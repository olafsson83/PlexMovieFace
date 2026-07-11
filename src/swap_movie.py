"""Stage 2: face-swap the whole movie using the character mapping from stage 1.

For each detected face, classifies it against every discovered character
(nearest cluster centroid from clusters.json, not just the ones with a
source photo -- this is what keeps identification accurate). Only actually
swaps if a matching characterN.jpg source photo was provided; otherwise the
face is left untouched (e.g. unnamed background extras).

Phase 2 additions over the naive Phase 1 version:
- Optical-flow tracking between full detection passes (tracking.py) --
  detection runs every DETECT_EVERY_N_FRAMES frames (or immediately after a
  scene cut); frames in between reuse the last classification and just
  track face position, which is the dominant cost saving. --no-tracking
  forces full per-frame detection, useful for a correctness check.
- Chunked, resumable encoding: output is built as a sequence of segment
  files: if a run is interrupted, re-running picks up from the last
  complete segment instead of starting over.
- --calibrate: processes a short sample of the movie and projects a
  full-length runtime estimate, instead of committing blind to a run that
  could take hours.
"""
import argparse
import json
import shutil
import sys
import time

import cv2
import numpy as np
from tqdm import tqdm

from config import (
    MOVIE_PATH, CHARACTERS_DIR, CLUSTERS_JSON, SOURCE_FACES_DIR, OUTPUT_DIR,
    MATCH_THRESHOLD, MAINTAIN_THRESHOLD, SEGMENT_FRAME_COUNT, USE_NVENC,
)
import face_engine
import identity
import plate_matching
import preflight
import tracking
import video_io

SEGMENTS_DIR = OUTPUT_DIR / "_segments"


def load_clusters():
    manifest = json.loads(CLUSTERS_JSON.read_text(encoding="utf-8"))
    return {number: np.array(data["centroid"], dtype=np.float32) for number, data in manifest.items()}


def load_source_faces(face_app):
    """Returns {character_number: insightface Face} for every characterN.jpg present."""
    sources = {}
    if not SOURCE_FACES_DIR.exists():
        return sources

    for path in SOURCE_FACES_DIR.glob("character*.jpg"):
        stem = path.stem  # e.g. "character1"
        number = stem[len("character"):]
        if not number.isdigit():
            continue

        img = cv2.imread(str(path))
        if img is None:
            print(f"  skipped (unreadable): {path.name}")
            continue

        faces = face_app.get(img)
        if not faces:
            print(f"  skipped (no face detected): {path.name}")
            continue

        sources[number] = faces[0]
        print(f"  loaded source face for character{number}: {path.name}")

    return sources


def classify(face, centroids, hint_number=None):
    """Single-frame nearest-centroid lookup, kept for diagnostics. The swap
    pipeline itself uses identity.TrackIdentityManager, which makes the
    decision at the track level (hysteresis, margins, confirmation).
    """
    best_number, best_score = None, None
    for number, centroid in centroids.items():
        score = face_engine.cosine_similarity(face.normed_embedding, centroid)
        threshold = MAINTAIN_THRESHOLD if number == hint_number else MATCH_THRESHOLD
        if score < threshold:
            continue
        if best_score is None or score > best_score:
            best_number, best_score = number, score
    return best_number


def process_frame(frame, face_app, identity_mgr, tracker, use_tracking, counts):
    """Runs detection + track-level identity decisions on detection frames,
    optical-flow tracking in between. Returns the TrackedFace list to swap."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if use_tracking else None

    do_full_detection = (not use_tracking) or tracker.due_for_detection(gray)
    if not do_full_detection:
        return tracker.track(gray)

    scene_cut = tracker.is_scene_cut(gray) if use_tracking else False
    detected = face_app.get(frame)
    swappable = identity_mgr.observe(detected, scene_cut=scene_cut, counts=counts)

    if use_tracking:
        all_kps = [face.kps for face in detected]
        return tracker.start_from_detection(gray, swappable, all_kps)
    return [tracking.TrackedFace(kps, number, track_id)
            for kps, number, track_id in swappable]


def swap_and_write(frame, active_faces, sources, plate_matcher, encoder, counts):
    for face in active_faces:
        frame = plate_matcher.swap(frame, face, sources[face.character_number])
        counts["swapped"] += 1
    try:
        encoder.stdin.write(frame.tobytes())
    except OSError as e:
        encoder.wait(timeout=5)
        tail = "\n".join(encoder.stderr_tail)
        sys.exit(
            f"ffmpeg encoder exited unexpectedly (exit code {encoder.returncode}): {e}\n"
            f"ffmpeg's last output:\n{tail}"
        )


def find_resume_point():
    """Returns (completed_segment_count, frames_already_done)."""
    completed = 0
    while (SEGMENTS_DIR / f"segment_{completed + 1:05d}.mp4").exists():
        completed += 1
    return completed, completed * SEGMENT_FRAME_COUNT


def run_calibration(cap, fps, width, height, face_app, plate_matcher, identity_mgr, sources, use_tracking, seconds):
    frame_limit = max(1, int(seconds * fps))
    tracker = tracking.FaceTracker()
    counts = {"swapped": 0, "no_photo_events": 0, "unmatched_events": 0}

    tmp_path = OUTPUT_DIR / "_calibrate.mp4"
    encoder = video_io.open_encoder_pipe(tmp_path, width, height, fps, use_nvenc=USE_NVENC)

    print(f"Calibrating on the first {seconds:.0f}s ({frame_limit} frames)...")
    start = time.time()
    processed = 0
    for _ in tqdm(range(frame_limit), desc="Calibrating", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
        active_faces = process_frame(frame, face_app, identity_mgr, tracker, use_tracking, counts)
        swap_and_write(frame, active_faces, sources, plate_matcher, encoder, counts)
        processed += 1
    elapsed = time.time() - start

    encoder.stdin.close()
    encoder.wait()
    tmp_path.unlink(missing_ok=True)

    if processed == 0 or elapsed == 0:
        sys.exit("Calibration processed no frames -- is the movie file valid?")

    real_fps = processed / elapsed
    processed_seconds = processed / fps
    speed_factor = elapsed / processed_seconds

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    print(f"\nProcessed {processed} frames in {elapsed:.1f}s ({real_fps:.2f} fps, "
          f"{speed_factor:.2f}x slower than real time).")
    if total_frames:
        projected_seconds = (total_frames / fps) * speed_factor
        hours = int(projected_seconds // 3600)
        minutes = int((projected_seconds % 3600) // 60)
        print(f"Full movie is {total_frames} frames (~{total_frames / fps / 60:.0f} min). "
              f"Projected full run time: ~{hours}h{minutes}m.")
    else:
        print("Could not determine total frame count to project a full-movie estimate.")


def finalize_output():
    segments = sorted(SEGMENTS_DIR.glob("segment_*.mp4"))
    if not segments:
        sys.exit("No segments were produced -- nothing to finalize.")

    video_only_path = OUTPUT_DIR / "_video_only.mp4"
    audio_path = OUTPUT_DIR / "_audio.mka"
    final_path = OUTPUT_DIR / f"{MOVIE_PATH.stem}_faceswapped.mp4"

    print("Joining segments...")
    video_io.concat_segments(segments, video_only_path)

    print("Extracting audio track...")
    video_io.extract_audio(MOVIE_PATH, audio_path)

    print("Muxing audio back in...")
    video_io.mux(video_only_path, audio_path, final_path)

    video_only_path.unlink(missing_ok=True)
    audio_path.unlink(missing_ok=True)
    shutil.rmtree(SEGMENTS_DIR, ignore_errors=True)

    return final_path


def main():
    parser = argparse.ArgumentParser(
        description="Swap the movie: analysis pass (detection/tracking/identity -> "
                    "swap-plan artifact) followed by the render pass (pixels)."
    )
    parser.add_argument("--no-tracking", action="store_true",
                         help="Force full per-frame detection in the analysis pass (slower, useful for a correctness check)")
    parser.add_argument("--calibrate", nargs="?", const=30.0, type=float, default=None, metavar="SECONDS",
                         help="Process only the first SECONDS (default 30) and project a full-movie runtime instead of doing a full run")
    parser.add_argument("--analyze-only", action="store_true",
                         help="Stop after writing the analysis artifact (inspect it before rendering)")
    parser.add_argument("--render-only", action="store_true",
                         help="Render from an existing analysis artifact without re-analyzing")
    parser.add_argument("--reanalyze", action="store_true",
                         help="Redo the analysis pass even if an artifact already exists")
    args = parser.parse_args()
    use_tracking = not args.no_tracking

    preflight.require_ready(need_movie=True, need_ffmpeg=True, need_model=True, need_discovery=True)

    face_app = face_engine.build_face_app()
    sources = load_source_faces(face_app)
    if not sources:
        sys.exit(
            f"No source photos found in {SOURCE_FACES_DIR}. Drop in at least one "
            f"characterN.jpg (matching a number from {CHARACTERS_DIR}) before running this stage."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.calibrate is not None:
        centroids = load_clusters()
        manifest = json.loads(CLUSTERS_JSON.read_text(encoding="utf-8"))
        groups = identity.group_sources(SOURCE_FACES_DIR)
        thresholds = identity.resolve_thresholds(groups, manifest)
        identity_mgr = identity.TrackIdentityManager(centroids, sources, groups, thresholds)
        import adaptive_detection
        face_app = adaptive_detection.AdaptiveDetector(face_app).bind(identity_mgr)
        plate_matcher = plate_matching.PlateMatcher(face_engine.build_swapper())
        cap = cv2.VideoCapture(str(MOVIE_PATH))
        if not cap.isOpened():
            sys.exit(f"Could not open movie file: {MOVIE_PATH}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        run_calibration(cap, fps, width, height, face_app, plate_matcher, identity_mgr, sources, use_tracking, args.calibrate)
        cap.release()
        return

    import analyze_movie
    import render_movie

    plan_path = analyze_movie.artifact_path()
    if args.render_only:
        if not plan_path.exists():
            sys.exit(f"--render-only but no analysis artifact at {plan_path}.")
    elif args.reanalyze or not plan_path.exists():
        analyze_movie.run_analysis(face_app, sources, use_tracking=use_tracking)
    else:
        print(f"Reusing analysis artifact {plan_path} (pass --reanalyze after changing "
              "thresholds, source photos, or character data).")

    if args.analyze_only:
        return

    swapper = face_engine.build_swapper()
    render_movie.run_render(plan_path, sources, swapper)


if __name__ == "__main__":
    main()
