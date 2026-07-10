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
    MATCH_THRESHOLD, SEGMENT_FRAME_COUNT, USE_NVENC,
)
import face_engine
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


def classify(face, centroids):
    """Returns the best-matching character number, or None if nothing is close enough."""
    best_number, best_score = None, MATCH_THRESHOLD
    for number, centroid in centroids.items():
        score = face_engine.cosine_similarity(face.normed_embedding, centroid)
        if score > best_score:
            best_number, best_score = number, score
    return best_number


def process_frame(frame, face_app, centroids, sources, tracker, use_tracking, counts):
    """Classifies/tracks faces in frame and returns the list of TrackedFace
    to actually swap this frame (empty if none)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if use_tracking else None

    do_full_detection = (not use_tracking) or tracker.due_for_detection(gray)
    if not do_full_detection:
        return tracker.track(gray)

    detected = face_app.get(frame)
    swappable = []
    for face in detected:
        number = classify(face, centroids)
        if number is None:
            counts["unmatched_events"] += 1
            continue
        if number not in sources:
            counts["no_photo_events"] += 1
            continue
        swappable.append((face.kps, number))

    if use_tracking:
        all_kps = [face.kps for face in detected]
        return tracker.start_from_detection(gray, swappable, all_kps)
    return [tracking.TrackedFace(kps, number) for kps, number in swappable]


def swap_and_write(frame, active_faces, sources, swapper, encoder, counts):
    for face in active_faces:
        frame = swapper.get(frame, face, sources[face.character_number], paste_back=True)
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


def run_calibration(cap, fps, width, height, face_app, swapper, centroids, sources, use_tracking, seconds):
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
        active_faces = process_frame(frame, face_app, centroids, sources, tracker, use_tracking, counts)
        swap_and_write(frame, active_faces, sources, swapper, encoder, counts)
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


def run_full(cap, fps, width, height, face_app, swapper, centroids, sources, use_tracking):
    completed_segments, frames_done = find_resume_point()
    if completed_segments:
        print(f"Resuming: {completed_segments} segment(s) already complete ({frames_done} frames). "
              "Skipping ahead...")
        for _ in tqdm(range(frames_done), desc="Skipping to resume point", unit="frame"):
            if not cap.read()[0]:
                sys.exit("Movie is shorter than the already-completed segments -- delete output/_segments/ and retry.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    tracker = tracking.FaceTracker()
    counts = {"swapped": 0, "no_photo_events": 0, "unmatched_events": 0}

    progress = tqdm(total=total_frames, initial=frames_done, desc="Swapping", unit="frame")
    segment_index = completed_segments
    video_ended = False

    try:
        while not video_ended:
            segment_index += 1
            segment_final = SEGMENTS_DIR / f"segment_{segment_index:05d}.mp4"
            segment_tmp = SEGMENTS_DIR / f"segment_{segment_index:05d}.mp4.part"
            encoder = video_io.open_encoder_pipe(segment_tmp, width, height, fps, use_nvenc=USE_NVENC)

            frames_in_segment = 0
            try:
                for _ in range(SEGMENT_FRAME_COUNT):
                    ret, frame = cap.read()
                    if not ret:
                        video_ended = True
                        break
                    active_faces = process_frame(frame, face_app, centroids, sources, tracker, use_tracking, counts)
                    swap_and_write(frame, active_faces, sources, swapper, encoder, counts)
                    frames_in_segment += 1
                    progress.update(1)
            finally:
                encoder.stdin.close()
                encoder.wait()

            if frames_in_segment == 0:
                segment_tmp.unlink(missing_ok=True)
                break
            segment_tmp.rename(segment_final)
    finally:
        progress.close()
        cap.release()

    return counts


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
    parser = argparse.ArgumentParser(description="Swap the movie using the discovered character mapping")
    parser.add_argument("--no-tracking", action="store_true",
                         help="Force full per-frame detection instead of optical-flow tracking (slower, useful for a correctness check)")
    parser.add_argument("--calibrate", nargs="?", const=30.0, type=float, default=None, metavar="SECONDS",
                         help="Process only the first SECONDS (default 30) and project a full-movie runtime instead of doing a full run")
    args = parser.parse_args()
    use_tracking = not args.no_tracking

    preflight.require_ready(need_movie=True, need_ffmpeg=True, need_model=True, need_discovery=True)

    centroids = load_clusters()
    face_app = face_engine.build_face_app()
    sources = load_source_faces(face_app)

    if not sources:
        sys.exit(
            f"No source photos found in {SOURCE_FACES_DIR}. Drop in at least one "
            f"characterN.jpg (matching a number from {CHARACTERS_DIR}) before running this stage."
        )

    swapper = face_engine.build_swapper()

    cap = cv2.VideoCapture(str(MOVIE_PATH))
    if not cap.isOpened():
        sys.exit(f"Could not open movie file: {MOVIE_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.calibrate is not None:
        run_calibration(cap, fps, width, height, face_app, swapper, centroids, sources, use_tracking, args.calibrate)
        cap.release()
        return

    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    counts = run_full(cap, fps, width, height, face_app, swapper, centroids, sources, use_tracking)
    final_path = finalize_output()

    print(
        f"\nDone. {counts['swapped']} faces swapped. {counts['no_photo_events']} recognized-but-unswapped "
        f"and {counts['unmatched_events']} unrecognized events at full-detection frames (not exhaustive -- "
        f"tracked frames reuse the last detection's classification). Output: {final_path}"
    )


if __name__ == "__main__":
    main()
