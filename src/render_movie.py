"""Render pass (v2 milestone 2): consumes the analysis artifact and produces
pixels -- swap inference, plate matching, compositing, segmented resumable
encode, audio mux. Runs no detection and no identity logic; every swap it
performs was decided (and is inspectable) in the artifact.
"""
import sys

import cv2
from tqdm import tqdm

from config import MOVIE_PATH, OUTPUT_DIR, SEGMENT_FRAME_COUNT, USE_NVENC
import analysis_store
import plate_matching
import swap_backend
import tracking
import video_io


def run_render(plan_path, sources, swapper):
    from swap_movie import SEGMENTS_DIR, find_resume_point, finalize_output, swap_and_write

    header, plan = analysis_store.load_plan(plan_path)
    if header["movie_path"] != str(MOVIE_PATH):
        print(f"  note: artifact was analyzed from {header['movie_path']}")

    backend = swap_backend.build_backend(swapper)
    print(f"  swap backend: {backend.capabilities()}")
    plate_matcher = plate_matching.PlateMatcher(backend)
    pose_gate = swap_backend.RenderPoseGate(backend)

    cap = cv2.VideoCapture(str(MOVIE_PATH))
    if not cap.isOpened():
        sys.exit(f"Could not open movie file: {MOVIE_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = header.get("frame_count") or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    completed_segments, frames_done = find_resume_point()
    if completed_segments:
        print(f"Resuming render: {completed_segments} segment(s) complete ({frames_done} frames).")
        for _ in tqdm(range(frames_done), desc="Skipping to resume point", unit="frame"):
            if not cap.read()[0]:
                sys.exit("Movie is shorter than the completed segments -- delete the "
                         "output _segments/ folder and retry.")

    counts = {"swapped": 0}
    frame_i = frames_done
    segment_index = completed_segments
    video_ended = False
    progress = tqdm(total=total_frames, initial=frames_done, desc="Rendering", unit="frame")
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
                    active = [
                        tracking.TrackedFace(kps, number, track_id, meta)
                        for track_id, number, kps, meta in plan.get(frame_i, [])
                    ]
                    swap_and_write(frame, active, sources, plate_matcher, encoder,
                                   counts, pose_gate=pose_gate)
                    frame_i += 1
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

    final_path = finalize_output()
    print(f"\n{plate_matcher.summary()}")
    print(pose_gate.summary())
    if hasattr(backend, "routed"):
        print(f"hybrid routing: {backend.routed['primary']} primary, "
              f"{backend.routed['extreme']} extreme-pose (SimSwap), "
              f"{backend.transition_count} transitions, "
              f"{backend.proxy_fallbacks} proxy fallbacks")
    print(f"Done. {counts['swapped']} faces swapped (from the analysis plan). Output: {final_path}")
    return final_path


def main():
    import face_engine
    import preflight
    from swap_movie import load_source_faces

    preflight.require_ready(need_movie=True, need_ffmpeg=True, need_model=True, need_discovery=True)
    from analyze_movie import artifact_path
    plan_path = artifact_path()
    if not plan_path.exists():
        sys.exit(f"No analysis artifact at {plan_path} -- run `python src/analyze_movie.py` first.")

    face_app = face_engine.build_face_app()
    sources = load_source_faces(face_app)
    swapper = face_engine.build_swapper()
    run_render(plan_path, sources, swapper)


if __name__ == "__main__":
    main()
