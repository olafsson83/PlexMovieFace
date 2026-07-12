"""Analysis pass (v2 milestone 2): detection, tracking, and identity
decisions over the whole movie, producing a swap-plan artifact and NO
pixels. The render pass (render_movie.py) consumes the artifact and never
re-runs identity logic.

Splitting the passes is what unlocks the upstream roadmap: the artifact can
be inspected before rendering, future milestones can revise it with evidence
from later frames (backward tracking), and a correction interface can edit
it -- none of which a single forward streaming pass can do.
"""
import json
import sys
import time

import cv2
import numpy as np
from tqdm import tqdm

from config import MOVIE_PATH, CLUSTERS_JSON, SOURCE_FACES_DIR, OUTPUT_DIR, DETECT_EVERY_N_FRAMES
import adaptive_detection
import analysis_store
import bridging
import face_engine
import identity
import tracking


def artifact_path():
    return OUTPUT_DIR / f"{MOVIE_PATH.stem}_analysis.npz"


def run_analysis(face_app, sources, use_tracking=True):
    from swap_movie import load_clusters, process_frame

    centroids = load_clusters()
    manifest = json.loads(CLUSTERS_JSON.read_text(encoding="utf-8"))
    groups = identity.group_sources(SOURCE_FACES_DIR)
    thresholds = identity.resolve_thresholds(groups, manifest)
    identity.describe_thresholds(groups, thresholds)
    identity_mgr = identity.TrackIdentityManager(centroids, sources, groups, thresholds)
    identity_mgr.record_observations = True
    detector = adaptive_detection.AdaptiveDetector(face_app).bind(identity_mgr)
    tracker = tracking.FaceTracker()
    counts = {"swapped": 0, "no_photo_events": 0, "unmatched_events": 0}

    cap = cv2.VideoCapture(str(MOVIE_PATH))
    if not cap.isOpened():
        sys.exit(f"Could not open movie file: {MOVIE_PATH}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    rows = []
    frame_i = 0
    start = time.time()
    progress = tqdm(total=total_frames, desc="Analyzing", unit="frame")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            active = process_frame(frame, detector, identity_mgr, tracker, use_tracking,
                                   counts, frame_index=frame_i)
            for face in active:
                rows.append((frame_i, face.track_id if face.track_id is not None else -1,
                             face.character_number, face.kps,
                             face.meta if face.meta is not None else analysis_store.make_meta()))
            frame_i += 1
            progress.update(1)
    finally:
        progress.close()
        cap.release()

    # Future-evidence passes: give confirmed tracks back the pre-confirmation
    # frames the acceptance gate withheld, then bridge interior gaps between
    # detector-verified anchors where propagation failed mid-shot.
    backfill = identity.backfill_swap_rows(
        identity_mgr.observation_log, thresholds,
        max_gap_frames=DETECT_EVERY_N_FRAMES * 3,
    )
    rows.extend(backfill)
    existing_pairs = {(r[0], r[1]) for r in rows}
    bridge_stats = {}
    bridged = bridging.bridge_swap_rows(
        identity_mgr.observation_log, existing_pairs,
        max_gap_frames=DETECT_EVERY_N_FRAMES * 3,
        frame_source=bridging.video_frame_source(MOVIE_PATH),
        stats=bridge_stats,
    )
    rows.extend(bridged)
    rows.sort(key=lambda r: r[0])

    header = {
        "movie_path": str(MOVIE_PATH),
        "frame_count": frame_i,
        "detect_every_n_frames": DETECT_EVERY_N_FRAMES,
        "thresholds": {
            gid: {"enter": t.enter, "keep": t.keep, "strong": t.strong, "source": t.source}
            for gid, t in thresholds.items()
        },
        "counts": counts,
        "adaptive_detection": detector.stats,
        "tracking_quality": tracker.stats,
        "backfilled_rows": len(backfill),
        "bridged_rows": len(bridged),
        "bridge_verification": bridge_stats,
        "analysis_seconds": round(time.time() - start, 1),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = analysis_store.save_plan(artifact_path(), header, rows)
    extreme = sum(1 for r in rows
                  if not np.isnan(r[4]["yaw"]) and abs(r[4]["yaw"]) > 65)
    print(detector.summary())
    print(tracker.summary())
    print(f"pose evidence: {extreme} rows past 65 degrees yaw kept for the "
          "render pass to gate per backend")
    print(f"retroactive backfill: {len(backfill)} pre-confirmation frames recovered")
    print(bridging.summary(bridge_stats))
    print(f"Analysis complete: {len(rows)} swap decisions across {frame_i} frames -> {path}")
    return path


def main():
    import preflight
    from swap_movie import load_source_faces

    preflight.require_ready(need_movie=True, need_model=True, need_discovery=True)
    face_app = face_engine.build_face_app()
    sources = load_source_faces(face_app)
    if not sources:
        sys.exit(f"No source photos found in {SOURCE_FACES_DIR}.")
    run_analysis(face_app, sources)


if __name__ == "__main__":
    main()
