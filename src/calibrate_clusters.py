"""Backfill identity-calibration stats into an existing clusters.json.

New discoveries write calibration stats automatically; this exists for
projects discovered before that (re-running full discovery would re-cluster
and could renumber characters, silently breaking the user's curated
source_faces mapping). It re-detects faces in the kept discovery frames and
assigns each to the nearest EXISTING centroid (same eps cut as clustering,
else noise), so numbering is preserved exactly.

Usage: python src/calibrate_clusters.py
"""
import json

import cv2
import numpy as np
from tqdm import tqdm

from config import DISCOVERY_FRAMES_DIR, CLUSTERS_JSON, CLUSTER_EPS
import face_engine
from discover_characters import (
    MIN_DET_SCORE, MIN_FACE_SIZE, calibration_stats, select_prototypes,
)


def main():
    if not CLUSTERS_JSON.exists():
        raise SystemExit(f"No clusters.json at {CLUSTERS_JSON} -- run discovery first.")
    frames = sorted(DISCOVERY_FRAMES_DIR.glob("frame_*.jpg"))
    if not frames:
        raise SystemExit(
            f"No discovery frames in {DISCOVERY_FRAMES_DIR} -- they're needed to "
            "measure score distributions. Re-run discovery instead (it now writes "
            "calibration automatically)."
        )

    manifest = json.loads(CLUSTERS_JSON.read_text(encoding="utf-8"))
    numbers = sorted(manifest.keys(), key=int)
    centroids = np.stack([
        np.array(manifest[n]["centroid"], dtype=np.float32) for n in numbers
    ])

    face_app = face_engine.build_face_app()
    embeddings = []
    for frame_path in tqdm(frames, desc="Embedding discovery frames", unit="frame"):
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        for face in face_app.get(img):
            x1, y1, x2, y2 = face.bbox
            if face.det_score < MIN_DET_SCORE or min(x2 - x1, y2 - y1) < MIN_FACE_SIZE:
                continue
            embeddings.append(face.normed_embedding)
    if not embeddings:
        raise SystemExit("No qualifying faces found in the discovery frames.")
    embeddings = np.stack(embeddings)

    # Assign each face to its nearest existing centroid using the same
    # similarity cut clustering used (1 - eps), else noise (-1).
    sims = embeddings @ centroids.T
    best = sims.argmax(axis=1)
    labels = np.where(sims.max(axis=1) >= 1.0 - CLUSTER_EPS, best, -1)

    label_to_number = {i: n for i, n in enumerate(numbers)}
    for i, n in enumerate(numbers):
        if not (labels == i).any():
            print(f"  character{n}: no faces re-assigned, skipping calibration")
            continue
        # Assignment stayed nearest-centroid (numbering stability); the
        # prototype bank and its calibration come from the assigned members.
        prototypes = select_prototypes(embeddings[labels == i])
        manifest[n]["prototypes"] = prototypes.tolist()
        manifest[n]["calibration"] = calibration_stats(
            i, prototypes, embeddings, labels, label_to_number
        )
        cal = manifest[n]["calibration"]
        worst_impostor = max(cal["impostor_p99_by_cluster"].values(), default=0.0)
        print(f"  character{n}: {len(prototypes)} prototypes, "
              f"genuine_p10={cal['genuine_p10']:.3f} "
              f"genuine_p50={cal['genuine_p50']:.3f} worst_impostor_p99={worst_impostor:.3f}")

    CLUSTERS_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nCalibration written to {CLUSTERS_JSON}")


if __name__ == "__main__":
    main()
