"""Stage 1: sample the movie, detect faces, cluster them into distinct
characters, and export a few representative crops per cluster for the user
to review and name.

Filenames encode the cluster number and a sample letter, e.g. character1a.jpg,
character1b.jpg, character2a.jpg -- the number is what matters for mapping to
a source photo later (see swap_movie.py); the letter just distinguishes
multiple sample crops of the same cluster so you can visually confirm it's
really one consistent person.
"""
import json
import shutil

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from config import (
    MOVIE_PATH, DISCOVERY_FRAMES_DIR, CHARACTERS_DIR, CLUSTERS_JSON,
    DISCOVERY_INTERVAL_SEC, CLUSTER_EPS, CLUSTER_MIN_SAMPLES,
)
import face_engine
import preflight
import video_io

MIN_DET_SCORE = 0.5
MIN_FACE_SIZE = 60  # shorter side, pixels
SAMPLES_PER_CHARACTER = 3
MIN_SAMPLE_GAP_SEC = 60.0
CROP_MARGIN = 0.3  # fraction of face box size, added on each side


def sample_frames():
    if DISCOVERY_FRAMES_DIR.exists():
        shutil.rmtree(DISCOVERY_FRAMES_DIR)
    print(f"Sampling frames every {DISCOVERY_INTERVAL_SEC}s...")
    video_io.extract_frames_at_interval(MOVIE_PATH, DISCOVERY_FRAMES_DIR, DISCOVERY_INTERVAL_SEC)
    frames = sorted(DISCOVERY_FRAMES_DIR.glob("frame_*.jpg"))
    if not frames:
        raise SystemExit(f"No frames extracted from {MOVIE_PATH} -- is the file readable by ffmpeg?")
    return frames


def detect_faces(face_app, frames):
    """Returns a flat list of face records: embedding, bbox, timestamp, frame path."""
    records = []
    for frame_path in tqdm(frames, desc="Detecting faces", unit="frame"):
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        frame_index = int(frame_path.stem.split("_")[1])
        timestamp = (frame_index - 1) * DISCOVERY_INTERVAL_SEC

        for face in face_app.get(img):
            x1, y1, x2, y2 = face.bbox
            width, height = x2 - x1, y2 - y1
            if face.det_score < MIN_DET_SCORE or min(width, height) < MIN_FACE_SIZE:
                continue
            records.append({
                "embedding": face.normed_embedding,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "timestamp": timestamp,
                "frame_path": frame_path,
            })
    return records


def cluster_faces(records):
    if not records:
        return {}, np.empty((0, 512)), np.empty(0, dtype=int)

    embeddings = np.stack([r["embedding"] for r in records])
    labels = DBSCAN(metric="cosine", eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES).fit_predict(embeddings)

    clusters = {}
    for record, label in zip(records, labels):
        if label == -1:
            continue  # noise -- not enough similar samples to count as a character
        clusters.setdefault(label, []).append(record)
    return clusters, embeddings, labels


PROTOTYPES_PER_CHARACTER = 5


def select_prototypes(embeddings, k=PROTOTYPES_PER_CHARACTER):
    """Prototype bank = the centroid PLUS diverse members chosen by
    farthest-point sampling. The centroid must stay in the bank: it is the
    best single representative for typical faces, and a bank of scattered
    member embeddings alone scores typical faces LOWER than the old
    centroid did (measured on the regression suite: coverage collapsed on
    two fixtures before the centroid was anchored). The FPS members add
    what the mean compresses away -- a profile face matches a profile
    prototype instead of being dragged down by a frontal-dominated mean.
    Pose/brightness aren't measured yet (milestone 6); embedding-space
    diversity is the v1 proxy.
    """
    embs = np.stack(embeddings)
    centroid = embs.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) or 1)
    chosen = [int(np.argmax(embs @ centroid))]
    while len(chosen) < min(k - 1, len(embs)):
        sims_to_chosen = embs @ embs[chosen].T          # N x len(chosen)
        nearest = sims_to_chosen.max(axis=1)
        nearest[chosen] = np.inf                        # don't re-pick
        candidate = int(np.argmin(nearest))
        if candidate in chosen:
            break
        chosen.append(candidate)
    return np.vstack([centroid[None], embs[chosen]])


def prototype_scores(embeddings, prototypes):
    """Max-over-prototypes cosine score for each embedding -- the same
    scoring the swap stage uses, so calibration stats stay consistent."""
    return (embeddings @ np.asarray(prototypes).T).max(axis=1)


def calibration_stats(label, prototypes, embeddings, labels, label_to_number):
    """Empirical score distributions for the swap stage's threshold
    calibration (see identity.py). These carry selection bias -- membership
    was itself decided by a similarity cut -- so identity.py treats them as
    evidence with safety margins, not ground truth.

    All scores are max-over-prototypes, the same function the swap stage
    applies, so the genuine and impostor distributions shift together when
    the representation changes.

    genuine_p10/p50: percentiles of members' scores against their own bank.
    impostor_p99_by_cluster: for every OTHER cluster (plus DBSCAN noise faces
    under the key "noise"), the 99th percentile of its faces' scores against
    THIS bank -- kept per-cluster so the swap stage can exclude duplicate
    clusters of the same actor (same source photo) when computing the real
    impostor tail.
    """
    member_sims = prototype_scores(embeddings[labels == label], prototypes)
    impostor_p99 = {}
    for other in set(labels.tolist()):
        if other == label:
            continue
        others = embeddings[labels == other]
        if len(others) == 0:
            continue
        sims = prototype_scores(others, prototypes)
        key = "noise" if other == -1 else label_to_number[other]
        impostor_p99[key] = round(float(np.percentile(sims, 99)), 4)
    return {
        "genuine_p10": round(float(np.percentile(member_sims, 10)), 4),
        "genuine_p50": round(float(np.percentile(member_sims, 50)), 4),
        "impostor_p99_by_cluster": impostor_p99,
    }


def select_representative_samples(members, k=SAMPLES_PER_CHARACTER, min_gap_sec=MIN_SAMPLE_GAP_SEC):
    """Picks up to k members spread out in time, favoring those most typical
    of the cluster (closest to its centroid) -- so the crops the user reviews
    actually differ in lighting/angle instead of being near-duplicate frames.
    """
    embeddings = np.stack([m["embedding"] for m in members])
    centroid = embeddings.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) or 1)

    ranked = sorted(members, key=lambda m: face_engine.cosine_similarity(m["embedding"], centroid), reverse=True)

    chosen = []
    chosen_ids = set()
    for candidate in ranked:
        if len(chosen) >= k:
            break
        if all(abs(candidate["timestamp"] - c["timestamp"]) >= min_gap_sec for c in chosen):
            chosen.append(candidate)
            chosen_ids.add(id(candidate))

    if len(chosen) < k:
        for candidate in ranked:
            if len(chosen) >= k:
                break
            if id(candidate) not in chosen_ids:
                chosen.append(candidate)
                chosen_ids.add(id(candidate))

    return chosen, centroid


def crop_face(frame_path, bbox):
    img = cv2.imread(str(frame_path))
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * CROP_MARGIN))
    y1 = max(0, int(y1 - bh * CROP_MARGIN))
    x2 = min(w, int(x2 + bw * CROP_MARGIN))
    y2 = min(h, int(y2 + bh * CROP_MARGIN))
    return img[y1:y2, x1:x2]


def export_characters(clusters, embeddings, labels):
    CHARACTERS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    letters = "abcdefghijklmnopqrstuvwxyz"

    # Number clusters 1..N in a stable order (largest cluster first) rather
    # than sklearn's arbitrary label order.
    ordered = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)
    label_to_number = {
        label: str(number) for number, (label, _) in enumerate(ordered, start=1)
    }

    for character_number, (label, members) in enumerate(ordered, start=1):
        samples, centroid = select_representative_samples(members)
        timestamps = []
        for letter, sample in zip(letters, samples):
            crop = crop_face(sample["frame_path"], sample["bbox"])
            if crop.size == 0:
                continue
            out_path = CHARACTERS_DIR / f"character{character_number}{letter}.jpg"
            cv2.imwrite(str(out_path), crop)
            timestamps.append(sample["timestamp"])

        prototypes = select_prototypes([m["embedding"] for m in members])
        manifest[str(character_number)] = {
            "centroid": centroid.tolist(),
            "prototypes": prototypes.tolist(),
            "member_count": len(members),
            "sample_timestamps": timestamps,
            "calibration": calibration_stats(
                label, prototypes, embeddings, labels, label_to_number
            ),
        }
        print(f"  character{character_number}: {len(members)} samples, "
              f"{len(prototypes)} prototypes")

    CLUSTERS_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    preflight.require_ready(need_movie=True, need_ffmpeg=True)

    frames = sample_frames()
    face_app = face_engine.build_face_app()

    records = detect_faces(face_app, frames)
    print(f"Detected {len(records)} qualifying faces across {len(frames)} sampled frames.")

    clusters, embeddings, labels = cluster_faces(records)
    if not clusters:
        raise SystemExit(
            "No recurring characters found. Try lowering CLUSTER_MIN_SAMPLES or "
            "DISCOVERY_INTERVAL_SEC in .env, or raising CLUSTER_EPS slightly."
        )

    print(f"Found {len(clusters)} distinct characters.")
    export_characters(clusters, embeddings, labels)

    print(
        f"\nDone. Review the crops in {CHARACTERS_DIR} -- for each character number you "
        f"want swapped, drop a replacement photo named e.g. character1.jpg into "
        f"{CHARACTERS_DIR.parent / 'source_faces'}, then run the swap stage."
    )


if __name__ == "__main__":
    main()
