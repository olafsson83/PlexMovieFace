"""Stage 2: face-swap the whole movie using the character mapping from stage 1.

For each detected face in each frame, classifies it against every discovered
character (nearest cluster centroid from clusters.json, not just the ones
with a source photo -- this is what keeps identification accurate). Only
actually swaps if a matching characterN.jpg source photo was provided;
otherwise the face is left untouched (e.g. unnamed background extras).
"""
import json
import sys

import cv2
import numpy as np
from tqdm import tqdm

from config import (
    MOVIE_PATH, CHARACTERS_DIR, CLUSTERS_JSON, SOURCE_FACES_DIR, OUTPUT_DIR, CLUSTER_EPS,
)
import face_engine
import preflight
import video_io

MATCH_THRESHOLD = 1.0 - CLUSTER_EPS  # cosine similarity floor to count as a known character


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


def main():
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = OUTPUT_DIR / "_audio.mka"
    video_only_path = OUTPUT_DIR / "_video_only.mp4"
    final_path = OUTPUT_DIR / f"{MOVIE_PATH.stem}_faceswapped.mp4"

    print("Extracting audio track...")
    video_io.extract_audio(MOVIE_PATH, audio_path)

    encoder = video_io.open_encoder_pipe(video_only_path, width, height, fps)

    counts = {"swapped": 0, "unmatched": 0, "no_photo": 0}
    progress = tqdm(total=total_frames, desc="Swapping", unit="frame")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            for face in face_app.get(frame):
                number = classify(face, centroids)
                if number is None:
                    counts["unmatched"] += 1
                    continue
                source_face = sources.get(number)
                if source_face is None:
                    counts["no_photo"] += 1
                    continue
                frame = swapper.get(frame, face, source_face, paste_back=True)
                counts["swapped"] += 1

            encoder.stdin.write(frame.tobytes())
            progress.update(1)
    finally:
        progress.close()
        cap.release()
        encoder.stdin.close()
        encoder.wait()

    print("Muxing audio back in...")
    video_io.mux(video_only_path, audio_path, final_path)
    video_only_path.unlink(missing_ok=True)
    audio_path.unlink(missing_ok=True)

    print(
        f"\nDone. {counts['swapped']} faces swapped, {counts['no_photo']} recognized but "
        f"no source photo provided, {counts['unmatched']} unrecognized. Output: {final_path}"
    )


if __name__ == "__main__":
    main()
