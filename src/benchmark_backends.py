"""Backend benchmark (v2 milestone 7): measures a swap backend on real
footage so candidates are compared on numbers, not screenshots.

For sampled frames of a clip, every face confidently matching a mapped
character is swapped through the ACTIVE backend (SWAP_BACKEND env) and the
full plate-matching composite, then measured:

- identity_similarity: re-detect the composited face and score its
  embedding against the SOURCE face's -- how much of the requested
  identity actually survived synthesis + compositing.
- crop_sharpness: Laplacian variance of the generated crop (proxy for the
  detail ceiling; higher is not automatically better, but a higher-res
  backend should separate clearly here).
- swap_ms: backend inference + compositing latency per face.

Usage (env selects project dirs exactly like the swap stage):
    python src/benchmark_backends.py --clip tests/fixtures/clips/brokeback_camp_clip.mp4
Writes tests/fixtures/results/benchmark_<backend>.json and prints a table.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Benchmark the active swap backend")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--every", type=int, default=15, help="sample every Nth frame")
    parser.add_argument("--min-score", type=float, default=0.7)
    parser.add_argument("--out", default=str(REPO_ROOT / "tests" / "fixtures" / "results"))
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import face_engine
    import identity
    import plate_matching
    import swap_backend
    from plate_matching import _interior_mask, sharpness_metrics
    from swap_movie import load_clusters, load_source_faces

    face_app = face_engine.build_face_app()
    sources = load_source_faces(face_app)
    centroids = load_clusters()
    backend = swap_backend.build_backend()
    matcher = plate_matching.PlateMatcher(backend)
    interior = _interior_mask(backend.capabilities().get("crop_size", 128))

    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.clip}")

    import tracking
    records = []
    frame_i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_i % args.every == 0:
            for face in face_app.get(frame):
                best_n = max(
                    sources,
                    key=lambda n: identity.prototype_max_score(face.normed_embedding, centroids[n]),
                )
                score = identity.prototype_max_score(face.normed_embedding, centroids[best_n])
                if score < args.min_score:
                    continue
                source = sources[best_n]
                tf = tracking.TrackedFace(face.kps, best_n, track_id=None)

                t0 = time.perf_counter()
                out_frame = matcher.swap(frame.copy(), tf, source)
                swap_ms = (time.perf_counter() - t0) * 1000

                bgr_fake, _ = backend.swap(frame, tf, backend.prepare_source(source))
                crop_sharp = sharpness_metrics(bgr_fake, interior)[0]

                # Identity survival: embed the composited face.
                out_faces = face_app.get(out_frame)
                sim = None
                if out_faces:
                    nearest = min(
                        out_faces,
                        key=lambda f: np.linalg.norm(
                            np.asarray(f.kps).mean(axis=0) - np.asarray(face.kps).mean(axis=0)
                        ),
                    )
                    sim = float(np.dot(nearest.normed_embedding, source.normed_embedding))
                records.append({
                    "frame": frame_i, "number": best_n, "swap_ms": swap_ms,
                    "crop_sharpness": crop_sharp, "identity_similarity": sim,
                })
        frame_i += 1
    cap.release()

    sims = [r["identity_similarity"] for r in records if r["identity_similarity"] is not None]
    report = {
        "backend": backend.capabilities(),
        "clip": args.clip,
        "swaps_measured": len(records),
        "identity_similarity_mean": round(float(np.mean(sims)), 4) if sims else None,
        "identity_similarity_p10": round(float(np.percentile(sims, 10)), 4) if sims else None,
        "crop_sharpness_mean": round(float(np.mean([r["crop_sharpness"] for r in records])), 1),
        "swap_ms_mean": round(float(np.mean([r["swap_ms"] for r in records])), 1),
        "records": records,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"benchmark_{backend.capabilities()['name']}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nbackend: {backend.capabilities()['name']}  ({len(records)} swaps measured)")
    print(f"  identity similarity: mean {report['identity_similarity_mean']}  "
          f"p10 {report['identity_similarity_p10']}")
    print(f"  crop sharpness (lap var): {report['crop_sharpness_mean']}")
    print(f"  swap+composite latency: {report['swap_ms_mean']} ms")
    print(f"  report: {out_path}")


if __name__ == "__main__":
    main()
