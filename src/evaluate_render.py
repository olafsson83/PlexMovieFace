"""Rendered-output regression harness.

evaluate.py validates the ANALYSIS plan (coverage, wrong-person windows); it
cannot see scrambled geometry, weak identity transfer, backend flicker or
malformed composites, because it never renders a pixel. This harness closes
that gap: for each probe window it renders the plan's rows through the real
backend + plate-matching path and measures the OUTPUT --

    out_sim        identity of the composited face vs the source photo
    orig_sim       identity of the untouched plate face vs the source photo
    identity_gain  out_sim - orig_sim (what the swap actually bought)
    align_residual mean landmark residual (px at 112 scale) of the
                   similarity fit used for alignment -- large values mean
                   the five points no longer describe one rigid face
    instability    per-frame change of the aligned output crop relative to
                   the plate's own change (ratio >> 1 = flicker/scramble)
    route          which backend arm rendered each row (hybrid only), plus
                   the number of per-track route transitions in the window

Embeddings are computed on aligned 112px crops directly through the
recognition model (no detector round-trip), so extreme poses the detector
would refuse still get measured. Each window also writes a crop strip
(plate row over output row) for human approval -- metrics rank, eyes accept.

Usage:
    python src/evaluate_render.py                       # all probes
    python src/evaluate_render.py --probe NAME
    python src/evaluate_render.py --compare A.json B.json   # regression diff

Probes run in subprocesses with their manifest env (config resolves at
import time), same pattern as evaluate.py.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "render_manifest.json"
DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "results"

# --compare acceptance tolerances
GAIN_DROP_TOLERANCE = 0.02       # mean identity_gain may not drop more
INSTABILITY_RISE_TOLERANCE = 1.2  # max instability may not grow past this factor
RESIDUAL_RISE_TOLERANCE = 1.2     # mean align_residual, same


# --- single-probe rendering (subprocess entry) -------------------------------

def _normed(feat):
    import numpy as np
    n = float(np.linalg.norm(feat))
    return feat / n if n > 0 else feat


def run_probe(probe, out_dir):
    import cv2
    import numpy as np
    from insightface.utils import face_align

    import analysis_store
    import face_engine
    import plate_matching
    import swap_backend
    import tracking
    from swap_movie import load_source_faces

    clip = REPO_ROOT / probe["clip"]
    face_app = face_engine.build_face_app()
    rec_model = face_app.models["recognition"]
    sources = load_source_faces(face_app)

    from analyze_movie import artifact_path, run_analysis
    plan_path = artifact_path()
    if not plan_path.exists() or not analysis_store.is_compatible(plan_path):
        print(f"  no compatible analysis artifact at {plan_path}; analyzing...")
        run_analysis(face_app, sources)
    header, plan = analysis_store.load_plan(plan_path)

    backend = swap_backend.build_backend()
    matcher = plate_matching.PlateMatcher(backend)
    source_embeddings = {
        n: _normed(np.asarray(f.normed_embedding, dtype=np.float32))
        for n, f in sources.items()
    }

    def identity_sim(frame, kps, number):
        crop = face_align.norm_crop(frame, np.asarray(kps, dtype=np.float32), 112)
        feat = _normed(rec_model.get_feat(crop).ravel().astype(np.float32))
        return float(np.dot(feat, source_embeddings[number]))

    def align_residual(kps):
        M = face_align.estimate_norm(np.asarray(kps, dtype=np.float32), 112)
        mapped = np.asarray(kps, dtype=np.float32) @ M[:, :2].T + M[:, 2]
        template = face_align.arcface_dst  # (5,2) at 112
        return float(np.linalg.norm(mapped - template, axis=1).mean())

    cap = cv2.VideoCapture(str(clip))
    results = {"probe": probe["name"], "backend": backend.capabilities(),
               "windows": [], "approved": None}

    for window in probe["windows"]:
        start, end = window["frames"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames_out = []
        prev_crops = {}  # track_id -> (orig_aligned128, out_aligned128)
        route_by_track = {}
        transitions = 0
        strip_orig, strip_out = [], []

        for frame_i in range(start, end + 1):
            ret, frame = cap.read()
            if not ret:
                break
            rows = plan.get(frame_i, [])
            out_frame = frame.copy()
            frame_rec = {"frame": frame_i, "rows": []}
            for track_id, number, kps in rows:
                if number not in sources:
                    continue
                tf = tracking.TrackedFace(np.asarray(kps), number, track_id)
                out_frame = matcher.swap(out_frame, tf, sources[number])
                route = getattr(backend, "last_route", None)
                if route is not None:
                    prev_route = route_by_track.get(track_id)
                    if prev_route is not None and prev_route != route:
                        transitions += 1
                    route_by_track[track_id] = route

                orig_sim = identity_sim(frame, kps, number)
                out_sim = identity_sim(out_frame, kps, number)
                orig128 = face_align.norm_crop(frame, np.asarray(kps, dtype=np.float32), 128)
                out128 = face_align.norm_crop(out_frame, np.asarray(kps, dtype=np.float32), 128)

                instability = None
                if track_id in prev_crops:
                    p_orig, p_out = prev_crops[track_id]
                    d_orig = float(cv2.absdiff(orig128, p_orig).mean())
                    d_out = float(cv2.absdiff(out128, p_out).mean())
                    instability = d_out / max(d_orig, 0.5)
                prev_crops[track_id] = (orig128, out128)

                frame_rec["rows"].append({
                    "track_id": track_id,
                    "number": number,
                    "orig_sim": round(orig_sim, 4),
                    "out_sim": round(out_sim, 4),
                    "identity_gain": round(out_sim - orig_sim, 4),
                    "align_residual": round(align_residual(kps), 3),
                    "instability": None if instability is None else round(instability, 3),
                    "route": route,
                })
                strip_orig.append(orig128)
                strip_out.append(out128)
            frames_out.append(frame_rec)

        all_rows = [r for f in frames_out for r in f["rows"]]
        gains = [r["identity_gain"] for r in all_rows]
        insts = [r["instability"] for r in all_rows if r["instability"] is not None]
        residuals = [r["align_residual"] for r in all_rows]
        summary = {
            "name": window["name"],
            "frames": window["frames"],
            "swapped_rows": len(all_rows),
            "frames_with_rows": sum(1 for f in frames_out if f["rows"]),
            "frames_total": len(frames_out),
            "mean_identity_gain": round(float(np.mean(gains)), 4) if gains else None,
            "min_out_sim": round(min((r["out_sim"] for r in all_rows), default=0.0), 4) if all_rows else None,
            "mean_align_residual": round(float(np.mean(residuals)), 3) if residuals else None,
            "max_instability": round(max(insts), 3) if insts else None,
            "route_counts": {r: sum(1 for x in all_rows if x["route"] == r)
                             for r in {x["route"] for x in all_rows}} if all_rows else {},
            "route_transitions": transitions,
            "per_frame": frames_out,
        }
        results["windows"].append(summary)

        if strip_orig:
            strip = np.vstack([np.hstack(strip_orig), np.hstack(strip_out)])
            strip_path = out_dir / f"render_{probe['name']}_{window['name']}.png"
            cv2.imwrite(str(strip_path), strip)

        gain_s = summary["mean_identity_gain"]
        print(f"  {window['name']}: {summary['frames_with_rows']}/{summary['frames_total']} frames rendered, "
              f"gain {gain_s}, min out_sim {summary['min_out_sim']}, "
              f"max instability {summary['max_instability']}, "
              f"routes {summary['route_counts']} ({transitions} transitions)")

    cap.release()
    out_path = out_dir / f"render_{probe['name']}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  -> {out_path}")


# --- comparison (regression gate) --------------------------------------------

def compare(baseline_path, candidate_path):
    base = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    cand = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    base_w = {w["name"]: w for w in base["windows"]}
    failures = []
    for w in cand["windows"]:
        b = base_w.get(w["name"])
        if b is None:
            print(f"  [new]  {w['name']}: no baseline")
            continue
        msgs = []
        bg, cg = b["mean_identity_gain"], w["mean_identity_gain"]
        if bg is not None and cg is not None and cg < bg - GAIN_DROP_TOLERANCE:
            msgs.append(f"identity_gain {bg} -> {cg}")
        if cg is not None and bg is None:
            msgs.append("")  # new coverage where baseline had none: fine
        bi, ci = b["max_instability"], w["max_instability"]
        if bi is not None and ci is not None and ci > bi * INSTABILITY_RISE_TOLERANCE and ci > 1.5:
            msgs.append(f"instability {bi} -> {ci}")
        br, cr = b["mean_align_residual"], w["mean_align_residual"]
        if br is not None and cr is not None and cr > br * RESIDUAL_RISE_TOLERANCE and cr > 2.0:
            msgs.append(f"align_residual {br} -> {cr}")
        if w["route_transitions"] > b["route_transitions"]:
            msgs.append(f"route_transitions {b['route_transitions']} -> {w['route_transitions']}")
        msgs = [m for m in msgs if m]
        status = "FAIL" if msgs else "ok"
        if msgs:
            failures.append(w["name"])
        print(f"  [{status}] {w['name']}"
              + (": " + "; ".join(msgs) if msgs else
                 f": gain {bg} -> {cg}, rows {b['swapped_rows']} -> {w['swapped_rows']}"))
    print(f"\nrender comparison: {'FAIL (' + ', '.join(failures) + ')' if failures else 'PASS'}")
    return 1 if failures else 0


# --- orchestration ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", help="run a single probe by name")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--run-inline", action="store_true",
                        help="(internal) run the named probe in this process")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"),
                        help="compare two result JSONs instead of rendering")
    args = parser.parse_args()

    if args.compare:
        sys.exit(compare(*args.compare))

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    probes = [p for p in manifest["probes"]
              if args.probe is None or p["name"] == args.probe]
    if not probes:
        sys.exit(f"no probe named {args.probe}")

    if args.run_inline:
        run_probe(probes[0], out_dir)
        return

    for probe in probes:
        print(f"\n=== {probe['name']} ===")
        env = dict(os.environ)
        env.update(probe.get("env", {}))
        env["MOVIE_PATH"] = str(REPO_ROOT / probe["clip"])
        proc = subprocess.run(
            [sys.executable, __file__, "--probe", probe["name"], "--run-inline",
             "--manifest", args.manifest, "--out", args.out],
            env=env, cwd=REPO_ROOT,
        )
        if proc.returncode != 0:
            sys.exit(f"probe {probe['name']} failed")


if __name__ == "__main__":
    main()
