"""Real-clip regression harness (Milestone 1 of IMPROVEMENT_PLAN.md).

Runs the analysis half of the pipeline (detection, tracking, identity
decisions -- no swap inference, no encoding) over the fixture clips in
tests/fixtures/manifest.json and checks the outcomes that were previously
verified by eye: wrong-person swaps (hard failures), coverage of known
shots, and per-character activity. Emits per-frame decision CSVs and a
summary JSON so any future change is measured instead of eyeballed.

Usage:
    python src/evaluate.py                      # whole suite
    python src/evaluate.py --fixture NAME       # one fixture
    python src/evaluate.py --out results_dir    # custom output dir

Each fixture runs in a subprocess with its manifest env applied, because
config.py resolves directories and thresholds at import time.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "manifest.json"
DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "results"


# --- single-fixture analysis (subprocess entry) ------------------------------

def run_fixture_analysis(fixture, out_dir):
    """Runs detection + identity over the clip, mirroring swap_movie's frame
    loop exactly (same process_frame), and records per-frame swap decisions.
    """
    import cv2
    import numpy as np

    import face_engine
    import identity
    import tracking
    from config import CLUSTERS_JSON, SOURCE_FACES_DIR
    from swap_movie import load_clusters, load_source_faces, process_frame

    centroids = load_clusters()
    face_app = face_engine.build_face_app()
    sources = load_source_faces(face_app)
    manifest = json.loads(CLUSTERS_JSON.read_text(encoding="utf-8"))
    groups = identity.group_sources(SOURCE_FACES_DIR)
    thresholds = identity.resolve_thresholds(groups, manifest)
    identity_mgr = identity.TrackIdentityManager(centroids, sources, groups, thresholds)
    tracker = tracking.FaceTracker()
    counts = {"swapped": 0, "no_photo_events": 0, "unmatched_events": 0}

    cap = cv2.VideoCapture(str(REPO_ROOT / fixture["clip"]))
    if not cap.isOpened():
        raise SystemExit(f"cannot open clip: {fixture['clip']}")

    rows = []  # (frame, number, cx, cy)
    frame_i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        active = process_frame(frame, face_app, identity_mgr, tracker, True, counts)
        for face in active:
            center = np.asarray(face.kps).mean(axis=0)
            rows.append((frame_i, face.character_number, float(center[0]), float(center[1])))
        frame_i += 1
    cap.release()

    csv_path = out_dir / f"{fixture['name']}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "number", "cx", "cy"])
        writer.writerows(rows)

    return evaluate_expectations(fixture, rows, frame_i)


def evaluate_expectations(fixture, rows, total_frames):
    swaps_by_frame = {}
    frames_by_number = {}
    for frame, number, cx, cy in rows:
        swaps_by_frame.setdefault(frame, []).append((number, cx, cy))
        frames_by_number.setdefault(number, set()).add(frame)

    results = []
    for exp in fixture.get("expectations", []):
        kind = exp["type"]
        if kind == "no_swap":
            a, b = exp["frames"]
            region = exp.get("region")
            violations = []
            for frame in range(a, b + 1):
                for number, cx, cy in swaps_by_frame.get(frame, []):
                    if region and not (region[0] <= cx <= region[2] and region[1] <= cy <= region[3]):
                        continue
                    violations.append({"frame": frame, "number": number, "cx": cx, "cy": cy})
            results.append({
                "type": kind, "note": exp.get("note", ""), "frames": [a, b],
                "passed": not violations, "wrong_person_swaps": len(violations),
                "violations": violations[:10],
            })
        elif kind == "coverage":
            a, b = exp["frames"]
            numbers = set(exp.get("numbers", []))
            hit = 0
            for frame in range(a, b + 1):
                swaps = swaps_by_frame.get(frame, [])
                if numbers:
                    swaps = [s for s in swaps if s[0] in numbers]
                if swaps:
                    hit += 1
            coverage = hit / max(b - a + 1, 1)
            results.append({
                "type": kind, "note": exp.get("note", ""), "frames": [a, b],
                "passed": coverage >= exp["min"],
                "coverage": round(coverage, 3), "required": exp["min"],
            })
        elif kind == "min_frames_swapped":
            per_number = {n: len(frames_by_number.get(n, ())) for n in exp["numbers"]}
            results.append({
                "type": kind, "note": exp.get("note", ""),
                "passed": all(v >= exp["min"] for v in per_number.values()),
                "frames_swapped": per_number, "required": exp["min"],
            })
        else:
            results.append({"type": kind, "passed": False, "error": "unknown expectation type"})

    return {
        "name": fixture["name"],
        "total_frames": total_frames,
        "total_swaps": len(rows),
        "frames_with_swaps": len(swaps_by_frame),
        "swapped_frames_by_number": {n: len(f) for n, f in sorted(frames_by_number.items())},
        "wrong_person_swaps": sum(r.get("wrong_person_swaps", 0) for r in results),
        "expectations": results,
        "passed": all(r["passed"] for r in results),
    }


# --- suite orchestration ------------------------------------------------------

def run_suite(manifest_path, out_dir, only=None):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for fixture in manifest["fixtures"]:
        if only and fixture["name"] != only:
            continue
        print(f"\n=== {fixture['name']} ===")
        env = os.environ.copy()
        env.update(fixture.get("env", {}))
        env["MOVIE_PATH"] = str(REPO_ROOT / fixture["clip"])
        proc = subprocess.run(
            [sys.executable, __file__, "--single", fixture["name"],
             "--manifest", str(manifest_path), "--out", str(out_dir)],
            env=env, cwd=str(REPO_ROOT),
        )
        summary_path = out_dir / f"{fixture['name']}.json"
        if proc.returncode != 0 or not summary_path.exists():
            summaries.append({"name": fixture["name"], "passed": False,
                              "error": f"analysis subprocess failed (exit {proc.returncode})"})
            continue
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

    suite = {
        "passed": all(s["passed"] for s in summaries),
        "wrong_person_swaps_total": sum(s.get("wrong_person_swaps", 0) for s in summaries),
        "fixtures": summaries,
    }
    (out_dir / "suite_summary.json").write_text(json.dumps(suite, indent=2), encoding="utf-8")

    print("\n=== suite summary ===")
    for s in summaries:
        status = "PASS" if s["passed"] else "FAIL"
        print(f"  [{status}] {s['name']}", end="")
        if "error" in s:
            print(f" -- {s['error']}")
            continue
        print(f" ({s['frames_with_swaps']}/{s['total_frames']} frames swapped, "
              f"{s.get('wrong_person_swaps', 0)} wrong-person)")
        for r in s["expectations"]:
            mark = "ok" if r["passed"] else "FAIL"
            detail = ""
            if r["type"] == "coverage":
                detail = f" coverage={r['coverage']} (need {r['required']})"
            elif r["type"] == "no_swap":
                detail = f" wrong_person_swaps={r['wrong_person_swaps']}"
            elif r["type"] == "min_frames_swapped":
                detail = f" {r['frames_swapped']} (need {r['required']})"
            print(f"      [{mark}] {r['type']}{detail} -- {r.get('note', '')}")
    print(f"\nwrong-person swaps across suite: {suite['wrong_person_swaps_total']} (hard-failure metric)")
    print(f"suite: {'PASS' if suite['passed'] else 'FAIL'}")
    return 0 if suite["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description="Real-clip regression suite")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--fixture", default=None, help="run only this fixture")
    parser.add_argument("--single", default=None, help=argparse.SUPPRESS)  # subprocess entry
    args = parser.parse_args()

    if args.single:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        fixture = next(f for f in manifest["fixtures"] if f["name"] == args.single)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = run_fixture_analysis(fixture, out_dir)
        (out_dir / f"{fixture['name']}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return 0

    return run_suite(args.manifest, Path(args.out), only=args.fixture)


if __name__ == "__main__":
    sys.exit(main())
