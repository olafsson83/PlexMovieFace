"""Shared readiness checks so a missing setup step fails with a clear fix,
not a stack trace halfway through a multi-hour run.
"""
from pathlib import Path

from config import MOVIE_PATH, CLUSTERS_JSON, REPO_ROOT

MODEL_PATH = Path.home() / ".insightface" / "models" / "inswapper_128.onnx"


def _ffmpeg_ok():
    try:
        import video_io
        video_io.get_ffmpeg_exe()
        return True
    except Exception:
        return False


def diagnose():
    """Full readiness checklist, used by setup.py to show a status report."""
    return [
        ((REPO_ROOT / ".env").exists(), ".env file exists"),
        (MOVIE_PATH is not None, "MOVIE_PATH is set"),
        (bool(MOVIE_PATH and MOVIE_PATH.exists()), "Movie file found"),
        (_ffmpeg_ok(), "ffmpeg available"),
        (MODEL_PATH.exists(), "Face-swap model downloaded"),
    ]


def print_report(checks=None):
    checks = checks if checks is not None else diagnose()
    for ok, label in checks:
        print(f"  [{'OK' if ok else 'MISSING'}] {label}")
    return all(ok for ok, _ in checks)


def require_ready(need_movie=False, need_ffmpeg=False, need_model=False, need_discovery=False):
    """Exits with a friendly message (not a traceback) if a stage's needs aren't met."""
    problems = []

    if not (REPO_ROOT / ".env").exists():
        problems.append("No .env file found.")
    else:
        if need_movie:
            if not MOVIE_PATH:
                problems.append("MOVIE_PATH is not set in .env.")
            elif not MOVIE_PATH.exists():
                problems.append(f"Movie file not found: {MOVIE_PATH}")

        if need_ffmpeg and not _ffmpeg_ok():
            problems.append("ffmpeg could not be found or run.")

        if need_model and not MODEL_PATH.exists():
            problems.append(f"Face-swap model not found: {MODEL_PATH}")

        if need_discovery and not CLUSTERS_JSON.exists():
            problems.append(f"No discovery results found ({CLUSTERS_JSON}).")

    if problems:
        print("Setup is incomplete:")
        for p in problems:
            print(f"  - {p}")
        if len(problems) == 1 and "discovery results" in problems[0]:
            raise SystemExit("\nRun character discovery first, then try again.")
        raise SystemExit("\nRun `python setup.py` to fix this.")
