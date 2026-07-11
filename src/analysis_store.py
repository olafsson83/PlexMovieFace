"""Versioned on-disk swap plan -- the contract between the analysis pass
(analyze_movie.py) and the render pass (render_movie.py).

The artifact holds one row per (frame, face) swap decision: frame index,
stable track id, character number, and the 5 alignment landmarks. The render
pass consumes it verbatim and never re-runs detection or identity logic; the
analysis artifact is the single place identity decisions live, and it can be
inspected (or later hand-corrected) before any pixel is rendered.
"""
import json

import numpy as np

FORMAT_VERSION = 2


def is_compatible(path):
    """Cheap header check used before deciding to reuse an analysis plan.
    Analysis algorithms are part of the artifact contract: version 2 adds
    scene-cut-safe and quality-gated optical-flow decisions, so a v1 plan
    must be regenerated rather than silently rendering its old bad rows."""
    try:
        with np.load(path, allow_pickle=False) as data:
            header = json.loads(str(data["header"]))
        return header.get("format_version") == FORMAT_VERSION
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def save_plan(path, header, rows):
    """rows: iterable of (frame_index, track_id, character_number, kps5x2)."""
    rows = list(rows)
    if rows:
        frames = np.array([r[0] for r in rows], dtype=np.int64)
        track_ids = np.array([r[1] for r in rows], dtype=np.int64)
        numbers = np.array([str(r[2]) for r in rows])
        kps = np.stack([np.asarray(r[3], dtype=np.float32) for r in rows])
    else:
        frames = np.empty(0, dtype=np.int64)
        track_ids = np.empty(0, dtype=np.int64)
        numbers = np.empty(0, dtype="U1")
        kps = np.empty((0, 5, 2), dtype=np.float32)

    header = dict(header)
    header["format_version"] = FORMAT_VERSION
    header["row_count"] = len(rows)

    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            header=np.array(json.dumps(header)),
            frames=frames, track_ids=track_ids, numbers=numbers, kps=kps,
        )
    tmp.replace(path)
    return path


def load_plan(path):
    """Returns (header, plan) where plan maps frame_index -> list of
    (track_id, character_number, kps5x2)."""
    with np.load(path, allow_pickle=False) as data:
        header = json.loads(str(data["header"]))
        if header.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"analysis artifact {path} has format_version "
                f"{header.get('format_version')}, expected {FORMAT_VERSION} -- "
                "re-run the analysis pass."
            )
        frames = data["frames"]
        track_ids = data["track_ids"]
        numbers = data["numbers"]
        kps = data["kps"]

    plan = {}
    for i in range(len(frames)):
        plan.setdefault(int(frames[i]), []).append(
            (int(track_ids[i]), str(numbers[i]), kps[i])
        )
    return header, plan
