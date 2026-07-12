"""Versioned on-disk swap plan -- the contract between the analysis pass
(analyze_movie.py) and the render pass (render_movie.py).

The artifact holds one row per (frame, face) swap decision. Since format v3
each row also carries the evidence behind the decision -- actual head pose
(pitch/yaw/roll from the 3D landmark model), detection score, identity
score and margin, observation provenance (detector / flow / backfill /
bridge) and a confidence -- so the RENDER pass can decide per backend
whether a row is renderable (extreme poses are no longer discarded during
analysis) and downstream tooling can audit every swap decision. The render
pass consumes the plan verbatim and never re-runs detection or identity
logic.
"""
import json
import math

import numpy as np

FORMAT_VERSION = 3

PROVENANCES = ("detector", "flow", "backfill", "bridge")

META_DEFAULTS = {
    "yaw": math.nan, "pitch": math.nan, "roll": math.nan,
    "det_score": math.nan, "identity_score": math.nan, "margin": math.nan,
    "provenance": "detector", "confidence": 1.0,
}


def make_meta(**kwargs):
    """A row's evidence record, with NaN/default fill for unknown fields."""
    meta = dict(META_DEFAULTS)
    for k, v in kwargs.items():
        if k not in META_DEFAULTS:
            raise KeyError(f"unknown meta field {k}")
        if v is not None:
            meta[k] = v
    return meta


def is_compatible(path):
    """Cheap header check used before deciding to reuse an analysis plan.
    Analysis algorithms are part of the artifact contract: v3 rows carry
    pose/score/provenance evidence and no longer pre-discard extreme poses,
    so older plans must be regenerated rather than silently rendered."""
    try:
        with np.load(path, allow_pickle=False) as data:
            header = json.loads(str(data["header"]))
        return header.get("format_version") == FORMAT_VERSION
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def save_plan(path, header, rows):
    """rows: iterable of (frame_index, track_id, character_number, kps5x2,
    meta) where meta comes from make_meta()."""
    rows = list(rows)
    if rows:
        frames = np.array([r[0] for r in rows], dtype=np.int64)
        track_ids = np.array([r[1] for r in rows], dtype=np.int64)
        numbers = np.array([str(r[2]) for r in rows])
        kps = np.stack([np.asarray(r[3], dtype=np.float32) for r in rows])
        pose = np.array([[r[4]["pitch"], r[4]["yaw"], r[4]["roll"]] for r in rows],
                        dtype=np.float32)
        scores = np.array([[r[4]["det_score"], r[4]["identity_score"],
                            r[4]["margin"], r[4]["confidence"]] for r in rows],
                          dtype=np.float32)
        provenance = np.array([r[4]["provenance"] for r in rows])
    else:
        frames = np.empty(0, dtype=np.int64)
        track_ids = np.empty(0, dtype=np.int64)
        numbers = np.empty(0, dtype="U1")
        kps = np.empty((0, 5, 2), dtype=np.float32)
        pose = np.empty((0, 3), dtype=np.float32)
        scores = np.empty((0, 4), dtype=np.float32)
        provenance = np.empty(0, dtype="U8")

    header = dict(header)
    header["format_version"] = FORMAT_VERSION
    header["row_count"] = len(rows)

    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            header=np.array(json.dumps(header)),
            frames=frames, track_ids=track_ids, numbers=numbers, kps=kps,
            pose=pose, scores=scores, provenance=provenance,
        )
    tmp.replace(path)
    return path


def load_plan(path):
    """Returns (header, plan) where plan maps frame_index -> list of
    (track_id, character_number, kps5x2, meta)."""
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
        pose = data["pose"]
        scores = data["scores"]
        provenance = data["provenance"]

    plan = {}
    for i in range(len(frames)):
        meta = {
            "pitch": float(pose[i, 0]), "yaw": float(pose[i, 1]),
            "roll": float(pose[i, 2]),
            "det_score": float(scores[i, 0]), "identity_score": float(scores[i, 1]),
            "margin": float(scores[i, 2]), "confidence": float(scores[i, 3]),
            "provenance": str(provenance[i]),
        }
        plan.setdefault(int(frames[i]), []).append(
            (int(track_ids[i]), str(numbers[i]), kps[i], meta)
        )
    return header, plan
