"""Bidirectional-tracking gap bridge (round-4 review, item: bridge rework).

The original bridge filled interior track gaps by LINEAR INTERPOLATION
between two detector-verified anchors. That restored rows through the very
intervals where optical-flow propagation had been REJECTED by the quality
gates -- quietly undoing the scramble protection: if the face actually
moved non-linearly (or the gap hides an occlusion), interpolated landmarks
paste the swap onto the wrong pixels.

This version earns each bridged frame with pixel evidence:

    forward:  LK-track the previous anchor's landmarks frame by frame
    backward: LK-track the next anchor's landmarks in reverse
    emit a frame only when BOTH trajectories survived the same per-step
    safety gates the live tracker uses (tracking.propagate_kps: fb
    round-trip, partial-affine plausibility, in-frame bounds) AND agree
    with each other within a face-scaled tolerance; the emitted landmarks
    are the anchor-weighted blend of the two trajectories.

Pure interpolation survives only for gaps of at most two interior frames,
where the straight line cannot meaningfully diverge from the motion.
"""
from __future__ import annotations

import cv2
import numpy as np

from identity import _interp_meta
from tracking import propagate_kps

# A gap with at most this many interior frames may be filled by plain
# interpolation (no video evidence needed).
MAX_INTERP_GAP = 2

# Forward and backward trajectories must agree within this fraction of the
# face's landmark spread (floored in px) for a frame to be emitted.
AGREE_FRACTION = 0.10
AGREE_MIN_PX = 3.0


def video_frame_source(movie_path):
    """frame_index -> grayscale frame, reading the movie lazily with cached
    sequential access (gaps arrive roughly in order)."""
    cap = cv2.VideoCapture(str(movie_path))
    state = {"next": None}

    def read(frame_index):
        if state["next"] != frame_index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            state["next"] = None
            return None
        state["next"] = frame_index + 1
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return read


def _candidate_pairs(observation_log, max_gap_frames):
    """Consecutive same-track observation pairs that were both swapped as
    the same character -- any contradicting evidence inside a gap appears
    as an intermediate observation and splits the pair, blocking the
    bridge by construction (cuts can't bridge: new track_id per cut)."""
    by_track = {}
    for obs in observation_log:
        by_track.setdefault(obs.track_id, []).append(obs)
    for track_id, obs_list in by_track.items():
        obs_list.sort(key=lambda o: o.frame_index)
        for a, b in zip(obs_list[:-1], obs_list[1:]):
            if a.swapped_number is None or a.swapped_number != b.swapped_number:
                continue
            span = b.frame_index - a.frame_index
            if span <= 1 or span > max_gap_frames:
                continue
            yield track_id, a, b


def _bridge_tracked(a, b, frame_source, stats):
    """Bidirectionally verified landmark sets for the interior of gap (a, b):
    {frame_index: kps} for exactly the frames where both trajectories
    survived the safety gates and agree."""
    grays = {}
    for frame in range(a.frame_index, b.frame_index + 1):
        gray = frame_source(frame)
        if gray is None:
            stats["gaps_unreadable"] += 1
            return {}
        grays[frame] = gray

    forward = {}
    kps = a.kps
    for frame in range(a.frame_index + 1, b.frame_index):
        kps = propagate_kps(grays[frame - 1], grays[frame], kps, stats)
        if kps is None:
            break
        forward[frame] = kps

    backward = {}
    kps = b.kps
    for frame in range(b.frame_index - 1, a.frame_index, -1):
        kps = propagate_kps(grays[frame + 1], grays[frame], kps, stats)
        if kps is None:
            break
        backward[frame] = kps

    spread = a.kps.max(axis=0) - a.kps.min(axis=0)
    tolerance = max(AGREE_MIN_PX, AGREE_FRACTION * float(np.hypot(*spread)))

    agreed = {}
    span = b.frame_index - a.frame_index
    for frame in range(a.frame_index + 1, b.frame_index):
        f, r = forward.get(frame), backward.get(frame)
        if f is None or r is None:
            stats["frames_unverified"] += 1
            continue
        if float(np.linalg.norm(f - r, axis=1).mean()) > tolerance:
            stats["frames_disagree"] += 1
            continue
        t = (frame - a.frame_index) / span
        agreed[frame] = ((1 - t) * f + t * r).astype(np.float32)
    return agreed


def bridge_swap_rows(observation_log, existing_pairs, max_gap_frames,
                     frame_source=None, stats=None):
    """Fills interior track gaps between two detector-verified swapped
    observations. Gaps of <= MAX_INTERP_GAP interior frames interpolate;
    longer gaps require bidirectional tracking evidence via `frame_source`
    (frame_index -> gray image; None disables long-gap bridging entirely).
    Frames the plan already covers are skipped via existing_pairs, a set of
    (frame_index, track_id).

    Returns rows shaped like the analysis plan: (frame, track_id, number,
    kps, meta).
    """
    if stats is None:
        stats = {}
    for key in ("interp_rows", "tracked_rows", "frames_unverified",
                "frames_disagree", "gaps_unreadable"):
        stats.setdefault(key, 0)

    rows = []
    for track_id, a, b in _candidate_pairs(observation_log, max_gap_frames):
        span = b.frame_index - a.frame_index
        gid = a.accepted_group

        if span - 1 <= MAX_INTERP_GAP:
            for frame in range(a.frame_index + 1, b.frame_index):
                if (frame, track_id) in existing_pairs:
                    continue
                t = (frame - a.frame_index) / span
                kps = ((1 - t) * a.kps + t * b.kps).astype(np.float32)
                meta = _interp_meta(a, b, t,
                                    a.group_scores.get(gid), b.group_scores.get(gid),
                                    provenance="bridge", confidence=0.5)
                rows.append((frame, track_id, a.swapped_number, kps, meta))
                stats["interp_rows"] += 1
            continue

        if frame_source is None:
            continue
        agreed = _bridge_tracked(a, b, frame_source, stats)
        for frame, kps in sorted(agreed.items()):
            if (frame, track_id) in existing_pairs:
                continue
            t = (frame - a.frame_index) / span
            meta = _interp_meta(a, b, t,
                                a.group_scores.get(gid), b.group_scores.get(gid),
                                provenance="bridge", confidence=0.5)
            rows.append((frame, track_id, a.swapped_number, kps, meta))
            stats["tracked_rows"] += 1
    return rows


def summary(stats):
    return (
        f"anchor bridging: {stats['tracked_rows']} frames verified by "
        f"bidirectional tracking + {stats['interp_rows']} short-gap "
        f"interpolations; withheld {stats['frames_unverified']} unverified / "
        f"{stats['frames_disagree']} trajectory-disagreement frames"
        + (f"; {stats['gaps_unreadable']} gaps unreadable" if stats["gaps_unreadable"] else "")
    )
