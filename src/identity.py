"""Track-based identity decisions for the swap stage.

Replaces per-frame nearest-centroid classification with decisions that belong
to a face *track*:

- Enter/keep hysteresis: a track must clear the higher enter threshold to
  start swapping, then survives moderately degraded frames on the lower keep
  threshold. A borderline score alone never activates a swap -- it needs
  either a strong-confidence score (instant accept) or several consecutive
  confirming observations, which is exactly what a one-frame false positive
  (the elderly-extra bug) can't produce.
- Best-vs-second margin: a 0.61 is much safer when the runner-up scores 0.28
  than 0.59. Duplicate clusters of the same actor (same source photo dropped
  in under several numbers) are grouped by photo content hash so they don't
  count as each other's impostors.
- Explicit unknown: a face matching nothing well enough is left alone, never
  force-assigned to the nearest discovered character.
- Calibrated thresholds: discovery stores per-cluster genuine/impostor score
  percentiles (see discover_characters.py). Those distributions carry
  selection bias -- cluster membership was itself chosen by a similarity
  cut -- so they are treated as calibration evidence with safety margins,
  not ground truth, and .env values remain the fallback/override.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

import face_engine
from config import (
    MATCH_THRESHOLD, MAINTAIN_THRESHOLD, MIN_MARGIN, STRONG_ENTER_MARGIN,
    CONFIRM_FRAMES, REJECT_FRAMES, TRACK_MISS_LIMIT, IDENTITY_AUTO_CALIBRATION,
)

# Below this, cosine scores are noise regardless of what calibration says.
ABSOLUTE_SCORE_FLOOR = 0.30


# --- Source-photo grouping ---------------------------------------------------

def group_sources(source_dir):
    """Maps character number -> group id, where numbers whose source photos
    have identical file content share a group (the user's own fix for a
    cluster-split actor is dropping the same photo in under both numbers).
    """
    groups = {}
    for path in sorted(source_dir.glob("character*.jpg")):
        number = path.stem[len("character"):]
        if not number.isdigit():
            continue
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        groups[number] = digest
    return groups


# --- Threshold calibration ---------------------------------------------------

@dataclass
class GroupThresholds:
    enter: float
    keep: float
    strong: float
    source: str  # "calibrated" or "env"


def resolve_thresholds(groups, manifest):
    """Per-group enter/keep/strong thresholds.

    With calibration data present (and IDENTITY_AUTO_CALIBRATION on), enter is
    anchored just above the group's worst observed impostor tail; without it,
    the .env MATCH/MAINTAIN values apply unchanged.
    """
    group_members = {}
    for number, gid in groups.items():
        group_members.setdefault(gid, []).append(number)

    thresholds = {}
    for gid, members in group_members.items():
        calibrated = IDENTITY_AUTO_CALIBRATION and all(
            "calibration" in manifest.get(n, {}) for n in members
        )
        if calibrated:
            impostor = 0.0
            genuine_p10 = 0.0
            for n in members:
                cal = manifest[n]["calibration"]
                genuine_p10 = max(genuine_p10, cal["genuine_p10"])
                for other, p99 in cal["impostor_p99_by_cluster"].items():
                    if groups.get(other) != gid:
                        impostor = max(impostor, p99)
            enter = max(impostor + 0.03, ABSOLUTE_SCORE_FLOOR + 0.05)
            keep = max(impostor + 0.01, enter - 0.15, ABSOLUTE_SCORE_FLOOR)
            strong = enter + STRONG_ENTER_MARGIN
            thresholds[gid] = GroupThresholds(enter, keep, strong, "calibrated")
            if genuine_p10 < enter:
                print(
                    f"  [identity] group {sorted(members, key=int)}: weak "
                    f"genuine/impostor separation (genuine_p10={genuine_p10:.2f} "
                    f"< enter={enter:.2f}) -- activation will rely on strong "
                    "frames + confirmation"
                )
        else:
            thresholds[gid] = GroupThresholds(
                MATCH_THRESHOLD, MAINTAIN_THRESHOLD,
                MATCH_THRESHOLD + STRONG_ENTER_MARGIN, "env",
            )
    return thresholds


def describe_thresholds(groups, thresholds):
    group_members = {}
    for number, gid in groups.items():
        group_members.setdefault(gid, []).append(number)
    for gid, members in sorted(group_members.items(), key=lambda kv: int(kv[1][0])):
        t = thresholds[gid]
        names = "+".join(f"character{n}" for n in sorted(members, key=int))
        print(f"  [identity] {names}: enter={t.enter:.2f} keep={t.keep:.2f} "
              f"strong={t.strong:.2f} ({t.source})")


# --- Track state -------------------------------------------------------------

@dataclass
class _Track:
    kps: np.ndarray
    track_id: int = -1
    identity: str | None = None         # accepted group id
    identity_number: str | None = None  # concrete cluster number for sources[]
    keep_fails: int = 0
    pending_group: str | None = None
    pending_count: int = 0
    missing: int = 0
    scores: list = field(default_factory=list)  # recent accepted-group scores


def _center(kps):
    return np.asarray(kps).mean(axis=0)


def _radius(kps):
    kps = np.asarray(kps)
    spread = kps.max(axis=0) - kps.min(axis=0)
    return 1.5 * float(np.hypot(*spread)) if spread.any() else 40.0


class TrackIdentityManager:
    """Owns the per-track identity state machine. observe() is called once per
    full-detection frame with every detected face; it returns (kps,
    character_number, track_id) triples for the faces that should actually be
    swapped this frame. track_id is stable for the life of a physical face
    track and is what downstream per-face temporal state (plate matching)
    must be keyed by -- NOT character_number, which conflates every
    appearance of an identity across shots and simultaneous instances.
    """

    def __init__(self, centroids, sources, groups, thresholds):
        self.centroids = centroids
        self.sources = sources
        self.groups = groups                    # number -> group id
        self.thresholds = thresholds            # group id -> GroupThresholds
        self.members = {}                       # group id -> [numbers]
        for number, gid in groups.items():
            self.members.setdefault(gid, []).append(number)
        self.unmapped = [n for n in centroids if n not in groups]
        self._tracks: list[_Track] = []
        self._next_track_id = 0

    def reset(self):
        self._tracks = []

    # -- association --

    def _associate(self, faces):
        """Greedy nearest-center matching of detections to existing tracks."""
        pairs = []
        for fi, face in enumerate(faces):
            fc = _center(face.kps)
            for ti, track in enumerate(self._tracks):
                dist = float(np.linalg.norm(fc - _center(track.kps)))
                if dist < _radius(track.kps):
                    pairs.append((dist, fi, ti))
        pairs.sort(key=lambda p: p[0])
        face_to_track = {}
        used_tracks = set()
        for _, fi, ti in pairs:
            if fi in face_to_track or ti in used_tracks:
                continue
            face_to_track[fi] = ti
            used_tracks.add(ti)
        return face_to_track

    # -- scoring --

    def _score(self, face):
        scores = {
            n: face_engine.cosine_similarity(face.normed_embedding, c)
            for n, c in self.centroids.items()
        }
        group_scores = {}
        group_best_number = {}
        for gid, members in self.members.items():
            best_n = max(members, key=lambda n: scores[n])
            group_scores[gid] = scores[best_n]
            group_best_number[gid] = best_n
        unmapped_best = max((scores[n] for n in self.unmapped), default=-1.0)
        return scores, group_scores, group_best_number, unmapped_best

    def _competitor(self, group_scores, unmapped_best, exclude_gid):
        rival = max(
            (s for g, s in group_scores.items() if g != exclude_gid),
            default=-1.0,
        )
        return max(rival, unmapped_best)

    # -- the decision --

    def observe(self, faces, scene_cut=False, counts=None):
        if scene_cut:
            self._tracks = []

        face_to_track = self._associate(faces)
        swappable = []
        touched = set()

        for fi, face in enumerate(faces):
            ti = face_to_track.get(fi)
            if ti is None:
                track = _Track(kps=np.asarray(face.kps).copy(),
                               track_id=self._next_track_id)
                self._next_track_id += 1
                self._tracks.append(track)
            else:
                track = self._tracks[ti]
                track.kps = np.asarray(face.kps).copy()
                track.missing = 0
            touched.add(id(track))

            scores, group_scores, group_best_number, unmapped_best = self._score(face)
            swapped = False

            if track.identity is not None:
                gid = track.identity
                t = self.thresholds[gid]
                s = group_scores.get(gid, -1.0)
                competitor = self._competitor(group_scores, unmapped_best, gid)
                if s >= t.keep:
                    track.keep_fails = 0
                    track.identity_number = group_best_number[gid]
                    track.scores.append(s)
                    swapped = True
                elif competitor >= t.enter and competitor - s >= MIN_MARGIN:
                    # Clear evidence this face is someone else entirely --
                    # don't ride out the rejection streak, drop now.
                    track.identity = None
                    track.identity_number = None
                    track.keep_fails = 0
                else:
                    track.keep_fails += 1
                    if track.keep_fails >= REJECT_FRAMES:
                        track.identity = None
                        track.identity_number = None
                        track.keep_fails = 0
                    else:
                        swapped = True  # hysteresis: ride out a brief dip

            if track.identity is None and not swapped:
                best_gid = max(group_scores, key=group_scores.get, default=None)
                if best_gid is not None:
                    t = self.thresholds[best_gid]
                    s = group_scores[best_gid]
                    margin = s - self._competitor(group_scores, unmapped_best, best_gid)
                    qualified = (
                        s >= t.enter
                        and s >= ABSOLUTE_SCORE_FLOOR
                        and margin >= MIN_MARGIN
                    )
                    if qualified and s >= t.strong:
                        self._accept(track, best_gid, group_best_number[best_gid], s)
                        swapped = True
                    elif qualified:
                        if track.pending_group == best_gid:
                            track.pending_count += 1
                        else:
                            track.pending_group = best_gid
                            track.pending_count = 1
                        if track.pending_count >= CONFIRM_FRAMES:
                            self._accept(track, best_gid, group_best_number[best_gid], s)
                            swapped = True
                    else:
                        track.pending_group = None
                        track.pending_count = 0

            if swapped:
                swappable.append((face.kps, track.identity_number, track.track_id))
            elif counts is not None:
                best_n = max(scores, key=scores.get) if scores else None
                if best_n is not None and best_n in self.groups:
                    counts["unmatched_events"] += 1
                else:
                    counts["no_photo_events"] += 1

        # Age out tracks no detection claimed this pass.
        for track in self._tracks:
            if id(track) not in touched:
                track.missing += 1
        self._tracks = [t for t in self._tracks if t.missing <= TRACK_MISS_LIMIT]

        return swappable

    def _accept(self, track, gid, number, score):
        track.identity = gid
        track.identity_number = number
        track.pending_group = None
        track.pending_count = 0
        track.keep_fails = 0
        track.scores = [score]
