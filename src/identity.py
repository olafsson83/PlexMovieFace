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
import math
from dataclasses import dataclass, field

import numpy as np

import analysis_store
import face_engine
from config import (
    MATCH_THRESHOLD, MAINTAIN_THRESHOLD, MIN_MARGIN, STRONG_ENTER_MARGIN,
    CONFIRM_FRAMES, REJECT_FRAMES, TRACK_MISS_LIMIT, IDENTITY_AUTO_CALIBRATION,
    THRESHOLDS_EXPLICIT, PROVEN_TRACK_MISS_LIMIT, PENDING_WINDOW,
)

# Below this, cosine scores are noise regardless of what calibration says.
ABSOLUTE_SCORE_FLOOR = 0.30


def prototype_max_score(embedding, prototypes):
    """Identity score against a prototype bank: best match over the bank
    (a profile face matches the profile prototype instead of being dragged
    down by a frontal-dominated mean). Accepts a single centroid vector for
    projects that haven't been recalibrated yet."""
    protos = np.atleast_2d(np.asarray(prototypes, dtype=np.float32))
    emb = np.asarray(embedding, dtype=np.float32)
    return float((protos @ emb).max())


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
        calibrated = (
            IDENTITY_AUTO_CALIBRATION
            and not THRESHOLDS_EXPLICIT  # operator's explicit .env wins
            and all("calibration" in manifest.get(n, {}) for n in members)
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
    pending_age: int = 0
    missing: int = 0
    scores: list = field(default_factory=list)  # recent accepted-group scores
    embedding: np.ndarray | None = None  # last observation's embedding
    needs_proof: bool = False            # just reacquired after a gap: must
                                         # clear KEEP before the ride-out applies

    @property
    def proven(self):
        """A track whose identity is backed by several strong observations
        earns longer survival through detection gaps (dark stretches,
        occlusion) -- see PROVEN_TRACK_MISS_LIMIT."""
        return self.identity is not None and len(self.scores) >= 3

    @property
    def miss_limit(self):
        return PROVEN_TRACK_MISS_LIMIT if self.proven else TRACK_MISS_LIMIT


@dataclass
class Observation:
    """One detection-frame sighting of a track, recorded during analysis so
    a later pass can revise decisions with evidence from the track's whole
    life (e.g. retroactive backfill of pre-confirmation frames)."""
    frame_index: int
    track_id: int
    kps: np.ndarray
    group_scores: dict           # group id -> best member score this frame
    swapped_number: str | None   # number swapped this frame, else None
    accepted_group: str | None   # track's accepted group AFTER this frame
    renderable: bool = True      # legacy field; pose gating now happens at render
    yaw: float | None = None     # buffalo_l 3D-landmark pose, when available
    pitch: float | None = None
    roll: float | None = None
    det_score: float | None = None
    margin: float | None = None  # accepted-group score minus best competitor


def _face_pose(face):
    """(pitch, yaw, roll) from buffalo_l's 3D landmark model, or None."""
    pose = getattr(face, "pose", None)
    if pose is None:
        return None
    return float(pose[0]), float(pose[1]), float(pose[2])


def _face_yaw(face):
    pose = _face_pose(face)
    return None if pose is None else pose[1]


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
        self.record_observations = False
        self.observation_log: list[Observation] = []

    def reset(self):
        self._tracks = []

    # -- association --

    # Combined-cost weights: position dominates, embedding breaks ties when
    # two people are close (the crossing-actors case greedy distance gets
    # wrong), scale guards against a near face grabbing a far track.
    W_POSITION = 1.0
    W_EMBEDDING = 1.5
    W_SCALE = 0.5
    MAX_ASSIGN_COST = 2.0

    def _pair_cost(self, face, track):
        dist = float(np.linalg.norm(_center(face.kps) - _center(track.kps)))
        radius = _radius(track.kps)
        if dist >= radius:
            return None  # gate: outside plausible motion since last sighting
        cost = self.W_POSITION * (dist / radius)
        if track.embedding is not None:
            sim = float(np.dot(face.normed_embedding, track.embedding))
            cost += self.W_EMBEDDING * max(0.0, 1.0 - sim)
        scale_ratio = _radius(face.kps) / max(radius, 1e-6)
        cost += self.W_SCALE * abs(float(np.log(max(scale_ratio, 1e-6))))
        return cost

    def _associate(self, faces):
        """Globally optimal detection-to-track assignment (Hungarian) over a
        combined position/embedding/scale cost, gated per pair. Greedy
        nearest-center mis-assigns when two people pass close to each other;
        the embedding term keeps identities on their own tracks.

        Uncontested pairs (one face, one gated track, mutually exclusive
        alternatives) skip the embedding/scale terms: embeddings exist to
        resolve COMPETITION, and in dark scenes a lone face's junk embedding
        must not break the continuity position alone supports (measured:
        dark-fixture coverage dropped 4% without this carve-out).
        """
        if not faces or not self._tracks:
            return {}
        cost = np.full((len(faces), len(self._tracks)), np.inf)
        position_ok = np.zeros((len(faces), len(self._tracks)), dtype=bool)
        for fi, face in enumerate(faces):
            for ti, track in enumerate(self._tracks):
                c = self._pair_cost(face, track)
                if c is not None:
                    cost[fi, ti] = c
                    position_ok[fi, ti] = True
        if not position_ok.any():
            return {}

        # Uncontested carve-out: face fi's only gated track is ti, and track
        # ti gates no other face -> assign on position evidence alone.
        for fi in range(len(faces)):
            gated = np.where(position_ok[fi])[0]
            if len(gated) == 1 and position_ok[:, gated[0]].sum() == 1:
                cost[fi, gated[0]] = min(cost[fi, gated[0]], self.MAX_ASSIGN_COST)

        from scipy.optimize import linear_sum_assignment
        finite_max = cost[np.isfinite(cost)].max()
        solver_cost = np.where(np.isfinite(cost), cost, finite_max + self.MAX_ASSIGN_COST + 1)
        rows, cols = linear_sum_assignment(solver_cost)
        return {
            int(fi): int(ti)
            for fi, ti in zip(rows, cols)
            if np.isfinite(cost[fi, ti]) and cost[fi, ti] <= self.MAX_ASSIGN_COST
        }

    # -- scoring --

    def _score(self, face):
        scores = {
            n: prototype_max_score(face.normed_embedding, c)
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

    def observe(self, faces, scene_cut=False, counts=None, frame_index=None):
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
                if track.missing > 0:
                    # Reacquired after a gap: someone else could have taken
                    # this spot, so swapping resumes only once the face
                    # scores >= KEEP again (no below-keep ride-out).
                    track.needs_proof = True
                track.missing = 0
            track.embedding = np.asarray(face.normed_embedding, dtype=np.float32)
            touched.add(id(track))

            # Pose is recorded as row evidence, never used to discard an
            # identity-certain observation here: whether a given yaw is
            # renderable depends on the swap backend, which analysis cannot
            # know. The render pass gates per backend (render_movie.py).
            pose = _face_pose(face)

            scores, group_scores, group_best_number, unmapped_best = self._score(face)
            swapped = False
            swap_score = None
            swap_margin = None

            if track.identity is not None:
                gid = track.identity
                t = self.thresholds[gid]
                s = group_scores.get(gid, -1.0)
                competitor = self._competitor(group_scores, unmapped_best, gid)
                if s >= t.keep:
                    track.keep_fails = 0
                    track.needs_proof = False
                    track.identity_number = group_best_number[gid]
                    track.scores.append(s)
                    del track.scores[:-8]  # bounded history for `proven`
                    swapped = True
                    swap_score, swap_margin = s, s - competitor
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
                    elif not track.needs_proof:
                        swapped = True  # hysteresis: ride out a brief dip
                        swap_score, swap_margin = s, s - competitor

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
                        swap_score, swap_margin = s, margin
                    elif qualified:
                        if track.pending_group == best_gid:
                            track.pending_count += 1
                        else:
                            # A different identity qualifying IS contradicting
                            # evidence; restart confirmation for the new one.
                            track.pending_group = best_gid
                            track.pending_count = 1
                            track.pending_age = 0
                        if track.pending_count >= CONFIRM_FRAMES:
                            self._accept(track, best_gid, group_best_number[best_gid], s)
                            swapped = True
                            swap_score, swap_margin = s, margin
                    elif track.pending_group is not None:
                        # Unqualified frames (blur dipping under the enter
                        # bar) merely AGE a pending candidate instead of
                        # resetting it -- oscillation around the threshold
                        # shouldn't restart confirmation from zero. The
                        # window bounds how long stale pendings survive.
                        track.pending_age += 1
                        if track.pending_age > PENDING_WINDOW:
                            track.pending_group = None
                            track.pending_count = 0
                            track.pending_age = 0

            if swapped:
                keep = self.thresholds[track.identity].keep
                meta = analysis_store.make_meta(
                    pitch=None if pose is None else pose[0],
                    yaw=None if pose is None else pose[1],
                    roll=None if pose is None else pose[2],
                    det_score=float(getattr(face, "det_score", math.nan)),
                    identity_score=swap_score,
                    margin=swap_margin,
                    provenance="detector",
                    # Ride-out frames (below keep, surviving on hysteresis)
                    # are weaker evidence than keep-cleared ones.
                    confidence=1.0 if swap_score is not None and swap_score >= keep else 0.7,
                )
                swappable.append((face.kps, track.identity_number, track.track_id, meta))
            elif counts is not None:
                best_n = max(scores, key=scores.get) if scores else None
                if best_n is not None and best_n in self.groups:
                    counts["unmatched_events"] += 1
                else:
                    counts["no_photo_events"] += 1

            if self.record_observations and frame_index is not None:
                self.observation_log.append(Observation(
                    frame_index=frame_index,
                    track_id=track.track_id,
                    kps=np.asarray(face.kps, dtype=np.float32).copy(),
                    group_scores=dict(group_scores),
                    swapped_number=track.identity_number if swapped else None,
                    accepted_group=track.identity,
                    yaw=None if pose is None else pose[1],
                    pitch=None if pose is None else pose[0],
                    roll=None if pose is None else pose[2],
                    det_score=float(getattr(face, "det_score", math.nan)),
                    margin=swap_margin,
                ))

        # Age out tracks no detection claimed this pass.
        for track in self._tracks:
            if id(track) not in touched:
                track.missing += 1
        self._tracks = [t for t in self._tracks if t.missing <= t.miss_limit]

        return swappable

    def _accept(self, track, gid, number, score):
        track.identity = gid
        track.identity_number = number
        track.pending_group = None
        track.pending_count = 0
        track.pending_age = 0
        track.keep_fails = 0
        track.needs_proof = False
        track.scores = [score]


def _lerp(a, b, t):
    """Linear interpolation tolerant of missing evidence (None/NaN)."""
    if a is None or b is None:
        return None
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return None
    return (1 - t) * a + t * b


def _interp_meta(a, b, t, score_a, score_b, provenance, confidence):
    """Row meta for a frame interpolated between two anchor Observations."""
    return analysis_store.make_meta(
        yaw=_lerp(a.yaw, b.yaw, t),
        pitch=_lerp(a.pitch, b.pitch, t),
        roll=_lerp(a.roll, b.roll, t),
        identity_score=_lerp(score_a, score_b, t),
        margin=_lerp(a.margin, b.margin, t),
        provenance=provenance,
        confidence=confidence,
    )


# --- retroactive backfill (future evidence) -----------------------------------

def backfill_swap_rows(observation_log, thresholds, max_gap_frames):
    """Extends each eventually-accepted track's swap range backward over its
    pre-confirmation observations -- the confirmation gate costs genuine
    tracks their first few detection frames, and only a pass that has seen
    the track's whole life can safely give them back.

    A pre-acceptance observation is backfilled only when its own score
    against the eventually-accepted group already cleared that group's KEEP
    bar (the same evidence standard an accepted track needs to stay
    swapped), walking backward contiguously from the acceptance point and
    stopping at the first failure or a gap longer than max_gap_frames. A
    track that never got accepted (the elderly-extra shape) contributes
    nothing. Frames between detection observations get linearly
    interpolated landmarks -- the same 5-frame-scale gap optical flow
    bridges on the forward path.

    Returns rows shaped like the analysis plan: (frame, track_id, number,
    kps, meta).
    """
    by_track = {}
    for obs in observation_log:
        by_track.setdefault(obs.track_id, []).append(obs)

    rows = []
    for track_id, obs_list in by_track.items():
        obs_list.sort(key=lambda o: o.frame_index)
        accept_i = next(
            (i for i, o in enumerate(obs_list) if o.swapped_number is not None),
            None,
        )
        if accept_i is None or accept_i == 0:
            continue
        gid = obs_list[accept_i].accepted_group
        number = obs_list[accept_i].swapped_number
        keep = thresholds[gid].keep

        anchors = [obs_list[accept_i]]
        for obs in reversed(obs_list[:accept_i]):
            score = obs.group_scores.get(gid, -1.0)
            if score < keep or not obs.renderable:
                break  # weak evidence or pose-gated: don't fill across it
            if anchors[0].frame_index - obs.frame_index > max_gap_frames:
                break
            anchors.insert(0, obs)
        if len(anchors) < 2:
            continue

        for a, b in zip(anchors[:-1], anchors[1:]):
            span = b.frame_index - a.frame_index
            for frame in range(a.frame_index, b.frame_index):
                if b.swapped_number is not None and frame == b.frame_index:
                    continue  # acceptance frame is already in the plan
                t = (frame - a.frame_index) / span if span else 0.0
                kps = (1 - t) * a.kps + t * b.kps
                meta = _interp_meta(a, b, t,
                                    a.group_scores.get(gid), b.group_scores.get(gid),
                                    provenance="backfill", confidence=0.6)
                rows.append((frame, track_id, number, kps.astype(np.float32), meta))
    return rows


# Gap bridging lives in bridging.py: long gaps are verified by
# bidirectional optical tracking through the same safety gates as the
# live tracker (linear interpolation restored rows through intervals
# where flow was REJECTED, undoing the scramble protection);
# interpolation survives only for gaps of at most two frames.
