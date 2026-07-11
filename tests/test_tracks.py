"""Unit tests for v2 milestone 4: Hungarian track association and
retroactive backfill. Run:

    python -m unittest tests.test_tracks
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import identity
from identity import (
    GroupThresholds, Observation, TrackIdentityManager, backfill_swap_rows,
)

DIM = 32


def basis(i):
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


CENTROIDS = {"1": basis(0), "2": basis(1)}
KPS_A = np.array([[45, 45], [55, 45], [50, 50], [45, 55], [55, 55]], dtype=np.float32)
KPS_B = KPS_A + np.array([14.0, 0.0])  # close enough to cross into A's gate


class FakeFace:
    def __init__(self, sims, kps):
        emb = np.zeros(DIM, dtype=np.float32)
        for number, s in sims.items():
            emb += s * CENTROIDS[number]
        rest = 1.0 - float(np.sum(emb ** 2))
        if rest > 0:
            emb[DIM - 1] = np.sqrt(rest)
        self.normed_embedding = emb / np.linalg.norm(emb)
        self.kps = kps.copy()
        self.det_score = 0.9


def manager():
    groups = {"1": "G1", "2": "G2"}
    thresholds = {
        "G1": GroupThresholds(0.7, 0.6, 0.82, "env"),
        "G2": GroupThresholds(0.7, 0.6, 0.82, "env"),
    }
    sources = {"1": object(), "2": object()}
    return TrackIdentityManager(CENTROIDS, sources, groups, thresholds)


class AssociationTests(unittest.TestCase):
    def test_crossing_faces_keep_their_tracks(self):
        """Two nearby faces swap positions; greedy nearest-center would keep
        each track at its old spot (wrong identity), the embedding term must
        keep each identity on its own track."""
        mgr = manager()
        a = FakeFace({"1": 0.9}, KPS_A)
        b = FakeFace({"2": 0.9}, KPS_B)
        out = mgr.observe([a, b])
        ids = {n: tid for _, n, tid in out}
        self.assertEqual(len(out), 2)

        # Positions cross; embeddings stay with their people.
        a2 = FakeFace({"1": 0.9}, KPS_B)
        b2 = FakeFace({"2": 0.9}, KPS_A)
        out2 = mgr.observe([a2, b2])
        ids2 = {n: tid for _, n, tid in out2}
        self.assertEqual(ids2["1"], ids["1"])  # same physical track for "1"
        self.assertEqual(ids2["2"], ids["2"])

    def test_far_face_not_assigned(self):
        mgr = manager()
        mgr.observe([FakeFace({"1": 0.9}, KPS_A)])
        out = mgr.observe([FakeFace({"1": 0.9}, KPS_A + 500.0)])
        # New location -> new track id (old one is out of the gate).
        self.assertEqual(len(out), 1)
        self.assertEqual(len(mgr._tracks), 2)


def obs(frame, track_id, score, swapped, kps_shift=0.0, gid="G1", number="1"):
    return Observation(
        frame_index=frame, track_id=track_id,
        kps=KPS_A + kps_shift,
        group_scores={gid: score},
        swapped_number=number if swapped else None,
        accepted_group=gid if swapped else None,
    )


THRESHOLDS = {"G1": GroupThresholds(0.7, 0.6, 0.82, "env")}


class BackfillTests(unittest.TestCase):
    def test_confirmed_track_backfills_pre_confirmation_frames(self):
        log = [
            obs(0, 5, 0.72, False),           # pending 1/3
            obs(5, 5, 0.71, False, 2.0),      # pending 2/3
            obs(10, 5, 0.73, True, 4.0),      # accepted here
        ]
        rows = backfill_swap_rows(log, THRESHOLDS, max_gap_frames=15)
        frames = sorted(r[0] for r in rows)
        self.assertEqual(frames, list(range(0, 10)))  # 0..9 recovered
        for frame, track_id, number, kps in rows:
            self.assertEqual((track_id, number), (5, "1"))
        # Interpolation: frame 5's kps equals the recorded observation's.
        kps5 = next(r[3] for r in rows if r[0] == 5)
        np.testing.assert_allclose(kps5, KPS_A + 2.0, atol=1e-5)

    def test_never_accepted_track_contributes_nothing(self):
        log = [obs(0, 7, 0.72, False), obs(5, 7, 0.30, False)]
        self.assertEqual(backfill_swap_rows(log, THRESHOLDS, 15), [])

    def test_backfill_stops_at_below_keep_observation(self):
        log = [
            obs(0, 5, 0.72, False),
            obs(5, 5, 0.40, False),   # dip below keep: backfill must stop here
            obs(10, 5, 0.72, False),
            obs(15, 5, 0.73, True),
        ]
        rows = backfill_swap_rows(log, THRESHOLDS, max_gap_frames=15)
        frames = sorted(r[0] for r in rows)
        self.assertEqual(frames, list(range(10, 15)))  # only the clean stretch

    def test_backfill_respects_gap_limit(self):
        log = [
            obs(0, 5, 0.75, False),
            obs(40, 5, 0.73, True),  # 40-frame gap: track was gone too long
        ]
        self.assertEqual(backfill_swap_rows(log, THRESHOLDS, max_gap_frames=15), [])


if __name__ == "__main__":
    unittest.main()
