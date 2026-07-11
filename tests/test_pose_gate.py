"""Unit tests for v2 milestone 6: pose gating / unrenderable frames. Run:

    python -m unittest tests.test_pose_gate
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from identity import GroupThresholds, Observation, TrackIdentityManager, backfill_swap_rows

DIM = 32
KPS = np.array([[45, 45], [55, 45], [50, 50], [45, 55], [55, 55]], dtype=np.float32)


def basis(i):
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


CENTROIDS = {"1": basis(0)}


class FakeFace:
    def __init__(self, score, yaw=0.0, kps=KPS):
        emb = score * CENTROIDS["1"]
        rest = 1.0 - float(np.sum(emb ** 2))
        if rest > 0:
            emb[DIM - 1] = np.sqrt(rest)
        self.normed_embedding = emb / np.linalg.norm(emb)
        self.kps = kps.copy()
        self.det_score = 0.9
        self.pose = np.array([0.0, yaw, 0.0], dtype=np.float32)  # pitch, yaw, roll


def manager():
    groups = {"1": "G"}
    thresholds = {"G": GroupThresholds(0.7, 0.6, 0.82, "env")}
    return TrackIdentityManager(CENTROIDS, {"1": object()}, groups, thresholds)


class PoseGateTests(unittest.TestCase):
    def test_extreme_yaw_withholds_swap_but_keeps_identity(self):
        mgr = manager()
        counts = {"unmatched_events": 0, "no_photo_events": 0}
        self.assertEqual(len(mgr.observe([FakeFace(0.9, yaw=0)], counts=counts)), 1)
        # Head turns past the limit: no swap emitted, but no re-confirmation
        # needed when it comes back.
        out = mgr.observe([FakeFace(0.9, yaw=80)], counts=counts)
        self.assertEqual(out, [])
        self.assertEqual(counts.get("unrenderable_events"), 1)
        out = mgr.observe([FakeFace(0.9, yaw=10)], counts=counts)
        self.assertEqual(len(out), 1)  # instant resume, identity was retained

    def test_hysteresis_holds_near_the_limit(self):
        mgr = manager()
        mgr.observe([FakeFace(0.9, yaw=0)])
        mgr.observe([FakeFace(0.9, yaw=80)])         # blocked
        out = mgr.observe([FakeFace(0.9, yaw=60)])   # inside limit but not exit margin
        self.assertEqual(out, [])                    # still blocked (65 - 8 = 57)
        out = mgr.observe([FakeFace(0.9, yaw=50)])   # safely inside
        self.assertEqual(len(out), 1)

    def test_faces_without_pose_are_unaffected(self):
        mgr = manager()
        face = FakeFace(0.9, yaw=0)
        del face.pose
        self.assertEqual(len(mgr.observe([face])), 1)


THRESHOLDS = {"G1": GroupThresholds(0.7, 0.6, 0.82, "env")}


def obs(frame, score, swapped, renderable=True):
    return Observation(
        frame_index=frame, track_id=5, kps=KPS,
        group_scores={"G1": score},
        swapped_number="1" if swapped else None,
        accepted_group="G1" if swapped else None,
        renderable=renderable,
    )


class BackfillPoseTests(unittest.TestCase):
    def test_backfill_stops_at_unrenderable_observation(self):
        log = [
            obs(0, 0.75, False),
            obs(5, 0.75, False, renderable=False),  # profile: don't fill across
            obs(10, 0.75, False),
            obs(15, 0.73, True),
        ]
        rows = backfill_swap_rows(log, THRESHOLDS, max_gap_frames=15)
        frames = sorted(r[0] for r in rows)
        self.assertEqual(frames, list(range(10, 15)))


if __name__ == "__main__":
    unittest.main()
