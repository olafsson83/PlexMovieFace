"""Unit tests for pose handling. Since plan v3 the analysis pass records
pose as row evidence and never discards identity-certain extreme-pose
observations; the RENDER pass gates per selected backend (RenderPoseGate),
with per-track hysteresis. Run:

    python -m unittest tests.test_pose_gate
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from identity import GroupThresholds, Observation, TrackIdentityManager, backfill_swap_rows
from swap_backend import RenderPoseGate
from tracking import TrackedFace

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


class AnalysisPoseEvidenceTests(unittest.TestCase):
    def test_extreme_yaw_row_is_kept_with_pose_evidence(self):
        # Analysis cannot know the backend, so an identity-certain profile
        # face must reach the plan -- with its yaw recorded for the render
        # gate to act on.
        mgr = manager()
        mgr.observe([FakeFace(0.9, yaw=0)])
        out = mgr.observe([FakeFace(0.9, yaw=80)])
        self.assertEqual(len(out), 1)
        kps, number, track_id, meta = out[0]
        self.assertAlmostEqual(meta["yaw"], 80.0, places=4)
        self.assertEqual(meta["provenance"], "detector")
        self.assertAlmostEqual(meta["det_score"], 0.9, places=4)
        self.assertGreater(meta["identity_score"], 0.85)
        self.assertGreater(meta["margin"], 0.0)

    def test_faces_without_pose_get_nan_yaw(self):
        mgr = manager()
        face = FakeFace(0.9, yaw=0)
        del face.pose
        out = mgr.observe([face])
        self.assertEqual(len(out), 1)
        self.assertTrue(np.isnan(out[0][3]["yaw"]))


class FakeBackend:
    def __init__(self, reliable=65):
        self.reliable = reliable

    def capabilities(self):
        return {"name": "fake", "reliable_abs_yaw": self.reliable}


def face_with_yaw(yaw, track_id=1):
    meta = {"yaw": yaw, "provenance": "detector"}
    return TrackedFace(KPS, "1", track_id, meta)


class RenderPoseGateTests(unittest.TestCase):
    def test_gate_withholds_beyond_backend_capability(self):
        gate = RenderPoseGate(FakeBackend(reliable=65))
        gate.enabled, gate.limit = True, 65.0
        self.assertTrue(gate.renderable(face_with_yaw(30)))
        self.assertFalse(gate.renderable(face_with_yaw(80)))
        self.assertEqual(gate.withheld, 1)

    def test_min_hold_hysteresis_near_the_limit(self):
        # Blocking is immediate; unblocking needs the blocked stint to have
        # lasted POSE_MIN_HOLD (3) rows -- jitter can't strobe the swap.
        gate = RenderPoseGate(FakeBackend(reliable=65))
        gate.enabled, gate.limit = True, 65.0
        self.assertFalse(gate.renderable(face_with_yaw(80)))   # blocked (stint 1)
        self.assertFalse(gate.renderable(face_with_yaw(60)))   # stint 2
        self.assertFalse(gate.renderable(face_with_yaw(60)))   # stint 3
        self.assertTrue(gate.renderable(face_with_yaw(60)))    # held long enough
        self.assertTrue(gate.renderable(face_with_yaw(60)))

    def test_rows_without_pose_keep_track_state(self):
        gate = RenderPoseGate(FakeBackend(reliable=65))
        gate.enabled, gate.limit = True, 65.0
        self.assertFalse(gate.renderable(face_with_yaw(80, track_id=3)))
        # A flow row without fresh pose inherits its track's blocked state...
        flow = TrackedFace(KPS, "1", 3, {"yaw": float("nan"), "provenance": "flow"})
        self.assertFalse(gate.renderable(flow))
        # ...and an unblocked track's pose-less rows render.
        other = TrackedFace(KPS, "1", 4, None)
        self.assertTrue(gate.renderable(other))

    def test_disabled_gate_passes_everything(self):
        gate = RenderPoseGate(FakeBackend(reliable=65))
        gate.enabled = False
        self.assertTrue(gate.renderable(face_with_yaw(89)))


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
