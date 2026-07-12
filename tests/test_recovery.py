"""Unit tests for hit-rate steps 3-4: ROI re-detection mapping, proven-track
survival, reacquisition proof, and the pending confirmation window. Run:

    python -m unittest tests.test_recovery
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import identity as identity_mod
from identity import GroupThresholds, TrackIdentityManager

DIM = 32
KPS = np.array([[45, 45], [55, 45], [50, 50], [45, 55], [55, 55]], dtype=np.float32)


def basis(i):
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


CENTROIDS = {"1": basis(0), "13": basis(2)}


class FakeFace:
    def __init__(self, sims, kps=KPS):
        emb = np.zeros(DIM, dtype=np.float32)
        for number, s in sims.items():
            emb += s * CENTROIDS[number]
        rest = 1.0 - float(np.sum(emb ** 2))
        if rest > 0:
            emb[DIM - 1] = np.sqrt(rest)
        self.normed_embedding = emb / np.linalg.norm(emb)
        self.kps = kps.copy()
        self.det_score = 0.9


def manager(enter=0.45, keep=0.25, strong=0.57):
    groups = {"1": "G"}
    thresholds = {"G": GroupThresholds(enter, keep, strong, "env")}
    return TrackIdentityManager(CENTROIDS, {"1": object()}, groups, thresholds)


def establish(mgr, n=3):
    for _ in range(n):
        out = mgr.observe([FakeFace({"1": 0.9})])
    return out


class ProvenSurvivalTests(unittest.TestCase):
    def test_proven_track_survives_long_gap_and_resumes_on_enter_evidence(self):
        mgr = manager()
        establish(mgr)  # 3 strong observations -> proven
        for _ in range(4):  # gap longer than TRACK_MISS_LIMIT(2), within proven(6)
            mgr.observe([])
        self.assertEqual(len(mgr._tracks), 1)
        # Re-detection at 0.47: clears enter (0.45) with margin, so the
        # long-gap proof requirement is satisfied and swapping resumes.
        out = mgr.observe([FakeFace({"1": 0.47})])
        self.assertEqual(len(out), 1)

    def test_unproven_track_still_dies_at_the_short_limit(self):
        mgr = manager()
        mgr.observe([FakeFace({"1": 0.9})])  # accepted, but only 1 strong hit
        for _ in range(3):
            mgr.observe([])
        self.assertEqual(mgr._tracks, [])

    def test_keep_level_ride_through_never_earns_proven(self):
        # The old definition counted ANY keep-level score toward proven
        # (len(scores) >= 3) despite documentation claiming strong
        # evidence. 0.30 clears keep=0.25 but not enter=0.45: two of them
        # after a strong acceptance must NOT make the track proven.
        mgr = manager()
        mgr.observe([FakeFace({"1": 0.9})])    # strong accept (hit 1)
        mgr.observe([FakeFace({"1": 0.30})])   # keep-level, still swapped
        mgr.observe([FakeFace({"1": 0.30})])
        self.assertEqual(mgr._tracks[0].strong_hits, 1)
        self.assertFalse(mgr._tracks[0].proven)
        for _ in range(3):  # dies at the SHORT limit
            mgr.observe([])
        self.assertEqual(mgr._tracks, [])

    def test_long_gap_reacquisition_demands_enter_level_evidence(self):
        # After a gap past TRACK_MISS_LIMIT, keep-level (0.30) evidence
        # must not resume the swap -- someone else had time to take the
        # spot. Enter-level with margin resumes.
        mgr = manager()
        establish(mgr)
        for _ in range(4):
            mgr.observe([])
        out = mgr.observe([FakeFace({"1": 0.30})])  # keep-level only
        self.assertEqual(out, [])
        out = mgr.observe([FakeFace({"1": 0.50})])  # enter + margin
        self.assertEqual(len(out), 1)

    def test_position_only_association_cannot_reacquire_missing_track(self):
        # A face at the old spot whose embedding actively contradicts the
        # track's must not be claimed by the uncontested position-only
        # carve-out once the track has been missing -- it becomes a new
        # track instead (and is never swapped on the old track's identity).
        mgr = manager()
        establish(mgr)
        old_id = mgr._tracks[0].track_id
        for _ in range(3):
            mgr.observe([])
        # Embedding maximally contradicts the track's: pair cost exceeds
        # MAX_ASSIGN_COST, so only the (formerly unconditional) position
        # carve-out could have claimed it.
        stranger = FakeFace({"1": 0.0})
        stranger.normed_embedding = -mgr._tracks[0].embedding
        out = mgr.observe([stranger])
        self.assertEqual(out, [])
        self.assertTrue(any(t.track_id != old_id for t in mgr._tracks))


class ReacquisitionProofTests(unittest.TestCase):
    def test_reacquired_track_gets_no_ride_out_below_keep(self):
        mgr = manager()
        establish(mgr)
        for _ in range(3):
            mgr.observe([])  # gap
        # Someone ELSE could now be at this spot: a below-keep face must NOT
        # be swapped on the old track's credit (the ride-out is withheld).
        out = mgr.observe([FakeFace({"1": 0.10, "13": 0.2})])
        self.assertEqual(out, [])
        # A keep-clearing observation restores normal behavior.
        out = mgr.observe([FakeFace({"1": 0.47})])
        self.assertEqual(len(out), 1)
        out = mgr.observe([FakeFace({"1": 0.10})])  # ride-out active again
        self.assertEqual(len(out), 1)


class PendingWindowTests(unittest.TestCase):
    def test_oscillating_borderline_scores_still_confirm(self):
        # 0.47/0.40/0.47/0.40/0.47 around enter=0.45: the old logic reset
        # pending on every dip and never confirmed.
        mgr = manager()
        seq = [0.47, 0.40, 0.47, 0.40, 0.47]
        results = [mgr.observe([FakeFace({"1": s})]) for s in seq]
        self.assertEqual([len(r) for r in results[:-1]], [0, 0, 0, 0])
        self.assertEqual(len(results[-1]), 1)  # 3rd qualified hit confirms

    def test_pending_expires_after_window(self):
        mgr = manager()
        mgr.observe([FakeFace({"1": 0.47})])  # pending 1/3
        for _ in range(7):  # > PENDING_WINDOW unqualified observations
            mgr.observe([FakeFace({"1": 0.30})])
        track = mgr._tracks[0]
        self.assertIsNone(track.pending_group)

    def test_rival_identity_still_resets_pending(self):
        groups = {"1": "G", "13": "H"}
        thresholds = {"G": GroupThresholds(0.45, 0.25, 0.57, "env"),
                      "H": GroupThresholds(0.45, 0.25, 0.57, "env")}
        mgr = TrackIdentityManager(CENTROIDS, {"1": object(), "13": object()},
                                   groups, thresholds)
        mgr.observe([FakeFace({"1": 0.47})])          # pending G 1/3
        mgr.observe([FakeFace({"13": 0.50})])         # rival qualifies -> reset to H
        track = mgr._tracks[0]
        self.assertEqual(track.pending_group, "H")
        self.assertEqual(track.pending_count, 1)


class RoiMappingTests(unittest.TestCase):
    def test_roi_coordinates_map_back_to_frame_space(self):
        from adaptive_detection import AdaptiveDetector

        class FakeDet:
            def detect(self, img, input_size=None, max_num=0, metric="default"):
                # One face at crop-space (upscaled) position (60, 60)-(100, 100).
                bboxes = np.array([[60, 60, 100, 100, 0.8]], dtype=np.float32)
                kpss = np.array([[[80, 80]] * 5], dtype=np.float32)
                return bboxes, kpss

        class FakeApp:
            det_model = FakeDet()
            models = {"detection": None}

            def get(self, frame):
                return []

        det = AdaptiveDetector(FakeApp(), enabled=True)
        frame = np.full((300, 300, 3), 40, dtype=np.uint8)
        center = np.array([150.0, 150.0])
        faces = det._detect_roi(frame, center, radius=30.0)
        self.assertEqual(len(faces), 1)
        # crop starts at 150-60=90; upscale 2.0 -> kps 80/2 + 90 = 130
        np.testing.assert_allclose(faces[0].kps[0], [130.0, 130.0], atol=0.01)
        np.testing.assert_allclose(faces[0].bbox[:2], [120.0, 120.0], atol=0.01)


if __name__ == "__main__":
    unittest.main()
