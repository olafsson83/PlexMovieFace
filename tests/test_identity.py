"""Unit tests for the track-based identity state machine (src/identity.py).

Uses synthetic orthonormal centroids so cosine scores are directly
controllable; no models, GPU, or video needed. Run:

    python -m unittest tests.test_identity
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import identity
from identity import GroupThresholds, TrackIdentityManager

DIM = 32
KPS_A = np.array([[45, 45], [55, 45], [50, 50], [45, 55], [55, 55]], dtype=np.float32)
KPS_FAR = KPS_A + 500.0


def basis(i):
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


# character1/character24 are the same actor (grouped); character13 is a
# discovered-but-unmapped cluster (e.g. the elderly extra's real identity).
CENTROIDS = {"1": basis(0), "24": basis(1), "13": basis(2), "2": basis(3)}


class FakeFace:
    def __init__(self, sims, kps=KPS_A):
        """sims: {number: target cosine similarity vs that centroid}."""
        emb = np.zeros(DIM, dtype=np.float32)
        for number, s in sims.items():
            emb += s * CENTROIDS[number]
        rest = 1.0 - float(np.sum(emb ** 2))
        if rest > 0:  # pad into an unused dimension so the norm is 1
            emb[DIM - 1] = np.sqrt(rest)
        self.normed_embedding = emb / np.linalg.norm(emb)
        self.kps = kps.copy()
        self.det_score = 0.9


def manager(sources=("1", "24"), enter=0.7, keep=0.6, strong=0.82,
            second_group=None):
    groups = {n: "G" for n in sources}
    thresholds = {"G": GroupThresholds(enter, keep, strong, "env")}
    if second_group:
        for n in second_group:
            groups[n] = "H"
        thresholds["H"] = GroupThresholds(enter, keep, strong, "env")
    fake_sources = {n: object() for n in groups}
    return TrackIdentityManager(CENTROIDS, fake_sources, groups, thresholds)


class IdentityDecisionTests(unittest.TestCase):
    def test_strong_score_swaps_immediately(self):
        mgr = manager()
        out = mgr.observe([FakeFace({"1": 0.90})])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][1], "1")

    def test_elderly_extra_regression(self):
        # The exact historical failure: an extra scored 0.638 vs Harry for one
        # frame, then detection re-scored him as his real (unmapped) cluster.
        # Under the movie's thresholds (enter 0.7) he never qualifies at all.
        mgr = manager()
        out = mgr.observe([FakeFace({"1": 0.638, "13": 0.2})])
        self.assertEqual(out, [])
        out = mgr.observe([FakeFace({"1": 0.30, "13": 0.80})])
        self.assertEqual(out, [])

    def test_borderline_then_reclassified_never_confirms(self):
        # Qualified-but-not-strong needs CONFIRM_FRAMES consecutive hits; a
        # reclassification in between resets the pending count.
        mgr = manager()  # enter 0.7, strong 0.82
        out = mgr.observe([FakeFace({"1": 0.75, "13": 0.2})])
        self.assertEqual(out, [])  # pending 1/3
        out = mgr.observe([FakeFace({"1": 0.30, "13": 0.80})])
        self.assertEqual(out, [])  # pending reset
        out = mgr.observe([FakeFace({"1": 0.75, "13": 0.2})])
        self.assertEqual(out, [])  # pending back to 1/3, still not swapped

    def test_borderline_confirms_after_n_frames(self):
        mgr = manager()  # enter 0.7, strong 0.82, CONFIRM_FRAMES default 3
        f = FakeFace({"1": 0.75})
        self.assertEqual(mgr.observe([f]), [])
        self.assertEqual(mgr.observe([FakeFace({"1": 0.75})]), [])
        out = mgr.observe([FakeFace({"1": 0.75})])
        self.assertEqual(len(out), 1)

    def test_keep_hysteresis_rides_out_dips(self):
        mgr = manager()
        mgr.observe([FakeFace({"1": 0.90})])           # instant accept
        out = mgr.observe([FakeFace({"1": 0.62})])     # below enter, above keep
        self.assertEqual(len(out), 1)
        out = mgr.observe([FakeFace({"1": 0.50})])     # below keep: ride out
        self.assertEqual(len(out), 1)
        out = mgr.observe([FakeFace({"1": 0.90})])     # recovers
        self.assertEqual(len(out), 1)

    def test_drops_after_reject_streak(self):
        mgr = manager()
        mgr.observe([FakeFace({"1": 0.90})])
        results = [mgr.observe([FakeFace({"1": 0.20})]) for _ in range(4)]
        self.assertEqual(len(results[0]), 1)  # riding out
        self.assertEqual(results[3], [])      # dropped on 4th consecutive fail

    def test_immediate_drop_on_clear_reclassification(self):
        mgr = manager()
        mgr.observe([FakeFace({"1": 0.90})])
        out = mgr.observe([FakeFace({"1": 0.30, "13": 0.80})])
        self.assertEqual(out, [])  # no riding out when it's clearly someone else

    def test_margin_blocks_ambiguous_acquisition(self):
        mgr = manager(sources=("1",), second_group=("2",))
        # Above enter for both groups, margin 0.05 < MIN_MARGIN 0.08.
        out = mgr.observe([FakeFace({"1": 0.75, "2": 0.70})])
        self.assertEqual(out, [])

    def test_duplicate_clusters_are_not_rivals(self):
        # Orthonormal fake centroids cap how high two sims can be at once
        # (sum of squares <= 1), so this uses moderate scores with matching
        # thresholds. "1" and "24" share a group: 0.6 vs "24" must not count
        # as a rival margin against "1".
        mgr = manager(enter=0.55, keep=0.45, strong=0.65)
        out = mgr.observe([FakeFace({"1": 0.70, "24": 0.60})])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][1], "1")  # best member number within the group

    def test_scene_cut_resets_tracks(self):
        mgr = manager()
        mgr.observe([FakeFace({"1": 0.90})])
        # After a cut, a borderline score must not inherit the old identity.
        out = mgr.observe([FakeFace({"1": 0.65})], scene_cut=True)
        self.assertEqual(out, [])

    def test_separate_locations_are_separate_tracks(self):
        mgr = manager()
        mgr.observe([FakeFace({"1": 0.90})])
        # A face far away must not inherit the accepted identity's keep bar.
        out = mgr.observe([
            FakeFace({"1": 0.90}),
            FakeFace({"1": 0.65}, kps=KPS_FAR),
        ])
        self.assertEqual(len(out), 1)

    def test_unknown_face_left_alone(self):
        mgr = manager()
        out = mgr.observe([FakeFace({"13": 0.85})])  # unmapped cluster only
        self.assertEqual(out, [])


class ThresholdResolutionTests(unittest.TestCase):
    def test_env_fallback_without_calibration(self):
        groups = {"1": "G", "24": "G"}
        thresholds = identity.resolve_thresholds(groups, {"1": {}, "24": {}})
        self.assertEqual(thresholds["G"].source, "env")

    def test_calibrated_enter_sits_above_impostor_tail(self):
        groups = {"1": "G", "24": "G"}
        manifest = {
            "1": {"calibration": {
                "genuine_p10": 0.72, "genuine_p50": 0.82,
                "impostor_p99_by_cluster": {"24": 0.60, "13": 0.48, "noise": 0.55},
            }},
            "24": {"calibration": {
                "genuine_p10": 0.70, "genuine_p50": 0.80,
                "impostor_p99_by_cluster": {"1": 0.62, "13": 0.44, "noise": 0.52},
            }},
        }
        thresholds = identity.resolve_thresholds(groups, manifest)
        t = thresholds["G"]
        self.assertEqual(t.source, "calibrated")
        # In-group duplicates ("1" vs "24" at 0.60/0.62) must be excluded;
        # the real impostor tail is noise at 0.55 -> enter = 0.58.
        self.assertAlmostEqual(t.enter, 0.58, places=6)
        self.assertGreater(t.strong, t.enter)
        self.assertLess(t.keep, t.enter)


if __name__ == "__main__":
    unittest.main()
