"""Unit tests for the swap backend interface (src/swap_backend.py). Run:

    python -m unittest tests.test_backend
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swap_backend import (
    ARCFACE_112_V1, HybridBackend, InswapperBackend, SimswapBackend,
    warp_by_template, yaw_proxy,
)


class FakeSwapper:
    def get(self, frame, target_face, source, paste_back=True):
        assert paste_back is False
        return np.zeros((128, 128, 3), dtype=np.uint8), np.eye(2, 3)


# Frontal 5-point landmarks (the alignment template itself, at 100px scale).
FRONTAL_KPS = ARCFACE_112_V1 * 100.0
# Near-profile: nose displaced far past the (compressed) eye span.
PROFILE_KPS = np.array(
    [[40, 46], [52, 46], [62, 61], [42, 78], [52, 78]], dtype=np.float32
)


class FakeTarget:
    def __init__(self, kps, meta=None, track_id=None):
        self.kps = np.asarray(kps, dtype=np.float32)
        self.meta = meta
        self.track_id = track_id


class FakeSimswapSession:
    """Echoes the target blob back, recording the feed names/shapes."""

    def __init__(self):
        self.last_feed = None

    def run(self, _, feed):
        self.last_feed = feed
        return [feed["target"]]


class FakeConverterSession:
    def run(self, _, feed):
        assert feed["input"].shape == (1, 512)
        return [feed["input"] * 4.0]  # non-normalized on purpose


class FakeArm:
    """Recording stand-in for a hybrid arm."""

    def __init__(self, tag, reliable=80):
        self.tag = tag
        self.reliable = reliable
        self.calls = []

    def prepare_source(self, source_face):
        return f"{self.tag}-prep"

    def swap(self, frame, target_face, prepared_source):
        self.calls.append(prepared_source)
        return np.zeros((8, 8, 3), dtype=np.uint8), np.eye(2, 3)

    def capabilities(self):
        return {"name": self.tag, "reliable_abs_yaw": self.reliable}


class BackendTests(unittest.TestCase):
    def test_inswapper_backend_contract(self):
        backend = InswapperBackend(FakeSwapper())
        prepared = backend.prepare_source("source-face")
        self.assertEqual(prepared, "source-face")  # passthrough for inswapper
        crop, M = backend.swap("frame", "target", prepared)
        self.assertEqual(crop.shape, (128, 128, 3))
        self.assertEqual(M.shape, (2, 3))
        caps = backend.capabilities()
        self.assertEqual(caps["crop_size"], 128)
        self.assertIn("reliable_abs_yaw", caps)

    def test_plate_matcher_wraps_raw_swapper(self):
        import plate_matching
        matcher = plate_matching.PlateMatcher(FakeSwapper())
        self.assertEqual(matcher.backend.name, "inswapper_128")

    def test_warp_by_template_maps_kps_to_template_positions(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        kps = ARCFACE_112_V1 * 512.0  # already in perfect template position
        crop, M, inliers = warp_by_template(frame, kps, ARCFACE_112_V1, 512)
        self.assertEqual(crop.shape, (512, 512, 3))
        self.assertEqual(int(inliers.sum()), 5)
        mapped = kps @ M[:, :2].T + M[:, 2]
        np.testing.assert_allclose(mapped, ARCFACE_112_V1 * 512.0, atol=0.5)

    def test_alignment_validation_accepts_genuine_and_rejects_scramble(self):
        from swap_backend import validate_alignment
        frame_shape = (300, 300, 3)
        good = ARCFACE_112_V1 * 200.0 + 40.0
        _, M, inl = warp_by_template(np.zeros(frame_shape, np.uint8), good,
                                     ARCFACE_112_V1, 512)
        report = validate_alignment(M, inl, good, ARCFACE_112_V1, 512, frame_shape)
        self.assertTrue(report["valid"])
        self.assertLess(report["residual_fraction"], 0.01)
        self.assertFalse(report["reflection"])

        # Scrambled landmarks (the LK-corruption shape): eyes collapsed onto
        # the mouth line -- no similarity to the template explains them.
        scrambled = np.array([[100, 200], [102, 200], [150, 90],
                              [100, 202], [104, 201]], dtype=np.float32)
        _, M2, inl2 = warp_by_template(np.zeros(frame_shape, np.uint8), scrambled,
                                       ARCFACE_112_V1, 512)
        report2 = validate_alignment(M2, inl2, scrambled, ARCFACE_112_V1, 512,
                                     frame_shape)
        self.assertFalse(report2["valid"])

    def test_alignment_validation_rejects_mostly_offframe_face(self):
        from swap_backend import validate_alignment
        frame_shape = (300, 300, 3)
        # Genuine geometry, but positioned so most of the crop samples
        # outside the frame (BORDER_REPLICATE smear territory).
        kps = ARCFACE_112_V1 * 200.0 + np.array([260.0, 40.0])
        _, M, inl = warp_by_template(np.zeros(frame_shape, np.uint8), kps,
                                     ARCFACE_112_V1, 512)
        report = validate_alignment(M, inl, kps, ARCFACE_112_V1, 512, frame_shape)
        self.assertFalse(report["valid"])
        self.assertEqual(report["reason"], "coverage")

    def test_simswap_withholds_invalid_alignment_and_matcher_passes_through(self):
        import plate_matching
        backend = SimswapBackend(FakeSimswapSession(), FakeConverterSession())
        scrambled = np.array([[100, 200], [102, 200], [150, 90],
                              [100, 202], [104, 201]], dtype=np.float32)
        frame = np.full((300, 300, 3), 90, dtype=np.uint8)
        crop, M = backend.swap(frame, FakeTarget(scrambled), np.zeros((1, 512), np.float32))
        self.assertIsNone(crop)
        self.assertEqual(backend.alignment_stats["withheld"], 1)

        matcher = plate_matching.PlateMatcher(backend)
        class FakeSource:
            embedding = np.ones(512, dtype=np.float32)
        out = matcher.swap(frame, FakeTarget(scrambled, track_id=1), FakeSource())
        np.testing.assert_array_equal(out, frame)  # plate untouched
        self.assertEqual(matcher.backend_withheld, 1)

    def test_simswap_backend_contract(self):
        session = FakeSimswapSession()
        backend = SimswapBackend(session, FakeConverterSession())

        class FakeSource:
            embedding = np.ones(512, dtype=np.float32)

        prepared = backend.prepare_source(FakeSource())
        self.assertEqual(prepared.shape, (1, 512))
        # converter output must be L2-normalized regardless of its scale
        self.assertAlmostEqual(float(np.linalg.norm(prepared)), 1.0, places=5)

        frame = np.full((300, 300, 3), 128, dtype=np.uint8)
        crop, M = backend.swap(frame, FakeTarget(FRONTAL_KPS), prepared)
        self.assertEqual(crop.shape, (512, 512, 3))
        self.assertEqual(M.shape, (2, 3))
        # echo session: BGR->RGB /255 in, *255 RGB->BGR out == round trip
        self.assertTrue(np.all(np.abs(crop.astype(int) - 128) <= 1))
        self.assertEqual(session.last_feed["target"].shape, (1, 3, 512, 512))
        self.assertEqual(backend.capabilities()["reliable_abs_yaw"], 80)

    def test_yaw_proxy_separates_frontal_from_profile(self):
        self.assertLess(yaw_proxy(FRONTAL_KPS), 0.2)
        self.assertGreater(yaw_proxy(PROFILE_KPS), 0.85)
        degenerate = np.zeros((5, 2), dtype=np.float32)
        self.assertEqual(yaw_proxy(degenerate), 10.0)

    def test_yaw_proxy_is_roll_invariant(self):
        theta = np.deg2rad(30)
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        self.assertAlmostEqual(
            yaw_proxy(FRONTAL_KPS @ rot.T), yaw_proxy(FRONTAL_KPS), places=6
        )
        self.assertAlmostEqual(
            yaw_proxy(PROFILE_KPS @ rot.T), yaw_proxy(PROFILE_KPS), places=6
        )

    def test_hybrid_falls_back_to_proxy_without_pose_evidence(self):
        primary, extreme = FakeArm("p"), FakeArm("e")
        backend = HybridBackend(primary, extreme, threshold=0.85)
        prepared = backend.prepare_source("src")
        self.assertEqual(prepared, {"primary": "p-prep", "extreme": "e-prep"})

        backend.swap(None, FakeTarget(FRONTAL_KPS), prepared)
        backend.swap(None, FakeTarget(PROFILE_KPS), prepared)
        self.assertEqual(primary.calls, ["p-prep"])   # frontal -> primary
        self.assertEqual(extreme.calls, ["e-prep"])   # profile -> SimSwap
        self.assertEqual(backend.routed, {"primary": 1, "extreme": 1})
        self.assertEqual(backend.proxy_fallbacks, 2)

    def test_hybrid_routes_on_stored_pose_over_proxy(self):
        primary, extreme = FakeArm("p", reliable=65), FakeArm("e", reliable=80)
        backend = HybridBackend(primary, extreme, threshold=0.85)
        prepared = backend.prepare_source("src")
        # FRONTAL landmarks (proxy would say primary) but stored yaw says 80:
        # the actual Buffalo pose must win.
        backend.swap(None, FakeTarget(FRONTAL_KPS, meta={"yaw": 80.0}, track_id=1),
                     prepared)
        self.assertEqual(extreme.calls, ["e-prep"])
        self.assertEqual(backend.proxy_fallbacks, 0)

    def test_hybrid_route_hysteresis_and_transition_log(self):
        primary, extreme = FakeArm("p", reliable=65), FakeArm("e", reliable=80)
        backend = HybridBackend(primary, extreme, threshold=0.85)
        prepared = backend.prepare_source("src")
        # Min-hold: entering extreme is immediate; returning to primary
        # needs the extreme stint to have lasted POSE_MIN_HOLD (3) swaps.
        for yaw in [80, 60, 50, 60]:
            backend.swap(None, FakeTarget(FRONTAL_KPS, meta={"yaw": float(yaw)},
                                          track_id=7), prepared)
        self.assertEqual(backend.routed, {"primary": 1, "extreme": 3})
        self.assertEqual(backend.transition_count, 1)
        self.assertEqual(backend.transitions[0]["from"], "extreme")
        self.assertEqual(backend.transitions[0]["to"], "primary")
        self.assertEqual(backend.transitions[0]["track_id"], 7)

    def test_hybrid_extreme_entry_is_immediate(self):
        # The safety direction must never wait out a hold: a fresh primary
        # track that turns past the limit routes extreme on that same swap.
        primary, extreme = FakeArm("p", reliable=65), FakeArm("e", reliable=80)
        backend = HybridBackend(primary, extreme, threshold=0.85)
        prepared = backend.prepare_source("src")
        for yaw in [30, 70]:
            backend.swap(None, FakeTarget(FRONTAL_KPS, meta={"yaw": float(yaw)},
                                          track_id=9), prepared)
        self.assertEqual(backend.routed, {"primary": 1, "extreme": 1})

    def test_unknown_backend_rejected(self):
        import os
        from swap_backend import build_backend
        os.environ["SWAP_BACKEND"] = "nonexistent"
        try:
            with self.assertRaises(SystemExit):
                build_backend(FakeSwapper())
        finally:
            del os.environ["SWAP_BACKEND"]


if __name__ == "__main__":
    unittest.main()
