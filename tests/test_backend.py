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
    def __init__(self, kps):
        self.kps = np.asarray(kps, dtype=np.float32)


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
        crop, M = warp_by_template(frame, kps, ARCFACE_112_V1, 512)
        self.assertEqual(crop.shape, (512, 512, 3))
        mapped = kps @ M[:, :2].T + M[:, 2]
        np.testing.assert_allclose(mapped, ARCFACE_112_V1 * 512.0, atol=0.5)

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

    def test_hybrid_routes_by_pose(self):
        class Recorder:
            def __init__(self, tag):
                self.tag = tag
                self.calls = []

            def prepare_source(self, source_face):
                return f"{self.tag}-prep"

            def swap(self, frame, target_face, prepared_source):
                self.calls.append(prepared_source)
                return np.zeros((8, 8, 3), dtype=np.uint8), np.eye(2, 3)

            def capabilities(self):
                return {"name": self.tag, "reliable_abs_yaw": 80}

        primary, extreme = Recorder("p"), Recorder("e")
        backend = HybridBackend(primary, extreme, threshold=0.85)
        prepared = backend.prepare_source("src")
        self.assertEqual(prepared, {"primary": "p-prep", "extreme": "e-prep"})

        backend.swap(None, FakeTarget(FRONTAL_KPS), prepared)
        backend.swap(None, FakeTarget(PROFILE_KPS), prepared)
        self.assertEqual(primary.calls, ["p-prep"])   # frontal -> primary
        self.assertEqual(extreme.calls, ["e-prep"])   # profile -> SimSwap
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
