"""Unit tests for the swap backend interface (src/swap_backend.py). Run:

    python -m unittest tests.test_backend
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swap_backend import InswapperBackend


class FakeSwapper:
    def get(self, frame, target_face, source, paste_back=True):
        assert paste_back is False
        return np.zeros((128, 128, 3), dtype=np.uint8), np.eye(2, 3)


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
