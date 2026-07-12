"""Unit tests for the analysis artifact (src/analysis_store.py). Run:

    python -m unittest tests.test_analysis_store
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import analysis_store


class AnalysisStoreTests(unittest.TestCase):
    def test_round_trip(self):
        kps_a = np.arange(10, dtype=np.float32).reshape(5, 2)
        kps_b = kps_a + 100
        meta_a = analysis_store.make_meta(yaw=71.5, pitch=-3.0, roll=1.0,
                                          det_score=0.82, identity_score=0.74,
                                          margin=0.31, provenance="detector",
                                          confidence=1.0)
        rows = [(0, 7, "1", kps_a, meta_a),
                (0, 8, "24", kps_b, analysis_store.make_meta(provenance="bridge",
                                                             confidence=0.5)),
                (5, 7, "1", kps_a + 1, analysis_store.make_meta())]
        header = {"movie_path": "x.mp4", "frame_count": 6}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.npz"
            analysis_store.save_plan(path, header, rows)
            loaded_header, plan = analysis_store.load_plan(path)

        self.assertEqual(loaded_header["movie_path"], "x.mp4")
        self.assertEqual(loaded_header["row_count"], 3)
        self.assertEqual(sorted(plan.keys()), [0, 5])
        self.assertEqual(len(plan[0]), 2)
        track_id, number, kps, meta = plan[0][0]
        self.assertEqual((track_id, number), (7, "1"))
        np.testing.assert_array_equal(kps, kps_a)
        self.assertAlmostEqual(meta["yaw"], 71.5, places=4)
        self.assertAlmostEqual(meta["identity_score"], 0.74, places=4)
        self.assertAlmostEqual(meta["margin"], 0.31, places=4)
        self.assertEqual(meta["provenance"], "detector")
        # bridge row keeps its provenance/confidence; unknown floats are NaN
        _, _, _, meta_b = plan[0][1]
        self.assertEqual(meta_b["provenance"], "bridge")
        self.assertAlmostEqual(meta_b["confidence"], 0.5, places=4)
        self.assertTrue(np.isnan(meta_b["yaw"]))
        self.assertEqual(plan[5][0][1], "1")

    def test_empty_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.npz"
            analysis_store.save_plan(path, {"frame_count": 0}, [])
            header, plan = analysis_store.load_plan(path)
        self.assertEqual(header["row_count"], 0)
        self.assertEqual(plan, {})

    def test_version_mismatch_rejected(self):
        rows = [(0, 1, "1", np.zeros((5, 2), dtype=np.float32),
                 analysis_store.make_meta())]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.npz"
            original = analysis_store.FORMAT_VERSION
            try:
                analysis_store.FORMAT_VERSION = 999
                analysis_store.save_plan(path, {}, rows)
            finally:
                analysis_store.FORMAT_VERSION = original
            with self.assertRaises(ValueError):
                analysis_store.load_plan(path)


if __name__ == "__main__":
    unittest.main()
