"""Unit tests for the tracked-frame safety gates (external review round 3):
forward/backward flow validation, unconditional scene-cut clearing, and
track_id-keyed carry-forward. Run:

    python -m unittest tests.test_flow_quality
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tracking
from tracking import FaceTracker

KPS_A = np.array([[45, 45], [55, 45], [50, 50], [45, 55], [55, 55]], dtype=np.float32)
KPS_B = KPS_A + 300.0  # a second face far away


def textured(seed=0, size=400):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (size, size), dtype=np.uint8)


class FlowQualityTests(unittest.TestCase):
    def test_static_texture_tracks_cleanly(self):
        frame = textured()
        t = FaceTracker(detect_every_n_frames=5)
        t.start_from_detection(frame, [(KPS_A, "1", 0, None)], [KPS_A])
        out = t.track(frame.copy())  # identical frame: fb error ~0
        self.assertEqual(len(out), 1)
        self.assertEqual(t.stats["flow_rejected_fb"], 0)
        self.assertEqual(t.stats["flow_rejected_geometry"], 0)

    def test_decorrelated_frame_is_rejected_not_scrambled(self):
        # LK against unrelated noise can report "success" with garbage
        # points; the quality gate must withhold the face instead.
        t = FaceTracker(detect_every_n_frames=5)
        t.start_from_detection(textured(0), [(KPS_A, "1", 0, None)], [KPS_A])
        out = t.track(textured(99))  # completely different content
        self.assertEqual(out, [])
        rejected = (t.stats["flow_rejected_status"] + t.stats["flow_rejected_fb"]
                    + t.stats["flow_rejected_geometry"])
        self.assertGreaterEqual(rejected, 1)


class SceneCutTests(unittest.TestCase):
    def test_cut_clears_tracks_unconditionally(self):
        frame = textured()
        t = FaceTracker(detect_every_n_frames=5)
        t.start_from_detection(frame, [(KPS_A, "1", 0, None)], [KPS_A])
        # New shot: detection found nothing near the old position -- without
        # the cut flag the old face would be LK-carried into the new shot.
        out = t.start_from_detection(frame.copy(), [], [], scene_cut=True)
        self.assertEqual(out, [])
        self.assertEqual(t.stats["scene_cut_tracks_cleared"], 1)

    def test_no_cut_still_carries_genuine_misses(self):
        frame = textured()
        t = FaceTracker(detect_every_n_frames=5)
        t.start_from_detection(frame, [(KPS_A, "1", 0, None)], [KPS_A])
        out = t.start_from_detection(frame.copy(), [], [], scene_cut=False)
        self.assertEqual(len(out), 1)  # carry-forward behavior unchanged


class TrackIdKeyingTests(unittest.TestCase):
    def test_same_character_tracks_do_not_suppress_each_other(self):
        # Two physical faces mapped to the SAME character (duplicate source
        # photo design). Re-detecting one of them must not suppress the
        # carry-forward of the other -- the old code keyed suppression by
        # character_number and dropped it.
        frame = textured()
        t = FaceTracker(detect_every_n_frames=5)
        t.start_from_detection(frame, [(KPS_A, "1", 10, None), (KPS_B, "1", 11, None)],
                               [KPS_A, KPS_B])
        # Next detection: only face A re-detected; B's region unexamined.
        out = t.start_from_detection(frame.copy(), [(KPS_A, "1", 10, None)], [KPS_A])
        track_ids = sorted(f.track_id for f in out)
        self.assertEqual(track_ids, [10, 11])


if __name__ == "__main__":
    unittest.main()
