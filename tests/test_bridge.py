"""Unit tests for the bidirectional-tracking gap bridge (src/bridging.py).
Run:

    python -m unittest tests.test_bridge
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bridging import MAX_INTERP_GAP, bridge_swap_rows
from identity import Observation

KPS = np.array([[45, 45], [55, 45], [50, 50], [45, 55], [55, 55]], dtype=np.float32)


def obs(frame, swapped, kps_shift=0.0, number="1", renderable=True, track_id=5):
    return Observation(
        frame_index=frame, track_id=track_id, kps=KPS + kps_shift,
        group_scores={"G1": 0.8},
        swapped_number=number if swapped else None,
        accepted_group="G1" if swapped else None,
        renderable=renderable,
    )


def moving_scene(shift_per_frame=1.0, frames=20, size=200, seed=3):
    """Synthetic clip: a textured patch translating over static noise, the
    face landmarks riding on the patch -- content LK can genuinely track."""
    rng = np.random.default_rng(seed)
    background = rng.integers(0, 60, (size, size), dtype=np.uint8)
    patch = rng.integers(80, 255, (48, 48), dtype=np.uint8)
    out = []
    for i in range(frames):
        img = background.copy()
        x = 30 + int(round(i * shift_per_frame))
        img[30:78, x:x + 48] = patch
        out.append(img)
    return out


class TrackedBridgeTests(unittest.TestCase):
    def test_long_gap_bridged_only_with_tracking_evidence(self):
        frames = moving_scene()
        kps0 = KPS.copy()          # inside the patch at frame 0
        kps8 = KPS + [8.0, 0.0]    # patch moved 8px right by frame 8
        log = [
            Observation(0, 5, kps0, {"G1": 0.8}, "1", "G1"),
            Observation(8, 5, kps8, {"G1": 0.8}, "1", "G1"),
        ]
        # Without a frame source, a long gap must NOT be interpolated.
        self.assertEqual(bridge_swap_rows(log, set(), 15, frame_source=None), [])

        stats = {}
        rows = bridge_swap_rows(log, set(), 15,
                                frame_source=lambda i: frames[i], stats=stats)
        emitted = sorted(r[0] for r in rows)
        self.assertEqual(emitted, list(range(1, 8)))
        self.assertEqual(stats["tracked_rows"], 7)
        # Landmarks follow the actual pixel motion (~1px/frame).
        kps4 = next(r[3] for r in rows if r[0] == 4)
        np.testing.assert_allclose(kps4, KPS + [4.0, 0.0], atol=1.5)
        for r in rows:
            self.assertEqual(r[4]["provenance"], "bridge")

    def test_gap_with_destroyed_content_is_not_bridged(self):
        # Mid-gap frames are unrelated noise: forward/backward trajectories
        # die at the gates or disagree -- nothing may be emitted (this is
        # exactly the interval the old linear interpolation filled blindly).
        frames = moving_scene()
        rng = np.random.default_rng(42)
        for i in range(3, 6):
            frames[i] = rng.integers(0, 255, frames[i].shape, dtype=np.uint8)
        log = [
            Observation(0, 5, KPS, {"G1": 0.8}, "1", "G1"),
            Observation(8, 5, KPS + [8.0, 0.0], {"G1": 0.8}, "1", "G1"),
        ]
        stats = {}
        rows = bridge_swap_rows(log, set(), 15,
                                frame_source=lambda i: frames[i], stats=stats)
        emitted = {r[0] for r in rows}
        self.assertFalse(emitted & {3, 4, 5})
        self.assertGreater(stats["frames_unverified"] + stats["frames_disagree"], 0)


class ShortGapInterpolationTests(unittest.TestCase):
    def test_two_frame_gap_interpolates_without_video(self):
        log = [obs(0, True), obs(3, True, kps_shift=3.0)]
        rows = bridge_swap_rows(log, {(0, 5), (3, 5)}, 15, frame_source=None)
        self.assertEqual(sorted(r[0] for r in rows), [1, 2])
        kps1 = next(r[3] for r in rows if r[0] == 1)
        np.testing.assert_allclose(kps1, KPS + 1.0, atol=1e-5)
        self.assertLessEqual(3 - 1, MAX_INTERP_GAP + 1)

    def test_covered_frames_not_duplicated(self):
        log = [obs(0, True), obs(3, True)]
        existing = {(0, 5), (1, 5), (2, 5), (3, 5)}
        self.assertEqual(bridge_swap_rows(log, existing, 15), [])


class CandidatePairSemanticsTests(unittest.TestCase):
    def test_intermediate_observation_splits_the_bridge(self):
        # An unswapped observation mid-gap is contradicting evidence: the
        # pairs around it are (0,2) and (2,4)... neither bridges because
        # the intermediate frame wasn't swapped.
        log = [obs(0, True), obs(2, False), obs(4, True)]
        self.assertEqual(bridge_swap_rows(log, {(0, 5), (4, 5)}, 15), [])

    def test_gap_over_limit_not_bridged(self):
        log = [obs(0, True), obs(40, True)]
        self.assertEqual(bridge_swap_rows(log, {(0, 5), (40, 5)}, 15), [])

    def test_identity_change_not_bridged(self):
        log = [obs(0, True, number="1"), obs(3, True, number="2")]
        self.assertEqual(bridge_swap_rows(log, {(0, 5), (3, 5)}, 15), [])


if __name__ == "__main__":
    unittest.main()
