"""Unit tests for anchor-bridged gap interpolation (identity.bridge_swap_rows).
Run:

    python -m unittest tests.test_bridge
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from identity import Observation, bridge_swap_rows

KPS = np.array([[45, 45], [55, 45], [50, 50], [45, 55], [55, 55]], dtype=np.float32)


def obs(frame, swapped, kps_shift=0.0, number="1", renderable=True, track_id=5):
    return Observation(
        frame_index=frame, track_id=track_id, kps=KPS + kps_shift,
        group_scores={"G1": 0.8},
        swapped_number=number if swapped else None,
        accepted_group="G1" if swapped else None,
        renderable=renderable,
    )


class BridgeTests(unittest.TestCase):
    def test_interior_gap_filled_with_interpolation(self):
        log = [obs(0, True), obs(10, True, kps_shift=10.0)]
        existing = {(0, 5), (1, 5), (10, 5)}  # propagation died after frame 1
        rows = bridge_swap_rows(log, existing, max_gap_frames=15)
        frames = sorted(r[0] for r in rows)
        self.assertEqual(frames, list(range(2, 10)))
        kps5 = next(r[3] for r in rows if r[0] == 5)
        np.testing.assert_allclose(kps5, KPS + 5.0, atol=1e-5)  # halfway
        for _, track_id, number, _ in rows:
            self.assertEqual((track_id, number), (5, "1"))

    def test_intermediate_observation_splits_the_bridge(self):
        # A pose-blocked (unswapped) observation mid-gap is contradicting
        # evidence: the pairs around it are (0,5) and (5,10), and neither
        # bridges because the intermediate frame wasn't swapped.
        log = [obs(0, True), obs(5, False, renderable=False), obs(10, True)]
        rows = bridge_swap_rows(log, {(0, 5), (10, 5)}, max_gap_frames=15)
        self.assertEqual(rows, [])

    def test_gap_over_limit_not_bridged(self):
        log = [obs(0, True), obs(40, True)]
        self.assertEqual(bridge_swap_rows(log, {(0, 5), (40, 5)}, 15), [])

    def test_covered_frames_not_duplicated(self):
        log = [obs(0, True), obs(6, True)]
        existing = {(0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5)}
        self.assertEqual(bridge_swap_rows(log, existing, 15), [])

    def test_identity_change_not_bridged(self):
        log = [obs(0, True, number="1"), obs(8, True, number="2")]
        self.assertEqual(bridge_swap_rows(log, {(0, 5), (8, 5)}, 15), [])

    def test_separate_tracks_not_cross_bridged(self):
        log = [obs(0, True, track_id=5), obs(8, True, track_id=6)]
        self.assertEqual(bridge_swap_rows(log, {(0, 5), (8, 6)}, 15), [])


if __name__ == "__main__":
    unittest.main()
