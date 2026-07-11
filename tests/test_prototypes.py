"""Unit tests for v2 milestone 5: prototype banks. Run:

    python -m unittest tests.test_prototypes
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discover_characters import prototype_scores, select_prototypes
from identity import prototype_max_score


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


class PrototypeSelectionTests(unittest.TestCase):
    def test_fps_picks_diverse_members(self):
        # Two tight sub-clusters (think frontal vs profile) plus the bulk:
        # farthest-point sampling must cover both sub-clusters instead of
        # picking five near-duplicates from the bulk.
        rng = np.random.default_rng(1)
        frontal = [unit([1, 0, 0] + 0.05 * rng.standard_normal(3)) for _ in range(30)]
        profile = [unit([0.4, 0.9, 0] + 0.05 * rng.standard_normal(3)) for _ in range(6)]
        protos = select_prototypes(frontal + profile, k=3)
        self.assertEqual(len(protos), 3)
        # At least one prototype must be close to the profile sub-cluster.
        profile_dir = unit([0.4, 0.9, 0])
        self.assertGreater(max(float(p @ profile_dir) for p in protos), 0.9)

    def test_fewer_members_than_k(self):
        members = [unit([1, 0, 0]), unit([0, 1, 0])]
        protos = select_prototypes(members, k=5)
        self.assertEqual(len(protos), 3)  # centroid anchor + both members

    def test_centroid_is_always_in_the_bank(self):
        rng = np.random.default_rng(2)
        members = [unit([1, 0.2, 0] + 0.1 * rng.standard_normal(3)) for _ in range(20)]
        protos = select_prototypes(members, k=4)
        centroid = unit(np.mean(np.stack(members), axis=0))
        self.assertGreater(max(float(p @ centroid) for p in protos), 0.9999)


class PrototypeScoringTests(unittest.TestCase):
    def test_max_over_bank_beats_mean(self):
        frontal = unit([1, 0, 0])
        profile = unit([0, 1, 0])
        bank = np.stack([frontal, profile])
        face = unit([0.1, 0.99, 0])  # a profile-ish face
        bank_score = prototype_max_score(face, bank)
        mean_score = float(face @ unit(frontal + profile))
        self.assertGreater(bank_score, mean_score)

    def test_single_centroid_backward_compat(self):
        centroid = unit([1, 0, 0])
        face = unit([0.9, 0.1, 0])
        self.assertAlmostEqual(
            prototype_max_score(face, centroid), float(face @ centroid), places=6
        )

    def test_batch_scores_match_scalar(self):
        bank = np.stack([unit([1, 0, 0]), unit([0, 1, 0])])
        faces = np.stack([unit([0.9, 0.1, 0]), unit([0.1, 0.9, 0])])
        batch = prototype_scores(faces, bank)
        for i in range(2):
            self.assertAlmostEqual(
                float(batch[i]), prototype_max_score(faces[i], bank), places=6
            )


if __name__ == "__main__":
    unittest.main()
