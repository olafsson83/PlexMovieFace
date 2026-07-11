"""Unit tests for grain matching (src/plate_matching.py GrainMatcher).
Synthetic images only. Run:

    python -m unittest tests.test_grain
"""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plate_matching import GrainMatcher, robust_noise_sigmas

CENTER = np.array([100.0, 100.0])


def flat_scene(size=200, value=120):
    return np.full((size, size, 3), value, dtype=np.uint8)


def add_noise(img, sigma, seed=1):
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float32) + rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def face_disk(size=200, radius=45):
    region = np.zeros((size, size), dtype=bool)
    yy, xx = np.ogrid[:size, :size]
    region[(yy - size // 2) ** 2 + (xx - size // 2) ** 2 <= radius ** 2] = True
    return region


class NoiseEstimatorTests(unittest.TestCase):
    def test_estimates_scale_with_noise(self):
        mask = np.ones((200, 200), dtype=bool)
        quiet = robust_noise_sigmas(add_noise(flat_scene(), 2.0), mask)
        loud = robust_noise_sigmas(add_noise(flat_scene(), 8.0), mask)
        self.assertIsNotNone(quiet)
        self.assertIsNotNone(loud)
        # Estimator units are attenuated but must preserve ordering/ratio.
        self.assertGreater(loud[0], quiet[0] * 2.5)

    def test_edges_do_not_inflate_estimate(self):
        clean = flat_scene()
        clean[:, 100:] = 240  # hard edge, zero noise
        mask = np.ones((200, 200), dtype=bool)
        sigmas = robust_noise_sigmas(clean, mask)
        self.assertIsNotNone(sigmas)
        self.assertLess(sigmas[0], 0.5)

    def test_too_few_pixels_returns_none(self):
        mask = np.zeros((200, 200), dtype=bool)
        mask[:10, :10] = True
        self.assertIsNone(robust_noise_sigmas(add_noise(flat_scene(), 5.0), mask))


class GrainMatcherTests(unittest.TestCase):
    def _run_match(self, plate_sigma, fake_sigma, smoothing=0.0, seed=3):
        frame = add_noise(flat_scene(), plate_sigma, seed=10)
        face = face_disk()
        fake_layer = flat_scene(value=118)
        if fake_sigma > 0:
            fake_layer = add_noise(fake_layer, fake_sigma, seed=11)
        matcher = GrainMatcher(smoothing=smoothing)
        out = matcher.match(frame, fake_layer, face, 90, "1", CENTER, seed)
        return matcher, out, fake_layer, face

    def test_grains_clean_fake_on_noisy_plate(self):
        matcher, out, fake_layer, face = self._run_match(plate_sigma=8.0, fake_sigma=0.0)
        self.assertGreaterEqual(matcher.applied_y[-1], matcher.min_sigma)
        interior = cv2.erode(face.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        before = float(out[interior].astype(np.float32).std())
        self.assertGreater(before, float(fake_layer[interior].astype(np.float32).std()))

    def test_deficit_only_no_double_graining(self):
        # Fake already as noisy as the plate -> nothing meaningful added.
        matcher, out, fake_layer, _ = self._run_match(plate_sigma=6.0, fake_sigma=6.0)
        self.assertLess(matcher.applied_y[-1], 2.0)

    def test_clean_plate_skips_grain(self):
        matcher, out, fake_layer, _ = self._run_match(plate_sigma=0.0, fake_sigma=0.0)
        self.assertLess(matcher.applied_y[-1], matcher.min_sigma)
        np.testing.assert_array_equal(out, fake_layer)

    def test_deterministic_per_seed(self):
        _, out_a, _, _ = self._run_match(8.0, 0.0, seed=42)
        _, out_b, _, _ = self._run_match(8.0, 0.0, seed=42)
        _, out_c, _, _ = self._run_match(8.0, 0.0, seed=43)
        np.testing.assert_array_equal(out_a, out_b)
        self.assertTrue((out_a != out_c).any())

    def test_synthesis_amplitude_matches_measurement_units(self):
        # Grain a clean fake against a noisy plate, then re-measure the
        # grained interior with the same estimator: it should land near the
        # plate's measured sigma (closing the estimator-units loop).
        frame = add_noise(flat_scene(), 8.0, seed=10)
        face = face_disk()
        fake_layer = flat_scene(value=118)
        matcher = GrainMatcher(smoothing=0.0)
        out = matcher.match(frame, fake_layer, face, 90, "1", CENTER, 5)
        interior = cv2.erode(face.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
        grained = robust_noise_sigmas(out, interior)
        ring_mask = face_disk(radius=75) & ~face_disk(radius=55)
        plate = robust_noise_sigmas(frame, ring_mask)
        self.assertIsNotNone(grained)
        self.assertIsNotNone(plate)
        self.assertAlmostEqual(grained[0], plate[0], delta=plate[0] * 0.3)


if __name__ == "__main__":
    unittest.main()
