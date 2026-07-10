"""Unit tests for sharpness matching (src/plate_matching.py). Synthetic
images only -- no models or GPU. Run:

    python -m unittest tests.test_plate_matching
"""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plate_matching import SharpnessMatcher, _interior_mask, sharpness_metrics


def textured_crop(seed=7, size=128):
    """A face-crop-like image with real high-frequency content."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (size // 4, size // 4, 3), dtype=np.uint8)
    img = cv2.resize(base, (size, size), interpolation=cv2.INTER_CUBIC)
    noise = rng.integers(0, 40, (size, size, 3), dtype=np.uint8)
    return cv2.add(img, noise)


CENTER = np.array([64.0, 64.0])


class SharpnessTests(unittest.TestCase):
    def test_recovers_known_blur(self):
        sharp = textured_crop()
        plate = cv2.GaussianBlur(sharp, (0, 0), 1.5)  # plate blurred by known sigma
        matcher = SharpnessMatcher(smoothing=0.0)
        sigma = matcher.choose_sigma(plate, sharp, "1", CENTER)
        self.assertGreaterEqual(sigma, 1.0)
        self.assertLessEqual(sigma, 2.0)

    def test_never_sharpens(self):
        sharp = textured_crop()
        soft_fake = cv2.GaussianBlur(sharp, (0, 0), 2.0)
        matcher = SharpnessMatcher(smoothing=0.0)
        # Plate sharper than the generated face -> no blur, never "unsharpen".
        sigma = matcher.choose_sigma(sharp, soft_fake, "1", CENTER)
        self.assertEqual(sigma, 0.0)

    def test_tolerance_leaves_close_matches_alone(self):
        crop = textured_crop()
        near_twin = cv2.GaussianBlur(crop, (0, 0), 0.3)
        matcher = SharpnessMatcher(smoothing=0.0)
        sigma = matcher.choose_sigma(near_twin, crop, "1", CENTER)
        self.assertLessEqual(sigma, 0.5)

    def test_temporal_smoothing_converges_and_resets(self):
        sharp = textured_crop()
        blurred_plate = cv2.GaussianBlur(sharp, (0, 0), 1.5)
        matcher = SharpnessMatcher(smoothing=0.7)
        # Seed the track in a sharp-plate state (raw sigma 0)...
        seeded = matcher.choose_sigma(sharp, sharp, "1", CENTER)
        self.assertEqual(seeded, 0.0)
        # ...then a sudden plate change must ramp gradually, not jump.
        raw = SharpnessMatcher(smoothing=0.0).choose_sigma(blurred_plate, sharp, "1", CENTER)
        first = matcher.choose_sigma(blurred_plate, sharp, "1", CENTER)
        self.assertLess(first, raw * 0.5)
        values = [matcher.choose_sigma(blurred_plate, sharp, "1", CENTER) for _ in range(12)]
        self.assertGreater(values[-1], values[0])       # rising toward raw
        self.assertAlmostEqual(values[-1], raw, delta=raw * 0.1)
        # A large position jump resets the EMA to the raw value immediately.
        far = CENTER + 500.0
        matcher2 = SharpnessMatcher(smoothing=0.7)
        matcher2.choose_sigma(sharp, sharp, "1", CENTER)
        jumped = matcher2.choose_sigma(blurred_plate, sharp, "1", far)
        self.assertAlmostEqual(jumped, raw, places=6)

    def test_metrics_use_interior_only(self):
        crop = textured_crop()
        interior = _interior_mask(crop.shape[0])
        # Corrupt only the border; interior metrics must not move.
        corrupted = crop.copy()
        corrupted[:8, :] = 255
        corrupted[-8:, :] = 0
        a = sharpness_metrics(crop, interior)
        b = sharpness_metrics(corrupted, interior)
        self.assertAlmostEqual(a[0], b[0], delta=a[0] * 0.02)


if __name__ == "__main__":
    unittest.main()
