"""Unit tests for motion blur matching (src/plate_matching.py
MotionBlurMatcher). Synthetic only. Run:

    python -m unittest tests.test_motion
"""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plate_matching import MotionBlurMatcher

KPS = np.array([[45, 45], [55, 45], [50, 50], [45, 55], [55, 55]], dtype=np.float32)
# Identity-ish alignment transform (frame == crop scale) for simplicity.
M_IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def crop_with_vertical_edge(size=128):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, size // 2:] = 200
    return img


def matcher(**kw):
    defaults = dict(smoothing=0.0, shutter_fraction=0.5)
    defaults.update(kw)
    return MotionBlurMatcher(**defaults)


class MotionBlurTests(unittest.TestCase):
    def test_first_observation_never_blurs(self):
        m = matcher()
        crop = crop_with_vertical_edge()
        out, length = m.blur_crop(crop, KPS, M_IDENTITY, "1", KPS.mean(axis=0))
        self.assertEqual(length, 0.0)
        np.testing.assert_array_equal(out, crop)

    def test_translation_produces_directional_blur(self):
        m = matcher()
        crop = crop_with_vertical_edge()
        m.blur_crop(crop, KPS, M_IDENTITY, "1", KPS.mean(axis=0))
        moved = KPS + np.array([8.0, 0.0])  # 8px right -> 4px blur at 180deg shutter
        out, length = m.blur_crop(crop, moved, M_IDENTITY, "1", moved.mean(axis=0))
        self.assertGreater(length, 2.0)
        self.assertLess(length, 6.0)
        # Horizontal motion must soften the vertical edge.
        col = crop.shape[1] // 2
        edge_before = np.abs(np.diff(crop[64, col - 3:col + 3, 0].astype(int))).max()
        edge_after = np.abs(np.diff(out[64, col - 3:col + 3, 0].astype(int))).max()
        self.assertLess(edge_after, edge_before * 0.8)

    def test_static_face_untouched(self):
        m = matcher()
        crop = crop_with_vertical_edge()
        m.blur_crop(crop, KPS, M_IDENTITY, "1", KPS.mean(axis=0))
        out, length = m.blur_crop(crop, KPS + 0.2, M_IDENTITY, "1", KPS.mean(axis=0))
        self.assertEqual(length, 0.0)
        np.testing.assert_array_equal(out, crop)

    def test_rotation_dominant_is_damped(self):
        m = matcher()
        crop = crop_with_vertical_edge()
        m.blur_crop(crop, KPS, M_IDENTITY, "1", KPS.mean(axis=0))
        # Rotate 12 degrees about the centroid plus a slight shift: the
        # linear-PSF model must be heavily damped, not fully applied.
        c = KPS.mean(axis=0)
        theta = np.radians(12)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rotated = (KPS - c) @ R.T + c + np.array([4.0, 0.0])
        out, length = m.blur_crop(crop, rotated.astype(np.float32), M_IDENTITY, "1",
                                  rotated.mean(axis=0))
        undamped = 4.0 * 0.5  # displacement x shutter fraction
        self.assertLess(length, undamped * 0.6)

    def test_glitch_capped_by_face_fraction(self):
        m = matcher(max_crop_fraction=0.08)
        crop = crop_with_vertical_edge()
        m.blur_crop(crop, KPS, M_IDENTITY, "1", KPS.mean(axis=0))
        teleport = KPS + np.array([70.0, 0.0])  # inside reset distance, huge motion
        out, length = m.blur_crop(crop, teleport, M_IDENTITY, "1", teleport.mean(axis=0))
        self.assertLessEqual(length, 128 * 0.08 + 1e-6)

    def test_position_jump_resets_state(self):
        m = matcher()
        crop = crop_with_vertical_edge()
        m.blur_crop(crop, KPS, M_IDENTITY, "1", KPS.mean(axis=0))
        far = KPS + 500.0  # beyond SMOOTHING_RESET_DISTANCE: cut/reacquire
        out, length = m.blur_crop(crop, far, M_IDENTITY, "1", far.mean(axis=0))
        self.assertEqual(length, 0.0)

    def test_ema_smooths_sudden_stop(self):
        m = matcher(smoothing=0.65)
        crop = crop_with_vertical_edge()
        kps = KPS.copy()
        m.blur_crop(crop, kps, M_IDENTITY, "1", kps.mean(axis=0))
        for _ in range(4):  # sustained motion builds the EMA
            kps = kps + np.array([8.0, 0.0])
            _, moving_len = m.blur_crop(crop, kps, M_IDENTITY, "1", kps.mean(axis=0))
        _, stopped_len = m.blur_crop(crop, kps, M_IDENTITY, "1", kps.mean(axis=0))
        self.assertGreater(stopped_len, 0.0)          # blur decays, not snaps
        self.assertLess(stopped_len, moving_len)

    def test_crop_space_scaling(self):
        # A 2x alignment scale doubles the blur length in crop space.
        m2 = matcher()
        M_2x = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        crop = crop_with_vertical_edge()
        m2.blur_crop(crop, KPS, M_2x, "1", KPS.mean(axis=0))
        moved = KPS + np.array([8.0, 0.0])
        _, length_2x = m2.blur_crop(crop, moved, M_2x, "1", moved.mean(axis=0))
        self.assertAlmostEqual(length_2x, 8.0, delta=0.5)  # 8 * 0.5 shutter * 2


if __name__ == "__main__":
    unittest.main()
