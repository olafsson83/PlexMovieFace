"""Unit tests for colour/lighting matching (src/plate_matching.py).
Synthetic images only -- no models or GPU. Run:

    python -m unittest tests.test_color
"""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plate_matching import ColorMatcher, _interior_mask

CENTER = np.array([64.0, 64.0])


def textured(seed=7, size=128, tint=(0, 0, 0), bias=110):
    rng = np.random.default_rng(seed)
    img = rng.normal(bias, 18, (size, size, 3)).astype(np.float32)
    img += np.array(tint, dtype=np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def interior_mean_lab(bgr):
    interior = _interior_mask(bgr.shape[0])
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    return lab[interior].mean(axis=0)


class ColorMatchTests(unittest.TestCase):
    def test_pulls_toward_plate_tone(self):
        # Fake is slightly blue/dark, plate slightly warm/bright (a realistic
        # backend tone gap, within the shift cap): the corrected fake's
        # interior Lab mean must move ~halfway toward the plate's.
        fake = textured(seed=1, tint=(24, 0, 0), bias=112)    # mildly blue/dark
        plate = textured(seed=2, tint=(0, 6, 22), bias=138)   # mildly warm/bright
        matcher = ColorMatcher(strength=0.5, smoothing=0.0)
        out = matcher.match(plate, fake, "1", CENTER)

        f, p, o = (interior_mean_lab(x) for x in (fake, plate, out))
        for ch in range(3):
            lo, hi = sorted((f[ch], p[ch]))
            self.assertGreater(o[ch], lo - 1)   # moved off the fake extreme
            self.assertLess(o[ch], hi + 1)      # toward, not past, the plate
        # Overall the corrected fake sits closer to the plate's Lab mean than
        # the raw fake did (per-channel can tie where they already matched).
        self.assertLess(float(np.linalg.norm(o - p)), float(np.linalg.norm(f - p)))
        # With strength 0.5 it should close roughly half the gap.
        self.assertLess(float(np.linalg.norm(o - p)), 0.7 * float(np.linalg.norm(f - p)))

    def test_already_matched_face_is_left_alone(self):
        # An already-matched face (tiny tone gap, within the dead-zone) must
        # get no correction -- otherwise the matcher tracks plate colour noise
        # and adds temporal jitter (the two-hander instability regression).
        fake = textured(seed=1, bias=120)
        plate = textured(seed=2, bias=122)  # ~2 L gap, under COLOR_MIN_SHIFT
        matcher = ColorMatcher(strength=0.5, smoothing=0.0,
                               min_shift=4.0, min_gain_delta=0.05)
        out = matcher.match(plate, fake, "1", CENTER)
        self.assertLess(float(np.abs(out.astype(int) - fake.astype(int)).mean()), 1.5)

    def test_strength_zero_is_identity(self):
        fake = textured(seed=1, tint=(60, 0, 0))
        plate = textured(seed=2, tint=(0, 0, 60), bias=150)
        out = ColorMatcher(strength=0.0, smoothing=0.0).match(plate, fake, "1", CENTER)
        # No correction: output within rounding of the input.
        self.assertLess(float(np.abs(out.astype(int) - fake.astype(int)).mean()), 1.5)

    def test_shift_is_bounded(self):
        # A huge tone gap must be capped by max_shift, not applied wholesale.
        fake = textured(seed=1, bias=40)
        plate = textured(seed=2, bias=220)
        matcher = ColorMatcher(strength=1.0, max_shift=10.0, smoothing=0.0)
        out = matcher.match(plate, fake, "1", CENTER)
        f, o = interior_mean_lab(fake), interior_mean_lab(out)
        # L-channel move cannot exceed the cap (plus gain interaction slack).
        self.assertLessEqual(abs(o[0] - f[0]), 10.0 + 6.0)

    def test_temporal_smoothing_and_reset(self):
        fake = textured(seed=1, tint=(60, 0, 0), bias=90)
        plate = textured(seed=2, tint=(0, 0, 60), bias=150)
        raw = ColorMatcher(strength=0.5, smoothing=0.0).match(plate, fake, "1", CENTER)
        m = ColorMatcher(strength=0.5, smoothing=0.7)
        m.match(fake, fake, "1", CENTER)               # seed: no correction
        first = m.match(plate, fake, "1", CENTER)      # must ramp, not jump
        self.assertLess(float(np.abs(first.astype(int) - fake.astype(int)).mean()),
                        float(np.abs(raw.astype(int) - fake.astype(int)).mean()))
        # A large position jump resets the EMA to the raw correction.
        m2 = ColorMatcher(strength=0.5, smoothing=0.7)
        m2.match(fake, fake, "1", CENTER)
        jumped = m2.match(plate, fake, "1", CENTER + 500.0)
        self.assertLess(float(np.abs(jumped.astype(int) - raw.astype(int)).mean()), 2.0)

    def test_only_interior_drives_the_match(self):
        # Corrupting only the border must not change the correction.
        fake = textured(seed=1, tint=(40, 0, 0))
        plate = textured(seed=2, tint=(0, 0, 40), bias=150)
        a = ColorMatcher(strength=0.5, smoothing=0.0).match(plate, fake, "1", CENTER)
        plate_b = plate.copy()
        plate_b[:8, :] = 255
        plate_b[-8:, :] = 0
        b = ColorMatcher(strength=0.5, smoothing=0.0).match(plate_b, fake, "1", CENTER)
        self.assertLess(float(np.abs(a.astype(int) - b.astype(int)).mean()), 1.5)


if __name__ == "__main__":
    unittest.main()
