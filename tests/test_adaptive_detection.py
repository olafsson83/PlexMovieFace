"""Unit tests for adaptive low-light detection (src/adaptive_detection.py).
Model-free: the detector wraps a fake face_app. Run:

    python -m unittest tests.test_adaptive_detection
"""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_detection import AdaptiveDetector, enhance_for_analysis, merge_detections

KPS = np.array([[45, 45], [55, 45], [50, 50], [45, 55], [55, 55]], dtype=np.float32)


class FakeFace:
    def __init__(self, bbox, kps=KPS, det_score=0.8):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.kps = kps.copy()
        self.det_score = det_score


class FakeDetModel:
    def __init__(self, retry_hits=1):
        self.retry_hits = retry_hits
        self.calls = []

    def detect(self, img, input_size=None, max_num=0, metric="default"):
        self.calls.append(input_size)
        if self.retry_hits == 0:
            return np.empty((0, 5), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)
        bboxes = np.tile(np.array([[40, 40, 60, 60, 0.7]], dtype=np.float32), (self.retry_hits, 1))
        kpss = np.tile(KPS[None], (self.retry_hits, 1, 1))
        return bboxes, kpss


class FakeFaceApp:
    def __init__(self, base_faces=(), retry_hits=1):
        self.base_faces = list(base_faces)
        self.det_model = FakeDetModel(retry_hits)
        self.models = {"detection": self.det_model}  # nothing else to run

    def get(self, frame):
        return list(self.base_faces)


class FakeTrack:
    def __init__(self, kps):
        self.kps = kps


class FakeIdentityMgr:
    def __init__(self, tracks=()):
        self._tracks = list(tracks)


def textured_frame(mean, spread=25, seed=3):
    """Flat frames are pathological for CLAHE (a constant tile equalizes to
    an extreme), so tests use noise-textured frames like real footage."""
    rng = np.random.default_rng(seed)
    img = rng.normal(mean, spread, (100, 100, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def dark_frame(value=20):
    return textured_frame(value, spread=10)


def bright_frame(value=150):
    return textured_frame(value, spread=25)


class EnhancementTests(unittest.TestCase):
    def test_dark_frame_lifted(self):
        frame = dark_frame(20)
        enhanced = enhance_for_analysis(frame)
        self.assertGreater(float(enhanced.mean()), float(frame.mean()) * 1.8)
        self.assertEqual(enhanced.shape, frame.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_bright_frame_gamma_untouched(self):
        frame = bright_frame(150)
        enhanced = enhance_for_analysis(frame)
        # Gamma clips to 1.0 above mid-gray; CLAHE may redistribute local
        # contrast but must not shift overall brightness much.
        self.assertLess(abs(float(enhanced.mean()) - float(frame.mean())), 15.0)


class MergeTests(unittest.TestCase):
    def test_overlapping_retry_dropped(self):
        base = [FakeFace([40, 40, 60, 60])]
        retry = [FakeFace([42, 42, 62, 62]), FakeFace([10, 10, 25, 25])]
        merged = merge_detections(base, retry)
        self.assertEqual(len(merged), 2)  # base + the novel one only
        self.assertIs(merged[0], base[0])


class RetryDecisionTests(unittest.TestCase):
    def test_bright_frame_never_retries(self):
        app = FakeFaceApp(base_faces=[], retry_hits=1)
        det = AdaptiveDetector(app, enabled=True)
        out = det.get(bright_frame())
        self.assertEqual(out, [])
        self.assertEqual(det.stats["retries"], 0)

    def test_dark_empty_frame_retries_and_recovers(self):
        app = FakeFaceApp(base_faces=[], retry_hits=1)
        det = AdaptiveDetector(app, enabled=True, retry_det_size=960)
        out = det.get(dark_frame())
        self.assertEqual(len(out), 1)
        self.assertEqual(det.stats["retries"], 1)
        self.assertEqual(det.stats["recovered_faces"], 1)
        self.assertIn((960, 960), app.det_model.calls)

    def test_dark_frame_with_expected_region_covered_skips_retry(self):
        base = [FakeFace([40, 40, 60, 60])]
        app = FakeFaceApp(base_faces=base, retry_hits=1)
        det = AdaptiveDetector(app, enabled=True)
        det.bind(FakeIdentityMgr([FakeTrack(KPS)]))  # track right where base face is
        out = det.get(dark_frame())
        self.assertEqual(len(out), 1)
        self.assertEqual(det.stats["retries"], 0)

    def test_dark_frame_with_missing_expected_region_retries(self):
        base = [FakeFace([40, 40, 60, 60])]
        app = FakeFaceApp(base_faces=base, retry_hits=1)
        det = AdaptiveDetector(app, enabled=True)
        far_track = FakeTrack(KPS + 500.0)  # a tracked face the base pass lost
        det.bind(FakeIdentityMgr([FakeTrack(KPS), far_track]))
        det.get(dark_frame())
        self.assertEqual(det.stats["retries"], 1)

    def test_disabled_passthrough(self):
        app = FakeFaceApp(base_faces=[], retry_hits=1)
        det = AdaptiveDetector(app, enabled=False)
        out = det.get(dark_frame())
        self.assertEqual(out, [])
        self.assertEqual(det.stats["retries"], 0)


class RecordingRecognition:
    def __init__(self):
        self.image_means = []

    def get(self, img, face):
        self.image_means.append(float(img.mean()))


class RecognitionSourceTests(unittest.TestCase):
    def test_enhanced_retry_embeds_on_the_original_plate(self):
        # The retry DETECTS on the brightened copy, but identity thresholds
        # were calibrated on plate pixels -- recognition must read the dark
        # ORIGINAL, not the gamma/CLAHE-lifted image.
        app = FakeFaceApp(base_faces=[], retry_hits=1)
        rec = RecordingRecognition()
        app.models = {"detection": app.det_model, "recognition": rec}
        det = AdaptiveDetector(app, enabled=True)
        frame = dark_frame(20)
        faces = det._detect_enhanced(frame)
        self.assertEqual(len(faces), 1)
        self.assertAlmostEqual(rec.image_means[0], float(frame.mean()), delta=1.0)

    def test_roi_retry_embeds_on_the_original_plate(self):
        app = FakeFaceApp(base_faces=[], retry_hits=1)
        rec = RecordingRecognition()
        app.models = {"detection": app.det_model, "recognition": rec}
        det = AdaptiveDetector(app, enabled=True)
        frame = dark_frame(20)
        # FakeDetModel's kps map back to ~(25, 25) at a ~10px landmark
        # radius; the expected region matches, so the filter keeps it.
        faces = det._detect_roi(frame, center=np.array([25.0, 25.0]), radius=15.0)
        self.assertEqual(len(faces), 1)
        self.assertAlmostEqual(rec.image_means[0], float(frame.mean()), delta=1.0)


class RoiFilterTests(unittest.TestCase):
    def test_detection_far_from_expected_center_is_filtered(self):
        app = FakeFaceApp(base_faces=[], retry_hits=1)
        det = AdaptiveDetector(app, enabled=True)
        # Detection maps back to ~(57, 57); expected region is at (80, 80)
        # with an 8px radius -- some other face, not the missing track.
        faces = det._detect_roi(dark_frame(20), center=np.array([80.0, 80.0]),
                                radius=8.0)
        self.assertEqual(faces, [])
        self.assertEqual(det.stats["roi_filtered"], 1)

    def test_detection_at_implausible_scale_is_filtered(self):
        app = FakeFaceApp(base_faces=[], retry_hits=1)
        det = AdaptiveDetector(app, enabled=True)
        # Center matches, but the track expects a face ~9x larger than the
        # ~10px-radius detection.
        faces = det._detect_roi(dark_frame(20), center=np.array([25.0, 25.0]),
                                radius=90.0)
        self.assertEqual(faces, [])
        self.assertEqual(det.stats["roi_filtered"], 1)


if __name__ == "__main__":
    unittest.main()
