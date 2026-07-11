"""Adaptive low-light detection (v2 milestone 3).

The base pipeline hands the detector one unmodified frame at a fixed 640px
canvas -- so dark scenes and small faces fail before identity logic ever
gets a chance. This wraps FaceAnalysis with a staged retry that runs ONLY
when the cheap pass looks insufficient:

    base:  detect on the original frame (unchanged behavior)
    retry: if the frame is dark AND (nothing was found, or a currently
           tracked face region came back empty) -> detect again on an
           analysis-only enhanced copy (adaptive gamma lift + CLAHE on
           luma) at a larger detector canvas, then merge by IoU.

The enhanced image exists purely for detection, landmarks, and embeddings.
The original plate is what gets swapped and composited -- enhancement never
touches output pixels. Full-frame enhanced retry coordinates are already in
original-frame space, so no mapping is needed (per-ROI upscale retries are
a later refinement; see IMPROVEMENT_PLAN.md).

Duck-types FaceAnalysis.get(), so analysis call sites wrap and pass it
wherever a face_app is expected. Disabled unless ADAPTIVE_DETECTION is on.
"""
from __future__ import annotations

import cv2
import numpy as np
from insightface.app.common import Face

from config import (
    ADAPTIVE_DETECTION, ADAPTIVE_DARK_LUMA, ADAPTIVE_RETRY_DET_SIZE,
    ADAPTIVE_GAMMA_MAX, ADAPTIVE_ROI_RETRY, ADAPTIVE_ROI_UPSCALE,
)


def enhance_for_analysis(frame, gamma_max=ADAPTIVE_GAMMA_MAX):
    """Analysis-only luminance lift: adaptive gamma toward mid-gray, then
    CLAHE for local contrast. Chroma is untouched (YCrCb luma path)."""
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y = ycc[:, :, 0]
    mean = float(y.mean())
    if mean <= 0:
        return frame.copy()
    # Gamma that maps the current mean toward 128: solving
    # (mean/255)^(1/gamma) = 128/255 gives gamma = log(mean/255)/log(128/255).
    # Clipped so bright frames are untouched (gamma < 1 would darken) and
    # black frames aren't amplified into pure noise.
    gamma = float(np.clip(np.log(max(mean, 1.0) / 255.0) / np.log(128.0 / 255.0),
                          1.0, gamma_max))
    lut = ((np.arange(256) / 255.0) ** (1.0 / gamma) * 255.0).astype(np.uint8)
    y = cv2.LUT(y, lut)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ycc[:, :, 0] = clahe.apply(y)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def merge_detections(base_faces, retry_faces, iou_threshold=0.4):
    """Base detections win (they came from the true plate); retry faces are
    added only where no base detection overlaps."""
    merged = list(base_faces)
    for face in retry_faces:
        if all(_iou(face.bbox, b.bbox) < iou_threshold for b in base_faces):
            merged.append(face)
    return merged


class AdaptiveDetector:
    """Drop-in face_app wrapper for the ANALYSIS side of the pipeline."""

    def __init__(self, face_app, enabled=ADAPTIVE_DETECTION,
                 dark_luma=ADAPTIVE_DARK_LUMA, retry_det_size=ADAPTIVE_RETRY_DET_SIZE):
        self.face_app = face_app
        self.enabled = enabled
        self.dark_luma = dark_luma
        self.retry_det_size = int(retry_det_size)
        self.identity_mgr = None  # bound after construction (expected regions)
        self.stats = {"frames": 0, "dark_frames": 0, "retries": 0, "recovered_faces": 0,
                      "roi_retries": 0, "roi_recovered": 0}

    def bind(self, identity_mgr):
        self.identity_mgr = identity_mgr
        return self

    def _expected_regions(self):
        """(center, radius) for every live identity-manager track -- a region
        that was a face moments ago and should usually still contain one."""
        if self.identity_mgr is None:
            return []
        regions = []
        for track in getattr(self.identity_mgr, "_tracks", []):
            kps = np.asarray(track.kps)
            spread = kps.max(axis=0) - kps.min(axis=0)
            radius = 1.5 * float(np.hypot(*spread)) if spread.any() else 40.0
            regions.append((kps.mean(axis=0), radius))
        return regions

    def _missing_regions(self, faces):
        missing = []
        for center, radius in self._expected_regions():
            hit = any(
                np.linalg.norm(np.asarray(f.kps).mean(axis=0) - center) < radius
                for f in faces
            )
            if not hit:
                missing.append((center, radius))
        return missing

    def _detect_enhanced(self, frame):
        enhanced = enhance_for_analysis(frame)
        size = (self.retry_det_size, self.retry_det_size)
        bboxes, kpss = self.face_app.det_model.detect(enhanced, input_size=size)
        faces = []
        for i in range(bboxes.shape[0]):
            face = Face(bbox=bboxes[i, 0:4], kps=kpss[i], det_score=bboxes[i, 4])
            # Landmarks/embedding models also read the enhanced image --
            # analysis only; the kps land in original-frame coordinates
            # because the enhancement is a pure per-pixel transform.
            for taskname, model in self.face_app.models.items():
                if taskname == "detection":
                    continue
                model.get(enhanced, face)
            faces.append(face)
        return faces

    def _detect_roi(self, frame, center, radius):
        """Crop the missing track region, enhance + upscale it, and detect
        there -- a face too small or too blurred for the full-frame canvas
        can still resolve on a magnified crop. All coordinates are mapped
        back to original-frame space; the crop exists for analysis only.
        """
        h, w = frame.shape[:2]
        r = max(radius * 2.0, 48.0)
        x0, y0 = int(max(0, center[0] - r)), int(max(0, center[1] - r))
        x1, y1 = int(min(w, center[0] + r)), int(min(h, center[1] + r))
        if x1 - x0 < 32 or y1 - y0 < 32:
            return []
        crop = enhance_for_analysis(frame[y0:y1, x0:x1])
        up = cv2.resize(crop, None, fx=ADAPTIVE_ROI_UPSCALE, fy=ADAPTIVE_ROI_UPSCALE,
                        interpolation=cv2.INTER_CUBIC)
        bboxes, kpss = self.face_app.det_model.detect(up, input_size=(640, 640))
        faces = []
        for i in range(bboxes.shape[0]):
            face = Face(bbox=bboxes[i, 0:4], kps=kpss[i], det_score=bboxes[i, 4])
            for taskname, model in self.face_app.models.items():
                if taskname == "detection":
                    continue
                model.get(up, face)  # landmarks/embedding read the crop
            offset = np.array([x0, y0], dtype=np.float32)
            face.kps = face.kps / ADAPTIVE_ROI_UPSCALE + offset
            face.bbox = np.concatenate([
                face.bbox[:2] / ADAPTIVE_ROI_UPSCALE + offset,
                face.bbox[2:4] / ADAPTIVE_ROI_UPSCALE + offset,
            ])
            faces.append(face)
        return faces

    def get(self, frame):
        base = self.face_app.get(frame)
        if not self.enabled:
            return base

        self.stats["frames"] += 1
        luma = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
        dark = luma < self.dark_luma
        if dark:
            self.stats["dark_frames"] += 1
        if not dark or (base and not self._missing_regions(base)):
            return base

        self.stats["retries"] += 1
        retry = self._detect_enhanced(frame)
        merged = merge_detections(base, retry)
        self.stats["recovered_faces"] += len(merged) - len(base)

        if ADAPTIVE_ROI_RETRY:
            for center, radius in self._missing_regions(merged):
                self.stats["roi_retries"] += 1
                roi_faces = self._detect_roi(frame, center, radius)
                if roi_faces:
                    before = len(merged)
                    merged = merge_detections(merged, roi_faces)
                    self.stats["roi_recovered"] += len(merged) - before
        return merged

    def summary(self):
        s = self.stats
        if not self.enabled:
            return "adaptive detection: disabled"
        return (
            f"adaptive detection: {s['retries']} retries on {s['dark_frames']} dark "
            f"frames (of {s['frames']}), {s['recovered_faces']} extra faces recovered; "
            f"{s['roi_retries']} ROI retries recovered {s['roi_recovered']} more"
        )
