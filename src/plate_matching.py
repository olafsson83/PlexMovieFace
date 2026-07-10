"""Phase 3 of IMPROVEMENT_PLAN.md: sharpness/defocus matching.

The swap model synthesizes the face "clean" -- sharper than the surrounding
plate whenever the plate face is defocused, soft-lensed, or low-res. A patch
that's crisper than its neighborhood reads as pasted on even when identity
and color are perfect. This module takes over inswapper's paste-back
(swapper.get(..., paste_back=False)) so the generated layer can be degraded
to match the plate BEFORE compositing -- blurring the finished composite
would smear original plate pixels around the face into a halo.

Everything happens in the aligned crop space (the model's 128x128 face
alignment): `aimg` (the original plate face) and `bgr_fake` (the generated
face) share that space, so a sigma measured there is applied there, and the
inverse warp rescales both identically. The crop is fully opaque, so
blurring it has no alpha-edge interaction; the compositing mask's own
erode+feather (replicated from insightface below) is far softer than any
sigma this module applies.

Measurement guards (why raw Laplacian matching is not enough): Laplacian
variance responds to wrinkles, facial hair, and identity texture, not just
lens sharpness -- a generated young face can measure "softer" than an old
plate face while being perceptually crisper. So blur is only applied when
the generated face is sharper on BOTH Laplacian variance and gradient
energy, the measurement is noise-suppressed and restricted to the crop
interior, sigma comes from a bounded grid search with tolerance (exact
matching makes sigma oscillate frame to frame), and the result is smoothed
temporally per character with a position-jump reset.
"""
from __future__ import annotations

import cv2
import numpy as np
from insightface.utils import face_align

from config import (
    SHARPNESS_MATCHING, SHARPNESS_TOLERANCE, SHARPNESS_MAX_SIGMA,
    SHARPNESS_TEMPORAL_SMOOTHING,
)

SIGMA_CANDIDATES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
# Crop border excluded from measurement: the warp pulls in background there,
# and the compositing mask erodes it away anyway.
INTERIOR_MARGIN_FRACTION = 0.16
# A track whose face center jumps farther than this (in frame pixels) gets
# its sigma smoothing reset -- it's a cut or a different instance.
SMOOTHING_RESET_DISTANCE = 80.0


def _interior_mask(size):
    margin = int(size * INTERIOR_MARGIN_FRACTION)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[margin:size - margin, margin:size - margin] = 1
    return mask.astype(bool)


def sharpness_metrics(bgr, interior):
    """(laplacian variance, gradient energy) over the crop interior, with a
    light denoise first so plate compression noise doesn't inflate the plate's
    apparent sharpness and mask a genuinely-too-crisp generated face.
    """
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    y = cv2.GaussianBlur(y, (0, 0), 0.5)
    lap = cv2.Laplacian(y, cv2.CV_32F)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1)
    lap_var = float(lap[interior].var())
    grad_energy = float((gx * gx + gy * gy)[interior].mean())
    return lap_var, grad_energy


def paste_back(frame, bgr_fake, M):
    """insightface INSwapper.get(paste_back=True)'s compositing, reproduced
    so the generated layer can be modified first. (Its fake_diff branch is
    dead code -- the merge only ever uses the eroded/feathered white mask.)
    With no modification, output is identical to the original paste-back.
    """
    IM = cv2.invertAffineTransform(M)
    h, w = frame.shape[:2]
    crop_h, crop_w = bgr_fake.shape[:2]

    warped_fake = cv2.warpAffine(bgr_fake, IM, (w, h), borderValue=0.0)
    white = np.full((crop_h, crop_w), 255, dtype=np.float32)
    mask = cv2.warpAffine(white, IM, (w, h), borderValue=0.0)
    mask[mask > 20] = 255

    inds = np.where(mask == 255)
    if len(inds[0]) == 0:
        return frame
    mask_h = inds[0].max() - inds[0].min()
    mask_w = inds[1].max() - inds[1].min()
    mask_size = int(np.sqrt(mask_h * mask_w))

    k = max(mask_size // 10, 10)
    mask = cv2.erode(mask, np.ones((k, k), np.uint8), iterations=1)
    k = max(mask_size // 20, 5)
    mask = cv2.GaussianBlur(mask, (2 * k + 1, 2 * k + 1), 0)
    mask = (mask / 255.0)[:, :, None]

    merged = mask * warped_fake + (1 - mask) * frame.astype(np.float32)
    return merged.astype(np.uint8)


class SharpnessMatcher:
    """Chooses (and temporally smooths) the residual Gaussian sigma that
    brings the generated crop's sharpness down to the plate crop's."""

    def __init__(self, tolerance=SHARPNESS_TOLERANCE, max_sigma=SHARPNESS_MAX_SIGMA,
                 smoothing=SHARPNESS_TEMPORAL_SMOOTHING):
        self.tolerance = tolerance
        self.max_sigma = max_sigma
        self.smoothing = smoothing  # weight on the PREVIOUS smoothed value
        self._state = {}            # character_number -> (ema_sigma, center)
        self.applied = []           # per-swap sigmas, for the run summary

    def reset(self):
        self._state.clear()

    def choose_sigma(self, plate_crop, fake_crop, track_key, center):
        interior = _interior_mask(fake_crop.shape[0])
        plate_lap, plate_grad = sharpness_metrics(plate_crop, interior)
        fake_lap, fake_grad = sharpness_metrics(fake_crop, interior)

        raw = 0.0
        # Only degrade when the generated face is clearly sharper on both
        # metrics -- identity texture differences alone must not trigger blur.
        if fake_lap > plate_lap * (1 + self.tolerance) and fake_grad > plate_grad:
            y = cv2.cvtColor(fake_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
            target = plate_lap * (1 + self.tolerance)
            raw = self.max_sigma
            for sigma in SIGMA_CANDIDATES:
                if sigma > self.max_sigma:
                    break
                blurred = cv2.GaussianBlur(y, (0, 0), sigma)
                blurred = cv2.GaussianBlur(blurred, (0, 0), 0.5)  # match metric path
                lap = cv2.Laplacian(blurred, cv2.CV_32F)
                if float(lap[interior].var()) <= target:
                    raw = sigma
                    break

        prev = self._state.get(track_key)
        if prev is not None and np.linalg.norm(center - prev[1]) < SMOOTHING_RESET_DISTANCE:
            smoothed = (1 - self.smoothing) * raw + self.smoothing * prev[0]
        else:
            smoothed = raw
        self._state[track_key] = (smoothed, center)
        return smoothed

    def summary(self):
        if not self.applied:
            return "sharpness matching: no faces processed"
        arr = np.array(self.applied)
        nonzero = arr[arr > 0.05]
        return (
            f"sharpness matching: {len(nonzero)}/{len(arr)} swaps blurred "
            f"(mean sigma {nonzero.mean():.2f}, max {arr.max():.2f})"
            if len(nonzero) else
            f"sharpness matching: 0/{len(arr)} swaps needed blur (plate as sharp as swap)"
        )


class PlateMatcher:
    """Orchestrates swap inference + plate matching + compositing. This is
    the insertion point for later phases (grain, motion blur) -- they slot
    into swap() between inference and paste_back in pipeline order."""

    def __init__(self, swapper):
        self.swapper = swapper
        self.sharpness = SharpnessMatcher() if SHARPNESS_MATCHING else None

    def swap(self, frame, tracked_face, source_face):
        bgr_fake, M = self.swapper.get(frame, tracked_face, source_face, paste_back=False)

        if self.sharpness is not None:
            # The plate must be measured from the UNswapped frame; frame is
            # untouched until paste_back below, so this crop is clean.
            plate_crop, _ = face_align.norm_crop2(
                frame, tracked_face.kps, bgr_fake.shape[0]
            )
            center = np.asarray(tracked_face.kps).mean(axis=0)
            sigma = self.sharpness.choose_sigma(
                plate_crop, bgr_fake, tracked_face.character_number, center
            )
            self.sharpness.applied.append(sigma)
            if sigma > 0.05:
                bgr_fake = cv2.GaussianBlur(bgr_fake, (0, 0), sigma)

        return paste_back(frame, bgr_fake, M)

    def summary(self):
        return self.sharpness.summary() if self.sharpness else "plate matching disabled"
