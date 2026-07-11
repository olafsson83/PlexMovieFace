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
    GRAIN_MATCHING, GRAIN_MAX_SIGMA, GRAIN_MIN_SIGMA,
    GRAIN_TEMPORAL_SMOOTHING, GRAIN_EDGE_REJECT_PERCENTILE,
    MOTION_BLUR_MATCHING, MOTION_MIN_DISPLACEMENT_PX, MOTION_SHUTTER_FRACTION,
    MOTION_MAX_CROP_FRACTION, MOTION_TEMPORAL_SMOOTHING,
    MOTION_MIN_INLIER_RATIO, MOTION_ROTATION_LIMIT_DEG,
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
    layer = warp_generated(frame, bgr_fake, M)
    if layer is None:
        return frame
    return composite(frame, *layer)


def warp_generated(frame, bgr_fake, M):
    """Inverse-warps the generated crop into frame space and builds the
    compositing mask (insightface's exact math). Returns (warped_fake,
    soft_mask, face_region, mask_size) or None when the warp lands entirely
    off-frame. face_region is the pre-erosion binary face area -- the grain
    stage samples the plate ring just outside it.
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
        return None
    face_region = mask == 255
    mask_h = inds[0].max() - inds[0].min()
    mask_w = inds[1].max() - inds[1].min()
    mask_size = int(np.sqrt(mask_h * mask_w))

    k = max(mask_size // 10, 10)
    mask = cv2.erode(mask, np.ones((k, k), np.uint8), iterations=1)
    k = max(mask_size // 20, 5)
    mask = cv2.GaussianBlur(mask, (2 * k + 1, 2 * k + 1), 0)
    mask = mask / 255.0

    return warped_fake, mask, face_region, mask_size


def composite(frame, warped_fake, soft_mask, face_region=None, mask_size=None):
    mask = soft_mask[:, :, None]
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


def robust_noise_sigmas(bgr, sample_mask, edge_reject_pct=GRAIN_EDGE_REJECT_PERCENTILE):
    """Per-channel (Y, Cr, Cb) noise scale from high-pass residuals over the
    sampled pixels, using MAD (1.4826 * median absolute deviation) so hair,
    background texture, and hard edges don't inflate the estimate the way a
    plain standard deviation would. Saturated/crushed and strong-gradient
    pixels are rejected first -- their residuals are structure, not noise.

    Note: the residual filter removes part of the noise itself, so these are
    consistent *estimator units*, systematically below the true sigma. All
    grain math (plate measurement, generated-face measurement, deficit, and
    synthesis rescaling) stays inside the same units, so the bias cancels.

    Returns np.array([sigma_y, sigma_cr, sigma_cb]) or None when too few
    valid pixels remain for a stable estimate.
    """
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y = ycc[:, :, 0]
    valid = sample_mask & (y > 10) & (y < 245)
    if int(valid.sum()) < 300:
        return None
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1)
    grad = gx * gx + gy * gy
    valid = valid & (grad <= np.percentile(grad[valid], edge_reject_pct))
    if int(valid.sum()) < 300:
        return None

    sigmas = []
    for c in range(3):
        ch = ycc[:, :, c]
        residual = ch - cv2.GaussianBlur(ch, (0, 0), 1.0)
        r = residual[valid]
        med = float(np.median(r))
        sigmas.append(1.4826 * float(np.median(np.abs(r - med))))
    return np.array(sigmas, dtype=np.float32)


class MotionBlurMatcher:
    """Directional blur from landmark motion, applied to the generated crop
    BEFORE the sharpness matcher measures it -- the plate's sharpness
    reading already contains its motion smear, so matching total sharpness
    first and then adding motion blur would double-blur.

    Deliberately conservative: wrong blur is more visibly destructive than
    missing blur. The estimate comes from a RANSAC partial-affine fit over
    the 5 tracked landmarks; it's damped or disabled when rotation
    dominates (a face-wide linear PSF can't represent rotational smear),
    when inliers are poor, or when the displacement is a tracker glitch
    (hard length cap). Blur length = displacement x shutter fraction
    (180-degree shutter default): inter-frame displacement is NOT the
    exposure displacement. The motion vector is EMA-smoothed, which also
    smooths the kernel angle and damps sign flips from tracking noise.
    """

    def __init__(self, shutter_fraction=MOTION_SHUTTER_FRACTION,
                 min_displacement=MOTION_MIN_DISPLACEMENT_PX,
                 max_crop_fraction=MOTION_MAX_CROP_FRACTION,
                 smoothing=MOTION_TEMPORAL_SMOOTHING,
                 min_inlier_ratio=MOTION_MIN_INLIER_RATIO,
                 rotation_limit_deg=MOTION_ROTATION_LIMIT_DEG):
        self.shutter_fraction = shutter_fraction
        self.min_displacement = min_displacement
        self.max_crop_fraction = max_crop_fraction
        self.smoothing = smoothing
        self.min_inlier_ratio = min_inlier_ratio
        self.rotation_limit_deg = rotation_limit_deg
        self._state = {}       # key -> (prev_kps, ema_crop_vector)
        self.applied = []      # applied crop-space blur lengths, for summary

    def reset(self):
        self._state.clear()

    def _frame_motion(self, prev_kps, kps):
        """Frame-space exposure-motion vector from consecutive landmarks,
        with RANSAC/rotation/residual guards. Returns (dx, dy) or None."""
        prev = np.asarray(prev_kps, dtype=np.float32)
        cur = np.asarray(kps, dtype=np.float32)
        A, inliers = cv2.estimateAffinePartial2D(prev, cur, method=cv2.RANSAC,
                                                 ransacReprojThreshold=2.0)
        if A is None or inliers is None:
            return None
        inlier_ratio = float(inliers.sum()) / len(prev)
        if inlier_ratio < self.min_inlier_ratio:
            return None

        motion = cur.mean(axis=0) - prev.mean(axis=0)
        if np.linalg.norm(motion) < self.min_displacement:
            return None

        rotation_deg = abs(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
        predicted = prev @ A[:, :2].T + A[:, 2]
        residual = float(np.linalg.norm(predicted - cur, axis=1).mean())
        damp = self.shutter_fraction
        if rotation_deg > self.rotation_limit_deg or residual > 2.5:
            damp *= 0.25  # rotation/deformation-dominant: barely trust it
        return motion * damp

    def blur_crop(self, bgr_fake, kps, M, key, center):
        """Returns (possibly blurred crop, applied crop-space length)."""
        prev = self._state.get(key)
        crop_vec = None

        if prev is not None and np.linalg.norm(center - prev[0].mean(axis=0)) < SMOOTHING_RESET_DISTANCE:
            frame_vec = self._frame_motion(prev[0], kps)
            if frame_vec is not None:
                # Into crop space through the alignment's linear part, so the
                # kernel is built where the blur is applied.
                crop_vec = M[:, :2] @ frame_vec
            else:
                crop_vec = np.zeros(2, dtype=np.float32)
            ema = prev[1]
            if ema is not None:
                crop_vec = (1 - self.smoothing) * crop_vec + self.smoothing * ema
        # else: new track / jump -- no motion evidence yet, no blur.

        self._state[key] = (np.asarray(kps, dtype=np.float32).copy(),
                            None if crop_vec is None else crop_vec.copy())

        if crop_vec is None:
            self.applied.append(0.0)
            return bgr_fake, 0.0

        max_len = bgr_fake.shape[0] * self.max_crop_fraction
        length = float(np.linalg.norm(crop_vec))
        if length > max_len:
            crop_vec = crop_vec * (max_len / length)
            length = max_len
        if length < 1.5:  # sub-1.5px linear PSF is a no-op
            self.applied.append(0.0)
            return bgr_fake, 0.0

        kernel = self._linear_kernel(crop_vec)
        self.applied.append(length)
        return cv2.filter2D(bgr_fake, -1, kernel), length

    @staticmethod
    def _linear_kernel(vec):
        n = int(np.ceil(np.linalg.norm(vec))) | 1  # odd
        n = max(n, 3)
        kernel = np.zeros((n, n), dtype=np.float32)
        c = (n - 1) / 2.0
        half = np.asarray(vec, dtype=np.float64) / 2.0
        p0 = (int(round(c - half[0])), int(round(c - half[1])))
        p1 = (int(round(c + half[0])), int(round(c + half[1])))
        cv2.line(kernel, p0, p1, 1.0, 1)
        total = kernel.sum()
        if total == 0:
            kernel[int(c), int(c)] = 1.0
            total = 1.0
        return kernel / total

    def summary(self):
        if not self.applied:
            return "motion blur: no faces processed"
        arr = np.array(self.applied)
        engaged = arr[arr > 0]
        if not len(engaged):
            return f"motion blur: 0/{len(arr)} swaps blurred (static faces)"
        return (
            f"motion blur: {len(engaged)}/{len(arr)} swaps blurred "
            f"(mean length {engaged.mean():.2f}px, max {arr.max():.2f}px in crop space)"
        )


class GrainMatcher:
    """Adds the plate's missing noise texture to the generated layer in
    frame space (crop space would rescale the noise spectrum by the warp).

    Only the *deficit* is added: measured plate noise minus what the
    generated face already carries, combined variance-wise -- the swap model
    inherits some of the plate's noise through its conditioning, and adding
    the full plate amount on top would double-grain. Noise is luma-dominant
    per actual measurement (Y/Cr/Cb estimated separately), spatially
    correlated (white noise through a small blur, not per-pixel RGB static),
    and deterministically seeded per swap so it evolves frame to frame but
    reproduces run to run.
    """

    def __init__(self, max_sigma=GRAIN_MAX_SIGMA, min_sigma=GRAIN_MIN_SIGMA,
                 smoothing=GRAIN_TEMPORAL_SMOOTHING):
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.smoothing = smoothing
        self._state = {}       # character_number -> (ema_sigmas, center)
        self.applied_y = []    # per-swap applied luma sigma, for the summary

    def reset(self):
        self._state.clear()

    def match(self, frame, warped_fake, face_region, mask_size, key, center, seed):
        """Returns warped_fake with matched grain added (or unchanged)."""
        rows = np.any(face_region, axis=1)
        cols = np.any(face_region, axis=0)
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        pad = max(mask_size // 3, 12)
        r0, r1 = max(0, r0 - pad), min(frame.shape[0], r1 + pad + 1)
        c0, c1 = max(0, c0 - pad), min(frame.shape[1], c1 + pad + 1)

        face_roi = face_region[r0:r1, c0:c1]
        plate_roi = frame[r0:r1, c0:c1]
        fake_roi = warped_fake[r0:r1, c0:c1]

        # Plate noise: a ring just outside the face; generated-face noise:
        # the face interior of the warped layer, same estimator/space.
        k_in = max(mask_size // 12, 3)
        k_out = k_in + max(mask_size // 5, 8)
        face_u8 = face_roi.astype(np.uint8)
        ring = (cv2.dilate(face_u8, np.ones((k_out, k_out), np.uint8)) > 0) & ~(
            cv2.dilate(face_u8, np.ones((k_in, k_in), np.uint8)) > 0
        )
        interior = cv2.erode(face_u8, np.ones((max(mask_size // 8, 5),) * 2, np.uint8)) > 0

        plate_sigmas = robust_noise_sigmas(plate_roi, ring)
        fake_sigmas = robust_noise_sigmas(fake_roi, interior)
        raw = None
        if plate_sigmas is not None and fake_sigmas is not None:
            deficit = np.sqrt(np.maximum(plate_sigmas ** 2 - fake_sigmas ** 2, 0.0))
            raw = np.minimum(deficit, self.max_sigma)

        prev = self._state.get(key)
        if raw is None:
            if prev is None or np.linalg.norm(center - prev[1]) >= SMOOTHING_RESET_DISTANCE:
                self.applied_y.append(0.0)
                return warped_fake
            target = prev[0]  # no fresh estimate; reuse the smoothed model
        elif prev is not None and np.linalg.norm(center - prev[1]) < SMOOTHING_RESET_DISTANCE:
            target = (1 - self.smoothing) * raw + self.smoothing * prev[0]
        else:
            target = raw
        self._state[key] = (target, center)

        self.applied_y.append(float(target[0]))
        if target[0] < self.min_sigma:
            return warped_fake

        warped_fake = warped_fake.copy()
        warped_fake[r0:r1, c0:c1] = self._grained(fake_roi, target, seed)
        return warped_fake

    def _grained(self, fake_roi, target_sigmas, seed):
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal((*fake_roi.shape[:2], 3), dtype=np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), 0.5)  # spatial correlation

        # Rescale each channel so OUR OWN estimator reads the target on the
        # synthetic field -- keeps synthesis in the same units as measurement
        # without deriving the residual filter's attenuation analytically.
        everywhere = np.ones(fake_roi.shape[:2], dtype=bool)
        for c in range(3):
            ch = noise[:, :, c]
            residual = ch - cv2.GaussianBlur(ch, (0, 0), 1.0)
            med = float(np.median(residual[everywhere]))
            est = 1.4826 * float(np.median(np.abs(residual[everywhere] - med)))
            noise[:, :, c] *= target_sigmas[c] / max(est, 1e-6)

        ycc = cv2.cvtColor(fake_roi, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        ycc += noise
        # Convert back via uint8: cv2's float color conversions assume [0,1]
        # range with a 0.5 chroma offset, which would wreck 0..255 data.
        ycc_u8 = np.clip(np.rint(ycc), 0, 255).astype(np.uint8)
        return cv2.cvtColor(ycc_u8, cv2.COLOR_YCrCb2BGR)

    def summary(self):
        if not self.applied_y:
            return "grain matching: no faces processed"
        arr = np.array(self.applied_y)
        engaged = arr[arr >= self.min_sigma]
        if not len(engaged):
            return f"grain matching: 0/{len(arr)} swaps needed grain"
        return (
            f"grain matching: {len(engaged)}/{len(arr)} swaps grained "
            f"(mean luma sigma {engaged.mean():.2f}, max {arr.max():.2f})"
        )


class PlateMatcher:
    """Orchestrates swap inference + plate matching + compositing, in
    pipeline order: motion blur -> residual sharpness in crop space ->
    warp to frame space -> grain -> composite. Grain is terminal before
    the merge because it's a sensor/encode phenomenon layered on top of
    all optics."""

    def __init__(self, swapper):
        self.swapper = swapper
        self.motion = MotionBlurMatcher() if MOTION_BLUR_MATCHING else None
        self.sharpness = SharpnessMatcher() if SHARPNESS_MATCHING else None
        self.grain = GrainMatcher() if GRAIN_MATCHING else None
        self._swap_counter = 0

    def swap(self, frame, tracked_face, source_face):
        bgr_fake, M = self.swapper.get(frame, tracked_face, source_face, paste_back=False)
        self._swap_counter += 1
        center = np.asarray(tracked_face.kps).mean(axis=0)

        if self.motion is not None:
            bgr_fake, _ = self.motion.blur_crop(
                bgr_fake, tracked_face.kps, M, tracked_face.character_number, center
            )

        if self.sharpness is not None:
            # The plate must be measured from the UNswapped frame; frame is
            # untouched until composite below, so this crop is clean.
            plate_crop, _ = face_align.norm_crop2(
                frame, tracked_face.kps, bgr_fake.shape[0]
            )
            sigma = self.sharpness.choose_sigma(
                plate_crop, bgr_fake, tracked_face.character_number, center
            )
            self.sharpness.applied.append(sigma)
            if sigma > 0.05:
                bgr_fake = cv2.GaussianBlur(bgr_fake, (0, 0), sigma)

        layer = warp_generated(frame, bgr_fake, M)
        if layer is None:
            return frame
        warped_fake, soft_mask, face_region, mask_size = layer

        if self.grain is not None:
            warped_fake = self.grain.match(
                frame, warped_fake, face_region, mask_size,
                tracked_face.character_number, center, self._swap_counter,
            )

        return composite(frame, warped_fake, soft_mask)

    def summary(self):
        parts = []
        if self.motion is not None:
            parts.append(self.motion.summary())
        if self.sharpness is not None:
            parts.append(self.sharpness.summary())
        if self.grain is not None:
            parts.append(self.grain.summary())
        return "\n".join(parts) if parts else "plate matching disabled"
