"""Swap backend interface (v2 milestone 7).

inswapper_128 is not a movie-resolution face renderer: it synthesizes the
aligned face at 128x128, softens eyes/teeth, and degrades at strong poses.
The pipeline therefore talks to a backend INTERFACE, with inswapper as the
fast baseline, so higher-resolution or temporally aware models can be
benchmarked against the same fixture suite and swapped in per-project.
No backend is accepted on screenshots alone -- benchmark_backends.py
measures identity preservation, sharpness, and latency on real fixtures.

A backend's contract:
    prepare_source(face)  -> backend-specific source representation
    swap(frame, target_face, prepared_source) -> (aligned_fake_crop, M)
    capabilities()        -> dict describing limits (crop size, yaw range)

The returned crop + alignment transform feed the plate-matching pipeline
(motion blur -> sharpness -> warp -> grain -> composite) unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

GFPGAN_MODEL_PATH = Path.home() / ".insightface" / "models" / "gfpgan_1.4.onnx"
SIMSWAP_MODEL_PATH = Path.home() / ".insightface" / "models" / "simswap_unofficial_512.onnx"
CROSSFACE_MODEL_PATH = Path.home() / ".insightface" / "models" / "crossface_simswap.onnx"

# facefusion's arcface_112_v1 five-point alignment template (fractions of
# the crop size). SimSwap was trained on this alignment, not insightface's
# arcface_128 variant, so the crop must be built from THIS template.
ARCFACE_112_V1 = np.array([
    [0.35473214, 0.45658929],
    [0.64526786, 0.45658929],
    [0.50000000, 0.61154464],
    [0.37913393, 0.77687500],
    [0.62086607, 0.77687500],
], dtype=np.float32)


def warp_by_template(frame, kps, template, size):
    """Aligns a face crop by similarity transform from its 5 landmarks to a
    normalized template, returning (crop, frame->crop affine) in the same
    convention the rest of the pipeline uses (insightface's norm_crop2)."""
    dst = template * size
    M, _ = cv2.estimateAffinePartial2D(
        np.asarray(kps, dtype=np.float32), dst, method=cv2.RANSAC,
        ransacReprojThreshold=100,
    )
    crop = cv2.warpAffine(frame, M, (size, size),
                          borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
    return crop, M


class InswapperBackend:
    """The baseline: insightface INSwapper at 128px aligned resolution."""

    name = "inswapper_128"

    def __init__(self, swapper):
        self.swapper = swapper

    def prepare_source(self, source_face):
        # INSwapper consumes the insightface Face directly (it projects the
        # 512-d embedding through its internal emap at swap time).
        return source_face

    def swap(self, frame, target_face, prepared_source):
        return self.swapper.get(frame, target_face, prepared_source, paste_back=False)

    def capabilities(self):
        return {
            "name": self.name,
            "crop_size": 128,
            # Beyond this the five-point alignment visibly breaks -- the
            # pose gate's MAX_ABS_YAW default matches it.
            "reliable_abs_yaw": 65,
            "temporally_aware": False,
        }


def _scale_affine(M, factor):
    """Rescale a frame->crop alignment to a larger crop resolution: points
    scale uniformly, so both the linear part and the translation multiply."""
    return np.asarray(M, dtype=np.float64) * float(factor)


class InswapperGfpganBackend:
    """inswapper identity synthesis at 128px, then GFPGAN v1.4 face
    restoration at 512px before compositing -- attacks the detail ceiling
    (soft eyes/teeth/skin) that no downstream plate matching can lift.
    Alignment limits are unchanged (still five-point inswapper underneath),
    so the pose gate stays at the same yaw range.

    GFPGAN can over-beautify (plastic skin, altered micro-identity), so the
    enhanced crop is blended over the plain upscale (GFPGAN_BLEND); the
    sharpness/grain matchers then fit the result to the plate as usual.
    """

    name = "inswapper_gfpgan"

    def __init__(self, swapper, session, blend=None):
        from config import GFPGAN_BLEND
        self.inner = InswapperBackend(swapper)
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.blend = GFPGAN_BLEND if blend is None else blend

    def prepare_source(self, source_face):
        return self.inner.prepare_source(source_face)

    def swap(self, frame, target_face, prepared_source):
        bgr_fake, M = self.inner.swap(frame, target_face, prepared_source)
        upscaled = cv2.resize(bgr_fake, (512, 512), interpolation=cv2.INTER_CUBIC)

        blob = upscaled[:, :, ::-1].astype(np.float32) / 255.0 * 2.0 - 1.0
        blob = blob.transpose(2, 0, 1)[None]
        out = self.session.run(None, {self.input_name: blob})[0][0]
        enhanced = np.clip((out.transpose(1, 2, 0) + 1.0) / 2.0 * 255.0, 0, 255)
        enhanced = enhanced[:, :, ::-1]  # back to BGR

        crop = np.clip(
            self.blend * enhanced + (1.0 - self.blend) * upscaled.astype(np.float32),
            0, 255,
        ).astype(np.uint8)
        return crop, _scale_affine(M, 512 / bgr_fake.shape[0])

    def capabilities(self):
        return {
            "name": self.name,
            "crop_size": 512,
            "reliable_abs_yaw": 65,  # alignment is still inswapper's 5-point
            "temporally_aware": False,
            "blend": self.blend,
        }


class SimswapBackend:
    """SimSwap 512 (neuralchen, via facefusion's ONNX conversion): a
    different synthesis family with materially better pose tolerance than
    inswapper's five-point-bound generator -- the candidate for recovering
    frames where identity is certain but yaw exceeds inswapper's range.

    Recipe replicated from facefusion 3.7: arcface_112_v1 alignment at
    512px, target as RGB/255 NCHW, source as the RAW insightface embedding
    passed through the crossface_simswap converter and L2-normalized.
    Non-commercial license (personal use only, same as this project).
    """

    name = "simswap_512"

    def __init__(self, session, converter):
        self.session = session
        self.converter = converter

    def prepare_source(self, source_face):
        embedding = np.asarray(source_face.embedding, dtype=np.float32).reshape(-1, 512)
        converted = self.converter.run(None, {"input": embedding})[0].ravel()
        converted = converted / np.linalg.norm(converted)
        return converted.reshape(1, -1).astype(np.float32)

    def swap(self, frame, target_face, prepared_source):
        crop, M = warp_by_template(frame, target_face.kps, ARCFACE_112_V1, 512)
        blob = crop[:, :, ::-1].astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None]
        out = self.session.run(None, {"target": blob, "source": prepared_source})[0][0]
        out = (out.transpose(1, 2, 0) * 255.0).round().clip(0, 255)
        return out[:, :, ::-1].astype(np.uint8), M

    def capabilities(self):
        return {
            "name": self.name,
            "crop_size": 512,
            "reliable_abs_yaw": 80,  # the reason this backend exists
            "temporally_aware": False,
        }


def yaw_proxy(kps):
    """Head-yaw magnitude estimated from the 5 alignment landmarks alone --
    the only pose signal available at render time, since the swap plan does
    not store the analysis pass's 3D-landmark yaw. Roll-correct by projecting
    onto the eye-line axis, then measure how far the nose leads the mouth
    midpoint along it, normalized by inter-eye distance (which shrinks with
    cos(yaw), usefully amplifying the signal). Calibrated against buffalo_l
    yaw on real fixtures: median 0.06 frontal, ~0.55 at 40-55 degrees, >1.4
    at 65+ degrees.
    """
    kps = np.asarray(kps, dtype=np.float64)
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_vec = right_eye - left_eye
    eye_dist = np.linalg.norm(eye_vec)
    if eye_dist < 1e-6:
        return 10.0  # degenerate landmarks: treat as extreme
    axis = eye_vec / eye_dist
    mid = (left_eye + right_eye) / 2.0
    nose_off = float(np.dot(nose - mid, axis))
    mouth_off = float(np.dot((kps[3] + kps[4]) / 2.0 - mid, axis))
    return abs(nose_off - mouth_off) / eye_dist


class HybridBackend:
    """Pose-routing composite: the primary backend wherever five-point
    alignment holds, SimSwap in the extreme-yaw band it cannot render.

    The two backends fail differently: inswapper transfers identity strongly
    (0.85 measured) but its five-point warp breaks past ~65 degrees yaw;
    SimSwap renders cleanly at 80+ degrees but only shifts identity halfway
    (0.53 measured). Routing exploits that the costs are asymmetric -- on a
    near-profile face a half-strength identity beats the untouched original
    (what the pose gate would otherwise leave), while frontal faces keep the
    strong swap. Run analysis with MAX_ABS_YAW near SimSwap's 80-degree
    ceiling so the extreme frames reach the plan at all.
    """

    name = "hybrid"

    def __init__(self, primary, extreme, threshold=None):
        from config import HYBRID_PROXY_THRESHOLD
        self.primary = primary
        self.extreme = extreme
        self.threshold = HYBRID_PROXY_THRESHOLD if threshold is None else threshold
        self.routed = {"primary": 0, "extreme": 0}

    def prepare_source(self, source_face):
        return {
            "primary": self.primary.prepare_source(source_face),
            "extreme": self.extreme.prepare_source(source_face),
        }

    def swap(self, frame, target_face, prepared_source):
        route = "extreme" if yaw_proxy(target_face.kps) > self.threshold else "primary"
        self.routed[route] += 1
        backend = self.extreme if route == "extreme" else self.primary
        return backend.swap(frame, target_face, prepared_source[route])

    def capabilities(self):
        return {
            "name": self.name,
            "primary": self.primary.capabilities(),
            "extreme": self.extreme.capabilities(),
            "reliable_abs_yaw": self.extreme.capabilities()["reliable_abs_yaw"],
            "proxy_threshold": self.threshold,
            "temporally_aware": False,
        }


def _load_onnx_session(path, label):
    if not path.exists():
        raise SystemExit(
            f"SWAP_BACKEND needs {path} -- download {path.name} from "
            "huggingface.co/facefusion (models-3.0.0 / models-3.4.0) into that folder."
        )
    import onnxruntime
    import gpu_runtime
    from config import CTX_ID
    gpu_requested = CTX_ID >= 0
    providers = gpu_runtime.requested_providers(gpu_requested)
    session = onnxruntime.InferenceSession(str(path), providers=providers)
    gpu_runtime.enforce_session_provider(session, gpu_requested, label)
    return session


def _load_gfpgan_session():
    if not GFPGAN_MODEL_PATH.exists():
        raise SystemExit(
            f"SWAP_BACKEND=inswapper_gfpgan needs {GFPGAN_MODEL_PATH} -- download "
            "gfpgan_1.4.onnx (~340MB) from huggingface.co/facefusion/models-3.0.0 "
            "into that folder."
        )
    import onnxruntime
    import gpu_runtime
    from config import CTX_ID
    gpu_requested = CTX_ID >= 0
    providers = gpu_runtime.requested_providers(gpu_requested)
    session = onnxruntime.InferenceSession(str(GFPGAN_MODEL_PATH), providers=providers)
    gpu_runtime.enforce_session_provider(session, gpu_requested, "GFPGAN enhancer")
    return session


def build_backend(swapper=None):
    """Backend factory keyed by SWAP_BACKEND (env)."""
    kind = os.environ.get("SWAP_BACKEND", "inswapper").lower()
    if kind in ("inswapper", "inswapper_gfpgan"):
        if swapper is None:
            import face_engine
            swapper = face_engine.build_swapper()
        if kind == "inswapper":
            return InswapperBackend(swapper)
        return InswapperGfpganBackend(swapper, _load_gfpgan_session())
    if kind == "simswap_512":
        return SimswapBackend(
            _load_onnx_session(SIMSWAP_MODEL_PATH, "SimSwap 512"),
            _load_onnx_session(CROSSFACE_MODEL_PATH, "crossface converter"),
        )
    if kind == "hybrid":
        from config import HYBRID_PRIMARY
        if HYBRID_PRIMARY not in ("inswapper", "inswapper_gfpgan"):
            raise SystemExit(
                f"HYBRID_PRIMARY '{HYBRID_PRIMARY}' must be inswapper or inswapper_gfpgan."
            )
        if swapper is None:
            import face_engine
            swapper = face_engine.build_swapper()
        primary = (InswapperBackend(swapper) if HYBRID_PRIMARY == "inswapper"
                   else InswapperGfpganBackend(swapper, _load_gfpgan_session()))
        extreme = SimswapBackend(
            _load_onnx_session(SIMSWAP_MODEL_PATH, "SimSwap 512"),
            _load_onnx_session(CROSSFACE_MODEL_PATH, "crossface converter"),
        )
        return HybridBackend(primary, extreme)
    raise SystemExit(
        f"Unknown SWAP_BACKEND '{kind}'. Available: inswapper, inswapper_gfpgan, "
        "simswap_512, hybrid. Candidate backends must pass benchmark_backends.py "
        "before being wired in."
    )
