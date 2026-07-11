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
    raise SystemExit(
        f"Unknown SWAP_BACKEND '{kind}'. Available: inswapper, inswapper_gfpgan. "
        "Candidate backends must pass benchmark_backends.py before being wired in."
    )
