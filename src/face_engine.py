"""Shared insightface model loading, used by both discover_characters.py and
swap_movie.py. Split into two builders since discovery only ever needs
detection + embeddings, not the swap model itself.
"""
import numpy as np
import insightface
from insightface.app import FaceAnalysis

from config import CTX_ID
import preflight
import gpu_runtime


GPU_REQUESTED = CTX_ID >= 0


def _providers():
    return gpu_runtime.requested_providers(GPU_REQUESTED)


def _protect_face_app_sessions(face_app):
    for task_name, model in face_app.models.items():
        session = getattr(model, "session", None)
        if session is not None:
            gpu_runtime.enforce_session_provider(session, GPU_REQUESTED, f"InsightFace {task_name} model")


def build_face_app() -> FaceAnalysis:
    """Loads the buffalo_l detector + recognition model (no swap model)."""
    # The GPU build's CUDA/cuDNN runtime DLLs live inside their own pip
    # packages (nvidia-cublas-cu12 etc.), not on the normal Windows DLL
    # search path -- without this, onnxruntime silently falls back to CPU
    # even though CUDAExecutionProvider is "available". Safe to call
    # unconditionally: it's a no-op on the CPU-only onnxruntime package.
    gpu_runtime.configure_cuda_runtime(GPU_REQUESTED)
    face_app = FaceAnalysis(name="buffalo_l", providers=_providers())
    face_app.prepare(ctx_id=CTX_ID, det_size=(640, 640))
    _protect_face_app_sessions(face_app)
    return face_app


def build_swapper():
    """Loads the inswapper_128 swap model."""
    gpu_runtime.configure_cuda_runtime(GPU_REQUESTED)
    # get_model() only joins a name with the model root if it does NOT end in
    # ".onnx" -- pass a name ending in .onnx and it's used as a literal path
    # instead (relative to the current working directory), which silently
    # fails unless you happen to run from inside ~/.insightface/models/.
    # Passing the fully resolved path sidesteps that entirely.
    swapper = insightface.model_zoo.get_model(
        str(preflight.MODEL_PATH),
        download=False,
        download_zip=False,
        providers=_providers(),
    )
    gpu_runtime.enforce_session_provider(swapper.session, GPU_REQUESTED, "INSwapper model")
    return swapper


def cosine_similarity(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
