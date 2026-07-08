"""Shared insightface model loading, used by both discover_characters.py and
swap_movie.py. Split into two builders since discovery only ever needs
detection + embeddings, not the swap model itself.
"""
import numpy as np
import onnxruntime
import insightface
from insightface.app import FaceAnalysis

from config import CTX_ID
import preflight


def build_face_app() -> FaceAnalysis:
    """Loads the buffalo_l detector + recognition model (no swap model)."""
    # The GPU build's CUDA/cuDNN runtime DLLs live inside their own pip
    # packages (nvidia-cublas-cu12 etc.), not on the normal Windows DLL
    # search path -- without this, onnxruntime silently falls back to CPU
    # even though CUDAExecutionProvider is "available". Safe to call
    # unconditionally: it's a no-op on the CPU-only onnxruntime package.
    onnxruntime.preload_dlls()
    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=CTX_ID, det_size=(640, 640))
    return face_app


def build_swapper():
    """Loads the inswapper_128 swap model."""
    onnxruntime.preload_dlls()
    # get_model() only joins a name with the model root if it does NOT end in
    # ".onnx" -- pass a name ending in .onnx and it's used as a literal path
    # instead (relative to the current working directory), which silently
    # fails unless you happen to run from inside ~/.insightface/models/.
    # Passing the fully resolved path sidesteps that entirely.
    return insightface.model_zoo.get_model(str(preflight.MODEL_PATH), download=False, download_zip=False)


def cosine_similarity(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
