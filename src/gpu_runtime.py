"""Windows CUDA runtime setup and verification for ONNX Runtime.

The project pins onnxruntime-gpu==1.26.0 + nvidia-cudnn-cu12==9.10.0.56
(see setup.py) instead of letting pip resolve an arbitrary cuDNN 9.x release.
cuDNN 9.24.0.43 (what pip picked before this pin) ships a lazily loaded
"tensor_ir" JIT-compilation engine plugin (cudnn_engines_tensor_ir64_9.dll)
that fails with CUDNN_BACKEND_API_FAILED on this hardware/driver combination
the first time a convolution actually runs -- session construction succeeds,
so the failure only surfaces mid-inference. cuDNN 9.10.0.56 does not ship
that engine plugin at all (only cudnn_engines_precompiled64_9.dll and
cudnn_engines_runtime_compiled64_9.dll), so pinning to it sidesteps the bug
entirely rather than working around it. Verified end-to-end: real face
detection + face swap inference on CUDAExecutionProvider with CPU fallback
disabled, no CUDNN_BACKEND_API_FAILED.

Keep both DLL-directory and explicitly loaded-DLL handles alive for the life
of the process. Discarding the object returned by ``os.add_dll_directory``
removes that directory from the process search path again.
"""
from __future__ import annotations

import ctypes
import os
import platform
import site
from pathlib import Path

import numpy as np
import onnxruntime


_DLL_DIRECTORY_HANDLES = []
_DLL_HANDLES = []
_CONFIGURED = False


def _site_package_roots() -> list[Path]:
    roots = [Path(path) for path in site.getsitepackages()]
    user_root = site.getusersitepackages()
    if user_root:
        roots.append(Path(user_root))

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(roots))


def _nvidia_bin_directories() -> list[Path]:
    directories = []
    for root in _site_package_roots():
        nvidia_root = root / "nvidia"
        if nvidia_root.is_dir():
            directories.extend(path for path in nvidia_root.rglob("bin") if path.is_dir())
    return list(dict.fromkeys(directories))


def _load_matching(directories: list[Path], pattern: str, required: bool = False) -> None:
    matches = []
    for directory in directories:
        matches.extend(sorted(directory.glob(pattern)))

    if required and not matches:
        raise RuntimeError(
            f"Required CUDA DLL matching {pattern!r} was not found inside this project's virtual environment. "
            "Delete .venv and run setup.bat again."
        )

    for path in matches:
        try:
            _DLL_HANDLES.append(ctypes.WinDLL(str(path)))
        except OSError as exc:
            if required:
                raise RuntimeError(f"Could not load required CUDA DLL {path}: {exc}") from exc


def configure_cuda_runtime(gpu_requested: bool) -> None:
    """Make this venv's CUDA/cuDNN DLLs win over system-wide installations."""
    global _CONFIGURED
    if _CONFIGURED or not gpu_requested:
        return

    if platform.system() != "Windows":
        preload = getattr(onnxruntime, "preload_dlls", None)
        if preload:
            preload(directory="")
        _CONFIGURED = True
        return

    directories = _nvidia_bin_directories()
    if not directories:
        raise RuntimeError(
            "No pip-packaged NVIDIA runtime DLLs were found in .venv. "
            "Delete .venv and run setup.bat again."
        )

    # Put the isolated venv runtime ahead of globally installed CUDA 11/12/13
    # toolkits for libraries that perform their own LoadLibrary-by-name calls.
    existing_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join([*(str(path) for path in directories), existing_path])

    for directory in directories:
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(directory)))

    preload = getattr(onnxruntime, "preload_dlls", None)
    if preload:
        preload(directory="")

    _load_matching(directories, "nvJitLink*.dll")
    _load_matching(directories, "nvrtc-builtins*.dll")
    _load_matching(directories, "nvrtc64*.dll")
    # Present on cuDNN 9.24.x (where it's the source of CUDNN_BACKEND_API_FAILED
    # on this hardware) but absent on the pinned 9.10.0.56 -- load it only if
    # a future cuDNN bump reintroduces it.
    _load_matching(directories, "cudnn_engines_tensor_ir64_9.dll", required=False)

    _CONFIGURED = True


def requested_providers(gpu_requested: bool) -> list[str]:
    if not gpu_requested:
        return ["CPUExecutionProvider"]

    configure_cuda_runtime(True)
    available = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "CUDAExecutionProvider is not available even though GPU mode is enabled. "
            f"ONNX Runtime reports: {available}. Delete .venv and run setup.bat again."
        )
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def enforce_session_provider(session, gpu_requested: bool, label: str) -> None:
    """Fail loudly instead of allowing ONNX Runtime to retry on CPU."""
    providers = session.get_providers()
    if gpu_requested and "CUDAExecutionProvider" not in providers:
        raise RuntimeError(f"{label} opened without CUDA. Applied providers: {providers}")
    if gpu_requested:
        session.disable_fallback()


def run_cuda_self_test() -> None:
    """Run a real convolution so cuDNN's lazy plugins are tested during setup."""
    from onnx import TensorProto, helper

    providers = requested_providers(True)
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 16, 16])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 16, 16])
    weights = np.ones((4, 3, 3, 3), dtype=np.float32)
    weight = helper.make_tensor("w", TensorProto.FLOAT, weights.shape, weights.ravel())
    conv = helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 1, 1, 1])
    graph = helper.make_graph([conv], "cuda_self_test", [x_info], [y_info], [weight])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10

    session = onnxruntime.InferenceSession(model.SerializeToString(), providers=providers)
    enforce_session_provider(session, True, "CUDA self-test")
    output = session.run(None, {"x": np.ones((1, 3, 16, 16), dtype=np.float32)})[0]
    if output.shape != (1, 4, 16, 16):
        raise RuntimeError(f"CUDA self-test returned an unexpected output shape: {output.shape}")
