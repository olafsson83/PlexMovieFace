# GPU/CUDA Investigation — Technical Summary

## System

- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM, Ada Lovelace architecture, compute capability SM 8.9)
- **OS**: Windows 11
- **Driver**: started at 577.02 (CUDA UMD 12.9), later updated to 610.74 (CUDA UMD 13.3) — confirmed a genuine GeForce Game Ready driver, not an enterprise/vGPU branch
- **Stack**: `insightface` → `onnxruntime` (CUDA execution provider), no PyTorch/TensorFlow involved anywhere in this pipeline

## Symptom

`onnxruntime`'s CUDA execution provider would either fail to load, or load but fail on first real inference, and silently fall back to CPU. **Correctness was never affected** — onnxruntime's CPU fallback is automatic and safe — only speed (CPU-bound processing instead of GPU-accelerated).

## Baseline: hardware and driver are not the problem

Verified directly, bypassing onnxruntime entirely: loaded `nvcuda.dll` (the driver's own CUDA usermode driver, in `System32`) via raw `ctypes` and called the CUDA Driver API directly.

```python
h = ctypes.WinDLL('nvcuda.dll')
h.cuInit(0)              # returned 0 = CUDA_SUCCESS
h.cuDeviceGetCount(...)  # correctly reported 1 device
```

This conclusively proved the GPU, driver, and low-level CUDA driver API are 100% healthy. Every failure below originates inside onnxruntime's own compiled binaries, not the system.

## Attempt 1: onnxruntime-gpu 1.26.0/1.27.0 + cuDNN 9.24.0.43 (newest, pip-bundled)

This is the "modern" install path: `pip install onnxruntime-gpu[cuda,cudnn]` pulls NVIDIA's pip-packaged CUDA/cuDNN runtime libraries automatically (no system Toolkit required).

**Result**: Session creation succeeds and reports `CUDAExecutionProvider` active — but this is misleading, since cuDNN 9's conv engine JIT-compiles lazily on first actual inference. That first real inference call fails:

```
Could not locate cudnn_engines_tensor_ir64_9.dll. Please make sure it is in your library path!
CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed... CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED
EP_FAIL: Failed to initialize CUDNN Frontend: CUDNN_FE failure 11: CUDNN_BACKEND_API_FAILED
Falling back to ['CPUExecutionProvider'] and retrying.
```

This is cuDNN 9's "tensor_ir" JIT graph-compilation engine specifically — one of several pluggable "engines" cuDNN 9's new graph API can select at runtime.

### Seven fix attempts, all failed identically:

1. **NVIDIA driver update** (577.02 → 610.74) — no change.
2. **Visual C++ Redistributable update** (14.50.35719 → 14.51.36247) — no change.
3. **Clean reinstall of `nvidia-cudnn-cu12`** (full removal, fresh 737MB download) — no change. Ruled out a corrupted pip install.
4. **`cudnn_conv_algo_search=HEURISTIC`** instead of `EXHAUSTIVE` (onnxruntime CUDA provider option, intended to avoid the JIT-heavy exhaustive search path) — no change.
5. **Root-caused the actual DLL-loading mechanism**: `onnxruntime.preload_dlls()` calls `ctypes.CDLL()` per known DLL, which does **not** register a persistent process-wide search directory. cuDNN 9's engine plugins are loaded via cuDNN's own internal `LoadLibrary(name)` calls at first-use time, which only resolve if the directory was registered via `os.add_dll_directory()`. Manually registered every relevant pip package directory this way (`cudnn/bin`, `cublas/bin`, `cuda_nvrtc/bin`, `cuda_runtime/bin`, `cufft/bin`, `curand/bin`) in addition to `preload_dlls()` — confirmed the specific DLL (`cudnn_engines_tensor_ir64_9.dll`) *is* now loadable in isolation via direct `ctypes.WinDLL()`, but the actual onnxruntime inference call still failed identically. This proved the DLL-search-path theory alone was insufficient.
6. **Installed CUDA Toolkit 13.3 system-wide** — no change. Root cause: our pip packages are "cu12"-suffixed (built for CUDA 12.x ABI); the v13.3 Toolkit only provides v13-suffixed files (`nvJitLink_130_0.dll`, `cublasLt64_13.dll`). Windows DLL loading requires **exact filename matches** — no automatic version substitution — so a mismatched-major-version Toolkit can't help regardless of what it contains.
7. **Installed CUDA Toolkit 12.0 (exact version match)**, confirmed `nvJitLink_120_0.dll` present, explicitly added its `bin` directories to the search path — **still the identical failure**. This disproved the version-mismatch hypothesis entirely; a perfectly matched Toolkit made no difference.

Also tested and dismissed: a suggested fix set included clearing a cuDNN JIT compute cache (no such cache existed on this system — nothing to clear), several cuDNN environment variables (`CUDNN_FRONTEND_ATTRIBUTE_DISABLE`, `CUDNN_ENGINE_FALLBACK`, `CUDNN_FALLBACK_TO_LEGACY`) that don't correspond to any documented, real cuDNN control and had no effect, and a claim that driver 610.74 was an enterprise/vGPU branch — directly contradicted by it being a confirmed GeForce Game Ready driver.

**Conclusion at this point**: a specific, narrow, unresolved bug in cuDNN 9.24's tensor_ir JIT engine on this exact GPU/driver — not a configuration problem.

## Attempt 2: downgrade to cuDNN 8.x (no tensor_ir architecture at all)

cuDNN 8.x predates the JIT graph API entirely, so this sidesteps the bug category altogether, at the cost of a much bigger setup lift.

- Installed **CUDA Toolkit 11.8** (component-only silent install — explicitly excluded `Display.Driver` to avoid downgrading the just-updated driver; verified via `nvidia-smi` that the driver stayed at 610.74 throughout).
- Manually downloaded **cuDNN 8.9.7.29** (for CUDA 11.x) from NVIDIA's Developer Program archive — this requires a free NVIDIA account and isn't automatable.
- Copying the extracted files into `Program Files\...\CUDA\v11.8\` failed with a permissions error (no admin rights in this shell) — worked around by leaving the files in a user-writable directory and pointing `os.add_dll_directory()` at it directly instead.
- Installed **`onnxruntime-gpu==1.17.1`** (Feb 2024) — old enough to predate the pip-bundled CUDA/cuDNN extras feature entirely (its metadata declares no `nvidia-*-cu12` dependencies at all).

**Result**: a **different** failure — `[WinError 1114] A dynamic link library (DLL) initialization routine failed`. This is a lower-level DLL-init crash, not a missing-file error — everything resolves, but the provider DLL's own init code fails when it runs. Most likely explanation: this old build lacks compiled kernel support for Ada Lovelace (RTX 40-series, SM 8.9), a GPU generation that may postdate what this release's prebuilt binaries were validated against.

## Attempt 3: middle-ground — onnxruntime-gpu 1.20.1 + an earlier cuDNN 9.x point release

Trying to find a version old enough to predate whatever regression is specific to cuDNN 9.24, but new enough to have proper hardware support.

- `onnxruntime-gpu==1.20.1` — also predates the pip extras feature (`preload_dlls()` doesn't even exist in this version).
- Manually installed `nvidia-cudnn-cu12==9.5.1.17` (an earlier 9.x point release than 9.24.0.43) plus matching cublas/nvrtc/runtime packages, manually registered DLL search directories.
- First failure: generic error 126 (missing dependency). **Root-caused precisely** by parsing `onnxruntime_providers_cuda.dll`'s PE import table directly with `pefile`, revealing its exact static dependencies: `cublasLt64_12.dll`, `cublas64_12.dll`, `cufft64_11.dll`, `cudart64_12.dll`, `onnxruntime_providers_shared.dll`, `cudnn64_9.dll`, plus standard VC runtime/API-set DLLs. This revealed `nvidia-cufft-cu12` had never been installed — installed it, which provided the exact-matching `cufft64_11.dll`.
- Retested: now hit **the same `WinError 1114` DLL-init crash** as the 1.17.1 attempt — not the tensor_ir bug this time, a different failure again.

## Final diagnosis

Two distinct failure modes across three onnxruntime generations, and they don't overlap by version age in a simple way:

| onnxruntime-gpu | CUDA/cuDNN | Failure mode | Likely cause |
|---|---|---|---|
| 1.17.1 | CUDA 11.8 + cuDNN 8.9.7 | `WinError 1114` (DLL init crash) | No Ada Lovelace kernel support |
| 1.20.1 | CUDA 12.x + cuDNN 9.5.1 | `WinError 1114` (DLL init crash) | Same — still no Ada Lovelace kernel support |
| 1.26.0 / 1.27.0 | CUDA 12.x + cuDNN 9.24.0 | `CUDNN_BACKEND_API_FAILED` (JIT engine failure, at runtime) | Specific bug in cuDNN 9.24's tensor_ir engine |

The newest build is actually the **most** hardware-compatible of the three (it's the only one that gets past DLL load and session creation successfully) — its failure is narrower and later-stage. Going to older versions traded a specific, well-isolated bug for a broader, more fundamental incompatibility. **The newest combination (1.26.0, restored as the final configuration) is the correct choice** even though it doesn't achieve GPU acceleration either.

## What was not attempted

- **Building onnxruntime from source** with explicit `-DCMAKE_CUDA_ARCHITECTURES` flags covering SM 8.9 — a multi-hour undertaking (Visual Studio + CUDA + cuDNN dev toolchain) with no guarantee of resolving the same underlying issue.
- **onnxruntime versions between 1.20.1 and 1.26.0** (the 1.21–1.25 range) — might have added Ada Lovelace support while still predating whatever changed in cuDNN 9.24 specifically. Untested due to time already invested.
- **A different inference framework entirely** (e.g., a PyTorch-based face-swap pipeline instead of onnxruntime) — not explored; would be a substantial rebuild, not a configuration change.

## Practical outcome

CPU-fallback was used for all real processing (posters and both movie runs). Correctness was never compromised — the fallback is automatic and safe. The full movie (~146 minutes of footage) took **~20 hours** on CPU; a working GPU path would likely have cut that to a few hours, based on typical CUDA vs. CPU inference throughput ratios for this class of model.
