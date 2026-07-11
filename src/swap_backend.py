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


def build_backend(swapper=None):
    """Backend factory keyed by SWAP_BACKEND (env). Only the baseline exists
    today; candidates register here once they pass the benchmark suite."""
    kind = os.environ.get("SWAP_BACKEND", "inswapper").lower()
    if kind == "inswapper":
        if swapper is None:
            import face_engine
            swapper = face_engine.build_swapper()
        return InswapperBackend(swapper)
    raise SystemExit(
        f"Unknown SWAP_BACKEND '{kind}'. Available: inswapper. "
        "Candidate backends must pass benchmark_backends.py before being wired in."
    )
