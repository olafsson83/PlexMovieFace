"""ffmpeg subprocess helpers. OpenCV's VideoWriter has no audio support and
unreliable codec availability on Windows, so all actual video encoding and
audio handling goes through ffmpeg directly instead.
"""
import shutil
import subprocess
import threading
from pathlib import Path

_ffmpeg_path = None
_has_nvenc = None


def get_ffmpeg_exe() -> str:
    """Resolves an ffmpeg executable: prefer one already on PATH, otherwise
    fall back to the static build bundled by the imageio-ffmpeg pip package
    (no manual system-wide ffmpeg install required).
    """
    global _ffmpeg_path
    if _ffmpeg_path:
        return _ffmpeg_path

    on_path = shutil.which("ffmpeg")
    if on_path:
        _ffmpeg_path = on_path
        return _ffmpeg_path

    import imageio_ffmpeg
    _ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    return _ffmpeg_path


def extract_frames_at_interval(movie_path: Path, out_dir: Path, interval_sec: float):
    """Samples frames from movie_path every interval_sec seconds into out_dir
    as frame_000001.jpg, frame_000002.jpg, ... Uses ffmpeg's own decoder
    rather than OpenCV seeking, which is unreliable across codecs/containers
    with B-frames or variable frame rate.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        get_ffmpeg_exe(), "-y", "-i", str(movie_path),
        "-vf", f"fps=1/{interval_sec}",
        "-q:v", "2",
        str(out_dir / "frame_%06d.jpg"),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def extract_audio(movie_path: Path, out_path: Path):
    """Stream-copies the audio track out of movie_path, no re-encoding.
    Uses a Matroska (.mka) container regardless of the source codec (AAC,
    AC3, DTS, ...) since it can wrap virtually any audio codec via copy.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        get_ffmpeg_exe(), "-y", "-i", str(movie_path),
        "-vn", "-acodec", "copy",
        str(out_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def has_nvenc() -> bool:
    """Checks once whether the resolved ffmpeg build has NVENC (hardware
    H.264 encode) compiled in. Not guaranteed even with an NVIDIA GPU
    present -- some static ffmpeg builds (including the one imageio-ffmpeg
    bundles, depending on version) don't include it.
    """
    global _has_nvenc
    if _has_nvenc is not None:
        return _has_nvenc
    try:
        result = subprocess.run([get_ffmpeg_exe(), "-hide_banner", "-encoders"],
                                 capture_output=True, text=True, timeout=15)
        _has_nvenc = "h264_nvenc" in result.stdout
    except Exception:
        _has_nvenc = False
    return _has_nvenc


def _drain_stderr(proc, tail):
    """Continuously reads a subprocess's stderr in the background so its
    pipe buffer never fills and blocks the process, while keeping the last
    lines around for diagnostics if something goes wrong.
    """
    for raw_line in proc.stderr:
        tail.append(raw_line.decode(errors="replace").rstrip())
        del tail[:-50]
    proc.stderr.close()


def open_encoder_pipe(out_path: Path, width: int, height: int, fps: float, use_nvenc: bool = False) -> subprocess.Popen:
    """Starts an ffmpeg subprocess that reads raw BGR frames from stdin and
    encodes them to out_path as H.264. Write frame.tobytes() (OpenCV BGR
    ndarray) to proc.stdin for each frame, then close stdin and wait().

    Defaults to libx264 (always correct, proven reliable) rather than
    auto-detecting and using NVENC, since hardware encode support varies by
    machine. Pass use_nvenc=True to opt in after confirming has_nvenc() and
    that it's stable on your setup.

    Explicitly forces the mp4 muxer via -f rather than relying on ffmpeg's
    filename-extension sniffing: out_path may be a temp name like
    "segment_00001.mp4.part" (for atomic rename-on-completion), whose real
    extension as far as ffmpeg is concerned is ".part", which it can't map
    to a container format on its own.

    The returned process has a `.stderr_tail` list (most recent ~50 lines)
    for diagnosing an encoder failure -- its own stderr is drained on a
    background thread rather than left to fill the pipe buffer and stall.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if use_nvenc and has_nvenc():
        codec_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20"]
    else:
        codec_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
    cmd = [
        get_ffmpeg_exe(), "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",
        *codec_args,
        "-pix_fmt", "yuv420p",
        "-f", "mp4",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    proc.stderr_tail = []
    threading.Thread(target=_drain_stderr, args=(proc, proc.stderr_tail), daemon=True).start()
    return proc


def mux(video_only_path: Path, audio_path: Path, final_out_path: Path):
    """Combines a video-only file and an audio file into final_out_path,
    stream-copying both (no re-encoding)."""
    final_out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        get_ffmpeg_exe(), "-y",
        "-i", str(video_only_path), "-i", str(audio_path),
        "-c", "copy", "-map", "0:v:0", "-map", "1:a:0",
        str(final_out_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def concat_segments(segment_paths, out_path: Path):
    """Losslessly joins a list of same-codec segment files (in order) into
    out_path via ffmpeg's concat demuxer (stream copy, no re-encoding).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = out_path.with_suffix(".concat.txt")
    lines = [f"file '{p.resolve()}'" for p in segment_paths]
    list_path.write_text("\n".join(lines), encoding="utf-8")
    try:
        cmd = [
            get_ffmpeg_exe(), "-y",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-c", "copy",
            str(out_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
    finally:
        list_path.unlink(missing_ok=True)
