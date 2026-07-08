"""ffmpeg subprocess helpers. OpenCV's VideoWriter has no audio support and
unreliable codec availability on Windows, so all actual video encoding and
audio handling goes through ffmpeg directly instead.
"""
import shutil
import subprocess
from pathlib import Path

_ffmpeg_path = None


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


def open_encoder_pipe(out_path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    """Starts an ffmpeg subprocess that reads raw BGR frames from stdin and
    encodes them to out_path as H.264. Write frame.tobytes() (OpenCV BGR
    ndarray) to proc.stdin for each frame, then close stdin and wait().
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        get_ffmpeg_exe(), "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
