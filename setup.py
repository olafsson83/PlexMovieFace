"""One-time interactive setup wizard.

Walks you through picking a movie file (typed directly, or looked up via
Plex), checking ffmpeg, installing the right onnxruntime for your hardware,
and downloading the face-swap model. Writes everything to .env. Safe to
re-run any time.
"""
import subprocess
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent
ENV_PATH = REPO_ROOT / ".env"

# Same URL used by ComfyUI-ReActor's own installer to fetch this model.
MODEL_URL = "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx"
MODEL_DIR = Path.home() / ".insightface" / "models"
MODEL_PATH = MODEL_DIR / "inswapper_128.onnx"


def ask(prompt, default=None):
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or default or ""


def ask_yes_no(prompt, default=False):
    suffix = "[Y/n]" if default else "[y/N]"
    value = input(f"{prompt} {suffix}: ").strip().lower()
    if not value:
        return default
    return value.startswith("y")


def step(n, title):
    print(f"\n--- Step {n}: {title} ---")


# --- Movie selection -------------------------------------------------------

def fetch_json(url, token, **params):
    try:
        r = requests.get(url, headers={"X-Plex-Token": token, "Accept": "application/json"},
                          params=params, timeout=10)
        r.raise_for_status()
        return r.json()["MediaContainer"]
    except requests.RequestException:
        return None


def choose_movie_via_plex():
    print("In Plex: play anything -> the (...) menu -> Get Info -> View XML.")
    print("Read the server address from your browser's address bar, and the value")
    print("after 'X-Plex-Token=' from that same URL.\n")

    while True:
        plex_url = ask("Plex server URL", "http://127.0.0.1:32400").rstrip("/")
        plex_token = ask("Plex token")

        print("Testing connection...")
        container = fetch_json(f"{plex_url}/library/sections", plex_token)
        if container is not None:
            break
        print("Could not reach Plex with that URL/token.")
        if not ask_yes_no("Try again?", default=True):
            return None

    sections = [s for s in container.get("Directory", []) if s.get("type") == "movie"]
    if not sections:
        print("No movie libraries found on that server.")
        return None

    for i, s in enumerate(sections, 1):
        print(f"  {i}) {s['title']}")
    choice = ask("Which library?", "1")
    try:
        section = sections[int(choice) - 1]
    except (ValueError, IndexError):
        print("Not a valid choice.")
        return None

    while True:
        query = ask("Search for the movie by (part of) its title")
        if not query:
            continue
        items_container = fetch_json(f"{plex_url}/library/sections/{section['key']}/all", plex_token)
        items = items_container.get("Metadata", []) if items_container else []
        matches = [it for it in items if query.lower() in it.get("title", "").lower()]
        if not matches:
            print("No matches, try again.")
            continue
        for i, it in enumerate(matches, 1):
            print(f"  {i}) {it['title']} ({it.get('year', '?')})")
        pick = ask("Which one?", "1")
        try:
            movie = matches[int(pick) - 1]
        except (ValueError, IndexError):
            print("Not a valid choice, try searching again.")
            continue
        break

    detail = fetch_json(f"{plex_url}/library/metadata/{movie['ratingKey']}", plex_token)
    items = detail.get("Metadata", []) if detail else []
    for media in (items[0].get("Media", []) if items else []):
        for part in media.get("Part", []):
            if part.get("file"):
                return Path(part["file"])

    print("Could not determine a file path for that movie from Plex's metadata.")
    return None


def choose_movie():
    if ask_yes_no("Look up the movie via Plex instead of typing a path?", default=True):
        path = choose_movie_via_plex()
        if path and path.exists():
            return path
        if path:
            print(f"Plex reports the file at {path}, but it's not reachable from this machine.")
        print("Falling back to typing the path directly.\n")

    while True:
        movie_path = Path(ask("Full path to the movie file"))
        if movie_path.exists():
            return movie_path
        print(f"Can't find that file: {movie_path}")


# --- Hardware / dependencies (same pattern as the sibling PlexAiFaceSwap repo) --

def ensure_onnxruntime(has_gpu):
    """Installs exactly one onnxruntime variant, idempotently.

    insightface declares plain (CPU) onnxruntime as a dependency, so a base
    `pip install -r requirements.txt` always pulls it in. The GPU variant
    additionally needs its own CUDA/cuDNN runtime DLLs -- plain
    `onnxruntime-gpu` does not include them and silently falls back to CPU.
    The `[cuda,cudnn]` extra pulls in NVIDIA's pip-packaged runtime libraries
    (no CUDA Toolkit installer or NVIDIA developer account needed);
    face_engine.py calls onnxruntime.preload_dlls() at runtime so those
    libraries -- which live in their own separate pip packages, not on the
    normal DLL search path -- actually get found.

    Uninstalling both variants and doing a --force-reinstall of the one we
    want avoids the "module 'onnxruntime' has no attribute 'InferenceSession'"
    breakage that can happen if both variants are ever installed at once
    (e.g. across repeated setup.py runs).
    """
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"],
        check=False,
    )
    package = "onnxruntime-gpu[cuda,cudnn]" if has_gpu else "onnxruntime"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-cache-dir", package],
        check=True,
    )
    # onnxruntime-gpu's own dependency resolution can pull numpy past the <2
    # pin insightface/scikit-image need; re-pin explicitly afterward.
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy<2"], check=True)


def check_ffmpeg():
    import video_io
    try:
        exe = video_io.get_ffmpeg_exe()
        subprocess.run([exe, "-version"], capture_output=True, check=True)
        print(f"  ffmpeg OK: {exe}")
        return True
    except Exception as e:
        print(f"  ffmpeg check failed: {e}")
        return False


def download_model():
    if MODEL_PATH.exists():
        print(f"  Already downloaded: {MODEL_PATH}")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = MODEL_PATH.with_suffix(".onnx.part")

    try:
        with requests.get(MODEL_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done * 100 // total
                        print(f"\r  {pct}% ({done // (1024 * 1024)}MB / {total // (1024 * 1024)}MB)", end="", flush=True)
            print()
    except requests.RequestException as e:
        tmp_path.unlink(missing_ok=True)
        sys.exit(
            f"\nDownload failed ({e}). Your connection likely dropped partway through "
            "this ~530MB file. Re-run `python setup.py` to try again."
        )

    tmp_path.rename(MODEL_PATH)


def write_env(movie_path, discovery_interval, ctx_id):
    lines = [
        f"MOVIE_PATH={movie_path}",
        "DISCOVERY_FRAMES_DIR=discovery_frames",
        "CHARACTERS_DIR=characters",
        "SOURCE_FACES_DIR=source_faces",
        "OUTPUT_DIR=output",
        f"DISCOVERY_INTERVAL_SEC={discovery_interval}",
        "CLUSTER_EPS=0.4",
        "CLUSTER_MIN_SAMPLES=3",
        f"CTX_ID={ctx_id}",
        "",
    ]
    ENV_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    print("PlexMovieFace setup wizard")
    print("You can re-run this any time to change settings.")

    sys.path.insert(0, str(REPO_ROOT / "src"))

    step(1, "Choose the movie file")
    movie_path = choose_movie()

    step(2, "Character discovery sampling")
    interval = ask("Sample a frame every how many seconds while discovering characters?", "2.0")

    step(3, "Hardware")
    has_gpu = ask_yes_no("Do you have an NVIDIA GPU you want to use (much faster)?", default=False)
    ctx_id = 0 if has_gpu else -1

    step(4, "Saving configuration")
    write_env(movie_path, interval, ctx_id)
    print(f"Wrote {ENV_PATH}")

    step(5, "Installing the right packages for your hardware")
    ensure_onnxruntime(has_gpu)

    step(6, "Checking ffmpeg")
    check_ffmpeg()

    step(7, "Downloading the face-swap model (about 530MB, one-time)")
    download_model()

    step(8, "Checking everything is ready")
    import preflight  # imported now so it picks up the .env we just wrote
    ok = preflight.print_report()
    if ok:
        print("\nAll set! Run `python run.py` to start (or double-click run.bat on Windows).")
    else:
        print("\nSome checks above are still failing - fix those and re-run `python setup.py`.")


if __name__ == "__main__":
    main()
