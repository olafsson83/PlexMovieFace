import sys
import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"

try:
    load_dotenv(ENV_PATH)
except UnicodeDecodeError:
    sys.exit(
        f"{ENV_PATH} isn't valid UTF-8 text (often caused by hand-editing it in a "
        "text editor that saves as ANSI/Windows-1252 instead of UTF-8 - e.g. Notepad's "
        "default 'Save As' encoding). Re-run `python setup.py` to regenerate it cleanly, "
        "or re-save the file with UTF-8 encoding."
    )

MOVIE_PATH = Path(os.environ["MOVIE_PATH"]) if os.environ.get("MOVIE_PATH") else None

DISCOVERY_FRAMES_DIR = REPO_ROOT / os.environ.get("DISCOVERY_FRAMES_DIR", "discovery_frames")
CHARACTERS_DIR = REPO_ROOT / os.environ.get("CHARACTERS_DIR", "characters")
SOURCE_FACES_DIR = REPO_ROOT / os.environ.get("SOURCE_FACES_DIR", "source_faces")
OUTPUT_DIR = REPO_ROOT / os.environ.get("OUTPUT_DIR", "output")

DISCOVERY_INTERVAL_SEC = float(os.environ.get("DISCOVERY_INTERVAL_SEC", "2.0"))
CLUSTER_EPS = float(os.environ.get("CLUSTER_EPS", "0.4"))
CLUSTER_MIN_SAMPLES = int(os.environ.get("CLUSTER_MIN_SAMPLES", "3"))

CTX_ID = int(os.environ.get("CTX_ID", "0"))

PLEX_URL = os.environ.get("PLEX_URL", "http://127.0.0.1:32400").rstrip("/")
PLEX_TOKEN = os.environ.get("PLEX_TOKEN", "")

CLUSTERS_JSON = CHARACTERS_DIR / "clusters.json"
