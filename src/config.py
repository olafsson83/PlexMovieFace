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
# Cosine similarity floor for classifying an in-movie face as a known
# character during the swap stage. Deliberately a separate, higher bar than
# CLUSTER_EPS's clustering distance (1.0 - CLUSTER_EPS): clustering can
# afford to be lenient (a false split just means duplicating a source photo
# under two numbers), but a lenient match threshold here means putting the
# wrong character's face on an unrelated person. Measured on this movie: a
# genuine same-character match against its own cluster centroid scores
# ~0.8+; an unrelated actor's face that happened to score just over a 0.6
# floor got misclassified and swapped -- 0.7 leaves clear margin on both
# sides.
MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.7"))

CTX_ID = int(os.environ.get("CTX_ID", "0"))

# Phase 2: run full face detection every Nth frame; track kps (facial
# landmarks) via optical flow on the frames in between instead of
# re-detecting, to avoid the dominant per-frame detection cost.
DETECT_EVERY_N_FRAMES = int(os.environ.get("DETECT_EVERY_N_FRAMES", "5"))
# Mean grayscale pixel difference between consecutive frames above this
# forces a full re-detection regardless of DETECT_EVERY_N_FRAMES, since a
# scene cut invalidates whatever was being tracked.
SCENE_CUT_THRESHOLD = float(os.environ.get("SCENE_CUT_THRESHOLD", "30.0"))
# Encode in chunks of this many frames so a multi-hour run can resume from
# the last complete chunk instead of restarting from scratch after a crash.
SEGMENT_FRAME_COUNT = int(os.environ.get("SEGMENT_FRAME_COUNT", "5000"))
# Off by default: hardware (NVENC) encode failed intermittently in testing
# on at least one machine. libx264 is slower but proven reliable; only
# enable this after confirming NVENC is stable on your own machine.
USE_NVENC = os.environ.get("USE_NVENC", "false").lower() == "true"

PLEX_URL = os.environ.get("PLEX_URL", "http://127.0.0.1:32400").rstrip("/")
PLEX_TOKEN = os.environ.get("PLEX_TOKEN", "")

CLUSTERS_JSON = CHARACTERS_DIR / "clusters.json"
