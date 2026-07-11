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
# Lower cosine similarity floor applied only to a character that's already
# being tracked at roughly the same on-screen spot (see tracking.hint_for).
# A single real, continuous shot's per-frame score is noisier than a clean
# discovery-crop match and can drift below MATCH_THRESHOLD for a frame or
# two even though it's clearly the same face throughout (measured range on
# a real continuous shot: 0.55-0.75) -- re-clearing the full acquire bar on
# every detection pass made the swap flicker on/off through shots like that.
# This is the old, pre-decoupling MATCH_THRESHOLD value: it was already a
# reasonable "is this still probably the same person" bar, just too lenient
# to be the bar for deciding that in the first place.
MAINTAIN_THRESHOLD = float(os.environ.get("MAINTAIN_THRESHOLD", "0.6"))
# An operator who explicitly sets thresholds in .env outranks auto-
# calibration: calibration derives from discovery-frame statistics, which
# can misrepresent a movie's extremes (measured: a dark-footage project
# with deliberately lowered thresholds lost half its coverage when
# calibration activated over them).
THRESHOLDS_EXPLICIT = ("MATCH_THRESHOLD" in os.environ) or ("MAINTAIN_THRESHOLD" in os.environ)

# Track-based identity decisions (see identity.py). MATCH_THRESHOLD acts as
# the "enter" bar and MAINTAIN_THRESHOLD as the "keep" bar unless discovery
# wrote calibration stats and auto-calibration is on.
#
# Required best-vs-second-best margin before a track can acquire an identity.
# Duplicate clusters mapped to the same source photo don't count as rivals.
MIN_MARGIN = float(os.environ.get("MIN_MARGIN", "0.08"))
# Score above enter+this activates a swap instantly; scores between enter and
# this need CONFIRM_FRAMES consecutive confirming detections first. Two-tier
# acceptance keeps clean shots swapping from their first frame while a single
# borderline false positive (the elderly-extra failure) never confirms.
STRONG_ENTER_MARGIN = float(os.environ.get("STRONG_ENTER_MARGIN", "0.12"))
CONFIRM_FRAMES = int(os.environ.get("CONFIRM_FRAMES", "3"))
# Consecutive below-keep detections before an accepted identity is dropped
# (a clear reclassification to someone else drops it immediately).
REJECT_FRAMES = int(os.environ.get("REJECT_FRAMES", "4"))
# Detection passes a track survives without being matched to any face.
TRACK_MISS_LIMIT = int(os.environ.get("TRACK_MISS_LIMIT", "2"))
# Derive enter/keep thresholds from discovery's calibration stats when
# present; set false to force the .env MATCH/MAINTAIN values.
IDENTITY_AUTO_CALIBRATION = os.environ.get("IDENTITY_AUTO_CALIBRATION", "true").lower() == "true"

# Plate matching (see plate_matching.py): degrade the generated face layer
# to match the plate's optics before compositing. Phase 3: sharpness.
SHARPNESS_MATCHING = os.environ.get("SHARPNESS_MATCHING", "true").lower() == "true"
# Fractional tolerance when matching Laplacian variance -- demanding exact
# equality makes sigma oscillate between frames.
SHARPNESS_TOLERANCE = float(os.environ.get("SHARPNESS_TOLERANCE", "0.12"))
SHARPNESS_MAX_SIGMA = float(os.environ.get("SHARPNESS_MAX_SIGMA", "3.0"))
# EMA weight on the previous smoothed sigma (0 = no smoothing).
SHARPNESS_TEMPORAL_SMOOTHING = float(os.environ.get("SHARPNESS_TEMPORAL_SMOOTHING", "0.7"))

# Phase 4: add the plate's noise/grain texture to the generated face --
# the swap output is missing the plate's high-frequency grain, which is the
# main "pasted-on" tell on grainy or compressed footage.
GRAIN_MATCHING = os.environ.get("GRAIN_MATCHING", "true").lower() == "true"
GRAIN_MAX_SIGMA = float(os.environ.get("GRAIN_MAX_SIGMA", "12.0"))
# Below this applied luma sigma the plate is clean enough to skip grain.
GRAIN_MIN_SIGMA = float(os.environ.get("GRAIN_MIN_SIGMA", "0.5"))
GRAIN_TEMPORAL_SMOOTHING = float(os.environ.get("GRAIN_TEMPORAL_SMOOTHING", "0.7"))
# Ring pixels with gradient energy above this percentile are rejected when
# estimating plate noise (their residuals are structure, not noise).
GRAIN_EDGE_REJECT_PERCENTILE = float(os.environ.get("GRAIN_EDGE_REJECT_PERCENTILE", "75"))

# Phase 5: directional motion blur on the generated face. The swap is
# synthesized sharp even when the plate face smeared during the exposure;
# landmark motion between consecutive frames drives a linear PSF. Applied
# BEFORE sharpness matching so that pass only adds the residual.
MOTION_BLUR_MATCHING = os.environ.get("MOTION_BLUR_MATCHING", "true").lower() == "true"
# Frame-space displacement below this is treated as static (no blur).
MOTION_MIN_DISPLACEMENT_PX = float(os.environ.get("MOTION_MIN_DISPLACEMENT_PX", "0.75"))
# Exposure fraction of the frame interval (180-degree shutter = 0.5).
MOTION_SHUTTER_FRACTION = float(os.environ.get("MOTION_SHUTTER_FRACTION", "0.5"))
# Blur length cap as a fraction of the aligned crop size -- a tracker glitch
# must not paint a 30px streak.
MOTION_MAX_CROP_FRACTION = float(os.environ.get("MOTION_MAX_CROP_FRACTION", "0.08"))
# EMA weight on the previous smoothed motion vector (angle smooths with it).
MOTION_TEMPORAL_SMOOTHING = float(os.environ.get("MOTION_TEMPORAL_SMOOTHING", "0.65"))
# Below this RANSAC inlier fraction the motion estimate is untrusted.
MOTION_MIN_INLIER_RATIO = float(os.environ.get("MOTION_MIN_INLIER_RATIO", "0.7"))
# Rotation (degrees/frame) above which the single-kernel model is damped;
# a face-wide linear PSF can't represent rotational smear.
MOTION_ROTATION_LIMIT_DEG = float(os.environ.get("MOTION_ROTATION_LIMIT_DEG", "3.0"))

# v2 milestone 3: staged low-light detection retry on an analysis-only
# enhanced copy of dark frames (gamma lift + CLAHE, larger detector canvas).
# The plate that gets swapped/composited is never altered.
ADAPTIVE_DETECTION = os.environ.get("ADAPTIVE_DETECTION", "true").lower() == "true"
# Mean frame luma below this counts as dark (0-255).
ADAPTIVE_DARK_LUMA = float(os.environ.get("ADAPTIVE_DARK_LUMA", "60"))
# Detector canvas for the retry pass (base pass stays at 640).
ADAPTIVE_RETRY_DET_SIZE = int(os.environ.get("ADAPTIVE_RETRY_DET_SIZE", "960"))
# Cap on the adaptive gamma lift (1.0 = no lift).
ADAPTIVE_GAMMA_MAX = float(os.environ.get("ADAPTIVE_GAMMA_MAX", "2.2"))

# v2 milestone 6: refuse to swap frames the backend can't render well
# instead of quietly producing a bad warped profile. inswapper_128's
# five-point alignment degrades hard past strong yaw; an original face for
# a few frames beats a broken swap.
POSE_GATE = os.environ.get("POSE_GATE", "true").lower() == "true"
# |yaw| in degrees beyond which an observation is unrenderable.
MAX_ABS_YAW = float(os.environ.get("MAX_ABS_YAW", "65"))
# Hysteresis: a blocked track becomes renderable again below
# MAX_ABS_YAW - POSE_EXIT_MARGIN, so yaw oscillating at the limit doesn't
# flicker the swap on and off.
POSE_EXIT_MARGIN = float(os.environ.get("POSE_EXIT_MARGIN", "8"))

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
