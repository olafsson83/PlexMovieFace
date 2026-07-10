# PlexMovieFace

Face-swap an entire movie, with support for **multiple different characters**
each mapped to a different replacement face — not just one face swapped onto
everyone. Sibling project to
[PlexAiFaceSwap](https://github.com/olafsson83/PlexAiFaceSwap) (posters); reuses
its proven GPU/CUDA setup and config patterns.

> Personal, non-commercial use only. Keep this on your own machine for your own
> laughs — don't redistribute altered copies of copyrighted movies, and don't
> use anyone's likeness as a source face without their consent.

## How it works

1. **Discover** — samples the movie, detects faces, and clusters them into
   distinct characters using face embeddings (nobody has to label anything
   yet). Exports a few sample crops per character as `character1a.jpg`,
   `character1b.jpg`, `character2a.jpg`, etc. — the **number** identifies a
   discovered character, the **letter** is just multiple sample crops of that
   same character so you can visually confirm it's really one consistent
   person.
2. **You name characters** — no code, just files: look at the crops in
   `characters/`, and for each character number you want swapped, drop a
   single replacement photo into `source_faces/` named to match, e.g.
   `character1.jpg`. Any number without a matching file is left untouched in
   the output (background extras, characters you don't care about, etc.).
   - **If the clustering accidentally splits one real actor into two
     different numbers** (happens sometimes — different scenes, lighting,
     makeup), the fix is simple: just also drop the same photo in as the
     second number, e.g. both `character1.jpg` and `character3.jpg` can be
     the same photo. No merge tool needed.
3. **Swap** — processes the whole movie, matching each detected face to the
   nearest discovered character and swapping in your photo where you've
   provided one, then re-muxes the original audio back in. Full detection
   only runs every few frames (tracking fills in the gap — see Performance
   below); output is written and encoded in resumable chunks. Your original
   movie file is never touched.

## Quick start

**Windows:** double-click `setup.bat`, then `run.bat`.

**Any OS:**

```bash
pip install -r requirements.txt
python setup.py
python run.py
```

`setup.py` will:

- ask for the movie file — either type a path directly, or look it up via
  Plex (searches your movie libraries, resolves the title you pick to its
  actual file path),
- ask how often to sample frames while discovering characters (default every
  2 seconds — frequent enough to catch anyone who's on screen for more than a
  few seconds, without scanning literally every frame),
- ask if you have an NVIDIA GPU and install the matching `onnxruntime` variant
  (with working CUDA/cuDNN, not just the package — see "GPU setup and
  verification" below for the pinned versions and why they matter),
- check ffmpeg is available (falls back to a bundled copy automatically if
  it's not already on your system),
- download the face-swap model (~530MB, one-time),
- write all of that to `.env`.

`run.py` (or `run.bat`) then gives you a menu to run discovery, run the swap,
run both, or **calibrate** (see below).

## Performance — read this before running on a full movie

Full-movie processing is **much** heavier than a poster or a single image:
a 90-minute movie is 130,000+ frames.

- **Always calibrate before a full run.** `run.py` menu option 4 (or
  `python run.py --calibrate` / `python run.py --calibrate 60` for a custom
  sample length) processes a short sample of the movie, measures your actual
  machine's real speed, and projects a full-movie time estimate — so you know
  what you're committing to *before* walking away for hours, not after.
- **Detection doesn't run on every frame.** Full face detection runs every
  `DETECT_EVERY_N_FRAMES` frames (default 5) or immediately after a detected
  scene cut; frames in between track the same faces' position via optical
  flow instead of re-detecting, reusing the classification from the last full
  detection. This is the dominant speedup lever. If you ever suspect tracking
  drift (a swap looking slightly misaligned partway between detections), run
  with `--no-tracking` to force full per-frame detection and compare — slower,
  but a useful correctness check.
- **Output is resumable.** The swap stage writes output in chunks
  (`SEGMENT_FRAME_COUNT` frames per chunk, default 5000) under
  `output/_segments/`. If a run is interrupted (crash, Ctrl+C, sleep), just
  re-run the swap stage — it picks up from the last complete chunk instead of
  starting over. Still worth disabling sleep/display-timeout-triggered sleep
  before a long unattended run.
- **Hardware encode is used automatically when available** (NVENC), freeing
  the CPU for tracking/detection work; falls back to `libx264` otherwise —
  always correct, just slower to encode.

### GPU setup and verification

The Windows setup pins ONNX Runtime 1.26 and cuDNN 9.10 instead of allowing
pip to select an arbitrary future cuDNN 9.x release. A later cuDNN 9.24 was
found to fail with `CUDNN_BACKEND_API_FAILED` on the reference RTX 4060
Laptop the first time a real convolution ran — session construction
succeeded, only actual inference failed, so the failure was easy to miss
until deep into a long run. cuDNN 9.10 doesn't ship the buggy lazily-loaded
tensor-IR engine plugin (`cudnn_engines_tensor_ir64_9.dll`) at all, so this
sidesteps the bug entirely rather than working around it. `gpu_runtime.py`
also registers the NVIDIA DLL directories for the lifetime of the process
(so this venv's isolated CUDA/cuDNN wins over any system-wide install).

Setup runs an actual CUDA convolution, not just a provider-availability
check. Look for `[OK] CUDA/cuDNN convolution test passed on GPU`. When GPU
mode is selected, runtime CPU fallback is disabled deliberately: a CUDA
failure stops with an error instead of quietly turning a movie run into a
20-hour CPU job.

If this project was installed before the pinned GPU stack was added, close
all PlexMovieFace windows, delete the project's `.venv` folder, pull the
latest code, and run `setup.bat` again. System-wide CUDA Toolkit
installations do not need to be removed; the project gives its isolated
runtime priority.

## Notes & troubleshooting

- **A character split into two numbers that are really the same person**:
  expected sometimes, not a bug. Fix: give both numbers the same source photo.
- **Faces too small, angled, or occluded** in a given frame just won't get
  swapped on that frame — the detector needs a reasonably clear, front-ish
  face to work with. Not an error, just a natural limit.
- **A face gets matched to a character but nothing happens**: you didn't
  provide a `characterN.jpg` for that number — that's intentional (unnamed
  faces are left alone), not a bug.
- **Re-running `setup.py`** is always safe — overwrites `.env`, skips
  re-downloading the model if it's already there.
- **Discovery finds too few / too many characters**: adjust `CLUSTER_EPS`
  (higher = more lenient, groups more loosely) or `CLUSTER_MIN_SAMPLES`
  (lower = counts a character from fewer appearances) in `.env`, then re-run
  discovery.
- **Who-is-who decisions are made per face track, not per frame**: a face
  must clear an "enter" bar (instantly if confident, else over several
  confirming detections) before swapping starts, then stays swapped through
  briefly degraded frames on a lower "keep" bar, and faces matching nothing
  well enough are explicitly left alone. The enter/keep bars are derived per
  movie from discovery's own score statistics (`clusters.json` calibration);
  for a project discovered before this existed, run
  `python src/calibrate_clusters.py` once to backfill them, or set
  `IDENTITY_AUTO_CALIBRATION=false` to force the manual `.env` thresholds.
- **Want to redo a run from scratch instead of resuming**: delete
  `output/_segments/` before re-running the swap stage.
