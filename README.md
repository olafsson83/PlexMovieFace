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
   provided one, then re-muxes the original audio back in. Output is a new
   file — your original movie is never touched.

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
  (with working CUDA/cuDNN, not just the package — see the sibling repo's
  README if you want the details on why that distinction matters),
- check ffmpeg is available (falls back to a bundled copy automatically if
  it's not already on your system),
- download the face-swap model (~530MB, one-time),
- write all of that to `.env`.

`run.py` (or `run.bat`) then gives you a menu to run discovery, run the swap,
or both.

## Performance — read this before running on a full movie

Full-movie processing is **much** heavier than a poster or a single image:
a 90-minute movie is 130,000+ frames. This first version processes every
frame with no shortcuts (no frame-skipping/tracking yet — that's planned as a
follow-up once this simpler version is proven correct), so:

- **Always test on a short clip first.** Cut one with ffmpeg:
  `ffmpeg -i movie.mp4 -t 180 -c copy test_clip.mp4` (~3 minutes), point
  `MOVIE_PATH` at that, and run the full discover → name → swap workflow on
  it. Note how long the swap stage actually takes for those 3 minutes and
  multiply up to estimate a full movie — the real number depends heavily on
  your CPU/GPU and how many named characters are on screen at once.
- **There's no crash-resumability yet.** If a multi-hour run on a full movie
  gets interrupted (crash, sleep, Ctrl+C), it starts over from scratch. Fine
  for a short clip; a real risk for a full movie until a future version adds
  resumable chunked processing. Disable sleep/display-timeout-triggered sleep
  on your machine before a long unattended run.

## Notes & troubleshooting

- **A character split into two numbers that are really the same person**:
  expected sometimes, not a bug. Fix: give both numbers the same source photo.
- **Faces too small, angled, or occluded** in a given frame just won't get
  swapped on that frame — the detector needs a reasonably clear, front-ish
  face to work with. Not an error, just a natural limit.
- **A face gets matched to a character but nothing happens**: you didn't
  provide a `characterN.jpg` for that number — that's intentional (unnamed
  faces are left alone), not a bug. The swap stage's summary at the end tells
  you how many faces were "recognized but no source photo provided" so you
  can tell whether that's what's happening.
- **Re-running `setup.py`** is always safe — overwrites `.env`, skips
  re-downloading the model if it's already there.
- **Discovery finds too few / too many characters**: adjust `CLUSTER_EPS`
  (higher = more lenient, groups more loosely) or `CLUSTER_MIN_SAMPLES`
  (lower = counts a character from fewer appearances) in `.env`, then re-run
  discovery.
