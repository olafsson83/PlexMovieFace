# Plate-matching & identity roadmap

## v2 milestones (from external review, 2026-07-11)

External review verdict, accepted: phases below fixed identity correctness
and finish quality, but the remaining failures (dark faces, strong profiles)
live UPSTREAM -- single-pass detection on the unmodified dark frame at fixed
640px, one centroid per character, five-point alignment, two disjoint
trackers with no shared track id, forward-only streaming that can't use
future evidence, and inswapper_128's synthesis ceiling. Known bug from the
review: plate-matching temporal state is keyed by character_number, not
(shot_id, track_id) -- same-identity faces can cross-contaminate.

Order of work (each gated on the regression suite not regressing, wrong-
person swaps weighted as hard failures):

1. DONE -- real-clip regression harness: `python src/evaluate.py` runs the
   fixture suite in tests/fixtures/manifest.json (clips gitignored, local).
   Baseline 2026-07-11: all fixtures PASS, 0 wrong-person swaps; coverage
   hp 967/4500 frames, diehard 479/3596, brokeback 706/720 (0.981).
2. DONE -- analyse-then-render two-pass architecture. analyze_movie.py
   writes a versioned swap-plan artifact (frame, track_id, character,
   landmarks -> <movie>_analysis.npz via analysis_store.py); render_movie.py
   consumes it and never re-runs identity logic. swap_movie.py orchestrates
   (--analyze-only / --render-only / --reanalyze; artifact reused when
   present). The identity manager now mints stable track_ids, threaded
   through TrackedFace into plate matching, whose temporal state is keyed
   by track_id instead of character_number -- the review's state-keying bug,
   fixed early since the plumbing was open (rest of milestone "correct
   temporal state ownership" lands with real shot ids in milestone 4).
   Verified: two-pass on the brokeback fixture produces exactly the
   single-pass baseline's 706 swap decisions; regression suite PASS,
   0 wrong-person swaps.
3. DONE -- adaptive low-light detection (adaptive_detection.py). Staged
   retry: base detection on the original frame; when the frame is dark AND
   (nothing found, or a live track's region came back empty), retry on an
   analysis-only enhanced copy (adaptive gamma toward mid-gray + CLAHE on
   luma) at a 960px detector canvas, merged by IoU with base detections
   winning. The plate is never altered; embeddings/landmarks for retry
   detections read the enhanced image (analysis only). Per-ROI upscale
   retries deferred. Measured on the diehard fixture: 1441 retries,
   119 extra detections, frames swapped 479 -> 547 (+14%) with the no-swap
   windows still clean; full suite PASS, 0 wrong-person swaps; HP and
   brokeback unchanged (bright footage, retry never fires). Honest ceiling:
   the deepest silhouette frames (early vent crawl) recover nothing --
   there is no face signal to enhance.
4. Persistent anonymous tracks with one authoritative track_id shared by
   identity, motion and plate-matching state (fixes the state-keying bug);
   Hungarian association on position+embedding+scale; forward-backward.
5. Pose-diverse prototype banks per character (replace single centroids);
   two-tier discovery (strict seeds + track-supported expansion faces).
6. Pose/landmark confidence + explicit "unrenderable frame" decisions.
7. Swap backend interface; benchmark higher-res backends against the
   fixture suite (inswapper_128 stays as the fast baseline).
8. Review/correction interface (track timelines, merge/split, approvals).

---

## v1 history (completed phases)

Goal: make swapped faces (a) never land on the wrong person, and (b) match the
plate's optical/sensor statistics (motion blur, defocus, grain) so they stop
reading as "pasted on". Distilled from two expert reviews; where they
conflicted, the second (identity-first) won because our worst real failures
were identity failures (the elderly-extra bug), not optics.

## Build order

1. **Identity: calibration, track-based hysteresis, rejection** (this phase)
   - Discovery stores per-cluster calibration stats (genuine score
     percentiles, per-pair impostor percentiles) in clusters.json.
     These are *calibration evidence, not ground truth* — cluster membership
     was itself selected by a similarity decision, so the distributions carry
     selection bias. Used with margins, not trusted blindly.
   - Swap-time identity decisions belong to the face *track*, not the frame:
     enter/keep thresholds (temporal hysteresis), best-vs-second-group margin,
     confirmation frames before activating, rejection streak before dropping,
     and an explicit unknown/no-swap outcome (never force-choose nearest).
   - Duplicate clusters of one actor (same source photo under several
     numbers) are grouped by source-photo content hash; margin is measured
     against the best *other* group so duplicates don't count as impostors.
   - Divergence from the review: strict N-frame confirmation for every
     activation would leave ~0.5s unswapped at each shot start. We use
     two-tier acceptance instead — instant accept above a strong-confidence
     bar, confirmation frames only for borderline scores (the elderly-extra
     regime: 0.638, one frame, would never have confirmed).

2. **Sharpness/defocus matching** (done -- see finding below)
   - Operate on the generated face layer *before* compositing
     (`swapper.get(..., paste_back=False)` returns the aligned fake crop +
     transform; we own the paste-back, byte-identical parity verified).
     Blurring the finished composite would halo original plate pixels.
   - Measure the *unswapped* plate crop (interior mask; luma; denoise first;
     Laplacian variance + gradient energy, not one metric). Both crops share
     the aligned 128px space, so sigma is measured and applied consistently.
   - Bounded sigma search with tolerance (~12%) + temporal smoothing per
     character with position-jump reset. Blur only when the fake is sharper
     on BOTH metrics (identity-texture differences must not trigger blur).
   - **Measured finding**: inswapper_128's output sharpness tracks its input
     almost exactly (fake/plate Laplacian ratio 0.98 on a sharp plate, 1.04
     on an artificially defocused one) -- the model synthesizes conditioned
     on the aligned plate crop and inherits its blur. So for THIS model the
     matcher is a safety net that rarely engages (18/967 swaps on the VCD
     Harry Potter clip, sigma <= 0.41), not a big visual win. The
     "plastic/pasted" impression is expected to come mostly from missing
     grain -- which the sharpness metric deliberately ignores (it denoises
     before measuring). Phase 3 (grain) is where the visible gain should be.

3. **Grain / compression-texture matching** (done)
   - Applied in FRAME space on the warped generated layer (crop space would
     rescale the noise spectrum by the warp), between warp and composite.
   - Plate noise measured from a ring just outside the face; the generated
     layer's own noise measured from the face interior in the same space;
     only the variance deficit is added (the swap inherits some plate noise
     through conditioning -- adding the full amount would double-grain).
   - Luma/chroma (YCrCb), robust MAD on high-pass residuals with
     edge/saturation rejection; spatially correlated noise rescaled so the
     same estimator reads the target on the synthetic field (keeps all
     amplitudes in consistent estimator units, no analytic filter constants);
     deterministic per-swap seeds; per-character EMA with position reset.
   - **Measured**: on the VCD Harry Potter clip, 773/967 swaps grained (mean
     luma sigma 2.13) -- engaging exactly where sharpness matching found
     nothing, confirming grain was the dominant texture gap. Post-encode the
     face/surroundings noise ratio moves from 1.01 to 1.11; the still-frame
     difference is subtle (x264 compresses it), the win is expected in
     motion where static-smooth skin against living grain is the tell.
   - Gotcha for posterity: cv2.cvtColor on float32 assumes [0,1] range with
     0.5 chroma offset -- YCrCb round-trips must go through uint8.

4. **Motion blur** (done)
   - Landmark-based partial-affine motion (RANSAC over the 5 tracked kps);
     translation-dominant cases get a linear PSF built in crop space (the
     alignment's linear part transforms the frame-space motion vector, so
     kernel and application share a space); rotation-dominant or
     high-residual cases are damped to 25%, low inlier ratio disables.
   - Blur length = displacement x shutter fraction (0.5 default -- the
     inter-frame displacement is NOT the exposure displacement), capped at
     8% of crop size, zero below 0.75px displacement; the motion VECTOR is
     EMA-smoothed so the angle smooths with it and a stop decays rather
     than snaps.
   - Applied before the sharpness matcher measures the crop, so that pass
     adds only the residual isotropic blur (no double-count of the plate's
     motion smear).
   - Measured on the Die Hard vent clip: 76/479 swaps blurred (mean 2.75px,
     max 5.88px, capped); static close-ups untouched.

5. **Quality dashboard**
   - Per-frame CSV (scores, margins, decisions, sigma, motion, noise) +
     side-by-side diagnostic crops (plate / raw swap / each matching stage /
     composite). Watch for: decision flips, sigma pumping, motion-angle
     flips, noise jumps at background changes.

## Module layout (target)

```
src/
  identity.py        # calibration, source-photo grouping, TrackIdentityManager
  plate_matching/    # phases 2-4: analysis + apply, premultiplied helpers
  quality_metrics.py # phase 5
```

## Test fixtures

Known regression cases from real runs (clips live outside the repo):
- Harry Potter ron-scene clip: elderly extra scored 0.638 vs Harry at frame
  ~328 (must stay unswapped); flicker window frames ~3480-3630 (must stay
  continuously swapped); genuine Harry at ~t=96-98s (must swap).
- Die Hard vent clip: dark low-score stretch needs enter=0.45/keep=0.25 to
  cover; Karl/Tony/Hans must never swap.
- Brokeback camp clip: near-total coverage two-hander; both identities must
  stay on the correct actor throughout.
```
