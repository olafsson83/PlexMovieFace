# Plate-matching & identity roadmap

## Round-4 external review (rendered-output quality, 2026-07-12)

Directive accepted: do not optimise total swap count further until rendered
output is measured -- the plan-level suite cannot see scrambled geometry,
weak identity transfer, backend flicker or malformed composites. Acceptance
now means the RENDERED OUTPUT improved (evaluate_render.py --compare), not
that the plan gained rows. Work items, gated per landing on both suites:

1. DONE -- rendered-output harness (src/evaluate_render.py +
   tests/fixtures/render_manifest.json): probe windows on the user's exact
   pain frames render through the real backend/plate-matching path and
   measure identity gain over the untouched plate, alignment residual,
   temporal instability, per-track routing/transitions, plus a crop strip
   per window for human approval. First baseline immediately quantified
   the SimSwap problem: identity gain 0.01 at yaw 81 vs 0.83 frontal.
2. DONE -- plan format v3 (analysis_store): every row carries actual
   pitch/yaw/roll, det_score, identity score, margin, provenance
   (detector/flow/backfill/bridge) and confidence. Analysis no longer
   discards identity-certain extreme-pose observations (it cannot know the
   backend); the RENDER pass gates them per selected backend capability
   (swap_backend.RenderPoseGate), with an explicit MAX_ABS_YAW .env value
   still overriding as operator intent.
3. DONE -- hybrid routing on actual buffalo_l pose from the plan row;
   yaw_proxy demoted to low-confidence fallback for rows without pose
   evidence; transitions logged. Hysteresis is asymmetric min-hold (enter
   the safe arm immediately, return after a 3-swap hold): the original
   margin-band exit (57 degrees) measurably kept 63-degree frames on the
   weak SimSwap arm, forfeiting 0.8+ identity gain (t23 probe); min-hold
   recovers them (window gain 0.212 -> 0.249) while still bounding
   route flicker. Render-probe compare vs pre-v3 baseline: PASS on all
   windows; t36 min out_sim 0.073 -> 0.557 (a proxy false-fire eliminated
   by real pose).
4. DONE -- SimSwap alignment validation before inference
   (swap_backend.validate_alignment): RANSAC reprojection threshold now
   scales with the crop (facefusion's hardcoded 100 rejects nothing at
   512); inlier count, residual normalised by crop size, scale, rotation,
   reflection (impossible under partial affine, recorded for audit) and
   frame coverage (BORDER_REPLICATE smear guard) are measured per swap;
   invalid transforms are withheld (backend returns None; the plate stays
   untouched and PlateMatcher counts it). Residual ceiling 0.18 grounded
   in measured genuine faces: frontal 0.03-0.06, genuine extreme 0.13.
5. DONE -- bidirectional-tracking bridge (bridging.py, replacing
   identity.bridge_swap_rows): long gaps are re-tracked forward from the
   previous anchor and backward from the next through the LITERAL same
   per-step gates as the live tracker (tracking.propagate_kps, shared
   code); a frame is emitted only when both trajectories survive and
   agree within a face-scaled tolerance. Pure interpolation only for
   gaps <= 2 frames. Field result vindicates the review: on the fixture
   clips ZERO long-gap frames survived verification (114 unverified + 10
   disagreeing on diehard alone) -- the old interpolation had been
   restoring exactly the frames the scramble protection withheld.
   Suite after: diehard 573 -> 552, brokeback 713 -> 700, hp 1021 held,
   0 wrong-person; render-probe compare PASS on every window.
6. TODO -- redefine `proven` on strong hits and margins; reacquisition
   after long gaps requires enter-level evidence + margin; no
   position-only uncontested reacquisition at keep level.
7. TODO -- ROI detections filtered to expected centre/scale; recognition
   embeddings computed on the original plate after enhanced detection.

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
4. PARTIAL -- persistent tracks, v1 landed:
   - Hungarian assignment over a combined position + embedding + scale cost
     (embedding term keeps crossing actors on their own tracks -- unit-tested
     with the crossing scenario greedy association gets wrong). Uncontested
     one-face-one-track pairs skip the embedding term: embeddings resolve
     competition, and a lone dark face's junk embedding must not break the
     continuity position alone supports (measured -4% dark-fixture coverage
     without the carve-out).
   - Retroactive backfill (first real future-evidence use the two-pass
     architecture enables): the analysis pass records every track
     observation; tracks that eventually confirm extend their swap range
     backward over pre-confirmation observations that already cleared the
     KEEP bar, contiguously, with landmark interpolation across detection
     gaps. Never-accepted tracks (the elderly-extra shape) contribute
     nothing by construction.
   - Suite after: hp 967 -> 985, diehard 547 -> 599, brokeback unchanged;
     0 wrong-person swaps throughout.
   - Still open for later: shot_id in the state keys, backward OPTICAL
     tracking (only decision backfill is implemented -- no new detections
     are created backward), tracklet merging across occlusion.
5. DONE (v1) -- prototype banks per character: centroid anchor + diverse
   members via farthest-point sampling in embedding space (pose/brightness
   measurement arrives in milestone 6), scored max-over-bank everywhere
   (identity, calibration stats, diagnostics). Two lessons the regression
   harness caught before they shipped:
   - A bank of scattered member embeddings WITHOUT the centroid scores
     typical faces lower than the old mean did (coverage collapsed on two
     fixtures); the centroid must stay in the bank as the floor.
   - Auto-calibration must not outrank an explicitly set .env threshold:
     calibration derives from discovery-frame statistics and misjudged the
     dark-footage project whose operator had deliberately lowered
     thresholds (coverage halved until precedence was fixed:
     explicit env > calibration > env defaults).
   Suite after: diehard 599 -> 613, hp 985 (held), brokeback 703 (-3,
   within noise); 0 wrong-person swaps. Known artifact: clusters smaller
   than the bank size self-score genuine_p10 = 1.0 (harmless -- thresholds
   anchor on impostor tails). Still open: two-tier discovery (strict seeds
   + track-supported expansion faces).
6. DONE (v1) -- pose gate / unrenderable frames. buffalo_l's 3D landmark
   model already computes head pose per detection (free); observations past
   MAX_ABS_YAW (default 65 deg) are marked unrenderable: the track keeps
   its identity and evidence, but no swap row is emitted -- a brief
   original face beats a broken five-point profile warp. Hysteresis
   (re-enable below limit - 8 deg) prevents flicker at the boundary;
   backfill refuses to fill across unrenderable observations. Measured
   trade (deliberate coverage reduction, awaiting operator judgment on
   samples): diehard 613 -> 510 frames (extreme-yaw vent crawling), hp
   985 -> 967, brokeback unchanged; suite PASS, 0 wrong-person. POSE_GATE
   and MAX_ABS_YAW are .env-tunable if samples argue for a looser limit.
   Still open: landmark confidence, occlusion estimates, per-backend
   support ranges (milestone 7).
7. DONE -- first candidate accepted as OPT-IN: inswapper_gfpgan chains
   GFPGAN v1.4 (ONNX, CUDA, no torch) at 512px after inswapper. Benchmark
   verdict on the brokeback fixture (fair sharpness metric: crops compared
   at a common 128px -- Laplacian variance is resolution-dependent):
   blend 0.8 = +53% detail (148 -> 226.5) at -0.049 identity similarity
   and +59% latency; blend 0.5 = +12% detail at -0.018 identity. Visual
   side-by-side confirms clearly sharper eyes/brows/skin. Stays opt-in
   (SWAP_BACKEND=inswapper_gfpgan) because of the measured identity cost;
   the alignment (and thus the pose gate range) is unchanged -- it does
   NOT fix profiles. Original milestone notes:
   swap backend interface (swap_backend.py):
   prepare_source / swap / capabilities, factory keyed by SWAP_BACKEND.
   PlateMatcher consumes the interface (raw INSwapper still accepted and
   auto-wrapped). benchmark_backends.py measures a backend on real footage:
   identity survival (composited-face embedding vs source), crop sharpness,
   swap+composite latency. Baseline recorded on the brokeback fixture,
   47 swaps: identity similarity 0.853 mean / 0.826 p10, crop lap-var 148,
   135 ms/swap. Candidates (higher-res synthesis or an ONNX face-enhancer
   chained after inswapper) must beat this on the same numbers -- no
   backend is accepted on screenshots. Smoke-verified: two-pass render
   through the interface reproduces the plan exactly (703 swaps).
8. Review/correction interface (track timelines, merge/split, approvals).

---

## Hit-rate plan (2026-07-11)

Loss ledger and fixes, in order of return on effort:
1. DONE -- anchor-bridged gap interpolation (identity.bridge_swap_rows):
   interior track gaps between two detector-verified swapped observations
   are filled by landmark interpolation -- far safer than the corrupt LK
   propagation the quality gate withholds. Contradicting evidence inside a
   gap (pose block, identity dropout) splits the anchor pair and blocks
   the bridge by construction; cuts can't bridge (new track_id per cut).
   Suite: diehard 432 -> 485, brokeback 651 -> 713 (best ever, reclaiming
   the gate's false positives), hp 953 -> 955; 0 wrong-person.
2. True-recall measurement: annotate one fixture with "a human can see the
   face here" ground truth so hit rate gets a meaningful denominator.
   (Partially served by the 5-frame diagnostic of 2026-07-11: the user's
   pain frames measured as 2x pose-gated at CERTAIN identity 0.74-0.85,
   1x luma-12 undetectable, 2x junk/borderline embeddings.)
3. DONE -- ROI re-detection (crop + enhance + 2x upscale around missing
   track regions, coordinates mapped back). Honest yield: 228 retries on
   the diehard fixture recovered 9 detections -- bounded cost, small gain.
4. DONE (v1) -- track-level identity robustness: proven tracks (3+ strong
   observations) survive 6 missed detection passes instead of 2, weak
   re-detections resume on the KEEP bar instead of failing cold
   acquisition, and pending confirmations AGE through sub-enter dips
   instead of resetting (only a qualifying rival resets them). Safety
   counterweight: a reacquired track must re-clear KEEP before the
   below-keep ride-out applies (someone else may have taken the spot).
   Suite: hp 985 -> 1002 (best ever), brokeback 713 (held), diehard
   485 -> 475 -- the dip is the reacquisition-proof guard withholding
   ride-out frames that previously swapped on stale credit; 0 wrong-person
   throughout.
5. DONE (2026-07-12) -- pose-capable backend (SimSwap 512, ONNX via
   facefusion's conversion + crossface embedding converter; no torch
   needed) and the hybrid router that makes it usable. Measured:
   - SimSwap renders cleanly at 80+ degree yaw (frames 527/551, the
     0:22/0:23 pain frames, now swap without warp breakage) BUT its
     identity transfer is half-strength: similarity 0.53 vs inswapper's
     0.85 (sharpness 114 vs 148, latency 208ms vs 134ms). Wholesale
     replacement fails the identity bar; as the only option on a
     near-profile frame it beats the untouched original face.
   - SWAP_BACKEND=hybrid therefore routes per face: HYBRID_PRIMARY where
     five-point alignment holds, SimSwap in the extreme band. Pose at
     render time is estimated from the plan's own 5 landmarks (nose vs
     mouth-midpoint offset along the roll-corrected eye axis, over
     inter-eye distance) -- calibrated against buffalo_l 3D yaw on 925
     fixture faces: threshold 0.85 gives 97% recall at |yaw|>65 with a
     3.5% false-fire rate on frontal faces. No plan-format change; works
     on flow-tracked/backfilled/bridged rows automatically.
   - Die Hard proof run (analysis at MAX_ABS_YAW=85): pose gate withheld
     20 detections instead of 88; 548 swaps = 448 inswapper + 100
     SimSwap-routed extreme poses.

## Round-3 external review (tracking safety, integrated 2026-07-11)

Three real tracking defects found by a second external review, integrated
after code review + our own tests (the zip shipped none) + suite gating:
1. LK "success" was trusted blindly; optical-flow propagation now requires
   forward/backward round-trip consistency AND one plausible partial-affine
   motion explaining all five landmarks (scale/rotation/residual limits) --
   scrambled-face flashes in dark/occluded/fast footage came from exactly
   these corrupt propagations. Feature-flagged (TRACK_FLOW_QUALITY_GATE).
2. Tracked landmarks could survive a scene cut when no new detection
   claimed their old position; start_from_detection now clears
   unconditionally on cuts.
3. Carry-forward suppression was keyed by character_number, so two
   simultaneous tracks of the same character suppressed each other; now
   keyed by track_id.
Artifact format bumped to v2 with an is_compatible() check -- old plans
regenerate automatically instead of silently rendering stale decisions.
Suite after (safety trade, all expectations still met, 0 wrong-person):
hp 967 -> 953, diehard 510 -> 432 (72/340 propagations withheld:
19 status, 30 fb, 23 geometry; 3 stale tracks cleared at cuts),
brokeback 703 -> 651.

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
