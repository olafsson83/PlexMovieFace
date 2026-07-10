# Plate-matching & identity roadmap

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

2. **Sharpness/defocus matching**
   - Operate on the generated face layer *before* compositing
     (`swapper.get(..., paste_back=False)` returns the aligned fake crop +
     transform; we own the paste-back), constrained by the warped swap alpha.
     Blurring the finished composite would halo original plate pixels.
   - Measure the *unswapped* plate crop (interior mask, eroded alpha; luma;
     denoise first; Laplacian variance + gradient energy, not one metric).
   - Bounded sigma search with tolerance (~12%) + temporal smoothing of
     sigma per track; premultiplied blur so RGB and alpha soften together.

3. **Grain / compression-texture matching**
   - Luma/chroma space, robust MAD estimator, annular sampling around the
     face with edge/saturation rejection; correlated (not white) noise;
     deterministic per-(track, frame) seeds; applied after all blurs,
     before encode.

4. **Motion blur** (last — wrong blur is more destructive than missing blur)
   - Landmark-based partial-affine motion (RANSAC), not one centroid vector;
     translation-dominant cases get a linear PSF, rotation-dominant cases are
     damped and flagged.
   - Blur length = displacement x shutter_fraction (default 0.5, configurable)
     x calibration factor, capped at a fraction of face width; zero below a
     displacement floor or when inlier ratio is poor.
   - Pipeline order: motion blur first, then the sharpness matcher adds only
     the *residual* isotropic blur needed — the plate measurement already
     contains motion smear, so matching total sharpness first and then adding
     motion blur would double-blur.

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
