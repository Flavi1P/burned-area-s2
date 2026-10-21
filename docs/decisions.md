# Decisions log

Dated record of choices that changed the protocol or the reading of the
results. Dates are the commit that introduced them. The README states the
current position. This file keeps how it was reached.

## 2026-08-22 — Role buffer of 1 km, not 2.56 km (`068dce8`)

The first role buffer was 2560 m, half a tile. The block seams cut through the
middle of the Saumos scar, so that buffer removed 98% of the burned area in the
calibration blocks (1 442 → 30 ha), which left nothing to calibrate a threshold
on. It is now 1000 m. That costs 10.7% of the footprint and keeps 666 ha. The
residual risk is correlation across a seam at 1–2 km, below the 17.5 km
super-block range. Decided on the training event only.

## 2026-08-23 — Sentinel-2 baseline-04.00 offset applied twice (`44e7906`)

The first published dNBR numbers were computed on reflectance with the
baseline-04.00 offset applied twice, once by the provider and once by
`src/data/stack.py`. The result was 90% of red and 71% of B12 pixels negative,
and NBR outside ±1 for about half the scene.
- **Effect:** it crippled the index but not the network, because per-channel
  standardisation absorbs an additive shift. The comparison was unfair in the
  network's favour.
- **Fix:** the bug was found while checking the network's inputs and fixed in
  `stack.py`, and the chain was re-run. Saumos dNBR IoU went from 0.521 to 0.702.
- **Guard:** any band more than 10% negative now fails loudly.

## 2026-08-24 — Per-pixel control added; context reading withdrawn (`a50ba66`)

The first README read the U-Net's lead over the dNBR oracle on pine (Saumos
0.863 vs 0.812, Biscarrosse 0.571 vs 0.526) as spatial context. That reading
confounded two differences: 6 extra bands and the receptive field. A
gradient-boosting model over the same channels, per pixel, reproduced almost
all of the lead (0.881 / 0.561, change-only 0.888 / 0.568). The context claim
was withdrawn. What remains for context is the Biscarrosse AP gap (0.702 vs
0.675), which is untested (T4.3).

## 2026-08-24 — Unsupervised threshold regime; Otsu chosen after seeing test events (`a50ba66`)

Added after E2 showed that the frozen threshold is the main transfer failure:
on Fontainebleau the dNBR oracle sits at 0.284 against a frozen 0.576.
- **What was tried:** Otsu and a 2-component GMM, both on the test events,
  before either was written to `config.yaml`. Otsu won.
- **This is a selection on the test events.** Both estimators stay in every
  table. The GMM fails everywhere (Fontainebleau GB 0.274, recall 1.000).
- **Remedy:** a pre-registered hold-out event evaluated once (handoff phase 3).

## 2026-08-24 — The U-Net gets an oracle row in the regime grid (`a50ba66`)

`src/model/evaluate.py` and `outputs/results.md` still refuse the U-Net an
oracle. With two methods, an oracle for the network would have dissolved the
decomposition the dNBR oracle exists for. In the methods × regimes grid, the
decomposition is the grid itself, so every method gets every regime. Oracle
cells are marked non-deployable in `regimes.json`, and a test asserts it.
`outputs/results.md` is kept as the historical two-method table.

## 2026-10-20 — Label rasters written by `src.data.ems` (`aeb668c`)

From a clean clone, the documented order failed: only `src.viz.quicklook` wrote
the label rasters. The rasters are unchanged byte for byte. See
`docs/reproduction.md`.

## 2026-10-20 — Five seeds per trained model; separation rule fixed before running

Written and committed before any multi-seed result existed.
- **Reversal.** `config.yaml` said: "No multi-seed spread: it measures optimiser
  variance, not the variance that matters." The second half still holds: seed
  spread says nothing about variance across events. But the U-Net vs GB gaps are
  0.01–0.03 IoU, and a single run cannot tell whether that is more than training
  noise. That is the question a reviewer asks first.
- **Seeds.** `model.seeds` and `pixel_model.seeds` are both
  `[20260821 … 20260825]`. The first is the original single run, so its rows
  must reproduce the August numbers. All five are reported. Choosing among them
  is forbidden, and a test enforces it.
- **Per seed:** training, epoch selection (AP on the calibration blocks) and the
  frozen threshold. The bootstrap and the GMM subsample now draw from
  `evaluation.seed`, with the same value as before, so they do not move.
- **Separation rule.** Two methods are *separated* on a metric only when their
  seed ranges [min, max] do not overlap. Otherwise the gap is
  "indistinguishable at this sample size". The sd is the sample sd (ddof = 1).
- **Historical table.** `outputs/results.md` keeps scoring the first seed only.

## 2026-10-21 — Five-seed results: what held, what did not (`c909fca`)

Seed 20260821 reproduced all 48 August cells exactly. Under the rule above:

| Claim (single run) | After 5 seeds | Status |
|---|---|---|
| GB matches the U-Net on pine (Saumos 0.881 vs 0.863, Biscarrosse 0.561 vs 0.571) | Saumos 0.887 vs 0.865, separated, GB ahead. Biscarrosse 0.563 vs 0.557, indistinguishable | holds (Saumos: GB ahead) |
| Context edge on Biscarrosse ranking (AP 0.702 vs 0.675) | 0.685 ± 0.011 vs 0.678 ± 0.004, indistinguishable. 0.702 was the U-Net's best seed | dropped |
| U-Net transfers better under the frozen threshold on Fontainebleau (0.556 vs 0.487) | 0.489 ± 0.068 (0.394–0.556) vs 0.500 ± 0.011, indistinguishable. 0.556 was the U-Net's best seed | dropped |
| GB ranks better than the U-Net on Fontainebleau (AP 0.887 vs 0.830) | 0.885 vs 0.828, separated | holds |
| dNBR frozen threshold fails on Fontainebleau (0.192 vs oracle 0.624) | no seed: unchanged | holds |
| GB + Otsu beats the dNBR oracle on Fontainebleau (0.696 vs 0.624) | 0.697 ± 0.006, separated. Also beats U-Net + Otsu (0.611 ± 0.030) | holds; still rests on Otsu chosen after the fact |
| Otsu costs IoU where frozen was right (Saumos GB 0.881 → 0.797) | 0.887 → 0.846 ± 0.028. 0.797 was GB's worst seed | weakened (smaller cost) |
| GMM fails everywhere | every seed | holds |
| — | The U-Net is never ahead of GB beyond seed noise, in any cell | new |
| — | The U-Net's frozen operating point is unstable in transfer: Fontainebleau sd 0.068, frozen threshold 0.814–0.889 | new |

- **Pattern.** The original run was a favourable draw for the U-Net in
  transfer: it was the best of the five seeds on Biscarrosse IoU and AP and on
  Fontainebleau frozen IoU. Nobody chose it; it was the only run.
- **Consequence for T4.3.** The clearcut hypothesis was proposed to explain the
  Biscarrosse AP gap, and that gap is gone. The error-map test (handoff phase 2)
  still runs as planned, but it now tests a hypothesis with nothing to explain.
- **Caveat on GB's narrow spread.** Its 1 410 610 train-role pixels are under
  `max_train_pixels`, so its seed moves only the early-stopping split. That is
  less training noise than the U-Net has, partly by construction.

## 2026-10-21 — Phase 2 narrowed to error maps; drawn seed fixed before any map

Written and committed before any error map existed.
- **Morphology test dropped, not run.** It was designed to explain the
  Biscarrosse AP gap by clearcuts: FPs that GB makes and the U-Net avoids would
  be more rectangular. Over five seeds that gap is indistinguishable (entry
  above). With no measured effect, an FP-shape statistic could only go looking
  for a story. The clearcut hypothesis stays unverified and is worded as such.
  If a context effect is ever measured (very high resolution is the expected
  place), the test can be pre-registered then.
- **What remains.** TP / FP / FN maps for every method (dNBR, GB, U-Net) at its
  frozen threshold, over the post-fire false-colour composite, on all three
  events. Diagnostics: no README number is read off them.
- **Drawn seed.** For each trained model, the seed whose frozen-threshold IoU on
  the Saumos calibration blocks is the median of the five (ties: lowest seed).
  No test event is consulted. The rule was chosen with the user before the
  selected seeds were looked up. The per-seed calibration F1 values were already
  public in `outputs/regimes.json`, and IoU is monotone in F1, so the selection
  was computable from published numbers; it was not computed first. The rule
  lives in `config.yaml` (`error_maps.representative_seed`).
- **Zoom.** Biscarrosse, a 128 px window placed where FPs are densest, summed
  over the methods drawn. A rule, so the window is not picked to flatter one
  method.
