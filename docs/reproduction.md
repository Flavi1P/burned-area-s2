# Reproduction from a clean clone

**Date:** 2026-10-20. **Reproduced commit:** `a50ba66` (`main`).
**Machine:** Windows 11, 16 CPU threads, no GPU.

## Procedure

1. Fresh `git clone` of https://github.com/Flavi1P/burned-area-s2 into an empty
   directory: no `data/`, no weights, nothing cached.
2. Fresh conda environment solved from `environment.yml` as committed (only the
   environment name changed, so the original env stayed untouched).
3. Every command from the README section *Running it*, in order, then
   `pytest -q -rs`.
4. Every `outputs/*.json` compared leaf by leaf to the committed version, every
   `outputs/*.png` pixel by pixel, and the intermediate rasters
   (`stack.tif`, `dnbr.tif`, `unet_prob.tif`, `gb_prob_full.tif`, label rasters)
   by checksum against the original working copy.

## Environment drift

`environment.yml` uses lower bounds, so the solve on 2026-10-20 is not the
August one:

| package      | 2026-08 env | 2026-10-20 solve |
|--------------|-------------|------------------|
| python       | 3.11.15     | 3.11.17          |
| pytorch      | 2.13.0      | 2.14.1           |
| scikit-learn | 1.9.0       | 1.9.1            |
| matplotlib   | 3.11.1      | 3.11.2           |
| numpy        | 2.4.6       | 2.4.6            |
| rasterio     | 1.4.4       | 1.4.4            |

## Result

**Every number is identical.** Not identical up to rounding: bit-identical.

| artefact | outcome |
|---|---|
| Sentinel-2 scenes (STAC Element 84), all three events | still available; `stack.tif` byte-identical |
| EMS delineation packages (EMSR894, EMSR899, EMSR902) | still available; label rasters byte-identical |
| dNBR: `baseline_results.json`, `thresholds.json` | every value identical |
| Spatial blocks, tiles, normalisation | identical |
| GB pixel model: `pixel_model.json`, `gb_prob_full.tif` | identical |
| U-Net: `training_e1.json` (per-epoch loss and AP, selected epoch), `unet_prob.tif` | identical; only wall-clock differs (738 s vs 610 s) |
| `results.json`, `regimes.json` and their `.md` tables | every value identical |
| Figures (`outputs/*.png`) | pixel-identical; bytes differ only in PNG metadata (matplotlib version) |
| `pytest -q` | 84 passed |

The U-Net reproducing bit for bit means the CPU training path is deterministic
for a fixed seed on this machine. It says nothing about the variance *between*
seeds. That variance is a separate question and is not measured here.

## Findings

1. **The documented run order failed from a clean clone.** `src.eval.blocks`
   crashed on a missing `data/interim/<event>/label_*.tif`. The label rasters
   were written only as a side effect of `src.viz.quicklook`, which the README
   ran near the end and only for `saumos`. The original working copy hid this
   because quicklook had been run earlier for every event.
   - **Fix:** `python -m src.data.ems` now writes the label rasters itself.
   - **Check:** re-running it in the clone reproduces the three label rasters
     byte for byte. No result is affected.
   - **How this run got past it:** it called quicklook for all three events
     right after `src.data.stack`, then continued the documented order.
2. **Three committed JSONs were behind the code that generates them.**
   `baseline_results.json`, `results.json` and `thresholds.json` predate the
   `regime` / `uses_labels` fields that `src/eval/threshold.py` now writes. Only
   those keys are missing; every value already present is identical. The
   committed versions will be regenerated when the outputs are next rewritten.
