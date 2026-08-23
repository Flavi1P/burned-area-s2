# Results — U-Net against the dNBR baseline

Generated 2026-08-22T20:37:14 from `config.yaml` by `python -m src.model.evaluate`, which recomputes both methods in one pass. Every cell below -- both methods, every event -- comes from the same domain, metric, threshold and bootstrap code in `src/eval/`; the network contributes a probability raster and nothing else. Both decision thresholds were fitted by maximising F1 on the same held-out calibration blocks of the training event, frozen there, and applied unchanged to every test event.

## The decision thresholds

Calibrated on 772,271 usable pixels of the saumos calibration blocks (r0c0, r1c2), of which 16,653 are burned (2.16% prevalence). Same pixels, same objective, same function, both sides — then frozen, and applied unchanged to every test event with no recalibration on any target.

| Score | Threshold | Objective attained on the calibration blocks |
|---|---|---|
| dnbr | ≥ **0.5763** | F1 = 0.383 |
| unet | ≥ **0.8852** | F1 = 0.940 |

The two objective values are not comparable to each other: each is the best F1 its own score can reach on those pixels, and it is the *procedure* that is shared, not the number it lands on.

## Results

| Exp. | Event | Method | IoU [95% CI] | F1 | Precision | Recall | AP | Predicted ha | EMS ha | Error |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | saumos | U-Net | 0.863 — | 0.927 | 0.939 | 0.915 | 0.976 | 5,107 | 5,239 | -2.5% |
| E1 | saumos | dNBR honest | 0.702 — | 0.825 | 0.918 | 0.750 | 0.906 | 4,281 | 5,239 | -18.3% |
| E1 | saumos | dNBR oracle | 0.812 — | 0.896 | 0.867 | 0.929 | 0.906 | 5,615 | 5,239 | +7.2% |
| E2 | biscarrosse | U-Net | 0.571 — | 0.727 | 0.618 | 0.883 | 0.702 | 2,504 | 1,752 | +42.9% |
| E2 | biscarrosse | dNBR honest | 0.481 — | 0.649 | 0.623 | 0.678 | 0.641 | 1,905 | 1,752 | +8.8% |
| E2 | biscarrosse | dNBR oracle | 0.526 — | 0.689 | 0.574 | 0.861 | 0.641 | 2,626 | 1,752 | +49.9% |
| E2 | fontainebleau | U-Net | 0.556 — | 0.715 | 0.839 | 0.623 | 0.830 | 617 | 831 | -25.7% |
| E2 | fontainebleau | dNBR honest | 0.192 — | 0.323 | 0.966 | 0.194 | 0.839 | 167 | 831 | -80.0% |
| E2 | fontainebleau | dNBR oracle | 0.624 — | 0.768 | 0.708 | 0.841 | 0.839 | 988 | 831 | +18.8% |

Intervals would be percentile bootstraps over ~17.5 km spatial blocks. A dash means no interval is publishable for that domain, and every dash here has the same cause: the protocol counts blocks that actually contain burned pixels, and no evaluation domain in this project reaches 4 of them. The two transfer events fit in one block each; the training event has four held-out blocks but 94% of their burned area lies in one of them. The point estimates stand, and the reason each interval is missing is written out in `baseline_results.json`.

`EMS ha` is the rasterised delineation inside the evaluated pixels, never a press figure. `AP` is average precision — threshold-free, so the honest and oracle rows share it by construction: it is one curve read at two operating points.

## dNBR ↔ EMS agreement — the circularity floor

How much of any "learned method beats the index" result is the index agreeing with the way the label was drawn in the first place. Average precision is threshold-free; the oracle columns are the best a dNBR threshold can do against this label, which is the same operating point as the oracle row above, read as a property of the label rather than of the method.

| Event | EMS production method | AP | Oracle IoU | Oracle κ |
|---|---|---|---|---|
| saumos | Semi-automatic extraction | 0.906 | 0.812 | 0.889 |
| biscarrosse | Semi-automatic extraction | 0.641 | 0.526 | 0.668 |
| fontainebleau | Semi-automatic extraction | 0.839 | 0.624 | 0.760 |

## What was actually evaluated

| Event | Footprint px | Usable | Evaluated px | Blocks (with burn) | Burned ha evaluated | Rasterised total ha | EMS reported ha | Press ha |
|---|---|---|---|---|---|---|---|---|
| saumos | 6,216,067 | 72.7% | 1,938,519 | 4 (3) | 5,239 | 31,592 | 31,602 | 42,000 |
| biscarrosse | 851,370 | 96.5% | 821,389 | 1 (1) | 1,752 | 1,752 | 1,753 | 2,200 |
| fontainebleau | 649,264 | 100.0% | 649,170 | 1 (1) | 831 | 831 | 832 | — |

`Usable` is the share of the geometric footprint left after cloud and cloud-shadow masking on both dates. The training event is evaluated only on its held-out test blocks, which is why its evaluated area is a fraction of its rasterised total. Press hectares are shown for reference only: they aggregate several fronts and several dates and are not a reference any metric here is computed against.

## Confusion matrices

| Exp. | Event | Method | TP | FP | FN | TN |
|---|---|---|---|---|---|---|
| E1 | saumos | U-Net | 119,846 | 7,820 | 11,140 | 1,799,713 |
| E1 | saumos | dNBR honest | 98,195 | 8,821 | 32,791 | 1,798,712 |
| E1 | saumos | dNBR oracle | 121,635 | 18,737 | 9,351 | 1,788,796 |
| E2 | biscarrosse | U-Net | 38,661 | 23,944 | 5,138 | 753,646 |
| E2 | biscarrosse | dNBR honest | 29,688 | 17,945 | 14,111 | 759,645 |
| E2 | biscarrosse | dNBR oracle | 37,714 | 27,940 | 6,085 | 749,650 |
| E2 | fontainebleau | U-Net | 12,945 | 2,490 | 7,839 | 625,896 |
| E2 | fontainebleau | dNBR honest | 4,025 | 142 | 16,759 | 628,244 |
| E2 | fontainebleau | dNBR oracle | 17,473 | 7,216 | 3,311 | 621,170 |
