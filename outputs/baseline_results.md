# Baseline results — dNBR

Generated 2026-08-22T16:14:10 from `config.yaml` by `python -m src.eval.baseline`. No model has been trained at this point: these rows exist to validate the data chain and the evaluation machinery before any deep learning is added to the picture.

## The decision threshold

dNBR ≥ **0.7435**, obtained by maximising F1 on 772,269 usable pixels of the saumos calibration blocks (r0c0, r1c2), of which 16,651 are burned (2.16% prevalence). It reaches F1 = 0.561 there, and is then frozen: every row below applies this same value, with no recalibration on any test event. The U-Net's threshold will be produced by the same function on the same pixels.

## Results

| Exp. | Event | Method | IoU [95% CI] | F1 | Precision | Recall | AP | Predicted ha | EMS ha | Error |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | saumos | dNBR honest | 0.521 — | 0.685 | 0.684 | 0.686 | 0.582 | 5,256 | 5,239 | +0.3% |
| E1 | saumos | dNBR oracle | 0.525 — | 0.689 | 0.716 | 0.664 | 0.582 | 4,860 | 5,239 | -7.2% |
| E2 | biscarrosse | dNBR honest | 0.427 — | 0.598 | 0.510 | 0.724 | 0.424 | 2,487 | 1,752 | +42.0% |
| E2 | biscarrosse | dNBR oracle | 0.429 — | 0.600 | 0.525 | 0.701 | 0.424 | 2,340 | 1,752 | +33.5% |
| E2 | fontainebleau | dNBR honest | 0.388 — | 0.559 | 0.890 | 0.407 | 0.547 | 380 | 831 | -54.2% |
| E2 | fontainebleau | dNBR oracle | 0.435 — | 0.606 | 0.730 | 0.518 | 0.547 | 590 | 831 | -29.1% |

Intervals would be percentile bootstraps over ~17.5 km spatial blocks. A dash means no interval is publishable for that domain, and every dash here has the same cause: the protocol counts blocks that actually contain burned pixels, and no evaluation domain in this project reaches 4 of them. The two transfer events fit in one block each; the training event has four held-out blocks but 94% of their burned area lies in one of them. The point estimates stand, and the reason each interval is missing is written out in `baseline_results.json`.

`EMS ha` is the rasterised delineation inside the evaluated pixels, never a press figure. `AP` is average precision — threshold-free, so the honest and oracle rows share it by construction: it is one curve read at two operating points.

## dNBR ↔ EMS agreement — the circularity floor

How much of any "learned method beats the index" result is the index agreeing with the way the label was drawn in the first place. Average precision is threshold-free; the oracle columns are the best a dNBR threshold can do against this label, which is the same operating point as the oracle row above, read as a property of the label rather than of the method.

| Event | EMS production method | AP | Oracle IoU | Oracle κ |
|---|---|---|---|---|
| saumos | Semi-automatic extraction | 0.582 | 0.525 | 0.667 |
| biscarrosse | Semi-automatic extraction | 0.424 | 0.429 | 0.574 |
| fontainebleau | Semi-automatic extraction | 0.547 | 0.435 | 0.595 |

## What was actually evaluated

| Event | Footprint px | Usable | Evaluated px | Blocks (with burn) | Burned ha evaluated | Rasterised total ha | EMS reported ha | Press ha |
|---|---|---|---|---|---|---|---|---|
| saumos | 6,216,067 | 72.7% | 1,938,505 | 4 (3) | 5,239 | 31,592 | 31,602 | 42,000 |
| biscarrosse | 851,370 | 96.5% | 821,383 | 1 (1) | 1,752 | 1,752 | 1,753 | 2,200 |
| fontainebleau | 649,264 | 100.0% | 649,169 | 1 (1) | 831 | 831 | 832 | — |

`Usable` is the share of the geometric footprint left after cloud and cloud-shadow masking on both dates. The training event is evaluated only on its held-out test blocks, which is why its evaluated area is a fraction of its rasterised total. Press hectares are shown for reference only: they aggregate several fronts and several dates and are not a reference any metric here is computed against.

## Confusion matrices

| Exp. | Event | Method | TP | FP | FN | TN |
|---|---|---|---|---|---|---|
| E1 | saumos | dNBR honest | 89,888 | 41,503 | 41,091 | 1,766,023 |
| E1 | saumos | dNBR oracle | 86,945 | 34,560 | 44,034 | 1,772,966 |
| E2 | biscarrosse | dNBR honest | 31,706 | 30,478 | 12,090 | 747,109 |
| E2 | biscarrosse | dNBR oracle | 30,693 | 27,796 | 13,103 | 749,791 |
| E2 | fontainebleau | dNBR honest | 8,463 | 1,048 | 12,320 | 627,338 |
| E2 | fontainebleau | dNBR oracle | 10,766 | 3,978 | 10,017 | 624,408 |
