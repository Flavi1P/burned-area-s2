# Methods x threshold regimes

Generated 2026-08-24T15:22:44 from `config.yaml` by `python -m src.eval.regimes`. Every cell comes from the same domain, threshold, metric and bootstrap code in `src/eval/`; each method contributes a score raster and nothing else, and all of them were scored on identically the same pixels of each event — asserted in `run()`, not hoped for.

## The methods

| Method | Per-pixel inputs | Spatial context | Role |
|---|---|---|---|
| dNBR | 2 bands x 2 dates | 1 px | reported |
| GB pixel | 16 features, 4 bands x 2 dates | 1 px | reported |
| GB pixel (change-only) | 9 features, no absolute post-fire level | 1 px | ablation |
| U-Net | 4 bands x 2 dates | 256 px tile | reported |

`GB pixel` is the control the original design lacked: every band the network sees, per pixel, with no context at all. It sits between the index and the U-Net on the only two axes that separate them, so the share of the network's lead it reproduces is the share that was never about spatial context. The estimator is `sklearn.HistGradientBoostingClassifier`.

## The regimes

| Regime | What it may look at | Deployable |
|---|---|---|
| frozen | the training event's calibration blocks, labels included | yes |
| unsupervised | the target event's score histogram, no labels | yes |
| oracle | the target event's labels | **no** |

Only the first two could be run on a fire whose perimeter nobody has drawn yet. The oracle is an instrument: the ceiling of what any threshold on that score can reach, so the distance to it separates a method's ranking from its operating point. Two unsupervised estimators are reported — otsu, gmm — because both were tried on the test events before either was written into `config.yaml`, and publishing only the winner would be a selection on the test events with the evidence removed.

## The frozen thresholds

| Score | Threshold | F1 attained on the calibration blocks |
|---|---|---|
| dNBR | ≥ **0.5763** | 0.383 |
| GB pixel | ≥ **0.8841** | 0.946 |
| GB pixel (change-only) | ≥ **0.8354** | 0.942 |
| U-Net | ≥ **0.8852** | 0.940 |

Same function, same pixels, every method — then frozen. The attained F1 values are not comparable to each other; each is the best its own score can do on those pixels, and it is the procedure that is shared.

## Results

### saumos

| Method | Regime | Deployable | IoU | F1 | Precision | Recall | AP | Threshold | Area error |
|---|---|---|---|---|---|---|---|---|---|
| dNBR | frozen | yes | **0.702** | 0.825 | 0.918 | 0.750 | 0.906 | 0.576 | -18.3% |
| dNBR | unsupervised (otsu) | yes | **0.811** | 0.896 | 0.859 | 0.936 | 0.906 | 0.393 | +8.9% |
| dNBR | unsupervised (gmm) | yes | **0.591** | 0.743 | 0.595 | 0.987 | 0.906 | 0.204 | +65.7% |
| dNBR | oracle | — *instrument* | **0.812** | 0.896 | 0.867 | 0.929 | 0.906 | 0.406 | +7.2% |
| GB pixel | frozen | yes | **0.881** | 0.937 | 0.980 | 0.898 | 0.981 | 0.884 | -8.4% |
| GB pixel | unsupervised (otsu) | yes | **0.797** | 0.887 | 0.814 | 0.975 | 0.981 | 0.470 | +19.9% |
| GB pixel | unsupervised (gmm) | yes | **0.289** | 0.449 | 0.289 | 0.998 | 0.981 | 0.029 | +244.7% |
| GB pixel | oracle | — *instrument* | **0.893** | 0.943 | 0.966 | 0.922 | 0.981 | 0.821 | -4.5% |
| GB pixel (change-only) | frozen | yes | **0.888** | 0.941 | 0.962 | 0.921 | 0.980 | 0.835 | -4.2% |
| GB pixel (change-only) | unsupervised (otsu) | yes | **0.871** | 0.931 | 0.892 | 0.973 | 0.980 | 0.483 | +9.1% |
| GB pixel (change-only) | unsupervised (gmm) | yes | **0.292** | 0.452 | 0.292 | 0.998 | 0.980 | 0.030 | +241.8% |
| GB pixel (change-only) | oracle | — *instrument* | **0.899** | 0.947 | 0.944 | 0.949 | 0.980 | 0.704 | +0.5% |
| U-Net | frozen | yes | **0.863** | 0.927 | 0.939 | 0.915 | 0.976 | 0.885 | -2.5% |
| U-Net | unsupervised (otsu) | yes | **0.857** | 0.923 | 0.874 | 0.978 | 0.976 | 0.492 | +12.0% |
| U-Net | unsupervised (gmm) | yes | **0.518** | 0.683 | 0.518 | 0.999 | 0.976 | 0.054 | +92.8% |
| U-Net | oracle | — *instrument* | **0.875** | 0.933 | 0.915 | 0.952 | 0.976 | 0.761 | +4.0% |

### biscarrosse

| Method | Regime | Deployable | IoU | F1 | Precision | Recall | AP | Threshold | Area error |
|---|---|---|---|---|---|---|---|---|---|
| dNBR | frozen | yes | **0.481** | 0.649 | 0.623 | 0.678 | 0.641 | 0.576 | +8.8% |
| dNBR | unsupervised (otsu) | yes | **0.501** | 0.668 | 0.517 | 0.943 | 0.641 | 0.321 | +82.2% |
| dNBR | unsupervised (gmm) | yes | **0.403** | 0.575 | 0.405 | 0.990 | 0.641 | 0.128 | +144.5% |
| dNBR | oracle | — *instrument* | **0.526** | 0.689 | 0.574 | 0.861 | 0.641 | 0.444 | +49.9% |
| GB pixel | frozen | yes | **0.561** | 0.719 | 0.629 | 0.839 | 0.675 | 0.884 | +33.3% |
| GB pixel | unsupervised (otsu) | yes | **0.558** | 0.717 | 0.577 | 0.946 | 0.675 | 0.470 | +64.1% |
| GB pixel | unsupervised (gmm) | yes | **0.310** | 0.474 | 0.311 | 0.998 | 0.675 | 0.007 | +221.5% |
| GB pixel | oracle | — *instrument* | **0.568** | 0.725 | 0.604 | 0.905 | 0.675 | 0.713 | +49.7% |
| GB pixel (change-only) | frozen | yes | **0.568** | 0.725 | 0.618 | 0.876 | 0.675 | 0.835 | +41.6% |
| GB pixel (change-only) | unsupervised (otsu) | yes | **0.560** | 0.718 | 0.578 | 0.945 | 0.675 | 0.471 | +63.5% |
| GB pixel (change-only) | unsupervised (gmm) | yes | **0.310** | 0.474 | 0.310 | 0.998 | 0.675 | 0.007 | +221.6% |
| GB pixel (change-only) | oracle | — *instrument* | **0.569** | 0.725 | 0.607 | 0.902 | 0.675 | 0.745 | +48.6% |
| U-Net | frozen | yes | **0.571** | 0.727 | 0.618 | 0.883 | 0.702 | 0.885 | +42.9% |
| U-Net | unsupervised (otsu) | yes | **0.533** | 0.696 | 0.546 | 0.959 | 0.702 | 0.483 | +75.7% |
| U-Net | unsupervised (gmm) | yes | **0.373** | 0.544 | 0.374 | 0.997 | 0.702 | 0.048 | +166.7% |
| U-Net | oracle | — *instrument* | **0.571** | 0.727 | 0.624 | 0.871 | 0.702 | 0.901 | +39.7% |

### fontainebleau

| Method | Regime | Deployable | IoU | F1 | Precision | Recall | AP | Threshold | Area error |
|---|---|---|---|---|---|---|---|---|---|
| dNBR | frozen | yes | **0.192** | 0.323 | 0.966 | 0.194 | 0.839 | 0.576 | -80.0% |
| dNBR | unsupervised (otsu) | yes | **0.466** | 0.636 | 0.469 | 0.989 | 0.839 | 0.173 | +111.0% |
| dNBR | unsupervised (gmm) | yes | **0.258** | 0.410 | 0.258 | 1.000 | 0.839 | 0.052 | +287.8% |
| dNBR | oracle | — *instrument* | **0.624** | 0.768 | 0.708 | 0.841 | 0.839 | 0.284 | +18.8% |
| GB pixel | frozen | yes | **0.487** | 0.655 | 0.931 | 0.506 | 0.887 | 0.884 | -45.7% |
| GB pixel | unsupervised (otsu) | yes | **0.696** | 0.821 | 0.801 | 0.842 | 0.887 | 0.412 | +5.1% |
| GB pixel | unsupervised (gmm) | yes | **0.274** | 0.430 | 0.274 | 1.000 | 0.887 | 0.006 | +265.3% |
| GB pixel | oracle | — *instrument* | **0.701** | 0.824 | 0.819 | 0.830 | 0.887 | 0.438 | +1.3% |
| GB pixel (change-only) | frozen | yes | **0.526** | 0.689 | 0.914 | 0.553 | 0.875 | 0.835 | -39.5% |
| GB pixel (change-only) | unsupervised (otsu) | yes | **0.653** | 0.790 | 0.728 | 0.864 | 0.875 | 0.395 | +18.7% |
| GB pixel (change-only) | unsupervised (gmm) | yes | **0.235** | 0.381 | 0.235 | 1.000 | 0.875 | 0.007 | +325.4% |
| GB pixel (change-only) | oracle | — *instrument* | **0.681** | 0.810 | 0.807 | 0.813 | 0.875 | 0.505 | +0.7% |
| U-Net | frozen | yes | **0.556** | 0.715 | 0.839 | 0.623 | 0.830 | 0.885 | -25.7% |
| U-Net | unsupervised (otsu) | yes | **0.599** | 0.749 | 0.656 | 0.874 | 0.830 | 0.437 | +33.3% |
| U-Net | unsupervised (gmm) | yes | **0.365** | 0.535 | 0.366 | 0.990 | 0.830 | 0.051 | +170.3% |
| U-Net | oracle | — *instrument* | **0.621** | 0.766 | 0.739 | 0.795 | 0.830 | 0.682 | +7.6% |

AP is threshold-free, so it is constant down each method's four rows by construction: it describes the score, and the regime only chooses where to cut it. Comparing AP across methods is therefore the cleanest statement of which score carries more separable signal, independent of any operating point.

No interval is published anywhere in this table. The rule is unchanged: the bootstrap resamples ~17.5 km blocks and counts only those containing burned pixels, and no evaluation domain in this project reaches 4. Each row carries its own reason in `regimes.json`. Adding methods and regimes multiplies the comparisons this table invites while adding no replicates to support them, and that is the honest caveat on every difference read off it.
