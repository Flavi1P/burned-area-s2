# Methods x threshold regimes

Generated 2026-10-21T10:24:31 from `config.yaml` by `python -m src.eval.regimes`. Every cell comes from the same domain, threshold, metric and bootstrap code in `src/eval/`; each method contributes a score raster and nothing else, and all of them were scored on identically the same pixels of each event — asserted in `run()`, not hoped for.

## The methods

| Method | Per-pixel inputs | Spatial context | Role | Seeds |
|---|---|---|---|---|
| dNBR | 2 bands x 2 dates | 1 px | reported | — (deterministic) |
| GB pixel | 16 features, 4 bands x 2 dates | 1 px | reported | 5 |
| GB pixel (change-only) | 9 features, no absolute post-fire level | 1 px | ablation | 5 |
| U-Net | 4 bands x 2 dates | 256 px tile | reported | 5 |

`GB pixel` is the control the original design lacked: every band the network sees, per pixel, with no context at all. It sits between the index and the U-Net on the only two axes that separate them, so the share of the network's lead it reproduces is the share that was never about spatial context. The estimator is `sklearn.HistGradientBoostingClassifier`.

## The regimes

| Regime | What it may look at | Deployable |
|---|---|---|
| frozen | the training event's calibration blocks, labels included | yes |
| unsupervised | the target event's score histogram, no labels | yes |
| oracle | the target event's labels | **no** |

Only the first two could be run on a fire whose perimeter nobody has drawn yet. The oracle is an instrument: the ceiling of what any threshold on that score can reach, so the distance to it separates a method's ranking from its operating point. Two unsupervised estimators are reported — otsu, gmm — because both were tried on the test events before either was written into `config.yaml`, and publishing only the winner would be a selection on the test events with the evidence removed.

## The frozen thresholds

| Score | Seeds | Threshold | F1 attained on the calibration blocks |
|---|---|---|---|
| dNBR | 1 | ≥ 0.576 | 0.383 |
| GB pixel | 5 | ≥ 0.867 ± 0.012 (0.851–0.884) | 0.946 ± 0.002 (0.944–0.948) |
| GB pixel (change-only) | 5 | ≥ 0.838 ± 0.007 (0.830–0.847) | 0.944 ± 0.004 (0.937–0.949) |
| U-Net | 5 | ≥ 0.864 ± 0.034 (0.814–0.889) | 0.934 ± 0.006 (0.924–0.940) |

Same function, same pixels, every method and every seed — then frozen, one threshold per seed. The attained F1 values are not comparable to each other; each is the best its own score can do on those pixels, and it is the procedure that is shared. Per-seed values are in `regimes.json`.

## Results

IoU and AP are **mean ± sd (min–max)** over the seeds; a single number is the dNBR, which has no seed. The other columns are seed means. Per-seed rows are in `regimes.json`.

### saumos

| Method | Regime | Deployable | IoU | AP | F1 | Precision | Recall | Threshold | Area error |
|---|---|---|---|---|---|---|---|---|---|
| dNBR | frozen | yes | **0.702** | 0.906 | 0.825 | 0.918 | 0.750 | 0.576 | -18.3% |
| dNBR | unsupervised (otsu) | yes | **0.811** | 0.906 | 0.896 | 0.859 | 0.936 | 0.393 | +8.9% |
| dNBR | unsupervised (gmm) | yes | **0.591** | 0.906 | 0.743 | 0.595 | 0.987 | 0.204 | +65.7% |
| dNBR | oracle | — *instrument* | **0.812** | 0.906 | 0.896 | 0.867 | 0.929 | 0.406 | +7.2% |
| GB pixel | frozen | yes | **0.887 ± 0.004 (0.881–0.891)** | 0.983 ± 0.002 (0.981–0.985) | 0.940 | 0.978 | 0.905 | 0.867 | -7.5% |
| GB pixel | unsupervised (otsu) | yes | **0.846 ± 0.028 (0.797–0.866)** | 0.983 ± 0.002 (0.981–0.985) | 0.916 | 0.865 | 0.975 | 0.476 | +12.8% |
| GB pixel | unsupervised (gmm) | yes | **0.294 ± 0.009 (0.280–0.305)** | 0.983 ± 0.002 (0.981–0.985) | 0.454 | 0.294 | 0.998 | 0.029 | +239.7% |
| GB pixel | oracle | — *instrument* | **0.902 ± 0.006 (0.893–0.910)** | 0.983 ± 0.002 (0.981–0.985) | 0.948 | 0.959 | 0.938 | 0.752 | -2.1% |
| GB pixel (change-only) | frozen | yes | **0.888 ± 0.001 (0.887–0.889)** | 0.979 ± 0.002 (0.975–0.981) | 0.941 | 0.963 | 0.919 | 0.838 | -4.5% |
| GB pixel (change-only) | unsupervised (otsu) | yes | **0.872 ± 0.006 (0.864–0.879)** | 0.979 ± 0.002 (0.975–0.981) | 0.932 | 0.893 | 0.974 | 0.483 | +9.0% |
| GB pixel (change-only) | unsupervised (gmm) | yes | **0.285 ± 0.006 (0.278–0.292)** | 0.979 ± 0.002 (0.975–0.981) | 0.444 | 0.286 | 0.998 | 0.030 | +249.5% |
| GB pixel (change-only) | oracle | — *instrument* | **0.899 ± 0.002 (0.896–0.901)** | 0.979 ± 0.002 (0.975–0.981) | 0.947 | 0.945 | 0.950 | 0.701 | +0.5% |
| U-Net | frozen | yes | **0.865 ± 0.006 (0.858–0.872)** | 0.969 ± 0.013 (0.948–0.982) | 0.927 | 0.927 | 0.927 | 0.864 | -0.0% |
| U-Net | unsupervised (otsu) | yes | **0.841 ± 0.019 (0.819–0.861)** | 0.969 ± 0.013 (0.948–0.982) | 0.913 | 0.854 | 0.982 | 0.491 | +15.2% |
| U-Net | unsupervised (gmm) | yes | **0.403 ± 0.117 (0.221–0.518)** | 0.969 ± 0.013 (0.948–0.982) | 0.566 | 0.403 | 1.000 | 0.054 | +172.2% |
| U-Net | oracle | — *instrument* | **0.870 ± 0.006 (0.864–0.875)** | 0.969 ± 0.013 (0.948–0.982) | 0.931 | 0.912 | 0.950 | 0.789 | +4.2% |

### biscarrosse

| Method | Regime | Deployable | IoU | AP | F1 | Precision | Recall | Threshold | Area error |
|---|---|---|---|---|---|---|---|---|---|
| dNBR | frozen | yes | **0.481** | 0.641 | 0.649 | 0.623 | 0.678 | 0.576 | +8.8% |
| dNBR | unsupervised (otsu) | yes | **0.501** | 0.641 | 0.668 | 0.517 | 0.943 | 0.321 | +82.2% |
| dNBR | unsupervised (gmm) | yes | **0.403** | 0.641 | 0.575 | 0.405 | 0.990 | 0.128 | +144.5% |
| dNBR | oracle | — *instrument* | **0.526** | 0.641 | 0.689 | 0.574 | 0.861 | 0.444 | +49.9% |
| GB pixel | frozen | yes | **0.563 ± 0.001 (0.561–0.564)** | 0.678 ± 0.004 (0.673–0.681) | 0.720 | 0.626 | 0.848 | 0.867 | +35.4% |
| GB pixel | unsupervised (otsu) | yes | **0.559 ± 0.001 (0.558–0.560)** | 0.678 ± 0.004 (0.673–0.681) | 0.717 | 0.577 | 0.946 | 0.470 | +63.8% |
| GB pixel | unsupervised (gmm) | yes | **0.309 ± 0.001 (0.307–0.310)** | 0.678 ± 0.004 (0.673–0.681) | 0.472 | 0.309 | 0.998 | 0.007 | +222.9% |
| GB pixel | oracle | — *instrument* | **0.569 ± 0.001 (0.568–0.570)** | 0.678 ± 0.004 (0.673–0.681) | 0.725 | 0.606 | 0.902 | 0.725 | +48.7% |
| GB pixel (change-only) | frozen | yes | **0.568 ± 0.000 (0.568–0.569)** | 0.674 ± 0.002 (0.671–0.676) | 0.725 | 0.619 | 0.875 | 0.838 | +41.4% |
| GB pixel (change-only) | unsupervised (otsu) | yes | **0.559 ± 0.000 (0.559–0.560)** | 0.674 ± 0.002 (0.671–0.676) | 0.717 | 0.578 | 0.946 | 0.471 | +63.7% |
| GB pixel (change-only) | unsupervised (gmm) | yes | **0.309 ± 0.003 (0.306–0.313)** | 0.674 ± 0.002 (0.671–0.676) | 0.473 | 0.310 | 0.998 | 0.007 | +222.4% |
| GB pixel (change-only) | oracle | — *instrument* | **0.569 ± 0.000 (0.568–0.569)** | 0.674 ± 0.002 (0.671–0.676) | 0.725 | 0.610 | 0.895 | 0.772 | +46.8% |
| U-Net | frozen | yes | **0.557 ± 0.009 (0.544–0.571)** | 0.685 ± 0.011 (0.670–0.702) | 0.716 | 0.606 | 0.873 | 0.864 | +44.1% |
| U-Net | unsupervised (otsu) | yes | **0.529 ± 0.009 (0.516–0.541)** | 0.685 ± 0.011 (0.670–0.702) | 0.692 | 0.543 | 0.954 | 0.480 | +75.6% |
| U-Net | unsupervised (gmm) | yes | **0.349 ± 0.046 (0.268–0.376)** | 0.685 ± 0.011 (0.670–0.702) | 0.516 | 0.349 | 0.998 | 0.049 | +190.4% |
| U-Net | oracle | — *instrument* | **0.559 ± 0.009 (0.547–0.571)** | 0.685 ± 0.011 (0.670–0.702) | 0.717 | 0.612 | 0.865 | 0.876 | +41.4% |

### fontainebleau

| Method | Regime | Deployable | IoU | AP | F1 | Precision | Recall | Threshold | Area error |
|---|---|---|---|---|---|---|---|---|---|
| dNBR | frozen | yes | **0.192** | 0.839 | 0.323 | 0.966 | 0.194 | 0.576 | -80.0% |
| dNBR | unsupervised (otsu) | yes | **0.466** | 0.839 | 0.636 | 0.469 | 0.989 | 0.173 | +111.0% |
| dNBR | unsupervised (gmm) | yes | **0.258** | 0.839 | 0.410 | 0.258 | 1.000 | 0.052 | +287.8% |
| dNBR | oracle | — *instrument* | **0.624** | 0.839 | 0.768 | 0.708 | 0.841 | 0.284 | +18.8% |
| GB pixel | frozen | yes | **0.500 ± 0.011 (0.487–0.516)** | 0.885 ± 0.002 (0.882–0.888) | 0.667 | 0.927 | 0.521 | 0.867 | -43.8% |
| GB pixel | unsupervised (otsu) | yes | **0.697 ± 0.006 (0.687–0.704)** | 0.885 ± 0.002 (0.882–0.888) | 0.821 | 0.802 | 0.841 | 0.412 | +4.9% |
| GB pixel | unsupervised (gmm) | yes | **0.262 ± 0.025 (0.219–0.279)** | 0.885 ± 0.002 (0.882–0.888) | 0.415 | 0.262 | 1.000 | 0.006 | +284.6% |
| GB pixel | oracle | — *instrument* | **0.700 ± 0.004 (0.693–0.704)** | 0.885 ± 0.002 (0.882–0.888) | 0.823 | 0.814 | 0.833 | 0.431 | +2.3% |
| GB pixel (change-only) | frozen | yes | **0.523 ± 0.007 (0.513–0.532)** | 0.879 ± 0.003 (0.875–0.882) | 0.687 | 0.915 | 0.550 | 0.838 | -39.9% |
| GB pixel (change-only) | unsupervised (otsu) | yes | **0.670 ± 0.012 (0.653–0.687)** | 0.879 ± 0.003 (0.875–0.882) | 0.802 | 0.752 | 0.860 | 0.401 | +14.4% |
| GB pixel (change-only) | unsupervised (gmm) | yes | **0.234 ± 0.002 (0.231–0.237)** | 0.879 ± 0.003 (0.875–0.882) | 0.380 | 0.234 | 1.000 | 0.007 | +326.9% |
| GB pixel (change-only) | oracle | — *instrument* | **0.686 ± 0.005 (0.681–0.693)** | 0.879 ± 0.003 (0.875–0.882) | 0.813 | 0.805 | 0.823 | 0.484 | +2.3% |
| U-Net | frozen | yes | **0.489 ± 0.068 (0.394–0.556)** | 0.828 ± 0.015 (0.806–0.848) | 0.655 | 0.863 | 0.533 | 0.864 | -38.1% |
| U-Net | unsupervised (otsu) | yes | **0.611 ± 0.030 (0.566–0.643)** | 0.828 ± 0.015 (0.806–0.848) | 0.758 | 0.711 | 0.825 | 0.423 | +17.2% |
| U-Net | unsupervised (gmm) | yes | **0.335 ± 0.044 (0.265–0.371)** | 0.828 ± 0.015 (0.806–0.848) | 0.500 | 0.336 | 0.988 | 0.060 | +198.4% |
| U-Net | oracle | — *instrument* | **0.629 ± 0.018 (0.608–0.657)** | 0.828 ± 0.015 (0.806–0.848) | 0.772 | 0.729 | 0.821 | 0.458 | +12.5% |

AP is threshold-free, so for each seed it is constant down a method's four rows by construction: it describes the score, and the regime only chooses where to cut it. Comparing AP across methods is therefore the cleanest statement of which score carries more separable signal, independent of any operating point.

## Gaps against seed spread

Rule, fixed before any multi-seed result: two methods are separated on a metric only when their seed ranges [min, max] do not overlap; otherwise indistinguishable at this sample size. Gap = mean of the first method minus mean of the second.

| Event | Regime | Comparison | IoU gap | IoU | AP gap | AP |
|---|---|---|---|---|---|---|
| saumos | frozen | GB pixel − dNBR | +0.184 | separated | +0.077 | separated |
| saumos | unsupervised (otsu) | GB pixel − dNBR | +0.035 | indistinguishable | +0.077 | separated |
| saumos | unsupervised (gmm) | GB pixel − dNBR | -0.297 | separated | +0.077 | separated |
| saumos | oracle | GB pixel − dNBR | +0.089 | separated | +0.077 | separated |
| saumos | frozen | U-Net − GB pixel | -0.022 | separated | -0.013 | indistinguishable |
| saumos | frozen | U-Net − dNBR | +0.162 | separated | +0.064 | separated |
| saumos | unsupervised (otsu) | U-Net − GB pixel | -0.005 | indistinguishable | -0.013 | indistinguishable |
| saumos | unsupervised (otsu) | U-Net − dNBR | +0.029 | separated | +0.064 | separated |
| saumos | unsupervised (gmm) | U-Net − GB pixel | +0.109 | indistinguishable | -0.013 | indistinguishable |
| saumos | unsupervised (gmm) | U-Net − dNBR | -0.188 | separated | +0.064 | separated |
| saumos | oracle | U-Net − GB pixel | -0.032 | separated | -0.013 | indistinguishable |
| saumos | oracle | U-Net − dNBR | +0.058 | separated | +0.064 | separated |
| biscarrosse | frozen | GB pixel − dNBR | +0.082 | separated | +0.037 | separated |
| biscarrosse | unsupervised (otsu) | GB pixel − dNBR | +0.058 | separated | +0.037 | separated |
| biscarrosse | unsupervised (gmm) | GB pixel − dNBR | -0.094 | separated | +0.037 | separated |
| biscarrosse | oracle | GB pixel − dNBR | +0.043 | separated | +0.037 | separated |
| biscarrosse | frozen | U-Net − GB pixel | -0.006 | indistinguishable | +0.007 | indistinguishable |
| biscarrosse | frozen | U-Net − dNBR | +0.076 | separated | +0.044 | separated |
| biscarrosse | unsupervised (otsu) | U-Net − GB pixel | -0.030 | separated | +0.007 | indistinguishable |
| biscarrosse | unsupervised (otsu) | U-Net − dNBR | +0.028 | separated | +0.044 | separated |
| biscarrosse | unsupervised (gmm) | U-Net − GB pixel | +0.040 | indistinguishable | +0.007 | indistinguishable |
| biscarrosse | unsupervised (gmm) | U-Net − dNBR | -0.054 | separated | +0.044 | separated |
| biscarrosse | oracle | U-Net − GB pixel | -0.010 | indistinguishable | +0.007 | indistinguishable |
| biscarrosse | oracle | U-Net − dNBR | +0.033 | separated | +0.044 | separated |
| fontainebleau | frozen | GB pixel − dNBR | +0.308 | separated | +0.046 | separated |
| fontainebleau | unsupervised (otsu) | GB pixel − dNBR | +0.230 | separated | +0.046 | separated |
| fontainebleau | unsupervised (gmm) | GB pixel − dNBR | +0.004 | indistinguishable | +0.046 | separated |
| fontainebleau | oracle | GB pixel − dNBR | +0.076 | separated | +0.046 | separated |
| fontainebleau | frozen | U-Net − GB pixel | -0.011 | indistinguishable | -0.057 | separated |
| fontainebleau | frozen | U-Net − dNBR | +0.297 | separated | -0.012 | indistinguishable |
| fontainebleau | unsupervised (otsu) | U-Net − GB pixel | -0.086 | separated | -0.057 | separated |
| fontainebleau | unsupervised (otsu) | U-Net − dNBR | +0.145 | separated | -0.012 | indistinguishable |
| fontainebleau | unsupervised (gmm) | U-Net − GB pixel | +0.073 | indistinguishable | -0.057 | separated |
| fontainebleau | unsupervised (gmm) | U-Net − dNBR | +0.077 | separated | -0.012 | indistinguishable |
| fontainebleau | oracle | U-Net − GB pixel | -0.071 | separated | -0.057 | separated |
| fontainebleau | oracle | U-Net − dNBR | +0.005 | indistinguishable | -0.012 | indistinguishable |

Seed spread is variance of the training procedure on one training event. It is a floor on the uncertainty of a gap, not an interval: a gap separated here can still be within the variance across events, which two transfer events cannot measure.

No interval is published anywhere in this table. The rule is unchanged: the bootstrap resamples ~17.5 km blocks and counts only those containing burned pixels, and no evaluation domain in this project reaches 4. Each row carries its own reason in `regimes.json`. Adding methods and regimes multiplies the comparisons this table invites while adding no replicates to support them, and that is the honest caveat on every difference read off it.
