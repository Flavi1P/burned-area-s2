# burned-area-s2

Burned-area segmentation from Sentinel-2 pre/post pairs — a U-Net measured
against the dNBR spectral index under a deliberately symmetric evaluation
protocol.

> **Status — active development, started 21 August 2026.**
> The data chain, the label provenance and the evaluation design are in place.
> **No performance number is published yet, here or anywhere else.** The
> baseline runs first, the network second, and the table appears when both have
> been through the same evaluation code. Any figure you see below describes the
> *data*, never a result.

---

## The question

That a CNN outperforms a thresholded spectral index for burned-area mapping is
not news. It has been shown repeatedly, on datasets far larger than three
fires, and a reader who works in this field already knows it. **This project is
not a scientific contribution and does not claim to be one.**

The question it does ask:

> **Where and why does a per-pixel spectral index fail, and does spatial
> context repair precisely those failures?**

Two structural limits of dNBR motivate it.

**Its threshold does not transfer.** This is its documented operational
weakness: thresholds need local recalibration by biome, by season, by
conditions. A service deploying over a new territory has to redo that work. So
the useful question is not "who scores the highest IoU" but **which method
degrades least when the territory changes and nobody recalibrates** — which is
what experiment E2 measures.

**It is blind to context, by construction.** A per-pixel index cannot separate
a clearcut from a burn scar, a cloud shadow from a scar, dark soil from
scorched soil. These are not calibration limits but information limits: the
pixel does not contain the answer, the shape and the neighbourhood do. That is
the only place where a network with a wide receptive field has a *structural*
advantage.

**A scenario where the baseline wins is a good scenario.** It would show when
not to deploy deep learning. Nothing in the protocol is arranged to avoid it.

### Scope

**Is:** binary segmentation of burned area from Sentinel-2 pre/post pairs.
Input = bi-temporal multispectral tiles, output = per-pixel mask. Supervised,
external labels, canonical baseline.

**Is not:** fire *risk* prediction. Risk depends on weather, topography and
ignition — none of which is in the image. A CNN on Sentinel-2 alone would learn
land cover and call it risk.

---

## Data

Three separate Copernicus EMS Rapid Mapping activations from the 2026 French
fire season. All facts below were read from the EMS public API and from the
delineation packages themselves on 21 August 2026; they live in
[`config.yaml`](config.yaml) and nowhere else.

| Event | Activation | Fire date | EMS burned area | Fuel | Role |
|---|---|---|---|---|---|
| Saumos (Gironde, 33) | `EMSR899` | 22 Jul 2026 | **31 602 ha** | maritime pine | **train** |
| Biscarrosse (Landes, 40) | `EMSR902` | 23 Jul 2026 | **1 753 ha** | maritime pine | **test — size control** |
| Fontainebleau (Seine-et-Marne, 77) | `EMSR894` | 12 Jul 2026 | **832 ha** | mixed broadleaf-dominated high forest | **test — biome transfer** |

### Why three events and not two

With only Landes → Fontainebleau, a drop in performance would confound two
causes: the **change of biome** (pine plantation → broadleaf) and the **change
of scale** (a small fire has a far worse perimeter-to-area ratio, hence far
more ambiguous edge pixels). "The model does not transfer between biomes" is a
much heavier claim than "the model is worse on small fires", and a two-event
design cannot tell them apart.

Biscarrosse dissolves the confound: **same biome as Saumos, same size class as
Fontainebleau.**

- Drop on Biscarrosse **and** Fontainebleau → size effect.
- Drop on Fontainebleau **only** → biome transfer, demonstrated rather than
  assumed.

The three events span a **38 : 2 : 1** range of burned area, so the size
control is doing real work.

### Press figures are not the reference

Saumos was widely reported at ~42 000 ha. The EMS delineation that this project
evaluates against covers **31 602 ha**. The press figure aggregates several
fronts across several dates; the evaluation reference is the rasterised
polygon, and only that. The gap is stated here rather than quietly ignored.

### Imagery

Four bands, `B04 / B8A / B11 / B12`, two dates, **8 input channels**. B02/B03
are dominated by residual atmospheric signal and add no burn information at
this dataset size.

**Grid: 20 m.** B8A, B11 and B12 are native there; resampling to 10 m would
manufacture information that does not exist. The decisive argument is not
compute, though: **the EMS polygons are digitised at 1:15 000 with a geometric
RMSE of 2.4–20 m depending on the event.** An IoU computed on 10 m pixels would
mostly be measuring the analyst's hand.

Scenes come from the Element 84 earth-search STAC catalogue — public COGs, no
account, no token.

**Scene choice is driven by the label, not by the calendar or by the weather.**
The post-fire scene is the acquisition closest to the date of the imagery the
EMS analyst actually delineated on. An image/label pair that disagrees produces
a systematic error no metric recovers from; residual smoke only produces noise.
Cloud *screens* candidates — measured on the SCL over the event footprint, not
on the scene-level figure — but it never ranks them. That distinction is not
academic: over Fontainebleau the candidate reported at 20% scene cloud is 45%
clouded over the burn, while the 0.4% candidate is completely clear.

---

## Labels, and what the metric actually measures

Copernicus EMS *delineation* vectors (`observedEventA`), rasterised on the
Sentinel-2 grid. **Labels are never produced by thresholding dNBR** — dNBR is
the baseline under test, and using it as truth would make the whole evaluation
circular.

EMS production is not homogeneous, so the method is read from the data rather
than assumed. Every polygon carries a `det_method` attribute and every layer an
ISO 19115 lineage; `python -m src.data.ems` reads both.

| Event | EMS product | Status | Detection method | Digitised on | Analysis scale | MMU | RMSE |
|---|---|---|---|---|---|---|---|
| Saumos | `EMSR899_AOI01_DEL_MONIT02_v1` | final | Semi-automatic extraction | Legion, 1.2 m, 29 Jul | 1:15 000 | 225 m² | **2.4 m** |
| Biscarrosse | `EMSR902_AOI01_DEL_MONIT01_v2` | final | Semi-automatic extraction | Legion 2.0 m + Sentinel-2 10 m, 26 Jul | 1:15 000 | 5 625 m² | **20.0 m** |
| Fontainebleau | `EMSR894_AOI01_DEL_MONIT01_v1` | final | Semi-automatic extraction | GeoSat-2, 4.0 m, 16 Jul | 1:15 000 | 576 m² | **8.0 m** |

Three consequences, stated in advance:

1. **All three labels are semi-automatic extractions.** So "U-Net > dNBR" will
   not measure agreement with ground truth; it will measure the network's
   ability to reproduce an operator-assisted classifier's decision better than
   a global threshold does. The result stays valid — it simply does not say
   what a hurried reader would assume it says. To put a number on the size of
   that problem, the results table will carry a **dNBR ↔ EMS agreement** row.
   That is the circularity floor, and it costs five minutes to compute.

2. **They were extracted from VHR imagery (1.2–4 m), not from Sentinel-2.**
   That weakens the circularity considerably — the operator's classifier saw a
   different sensor at a different resolution than the baseline under test —
   but it does not remove it.

3. **The label geometry is not equally good across events.** Biscarrosse's
   RMSE is 20 m, exactly one pixel, against 2.4 m for Saumos. The size control
   therefore also carries the least precise boundary, which caps the IoU
   attainable on it regardless of method. This will be repeated next to its
   number rather than discovered afterwards.

Rasterising at 20 m costs almost nothing in area: 31 596 → 31 592 ha on Saumos,
831.3 → 831.4 ha on Fontainebleau. Whatever area error the models produce, it
will not be the grid's fault.

---

## Evaluation protocol

This is the part that matters, and it is built before the model.

**Split by event, never at random.** Randomly drawing neighbouring tiles leaks
adjacent pixels between train and test.

**Two tiling regimes.** Training tiles overlap. **Test tiles do not** —
overlap manufactures dependence between evaluation units and invalidates the
bootstrap.

**Negative tiles, two different policies.** In test, *nothing is filtered*: the
footprint is geometric (EMS area of interest + 2 km buffer) and every tile
inside it is scored, including entirely unburned ones. Filtering negatives
would inflate precision by an arbitrary factor and make the number comparable
to nothing. In training, roughly 2 negatives per positive, deliberately
over-sampling clearcuts and cloud shadows.

### Threshold symmetry — non-negotiable

Both methods emit a **continuous per-pixel score**: dNBR an index difference,
the U-Net a sigmoid probability. Neither emits a mask. The threshold does.

Carefully calibrating the dNBR threshold and leaving the U-Net at the default
0.5 compares an **optimised** operating point to an **arbitrary** one, and the
resulting bias has no known sign: the BCE term pushes toward under-predicting
the positive class under heavy imbalance, the Dice term pushes outputs toward
the extremes and moves the optimum elsewhere. A number whose bias you cannot
sign is indefensible even when it is unfavourable to you.

So: **calibration blocks are held out spatially inside Saumos, both thresholds
are fitted on exactly the same pixels by maximising F1, both are frozen, and
both are applied as-is to every test event. No recalibration on the target, on
either side.**

> Both methods have exactly one decision parameter, calibrated by the same
> procedure on the same pixels and frozen before test.

That sentence exists to defuse the reflex objection — *your network has 24
million parameters and your baseline has one*.

This matters more in E2 than in E1: under domain shift the probability
distribution moves, and 0.5 stops corresponding to anything observed. The
transfer metric, which is the reason this project exists, would then partly be
measuring uncontrolled calibration drift.

### The dNBR oracle

Alongside the **honest dNBR** (threshold calibrated on Saumos, applied as-is), a
**dNBR oracle** whose threshold is optimised *on the test event itself*. It is
not a competitor — it is impossible in production, since it requires already
knowing the answer. It is a decomposition instrument:

- **CNN ≈ oracle > honest** → the network's whole advantage is better
  calibration transfer. dNBR's problem is its threshold, not its information,
  and cheap local recalibration would do — no deep learning required.
- **CNN > oracle** → the network exploits information the index does not
  contain at any threshold. That is the spatial-context gain, demonstrated.

One extra row, all the machinery already exists, and it is the only analysis
here whose answer a practitioner does not already know.

### Metrics

IoU and F1 first. Precision and recall **reported separately** — the
operational cost is asymmetric. **Precision-recall curve** rather than ROC,
given the class imbalance. Confusion matrix. Total area error in hectares,
compared against **the rasterised EMS polygon**, never against press figures.

**Global accuracy is never reported.** It will sit near 98% and mean nothing.

### Uncertainty, and its declared hole

**No multi-seed spread.** Inter-seed standard deviation measures optimiser
variance, not the variance that matters, and printing it beside a metric drawn
from a single test event fakes precision.

**Spatial block bootstrap on ~17.5 km super-blocks** — beyond the
autocorrelation range of a burn scar. A 31 600 ha scar spans about 20 km, so two
neighbouring 5 km tiles are massively correlated, and using the tile as the
bootstrap unit would reproduce, more subtly, the leak the protocol exists to
remove.

The consequence is accepted and will be published as such: Saumos spans roughly
50 × 49 km and supports an interval; Biscarrosse (17.7 × 19.2 km) and
Fontainebleau (21.1 × 12.3 km) yield one or two independent blocks and
therefore **no publishable interval**. It will be written literally — *"IoU
Fontainebleau = 0.XX, interval not estimable — one independent spatial block"*.
An honest interval next to a declared hole beats two intervals of which one is
false. With N = 3 events, what can be reported across events is an observed
range, not a confidence interval.

### Experiments

| Exp. | Train | Test | What it measures |
|---|---|---|---|
| **E1** | Saumos (disjoint blocks, buffered) | Saumos (held-out blocks) | In-domain reference ceiling |
| **E2** | Saumos | Biscarrosse **and** Fontainebleau | Transfer. Comparing the two drops separates size from biome |
| **E3** | E1 fine-tuned on 0 / 10 / 25% of Fontainebleau | Fontainebleau, disjoint blocks | How many annotations does opening a new territory cost |

E3 leaks by default: adaptation and evaluation tiles come from the same event
on contiguous scars, so a random 10% draw reinstates exactly the spatial leak
the rest of the protocol removes. The adaptation tiles are spatially disjoint
and buffered. But Fontainebleau fits in one or two blocks, so a single
partition would leave a tiny, luck-dominated evaluation set — hence **5
alternating disjoint partitions, with mean and observed range**, published as a
band rather than a line.

The reverse split (train Fontainebleau → test Landes) is rejected: 832 ha of
positives cannot distinguish "the model does not generalise" from "the model
had nothing to learn from", and anyone doing the arithmetic will see it in
thirty seconds.

---

## A hypothesis stated before the results

The Landes de Gascogne is a production pine forest, dotted with clearcuts whose
NBR drop closely resembles a fire scar's. A per-pixel threshold cannot separate
them. A network with a wide receptive field should be able to, because
clearcuts are rectangular, aligned on the cadastral parcel grid, and
sharp-edged.

**This is the mechanism by which the CNN is expected to beat the baseline.** It
will be checked explicitly on the error maps, and the verdict will be written
down **including if the hypothesis is wrong**. A prediction made in advance and
then tested is worth more than three points of IoU.

---

## Data chain, end to end

Every panel below is produced by `python -m src.viz.quicklook --event <id>`,
driven entirely by `config.yaml`. False colour **SWIR-2 / NIR / Red
(B12 / B8A / B04)** at 20 m — three of the four bands the model actually
receives, so the figure shows the model's input rather than a prettier picture
of something else. Burn scars read dark maroon; healthy vegetation reads green.

**Saumos — the training event, 31 592 ha rasterised at 20 m:**

![Saumos before / after / EMS label](outputs/quicklook_saumos.png)

**Fontainebleau — the biome-transfer test event, 831 ha:**

![Fontainebleau before / after / EMS label](outputs/quicklook_fontainebleau.png)

Biscarrosse, the size control, is at
[`outputs/quicklook_biscarrosse.png`](outputs/quicklook_biscarrosse.png).

### One thing the chain already surfaced

Over Saumos, the Scene Classification Layer flags **26% of the label's pixels**
as cloud or cirrus on the chosen post-fire scene — a scene that is visibly
clear over the scar. Most of it is class 10 (thin cirrus) over the burn itself.
Separately, fresh scars land overwhelmingly in SCL classes 2 (dark) and 7
(unclassified): on Saumos and Fontainebleau, 78–89% of the pixels carrying
those classes fall inside the EMS label. SCL has no burned class at all.

Applying a blanket cloud/shadow mask would therefore delete a quarter of the
positive class from the training event. Cirrus will have to be handled
separately from opaque cloud and shadow. That SCL cannot itself decide what a
burn scar is, incidentally, is a small piece of evidence for the premise of
this whole project: the per-pixel spectral signature of a burn is ambiguous.

---

## Reproducing

```bash
conda env create -f environment.yml
conda activate burned-area-s2

python -m src.config                              # the plan, as loaded
python -m src.data.ems                            # EMS label provenance, all events
python -m src.data.stac --event saumos --list     # candidate scenes + footprint cloud
python -m src.viz.quicklook --event saumos        # before / after / label figure
pytest -q                                         # protocol guards
```

No credentials are needed for anything.

### Layout

```
config.yaml     the experimental plan — events, experiments, protocol.
                No event name, EMS code or date exists anywhere else.
src/data/       STAC access, EMS labels, target grids
src/model/      dataset, U-Net, training loop        (phase 3)
src/eval/       dNBR, thresholds, metrics, blocks, bootstrap  (phase 2)
src/viz/        figures
outputs/        versioned figures and tables, never rasters
tests/          protocol guards
```

`config.yaml` is not a bag of paths and hyper-parameters. The experimental plan
has to be readable *in the config file*, not reconstructed by reading
`train.py`, and the results table has to be a direct translation of its
`experiments:` block. The test suite enforces the parts of that which can be
enforced: a test event cannot appear in training under a by-event split, the
decision threshold cannot be calibrated on a test event, test tiles cannot
overlap, negative test tiles cannot be filtered, global accuracy cannot be
added to the metric list, the bootstrap block cannot shrink to a tile, and E3
cannot fall back to a single partition. The failure mode this project has to
defend against is not a crash — it is a plausible number produced by a
configuration that leaked.

---

## Known traps and accepted limits

- **Spatial leakage** — trap number one. Split by event, test tiling without
  overlap, buffered blocks, disjoint E3 partitions.
- **Normalisation leakage** — statistics computed on training data only.
- **Threshold leakage** — no recalibration on the target, on either side.
- **Partial circularity of the EMS label** — all three are semi-automatic
  extractions; see above.
- **Uneven label geometry** — 2.4 m RMSE on Saumos against 20 m on Biscarrosse.
- **Cloud shadows** look like burned surfaces and are a classic false-positive
  source; **cirrus flags overlap the scar itself** on Saumos.
- **Class imbalance** — BCE + Dice, never global accuracy.
- **Post-image / label-date offset**, particularly around the Fontainebleau
  reignitions: EMS stopped delineating on 16 July while the fire flared up into
  late July, so the post scene is deliberately held close to 16 July. Burn that
  the label does not contain would otherwise be scored as false positives.
- **N = 3 events** — observed range, not an inter-event confidence interval.

## Next steps, deliberately not started

Multi-class severity (graded dNBR), multi-temporal detection over Sentinel-2
series, Sentinel-1 fusion to see through cloud, a fourth event for temporal
validation (**Landiras 2022, `EMSR592` / `EMSR619`** — same massif, mature
labels), fire susceptibility with weather and topography, DAG orchestration,
deployment.

A project half-finished on five fronts proves the opposite of knowing where it
is going.

## Licence

MIT. Copernicus EMS products and Sentinel-2 data are used under their
respective terms.
