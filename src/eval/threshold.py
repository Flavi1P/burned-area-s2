"""Threshold calibration -- one function, both methods.

The dNBR produces a difference of indices, the U-Net a sigmoid probability.
Neither produces a mask; the threshold does. So calibrating the baseline's
threshold carefully and leaving the network at the default 0.5 would compare an
optimised operating point against an arbitrary one, and the resulting bias does
not even have a known sign: the BCE term pushes a model to under-predict the
positive class under strong imbalance, while the Dice term pushes outputs to
the extremes and moves the optimum somewhere else again. A number whose bias
nobody can sign is indefensible even when it is unfavourable.

**Three regimes, and the difference between them is who is allowed to look at
what.** ``calibrate`` on the training event's calibration blocks is *frozen*:
it consults labels, never the target's. ``unsupervised`` reads the shape of the
score's own histogram on the target event and consults no labels at all, which
is why it is deployable where the oracle is not. ``calibrate`` pointed at a
test event is the *oracle*: it consults the target's labels and is therefore
not deployable at all. A regime is applied to every method or to none -- the
same symmetry that made one calibration function serve both sides in the first
place.

Hence: one function, taking a continuous score and a truth mask, used for the
dNBR and for the network, on the same held-out calibration blocks of the
training event, frozen before any test event is touched.

The same function does duty as the **oracle**, by being pointed at a test event
instead. That is not a competitor -- it is impossible in production, since it
requires already knowing the answer -- it is an instrument. If the network
lands near the oracle but above the honest baseline, its advantage is better
calibration transfer and a cheap local recalibration would have bought the same
thing. If it beats the oracle, it is using information no threshold on the
index can reach. That is the only question in this project whose answer a
practitioner does not know in advance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.mixture import GaussianMixture

from ..config import Config

OBJECTIVES = ("f1",)
# Label-free estimators. They do not maximise a metric -- there is no metric to
# maximise without labels -- so their `objective_value` carries the estimator's
# own separability criterion, which is not comparable to an F1 or to each other.
ESTIMATORS = ("otsu", "gmm")

FROZEN, UNSUPERVISED, ORACLE = "frozen", "unsupervised", "oracle"


@dataclass(frozen=True)
class Threshold:
    """A calibrated decision threshold and everything needed to audit it."""

    value: float
    objective: str
    score_name: str
    calibrated_on: str  # human-readable description of the pixels used
    n_pixels: int
    n_positive: int
    objective_value: float  # the objective attained at `value`, on those pixels
    frozen: bool = True
    # Which regime produced it, and whether producing it needed labels. The
    # second is the field that decides what a row is allowed to claim: a
    # threshold fitted with the target's own labels is an instrument, never a
    # deployable result, and `regimes.py` reads this rather than parsing a name.
    regime: str = FROZEN
    uses_labels: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


def calibrate(
    score: np.ndarray,
    truth: np.ndarray,
    *,
    score_name: str,
    calibrated_on: str,
    objective: str = "f1",
    regime: str = FROZEN,
) -> Threshold:
    """The threshold maximising ``objective`` over a set of scored pixels.

    Exact rather than a grid search: ``precision_recall_curve`` enumerates
    every distinct operating point the data actually contains, so the maximum
    is the true one and does not depend on a grid resolution nobody would think
    to report.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown calibration objective {objective!r}")
    truth = np.asarray(truth).astype(bool)
    score = np.asarray(score, dtype="float64")
    if score.shape != truth.shape:
        raise ValueError(f"score {score.shape} and truth {truth.shape} disagree")
    if not truth.any():
        raise ValueError(
            f"{calibrated_on}: no positive pixel to calibrate on. A threshold fitted "
            "on pure background is not a threshold."
        )

    precision, recall, thresholds = precision_recall_curve(truth, score)
    precision, recall = precision[:-1], recall[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = 2 * precision * recall / (precision + recall)
    f1 = np.nan_to_num(f1, nan=0.0)
    best = int(np.argmax(f1))

    return Threshold(
        value=float(thresholds[best]),
        objective=objective,
        score_name=score_name,
        calibrated_on=calibrated_on,
        n_pixels=int(truth.size),
        n_positive=int(truth.sum()),
        objective_value=float(f1[best]),
        frozen=regime == FROZEN,
        regime=regime,
        uses_labels=True,
    )


# --------------------------------------------------------------------------- #
# the label-free regime
# --------------------------------------------------------------------------- #


def _otsu(values: np.ndarray, bins: int, clip: tuple[float, float]) -> tuple[float, float]:
    """Otsu's threshold and the between-class variance it attains.

    The tails are clipped before the histogram is built: a handful of pixels at
    dNBR 1.9 would otherwise pull the optimal split towards them and describe an
    outlier rather than the burn.
    """
    lo, hi = np.percentile(values, list(clip))
    counts, edges = np.histogram(np.clip(values, lo, hi), bins=bins)
    weight = counts / counts.sum()
    centres = (edges[:-1] + edges[1:]) / 2

    w0 = np.cumsum(weight)
    w1 = 1.0 - w0
    cumulative = np.cumsum(weight * centres)
    total = cumulative[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        m0 = cumulative / np.maximum(w0, 1e-12)
        m1 = (total - cumulative) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (m0 - m1) ** 2
    best = int(np.nanargmax(between))
    variance = float(np.var(values))
    # Normalised so it reads as "share of the scene's variance the split
    # explains" -- a separability figure, not an F1, and labelled as such.
    return float(centres[best]), float(between[best] / variance) if variance else 0.0


def _gmm(
    values: np.ndarray, components: int, max_samples: int, seed: int
) -> tuple[float, float]:
    """Where a ``components``-way Gaussian mixture stops calling a pixel background.

    The threshold is the lowest score at which the highest-mean component wins
    the posterior. Two Gaussians are a poor description of a burn histogram --
    the background is neither Gaussian nor unimodal -- and this estimator is
    kept precisely so that the reader can see that, rather than being shown only
    the estimator that happened to work.
    """
    rng = np.random.default_rng(seed)
    sample = values
    if sample.size > max_samples:
        sample = rng.choice(sample, size=max_samples, replace=False)
    mixture = GaussianMixture(components, random_state=seed).fit(sample.reshape(-1, 1))

    burned = int(np.argmax(mixture.means_.ravel()))
    lo, hi = np.percentile(values, [50, 99.9])
    grid = np.linspace(lo, hi, 4000).reshape(-1, 1)
    posterior = mixture.predict_proba(grid)[:, burned]
    above = np.flatnonzero(posterior >= 0.5)
    value = float(grid[above[0], 0]) if above.size else float(hi)
    return value, float(mixture.weights_[burned])


def unsupervised(
    score: np.ndarray,
    *,
    score_name: str,
    selected_on: str,
    estimator: str,
    settings: dict,
    seed: int,
) -> Threshold:
    """A threshold read off the score's own histogram, with no labels at all.

    This is the regime the oracle is not: it looks at the target event, which
    the frozen threshold may not, but it looks only at the distribution of the
    scores, which is available the moment the imagery lands. Everything it needs
    exists in production, so a row produced by it is a row somebody could
    actually deploy.

    It is not free. It assumes the burned mode is present and separable in the
    score's histogram, which fails on a scene with almost no burn in it -- there
    the estimator will happily split the background against itself. The
    assumption is the price of not needing labels, and it belongs in the table
    next to the number, not in a footnote.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(
            f"unknown unsupervised estimator {estimator!r}; config.yaml declares "
            f"{', '.join(ESTIMATORS)}"
        )
    values = np.asarray(score, dtype="float64")
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"{selected_on}: no finite score to read a threshold from")

    if estimator == "otsu":
        value, criterion = _otsu(
            values, int(settings["bins"]), tuple(settings["clip_percentiles"])
        )
    else:
        value, criterion = _gmm(
            values,
            int(settings["gmm_components"]),
            int(settings["max_samples"]),
            seed,
        )

    return Threshold(
        value=value,
        objective=estimator,
        score_name=score_name,
        calibrated_on=selected_on,
        n_pixels=int(values.size),
        # No labels were consulted, so there is no count of positives to report.
        # The honest entry is the number of pixels the threshold *predicts*
        # positive, and it is never to be read as a count of burned ground.
        n_positive=int((values >= value).sum()),
        objective_value=criterion,
        frozen=False,
        regime=UNSUPERVISED,
        uses_labels=False,
    )


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #


def thresholds_path(cfg: Config) -> Path:
    return cfg.path_for("outputs", "thresholds.json")


def save(cfg: Config, thresholds: dict[str, Threshold]) -> Path:
    """Freeze the calibrated thresholds to a versioned file.

    They live in ``outputs/`` and are committed, so the value applied to every
    test event is the value a reader can see, and re-running a test cannot
    quietly move it.
    """
    dest = thresholds_path(cfg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "objective": cfg.evaluation["threshold"]["objective"],
        "calibrated_on_event": cfg.evaluation["threshold"]["calibrated_on"],
        "block_role": cfg.evaluation["threshold"]["block_role"],
        "recalibrate_on_target": cfg.evaluation["threshold"]["recalibrate_on_target"],
        "thresholds": {name: t.as_dict() for name, t in thresholds.items()},
    }
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest


def load(cfg: Config) -> dict[str, Threshold]:
    payload = json.loads(thresholds_path(cfg).read_text(encoding="utf-8"))
    return {name: Threshold(**t) for name, t in payload["thresholds"].items()}


def update(cfg: Config, name: str, threshold: Threshold) -> Path:
    """Add or replace one entry, keeping the others frozen as they are.

    This is how the U-Net's threshold joins the baseline's: same function, same
    pixels, same file, written at a different time.

    Only frozen thresholds may enter this file. An unsupervised threshold is a
    property of one target event and an oracle threshold is fitted on that
    event's labels; writing either here would put a per-event number in the
    file whose whole purpose is to show that one value was applied unchanged
    everywhere. The guard makes that mistake impossible rather than unlikely.
    """
    if threshold.regime != FROZEN or not threshold.frozen:
        raise ValueError(
            f"threshold {name!r} was produced by the {threshold.regime!r} regime and "
            f"cannot be frozen to {thresholds_path(cfg).name}: that file records the "
            "single value applied unchanged to every test event. Per-event "
            "thresholds belong in the results rows, beside the row they produced."
        )
    existing = load(cfg) if thresholds_path(cfg).exists() else {}
    existing[name] = threshold
    return save(cfg, existing)
