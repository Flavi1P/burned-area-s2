"""Threshold calibration -- one function, both methods.

The dNBR produces a difference of indices, the U-Net a sigmoid probability.
Neither produces a mask; the threshold does. So calibrating the baseline's
threshold carefully and leaving the network at the default 0.5 would compare an
optimised operating point against an arbitrary one, and the resulting bias does
not even have a known sign: the BCE term pushes a model to under-predict the
positive class under strong imbalance, while the Dice term pushes outputs to
the extremes and moves the optimum somewhere else again. A number whose bias
nobody can sign is indefensible even when it is unfavourable.

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

from ..config import Config

OBJECTIVES = ("f1",)


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

    def as_dict(self) -> dict:
        return asdict(self)


def calibrate(
    score: np.ndarray,
    truth: np.ndarray,
    *,
    score_name: str,
    calibrated_on: str,
    objective: str = "f1",
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
    """
    existing = load(cfg) if thresholds_path(cfg).exists() else {}
    existing[name] = threshold
    return save(cfg, existing)
