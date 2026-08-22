"""Metrics, and the confusion counts everything else is rebuilt from.

Every number in the results table comes out of this module, for the baseline
and for the network alike. That is the point: "the two methods are measured by
the same code" is a claim the repository has to be able to back, and the way to
back it is to have exactly one implementation and to give it a continuous score
plus a threshold, whatever produced them.

Three deliberate choices.

**Counts first, ratios second.** A ``Confusion`` holds four integers and knows
how to add itself to another. Every ratio is a property derived from them. That
is what makes the spatial block bootstrap cheap -- resampling blocks is summing
their counts -- and it means a metric can never disagree with the confusion
matrix printed next to it.

**No global accuracy, enforced.** Under this class imbalance accuracy sits near
98% whatever the method does. ``config.yaml`` lists it as forbidden and
``as_row`` refuses to emit any forbidden metric, so the rule is executable
rather than a note in a README.

**Area error against the rasterised polygon.** Not against the press figure:
press hectares aggregate several fronts and several dates, and comparing a
prediction to them measures the difference between two definitions rather than
the quality of anything. The press number is carried separately, as a note.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


@dataclass(frozen=True)
class Confusion:
    """Pixel counts of one binary decision. The atom of every other number."""

    tp: int
    fp: int
    fn: int
    tn: int

    def __add__(self, other: "Confusion") -> "Confusion":
        return Confusion(
            self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.tn + other.tn
        )

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else float("nan")

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else float("nan")

    @property
    def f1(self) -> float:
        denominator = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denominator if denominator else float("nan")

    @property
    def iou(self) -> float:
        denominator = self.tp + self.fp + self.fn
        return self.tp / denominator if denominator else float("nan")

    @property
    def kappa(self) -> float:
        """Cohen's kappa -- agreement corrected for what chance alone gives.

        Carried because the circularity question (section 5) is about how much
        two maps agree, and raw agreement is meaningless at 6% prevalence.
        """
        n = self.n
        if n == 0:
            return float("nan")
        observed = (self.tp + self.tn) / n
        expected = (
            (self.tp + self.fp) * (self.tp + self.fn) + (self.fn + self.tn) * (self.fp + self.tn)
        ) / n**2
        return (observed - expected) / (1 - expected) if expected < 1 else float("nan")

    @property
    def prevalence(self) -> float:
        return (self.tp + self.fn) / self.n if self.n else float("nan")

    def as_dict(self) -> dict[str, int]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn}


def confusion(
    score: np.ndarray, truth: np.ndarray, threshold: float
) -> Confusion:
    """Confusion counts of thresholding a continuous score. 1-D inputs."""
    predicted = score >= threshold
    truth = truth.astype(bool)
    tp = int(np.count_nonzero(predicted & truth))
    fp = int(np.count_nonzero(predicted & ~truth))
    fn = int(np.count_nonzero(~predicted & truth))
    return Confusion(tp=tp, fp=fp, fn=fn, tn=int(truth.size) - tp - fp - fn)


def confusion_by_group(
    score: np.ndarray, truth: np.ndarray, threshold: float, group: np.ndarray, n_groups: int
) -> list[Confusion]:
    """The same counts, split by group index -- the input of the block bootstrap."""
    predicted = score >= threshold
    truth = truth.astype(bool)
    counts = {
        key: np.bincount(group[mask], minlength=n_groups)
        for key, mask in (
            ("tp", predicted & truth),
            ("fp", predicted & ~truth),
            ("fn", ~predicted & truth),
            ("tn", ~predicted & ~truth),
        )
    }
    return [
        Confusion(
            tp=int(counts["tp"][i]),
            fp=int(counts["fp"][i]),
            fn=int(counts["fn"][i]),
            tn=int(counts["tn"][i]),
        )
        for i in range(n_groups)
    ]


def pr_curve(
    score: np.ndarray, truth: np.ndarray, max_points: int = 400
) -> dict[str, list[float]]:
    """A precision-recall curve, thinned to something a figure can carry.

    PR rather than ROC: at 6% prevalence the false-positive rate hides an order
    of magnitude of false positives behind a flat-looking curve. PR is also
    threshold-free, which is what makes it the honest companion of a table
    whose every other line depends on one calibrated threshold.
    """
    precision, recall, thresholds = precision_recall_curve(truth.astype(bool), score)
    # precision_recall_curve appends the (recall=0, precision=1) endpoint, which
    # has no threshold; drop it so the three arrays stay aligned.
    precision, recall = precision[:-1], recall[:-1]
    if precision.size > max_points:
        keep = np.linspace(0, precision.size - 1, max_points).round().astype(int)
        precision, recall, thresholds = precision[keep], recall[keep], thresholds[keep]
    return {
        "precision": [round(float(v), 5) for v in precision],
        "recall": [round(float(v), 5) for v in recall],
        "threshold": [round(float(v), 5) for v in thresholds],
    }


@dataclass
class Result:
    """One (method, evaluation domain) cell of the results table."""

    method: str
    event: str
    experiment: str
    threshold: float
    threshold_source: str
    counts: Confusion
    pixel_area_ha: float
    label_area_ha: float  # EMS polygon rasterised, inside this domain
    average_precision: float
    domain_pixels: int  # usable pixels this domain was scored on
    footprint_pixels: int  # pixels in the geometric footprint, before masking
    curve: dict[str, list[float]] | None = None
    interval: dict[str, object] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def predicted_area_ha(self) -> float:
        return (self.counts.tp + self.counts.fp) * self.pixel_area_ha

    @property
    def area_error_ha(self) -> float:
        return self.predicted_area_ha - self.label_area_ha

    @property
    def area_error_pct(self) -> float:
        return (
            100.0 * self.area_error_ha / self.label_area_ha
            if self.label_area_ha
            else float("nan")
        )

    def as_row(self, forbidden: tuple[str, ...] = ()) -> dict[str, object]:
        row = {
            "experiment": self.experiment,
            "event": self.event,
            "method": self.method,
            "threshold": round(self.threshold, 5),
            "threshold_source": self.threshold_source,
            "iou": self.counts.iou,
            "f1": self.counts.f1,
            "precision": self.counts.precision,
            "recall": self.counts.recall,
            "average_precision": self.average_precision,
            "kappa": self.counts.kappa,
            "predicted_area_ha": self.predicted_area_ha,
            "label_area_ha": self.label_area_ha,
            "area_error_ha": self.area_error_ha,
            "area_error_pct": self.area_error_pct,
            "prevalence": self.counts.prevalence,
            "domain_pixels": self.domain_pixels,
            "footprint_pixels": self.footprint_pixels,
            "confusion": self.counts.as_dict(),
            "interval": self.interval,
            "pr_curve": self.curve,
            "notes": self.notes,
        }
        offenders = sorted(set(forbidden) & set(row))
        if offenders:
            raise ValueError(
                f"config.yaml forbids reporting {', '.join(offenders)}, and this row "
                "carries it. Under this class imbalance it would be near 98% for "
                "every method and would mean nothing."
            )
        return row


def evaluate(
    score: np.ndarray,
    truth: np.ndarray,
    threshold: float,
    *,
    method: str,
    event: str,
    experiment: str,
    threshold_source: str,
    pixel_area_ha: float,
    footprint_pixels: int,
    with_curve: bool = True,
) -> Result:
    """Score one method on one evaluation domain. Inputs are 1-D and already
    restricted to the usable pixels of that domain."""
    truth = truth.astype(bool)
    counts = confusion(score, truth, threshold)
    return Result(
        method=method,
        event=event,
        experiment=experiment,
        threshold=float(threshold),
        threshold_source=threshold_source,
        counts=counts,
        pixel_area_ha=pixel_area_ha,
        label_area_ha=float(truth.sum()) * pixel_area_ha,
        average_precision=float(average_precision_score(truth, score))
        if truth.any()
        else float("nan"),
        domain_pixels=int(truth.size),
        footprint_pixels=int(footprint_pixels),
        curve=pr_curve(score, truth) if with_curve and truth.any() else None,
    )
