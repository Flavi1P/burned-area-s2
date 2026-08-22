"""Spatial block bootstrap, and the case where it must refuse to answer.

The resampling unit is the super-block built by ``src.eval.blocks``, which is
also where the argument for its size lives. This module only consumes it, and
adds two things of its own.

**The refusal.** The honest consequence of that block size, published rather
than hidden and larger than the design anticipated: no evaluation domain in
this project supports an interval. The two transfer events hold one block each.
The training event holds nine, four of them held out for test -- but its scar
is a single blob and only one of those four carries a meaningful amount of
burned area, so an interval over them would describe which block the draw
happened to pick. Every point estimate stands; every interval cell says why it
is empty, in words, rather than being left suspiciously blank.
``min_blocks_for_interval`` in ``config.yaml`` sets where the refusal kicks in.

**A ratio of sums, not a mean of ratios.** Blocks are resampled with
replacement and the metric is recomputed from the summed confusion counts of
the draw. A mean of per-block IoUs would weight a block holding forty burned
pixels the same as one holding forty thousand, and would drift away from the
point estimate it is supposed to bracket.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import Confusion

METRICS = ("iou", "f1", "precision", "recall")


@dataclass(frozen=True)
class Interval:
    """A percentile interval, or a stated refusal to produce one."""

    metric: str
    point: float
    n_blocks: int
    n_burned_blocks: int = 0
    low: float | None = None
    high: float | None = None
    level: float = 0.95
    iterations: int = 0
    reason: str | None = None  # set exactly when the interval is missing

    @property
    def estimable(self) -> bool:
        return self.low is not None

    def as_dict(self) -> dict:
        payload = {
            "metric": self.metric,
            "point": self.point,
            "n_blocks": self.n_blocks,
            "n_burned_blocks": self.n_burned_blocks,
            "estimable": self.estimable,
        }
        if self.estimable:
            payload |= {
                "low": self.low,
                "high": self.high,
                "level": self.level,
                "iterations": self.iterations,
            }
        else:
            payload["reason"] = self.reason
        return payload

    def render(self, digits: int = 3) -> str:
        """The cell as it appears in the table, refusal included."""
        if self.estimable:
            return (
                f"{self.point:.{digits}f} "
                f"[{self.low:.{digits}f}, {self.high:.{digits}f}]"
            )
        return f"{self.point:.{digits}f} (interval not estimable)"


def _value(counts: Confusion, metric: str) -> float:
    return getattr(counts, metric)


def block_bootstrap(
    blocks: list[Confusion],
    *,
    metrics: tuple[str, ...] = METRICS,
    iterations: int = 1000,
    min_blocks: int = 4,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, Interval]:
    """Percentile intervals over spatial blocks, or a stated refusal.

    ``blocks`` holds one confusion per independent spatial block of the
    evaluation domain. Empty blocks are dropped first: a block the cloud mask
    emptied is not an independent observation, it is nothing, and leaving it in
    would let a draw of pure zeros land in the distribution.

    The count that decides whether an interval exists is the number of blocks
    that actually **contain burned pixels**, not the number of blocks. Every
    metric here is a statement about burned area; a block holding nothing but
    background contributes to precision but is not a replicate of that
    statement, and counting it would let a domain whose entire scar sits in one
    block advertise four independent observations. Those background blocks stay
    in the resampling -- their false positives are real -- they simply do not
    buy the right to publish an interval.
    """
    populated = [b for b in blocks if b.n > 0]
    total = sum(populated, Confusion(0, 0, 0, 0))
    n = len(populated)
    n_burned = sum(1 for b in populated if b.tp + b.fn > 0)

    if n_burned < min_blocks:
        reason = (
            f"{n_burned} of the {n} spatial blocks in this evaluation domain contain "
            f"burned pixels, below the {min_blocks} the protocol requires. A "
            "percentile interval built on so few would describe which block the draw "
            "happened to pick, not the uncertainty of the metric."
        )
        return {
            m: Interval(
                metric=m,
                point=_value(total, m),
                n_blocks=n,
                n_burned_blocks=n_burned,
                reason=reason,
            )
            for m in metrics
        }

    rng = np.random.default_rng(seed)
    draws = {m: np.empty(iterations, dtype="float64") for m in metrics}
    for i in range(iterations):
        picks = rng.integers(0, n, size=n)
        drawn = sum((populated[p] for p in picks), Confusion(0, 0, 0, 0))
        for m in metrics:
            draws[m][i] = _value(drawn, m)

    lo_q, hi_q = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    out = {}
    for m in metrics:
        values = draws[m][np.isfinite(draws[m])]
        low, high = np.percentile(values, [lo_q, hi_q])
        out[m] = Interval(
            metric=m,
            point=_value(total, m),
            n_blocks=n,
            n_burned_blocks=n_burned,
            low=float(low),
            high=float(high),
            level=level,
            iterations=iterations,
        )
    return out
