"""The U-Net through the baseline's own evaluation machinery.

Nothing in this module computes a metric, an interval or a table cell. Every one
of those comes from ``src.eval`` -- the same functions, called with the same
arguments, that produced the dNBR rows before the network existed. What this
module does is narrow: turn the probability raster into a score column, obtain a
threshold for it by the one calibration function, and hand both to code that has
never heard of a neural network.

**The threshold is the point.** ``src.eval.threshold.calibrate`` fits it by
maximising F1 on the calibration blocks of the training event -- the same
pixels, the same objective, the same function that gave the dNBR its 0.58. It is
then frozen and applied unchanged to every test event, with no recalibration on
the target. Leaving the network at 0.5 while the baseline got a fitted threshold
would compare an arbitrary operating point to an optimised one, and the sign of
that bias is not knowable: BCE under-predicts the positive class at 2%
prevalence, Dice pushes probabilities to the extremes, and the two move the
optimum in opposite directions. A number whose bias cannot be signed is
indefensible even when it is unfavourable.

**The U-Net gets no oracle row, and that asymmetry is deliberate.** The oracle
exists to bound what the *index* can do at its best threshold, so that any gain
the network shows can be split into calibration transfer and genuine context.
Giving the network the same refit would answer a question nobody asked and would
quietly restore the symmetry the oracle exists to break. The U-Net's honest row
is compared against the dNBR's oracle row: that is the demanding comparison, and
it is the one the design note asks for.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from ..config import Config, Experiment, load_config
from ..eval import baseline
from ..eval import threshold as thresholds_mod
from ..eval.metrics import evaluate as evaluate_domain
from .predict import read_probability

UNET = "unet"
METHOD = "U-Net"


def _experiments(cfg: Config, experiment_ids: list[str] | None) -> list[Experiment]:
    """The experiments a single trained model can be scored on.

    E3 fine-tunes per partition and produces its own curve; it is not a row in
    this table and is handled in phase 4.
    """
    return [
        cfg.experiment(eid)
        for eid in (experiment_ids or list(cfg.experiments))
        if cfg.experiment(eid).finetune_on is None
    ]


def calibrate_threshold(cfg: Config) -> thresholds_mod.Threshold:
    """One threshold for the network, from the function that gave the dNBR its own."""
    settings = cfg.evaluation["threshold"]
    event = cfg.event(settings["calibrated_on"])
    domain = baseline.build_domain(
        cfg, event, role=settings["block_role"], score=read_probability(cfg, event)
    )
    frozen = thresholds_mod.calibrate(
        domain.score,
        domain.truth,
        score_name=UNET,
        calibrated_on=(
            f"{domain.event.id}, {settings['block_role']} blocks "
            f"({', '.join(domain.blocks.blocks_with_role(settings['block_role']))}), "
            f"{domain.truth.size:,} usable px, {int(domain.truth.sum()):,} burned"
        ),
        objective=settings["objective"],
    )
    thresholds_mod.update(cfg, UNET, frozen)
    return frozen


def rows_for(cfg: Config, experiment_ids: list[str] | None = None) -> tuple[list[dict], dict]:
    """The U-Net's rows, produced by the baseline's own domain and metric code."""
    frozen = calibrate_threshold(cfg)
    forbidden = tuple(cfg.evaluation.get("forbidden_metrics", ()))

    rows: list[dict] = []
    for experiment in _experiments(cfg, experiment_ids):
        for event_id in experiment.test:
            event = cfg.event(event_id)
            domain = baseline.build_domain(
                cfg, event, score=read_probability(cfg, event)
            )
            result = evaluate_domain(
                domain.score,
                domain.truth,
                frozen.value,
                method=METHOD,
                event=event.id,
                experiment=experiment.id,
                threshold_source=f"frozen, {frozen.calibrated_on}",
                pixel_area_ha=domain.pixel_area_ha,
                footprint_pixels=domain.footprint_pixels,
            )
            rows.append(
                baseline.with_interval(cfg, domain, result).as_row(forbidden)
            )
    return rows, frozen.as_dict()


# --------------------------------------------------------------------------- #
# the merged table
# --------------------------------------------------------------------------- #

_METHOD_ORDER = {METHOD: 0, baseline.HONEST: 1, baseline.ORACLE: 2}


def run(cfg: Config, experiment_ids: list[str] | None = None) -> dict:
    """Recompute both methods in one pass and merge them into one table.

    The dNBR rows are recomputed here rather than read from the phase-2 file, so
    every number in the merged table comes out of one execution of one code path
    on one state of the data.
    """
    payload = baseline.run(cfg, experiment_ids)
    unet_rows, unet_threshold = rows_for(cfg, experiment_ids)

    payload["rows"] = sorted(
        payload["rows"] + unet_rows,
        key=lambda r: (r["experiment"], r["event"], _METHOD_ORDER.get(r["method"], 9)),
    )
    payload["thresholds"] = {"dnbr": payload["threshold"], "unet": unet_threshold}
    payload["title"] = "Results — U-Net against the dNBR baseline"
    payload["preamble"] = (
        f"Generated {payload['generated']} from `{payload['config']}` by "
        "`python -m src.model.evaluate`, which recomputes both methods in one pass. "
        "Every cell below -- both methods, every event -- comes from the same "
        "domain, metric, threshold and bootstrap code in `src/eval/`; the network "
        "contributes a probability raster and nothing else. Both decision "
        "thresholds were fitted by maximising F1 on the same held-out calibration "
        "blocks of the training event, frozen there, and applied unchanged to "
        "every test event."
    )
    return payload


def write(cfg: Config, payload: dict) -> tuple[Path, Path]:
    out = cfg.path_for("outputs")
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "results.json"
    json_path.write_text(
        json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
    )
    md_path = out / "results.md"
    md_path.write_text(baseline.render_markdown(payload), encoding="utf-8")
    return json_path, md_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--experiment", action="append", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    payload = run(cfg, args.experiment)
    json_path, md_path = write(cfg, payload)

    for name, entry in payload["thresholds"].items():
        print(
            f"threshold  : {name:5s} >= {entry['value']:.4f}  "
            f"({entry['objective']} = {entry['objective_value']:.3f} on "
            f"{entry['n_pixels']:,} calibration px)"
        )
    print(
        f"{'exp':4s} {'event':14s} {'method':13s} {'IoU':>7s} {'F1':>7s} "
        f"{'prec':>7s} {'rec':>7s} {'AP':>7s} {'area err':>9s}"
    )
    for row in payload["rows"]:
        print(
            f"{row['experiment']:4s} {row['event']:14s} {row['method']:13s} "
            f"{row['iou']:7.3f} {row['f1']:7.3f} {row['precision']:7.3f} "
            f"{row['recall']:7.3f} {row['average_precision']:7.3f} "
            f"{row['area_error_pct']:+8.1f}%"
        )
    print(f"written    : {json_path.name}, {md_path.name}")


if __name__ == "__main__":  # pragma: no cover
    main()
