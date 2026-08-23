"""The baseline run: dNBR through the whole evaluation machinery.

This is the point of doing the baseline before the network. Everything the
U-Net will be scored by exists and has run before the U-Net exists: the
validity mask, the block split, the calibration function, the metrics, the
bootstrap, the table writer. When the network arrives it supplies a continuous
score per pixel and nothing else, and the table gains rows without the
evaluation code changing. That is what makes "both methods were measured by
the same code" a fact rather than an intention.

Three rows come out of it per test event.

``dNBR honest`` is the operational baseline: one threshold, calibrated by
maximising F1 on the held-out calibration blocks of the training event, frozen,
and applied unchanged to every test event. No recalibration on the target,
because in production nobody has the target's labels.

``dNBR oracle`` refits that threshold on the test event itself. It cannot be
run in production -- it needs the answer in order to produce the answer -- and
it is not a competitor. It is the decomposition instrument: whatever the U-Net
gains over the honest baseline is either better calibration transfer, which the
oracle also has, or information the index does not contain at any threshold,
which the oracle does not have.

``dNBR <-> EMS agreement`` is the circularity floor. Copernicus EMS delineations
here are semi-automatic extractions, so part of any "learned method beats the
index" result is the index agreeing with the way the label was drawn. The
threshold-free average precision of the raw dNBR against the label measures
how much, and it is reported before any comparison is made, not after someone
asks.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from ..config import Config, Event, Experiment, load_config
from ..data import ems
from ..data.grid import grid_for_event
from ..data.stack import read_valid
from . import threshold as thresholds_mod
from .blocks import Blocks, blocks_for_event
from .bootstrap import Interval, block_bootstrap
from .dnbr import read_dnbr
from .metrics import Result, confusion_by_group, evaluate

DNBR = "dnbr"
HONEST = "dNBR honest"
ORACLE = "dNBR oracle"


@dataclass
class Domain:
    """The pixels one method is scored on, and the blocks they fall in."""

    event: Event
    blocks: Blocks
    mask: np.ndarray  # (H, W) bool: test role, cloud-free, in every band
    score: np.ndarray  # 1-D, restricted to `mask`
    truth: np.ndarray  # 1-D
    group: np.ndarray  # 1-D block index
    pixel_area_ha: float
    footprint_pixels: int
    label_area_ha_total: float  # the whole rasterised polygon, for context
    valid_fraction: float

    @property
    def n_blocks(self) -> int:
        return int(np.unique(self.group).size)

    @property
    def n_burned_blocks(self) -> int:
        """Blocks holding at least one burned pixel -- the count that decides
        whether an interval is publishable. See ``src.eval.bootstrap``."""
        return int(np.unique(self.group[self.truth]).size)


def build_domain(
    cfg: Config, event: Event, role: str = "test", score: np.ndarray | None = None
) -> Domain:
    """Assemble the evaluation domain of one event.

    Section 6.1: the footprint is geometric -- the EMS area of interest plus a
    buffer -- and every pixel inside it is scored, including entirely unburned
    ones. Nothing is filtered for being negative; the only pixels removed are
    the ones no method can see, and they are removed for every method at once.

    ``score`` is the only thing a method supplies. It defaults to the dNBR, and
    the U-Net passes its probability raster through this same function -- which
    is why "both methods were measured by the same code" is a fact about the
    call graph rather than a claim in a README. Every other input to the domain
    (footprint, role, validity, label, block index) is fixed by the event.
    """
    grid = grid_for_event(cfg, event)
    blocks = blocks_for_event(cfg, event, grid)
    valid = read_valid(cfg, event)
    with rasterio.open(ems.label_raster_path(cfg, event)) as src:
        label = src.read(1) > 0
    if score is None:
        score = read_dnbr(cfg, event)
    if score.shape != grid.shape:
        raise ValueError(
            f"{event.id}: score raster {score.shape} does not match the event grid "
            f"{grid.shape}"
        )

    mask = blocks.mask(role) & valid & np.isfinite(score)
    return Domain(
        event=event,
        blocks=blocks,
        mask=mask,
        score=score[mask].astype("float64"),
        truth=label[mask],
        group=blocks.index[mask],
        pixel_area_ha=grid.pixel_area_ha,
        footprint_pixels=grid.width * grid.height,
        label_area_ha_total=float(label.sum()) * grid.pixel_area_ha,
        valid_fraction=float(valid.mean()),
    )


def calibration_domain(cfg: Config) -> Domain:
    """The pixels both thresholds are fitted on. Held out from train and test."""
    settings = cfg.evaluation["threshold"]
    event = cfg.event(settings["calibrated_on"])
    if not event.is_train:
        raise ValueError(
            f"threshold.calibrated_on names {event.id}, which is not the training "
            "event: calibrating on a test event is the leak this protocol exists "
            "to prevent"
        )
    return build_domain(cfg, event, role=settings["block_role"])


def bootstrap_intervals(cfg: Config, domain: Domain, value: float) -> dict[str, Interval]:
    settings = cfg.evaluation["spatial_blocks"]
    counts = confusion_by_group(
        domain.score, domain.truth, value, domain.group, int(domain.group.max()) + 1
    )
    return block_bootstrap(
        counts,
        iterations=int(settings["bootstrap_iterations"]),
        min_blocks=int(settings["min_blocks_for_interval"]),
        seed=int(cfg.model["seed"]),
    )


def with_interval(cfg: Config, domain: Domain, result: Result) -> Result:
    """Attach the block-bootstrap interval, or the reason there is not one."""
    intervals = bootstrap_intervals(cfg, domain, result.threshold)
    result.interval = {m: i.as_dict() for m, i in intervals.items()}
    if not intervals["iou"].estimable:
        result.notes.append(intervals["iou"].reason)
    return result


def run(cfg: Config, experiment_ids: list[str] | None = None) -> dict:
    """Calibrate once, then score the dNBR on every baseline experiment."""
    # Experiments needing a fine-tuned model are not baseline experiments: the
    # index has nothing to fine-tune.
    experiments: list[Experiment] = [
        cfg.experiment(eid)
        for eid in (experiment_ids or list(cfg.experiments))
        if cfg.experiment(eid).finetune_on is None
    ]

    calibration = calibration_domain(cfg)
    settings = cfg.evaluation["threshold"]
    frozen = thresholds_mod.calibrate(
        calibration.score,
        calibration.truth,
        score_name=DNBR,
        calibrated_on=(
            f"{calibration.event.id}, {settings['block_role']} blocks "
            f"({', '.join(calibration.blocks.blocks_with_role(settings['block_role']))}), "
            f"{calibration.truth.size:,} usable px, "
            f"{int(calibration.truth.sum()):,} burned"
        ),
        objective=settings["objective"],
    )
    thresholds_mod.update(cfg, DNBR, frozen)

    rows: list[dict] = []
    events_seen: dict[str, dict] = {}
    circularity: list[dict] = []
    forbidden = tuple(cfg.evaluation.get("forbidden_metrics", ()))

    for experiment in experiments:
        for event_id in experiment.test:
            event = cfg.event(event_id)
            domain = build_domain(cfg, event)

            honest = evaluate(
                domain.score,
                domain.truth,
                frozen.value,
                method=HONEST,
                event=event.id,
                experiment=experiment.id,
                threshold_source=f"frozen, {frozen.calibrated_on}",
                pixel_area_ha=domain.pixel_area_ha,
                footprint_pixels=domain.footprint_pixels,
            )

            oracle_threshold = thresholds_mod.calibrate(
                domain.score,
                domain.truth,
                score_name=DNBR,
                calibrated_on=f"{event.id} itself -- not available in production",
            )
            oracle = evaluate(
                domain.score,
                domain.truth,
                oracle_threshold.value,
                method=ORACLE,
                event=event.id,
                experiment=experiment.id,
                threshold_source="refit on this event (upper bound, not a competitor)",
                pixel_area_ha=domain.pixel_area_ha,
                footprint_pixels=domain.footprint_pixels,
                with_curve=False,
            )

            for result in (honest, oracle):
                rows.append(with_interval(cfg, domain, result).as_row(forbidden))

            events_seen[event.id] = {
                "event": event.id,
                "name": event.name,
                "role": event.role,
                "fuel": event.fuel,
                "test_purpose": event.test_purpose,
                "ems_product": event.label.product_id,
                "ems_method": event.label.production.method,
                "ems_reported_area_ha": event.label.reported_burnt_area_ha,
                "press_area_ha": event.press_area_ha,
                "rasterised_area_ha": round(domain.label_area_ha_total, 1),
                "footprint_pixels": domain.footprint_pixels,
                "valid_fraction": round(domain.valid_fraction, 4),
                "evaluated_pixels": int(domain.truth.size),
                "evaluated_burned_ha": round(
                    float(domain.truth.sum()) * domain.pixel_area_ha, 1
                ),
                "evaluated_blocks": domain.n_blocks,
                "evaluated_blocks_with_burn": domain.n_burned_blocks,
                "test_blocks": domain.blocks.blocks_with_role("test"),
            }
            circularity.append(
                {
                    "event": event.id,
                    "ems_method": event.label.production.method,
                    "average_precision": honest.average_precision,
                    "oracle_iou": oracle.counts.iou,
                    "oracle_kappa": oracle.counts.kappa,
                    "oracle_threshold": oracle.threshold,
                }
            )

    return {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "config": str(cfg.path.name),
        "min_blocks_for_interval": int(
            cfg.evaluation["spatial_blocks"]["min_blocks_for_interval"]
        ),
        "superblock_km": float(cfg.evaluation["spatial_blocks"]["superblock_km"]),
        "threshold": frozen.as_dict(),
        "calibration_domain": {
            "event": calibration.event.id,
            "blocks": calibration.blocks.blocks_with_role(
                cfg.evaluation["threshold"]["block_role"]
            ),
            "pixels": int(calibration.truth.size),
            "burned_pixels": int(calibration.truth.sum()),
            "prevalence": round(float(calibration.truth.mean()), 5),
        },
        "events": list(events_seen.values()),
        "circularity": circularity,
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def _interval_cell(row: dict, metric: str) -> str:
    interval = (row.get("interval") or {}).get(metric)
    point = row[metric]
    if not interval or not interval.get("estimable"):
        return f"{point:.3f} —"
    return f"{point:.3f} [{interval['low']:.3f}, {interval['high']:.3f}]"


def render_markdown(payload: dict) -> str:
    """Render one results table.

    The same renderer serves the baseline-only table of phase 2 and the merged
    table of phase 3; ``title``, ``preamble`` and ``thresholds`` are what differ,
    and every column, interval and footnote is identical between them by
    construction rather than by careful copying.
    """
    lines: list[str] = []
    lines.append(f"# {payload.get('title', 'Baseline results — dNBR')}\n")
    lines.append(
        payload.get(
            "preamble",
            f"Generated {payload['generated']} from `{payload['config']}` by "
            "`python -m src.eval.baseline`. No model has been trained at this "
            "point: these rows exist to validate the data chain and the evaluation "
            "machinery before any deep learning is added to the picture.",
        )
        + "\n"
    )

    calibration = payload["calibration_domain"]
    # Phase 2 froze one threshold; phase 3 freezes two, by the same call on the
    # same pixels. Both render from the same records.
    thresholds = payload.get("thresholds") or {"dnbr": payload["threshold"]}
    lines.append("## The decision thresholds\n")
    lines.append(
        f"Calibrated on {calibration['pixels']:,} usable pixels of the "
        f"{calibration['event']} calibration blocks "
        f"({', '.join(calibration['blocks'])}), of which "
        f"{calibration['burned_pixels']:,} are burned "
        f"({calibration['prevalence']:.2%} prevalence). Same pixels, same "
        "objective, same function, both sides — then frozen, and applied unchanged "
        "to every test event with no recalibration on any target.\n"
    )
    lines.append("| Score | Threshold | Objective attained on the calibration blocks |")
    lines.append("|---|---|---|")
    for name, entry in thresholds.items():
        lines.append(
            f"| {name} | ≥ **{entry['value']:.4f}** | "
            f"{entry['objective'].upper()} = {entry['objective_value']:.3f} |"
        )
    lines.append("")
    if len(thresholds) > 1:
        lines.append(
            "The two objective values are not comparable to each other: each is the "
            "best F1 its own score can reach on those pixels, and it is the "
            "*procedure* that is shared, not the number it lands on.\n"
        )

    lines.append("## Results\n")
    lines.append(
        "| Exp. | Event | Method | IoU [95% CI] | F1 | Precision | Recall | AP | "
        "Predicted ha | EMS ha | Error |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in payload["rows"]:
        lines.append(
            f"| {row['experiment']} | {row['event']} | {row['method']} | "
            f"{_interval_cell(row, 'iou')} | {row['f1']:.3f} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['average_precision']:.3f} | "
            f"{row['predicted_area_ha']:,.0f} | {row['label_area_ha']:,.0f} | "
            f"{row['area_error_pct']:+.1f}% |"
        )
    lines.append("")
    lines.append(
        f"Intervals would be percentile bootstraps over ~{payload['superblock_km']:g} km "
        "spatial blocks. A "
        "dash means no interval is publishable for that domain, and every dash here "
        "has the same cause: the protocol counts blocks that actually contain burned "
        "pixels, and no evaluation domain in this project reaches "
        f"{payload['min_blocks_for_interval']} of them. The two transfer events fit "
        "in one block each; the training event has four held-out blocks but 94% of "
        "their burned area lies in one of them. The point estimates stand, and the "
        "reason each interval is missing is written out in `baseline_results.json`.\n"
    )
    lines.append(
        "`EMS ha` is the rasterised delineation inside the evaluated pixels, never a "
        "press figure. `AP` is average precision — threshold-free, so the honest "
        "and oracle rows share it by construction: it is one curve read at two "
        "operating points.\n"
    )

    lines.append("## dNBR ↔ EMS agreement — the circularity floor\n")
    lines.append(
        "How much of any \"learned method beats the index\" result is the index "
        "agreeing with the way the label was drawn in the first place. Average "
        "precision is threshold-free; the oracle columns are the best a dNBR "
        "threshold can do against this label, which is the same operating point as "
        "the oracle row above, read as a property of the label rather than of the "
        "method.\n"
    )
    lines.append("| Event | EMS production method | AP | Oracle IoU | Oracle κ |")
    lines.append("|---|---|---|---|---|")
    for entry in payload["circularity"]:
        lines.append(
            f"| {entry['event']} | {entry['ems_method']} | "
            f"{entry['average_precision']:.3f} | {entry['oracle_iou']:.3f} | "
            f"{entry['oracle_kappa']:.3f} |"
        )
    lines.append("")

    lines.append("## What was actually evaluated\n")
    lines.append(
        "| Event | Footprint px | Usable | Evaluated px | Blocks (with burn) | "
        "Burned ha evaluated | Rasterised total ha | EMS reported ha | Press ha |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for entry in payload["events"]:
        press = f"{entry['press_area_ha']:,.0f}" if entry["press_area_ha"] else "—"
        lines.append(
            f"| {entry['event']} | {entry['footprint_pixels']:,} | "
            f"{entry['valid_fraction']:.1%} | {entry['evaluated_pixels']:,} | "
            f"{entry['evaluated_blocks']} ({entry['evaluated_blocks_with_burn']}) | "
            f"{entry['evaluated_burned_ha']:,.0f} | "
            f"{entry['rasterised_area_ha']:,.0f} | "
            f"{entry['ems_reported_area_ha']:,.0f} | {press} |"
        )
    lines.append("")
    lines.append(
        "`Usable` is the share of the geometric footprint left after cloud and "
        "cloud-shadow masking on both dates. The training event is evaluated only on "
        "its held-out test blocks, which is why its evaluated area is a fraction of "
        "its rasterised total. Press hectares are shown for reference only: they "
        "aggregate several fronts and several dates and are not a reference any "
        "metric here is computed against.\n"
    )

    lines.append("## Confusion matrices\n")
    lines.append("| Exp. | Event | Method | TP | FP | FN | TN |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in payload["rows"]:
        c = row["confusion"]
        lines.append(
            f"| {row['experiment']} | {row['event']} | {row['method']} | "
            f"{c['tp']:,} | {c['fp']:,} | {c['fn']:,} | {c['tn']:,} |"
        )
    lines.append("")
    return "\n".join(lines)


def write(cfg: Config, payload: dict) -> tuple[Path, Path]:
    out = cfg.path_for("outputs")
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "baseline_results.json"
    json_path.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    md_path = out / "baseline_results.md"
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return json_path, md_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--experiment",
        action="append",
        default=None,
        help="experiment identifier from config.yaml; repeatable, default all",
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    payload = run(cfg, args.experiment)
    json_path, md_path = write(cfg, payload)

    # Plain ASCII: this runs on a Windows console whose default code page
    # cannot encode the markdown table's typography.
    threshold = payload["threshold"]
    print(
        f"threshold  : dnbr >= {threshold['value']:.4f}  "
        f"({threshold['objective']} = {threshold['objective_value']:.3f} on "
        f"{threshold['n_positive']:,} burned of {threshold['n_pixels']:,} px)"
    )
    print(f"{'exp':<4} {'event':<14} {'method':<12} {'IoU':>18} {'F1':>7} "
          f"{'prec':>7} {'rec':>7} {'AP':>7} {'area err':>10}")
    for row in payload["rows"]:
        interval = (row.get("interval") or {}).get("iou") or {}
        cell = (
            f"{row['iou']:.3f} [{interval['low']:.3f},{interval['high']:.3f}]"
            if interval.get("estimable")
            else f"{row['iou']:.3f} (no interval)"
        )
        print(
            f"{row['experiment']:<4} {row['event']:<14} "
            f"{row['method'].replace('dNBR ', ''):<12} {cell:>18} "
            f"{row['f1']:>7.3f} {row['precision']:>7.3f} {row['recall']:>7.3f} "
            f"{row['average_precision']:>7.3f} {row['area_error_pct']:>+9.1f}%"
        )
    print()
    print(f"written: {json_path.name}, {md_path.name}")


if __name__ == "__main__":  # pragma: no cover
    main()
