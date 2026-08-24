"""Methods x threshold regimes, as one grid.

E2 left a question the original table could not answer, because it varied two
things at once. The honest dNBR loses to the U-Net on the unseen biome by 0.36
IoU, and the dNBR *oracle* then beats the U-Net. So part of every gap in that
table is the method and part is the threshold, and the two were never
separated for anything except the index.

This module separates them for everything, by crossing every method with every
threshold regime and running the same evaluation code over all of it:

* **frozen** -- fitted on the training event's calibration blocks and applied
  unchanged. Deployable. This is the protocol the README describes.
* **unsupervised** -- read off the shape of the score's own histogram on the
  target event, consulting no labels. Also deployable, and the regime the
  original design simply did not consider.
* **oracle** -- refitted on the target event's labels. Not deployable, and not
  a competitor: it is the ceiling of what any threshold on that score could
  reach, and the distance from it says how much of a method's failure is its
  threshold rather than its score.

Read a row against its own oracle and you learn whether the method's ranking is
good; read it against another method's oracle and you learn whether its ranking
carries information the other one's does not. Those are different questions and
the old table could only ask the first one of the index.

**The asymmetry the original design imposed is dropped here, deliberately, and
it should be argued with rather than assumed.** ``src/model/evaluate.py``
refuses the U-Net an oracle row, on the grounds that the oracle exists to bound
the index and granting one to the network would dissolve the decomposition. That
was right while there were two methods and one question. With three methods and
three regimes the decomposition *is* the grid, and a missing cell in it is not
a guard -- it is a hole. So every method gets every regime, and what keeps an
oracle number from being read as a result is that no oracle row is ever
deployable and every one of them is labelled so, in the JSON and in the table.
The shipped two-method table in ``outputs/results.md`` is untouched and still
carries the original asymmetry.

**Provenance of the unsupervised regime, stated because it is a selection.**
Otsu and a two-component Gaussian mixture were both tried on the test events
before either was written into ``config.yaml``. Otsu won. Reporting only Otsu
would be a choice made on the test events and hidden, so both estimators are
carried in every table this module writes, and the mixture's poor showing is
part of the result rather than something that was quietly dropped.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import Config, Event, load_config
from . import pixel_model
from . import threshold as thresholds_mod
from .baseline import Domain, build_domain, with_interval
from .dnbr import read_dnbr
from .metrics import Result, evaluate


@dataclass(frozen=True)
class Method:
    """One continuous score, and where it comes from.

    A method contributes a raster and nothing else. Everything downstream --
    domain, threshold, metric, interval -- is the same code for all of them,
    which is what makes the columns of this grid comparable.
    """

    name: str
    score_name: str
    inputs: str  # what the method is allowed to see, per pixel
    context: str  # its spatial extent: the axis the U-Net is supposed to win on
    load: Callable[[Config, Event], np.ndarray]
    ablation: bool = False


def methods(cfg: Config) -> list[Method]:
    """The three rungs, plus the feature ablation of the middle one."""
    reported = cfg.pixel_model["reported"]
    ablations = [s for s in cfg.pixel_model["features"] if s != reported]
    built = [
        Method(
            name="dNBR",
            score_name="dnbr",
            inputs="2 bands x 2 dates",
            context="1 px",
            load=lambda c, e: read_dnbr(c, e),
        ),
        Method(
            name="GB pixel",
            score_name="gb",
            inputs=f"{len(cfg.pixel_model['features'][reported])} features, 4 bands x 2 dates",
            context="1 px",
            load=lambda c, e, s=reported: pixel_model.read_probability(c, e, s),
        ),
    ]
    for feature_set in ablations:
        built.append(
            Method(
                name=f"GB pixel ({feature_set.replace('_', '-')})",
                score_name=f"gb_{feature_set}",
                inputs=f"{len(cfg.pixel_model['features'][feature_set])} features, no absolute post-fire level",
                context="1 px",
                load=lambda c, e, s=feature_set: pixel_model.read_probability(c, e, s),
                ablation=True,
            )
        )
    built.append(
        Method(
            name="U-Net",
            score_name="unet",
            inputs="4 bands x 2 dates",
            context="256 px tile",
            load=_load_unet,
        )
    )
    return built


def _load_unet(cfg: Config, event: Event) -> np.ndarray:
    # Imported here rather than at module scope: the U-Net's reader pulls in
    # torch, and the dNBR and pixel-model rows must not need it to be installed.
    from ..model.predict import read_probability

    return read_probability(cfg, event)


# --------------------------------------------------------------------------- #
# the grid
# --------------------------------------------------------------------------- #


def frozen_threshold(cfg: Config, method: Method) -> thresholds_mod.Threshold:
    """One threshold per method, from the one calibration function.

    The same pixels for every method -- the calibration blocks of the training
    event -- so the procedure is shared even though the value is not. That is
    the property ``config.yaml`` calls threshold symmetry, and it is the reason
    a comparison between two of these rows means anything at all.
    """
    settings = cfg.evaluation["threshold"]
    event = cfg.event(settings["calibrated_on"])
    if not event.is_train:
        raise ValueError(
            f"threshold.calibrated_on names {event.id}, which is not the training "
            "event: calibrating on a test event is the leak this protocol prevents"
        )
    domain = build_domain(
        cfg, event, role=settings["block_role"], score=method.load(cfg, event)
    )
    return thresholds_mod.calibrate(
        domain.score,
        domain.truth,
        score_name=method.score_name,
        calibrated_on=(
            f"{event.id}, {settings['block_role']} blocks "
            f"({', '.join(domain.blocks.blocks_with_role(settings['block_role']))}), "
            f"{domain.truth.size:,} usable px, {int(domain.truth.sum()):,} burned"
        ),
        objective=settings["objective"],
    )


def thresholds_for(
    cfg: Config, method: Method, domain: Domain, frozen: thresholds_mod.Threshold
) -> list[thresholds_mod.Threshold]:
    """Every regime's threshold for one method on one event, in reading order."""
    settings = cfg.evaluation["threshold"]["unsupervised"]
    out = [frozen]
    for estimator in settings["estimators"]:
        out.append(
            thresholds_mod.unsupervised(
                domain.score,
                score_name=method.score_name,
                selected_on=(
                    f"{domain.event.id}, the {domain.truth.size:,} evaluated pixels, "
                    "score histogram only -- no label consulted"
                ),
                estimator=estimator,
                settings=settings,
                seed=int(cfg.pixel_model["seed"]),
            )
        )
    out.append(
        thresholds_mod.calibrate(
            domain.score,
            domain.truth,
            score_name=method.score_name,
            calibrated_on=f"{domain.event.id} itself -- not available in production",
            objective=settings_objective(cfg),
            regime=thresholds_mod.ORACLE,
        )
    )
    return out


def settings_objective(cfg: Config) -> str:
    return cfg.evaluation["threshold"]["objective"]


def _label(threshold: thresholds_mod.Threshold) -> str:
    """How a regime is named in the table."""
    if threshold.regime == thresholds_mod.UNSUPERVISED:
        return f"unsupervised ({threshold.objective})"
    return threshold.regime


def run(cfg: Config, experiment_ids: list[str] | None = None) -> dict:
    """Every method x every regime x every test event, through one code path."""
    experiments = [
        cfg.experiment(eid)
        for eid in (experiment_ids or list(cfg.experiments))
        if cfg.experiment(eid).finetune_on is None
    ]
    catalogue = methods(cfg)
    forbidden = tuple(cfg.evaluation.get("forbidden_metrics", ()))
    frozen = {m.name: frozen_threshold(cfg, m) for m in catalogue}

    rows: list[dict] = []
    domains: dict[tuple[str, str], int] = {}
    for experiment in experiments:
        for event_id in experiment.test:
            event = cfg.event(event_id)
            for method in catalogue:
                domain = build_domain(cfg, event, score=method.load(cfg, event))
                domains[(event_id, method.name)] = int(domain.truth.size)

                for threshold in thresholds_for(cfg, method, domain, frozen[method.name]):
                    result: Result = evaluate(
                        domain.score,
                        domain.truth,
                        threshold.value,
                        method=method.name,
                        event=event.id,
                        experiment=experiment.id,
                        threshold_source=f"{_label(threshold)}: {threshold.calibrated_on}",
                        pixel_area_ha=domain.pixel_area_ha,
                        footprint_pixels=domain.footprint_pixels,
                        with_curve=False,
                    )
                    row = with_interval(cfg, domain, result).as_row(forbidden)
                    row |= {
                        "regime": _label(threshold),
                        "regime_family": threshold.regime,
                        "deployable": not threshold.uses_labels
                        or threshold.regime == thresholds_mod.FROZEN,
                        "uses_target_labels": threshold.regime == thresholds_mod.ORACLE,
                        "inputs": method.inputs,
                        "context": method.context,
                        "ablation": method.ablation,
                    }
                    rows.append(row)

    _assert_identical_domains(domains)

    return {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "config": str(cfg.path.name),
        "estimator": cfg.pixel_model["estimator"],
        "min_blocks_for_interval": int(
            cfg.evaluation["spatial_blocks"]["min_blocks_for_interval"]
        ),
        "superblock_km": float(cfg.evaluation["spatial_blocks"]["superblock_km"]),
        "methods": [
            {
                "name": m.name,
                "score": m.score_name,
                "inputs": m.inputs,
                "context": m.context,
                "ablation": m.ablation,
            }
            for m in catalogue
        ],
        "frozen_thresholds": {name: t.as_dict() for name, t in frozen.items()},
        "regimes": ["frozen", "unsupervised", "oracle"],
        "unsupervised_estimators": list(
            cfg.evaluation["threshold"]["unsupervised"]["estimators"]
        ),
        "rows": rows,
    }


def _assert_identical_domains(domains: dict[tuple[str, str], int]) -> None:
    """Every method saw the same pixels of an event, or the grid means nothing.

    Checked here rather than only in the test suite: a domain mismatch produces
    a plausible table rather than a crash, and this module exists to compare
    columns of that table against each other.
    """
    by_event: dict[str, dict[str, int]] = {}
    for (event_id, method), pixels in domains.items():
        by_event.setdefault(event_id, {})[method] = pixels
    for event_id, counts in by_event.items():
        if len(set(counts.values())) > 1:
            detail = ", ".join(f"{m}: {n:,}" for m, n in sorted(counts.items()))
            raise ValueError(
                f"{event_id}: the methods were scored on different pixel counts "
                f"({detail}). Every method must contribute a score that is finite "
                "wherever the shared validity mask is, or the comparison is between "
                "different evaluations."
            )


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def render_markdown(payload: dict) -> str:
    lines: list[str] = []
    lines.append("# Methods x threshold regimes\n")
    lines.append(
        f"Generated {payload['generated']} from `{payload['config']}` by "
        "`python -m src.eval.regimes`. Every cell comes from the same domain, "
        "threshold, metric and bootstrap code in `src/eval/`; each method "
        "contributes a score raster and nothing else, and all of them were "
        "scored on identically the same pixels of each event — asserted in "
        "`run()`, not hoped for.\n"
    )

    lines.append("## The methods\n")
    lines.append("| Method | Per-pixel inputs | Spatial context | Role |")
    lines.append("|---|---|---|---|")
    for method in payload["methods"]:
        role = "ablation" if method["ablation"] else "reported"
        lines.append(
            f"| {method['name']} | {method['inputs']} | {method['context']} | {role} |"
        )
    lines.append("")
    lines.append(
        "`GB pixel` is the control the original design lacked: every band the "
        "network sees, per pixel, with no context at all. It sits between the "
        "index and the U-Net on the only two axes that separate them, so the "
        "share of the network's lead it reproduces is the share that was never "
        "about spatial context. The estimator is "
        f"`{payload['estimator']}`.\n"
    )

    lines.append("## The regimes\n")
    lines.append(
        "| Regime | What it may look at | Deployable |\n|---|---|---|\n"
        "| frozen | the training event's calibration blocks, labels included | yes |\n"
        "| unsupervised | the target event's score histogram, no labels | yes |\n"
        "| oracle | the target event's labels | **no** |\n"
    )
    lines.append(
        "Only the first two could be run on a fire whose perimeter nobody has "
        "drawn yet. The oracle is an instrument: the ceiling of what any "
        "threshold on that score can reach, so the distance to it separates a "
        "method's ranking from its operating point. Two unsupervised estimators "
        f"are reported — {', '.join(payload['unsupervised_estimators'])} — because "
        "both were tried on the test events before either was written into "
        "`config.yaml`, and publishing only the winner would be a selection on "
        "the test events with the evidence removed.\n"
    )

    lines.append("## The frozen thresholds\n")
    lines.append("| Score | Threshold | F1 attained on the calibration blocks |")
    lines.append("|---|---|---|")
    for name, entry in payload["frozen_thresholds"].items():
        lines.append(
            f"| {name} | ≥ **{entry['value']:.4f}** | {entry['objective_value']:.3f} |"
        )
    lines.append("")
    lines.append(
        "Same function, same pixels, every method — then frozen. The attained F1 "
        "values are not comparable to each other; each is the best its own score "
        "can do on those pixels, and it is the procedure that is shared.\n"
    )

    lines.append("## Results\n")
    by_event: dict[str, list[dict]] = {}
    for row in payload["rows"]:
        by_event.setdefault(row["event"], []).append(row)

    for event_id, rows in by_event.items():
        lines.append(f"### {event_id}\n")
        lines.append(
            "| Method | Regime | Deployable | IoU | F1 | Precision | Recall | AP | "
            "Threshold | Area error |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for row in rows:
            mark = "yes" if row["deployable"] else "— *instrument*"
            lines.append(
                f"| {row['method']} | {row['regime']} | {mark} | "
                f"**{row['iou']:.3f}** | {row['f1']:.3f} | {row['precision']:.3f} | "
                f"{row['recall']:.3f} | {row['average_precision']:.3f} | "
                f"{row['threshold']:.3f} | {row['area_error_pct']:+.1f}% |"
            )
        lines.append("")

    lines.append(
        "AP is threshold-free, so it is constant down each method's four rows by "
        "construction: it describes the score, and the regime only chooses where "
        "to cut it. Comparing AP across methods is therefore the cleanest "
        "statement of which score carries more separable signal, independent of "
        "any operating point.\n"
    )
    lines.append(
        f"No interval is published anywhere in this table. The rule is unchanged: "
        f"the bootstrap resamples ~{payload['superblock_km']:g} km blocks and counts "
        f"only those containing burned pixels, and no evaluation domain in this "
        f"project reaches {payload['min_blocks_for_interval']}. Each row carries its "
        "own reason in `regimes.json`. Adding methods and regimes multiplies the "
        "comparisons this table invites while adding no replicates to support "
        "them, and that is the honest caveat on every difference read off it.\n"
    )
    return "\n".join(lines)


def write(cfg: Config, payload: dict) -> tuple[Path, Path]:
    out = cfg.path_for("outputs")
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "regimes.json"
    json_path.write_text(
        json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
    )
    md_path = out / "regimes.md"
    md_path.write_text(render_markdown(payload), encoding="utf-8")
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

    header = (
        f"{'event':<14} {'method':<22} {'regime':<22} {'IoU':>7} {'prec':>7} "
        f"{'rec':>7} {'AP':>7} {'depl':>6}"
    )
    print(header)
    print("-" * len(header))
    last = None
    for row in payload["rows"]:
        if last is not None and row["event"] != last:
            print()
        last = row["event"]
        print(
            f"{row['event']:<14} {row['method']:<22} {row['regime']:<22} "
            f"{row['iou']:>7.3f} {row['precision']:>7.3f} {row['recall']:>7.3f} "
            f"{row['average_precision']:>7.3f} {'yes' if row['deployable'] else 'no':>6}"
        )
    print()
    print(f"written: {json_path.name}, {md_path.name}")


if __name__ == "__main__":  # pragma: no cover
    main()
