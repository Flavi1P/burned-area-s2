"""Where each method is wrong: TP / FP / FN maps at the frozen threshold.

The grid in ``outputs/regimes.md`` says how much each method gets wrong. This
module says where. One figure per evaluated event, one panel per method, over
the same post-fire false-colour composite as the quicklooks, so the error
classes can be read against what the scene actually shows.

These are diagnostics, not a result. No number in the README is read off them,
and nothing computed here feeds back into a model, a threshold or a seed. The
clearcut morphology test this module was first meant to carry was dropped
before any map existed -- see ``docs/decisions.md``, 2026-10-21.

**Which seed is drawn is a rule, not a look.** A map is one seed, and a trained
model has five. Drawing the best one would flatter the model; drawing the
first one would too, since it was the U-Net's best draw in transfer. So each
trained model is drawn at the seed whose frozen-threshold IoU on the training
event's calibration blocks is the median of its seeds. That reads only the
pixels the threshold was fitted on, never a test event, and the rule was fixed
in ``config.yaml`` before the first map was drawn.

**The pixels are the grid's pixels.** Every panel colours exactly the domain
``build_domain`` scores, at exactly the threshold ``regimes.frozen_threshold``
fits, so the counts in ``outputs/errors.json`` reproduce the frozen rows of
``outputs/regimes.json`` for the drawn seed -- a test holds them to it. Pixels
outside the domain (other roles, cloud, the 1 km role buffer) are greyed out
rather than coloured, so a blank area is never mistaken for a true negative.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Patch, Rectangle

from ..config import Config, Event, load_config
from ..data import ems
from ..data.stack import read_valid, stack_path
from ..eval import regimes
from ..eval import threshold as thresholds_mod
from ..eval.baseline import build_domain
from ..eval.metrics import confusion
from .quicklook import COMPOSITE, _composite, _stretch_limits

SEED_RULES = ("median_calibration_iou",)

# Class codes of an error raster. 0 is "not scored", which is not a negative.
OUTSIDE, TN, TP, FP, FN = 0, 1, 2, 3, 4

# Okabe-Ito blue, orange and reddish purple: separable under the common colour
# vision deficiencies, drawn opaque over a greyscale composite (see `_paint`).
CLASS_COLOURS = {
    TP: ("#0072B2", "true positive"),
    FP: ("#E69F00", "false positive"),
    FN: ("#CC79A7", "false negative"),
}


# --------------------------------------------------------------------------- #
# the drawn seed
# --------------------------------------------------------------------------- #


def median_seed(values: dict[int, float]) -> int:
    """The seed holding the median value; ties to the lowest seed.

    Refuses an even count rather than averaging two maps or picking one of the
    two middle seeds by a rule nobody wrote down.
    """
    if len(values) % 2 == 0:
        raise ValueError(
            f"{len(values)} seeds: the median of an even count is not one of the "
            "seeds. Configure an odd number of seeds or change "
            "error_maps.representative_seed."
        )
    ranked = sorted(values, key=lambda seed: (values[seed], seed))
    middle = values[ranked[len(ranked) // 2]]
    return min(seed for seed, value in values.items() if value == middle)


def calibration_iou(
    cfg: Config, method: regimes.Method, seed: Optional[int]
) -> tuple[float, thresholds_mod.Threshold]:
    """Frozen-threshold IoU on the calibration blocks, and that threshold."""
    frozen = regimes.frozen_threshold(cfg, method, seed)
    settings = cfg.evaluation["threshold"]
    event = cfg.event(settings["calibrated_on"])
    domain = build_domain(
        cfg, event, role=settings["block_role"], score=method.load(cfg, event, seed)
    )
    return confusion(domain.score, domain.truth, frozen.value).iou, frozen


def representative_seed(cfg: Config, method: regimes.Method) -> dict:
    """Apply ``error_maps.representative_seed`` to one method.

    Returns the drawn seed with every candidate's calibration IoU, so the
    choice can be audited from ``errors.json`` without re-running anything.
    """
    rule = cfg.raw["error_maps"]["representative_seed"]
    if rule not in SEED_RULES:
        raise ValueError(f"error_maps.representative_seed: unknown rule {rule!r}")
    if method.seeds == (None,):
        return {
            "seed": None,
            "rule": "no seed",
            "calibration_iou": {},
            "threshold": regimes.frozen_threshold(cfg, method, None),
        }
    fitted = {seed: calibration_iou(cfg, method, seed) for seed in method.seeds}
    seed = median_seed({s: iou for s, (iou, _) in fitted.items()})
    return {
        "seed": seed,
        "rule": rule,
        "calibration_iou": {str(s): iou for s, (iou, _) in fitted.items()},
        "threshold": fitted[seed][1],
    }


# --------------------------------------------------------------------------- #
# error rasters
# --------------------------------------------------------------------------- #


def error_classes(
    score: np.ndarray, label: np.ndarray, mask: np.ndarray, threshold: float
) -> np.ndarray:
    """(H, W) class codes. Uses the same ``score >= threshold`` as the metrics."""
    predicted = score >= threshold
    truth = label.astype(bool)
    out = np.full(mask.shape, OUTSIDE, dtype="uint8")
    out[mask & ~predicted & ~truth] = TN
    out[mask & predicted & truth] = TP
    out[mask & predicted & ~truth] = FP
    out[mask & ~predicted & truth] = FN
    return out


def densest_window(count: np.ndarray, size: int) -> tuple[int, int]:
    """Top-left (row, col) of the size x size window with the largest sum."""
    if size > min(count.shape):
        raise ValueError(f"window {size} px does not fit in a {count.shape} raster")
    integral = np.pad(count.astype("int64"), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    row, col = np.unravel_index(int(np.argmax(sums)), sums.shape)
    return int(row), int(col)


def _post_composite(cfg: Config, event: Event, valid: np.ndarray) -> np.ndarray:
    with rasterio.open(stack_path(cfg, event)) as src:
        channels = list(src.descriptions)
        bands = {
            name: src.read(channels.index(f"post_{name}") + 1) for name in COMPOSITE
        }
    return _composite(bands, _stretch_limits(bands, valid))


def _paint(rgb: np.ndarray, classes: np.ndarray) -> np.ndarray:
    # The composite goes to grey so the error classes are the only colour on
    # the map: in colour, bare soil and built-up areas render in the same pink
    # as a false negative. Structure still reads; hue belongs to the classes.
    grey = np.nan_to_num(rgb.mean(axis=-1, keepdims=True), nan=1.0)
    out = np.repeat(0.15 + 0.7 * grey, 3, axis=-1)
    outside = classes == OUTSIDE
    out[outside] = 0.3 * out[outside] + 0.65  # washed out: not scored, not negative
    for code, (colour, _) in CLASS_COLOURS.items():
        out[classes == code] = matplotlib.colors.to_rgb(colour)
    return out


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


def drawn_methods(cfg: Config) -> list[regimes.Method]:
    """The three rungs; the feature ablation is a row in the grid, not a map."""
    return [m for m in regimes.methods(cfg) if not m.ablation]


def evaluated_events(cfg: Config) -> list[Event]:
    """Every event the grid scores, in experiment order, each once."""
    seen: list[str] = []
    for experiment in cfg.experiments.values():
        if experiment.finetune_on is not None:
            continue
        seen += [e for e in experiment.test if e not in seen]
    return [cfg.event(e) for e in seen]


def run(cfg: Config) -> dict:
    settings = cfg.raw["error_maps"]
    if settings["regime"] != thresholds_mod.FROZEN:
        raise ValueError(
            "error_maps.regime: only the frozen regime is drawn. It is the one "
            "fitted before any test pixel is seen."
        )

    selection, panels = {}, []
    for method in drawn_methods(cfg):
        selection[method.name] = representative_seed(cfg, method)

    events = []
    for event in evaluated_events(cfg):
        per_method = []
        for method in drawn_methods(cfg):
            chosen = selection[method.name]
            score = method.load(cfg, event, chosen["seed"])
            domain = build_domain(cfg, event, score=score)
            with rasterio.open(ems.label_raster_path(cfg, event)) as src:
                label = src.read(1) > 0
            classes = error_classes(score, label, domain.mask, chosen["threshold"].value)
            counts = confusion(domain.score, domain.truth, chosen["threshold"].value)
            per_method.append((method, classes, counts, domain))
        events.append((event, per_method))

    payload = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "config": str(cfg.path.name),
        "regime": settings["regime"],
        "seed_rule": settings["representative_seed"],
        "note": "diagnostics: no README number is read off these maps",
        "seeds": {
            name: {k: v for k, v in s.items() if k != "threshold"}
            | {"threshold": s["threshold"].value}
            for name, s in selection.items()
        },
        "events": {},
    }
    zoom = settings.get("zoom")
    for event, per_method in events:
        entry = {"methods": {}}
        for method, classes, counts, domain in per_method:
            entry["methods"][method.name] = {
                "seed": selection[method.name]["seed"],
                "threshold": selection[method.name]["threshold"].value,
                **counts.as_dict(),
                "iou": counts.iou,
                "fp_ha": counts.fp * domain.pixel_area_ha,
                "fn_ha": counts.fn * domain.pixel_area_ha,
            }
        window = None
        if zoom and zoom["event"] == event.id:
            fp = sum((classes == FP).astype("int32") for _, classes, _, _ in per_method)
            row, col = densest_window(fp, int(zoom["size_px"]))
            window = {"row": row, "col": col, "size_px": int(zoom["size_px"])}
            entry["zoom"] = window
        payload["events"][event.id] = entry
        panels.append((event, per_method, window))

    return payload | {"_panels": panels}


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def _panel_title(method, info: dict, n_seeds: int) -> str:
    seed = (
        "no seed"
        if info["seed"] is None
        else f"seed {info['seed']} (median of {n_seeds})"
    )
    return (
        f"{method.name} — {seed}\nfrozen threshold ≥ {info['threshold']:.3f}  ·  "
        f"IoU {info['iou']:.3f}  ·  FP {info['fp_ha']:.0f} ha  ·  "
        f"FN {info['fn_ha']:.0f} ha"
    )


def _legend(fig) -> None:
    handles = [Patch(color=c, label=l) for c, l in CLASS_COLOURS.values()]
    handles.append(Patch(facecolor="#e0e0e0", edgecolor="#999999", label="not scored"))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )


def draw(cfg: Config, payload: dict) -> list[Path]:
    written = []
    for event, per_method, window in payload["_panels"]:
        rgb = _post_composite(cfg, event, read_valid(cfg, event))
        info = payload["events"][event.id]["methods"]

        crops = [(None, f"errors_{event.id}.png")]
        if window:
            crops.append((window, f"errors_{event.id}_zoom.png"))
        for crop, name in crops:
            if crop:
                rows = slice(crop["row"], crop["row"] + crop["size_px"])
                cols = slice(crop["col"], crop["col"] + crop["size_px"])
            else:
                rows = cols = slice(None)
            height, width = rgb[rows, cols].shape[:2]
            panel_w = 5.4
            fig, axes = plt.subplots(
                1,
                len(per_method),
                figsize=(len(per_method) * panel_w, panel_w * height / width + 1.4),
                constrained_layout=True,
            )
            for ax, (method, classes, _, _) in zip(axes, per_method):
                ax.imshow(_paint(rgb, classes)[rows, cols], interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(
                    _panel_title(method, info[method.name], len(method.seeds)),
                    fontsize=9.5,
                )
                if window and not crop:
                    ax.add_patch(
                        Rectangle(
                            (window["col"] - 0.5, window["row"] - 0.5),
                            window["size_px"],
                            window["size_px"],
                            fill=False,
                            edgecolor="black",
                            linewidth=1.0,
                        )
                    )
            where = (
                f"zoom, {crop['size_px'] * cfg.resolution_m / 1000:g} km window "
                "with the most false positives over the three methods"
                if crop
                else "whole footprint"
            )
            fig.suptitle(
                f"{event.name} — errors at the frozen threshold ({where})  |  "
                f"post-fire {'/'.join(COMPOSITE)} composite in grey, "
                f"{cfg.resolution_m:g} m",
                fontsize=12,
            )
            _legend(fig)
            dest = cfg.path_for("outputs", name)
            fig.savefig(dest, dpi=130, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            written.append(dest)
    return written


def write(cfg: Config, payload: dict) -> Path:
    out = cfg.path_for("outputs")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "errors.json"
    public = {k: v for k, v in payload.items() if not k.startswith("_")}
    path.write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    payload = run(cfg)
    for name, chosen in payload["seeds"].items():
        print(f"{name:<12} drawn seed {chosen['seed']}  ({chosen['rule']})")
    for path in [write(cfg, payload), *draw(cfg, payload)]:
        print(f"written: {path.name}")


if __name__ == "__main__":  # pragma: no cover
    main()
