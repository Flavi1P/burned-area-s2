"""Figures of the evaluation setup and of what the baseline did.

Two figures, and both of them are about the protocol rather than about a score.

``pr_baseline.png`` is the precision-recall curve of the dNBR on each test
event, with the frozen operating point and the oracle operating point marked on
it. It is the figure that makes the threshold argument visible: the curve is a
property of the index, the two dots are two ways of choosing a point on it, and
the distance between them is the part of any later "the network wins" that a
recalibration would also have bought.

``split_map.png`` is the training event's block split. It exists because "we
used a spatially blocked split" is a sentence anybody can write; a map showing
which ground trained, which calibrated the threshold, which was held out, and
how much of it the clouds took, is a claim a reader can check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..config import Config, load_config
from ..data import ems
from ..data.grid import grid_for_event
from ..data.stack import read_valid
from ..eval.blocks import blocks_for_event

# One colour per method, reused wherever the two appear together.
METHOD_COLOURS = {"dNBR honest": "#c1440e", "dNBR oracle": "#4a7ba7"}
CLOUD_COLOUR = "#b9c6d2"
ROLE_COLOURS = {
    "buffer": "#e8e4dd",
    "train": "#8fb996",
    "calibration": "#e5c185",
    "test": "#8aa9c9",
}


def _results(cfg: Config) -> dict:
    path = cfg.path_for("outputs", "baseline_results.json")
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found -- run `python -m src.eval.baseline` first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def pr_figure(cfg: Config, dest: Path | None = None) -> Path:
    """Precision-recall per event, with both operating points marked."""
    payload = _results(cfg)
    events = [entry["event"] for entry in payload["events"]]
    by_event = {
        (row["event"], row["method"]): row for row in payload["rows"]
    }

    fig, axes = plt.subplots(
        1, len(events), figsize=(4.2 * len(events), 4.4), constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    for ax, event in zip(axes, events):
        honest = by_event[(event, "dNBR honest")]
        oracle = by_event[(event, "dNBR oracle")]
        curve = honest["pr_curve"]

        ax.plot(curve["recall"], curve["precision"], color="#333333", lw=1.6, zorder=2)
        ax.axhline(
            honest["prevalence"],
            color="#999999",
            ls=(0, (4, 3)),
            lw=1.0,
            zorder=1,
        )

        for row, marker in ((honest, "o"), (oracle, "s")):
            ax.plot(
                row["recall"],
                row["precision"],
                marker,
                color=METHOD_COLOURS[row["method"]],
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=1.2,
                zorder=3,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("recall")
        ax.set_aspect("equal")
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_title(
            f"{event}   AP = {honest['average_precision']:.3f}", fontsize=11
        )
    axes[0].set_ylabel("precision")

    handles = [
        Line2D([], [], color="#333333", lw=1.6, label="dNBR, all thresholds"),
        Line2D(
            [], [], color=METHOD_COLOURS["dNBR honest"], marker="o", ls="",
            markersize=8, label="frozen threshold (honest)",
        ),
        Line2D(
            [], [], color=METHOD_COLOURS["dNBR oracle"], marker="s", ls="",
            markersize=8, label="threshold refit on this event (oracle)",
        ),
        Line2D(
            [], [], color="#999999", ls=(0, (4, 3)), lw=1.0,
            label="prevalence — precision of a coin flip",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.10),
    )
    fig.suptitle(
        "dNBR baseline — precision-recall per evaluation domain, "
        f"threshold frozen at {payload['threshold']['value']:.3f}",
        fontsize=12,
    )

    dest = dest or cfg.path_for("outputs", "pr_baseline.png")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def split_figure(cfg: Config, event_id: str | None = None, dest: Path | None = None) -> Path:
    """The block split of the training event, with the scar and the cloud loss."""
    event = cfg.event(event_id or cfg.evaluation["threshold"]["calibrated_on"])
    grid = grid_for_event(cfg, event)
    blocks = blocks_for_event(cfg, event, grid)
    valid = read_valid(cfg, event)
    with rasterio.open(ems.label_raster_path(cfg, event)) as src:
        label = src.read(1) > 0

    order = ["buffer", "train", "calibration", "test"]
    cmap = ListedColormap([ROLE_COLOURS[name] for name in order])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.4), constrained_layout=True)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].imshow(blocks.role, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    axes[0].contour(label, levels=[0.5], colors="#7a1f0d", linewidths=0.8)
    for name in blocks.names:
        inside = blocks.index == blocks.index_of(name)
        rows, cols = np.where(inside)
        axes[0].text(
            cols.mean(), rows.mean(), name,
            ha="center", va="center", fontsize=9, color="#333333",
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1.5},
        )
    axes[0].set_title("spatial blocks and their role", fontsize=11)

    # Cloud and shadow read as cloud, not as background: the eye has to be able
    # to tell "nothing burned here" from "nobody can tell what happened here".
    axes[1].imshow(
        np.ones_like(label, dtype=float),
        cmap=ListedColormap([CLOUD_COLOUR]),
        interpolation="nearest",
    )
    axes[1].imshow(
        np.where(valid, 1.0, np.nan),
        cmap=ListedColormap(["#f4f1ec"]),
        interpolation="nearest",
    )
    burned_visible = label & valid
    burned_lost = label & ~valid
    axes[1].imshow(
        np.where(burned_visible, 1.0, np.nan),
        cmap=ListedColormap(["#c1440e"]),
        interpolation="nearest",
    )
    axes[1].imshow(
        np.where(burned_lost, 1.0, np.nan),
        cmap=ListedColormap(["#7a7a7a"]),
        interpolation="nearest",
    )
    axes[1].set_title("what the cloud mask leaves of the scar", fontsize=11)

    ha = grid.pixel_area_ha
    settings = cfg.evaluation["spatial_blocks"]
    role_handles = [
        Patch(facecolor=ROLE_COLOURS[name], label=f"{name}") for name in order
    ]
    axes[0].legend(
        handles=role_handles
        + [Line2D([], [], color="#7a1f0d", lw=1.5, label="EMS delineation")],
        loc="upper right",
        fontsize=8.5,
        framealpha=0.9,
    )
    axes[1].legend(
        handles=[
            Patch(facecolor="#c1440e", label=f"burned, usable ({burned_visible.sum() * ha:,.0f} ha)"),
            Patch(facecolor="#7a7a7a", label=f"burned, cloud or shadow ({burned_lost.sum() * ha:,.0f} ha)"),
            Patch(
                facecolor=CLOUD_COLOUR,
                label=f"cloud or shadow ({1 - valid.mean():.0%} of the footprint)",
            ),
        ],
        loc="lower right",
        fontsize=8.5,
        framealpha=0.9,
    )

    fig.suptitle(
        f"{event.name} — {grid.width}×{grid.height} px at {grid.resolution:g} m, "
        f"{blocks.n_blocks} blocks of ~{settings['superblock_km']} km, "
        f"{settings['role_buffer_m']} m buffer between roles",
        fontsize=12,
    )
    fig.text(
        0.005, -0.015,
        "The buffer band belongs to no role: no training pixel is within "
        f"{2 * settings['role_buffer_m'] / 1000:g} km of a calibration or test pixel. "
        "At this block size only one held-out block carries meaningful burned area, "
        "which is why no bootstrap interval is publishable here.",
        fontsize=8, va="top", color="#444444", transform=fig.transFigure,
    )

    dest = dest or cfg.path_for("outputs", f"split_{event.id}.png")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--event", default=None, help="event for the split map")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    for path in (pr_figure(cfg), split_figure(cfg, args.event)):
        print(f"written: {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
