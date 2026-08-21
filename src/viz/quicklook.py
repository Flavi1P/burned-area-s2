"""Before / after / label quicklook for one event.

A false-colour SWIR-2 / NIR / Red composite, not true colour: with B12 in the
red channel a burn scar reads as saturated orange against green vegetation,
which is exactly the contrast the whole project is about. It also happens to be
made of three of the four bands the model actually receives, so the figure
shows the model's input rather than a prettier picture of something else.

The stretch is computed once, on the post-fire scene, and applied to both
dates. Stretching each panel independently would normalise away the very
radiometric change the pair exists to carry.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.lines import Line2D

from ..config import Config, load_config
from ..data import ems, stac
from ..data.stac import SCL_CLOUD_SHADOW, SCL_NO_DATA, cloud_shadow_fraction
from ..data.grid import grid_for_event


COMPOSITE = ("swir22", "nir08", "red")  # B12, B8A, B04


def _stretch_limits(
    bands: dict[str, np.ndarray], valid: np.ndarray, low: float = 2.0, high: float = 98.0
) -> dict[str, tuple[float, float]]:
    limits = {}
    for name in COMPOSITE:
        values = bands[name][valid]
        values = values[np.isfinite(values)]
        limits[name] = tuple(np.percentile(values, [low, high])) if values.size else (0.0, 1.0)
    return limits


def _composite(
    bands: dict[str, np.ndarray], limits: dict[str, tuple[float, float]]
) -> np.ndarray:
    channels = []
    for name in COMPOSITE:
        lo, hi = limits[name]
        scaled = (bands[name].astype("float32") - lo) / max(hi - lo, 1e-6)
        channels.append(np.clip(scaled, 0, 1))
    return np.dstack(channels)


def _load_phase(cfg: Config, event, phase: str, grid) -> tuple[stac.Scene, dict[str, np.ndarray]]:
    scene, paths = stac.fetch_scene(cfg, event, phase, grid)
    arrays = {}
    for asset, path in paths.items():
        with rasterio.open(path) as src:
            arrays[asset] = src.read(1)
    return scene, arrays


def quicklook(cfg: Config, event_id: str, dest: Path | None = None) -> Path:
    event = cfg.event(event_id)
    grid = grid_for_event(cfg, event)

    post_scene, post = _load_phase(cfg, event, "post", grid)
    pre_scene, pre = _load_phase(cfg, event, "pre", grid)

    label_path, areas = ems.write_label_raster(cfg, event, grid)
    with rasterio.open(label_path) as src:
        label = src.read(1)

    scl = post["scl"]
    invalid = np.isin(scl, SCL_CLOUD_SHADOW + (SCL_NO_DATA,))
    valid = ~invalid
    cloud_fraction = cloud_shadow_fraction(scl)

    limits = _stretch_limits(post, valid)
    rgb_pre = _composite(pre, limits)
    rgb_post = _composite(post, limits)

    # Size the canvas from the footprint so the panels carry the figure and the
    # margins do not.
    aspect = grid.width / grid.height
    panel_w = 5.4
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(3 * panel_w + 0.6, panel_w / aspect + 1.5),
        constrained_layout=True,
    )
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].imshow(rgb_pre)
    axes[0].set_title(f"pre-fire  {pre_scene.date}", fontsize=11)

    axes[1].imshow(rgb_post)
    axes[1].set_title(f"post-fire  {post_scene.date}", fontsize=11)

    axes[2].imshow(rgb_post)
    overlay = np.zeros((*label.shape, 4), dtype="float32")
    overlay[label == 1] = (0.15, 0.55, 1.0, 0.35)
    axes[2].imshow(overlay)
    axes[2].contour(label, levels=[0.5], colors="#2b8cff", linewidths=0.7)
    axes[2].set_title(
        f"post-fire + EMS label  {event.label.product_id}", fontsize=11
    )
    axes[2].legend(
        handles=[Line2D([], [], color="#2b8cff", lw=2, label="EMS observed event")],
        loc="lower right",
        fontsize=9,
        framealpha=0.85,
    )

    prod = event.label.production
    fig.suptitle(
        f"{event.name} — {event.admin}   |   fire of {event.event_datetime:%d %b %Y}"
        f"   |   Sentinel-2 L2A, {COMPOSITE[0]}/{COMPOSITE[1]}/{COMPOSITE[2]} "
        f"(B12/B8A/B04) at {grid.resolution:g} m",
        fontsize=13,
    )
    footer = [
        f"EMS {event.label.product_id} ({event.label.status}), delineated on "
        f"{prod.delineated_on}",
        f"method: {prod.method}, {prod.analysis_scale}, MMU {prod.mmu_m2:g} m2, "
        f"RMSE {prod.geometric_rmse_m:g} m",
        f"polygon {areas['polygon_area_ha']:.0f} ha -> rasterised at "
        f"{grid.resolution:g} m {areas['rasterised_area_ha']:.0f} ha",
        f"SCL cloud/shadow over the footprint: {cloud_fraction:.1%}",
        f"post scene: {post_scene.reason}",
    ]
    fig.text(
        0.005,
        -0.01,
        "\n".join(["  ·  ".join(footer[:3]), "  ·  ".join(footer[3:])]),
        fontsize=7.5,
        va="top",
        color="#444444",
        linespacing=1.6,
        transform=fig.transFigure,
    )

    dest = dest or cfg.path_for("outputs", f"quicklook_{event.id}.png")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"event            : {event.id}")
    print(f"grid             : {grid}")
    print(f"pre  scene       : {pre_scene.item_id} ({pre_scene.reason})")
    print(f"post scene       : {post_scene.item_id} ({post_scene.reason})")
    print(f"label            : {event.label.product_id}, {prod.method}")
    print(
        f"areas            : polygon {areas['polygon_area_ha']:.1f} ha, "
        f"rasterised {areas['rasterised_area_ha']:.1f} ha, "
        f"EMS-reported {areas['reported_area_ha']:.1f} ha"
    )
    print(f"cloud/shadow     : {cloud_fraction:.2%} of the footprint (SCL, post scene)")
    print(f"figure           : {dest}")
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--event", required=True, help="event identifier from config.yaml")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    quicklook(cfg, args.event, Path(args.out) if args.out else None)


if __name__ == "__main__":  # pragma: no cover
    main()
