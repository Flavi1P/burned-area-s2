"""The dNBR baseline.

NBR = (NIR - SWIR2) / (NIR + SWIR2), dNBR = NBR_pre - NBR_post. Combustion
removes the chlorophyll that lifts NIR and exposes char and ash, which raise
SWIR2; the normalised difference of the two is the canonical burn index and
has been the operational standard since Key & Benson. It is the right baseline
precisely because it is not a straw man: it is what an agency would actually
run, and if it wins here that is a publishable result worth stating plainly.

Two things this module refuses to do.

**It does not threshold.** A dNBR raster is a continuous score, exactly like
the network's sigmoid output, and the threshold arrives from
``src.eval.threshold`` -- the same function, the same calibration pixels. Fixed
severity breakpoints from the literature (0.10, 0.27, 0.44 ...) are deliberately
not used: they were derived for Composite Burn Index severity classes in North
American conifer, and importing them here would import a calibration nobody in
this project can defend.

**It does not decide what is visible.** Validity comes from the shared mask in
``src.data.stack``, so the baseline and the network are scored on identically
the same pixels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio

from ..config import Config, Event, load_config
from ..data.grid import Grid, grid_for_event
from ..data.stack import Stack, build_stack, stack_path, valid_path

# The two bands the index is made of, as named in config.yaml's `bands` block.
NIR = "nir08"   # B8A
SWIR2 = "swir22"  # B12


def nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """Normalised Burn Ratio. Undefined where both bands vanish; NaN there."""
    denominator = nir + swir2
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (nir - swir2) / denominator
    out[denominator == 0] = np.nan
    return out.astype("float32")


def dnbr_from_stack(stack: Stack) -> np.ndarray:
    """dNBR of one event: positive where the scene darkened in NIR relative to
    SWIR2, i.e. where vegetation burned."""
    pre = nbr(stack.band("pre", NIR), stack.band("pre", SWIR2))
    post = nbr(stack.band("post", NIR), stack.band("post", SWIR2))
    return (pre - post).astype("float32")


def dnbr_path(cfg: Config, event: Event) -> Path:
    return cfg.path_for("data_interim", event.id, "dnbr.tif")


def write_dnbr(cfg: Config, event: Event, grid: Grid | None = None) -> tuple[Path, np.ndarray]:
    grid = grid or grid_for_event(cfg, event)
    stack = build_stack(cfg, event, grid)
    score = dnbr_from_stack(stack)

    dest = dnbr_path(cfg, event)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dest, "w", **grid.profile("float32", nodata=float("nan"))) as dst:
        dst.write(score, 1)
        dst.set_band_description(1, "dnbr")
        dst.update_tags(
            event=event.id,
            index="dNBR = NBR_pre - NBR_post",
            nbr=f"(({NIR} - {SWIR2}) / ({NIR} + {SWIR2}))",
            pre_date=stack.dates["pre"],
            post_date=stack.dates["post"],
            valid_fraction=f"{stack.valid_fraction:.4f}",
            note="continuous score; the threshold lives in outputs/thresholds.json",
        )
    return dest, score


def read_dnbr(cfg: Config, event: Event) -> np.ndarray:
    """The dNBR raster, computed on demand if it is not on disk yet."""
    path = dnbr_path(cfg, event)
    if not path.exists():
        return write_dnbr(cfg, event)[1]
    with rasterio.open(path) as src:
        return src.read(1)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--event", default=None, help="default: every event")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    for event_id in [args.event] if args.event else list(cfg.events):
        event = cfg.event(event_id)
        dest, score = write_dnbr(cfg, event)
        finite = score[np.isfinite(score)]
        quartiles = np.percentile(finite, [5, 50, 95]) if finite.size else [np.nan] * 3
        print(f"{event.id}")
        print(f"  written : {dest.name}")
        print(
            f"  dNBR    : p05 {quartiles[0]:+.3f}  median {quartiles[1]:+.3f}  "
            f"p95 {quartiles[2]:+.3f}  over {finite.size:,} defined px"
        )
        print(f"  stack   : {stack_path(cfg, event).name}, {valid_path(cfg, event).name}")


if __name__ == "__main__":  # pragma: no cover
    main()
