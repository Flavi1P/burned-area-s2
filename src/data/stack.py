"""The per-event analysis stack, and the validity mask that goes with it.

Every raster the project produces for one event lives on that event's 20 m
grid, so stacking is a matter of stapling the eight already-clipped bands
together and converting the digital numbers to reflectance. Two things earn
this module its own file.

**Reflectance, not DN.** Since processing baseline 04.00 the L2A products carry
a -1000 DN offset, so the familiar ``DN / 10000`` is wrong by 0.1 reflectance.
That is not cosmetic for this project: NBR is a normalised ratio, and a
constant additive error on both terms does not cancel. It shifts every dNBR,
and therefore the calibrated threshold, in a way that would silently differ
between an old scene and a new one. The conversion is read from the band
metadata that ``src.data.stac`` stamped on each COG, never hard-coded.

**One validity mask for both dates and both methods.** A pixel counts only if
the Scene Classification Layer calls it usable on the pre date *and* the post
date, and if it carries data in every band. Anything else is dropped -- from
the dNBR, from the U-Net, from the metrics, from the bootstrap, identically.
The alternative, letting each method decide for itself what it can see, would
mean the two methods are scored on different pixels, and the comparison table
would be measuring the masks as much as the methods.

Cloud shadow is in the invalid set for a substantive reason, not for tidiness:
a shadow over vegetation darkens NIR and SWIR much the way a burn scar does,
and it is the textbook false positive of every index-based method. Leaving
shadows in would hand the CNN an advantage the baseline cannot have, which is
exactly the kind of unearned margin this project exists not to publish.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from ..config import Config, Event, load_config
from .grid import Grid, grid_for_event
from .stac import SCL_CLOUD_SHADOW, SCL_NO_DATA, fetch_scene

PHASES = ("pre", "post")


@dataclass(frozen=True)
class Stack:
    """The eight reflectance channels of one event, plus what is usable."""

    event_id: str
    grid: Grid
    channels: tuple[str, ...]  # e.g. ("pre_red", ..., "post_swir22")
    data: np.ndarray  # (8, H, W) float32 reflectance, NaN where invalid
    valid: np.ndarray  # (H, W) bool
    dates: dict[str, str]
    scenes: dict[str, str]

    def band(self, phase: str, asset: str) -> np.ndarray:
        return self.data[self.channels.index(f"{phase}_{asset}")]

    @property
    def valid_fraction(self) -> float:
        return float(self.valid.mean())


def stack_path(cfg: Config, event: Event) -> Path:
    return cfg.path_for("data_interim", event.id, "stack.tif")


def valid_path(cfg: Config, event: Event) -> Path:
    return cfg.path_for("data_interim", event.id, "valid.tif")


def _read_reflectance(path: Path) -> np.ndarray:
    """One band as float32 reflectance, NaN where the source says nodata."""
    with rasterio.open(path) as src:
        dn = src.read(1)
        scale = src.scales[0]
        offset = src.offsets[0]
        nodata = src.nodata
    out = dn.astype("float32") * scale + offset
    if nodata is not None:
        out[dn == nodata] = np.nan
    return out


def _scl_valid(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        scl = src.read(1)
    return ~np.isin(scl, SCL_CLOUD_SHADOW + (SCL_NO_DATA,))


def build_stack(cfg: Config, event: Event, grid: Grid | None = None) -> Stack:
    """Assemble the eight channels and the joint validity mask, in memory."""
    grid = grid or grid_for_event(cfg, event)
    qa = cfg.project["qa_band"]

    channels: list[str] = []
    arrays: list[np.ndarray] = []
    valid = np.ones(grid.shape, dtype=bool)
    dates: dict[str, str] = {}
    scenes: dict[str, str] = {}

    for phase in PHASES:
        scene, paths = fetch_scene(cfg, event, phase, grid)
        dates[phase] = f"{scene.date:%Y-%m-%d}"
        scenes[phase] = scene.item_id
        valid &= _scl_valid(paths[qa])
        for asset in cfg.band_assets:
            band = _read_reflectance(paths[asset])
            valid &= np.isfinite(band)
            channels.append(f"{phase}_{asset}")
            arrays.append(band)

    data = np.stack(arrays).astype("float32")
    data[:, ~valid] = np.nan
    return Stack(
        event_id=event.id,
        grid=grid,
        channels=tuple(channels),
        data=data,
        valid=valid,
        dates=dates,
        scenes=scenes,
    )


def write_stack(cfg: Config, event: Event, stack: Stack | None = None) -> tuple[Path, Path]:
    """Serialise the stack and its validity mask next to the event's label."""
    stack = stack or build_stack(cfg, event)
    grid = stack.grid
    tags = {
        "event": stack.event_id,
        "channels": ",".join(stack.channels),
        "pre_date": stack.dates["pre"],
        "post_date": stack.dates["post"],
        "pre_item": stack.scenes["pre"],
        "post_item": stack.scenes["post"],
        "valid_fraction": f"{stack.valid_fraction:.4f}",
    }

    dest = stack_path(cfg, event)
    dest.parent.mkdir(parents=True, exist_ok=True)
    profile = grid.profile("float32", nodata=float("nan"), count=len(stack.channels))
    with rasterio.open(dest, "w", **profile) as dst:
        dst.write(stack.data)
        for index, name in enumerate(stack.channels, start=1):
            dst.set_band_description(index, name)
        dst.update_tags(**tags)

    mask_dest = valid_path(cfg, event)
    with rasterio.open(mask_dest, "w", **grid.profile("uint8", nodata=None)) as dst:
        dst.write(stack.valid.astype("uint8"), 1)
        dst.update_tags(**tags, meaning="1 = usable on both dates, in every band")
    return dest, mask_dest


def read_valid(cfg: Config, event: Event) -> np.ndarray:
    """The validity mask, built on demand if it is not on disk yet."""
    path = valid_path(cfg, event)
    if not path.exists():
        write_stack(cfg, event)
    with rasterio.open(path) as src:
        return src.read(1).astype(bool)


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
        stack = build_stack(cfg, event)
        dest, mask_dest = write_stack(cfg, event, stack)
        print(f"{event.id}")
        print(f"  grid    : {stack.grid}")
        print(f"  dates   : pre {stack.dates['pre']}  post {stack.dates['post']}")
        print(f"  channels: {len(stack.channels)}  {', '.join(stack.channels)}")
        print(
            f"  valid   : {stack.valid_fraction:.2%} of the footprint "
            f"({int(stack.valid.sum()):,} px)"
        )
        print(f"  written : {dest.name}, {mask_dest.name}")


if __name__ == "__main__":  # pragma: no cover
    main()
