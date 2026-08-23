"""Tiling, in two regimes that are deliberately not the same.

Section 6.1 of the design note asks for two tiling policies, and the difference
between them is a protocol decision rather than an implementation detail.

**Training tiles overlap** (stride = tile - ``overlap_px``), because 65 tiles is
what this project has and overlap is free sample count. They must lie *entirely*
inside pixels carrying the requested role. That is stricter than the role buffer
and costs nothing: a tile straddling a seam would put calibration or test ground
into a training batch, and the receptive field of a U-Net would carry it across
the whole tile. Requiring containment means no training pixel is ever closer
than the 1 km buffer to a pixel it will later be scored on.

**Test tiles do not overlap.** Overlapping them would manufacture dependence
between evaluation units. The grid rarely divides into whole tiles, so the last
tile on each axis is shifted back against the edge rather than padded -- and it
*owns* only the strip no earlier tile covered, so every pixel of the footprint
is predicted exactly once by exactly one tile. "No test pixel is scored twice"
is then true literally, not approximately.

The seam cost of that choice is real, and it is paid by the network alone:
pixels at a tile border are predicted with truncated context, while the dNBR has
no tiles and no borders. The comparison is therefore conservative towards the
U-Net, which is the right direction for it to be wrong in.

Nothing here filters test tiles. The test footprint is geometric -- the EMS area
of interest plus a buffer -- and an entirely unburned tile inside it is part of
the problem, not noise to be removed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from ..config import Config, Event, load_config
from ..eval.blocks import Blocks, blocks_for_event
from . import ems
from .grid import Grid, grid_for_event
from .stack import read_valid

TRAIN, TEST = "train", "test"


@dataclass(frozen=True)
class Tile:
    """One square window on an event grid, and what is inside it."""

    event_id: str
    regime: str  # "train" (overlapping, role-contained) or "test" (paving)
    role: str  # the role its pixels carry
    row: int
    col: int
    size: int
    valid_fraction: float  # share of the tile usable on both dates
    burned_fraction: float  # share of the tile inside the rasterised EMS polygon
    owns: tuple[int, int, int, int]  # (row0, row1, col0, col1) written at inference

    @property
    def window(self) -> Window:
        return Window(
            col_off=self.col, row_off=self.row, width=self.size, height=self.size
        )

    @property
    def slices(self) -> tuple[slice, slice]:
        return (
            slice(self.row, self.row + self.size),
            slice(self.col, self.col + self.size),
        )

    @property
    def owned_slices(self) -> tuple[slice, slice]:
        """The sub-window this tile is responsible for writing at inference."""
        r0, r1, c0, c1 = self.owns
        return slice(r0, r1), slice(c0, c1)

    @property
    def is_positive(self) -> bool:
        return self.burned_fraction > 0.0


def _origins(extent: int, size: int, stride: int) -> list[int]:
    """Tile starts along one axis, with the last one flush against the edge."""
    if extent < size:
        raise ValueError(
            f"a {size} px tile does not fit in a {extent} px axis; either the event "
            "footprint or project.tile_size_px is wrong"
        )
    starts = list(range(0, extent - size + 1, stride))
    if starts[-1] + size < extent:
        starts.append(extent - size)
    return starts


def _spans(starts: list[int], size: int) -> list[tuple[int, int]]:
    """The strip each start owns, so the strips are disjoint and cover the axis."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for start in starts:
        end = start + size
        spans.append((max(start, cursor), end))
        cursor = end
    return spans


def _layers(cfg: Config, event: Event) -> tuple[np.ndarray, np.ndarray]:
    valid = read_valid(cfg, event)
    with rasterio.open(ems.label_raster_path(cfg, event)) as src:
        label = src.read(1) > 0
    return valid, label


def tiles_for_event(
    cfg: Config,
    event: Event,
    regime: str = TEST,
    role: str = TRAIN,
    grid: Grid | None = None,
    blocks: Blocks | None = None,
) -> list[Tile]:
    """Every tile of one event under one regime.

    ``role`` is used by the training regime only, and names the role a tile has
    to be made entirely of. The test regime paves the whole geometric footprint
    and asks no questions about roles: restricting it would be the negative
    filtering section 6.1 forbids.
    """
    if regime not in (TRAIN, TEST):
        raise ValueError(f"unknown tiling regime {regime!r}")
    grid = grid or grid_for_event(cfg, event)
    size = int(cfg.project["tile_size_px"])
    settings = cfg.evaluation["tiling"][regime]
    stride = size - int(settings.get("overlap_px", 0))
    valid, label = _layers(cfg, event)

    row_starts = _origins(grid.height, size, stride)
    col_starts = _origins(grid.width, size, stride)
    row_spans = dict(zip(row_starts, _spans(row_starts, size)))
    col_spans = dict(zip(col_starts, _spans(col_starts, size)))

    contained: np.ndarray | None = None
    minimum = 0.0
    if regime == TRAIN:
        blocks = blocks or blocks_for_event(cfg, event, grid)
        contained = blocks.mask(role)
        minimum = float(settings.get("min_valid_fraction", 0.0))

    tiles: list[Tile] = []
    for row in row_starts:
        for col in col_starts:
            sl = (slice(row, row + size), slice(col, col + size))
            if contained is not None and not contained[sl].all():
                continue
            valid_fraction = float(valid[sl].mean())
            if valid_fraction < minimum:
                continue
            r0, r1 = row_spans[row]
            c0, c1 = col_spans[col]
            owns = (
                (r0, r1, c0, c1)
                if regime == TEST
                else (row, row + size, col, col + size)
            )
            tiles.append(
                Tile(
                    event_id=event.id,
                    regime=regime,
                    role=role if regime == TRAIN else TEST,
                    row=row,
                    col=col,
                    size=size,
                    valid_fraction=valid_fraction,
                    burned_fraction=float(label[sl].mean()),
                    owns=owns,
                )
            )
    return tiles


def training_tiles(cfg: Config, event: Event, role: str = TRAIN) -> list[Tile]:
    """Overlapping tiles lying wholly inside one role of the training event."""
    return tiles_for_event(cfg, event, regime=TRAIN, role=role)


def test_tiles(cfg: Config, event: Event) -> list[Tile]:
    """The non-overlapping paving of one event's whole footprint."""
    return tiles_for_event(cfg, event, regime=TEST)


# --------------------------------------------------------------------------- #
# inventory
# --------------------------------------------------------------------------- #


def _census(tiles: list[Tile]) -> dict:
    positive = [t for t in tiles if t.is_positive]
    negative = len(tiles) - len(positive)
    return {
        "tiles": len(tiles),
        "positive": len(positive),
        "negative": negative,
        "negatives_per_positive": round(negative / len(positive), 2) if positive else None,
        "mean_valid_fraction": round(float(np.mean([t.valid_fraction for t in tiles])), 4)
        if tiles
        else None,
        "mean_burned_fraction_of_positives": round(
            float(np.mean([t.burned_fraction for t in positive])), 4
        )
        if positive
        else None,
    }


def inventory(cfg: Config) -> dict:
    """The composition of every tile set, reported rather than engineered.

    The design note asked for a 2:1 negative-to-positive training ratio with
    clearcuts oversampled. Neither survived contact with the data, and both
    refusals are recorded here rather than in a commit message: the training
    role yields ~65 tiles of which ~34 are positive, so enforcing 2:1 could only
    be done by throwing positives away; and the only way to build a clearcut
    layer in scope would be to threshold the dNBR, which is the baseline under
    test and would make the training set circular.
    """
    train_event = next(e for e in cfg.events.values() if e.is_train)
    calibration_role = cfg.evaluation["threshold"]["block_role"]
    size = int(cfg.project["tile_size_px"])
    train_settings = cfg.evaluation["tiling"][TRAIN]

    sets: list[dict] = []
    for role in (TRAIN, calibration_role):
        sets.append(
            {
                "event": train_event.id,
                "regime": TRAIN,
                "role": role,
                "stride_px": size - int(train_settings["overlap_px"]),
                "min_valid_fraction": float(train_settings.get("min_valid_fraction", 0.0)),
                **_census(training_tiles(cfg, train_event, role=role)),
            }
        )
    for event in cfg.events.values():
        sets.append(
            {
                "event": event.id,
                "regime": TEST,
                "role": TEST,
                "stride_px": size,
                "min_valid_fraction": 0.0,
                **_census(test_tiles(cfg, event)),
            }
        )

    return {
        "tile_size_px": size,
        "tile_km": round(size * cfg.resolution_m / 1000.0, 3),
        "sets": sets,
    }


def inventory_path(cfg: Config) -> Path:
    return cfg.path_for("outputs", "tiles.json")


def write_inventory(cfg: Config) -> Path:
    dest = inventory_path(cfg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(inventory(cfg), indent=2) + "\n", encoding="utf-8")
    return dest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    payload = inventory(cfg)
    path = write_inventory(cfg)

    print(f"tile {payload['tile_size_px']} px = {payload['tile_km']:.2f} km")
    for entry in payload["sets"]:
        ratio = entry["negatives_per_positive"]
        ratio_text = "n/a" if ratio is None else format(ratio, ".2f")
        print(
            f"  {entry['event']:14s} {entry['regime']:5s} {entry['role']:12s} "
            f"stride {entry['stride_px']:3d}  tiles {entry['tiles']:4d}  "
            f"positive {entry['positive']:4d}  neg/pos {ratio_text}"
        )
    print(f"written    : {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
