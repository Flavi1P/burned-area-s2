"""Spatial blocks: the unit of independence for splitting and for resampling.

Two jobs, one geometry, because they are the same question asked twice.

**Splitting.** Inside the training event, some ground trains the model, some
calibrates the decision threshold, and some tests it. If those three touch,
every number downstream is inflated by pixels the model has effectively
already seen -- neighbouring 20 m pixels of one burn scar are near-copies of
each other.

**Resampling.** A bootstrap that resamples pixels, or even 5 km tiles, assumes
its units are independent. They are not: a 30 000 ha scar spans some 20 km, so
two neighbouring tiles carry the same fire, the same fuel, the same smoke and
the same analyst. Resampling them would produce a confidence interval an order
of magnitude too narrow -- a more subtle version of the very leak the split
exists to prevent. So the resampling unit is a ~17.5 km super-block, chosen to
sit beyond that autocorrelation range.

The consequence is deliberately not hidden: at that size the two test events
hold a single block each, and no interval is estimable for them. That hole is
reported as a hole. See ``src.eval.bootstrap``.

Blocks are named ``r{row}c{col}`` from the north-west corner of the event grid,
so the identifiers in ``config.yaml`` read as the map they are.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

from ..config import Config, Event, load_config
from ..data import ems
from ..data.grid import Grid, grid_for_event
from ..data.stack import read_valid

# Role codes, as written in roles.tif. 0 stays free for "no role".
ROLE_CODES = {"buffer": 0, "train": 1, "calibration": 2, "test": 3}
ROLE_NAMES = {code: name for name, code in ROLE_CODES.items()}
DEFAULT_ROLE = "train"


@dataclass(frozen=True)
class Blocks:
    """The block partition of one event grid, with a role for every pixel."""

    event_id: str
    grid: Grid
    index: np.ndarray  # (H, W) int16, index into `names`
    names: tuple[str, ...]
    role: np.ndarray  # (H, W) uint8, ROLE_CODES
    block_role: dict[str, str]

    @property
    def n_blocks(self) -> int:
        return len(self.names)

    def mask(self, role: str) -> np.ndarray:
        """Pixels carrying one role, buffer already removed."""
        if role not in ROLE_CODES:
            raise KeyError(f"unknown role {role!r}; known: {sorted(ROLE_CODES)}")
        return self.role == ROLE_CODES[role]

    def blocks_with_role(self, role: str) -> list[str]:
        return [n for n in self.names if self.block_role[n] == role]

    def index_of(self, name: str) -> int:
        return self.names.index(name)


def _block_edges(extent_px: int, resolution: float, target_km: float) -> np.ndarray:
    """Cut one axis into whole blocks as close as possible to the target size.

    ``round`` rather than ``ceil``: ceiling a 17.7 km event into 17.5 km blocks
    would produce a second block 200 m wide, an "independent" unit holding a
    few hundred pixels. One slightly oversized block is an honest description
    of an event that holds one block.
    """
    span_km = extent_px * resolution / 1000.0
    n = max(1, int(round(span_km / target_km)))
    return np.linspace(0, extent_px, n + 1).round().astype(int)


def partition(cfg: Config, grid: Grid) -> tuple[np.ndarray, tuple[str, ...]]:
    """Cut a grid into super-blocks. Geometry only -- no label is consulted."""
    target_km = float(cfg.evaluation["spatial_blocks"]["superblock_km"])
    rows = _block_edges(grid.height, grid.resolution, target_km)
    cols = _block_edges(grid.width, grid.resolution, target_km)

    index = np.zeros(grid.shape, dtype="int16")
    names: list[str] = []
    for i in range(len(rows) - 1):
        for j in range(len(cols) - 1):
            index[rows[i] : rows[i + 1], cols[j] : cols[j + 1]] = len(names)
            names.append(f"r{i}c{j}")
    return index, tuple(names)


def _buffered(role: np.ndarray, buffer_px: float) -> np.ndarray:
    """Blank out pixels lying within ``buffer_px`` of a role boundary.

    The boundary is where the role changes between 4-neighbours, and the
    distance transform measures how far every pixel sits from the nearest such
    seam. The outer edge of the grid is not a seam, so nothing is eroded there.
    """
    if buffer_px <= 0:
        return role
    seam = np.zeros(role.shape, dtype=bool)
    seam[:-1, :] |= role[:-1, :] != role[1:, :]
    seam[1:, :] |= role[:-1, :] != role[1:, :]
    seam[:, :-1] |= role[:, :-1] != role[:, 1:]
    seam[:, 1:] |= role[:, :-1] != role[:, 1:]
    if not seam.any():
        return role
    distance = ndimage.distance_transform_edt(~seam)
    out = role.copy()
    out[distance <= buffer_px] = ROLE_CODES["buffer"]
    return out


def blocks_for_event(cfg: Config, event: Event, grid: Grid | None = None) -> Blocks:
    """The block partition and the pixel-level role map of one event."""
    grid = grid or grid_for_event(cfg, event)
    settings = cfg.evaluation["spatial_blocks"]
    index, names = partition(cfg, grid)

    declared = (settings.get("roles") or {}).get(event.id, {}) or {}
    block_role = {name: DEFAULT_ROLE for name in names}
    if not event.is_train:
        # A test event is never split: section 6.1 defines its footprint
        # geometrically and evaluates everything inside it, unburned included.
        block_role = {name: "test" for name in names}
    for role, block_names in declared.items():
        if role not in ROLE_CODES:
            raise KeyError(f"{event.id}: unknown block role {role!r} in config.yaml")
        for name in block_names:
            if name not in block_role:
                raise KeyError(
                    f"{event.id}: config.yaml assigns role {role!r} to block "
                    f"{name!r}, which this grid does not contain "
                    f"({len(names)} blocks: {', '.join(names)})"
                )
            block_role[name] = role

    role = np.zeros(grid.shape, dtype="uint8")
    for name in names:
        role[index == names.index(name)] = ROLE_CODES[block_role[name]]

    buffer_px = float(settings.get("role_buffer_m", 0.0)) / grid.resolution
    role = _buffered(role, buffer_px)
    return Blocks(
        event_id=event.id,
        grid=grid,
        index=index,
        names=names,
        role=role,
        block_role=block_role,
    )


# --------------------------------------------------------------------------- #
# serialisation and inventory
# --------------------------------------------------------------------------- #


def roles_path(cfg: Config, event: Event) -> Path:
    return cfg.path_for("data_interim", event.id, "roles.tif")


def write_roles(cfg: Config, event: Event, blocks: Blocks | None = None) -> Path:
    """Serialise the role map, and only the role map.

    The block index is not written: ``partition`` derives it from the grid
    geometry alone, so a file holding it would be a copy of something already
    reproducible from ``config.yaml``, free to recompute, and able to go stale.
    The role map is different -- it also carries the buffer erosion, which
    depends on the declared roles -- and it is what the dataset and the figures
    actually read.
    """
    blocks = blocks or blocks_for_event(cfg, event)
    dest = roles_path(cfg, event)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dest, "w", **blocks.grid.profile("uint8", nodata=None)) as dst:
        dst.write(blocks.role, 1)
        dst.update_tags(
            event=blocks.event_id,
            blocks=",".join(blocks.names),
            roles=",".join(f"{n}={blocks.block_role[n]}" for n in blocks.names),
            superblock_km=str(cfg.evaluation["spatial_blocks"]["superblock_km"]),
            role_buffer_m=str(cfg.evaluation["spatial_blocks"].get("role_buffer_m", 0)),
            codes=",".join(f"{v}={k}" for k, v in ROLE_CODES.items()),
        )
    return dest


def inventory(cfg: Config, event: Event, blocks: Blocks | None = None) -> dict:
    """Per-block census: role, usable pixels, burned pixels.

    Written out so the composition of every split is a file in the repository
    rather than a claim in a README.
    """
    blocks = blocks or blocks_for_event(cfg, event)
    valid = read_valid(cfg, event)
    with rasterio.open(ems.label_raster_path(cfg, event)) as src:
        label = src.read(1) > 0

    ha = blocks.grid.pixel_area_ha
    entries = []
    for name in blocks.names:
        inside = blocks.index == blocks.index_of(name)
        usable = inside & valid & (blocks.role != ROLE_CODES["buffer"])
        entries.append(
            {
                "block": name,
                "role": blocks.block_role[name],
                "pixels": int(inside.sum()),
                "usable_pixels": int(usable.sum()),
                "burned_pixels": int((usable & label).sum()),
                "burned_ha": round(float((usable & label).sum()) * ha, 1),
            }
        )

    per_role = {}
    for role in ("train", "calibration", "test"):
        mask = blocks.mask(role) & valid
        per_role[role] = {
            "blocks": blocks.blocks_with_role(role),
            "usable_pixels": int(mask.sum()),
            "burned_pixels": int((mask & label).sum()),
            "burned_ha": round(float((mask & label).sum()) * ha, 1),
        }

    dropped = float((blocks.role == ROLE_CODES["buffer"]).mean())
    return {
        "event": event.id,
        "grid": {
            "crs": blocks.grid.crs.to_string(),
            "width": blocks.grid.width,
            "height": blocks.grid.height,
            "resolution_m": blocks.grid.resolution,
        },
        "n_blocks": blocks.n_blocks,
        "buffer_fraction_of_footprint": round(dropped, 4),
        "valid_fraction_of_footprint": round(float(valid.mean()), 4),
        "roles": per_role,
        "blocks": entries,
    }


def write_inventory(cfg: Config, event_ids: list[str] | None = None) -> Path:
    cfg_events = event_ids or list(cfg.events)
    payload = {
        "superblock_km": cfg.evaluation["spatial_blocks"]["superblock_km"],
        "role_buffer_m": cfg.evaluation["spatial_blocks"].get("role_buffer_m", 0),
        "min_blocks_for_interval": cfg.evaluation["spatial_blocks"][
            "min_blocks_for_interval"
        ],
        "events": [],
    }
    for event_id in cfg_events:
        event = cfg.event(event_id)
        blocks = blocks_for_event(cfg, event)
        write_roles(cfg, event, blocks)
        payload["events"].append(inventory(cfg, event, blocks))

    dest = cfg.path_for("outputs", "spatial_blocks.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--event", default=None, help="default: every event")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    event_ids = [args.event] if args.event else list(cfg.events)
    dest = write_inventory(cfg, event_ids)
    payload = json.loads(dest.read_text(encoding="utf-8"))

    for entry in payload["events"]:
        print(f"{entry['event']}  --  {entry['n_blocks']} block(s) of "
              f"~{payload['superblock_km']} km")
        for block in entry["blocks"]:
            print(
                f"    {block['block']}  {block['role']:<11} "
                f"{block['usable_pixels']:>9,} usable px  "
                f"{block['burned_ha']:>9,.0f} ha burned"
            )
        for role, summary in entry["roles"].items():
            if summary["blocks"]:
                print(
                    f"  {role:<11} {len(summary['blocks'])} block(s), "
                    f"{summary['usable_pixels']:,} usable px, "
                    f"{summary['burned_ha']:,.0f} ha burned"
                )
        print(
            f"  buffer dropped {entry['buffer_fraction_of_footprint']:.1%} of the "
            f"footprint; cloud/shadow left "
            f"{entry['valid_fraction_of_footprint']:.1%} usable"
        )
    print(f"\nwritten: {dest}")


if __name__ == "__main__":  # pragma: no cover
    main()
