"""Tiles as tensors: normalisation, augmentation, and the validity weight.

Three decisions here are protocol rather than plumbing.

**Normalisation statistics come from the training pixels of the training event
and from nowhere else.** Not from the calibration blocks, not from the test
events, not from the pooled corpus. Per-channel means over a test event would
carry that event's radiometry into the network's input scaling, which is a leak
that never shows up as a crash and moves every downstream number a little.

**The validity mask travels with the tile as a loss weight.** Cloud and shadow
pixels are NaN in the stack; the network cannot be given NaN, so they are set to
zero *after* normalisation -- the per-channel mean, which is the least
informative value the input distribution has -- and their weight is set to zero
so no gradient is ever computed from them. Filling with the mean and then
learning from it would teach the network that mean reflectance means unburned,
which is a fact about the cloud mask and not about fire.

Adding a ninth "validity" channel was the alternative and was rejected: the
design fixes the input at 8 channels, and the dNBR gets no such channel, so it
would be information given to one method and not the other.

**Augmentation is the D4 group and nothing else** -- four rotations and their
mirrors. A burn scar has no canonical orientation, so D4 is label-preserving by
construction. Photometric jitter, the reflex augmentation, would be wrong on
calibrated surface reflectance: it would simulate radiometry that the sensor
cannot produce, and the dNBR baseline is a fixed function of that same
radiometry. Scaling and elastic warps would break the 20 m grid the whole
evaluation is defined on.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from ..config import Config, Event, load_config
from ..data import ems
from ..data.stack import stack_path, valid_path
from ..data.tiles import Tile, training_tiles
from ..eval.blocks import blocks_for_event


# --------------------------------------------------------------------------- #
# event arrays
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EventArrays:
    """One event held in memory: reflectance, label, validity."""

    event_id: str
    channels: tuple[str, ...]
    data: np.ndarray  # (C, H, W) float32, NaN where invalid
    label: np.ndarray  # (H, W) bool
    valid: np.ndarray  # (H, W) bool


@lru_cache(maxsize=4)
def _load(config_path: str, event_id: str) -> EventArrays:
    cfg = load_config(config_path)
    event = cfg.event(event_id)
    with rasterio.open(stack_path(cfg, event)) as src:
        data = src.read().astype("float32")
        channels = tuple(src.descriptions)
    with rasterio.open(valid_path(cfg, event)) as src:
        valid = src.read(1).astype(bool)
    with rasterio.open(ems.label_raster_path(cfg, event)) as src:
        label = src.read(1) > 0
    return EventArrays(
        event_id=event_id, channels=channels, data=data, label=label, valid=valid
    )


def load_event(cfg: Config, event: Event) -> EventArrays:
    """The whole event in memory, cached across datasets and inference passes."""
    return _load(str(cfg.path), event.id)


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Normalisation:
    """Per-channel statistics, and a record of exactly which pixels made them."""

    channels: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    fitted_on: str
    n_pixels: int

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Standardise a (C, H, W) block; invalid pixels land on zero."""
        mean = np.asarray(self.mean, dtype="float32")[:, None, None]
        std = np.asarray(self.std, dtype="float32")[:, None, None]
        out = (data.astype("float32") - mean) / std
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def as_dict(self) -> dict:
        return asdict(self)


def fit_normalisation(cfg: Config, role: str = "train") -> Normalisation:
    """Fit on one role of the training event. Never on a test event."""
    event = next(e for e in cfg.events.values() if e.is_train)
    arrays = load_event(cfg, event)
    blocks = blocks_for_event(cfg, event)
    mask = blocks.mask(role) & arrays.valid
    if not mask.any():
        raise ValueError(f"{event.id}: no usable pixel with role {role!r} to fit on")

    pixels = arrays.data[:, mask]
    mean = np.nanmean(pixels, axis=1)
    std = np.nanstd(pixels, axis=1)
    # A dead channel would divide by zero and produce a silently constant input.
    if not np.all(std > 0):
        dead = [arrays.channels[i] for i in np.flatnonzero(std <= 0)]
        raise ValueError(f"{event.id}: channels with zero variance: {dead}")

    return Normalisation(
        channels=tuple(arrays.channels),
        mean=tuple(float(v) for v in mean),
        std=tuple(float(v) for v in std),
        fitted_on=(
            f"{event.id}, {role} blocks "
            f"({', '.join(blocks.blocks_with_role(role))}), cloud-free pixels only"
        ),
        n_pixels=int(mask.sum()),
    )


def normalisation_path(cfg: Config) -> Path:
    return cfg.path_for("outputs", "normalisation.json")


def save_normalisation(cfg: Config, norm: Normalisation) -> Path:
    """Versioned, like the thresholds: a reader can see what scaling was used."""
    dest = normalisation_path(cfg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(norm.as_dict(), indent=2) + "\n", encoding="utf-8")
    return dest


def load_normalisation(cfg: Config) -> Normalisation:
    payload = json.loads(normalisation_path(cfg).read_text(encoding="utf-8"))
    return Normalisation(
        channels=tuple(payload["channels"]),
        mean=tuple(payload["mean"]),
        std=tuple(payload["std"]),
        fitted_on=payload["fitted_on"],
        n_pixels=int(payload["n_pixels"]),
    )


# --------------------------------------------------------------------------- #
# augmentation
# --------------------------------------------------------------------------- #

# The dihedral group of the square: k quarter-turns, optionally mirrored.
D4 = tuple((k, flip) for k in range(4) for flip in (False, True))


def apply_d4(arrays: list[np.ndarray], element: tuple[int, bool]) -> list[np.ndarray]:
    """Apply one D4 element to every array, identically. Last two axes only."""
    k, flip = element
    out = [np.rot90(a, k=k, axes=(-2, -1)) for a in arrays]
    if flip:
        out = [a[..., ::-1] for a in out]
    return [np.ascontiguousarray(a) for a in out]


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #


class TileDataset(Dataset):
    """Tiles of one event, standardised, optionally augmented.

    Yields ``(x, y, w)``: reflectance ``(C, H, W)``, label ``(1, H, W)``, and the
    validity weight ``(1, H, W)`` that keeps clouded pixels out of the loss.
    """

    def __init__(
        self,
        cfg: Config,
        event: Event,
        tiles: list[Tile],
        normalisation: Normalisation,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        if not tiles:
            raise ValueError(f"{event.id}: empty tile set")
        self.cfg = cfg
        self.event = event
        self.tiles = list(tiles)
        self.normalisation = normalisation
        self.augment = augment
        self.arrays = load_event(cfg, event)
        if tuple(self.arrays.channels) != tuple(normalisation.channels):
            raise ValueError(
                f"{event.id}: channel order {self.arrays.channels} does not match the "
                f"normalisation fitted on {normalisation.channels}"
            )
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.tiles)

    @property
    def n_positive(self) -> int:
        return sum(1 for t in self.tiles if t.is_positive)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tile = self.tiles[index]
        rows, cols = tile.slices
        x = self.normalisation.apply(self.arrays.data[:, rows, cols])
        y = self.arrays.label[rows, cols][None].astype("float32")
        w = self.arrays.valid[rows, cols][None].astype("float32")

        if self.augment:
            element = D4[int(self._rng.integers(len(D4)))]
            x, y, w = apply_d4([x, y, w], element)

        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w)


def training_dataset(
    cfg: Config,
    event: Event,
    normalisation: Normalisation,
    role: str = "train",
    augment: bool = True,
) -> TileDataset:
    return TileDataset(
        cfg,
        event,
        training_tiles(cfg, event, role=role),
        normalisation,
        augment=augment,
        seed=int(cfg.model["seed"]),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    norm = fit_normalisation(cfg)
    path = save_normalisation(cfg, norm)

    train_event = next(e for e in cfg.events.values() if e.is_train)
    dataset = training_dataset(cfg, train_event, norm)
    x, y, w = dataset[0]

    print(f"fitted on  : {norm.fitted_on}")
    print(f"pixels     : {norm.n_pixels:,}")
    for name, mean, std in zip(norm.channels, norm.mean, norm.std):
        print(f"  {name:14s} mean {mean:8.4f}  std {std:8.4f}")
    print(f"tiles      : {len(dataset)} ({dataset.n_positive} positive)")
    print(f"batch shape: x {tuple(x.shape)}  y {tuple(y.shape)}  w {tuple(w.shape)}")
    print(f"written    : {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
