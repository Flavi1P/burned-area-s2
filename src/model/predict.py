"""Inference: one continuous probability raster per event.

This is the whole interface between the network and the evaluation. The U-Net
hands over a score per pixel, exactly as the dNBR does, and everything after
this point -- threshold, metrics, bootstrap, table -- is code that already
existed and ran before the network did.

The paving is the non-overlapping test regime of ``src.data.tiles``, and each
tile writes only the strip it owns, so every pixel is predicted exactly once. No
test-time augmentation and no overlap averaging: both would improve the numbers,
and both would break the symmetry the comparison rests on, since the dNBR gets
neither. The tile-seam cost is left in and paid by the network.

Invalid pixels come back NaN, the same way the dNBR does, so the two methods are
scored on exactly the same pixel set rather than on two sets that happen to look
alike.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch

from ..config import Config, Event, load_config
from ..data.grid import grid_for_event
from ..data.tiles import Tile, test_tiles
from . import unet
from .dataset import EventArrays, Normalisation, load_event, load_normalisation


def probability_path(cfg: Config, event: Event) -> Path:
    return cfg.path_for("data_interim", event.id, "unet_prob.tif")


def weights_path(cfg: Config, experiment_id: str) -> Path:
    return cfg.path_for("models", f"unet_{experiment_id.lower()}.pt")


def load_model(cfg: Config, path: Path) -> tuple[torch.nn.Module, dict]:
    """Rebuild the architecture from config and load frozen weights into it."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = unet.build(cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict_tiles(
    model: torch.nn.Module,
    arrays: EventArrays,
    normalisation: Normalisation,
    tiles: list[Tile],
    batch_size: int = 8,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Sigmoid probabilities over a set of tiles, on the event's own grid."""
    height, width = arrays.valid.shape
    if out is None:
        out = np.full((height, width), np.nan, dtype="float32")

    model.eval()
    for start in range(0, len(tiles), batch_size):
        batch = tiles[start : start + batch_size]
        stacked = np.stack(
            [
                normalisation.apply(arrays.data[(slice(None), *tile.slices)])
                for tile in batch
            ]
        )
        probability = torch.sigmoid(model(torch.from_numpy(stacked))).numpy()[:, 0]
        for tile, prediction in zip(batch, probability):
            rows, cols = tile.owned_slices
            # The owned strip, expressed inside the tile's own frame.
            out[rows, cols] = prediction[
                rows.start - tile.row : rows.stop - tile.row,
                cols.start - tile.col : cols.stop - tile.col,
            ]

    out[~arrays.valid] = np.nan
    return out


def predict_event(
    cfg: Config,
    event: Event,
    model: torch.nn.Module,
    normalisation: Normalisation | None = None,
) -> np.ndarray:
    """The full probability raster of one event, over its whole footprint."""
    normalisation = normalisation or load_normalisation(cfg)
    arrays = load_event(cfg, event)
    return predict_tiles(
        model,
        arrays,
        normalisation,
        test_tiles(cfg, event),
        batch_size=int(cfg.model["batch_size"]),
    )


def write_probability(
    cfg: Config, event: Event, probability: np.ndarray, tags: dict | None = None
) -> Path:
    dest = probability_path(cfg, event)
    dest.parent.mkdir(parents=True, exist_ok=True)
    grid = grid_for_event(cfg, event)
    with rasterio.open(dest, "w", **grid.profile("float32", nodata=float("nan"))) as dst:
        dst.write(probability.astype("float32"), 1)
        dst.set_band_description(1, "burned probability")
        dst.update_tags(event=event.id, **{k: str(v) for k, v in (tags or {}).items()})
    return dest


def read_probability(cfg: Config, event: Event) -> np.ndarray:
    path = probability_path(cfg, event)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing: run `python -m src.model.predict --event {event.id}`"
        )
    with rasterio.open(path) as src:
        return src.read(1).astype("float32")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--experiment", default="E1", help="which trained weights")
    parser.add_argument("--event", action="append", default=None, help="default: all")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    path = weights_path(cfg, args.experiment)
    if not path.exists():
        raise SystemExit(
            f"{path} is missing: run `python -m src.model.train --experiment "
            f"{args.experiment}` first"
        )
    model, checkpoint = load_model(cfg, path)
    normalisation = load_normalisation(cfg)

    print(f"weights    : {path.name}  (epoch {checkpoint.get('epoch')})")
    for event_id in args.event or list(cfg.events):
        event = cfg.event(event_id)
        probability = predict_event(cfg, event, model, normalisation)
        dest = write_probability(
            cfg,
            event,
            probability,
            tags={
                "experiment": args.experiment,
                "weights": path.name,
                "selected_epoch": checkpoint.get("epoch"),
                "tiling": "test regime, no overlap, no test-time augmentation",
            },
        )
        finite = np.isfinite(probability)
        print(
            f"  {event.id:14s} {int(finite.sum()):>9,} px predicted  "
            f"mean p {float(np.nanmean(probability)):.4f}  -> {dest}"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
