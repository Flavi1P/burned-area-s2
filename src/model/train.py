"""The training loop for one experiment.

**What the model is selected on, and why it is not the test set.** Every epoch
is scored on the calibration blocks of the training event -- the same blocks
that will later produce the decision threshold, held out from both training and
test -- using average precision, which is threshold-free. Selecting on a
thresholded metric would fold the choice of epoch into the choice of threshold,
and those two are kept apart deliberately.

The consequence, declared rather than buried: the calibration blocks do double
duty here, model selection and threshold calibration, so the F1 the threshold
reports on them is mildly optimistic about itself. Neither use touches a test
pixel, which is the property the protocol actually needs. The alternative --
carving a fourth role out of Saumos -- was rejected: the calibration blocks
already hold only 666 burned hectares after the buffer, and splitting them again
would leave both halves too thin to do either job.

**The validation pass runs the inference regime, not the training regime.** It
uses the non-overlapping paving and scores the pooled calibration pixels, so the
number that selects the epoch is produced the same way as the number in the
results table. An epoch selected on an overlapping, cloud-filtered tile set
would be selected on a distribution the model never meets again.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from ..config import Config, Event, Experiment, load_config
from ..data.tiles import Tile, test_tiles
from ..eval.blocks import blocks_for_event
from . import unet
from .dataset import (
    EventArrays,
    Normalisation,
    fit_normalisation,
    load_event,
    save_normalisation,
    training_dataset,
)
from .loss import BceDiceLoss
from .predict import weights_path


def history_path(cfg: Config, experiment_id: str) -> Path:
    return cfg.path_for("outputs", f"training_{experiment_id.lower()}.json")


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


def validation_tiles(cfg: Config, event: Event, role: str) -> list[Tile]:
    """Inference-regime tiles covering the pixels the threshold will be fitted on.

    Every test tile whose owned strip touches a pixel of ``role``. Restricting to
    tiles *contained* in the role would have thrown away almost all of it: the
    calibration blocks hold their burned ground near their own edges, and only
    one contained tile carries any burn at all.
    """
    mask = blocks_for_event(cfg, event).mask(role)
    kept = []
    for tile in test_tiles(cfg, event):
        rows, cols = tile.owned_slices
        if mask[rows, cols].any():
            kept.append(tile)
    return kept


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    arrays: EventArrays,
    normalisation: Normalisation,
    tiles: list[Tile],
    role_mask: np.ndarray,
    criterion: BceDiceLoss,
    batch_size: int,
) -> dict[str, float]:
    """Score the pooled pixels of one role. Owned strips only, so nothing counts
    twice."""
    model.eval()
    scores: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    losses: list[float] = []

    for start in range(0, len(tiles), batch_size):
        batch = tiles[start : start + batch_size]
        x = np.stack(
            [
                normalisation.apply(arrays.data[(slice(None), *tile.slices)])
                for tile in batch
            ]
        )
        y = np.stack([arrays.label[tile.slices][None] for tile in batch]).astype("float32")
        # Weight = usable AND inside the role. Pixels of another role sit in the
        # tile because the paving is geometric; they must not enter the metric.
        w = np.stack(
            [(arrays.valid & role_mask)[tile.slices][None] for tile in batch]
        ).astype("float32")

        logits = model(torch.from_numpy(x))
        parts = criterion(logits, torch.from_numpy(y), torch.from_numpy(w))
        losses.append(float(parts["loss"]))

        probability = torch.sigmoid(logits).numpy()[:, 0]
        for tile, prediction in zip(batch, probability):
            rows, cols = tile.owned_slices
            inner = (
                slice(rows.start - tile.row, rows.stop - tile.row),
                slice(cols.start - tile.col, cols.stop - tile.col),
            )
            keep = (arrays.valid & role_mask)[rows, cols]
            if keep.any():
                scores.append(prediction[inner][keep])
                truths.append(arrays.label[rows, cols][keep])

    score = np.concatenate(scores)
    truth = np.concatenate(truths)
    return {
        "loss": float(np.mean(losses)),
        "average_precision": float(average_precision_score(truth, score)),
        "pixels": int(truth.size),
        "burned_pixels": int(truth.sum()),
        "mean_probability": float(score.mean()),
    }


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #


def train(cfg: Config, experiment: Experiment, quick: int | None = None) -> dict:
    """Fit one experiment and keep the epoch that scores best on calibration."""
    if len(experiment.train) != 1:
        raise ValueError(
            f"{experiment.id}: this loop trains on exactly one event, got "
            f"{experiment.train}"
        )
    settings = cfg.model
    seed = int(settings["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    event = cfg.event(experiment.train[0])
    calibration_role = cfg.evaluation["threshold"]["block_role"]

    normalisation = fit_normalisation(cfg)
    save_normalisation(cfg, normalisation)

    dataset = training_dataset(cfg, event, normalisation, augment=settings["augment"] == "d4")
    loader = DataLoader(
        dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        drop_last=False,
        num_workers=int(settings.get("num_workers", 0)),
        generator=torch.Generator().manual_seed(seed),
    )

    arrays = load_event(cfg, event)
    role_mask = blocks_for_event(cfg, event).mask(calibration_role)
    val_tiles = validation_tiles(cfg, event, calibration_role)

    model = unet.build(cfg)
    criterion = BceDiceLoss.from_config(cfg)
    epochs = int(quick or settings["epochs"])
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=float(settings["lr"]),
        weight_decay=float(settings.get("weight_decay", 0.0)),
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
        if settings.get("lr_schedule") == "cosine"
        else None
    )

    selection_metric = str(settings.get("select_on", "average_precision"))
    history: list[dict] = []
    best = {"epoch": -1, "value": -np.inf}
    best_state: dict | None = None
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"loss": 0.0, "bce": 0.0, "dice": 0.0}
        for x, y, w in loader:
            optimiser.zero_grad(set_to_none=True)
            parts = criterion(model(x), y, w)
            parts["loss"].backward()
            optimiser.step()
            for key in totals:
                totals[key] += float(parts[key].detach()) * x.shape[0]
        for key in totals:
            totals[key] /= len(dataset)

        validation = validate(
            model,
            arrays,
            normalisation,
            val_tiles,
            role_mask,
            criterion,
            int(settings["batch_size"]),
        )
        if scheduler is not None:
            scheduler.step()

        entry = {
            "epoch": epoch,
            "lr": optimiser.param_groups[0]["lr"],
            "train": {k: round(v, 5) for k, v in totals.items()},
            "validation": {
                k: (round(v, 5) if isinstance(v, float) else v)
                for k, v in validation.items()
            },
            "seconds": round(time.time() - started, 1),
        }
        history.append(entry)

        if validation[selection_metric] > best["value"]:
            best = {"epoch": epoch, "value": validation[selection_metric]}
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        print(
            f"  epoch {epoch:3d}/{epochs}  train {totals['loss']:.4f}  "
            f"val {validation['loss']:.4f}  val AP {validation['average_precision']:.4f}"
            f"{'  *' if best['epoch'] == epoch else ''}"
        )

    payload = {
        "experiment": experiment.id,
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "train_event": event.id,
        "epochs": epochs,
        "seed": seed,
        "architecture": f"{settings['architecture']}/{settings['encoder']}",
        "encoder_weights": settings.get("encoder_weights"),
        "parameters": unet.n_parameters(model),
        "loss": dict(settings["loss"]),
        "augment": settings["augment"],
        "optimiser": {
            "name": "adam",
            "lr": float(settings["lr"]),
            "weight_decay": float(settings.get("weight_decay", 0.0)),
            "schedule": settings.get("lr_schedule"),
        },
        "train_tiles": {
            "role": "train",
            "tiles": len(dataset),
            "positive": dataset.n_positive,
            "overlap_px": int(cfg.evaluation["tiling"]["train"]["overlap_px"]),
        },
        "selection": {
            "metric": selection_metric,
            "role": calibration_role,
            "blocks": blocks_for_event(cfg, event).blocks_with_role(calibration_role),
            "tiles": len(val_tiles),
            "pixels": history[-1]["validation"]["pixels"],
            "burned_pixels": history[-1]["validation"]["burned_pixels"],
            "best_epoch": best["epoch"],
            "best_value": round(float(best["value"]), 5),
        },
        "normalisation": normalisation.as_dict(),
        "wall_seconds": round(time.time() - started, 1),
        "history": history,
    }

    dest = weights_path(cfg, experiment.id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "experiment": experiment.id,
            "epoch": best["epoch"],
            "selection": payload["selection"],
            "normalisation": normalisation.as_dict(),
            "config": str(cfg.path.name),
        },
        dest,
    )
    history_path(cfg, experiment.id).parent.mkdir(parents=True, exist_ok=True)
    history_path(cfg, experiment.id).write_text(
        json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
    )
    payload["weights"] = str(dest)
    return payload


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--experiment", default="E1")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override model.epochs; for smoke tests only",
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    experiment = cfg.experiment(args.experiment)
    print(
        f"experiment : {experiment.id}  train {experiment.train} -> test "
        f"{experiment.test}"
    )
    payload = train(cfg, experiment, quick=args.epochs)

    selection = payload["selection"]
    print(
        f"selected   : epoch {selection['best_epoch']} of {payload['epochs']}, "
        f"{selection['metric']} = {selection['best_value']:.4f} on the "
        f"{selection['role']} blocks ({', '.join(selection['blocks'])})"
    )
    print(f"wall time  : {payload['wall_seconds'] / 60:.1f} min")
    print(f"weights    : {payload['weights']}")
    print(f"history    : {history_path(cfg, experiment.id)}")


if __name__ == "__main__":  # pragma: no cover
    main()
