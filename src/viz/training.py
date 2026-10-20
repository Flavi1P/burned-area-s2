"""Training curves, every seed: the loss on the left, the metric that chose the epoch on the right.

Two panels rather than one, because they answer different questions. The loss
panel shows whether the fit converged; the selection panel shows *which* epoch
was kept and on what evidence, with the chosen epoch marked. A reader who
suspects the model was picked on the test set can see from this figure that it
was not: the curve that drives the choice is measured on the calibration blocks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..config import Config, load_config  # noqa: E402
from ..model.train import history_path  # noqa: E402


def curve_figure(
    cfg: Config, experiment_id: str = "E1", dest: Path | None = None
) -> Path:
    """Every seed on the same axes, so the spread is visible and none is hidden."""
    payloads = []
    for seed in cfg.model_seeds:
        path = history_path(cfg, experiment_id, seed)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing: run `python -m src.model.train --experiment "
                f"{experiment_id} --seed {seed}` first"
            )
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    first = payloads[0]
    selection = first["selection"]
    colours = plt.get_cmap("tab10").colors

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for payload, colour in zip(payloads, colours):
        history = payload["history"]
        epochs = [h["epoch"] for h in history]
        axes[0].plot(epochs, [h["train"]["loss"] for h in history], color=colour, lw=1.2)
        axes[0].plot(
            epochs, [h["validation"]["loss"] for h in history], color=colour, lw=1.2,
            ls="--",
        )
        chosen = payload["selection"]
        axes[1].plot(
            epochs,
            [h["validation"]["average_precision"] for h in history],
            color=colour,
            lw=1.4,
            label=f"seed {payload['seed']}: epoch {chosen['best_epoch']}, "
            f"AP {chosen['best_value']:.3f}",
        )
        axes[1].plot([chosen["best_epoch"]], [chosen["best_value"]], "o", color=colour, ms=5)

    ax = axes[0]
    ax.plot([], [], color="0.3", lw=1.2, label="train")
    ax.plot([], [], color="0.3", lw=1.2, ls="--", label=f"{selection['role']} blocks")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE + Dice")
    ax.set_title("Loss, one colour per seed", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)

    ax = axes[1]
    ax.set_xlabel("epoch")
    ax.set_ylabel("average precision")
    ax.set_title(
        f"Model selection — {selection['role']} blocks "
        f"({', '.join(selection['blocks'])}), threshold-free",
        loc="left",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(alpha=0.25, lw=0.5)

    fig.suptitle(
        f"{first['experiment']} — {first['architecture']}, "
        f"{first['train_tiles']['tiles']} training tiles "
        f"({first['train_tiles']['positive']} positive), "
        f"{first['epochs']} epochs on CPU, {len(payloads)} seeds",
        fontsize=11,
        x=0.005,
        ha="left",
    )

    dest = dest or cfg.path_for("outputs", f"training_{experiment_id.lower()}.png")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--experiment", default="E1")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    print(f"written    : {curve_figure(cfg, args.experiment)}")


if __name__ == "__main__":  # pragma: no cover
    main()
