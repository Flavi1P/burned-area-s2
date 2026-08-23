"""Training curves: the loss on the left, the metric that chose the epoch on the right.

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
    path = history_path(cfg, experiment_id)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing: run `python -m src.model.train --experiment "
            f"{experiment_id}` first"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload["history"]
    epochs = [h["epoch"] for h in history]
    selection = payload["selection"]
    best = selection["best_epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(epochs, [h["train"]["loss"] for h in history], label="train", lw=1.8)
    ax.plot(
        epochs,
        [h["validation"]["loss"] for h in history],
        label=f"{selection['role']} blocks",
        lw=1.8,
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE + Dice")
    ax.set_title("Loss", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)

    ax = axes[1]
    values = [h["validation"]["average_precision"] for h in history]
    ax.plot(epochs, values, color="#b2182b", lw=1.8)
    ax.axvline(best, color="0.35", ls="--", lw=1.0)
    ax.plot([best], [selection["best_value"]], "o", color="#b2182b", ms=6)
    ax.annotate(
        f"epoch {best}\nAP = {selection['best_value']:.3f}",
        xy=(best, selection["best_value"]),
        xytext=(6, -28),
        textcoords="offset points",
        fontsize=9,
        color="0.25",
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("average precision")
    ax.set_title(
        f"Model selection — {selection['role']} blocks "
        f"({', '.join(selection['blocks'])}), threshold-free",
        loc="left",
        fontsize=11,
    )
    ax.grid(alpha=0.25, lw=0.5)

    fig.suptitle(
        f"{payload['experiment']} — {payload['architecture']}, "
        f"{payload['train_tiles']['tiles']} training tiles "
        f"({payload['train_tiles']['positive']} positive), "
        f"{payload['epochs']} epochs on CPU",
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
