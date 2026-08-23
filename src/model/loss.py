"""BCE + Dice, both masked by the validity weight.

The two terms do different jobs and the design note is explicit about why both
are there: BCE stabilises the start of training, when almost nothing is
predicted positive and the Dice gradient is nearly flat; Dice then carries the
class imbalance, which BCE alone handles badly at ~2% prevalence.

**Dice is computed over the batch, not per tile.** A per-tile Dice on a wholly
unburned tile is 0/0, and whatever value the smoothing term assigns to it is an
arbitrary constant that the optimiser will happily chase. Half the training
tiles here are unburned, so per-tile averaging would spend half the gradient on
an artefact of the smoothing. Pooling the batch makes those tiles contribute
what they actually should: false-positive area in the denominator.

**Every pixel enters through the same weight.** Clouded pixels have weight zero
in both terms, so they contribute neither loss nor gradient -- they are not
pixels the network is being asked to get right.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..config import Config


@dataclass(frozen=True)
class BceDiceLoss:
    """The weighted sum named by ``model.loss`` in config.yaml."""

    bce_weight: float = 1.0
    dice_weight: float = 1.0
    smooth: float = 1.0

    @classmethod
    def from_config(cls, cfg: Config) -> "BceDiceLoss":
        settings = cfg.model["loss"]
        return cls(
            bce_weight=float(settings["bce_weight"]),
            dice_weight=float(settings["dice_weight"]),
        )

    def __call__(
        self, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Returns the total and its two parts, so the curves can be read apart."""
        total_weight = weight.sum().clamp_min(1.0)

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        bce = (bce * weight).sum() / total_weight

        probability = torch.sigmoid(logits) * weight
        truth = target * weight
        intersection = (probability * truth).sum()
        denominator = probability.sum() + truth.sum()
        dice = 1.0 - (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        return {
            "loss": self.bce_weight * bce + self.dice_weight * dice,
            "bce": bce.detach(),
            "dice": dice.detach(),
        }
