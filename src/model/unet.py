"""The network, built from config and not reimplemented.

``segmentation_models_pytorch`` supplies the architecture. Writing a U-Net by
hand proves nothing that is in question here -- the question this project asks
is about evaluation, and a hand-rolled decoder would only add a surface for
silent bugs to live on.

**On the pretrained encoder.** ``in_channels=8`` makes smp inflate the first
convolution: the pretrained RGB kernels are repeated and rescaled across the
eight channels. On SWIR bands that inflation is close to meaningless -- an
ImageNet first layer encodes opponent-colour and edge filters tuned to visible
light, and B11/B12 are not visible light. The benefit that does survive is in
the deeper layers, whose texture and shape priors are largely wavelength-blind,
and it arrives as a regularising initialisation. On a training set of 70 tiles
that regularisation is worth more than the first layer ever could be.
"""

from __future__ import annotations

import argparse

import segmentation_models_pytorch as smp
import torch

from ..config import Config, load_config

ARCHITECTURES = {"unet": smp.Unet}


def build(cfg: Config) -> torch.nn.Module:
    """The model named by ``model:`` in config.yaml."""
    settings = cfg.model
    architecture = str(settings["architecture"]).lower()
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"model.architecture={architecture!r} is not one of {sorted(ARCHITECTURES)}"
        )
    expected = len(cfg.band_assets) * 2
    if int(settings["in_channels"]) != expected:
        raise ValueError(
            f"model.in_channels={settings['in_channels']} but the stack carries "
            f"{expected} channels ({len(cfg.band_assets)} bands x 2 dates)"
        )
    return ARCHITECTURES[architecture](
        encoder_name=str(settings["encoder"]),
        encoder_weights=settings.get("encoder_weights"),
        in_channels=int(settings["in_channels"]),
        classes=int(settings["classes"]),
    )


def n_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    model = build(cfg)
    size = int(cfg.project["tile_size_px"])
    channels = int(cfg.model["in_channels"])

    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(2, channels, size, size))

    print(f"model      : {cfg.model['architecture']} / {cfg.model['encoder']}")
    print(f"weights    : {cfg.model.get('encoder_weights')}")
    print(f"parameters : {n_parameters(model) / 1e6:.2f} M")
    print(f"forward    : (2, {channels}, {size}, {size}) -> {tuple(logits.shape)}")


if __name__ == "__main__":  # pragma: no cover
    main()
