"""The per-pixel multivariate control, and the reason it had to exist.

The original comparison had two rungs and a gap between them. The dNBR is two
bands and one threshold. The U-Net is eight bands and a receptive field of a
few hundred metres. When the network wins, *both* differences are available to
explain it, and the README's pine argument silently attributes the whole gain
to the second one. Nothing in the design could tell them apart, because nothing
in the design sat in between.

This module is the rung in between: every band the network sees, per pixel,
with no spatial context whatsoever. Histogram gradient boosting over the eight
reflectance channels and the indices derived from them, fitted on the same
train-role pixels the network was fitted on, thresholded by the same function
on the same calibration blocks, scored by the same metric code. The only thing
it does not have is context. So whatever it recovers of the network's lead was
never context to begin with, and whatever it fails to recover is the honest
size of the spatial argument.

**It is a control, not a bid.** If it beats the U-Net, that is a finding about
the U-Net, not a new champion to promote -- the interesting quantity is the
difference between the two, and it is only interesting because everything else
about them is held equal.

**Where the training pixels come from.** The train-role blocks of the training
event, and nothing else. Those blocks are already disjoint from the calibration
and test blocks and separated from them by the 1 km role buffer, so this model
inherits the separation guarantee ``src/eval/blocks.py`` provides without
needing a tiling rule of its own -- and unlike the U-Net it cannot straddle a
seam, because a pixel is not a tile.

**Unusable pixels are neither dropped nor imputed.** The histogram splitter
routes NaN natively, so a pixel whose SWIR is missing still receives a
probability. That is what keeps this method's evaluation domain identical to
the dNBR's, pixel for pixel, which is the property the whole comparison rests
on. Imputing would have invented reflectance; dropping would have made the
domains differ and quietly broken the comparison instead of the test.

CatBoost, LightGBM and this estimator are the same algorithm. scikit-learn is
already a dependency and the other two are not, and no conclusion here turns on
the third decimal, so the dependency is not worth adding. ``config.yaml``
records which estimator produced the numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from ..config import Config, Event, load_config
from ..data import ems
from ..data.grid import Grid, grid_for_event
from ..data.stack import Stack, build_stack, read_valid
from .blocks import blocks_for_event
from .dnbr import NIR, SWIR2, nbr

METHOD = "GB pixel"
SCORE = "gb"


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #


def _normalised_difference(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        out = ((x - y) / (x + y)).astype("float32")
    out[(x + y) == 0] = np.nan
    return out


def features(stack: Stack) -> dict[str, np.ndarray]:
    """Every feature ``config.yaml`` is allowed to name, computed once.

    Three families, and the split between them is what the ``change_only``
    feature set in ``config.yaml`` exists to test:

    * **differences** -- ``dnbr``, ``dndvi``, ``dnbr2`` and the four raw band
      deltas. These describe what changed, and a change of a given size means
      roughly the same thing in pine and in oak.
    * **pre-fire context** -- ``pre_nbr``, ``pre_ndvi``. How much there was to
      lose, which is what the relativised indices of the literature divide by.
    * **absolute post-fire levels** -- ``post_*``. The most informative family
      in-domain and the most biome-specific: what a burned oak stand reflects
      is not what a burned pine plantation reflects, so a split learned on one
      is a guess on the other.
    """
    bands = {
        (phase, asset): stack.band(phase, asset)
        for phase in ("pre", "post")
        for asset in ("red", "nir08", "swir16", "swir22")
    }
    pre_nbr = nbr(bands[("pre", NIR)], bands[("pre", SWIR2)])
    post_nbr = nbr(bands[("post", NIR)], bands[("post", SWIR2)])
    pre_ndvi = _normalised_difference(bands[("pre", "nir08")], bands[("pre", "red")])
    post_ndvi = _normalised_difference(bands[("post", "nir08")], bands[("post", "red")])
    pre_nbr2 = _normalised_difference(bands[("pre", "swir16")], bands[("pre", "swir22")])
    post_nbr2 = _normalised_difference(
        bands[("post", "swir16")], bands[("post", "swir22")]
    )

    computed = {
        "dnbr": (pre_nbr - post_nbr).astype("float32"),
        "pre_nbr": pre_nbr,
        "post_nbr": post_nbr,
        "dndvi": (pre_ndvi - post_ndvi).astype("float32"),
        "pre_ndvi": pre_ndvi,
        "post_ndvi": post_ndvi,
        "dnbr2": (pre_nbr2 - post_nbr2).astype("float32"),
        "post_nbr2": post_nbr2,
    }
    for asset in ("red", "nir08", "swir16", "swir22"):
        computed[f"d_{asset}"] = (
            bands[("post", asset)] - bands[("pre", asset)]
        ).astype("float32")
        computed[f"post_{asset}"] = bands[("post", asset)].astype("float32")
    return computed


def feature_names(cfg: Config, feature_set: str) -> list[str]:
    sets = cfg.pixel_model["features"]
    try:
        names = list(sets[feature_set])
    except KeyError:
        raise ValueError(
            f"unknown feature set {feature_set!r}; config.yaml declares "
            f"{', '.join(sorted(sets))}"
        ) from None
    return names


def design_matrix(
    cfg: Config, event: Event, feature_set: str, grid: Grid | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """``(n_pixels, n_features)`` for every valid pixel of one event.

    Returns the matrix, the boolean mask of the pixels it covers, and the
    column names in the order ``config.yaml`` lists them -- the order is part of
    the model's contract and must not depend on a dict's iteration order.
    """
    grid = grid or grid_for_event(cfg, event)
    computed = features(build_stack(cfg, event, grid))
    names = feature_names(cfg, feature_set)
    unknown = [n for n in names if n not in computed]
    if unknown:
        raise ValueError(
            f"config.yaml names features this module does not compute: "
            f"{', '.join(unknown)}"
        )
    # Validity is the shared cloud/shadow mask and nothing else. A pixel whose
    # features are partly NaN inside that mask still gets a row, because the
    # splitter can route it -- see the module docstring.
    valid = read_valid(cfg, event)
    matrix = np.stack([computed[n][valid] for n in names], axis=1)
    return matrix, valid, names


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #


def training_pixels(
    cfg: Config, feature_set: str
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """The train-role pixels of the training event, and nothing else."""
    settings = cfg.pixel_model
    event = cfg.event(cfg.evaluation["threshold"]["calibrated_on"])
    if not event.is_train:
        raise ValueError(
            f"{event.id} is not the training event: fitting a pixel model on a test "
            "event is the leak the whole protocol exists to prevent"
        )
    grid = grid_for_event(cfg, event)
    blocks = blocks_for_event(cfg, event, grid)
    matrix, valid, names = design_matrix(cfg, event, feature_set, grid)

    with rasterio.open(ems.label_raster_path(cfg, event)) as src:
        label = src.read(1) > 0

    # `valid` indexes the matrix rows; restrict them to the train role.
    train = blocks.mask("train")[valid]
    matrix, truth = matrix[train], label[valid][train]

    rng = np.random.default_rng(int(settings["seed"]))
    cap = int(settings["max_train_pixels"])
    if matrix.shape[0] > cap:
        keep = rng.choice(matrix.shape[0], size=cap, replace=False)
        matrix, truth = matrix[keep], truth[keep]

    provenance = {
        "event": event.id,
        "role": "train",
        "blocks": blocks.blocks_with_role("train"),
        "pixels": int(matrix.shape[0]),
        "burned_pixels": int(truth.sum()),
        "prevalence": round(float(truth.mean()), 5),
        "features": names,
        "feature_set": feature_set,
    }
    return matrix, truth, names, provenance


def fit(cfg: Config, feature_set: str | None = None):
    """Train the booster. No test event is opened anywhere in this call."""
    settings = cfg.pixel_model
    feature_set = feature_set or settings["reported"]
    matrix, truth, names, provenance = training_pixels(cfg, feature_set)

    estimator = HistGradientBoostingClassifier(
        max_iter=int(settings["max_iter"]),
        learning_rate=float(settings["learning_rate"]),
        max_leaf_nodes=int(settings["max_leaf_nodes"]),
        random_state=int(settings["seed"]),
    )
    estimator.fit(matrix, truth)
    estimator.feature_names_in_order_ = names  # type: ignore[attr-defined]
    return estimator, provenance


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #


def probability_path(cfg: Config, event: Event, feature_set: str) -> Path:
    return cfg.path_for("data_interim", event.id, f"gb_prob_{feature_set}.tif")


def predict(
    cfg: Config, estimator, event: Event, feature_set: str
) -> tuple[Path, np.ndarray]:
    """One probability raster, NaN outside the shared validity mask."""
    grid = grid_for_event(cfg, event)
    matrix, valid, _ = design_matrix(cfg, event, feature_set, grid)

    score = np.full(grid.shape, np.nan, dtype="float32")
    score[valid] = estimator.predict_proba(matrix)[:, 1].astype("float32")

    dest = probability_path(cfg, event, feature_set)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dest, "w", **grid.profile("float32", nodata=float("nan"))) as dst:
        dst.write(score, 1)
        dst.set_band_description(1, "P(burned)")
        dst.update_tags(
            event=event.id,
            method=METHOD,
            estimator=cfg.pixel_model["estimator"],
            feature_set=feature_set,
            trained_on=cfg.evaluation["threshold"]["calibrated_on"] + ", train blocks",
            note="continuous score; thresholds live in outputs/regimes.json",
        )
    return dest, score


def read_probability(cfg: Config, event: Event, feature_set: str) -> np.ndarray:
    path = probability_path(cfg, event, feature_set)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing: run `python -m src.eval.pixel_model`"
        )
    with rasterio.open(path) as src:
        return src.read(1).astype("float32")


def importance(
    cfg: Config, estimator, names: list[str], feature_set: str
) -> list[dict]:
    """Permutation importance on the calibration blocks -- held out from the fit.

    Reported because it is the evidence for the biome argument: if the model
    leans on absolute post-fire reflectance rather than on change, then the
    thing it learned is what a burned *pine* stand looks like, and its threshold
    has no reason to survive a move to oak. That is a claim about the model's
    innards, and it should be measured rather than asserted.
    """
    settings = cfg.pixel_model
    event = cfg.event(cfg.evaluation["threshold"]["calibrated_on"])
    grid = grid_for_event(cfg, event)
    blocks = blocks_for_event(cfg, event, grid)
    matrix, valid, _ = design_matrix(cfg, event, feature_set, grid)
    with rasterio.open(ems.label_raster_path(cfg, event)) as src:
        label = src.read(1) > 0

    held_out = blocks.mask(cfg.evaluation["threshold"]["block_role"])[valid]
    matrix, truth = matrix[held_out], label[valid][held_out]

    rng = np.random.default_rng(int(settings["seed"]))
    cap = 120_000
    if matrix.shape[0] > cap:
        keep = rng.choice(matrix.shape[0], size=cap, replace=False)
        matrix, truth = matrix[keep], truth[keep]

    scores = permutation_importance(
        estimator,
        matrix,
        truth,
        n_repeats=3,
        random_state=int(settings["seed"]),
        scoring="average_precision",
        n_jobs=1,
    )
    ranked = sorted(
        (
            {"feature": n, "drop_in_ap": float(m), "sd": float(s)}
            for n, m, s in zip(names, scores.importances_mean, scores.importances_std)
        ),
        key=lambda entry: -entry["drop_in_ap"],
    )
    return ranked


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def run(cfg: Config, feature_sets: list[str] | None = None) -> dict:
    """Fit and predict each configured feature set, and report the fit."""
    sets = feature_sets or list(cfg.pixel_model["features"])
    payload: dict = {"estimator": cfg.pixel_model["estimator"], "runs": []}
    for feature_set in sets:
        estimator, provenance = fit(cfg, feature_set)
        written = []
        for event in cfg.events.values():
            dest, _ = predict(cfg, estimator, event, feature_set)
            # The basename is the same for every event; the event is the path.
            written.append(f"{event.id}/{dest.name}")
        payload["runs"].append(
            {
                **provenance,
                "rasters": written,
                "importance": importance(
                    cfg, estimator, provenance["features"], feature_set
                ),
            }
        )
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--features",
        action="append",
        default=None,
        help="feature set from config.yaml; repeatable, default all",
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    payload = run(cfg, args.features)

    dest = cfg.path_for("outputs", "pixel_model.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for entry in payload["runs"]:
        print(f"{entry['feature_set']}  ({len(entry['features'])} features)")
        print(
            f"  trained on : {entry['event']}, {entry['role']} blocks, "
            f"{entry['pixels']:,} px, {entry['prevalence']:.1%} burned"
        )
        print(f"  rasters    : {', '.join(entry['rasters'])}")
        print("  importance : (drop in AP on the held-out calibration blocks)")
        for item in entry["importance"][:6]:
            print(f"      {item['feature']:<14} {item['drop_in_ap']:+.4f}")
    print(f"written      : {dest.name}")


if __name__ == "__main__":  # pragma: no cover
    main()
