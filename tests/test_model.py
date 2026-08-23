"""Phase 3 tests: tiling, normalisation, loss masking, and the U-Net rows.

The tests worth having here are not "does the tensor have the right shape" --
though one of those is the stated completion criterion for T3.1 and is included.
They are the ones that fail when the protocol quietly breaks: a training tile
touching calibration ground, a normalisation statistic fitted on a test event,
a clouded pixel contributing a gradient, or the two methods being scored on
domains that merely resemble each other.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.config import load_config
from src.data.grid import grid_for_event
from src.data.stack import NEGATIVE_REFLECTANCE_LIMIT
# Imported as a module, not by name: `test_tiles` is protocol vocabulary
# (the test tiling regime), and a bare import would have pytest collect it
# as a test case.
from src.data import tiles as tiling
from src.eval.blocks import blocks_for_event


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def train_event(cfg):
    return next(e for e in cfg.events.values() if e.is_train)


# --------------------------------------------------------------------------- #
# tiling
# --------------------------------------------------------------------------- #


def test_training_tiles_lie_entirely_inside_their_role(cfg, train_event):
    """The leak this project exists to avoid. A tile straddling a role seam puts
    calibration or test ground into a training batch, and a U-Net's receptive
    field spreads it over the whole tile."""
    blocks = blocks_for_event(cfg, train_event)
    for role in ("train", "calibration"):
        mask = blocks.mask(role)
        for tile in tiling.training_tiles(cfg, train_event, role=role):
            assert mask[tile.slices].all(), (
                f"{role} tile at ({tile.row}, {tile.col}) contains pixels of "
                "another role"
            )


def test_training_and_calibration_tiles_never_share_a_pixel(cfg, train_event):
    grid = grid_for_event(cfg, train_event)
    seen = np.zeros(grid.shape, dtype=bool)
    for tile in tiling.training_tiles(cfg, train_event, role="train"):
        seen[tile.slices] = True
    for tile in tiling.training_tiles(cfg, train_event, role="calibration"):
        assert not seen[tile.slices].any()


def test_test_tiles_partition_the_footprint_exactly(cfg):
    """Section 6.1: no overlap between evaluation units, and no gap either --
    the test footprint is geometric and every pixel in it is scored."""
    for event in cfg.events.values():
        grid = grid_for_event(cfg, event)
        covered = np.zeros(grid.shape, dtype="uint8")
        for tile in tiling.test_tiles(cfg, event):
            covered[tile.owned_slices] += 1
        assert covered.max() == 1, f"{event.id}: some pixel is predicted twice"
        assert covered.min() == 1, f"{event.id}: some pixel is never predicted"


def test_test_tiles_are_not_filtered(cfg):
    """Dropping unburned tiles would inflate precision by an arbitrary factor."""
    for event in cfg.events.values():
        tiles = tiling.test_tiles(cfg, event)
        assert any(not t.is_positive for t in tiles), (
            f"{event.id}: every test tile contains burn, which means the footprint "
            "was filtered"
        )


def test_tile_inventory_is_reported_not_engineered(cfg):
    """The design note asked for 2:1 negatives; the data could not supply it. The
    inventory must record what the training set actually is."""
    path = cfg.path_for("outputs", "tiles.json")
    if not path.exists():
        pytest.skip("run `python -m src.data.tiles` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    train = next(
        s for s in payload["sets"] if s["regime"] == "train" and s["role"] == "train"
    )
    assert train["positive"] > 0
    assert train["tiles"] == train["positive"] + train["negative"]


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #


def test_normalisation_is_fitted_on_training_pixels_only(cfg, train_event):
    torch = pytest.importorskip("torch")  # noqa: F841
    from src.model.dataset import fit_normalisation, load_event

    norm = fit_normalisation(cfg)
    arrays = load_event(cfg, train_event)
    expected = int((blocks_for_event(cfg, train_event).mask("train") & arrays.valid).sum())

    assert norm.n_pixels == expected
    assert train_event.id in norm.fitted_on
    for other in cfg.events.values():
        if other.id != train_event.id:
            assert other.id not in norm.fitted_on
    assert all(s > 0 for s in norm.std)


def test_shipped_normalisation_matches_the_training_event(cfg, train_event):
    path = cfg.path_for("outputs", "normalisation.json")
    if not path.exists():
        pytest.skip("run `python -m src.model.dataset` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert train_event.id in payload["fitted_on"]
    assert len(payload["mean"]) == int(cfg.model["in_channels"])


def test_reflectance_guard_threshold_is_meaningful():
    """The guard that catches a double-applied band offset. Loose enough for
    water and shadow, tight enough to fail on 70% negative reflectance."""
    assert 0.0 < NEGATIVE_REFLECTANCE_LIMIT < 0.5


# --------------------------------------------------------------------------- #
# augmentation
# --------------------------------------------------------------------------- #


def test_d4_moves_image_and_label_together():
    pytest.importorskip("torch")
    from src.model.dataset import D4, apply_d4

    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 16, 16)).astype("float32")
    y = (rng.random((1, 16, 16)) > 0.7).astype("float32")

    assert len(D4) == 8
    for element in D4:
        xa, ya = apply_d4([x, y], element)
        assert xa.shape == x.shape and ya.shape == y.shape
        assert ya.sum() == y.sum(), "augmentation changed the amount of burned area"
        # The transform is a permutation of pixels: the same one on both arrays.
        assert np.array_equal(apply_d4([y], element)[0], ya)


def test_d4_elements_are_distinct():
    pytest.importorskip("torch")
    from src.model.dataset import D4, apply_d4

    # An asymmetric pattern: every group element must move it somewhere new.
    a = np.arange(16, dtype="float32").reshape(1, 4, 4)
    seen = {apply_d4([a], element)[0].tobytes() for element in D4}
    assert len(seen) == len(D4)


# --------------------------------------------------------------------------- #
# dataset and model
# --------------------------------------------------------------------------- #


def test_a_batch_has_the_shape_the_network_expects(cfg, train_event):
    """T3.1's stated completion criterion: (B, 8, 256, 256)."""
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from src.model.dataset import fit_normalisation, training_dataset

    dataset = training_dataset(cfg, train_event, fit_normalisation(cfg))
    batch_size = int(cfg.model["batch_size"])
    x, y, w = next(iter(DataLoader(dataset, batch_size=batch_size)))

    size = int(cfg.project["tile_size_px"])
    channels = int(cfg.model["in_channels"])
    assert x.shape == (batch_size, channels, size, size)
    assert y.shape == (batch_size, 1, size, size)
    assert w.shape == (batch_size, 1, size, size)
    assert torch.isfinite(x).all(), "clouded pixels must arrive as zeros, never NaN"
    assert set(np.unique(y.numpy())) <= {0.0, 1.0}


def test_the_network_maps_eight_channels_to_one(cfg):
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from src.model import unet

    model = unet.build(cfg)
    model.eval()
    size = int(cfg.project["tile_size_px"])
    with torch.no_grad():
        logits = model(torch.zeros(2, int(cfg.model["in_channels"]), size, size))
    assert logits.shape == (2, 1, size, size)


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #


def test_masked_pixels_contribute_no_loss_and_no_gradient(cfg):
    """Cloud and shadow pixels are filled with the channel mean so the network
    can run at all. If they also entered the loss, the network would be taught
    that mean reflectance means unburned -- a fact about the cloud mask."""
    torch = pytest.importorskip("torch")
    from src.model.loss import BceDiceLoss

    criterion = BceDiceLoss.from_config(cfg)
    rng = np.random.default_rng(1)
    logits = torch.tensor(rng.normal(size=(2, 1, 8, 8)), dtype=torch.float32)
    target = torch.tensor(
        (rng.random((2, 1, 8, 8)) > 0.6).astype("float32"), dtype=torch.float32
    )
    weight = torch.ones_like(target)
    weight[:, :, :, 4:] = 0.0

    before = float(criterion(logits, target, weight)["loss"])
    perturbed = logits.clone()
    perturbed[:, :, :, 4:] += 25.0  # nonsense predictions, all of it masked out
    after = float(criterion(perturbed, target, weight)["loss"])

    assert after == pytest.approx(before, abs=1e-6)


def test_the_loss_carries_both_terms(cfg):
    torch = pytest.importorskip("torch")
    from src.model.loss import BceDiceLoss

    criterion = BceDiceLoss.from_config(cfg)
    logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
    target = torch.ones(1, 1, 4, 4)
    parts = criterion(logits, target, torch.ones_like(target))

    assert {"loss", "bce", "dice"} <= set(parts)
    assert float(parts["bce"]) > 0 and float(parts["dice"]) > 0
    parts["loss"].backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0


def test_a_perfect_prediction_beats_an_inverted_one(cfg):
    torch = pytest.importorskip("torch")
    from src.model.loss import BceDiceLoss

    criterion = BceDiceLoss.from_config(cfg)
    target = torch.zeros(1, 1, 8, 8)
    target[:, :, :4] = 1.0
    weight = torch.ones_like(target)
    good = torch.where(target > 0, 6.0, -6.0)

    assert float(criterion(good, target, weight)["loss"]) < float(
        criterion(-good, target, weight)["loss"]
    )


# --------------------------------------------------------------------------- #
# the shipped results
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def results(cfg):
    path = cfg.path_for("outputs", "results.json")
    if not path.exists():
        pytest.skip("run `python -m src.model.evaluate` first")
    return json.loads(path.read_text(encoding="utf-8"))


def test_both_methods_are_scored_on_identical_domains(results):
    """The claim the whole comparison rests on. Not "similar" domains: the same
    pixels, the same count, the same label area, event by event."""
    by_event: dict[str, list[dict]] = {}
    for row in results["rows"]:
        by_event.setdefault(row["event"], []).append(row)

    for event, rows in by_event.items():
        assert len({r["domain_pixels"] for r in rows}) == 1, (
            f"{event}: the methods were scored on different numbers of pixels"
        )
        assert len({round(r["label_area_ha"], 6) for r in rows}) == 1, (
            f"{event}: the methods were scored against different label areas"
        )
        assert {"U-Net", "dNBR honest", "dNBR oracle"} <= {r["method"] for r in rows}


def test_the_unet_threshold_is_frozen_and_calibrated_off_the_test_events(results, cfg):
    unet = results["thresholds"]["unet"]
    assert unet["frozen"] is True
    assert unet["objective"] == cfg.evaluation["threshold"]["objective"]
    assert cfg.evaluation["threshold"]["calibrated_on"] in unet["calibrated_on"]
    for event in cfg.events.values():
        if event.role == "test":
            assert event.id not in unet["calibrated_on"]

    rows = [r for r in results["rows"] if r["method"] == "U-Net"]
    assert len({r["threshold"] for r in rows}) == 1, (
        "the network used more than one threshold across events, which is a "
        "recalibration on the target by another name"
    )
    assert rows[0]["threshold"] == pytest.approx(unet["value"], abs=1e-5)


def test_both_thresholds_were_calibrated_on_the_same_pixels(results):
    """Section 6.2, the sentence the README makes: one procedure, one pixel set,
    both sides."""
    dnbr, unet = results["thresholds"]["dnbr"], results["thresholds"]["unet"]
    assert dnbr["n_pixels"] == unet["n_pixels"]
    assert dnbr["n_positive"] == unet["n_positive"]
    assert dnbr["objective"] == unet["objective"]


def test_the_network_gets_no_oracle_row(results):
    """The oracle bounds what the index can do. Giving the network one too would
    dissolve the decomposition it exists to provide."""
    assert not any(
        r["method"].startswith("U-Net") and "oracle" in r["method"].lower()
        for r in results["rows"]
    )
