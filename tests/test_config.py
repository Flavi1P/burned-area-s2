"""Config and grid tests.

These run offline. Most of them are not testing that the code works -- they are
testing that the *protocol* cannot be broken quietly. The failure mode this
project has to defend against is not a crash; it is a plausible number produced
by a configuration that leaked. So each guard here corresponds to one leak the
design note refuses to accept, and each one is checked by feeding the loader a
configuration that commits the mistake.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from src.config import ConfigError, load_config
from src.data.grid import grid_for_event, grid_from_bounds, snap_bounds, utm_crs_for_bbox


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture()
def raw(cfg, tmp_path):
    """A mutable copy of the real config, plus a writer that reloads it."""

    def write(mutated):
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(mutated), encoding="utf-8")
        return load_config(path)

    return copy.deepcopy(cfg.raw), write


# --------------------------------------------------------------------------- #
# the shipped configuration
# --------------------------------------------------------------------------- #


def test_config_loads(cfg):
    assert cfg.events and cfg.experiments


def test_every_event_has_a_final_ems_product(cfg):
    """T0.1: status and production method recorded for all three events."""
    for event in cfg.events.values():
        assert event.label.is_final, f"{event.id}: label is not a final EMS product"
        assert event.label.production.is_documented, (
            f"{event.id}: EMS production method not recorded -- section 5 requires it, "
            "because a semi-automatic label makes the comparison partly circular"
        )


def test_exactly_one_training_event(cfg):
    """The whole transfer argument rests on training on one biome only."""
    assert [e.id for e in cfg.events.values() if e.is_train] == ["saumos"]


def test_test_events_state_what_they_control_for(cfg):
    for event in cfg.events.values():
        if not event.is_train:
            assert event.test_purpose, f"{event.id}: no stated role in the design"


def test_size_control_shares_the_biome_of_the_training_event(cfg):
    """Biscarrosse only dissolves the confound if its fuel matches Saumos and
    its size class matches Fontainebleau. If someone edits the events, this is
    the assumption that silently breaks."""
    train = next(e for e in cfg.events.values() if e.is_train)
    control = cfg.event("biscarrosse")
    other = cfg.event("fontainebleau")

    assert control.fuel == train.fuel
    assert control.label.reported_burnt_area_ha < train.label.reported_burnt_area_ha / 5
    ratio = control.label.reported_burnt_area_ha / other.label.reported_burnt_area_ha
    assert 0.2 < ratio < 5, f"size classes no longer comparable (ratio {ratio:.1f})"


def test_no_event_name_outside_the_yaml():
    """Section 9: the experimental plan must be readable in config.yaml, not
    reconstructed by reading train.py."""
    from pathlib import Path

    from src.config import REPO_ROOT

    cfg = load_config()
    forbidden = {e.label.activation for e in cfg.events.values()}
    offenders = []
    for path in (REPO_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            # An EMS code may appear in a comment as an illustration, never in
            # code. Crude but effective: flag it if it is inside a string.
            for line in text.splitlines():
                stripped = line.strip()
                if token in line and not stripped.startswith("#"):
                    offenders.append(f"{path.name}: {stripped}")
    assert not offenders, "EMS codes hard-coded outside config.yaml:\n" + "\n".join(offenders)


# --------------------------------------------------------------------------- #
# leak guards
# --------------------------------------------------------------------------- #


def test_rejects_shared_event_under_by_event_split(raw):
    data, write = raw
    exp = next(e for e in data["experiments"] if e["id"] == "E2")
    exp["train"] = ["saumos"]
    exp["test"] = ["saumos", "fontainebleau"]
    with pytest.raises(ConfigError, match="train and test"):
        write(data)


def test_rejects_threshold_calibrated_on_a_test_event(raw):
    data, write = raw
    data["evaluation"]["threshold"]["calibrated_on"] = "fontainebleau"
    with pytest.raises(ConfigError, match="test event"):
        write(data)


def test_rejects_recalibration_on_the_target(raw):
    """Section 6.2: recalibrating on the target erases the very transfer cost
    E2 exists to measure, on both sides of the comparison."""
    data, write = raw
    data["evaluation"]["threshold"]["recalibrate_on_target"] = True
    with pytest.raises(ConfigError, match="recalibrate_on_target"):
        write(data)


def test_rejects_overlapping_test_tiles(raw):
    data, write = raw
    data["evaluation"]["tiling"]["test"]["overlap_px"] = 64
    with pytest.raises(ConfigError, match="overlap"):
        write(data)


def test_rejects_filtering_negative_test_tiles(raw):
    data, write = raw
    data["evaluation"]["tiling"]["test"]["filter_negatives"] = True
    with pytest.raises(ConfigError, match="filter_negatives"):
        write(data)


def test_rejects_global_accuracy(raw):
    data, write = raw
    data["evaluation"]["metrics"].append("accuracy")
    with pytest.raises(ConfigError, match="forbidden"):
        write(data)


def test_rejects_bootstrap_blocks_smaller_than_a_tile(raw):
    """Section 6.5: the tile cannot be the bootstrap unit -- neighbouring tiles
    inside one scar are massively correlated."""
    data, write = raw
    data["evaluation"]["spatial_blocks"]["superblock_km"] = 3.0
    with pytest.raises(ConfigError, match="superblock_km"):
        write(data)


def test_rejects_single_partition_few_shot(raw):
    """Section 7: fine-tuning and evaluating inside one small event needs
    several alternating partitions, or the curve is one lucky split."""
    data, write = raw
    exp = next(e for e in data["experiments"] if e["id"] == "E3")
    exp["n_partitions"] = 1
    with pytest.raises(ConfigError, match="partitions"):
        write(data)


def test_rejects_unknown_event_reference(raw):
    data, write = raw
    data["experiments"][0]["test"] = ["landiras"]
    with pytest.raises(ConfigError, match="unknown event"):
        write(data)


# --------------------------------------------------------------------------- #
# grid
# --------------------------------------------------------------------------- #


def test_snap_bounds_only_ever_grows():
    bounds = (10.0, 10.0, 90.0, 90.0)
    snapped = snap_bounds(bounds, 20.0)
    assert snapped == (0.0, 0.0, 100.0, 100.0)
    for got, want in zip(snapped[:2], bounds[:2]):
        assert got <= want
    for got, want in zip(snapped[2:], bounds[2:]):
        assert got >= want


def test_grid_origin_is_a_multiple_of_the_resolution():
    """Alignment with the Sentinel-2 20 m grid is what makes B8A/B11/B12 land
    as an identity instead of being needlessly resampled."""
    grid = grid_from_bounds((417_233.0, 4_951_117.0, 438_112.0, 4_963_004.0),
                            utm_crs_for_bbox((2.47, 48.35, 2.70, 48.43)), 20.0)
    assert grid.transform.c % 20 == 0
    assert grid.transform.f % 20 == 0


def test_event_grids_are_utm_and_cover_the_aoi(cfg):
    for event in cfg.events.values():
        grid = grid_for_event(cfg, event)
        assert grid.crs.to_epsg() // 100 == 326  # UTM north
        assert grid.resolution == cfg.resolution_m
        assert grid.width > 0 and grid.height > 0


def test_tile_fits_inside_every_event_grid(cfg):
    tile = int(cfg.project["tile_size_px"])
    for event in cfg.events.values():
        grid = grid_for_event(cfg, event)
        assert min(grid.shape) >= tile, f"{event.id}: footprint smaller than one tile"


def test_pixel_area(cfg):
    grid = grid_for_event(cfg, next(iter(cfg.events.values())))
    assert grid.pixel_area_ha == pytest.approx(0.04)  # 20 m x 20 m
