"""Evaluation-machinery tests.

Same spirit as ``test_config.py``: these are not there to prove the arithmetic
works, they are there so that the protocol cannot be broken quietly. Phase 2
builds the machinery both methods will be measured by, so the failure that
matters is not a crash -- it is a comparison that looks fair and is not. Each
guard below corresponds to one way that could happen:

* the two methods getting different thresholds, or different pixels;
* an interval published over a domain that cannot support one;
* a metric disagreeing with the confusion matrix printed beside it;
* accuracy sneaking into a table where it would read 98% and mean nothing;
* train, calibration and test blocks touching.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.metrics import f1_score, jaccard_score

from src.config import load_config
from src.data.grid import grid_for_event
from src.eval import threshold as threshold_mod
from src.eval.blocks import ROLE_CODES, blocks_for_event, partition
from src.eval.bootstrap import block_bootstrap
from src.eval.metrics import Confusion, confusion, confusion_by_group, evaluate


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def toy():
    """A score, a truth mask and a block index, deterministic."""
    rng = np.random.default_rng(7)
    truth = rng.random(20_000) < 0.08
    score = rng.normal(0.0, 1.0, truth.size) + 2.0 * truth
    group = rng.integers(0, 6, truth.size)
    return score, truth, group


# --------------------------------------------------------------------------- #
# threshold symmetry -- section 6.2
# --------------------------------------------------------------------------- #


def test_calibration_is_score_agnostic(toy):
    """The same function must serve a dNBR difference and a sigmoid probability,
    or the two methods are not calibrated by the same procedure."""
    score, truth, _ = toy
    index_like = threshold_mod.calibrate(
        score, truth, score_name="index", calibrated_on="toy"
    )
    probability_like = threshold_mod.calibrate(
        1 / (1 + np.exp(-score)), truth, score_name="probability", calibrated_on="toy"
    )
    # A monotone transform of the score moves the threshold but cannot move the
    # operating point it selects.
    assert index_like.objective_value == pytest.approx(
        probability_like.objective_value, abs=1e-12
    )
    assert probability_like.value == pytest.approx(
        1 / (1 + np.exp(-index_like.value)), abs=1e-9
    )


def test_calibration_finds_the_best_operating_point(toy):
    score, truth, _ = toy
    best = threshold_mod.calibrate(score, truth, score_name="s", calibrated_on="toy")
    for candidate in np.linspace(score.min(), score.max(), 200):
        counts = confusion(score, truth, candidate)
        assert counts.f1 <= best.objective_value + 1e-12


def test_calibration_refuses_a_set_with_no_positive():
    score = np.linspace(0, 1, 100)
    with pytest.raises(ValueError, match="no positive pixel"):
        threshold_mod.calibrate(
            score, np.zeros(100, bool), score_name="s", calibrated_on="empty blocks"
        )


def test_shipped_threshold_was_not_calibrated_on_a_test_event(cfg):
    """The whole point of freezing it: the target's labels are never consulted."""
    path = threshold_mod.thresholds_path(cfg)
    if not path.exists():
        pytest.skip("run `python -m src.eval.baseline` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    calibrated_on = payload["calibrated_on_event"]
    assert cfg.event(calibrated_on).is_train
    assert payload["recalibrate_on_target"] is False
    for name, entry in payload["thresholds"].items():
        for event in cfg.events.values():
            if not event.is_train:
                assert event.id not in entry["calibrated_on"], (
                    f"threshold {name!r} mentions the test event {event.id} among the "
                    "pixels it was fitted on"
                )


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def test_metrics_agree_with_sklearn(toy):
    score, truth, _ = toy
    counts = confusion(score, truth, 0.5)
    predicted = score >= 0.5
    assert counts.iou == pytest.approx(jaccard_score(truth, predicted))
    assert counts.f1 == pytest.approx(f1_score(truth, predicted))
    assert counts.n == truth.size


def test_confusion_splits_and_sums_back(toy):
    """The bootstrap resamples per-block counts, so the split has to be exact."""
    score, truth, group = toy
    parts = confusion_by_group(score, truth, 0.5, group, group.max() + 1)
    assert sum(parts, Confusion(0, 0, 0, 0)) == confusion(score, truth, 0.5)


def test_forbidden_metrics_cannot_be_emitted(cfg, toy):
    score, truth, _ = toy
    result = evaluate(
        score, truth, 0.5,
        method="toy", event="toy", experiment="toy", threshold_source="toy",
        pixel_area_ha=0.04, footprint_pixels=truth.size, with_curve=False,
    )
    row = result.as_row(tuple(cfg.evaluation["forbidden_metrics"]))
    assert "accuracy" not in row
    # and the guard is live, not decorative
    with pytest.raises(ValueError, match="forbids reporting"):
        result.as_row(("iou",))


def test_area_error_is_measured_against_the_rasterised_label(toy):
    score, truth, _ = toy
    result = evaluate(
        score, truth, 0.5,
        method="toy", event="toy", experiment="toy", threshold_source="toy",
        pixel_area_ha=0.04, footprint_pixels=truth.size, with_curve=False,
    )
    assert result.label_area_ha == pytest.approx(truth.sum() * 0.04)
    assert result.predicted_area_ha == pytest.approx(
        (result.counts.tp + result.counts.fp) * 0.04
    )


# --------------------------------------------------------------------------- #
# bootstrap -- section 6.5
# --------------------------------------------------------------------------- #


def _blocks(n_with_burn: int, n_total: int) -> list[Confusion]:
    return [
        Confusion(tp=100, fp=20, fn=30, tn=5_000)
        if i < n_with_burn
        else Confusion(tp=0, fp=15, fn=0, tn=5_000)
        for i in range(n_total)
    ]


def test_refuses_an_interval_when_too_few_blocks_carry_burn():
    """A domain whose scar sits in one block does not get to advertise four
    independent observations because three background blocks came along."""
    intervals = block_bootstrap(_blocks(1, 6), iterations=50, min_blocks=4)
    assert not intervals["iou"].estimable
    assert intervals["iou"].n_blocks == 6
    assert intervals["iou"].n_burned_blocks == 1
    assert "burned pixels" in intervals["iou"].reason


def test_publishes_an_interval_when_enough_blocks_carry_burn():
    intervals = block_bootstrap(_blocks(6, 6), iterations=200, min_blocks=4)
    interval = intervals["iou"]
    assert interval.estimable
    assert interval.low <= interval.point <= interval.high


def test_point_estimate_is_the_pooled_metric_not_a_mean_of_ratios():
    blocks = [
        Confusion(tp=1, fp=0, fn=0, tn=10),        # IoU 1.0 on one pixel
        Confusion(tp=500, fp=500, fn=500, tn=10),  # IoU 0.33 on a thousand
    ]
    intervals = block_bootstrap(blocks, iterations=10, min_blocks=1)
    pooled = sum(blocks, Confusion(0, 0, 0, 0))
    assert intervals["iou"].point == pytest.approx(pooled.iou)
    assert intervals["iou"].point < 0.4  # a mean of ratios would say 0.67


def test_bootstrap_is_reproducible():
    a = block_bootstrap(_blocks(6, 6), iterations=100, min_blocks=4, seed=3)
    b = block_bootstrap(_blocks(6, 6), iterations=100, min_blocks=4, seed=3)
    assert a["f1"].low == b["f1"].low and a["f1"].high == b["f1"].high


# --------------------------------------------------------------------------- #
# spatial blocks -- section 6.1 and 6.5
# --------------------------------------------------------------------------- #


def test_block_size_stays_near_the_configured_target(cfg):
    target = float(cfg.evaluation["spatial_blocks"]["superblock_km"])
    for event in cfg.events.values():
        grid = grid_for_event(cfg, event)
        index, names = partition(cfg, grid)
        assert set(np.unique(index)) == set(range(len(names)))
        rows = len({n.split("c")[0] for n in names})
        cols = len({n.split("c")[1] for n in names})
        for span_km, n in (
            (grid.height * grid.resolution / 1000, rows),
            (grid.width * grid.resolution / 1000, cols),
        ):
            # Never a sliver: no axis is cut into blocks under half the target.
            assert span_km / n >= target / 2, f"{event.id}: {span_km / n:.1f} km blocks"


def test_every_test_event_is_test_in_full(cfg):
    """Section 6.1: the test footprint is geometric and nothing inside it is
    filtered out, negatives included."""
    for event in cfg.events.values():
        if event.is_train:
            continue
        blocks = blocks_for_event(cfg, event)
        assert blocks.blocks_with_role("test") == list(blocks.names)
        assert not blocks.mask("train").any()


def test_roles_are_disjoint_and_buffered(cfg):
    event = cfg.event(cfg.evaluation["threshold"]["calibrated_on"])
    blocks = blocks_for_event(cfg, event)
    train = blocks.mask("train")
    held_out = blocks.mask("calibration") | blocks.mask("test")

    assert not (train & held_out).any()
    assert train.any() and blocks.mask("calibration").any() and blocks.mask("test").any()

    # No train pixel is adjacent to a held-out pixel: the buffer is real, not a
    # parameter that happens to be in the YAML.
    from scipy import ndimage

    buffer_px = (
        float(cfg.evaluation["spatial_blocks"]["role_buffer_m"]) / blocks.grid.resolution
    )
    distance = ndimage.distance_transform_edt(~held_out)
    assert distance[train].min() > buffer_px


def test_calibration_blocks_hold_burned_ground(cfg):
    """A threshold fitted on pure background is not a threshold. This is the
    guard that caught a 2560 m buffer eating 98% of the calibration burn."""
    path = cfg.path_for("outputs", "spatial_blocks.json")
    if not path.exists():
        pytest.skip("run `python -m src.eval.blocks` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    train_event = cfg.evaluation["threshold"]["calibrated_on"]
    entry = next(e for e in payload["events"] if e["event"] == train_event)
    for role in ("train", "calibration", "test"):
        assert entry["roles"][role]["burned_pixels"] > 1000, (
            f"{role} blocks hold {entry['roles'][role]['burned_ha']} ha of burn"
        )


def test_unknown_block_name_in_config_is_refused(cfg):
    import copy

    from src.config import load_config as _load

    raw = copy.deepcopy(cfg.raw)
    raw["evaluation"]["spatial_blocks"]["roles"] = {
        cfg.evaluation["threshold"]["calibrated_on"]: {"test": ["r9c9"]}
    }
    import tempfile
    from pathlib import Path

    import yaml

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        mutated = _load(path)
        with pytest.raises(KeyError, match="r9c9"):
            blocks_for_event(mutated, mutated.event(raw["evaluation"]["threshold"]["calibrated_on"]))


def test_buffer_erodes_seams_and_not_the_outer_edge():
    from src.eval.blocks import _buffered

    role = np.full((40, 40), ROLE_CODES["train"], dtype="uint8")
    role[:, 20:] = ROLE_CODES["test"]
    out = _buffered(role, buffer_px=3)
    assert (out[:, 0] == ROLE_CODES["train"]).all()
    assert (out[:, -1] == ROLE_CODES["test"]).all()
    assert (out[:, 18:22] == ROLE_CODES["buffer"]).all()


# --------------------------------------------------------------------------- #
# the shipped results
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def results(cfg):
    path = cfg.path_for("outputs", "baseline_results.json")
    if not path.exists():
        pytest.skip("run `python -m src.eval.baseline` first")
    return json.loads(path.read_text(encoding="utf-8"))


def test_one_frozen_threshold_for_every_test_event(results):
    honest = [r for r in results["rows"] if r["method"] == "dNBR honest"]
    assert len({r["threshold"] for r in honest}) == 1, (
        "the honest baseline used more than one threshold across events, which is a "
        "recalibration on the target by another name"
    )
    # The row rounds for display; the exact frozen value lives in thresholds.json.
    assert honest[0]["threshold"] == pytest.approx(
        results["threshold"]["value"], abs=1e-5
    )


def test_oracle_is_never_worse_than_honest(results):
    """It refits on the event itself, so by construction it cannot lose. If it
    does, the two are not being scored on the same pixels."""
    by_key = {(r["event"], r["method"]): r for r in results["rows"]}
    for event in {e for e, _ in by_key}:
        honest = by_key[(event, "dNBR honest")]
        oracle = by_key[(event, "dNBR oracle")]
        assert oracle["f1"] >= honest["f1"] - 1e-9
        assert oracle["average_precision"] == pytest.approx(
            honest["average_precision"]
        ), "average precision is threshold-free and must not move with the threshold"


def test_confusion_matches_the_evaluated_domain(results):
    for row in results["rows"]:
        counts = row["confusion"]
        assert sum(counts.values()) == row["domain_pixels"]
        assert row["domain_pixels"] <= row["footprint_pixels"]


def test_no_interval_is_published_without_enough_burned_blocks(results):
    minimum = results["min_blocks_for_interval"]
    for row in results["rows"]:
        for metric, interval in (row["interval"] or {}).items():
            if interval["estimable"]:
                assert interval["n_burned_blocks"] >= minimum
            else:
                assert interval["reason"]


def test_evaluated_domain_keeps_its_negatives(results):
    """Section 6.1: filtering unburned pixels would inflate precision by an
    arbitrary factor. Prevalence has to stay low."""
    for row in results["rows"]:
        assert row["prevalence"] < 0.2, (
            f"{row['event']}: prevalence {row['prevalence']:.1%} -- the evaluation "
            "footprint looks filtered"
        )
