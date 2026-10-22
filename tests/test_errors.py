"""Guards on the error maps.

A map is one seed of five, so the ways it can mislead are about which seed and
which pixels:

* the drawn seed chosen by looking at a test event;
* a median rule that silently picks between two middle seeds;
* error classes that disagree with the confusion counts the grid reports;
* a map whose counts are not the grid's frozen row for the seed it claims.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import load_config
from src.eval import regimes
from src.eval.metrics import confusion
from src.viz import errors


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# --------------------------------------------------------------------------- #
# the drawn seed
# --------------------------------------------------------------------------- #


def test_median_seed_is_the_middle_value_not_the_middle_seed():
    values = {1: 0.90, 2: 0.10, 3: 0.50, 4: 0.70, 5: 0.30}
    assert errors.median_seed(values) == 3


def test_median_seed_breaks_ties_to_the_lowest_seed():
    assert errors.median_seed({9: 0.5, 4: 0.5, 7: 0.5}) == 4
    assert errors.median_seed({9: 0.1, 4: 0.5, 7: 0.5, 2: 0.9, 3: 0.9}) == 4


def test_median_seed_refuses_an_even_count():
    with pytest.raises(ValueError, match="even count"):
        errors.median_seed({1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4})


def test_seed_selection_never_reads_a_test_event(cfg, monkeypatch):
    """Every domain built while choosing a seed is the training event's
    calibration role. Recorded from the call graph, not assumed from names."""
    seen: list[tuple[str, str]] = []
    rng = np.random.default_rng(0)

    def fake_domain(cfg_, event, role="test", score=None):
        seen.append((event.id, role))
        truth = rng.random(2_000) < 0.2
        return SimpleNamespace(
            score=rng.random(truth.size) + truth,
            truth=truth,
            blocks=SimpleNamespace(blocks_with_role=lambda r: ["x"]),
        )

    monkeypatch.setattr(regimes, "build_domain", fake_domain)
    monkeypatch.setattr(errors, "build_domain", fake_domain)
    method = regimes.Method(
        name="fake",
        score_name="fake",
        inputs="-",
        context="-",
        load=lambda c, e, s: np.zeros(1),
        seeds=(1, 2, 3),
    )
    chosen = errors.representative_seed(cfg, method)

    settings = cfg.evaluation["threshold"]
    assert chosen["seed"] in (1, 2, 3)
    assert seen and set(seen) == {(settings["calibrated_on"], settings["block_role"])}
    assert cfg.event(settings["calibrated_on"]).is_train


def test_only_the_frozen_regime_can_be_drawn(cfg):
    bad = SimpleNamespace(
        raw=cfg.raw | {"error_maps": cfg.raw["error_maps"] | {"regime": "oracle"}}
    )
    with pytest.raises(ValueError, match="frozen"):
        errors.run(bad)


# --------------------------------------------------------------------------- #
# error rasters
# --------------------------------------------------------------------------- #


def test_error_classes_reproduce_the_confusion_counts():
    rng = np.random.default_rng(1)
    label = rng.random((60, 70)) < 0.3
    score = rng.random(label.shape) + 0.4 * label
    mask = rng.random(label.shape) < 0.8
    classes = errors.error_classes(score, label, mask, 0.7)

    counts = confusion(score[mask], label[mask], 0.7)
    assert int((classes == errors.TP).sum()) == counts.tp
    assert int((classes == errors.FP).sum()) == counts.fp
    assert int((classes == errors.FN).sum()) == counts.fn
    assert int((classes == errors.TN).sum()) == counts.tn
    assert np.all(classes[~mask] == errors.OUTSIDE)


def test_densest_window_finds_the_planted_cluster():
    count = np.zeros((100, 120), dtype="int32")
    count[5, 5] = 3  # a lone dense pixel loses to a larger cluster
    count[60:70, 80:90] = 1
    assert errors.densest_window(count, 10) == (60, 80)
    with pytest.raises(ValueError):
        errors.densest_window(count, 101)


# --------------------------------------------------------------------------- #
# the shipped maps
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def shipped(cfg):
    paths = [cfg.path_for("outputs", n) for n in ("errors.json", "regimes.json")]
    if not all(p.exists() for p in paths):
        pytest.skip("run `python -m src.eval.regimes` and `python -m src.viz.errors`")
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def test_maps_are_the_grids_frozen_rows(shipped):
    """Same pixels, same threshold: every map's counts are the frozen row of
    regimes.json for the seed it says it draws."""
    maps, grid = shipped
    for event_id, entry in maps["events"].items():
        for method, info in entry["methods"].items():
            row = next(
                r
                for r in grid["rows"]
                if r["event"] == event_id
                and r["method"] == method
                and r["regime"] == "frozen"
                and r["seed"] == info["seed"]
            )
            assert info["iou"] == row["iou"], (event_id, method)
            assert info["threshold"] == pytest.approx(row["threshold"], abs=5e-6)


def test_drawn_seed_is_the_configured_rule_applied(shipped):
    maps, _ = shipped
    for name, chosen in maps["seeds"].items():
        if chosen["seed"] is None:
            continue
        values = {int(s): v for s, v in chosen["calibration_iou"].items()}
        assert chosen["seed"] == errors.median_seed(values), name
