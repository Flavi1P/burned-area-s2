"""Guards on the methods x regimes grid.

Adding a third method and two more threshold regimes multiplies the ways a
comparison can look fair without being fair, so it multiplies the guards. Each
test below names one such way:

* an unsupervised or oracle threshold leaking into the file whose whole purpose
  is to record the one value applied unchanged everywhere;
* a label-free threshold that is not actually label-free;
* the new method being scored on different pixels from the old ones, which
  would make every column comparison in the grid meaningless;
* an oracle row presented as something somebody could deploy;
* the pixel model trained on ground it is later evaluated on;
* the losing unsupervised estimator quietly disappearing from the table, which
  would turn "we tried two" into a selection nobody can see.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import load_config
from src.eval import pixel_model, regimes
from src.eval import threshold as threshold_mod
from src.eval.blocks import blocks_for_event
from src.data.grid import grid_for_event


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def grid(cfg):
    path = cfg.path_for("outputs", "regimes.json")
    if not path.exists():
        pytest.skip("run `python -m src.eval.regimes` first")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# the regimes themselves
# --------------------------------------------------------------------------- #


def test_unsupervised_threshold_consults_no_labels():
    """The claim that makes the regime deployable, tested on its signature and
    on its behaviour: the same scores must give the same threshold whatever the
    truth is, because the truth is not an argument it has."""
    rng = np.random.default_rng(3)
    truth = rng.random(50_000) < 0.1
    score = rng.normal(0.0, 1.0, truth.size) + 3.0 * truth
    settings = load_config().evaluation["threshold"]["unsupervised"]

    for estimator in settings["estimators"]:
        picked = threshold_mod.unsupervised(
            score,
            score_name="toy",
            selected_on="toy",
            estimator=estimator,
            settings=settings,
            seed=1,
        )
        assert picked.uses_labels is False
        assert picked.frozen is False
        assert picked.regime == threshold_mod.UNSUPERVISED
        # Its n_positive is a count of predicted pixels, never of burned ground.
        assert picked.n_positive == int((score[np.isfinite(score)] >= picked.value).sum())


def test_unsupervised_threshold_is_reproducible():
    rng = np.random.default_rng(11)
    score = rng.normal(0.0, 1.0, 40_000) + 3.0 * (rng.random(40_000) < 0.1)
    settings = load_config().evaluation["threshold"]["unsupervised"]
    for estimator in settings["estimators"]:
        kwargs = dict(
            score_name="toy", selected_on="toy", estimator=estimator,
            settings=settings, seed=5,
        )
        assert threshold_mod.unsupervised(score, **kwargs).value == pytest.approx(
            threshold_mod.unsupervised(score, **kwargs).value
        )


def test_an_unknown_estimator_is_refused():
    settings = load_config().evaluation["threshold"]["unsupervised"]
    with pytest.raises(ValueError, match="unknown unsupervised estimator"):
        threshold_mod.unsupervised(
            np.linspace(0, 1, 100),
            score_name="toy",
            selected_on="toy",
            estimator="triangle",
            settings=settings,
            seed=1,
        )


def test_only_a_frozen_threshold_can_reach_the_frozen_file(cfg, tmp_path):
    """`thresholds.json` records the one value applied unchanged to every test
    event. A per-event threshold in it would make that sentence false."""
    per_event = threshold_mod.Threshold(
        value=0.3,
        objective="otsu",
        score_name="dnbr",
        calibrated_on="fontainebleau, score histogram only",
        n_pixels=10,
        n_positive=3,
        objective_value=0.5,
        frozen=False,
        regime=threshold_mod.UNSUPERVISED,
        uses_labels=False,
    )
    with pytest.raises(ValueError, match="cannot be frozen"):
        threshold_mod.update(cfg, "dnbr", per_event)


def test_shipped_frozen_file_holds_only_frozen_thresholds(cfg):
    path = threshold_mod.thresholds_path(cfg)
    if not path.exists():
        pytest.skip("run `python -m src.eval.baseline` first")
    for name, entry in json.loads(path.read_text(encoding="utf-8"))["thresholds"].items():
        assert entry.get("regime", threshold_mod.FROZEN) == threshold_mod.FROZEN, name
        assert entry["frozen"] is True, name


# --------------------------------------------------------------------------- #
# the pixel model
# --------------------------------------------------------------------------- #


def test_pixel_model_trains_only_on_train_role_ground(cfg):
    """It has no tiles, so it cannot straddle a seam -- but it can still be
    handed the wrong pixels, and that is the leak this checks."""
    reported = cfg.pixel_model["reported"]
    _, truth, _, provenance = pixel_model.training_pixels(cfg, reported)
    event = cfg.event(cfg.evaluation["threshold"]["calibrated_on"])
    assert event.is_train
    assert provenance["event"] == event.id
    assert provenance["role"] == "train"

    blocks = blocks_for_event(cfg, event, grid_for_event(cfg, event))
    held_out = set(blocks.blocks_with_role("calibration")) | set(
        blocks.blocks_with_role("test")
    )
    assert not held_out & set(provenance["blocks"])
    assert truth.any(), "a classifier fitted on pure background is not a classifier"


def test_pixel_model_refuses_to_fit_on_a_test_event(cfg):
    """Point the training pointer at a test event and the fit must refuse.
    The configuration is what decides where training pixels come from, so this
    is the leak that a typo in one YAML line could open."""
    test_event = next(e for e in cfg.events.values() if not e.is_train)
    hijacked = copy.deepcopy(cfg)
    hijacked.raw["evaluation"]["threshold"]["calibrated_on"] = test_event.id
    with pytest.raises(ValueError, match="not the training event"):
        pixel_model.training_pixels(hijacked, cfg.pixel_model["reported"])


def _toy_stack(size: int = 8):
    """Eight synthetic channels, enough to enumerate what `features` computes."""
    rng = np.random.default_rng(2)
    channels = tuple(
        f"{phase}_{asset}"
        for phase in ("pre", "post")
        for asset in ("red", "nir08", "swir16", "swir22")
    )
    return SimpleNamespace(
        channels=channels,
        data=rng.random((len(channels), size, size)).astype("float32") + 0.1,
        band=lambda phase, asset, _c=channels: None,
    )


def test_every_configured_feature_is_actually_computed(cfg):
    """config.yaml may only name features this module knows how to build, and
    the check must not wait for a four-minute fit to discover otherwise."""
    stack = _toy_stack()
    stack.band = lambda phase, asset: stack.data[stack.channels.index(f"{phase}_{asset}")]
    computed = set(pixel_model.features(stack))

    for feature_set in cfg.pixel_model["features"]:
        names = pixel_model.feature_names(cfg, feature_set)
        assert names, feature_set
        assert len(names) == len(set(names)), f"{feature_set} repeats a feature"
        unknown = sorted(set(names) - computed)
        assert not unknown, f"{feature_set} names uncomputed features: {unknown}"

    assert cfg.pixel_model["reported"] in cfg.pixel_model["features"]


def test_an_unknown_feature_set_is_refused(cfg):
    with pytest.raises(ValueError, match="unknown feature set"):
        pixel_model.feature_names(cfg, "everything")


def test_change_only_holds_no_absolute_post_fire_level(cfg):
    """The ablation is only interesting if it actually ablates something."""
    change_only = cfg.pixel_model["features"].get("change_only")
    if change_only is None:
        pytest.skip("no change_only feature set configured")
    assert not [n for n in change_only if n.startswith("post_")]
    full = cfg.pixel_model["features"][cfg.pixel_model["reported"]]
    assert [n for n in full if n.startswith("post_")], (
        "the reported feature set has no absolute post-fire level, so the "
        "ablation removes nothing and proves nothing"
    )


# --------------------------------------------------------------------------- #
# the shipped grid
# --------------------------------------------------------------------------- #


def test_every_method_saw_the_same_pixels(grid):
    """The property every column comparison in the table rests on."""
    by_event: dict[str, set] = {}
    for row in grid["rows"]:
        by_event.setdefault(row["event"], set()).add(
            (row["domain_pixels"], round(row["label_area_ha"], 6))
        )
    for event, seen in by_event.items():
        assert len(seen) == 1, f"{event}: methods scored on different domains: {seen}"


def test_the_grid_is_complete(grid):
    """A missing cell is a hole, not a guard: the decomposition is the grid."""
    expected = len(grid["unsupervised_estimators"]) + 2  # frozen + oracle
    counted: dict[tuple[str, str], int] = {}
    for row in grid["rows"]:
        counted[(row["event"], row["method"])] = counted.get(
            (row["event"], row["method"]), 0
        ) + 1
    assert counted, "no rows"
    for key, n in counted.items():
        assert n == expected, f"{key} has {n} regimes, expected {expected}"


def test_both_unsupervised_estimators_are_reported(grid):
    """Otsu and the mixture were both tried on the test events before either
    was configured. Publishing only the winner would hide that selection."""
    reported = {r["regime"] for r in grid["rows"] if r["regime_family"] == "unsupervised"}
    for estimator in grid["unsupervised_estimators"]:
        assert f"unsupervised ({estimator})" in reported


def test_no_oracle_row_is_marked_deployable(grid):
    """The oracle needs the answer to produce the answer. It is an instrument,
    and nothing in the payload may suggest otherwise."""
    for row in grid["rows"]:
        if row["regime_family"] == "oracle":
            assert row["deployable"] is False
            assert row["uses_target_labels"] is True
        else:
            assert row["uses_target_labels"] is False


def test_the_oracle_bounds_its_own_method(grid):
    """Refit on the event itself, so by construction no other regime of the
    same score can beat it. If one does, the regimes are not sharing pixels."""
    best: dict[tuple[str, str], float] = {}
    oracle: dict[tuple[str, str], float] = {}
    for row in grid["rows"]:
        key = (row["event"], row["method"])
        if row["regime_family"] == "oracle":
            oracle[key] = row["f1"]
        else:
            best[key] = max(best.get(key, 0.0), row["f1"])
    for key, ceiling in oracle.items():
        assert ceiling >= best[key] - 1e-9, key


def test_average_precision_does_not_move_with_the_regime(grid):
    """AP is threshold-free, so it belongs to the score, not the operating
    point. If it moves down a method's rows, the rows are not one score."""
    seen: dict[tuple[str, str], set] = {}
    for row in grid["rows"]:
        seen.setdefault((row["event"], row["method"]), set()).add(
            round(row["average_precision"], 9)
        )
    for key, values in seen.items():
        assert len(values) == 1, f"{key}: average precision moved with the threshold"


def test_one_frozen_threshold_per_method_across_events(grid):
    """The frozen regime is only frozen if it is the same number everywhere."""
    by_method: dict[str, set] = {}
    for row in grid["rows"]:
        if row["regime_family"] == "frozen":
            by_method.setdefault(row["method"], set()).add(round(row["threshold"], 6))
    for method, values in by_method.items():
        assert len(values) == 1, (
            f"{method} used {len(values)} frozen thresholds across events, which is a "
            "recalibration on the target by another name"
        )
        frozen = grid["frozen_thresholds"][method]
        assert values.pop() == pytest.approx(frozen["value"], abs=1e-5)


def test_frozen_thresholds_were_calibrated_off_every_test_event(grid, cfg):
    for method, entry in grid["frozen_thresholds"].items():
        assert entry["frozen"] is True, method
        assert entry["uses_labels"] is True, method
        assert cfg.evaluation["threshold"]["calibrated_on"] in entry["calibrated_on"]
        for event in cfg.events.values():
            if not event.is_train:
                assert event.id not in entry["calibrated_on"], method


def test_frozen_thresholds_share_their_calibration_pixels(grid):
    """Threshold symmetry, extended to every method in the grid."""
    entries = list(grid["frozen_thresholds"].values())
    assert len({e["n_pixels"] for e in entries}) == 1
    assert len({e["n_positive"] for e in entries}) == 1
    assert len({e["objective"] for e in entries}) == 1


def test_no_interval_is_published_without_enough_burned_blocks(grid):
    minimum = grid["min_blocks_for_interval"]
    for row in grid["rows"]:
        for interval in (row["interval"] or {}).values():
            if interval["estimable"]:
                assert interval["n_burned_blocks"] >= minimum
            else:
                assert interval["reason"]


def test_forbidden_metrics_stay_out_of_the_grid(grid, cfg):
    for metric in cfg.evaluation.get("forbidden_metrics", ()):
        for row in grid["rows"]:
            assert metric not in row


def test_the_original_two_method_table_is_untouched(cfg):
    """The grid is an addition. The shipped protocol table keeps its own rules,
    including the asymmetry that denies the network an oracle row."""
    path = cfg.path_for("outputs", "results.json")
    if not path.exists():
        pytest.skip("run `python -m src.model.evaluate` first")
    results = json.loads(path.read_text(encoding="utf-8"))
    assert not any(
        r["method"].startswith("U-Net") and "oracle" in r["method"].lower()
        for r in results["rows"]
    )
    assert {r["method"] for r in results["rows"]} == {
        "U-Net",
        "dNBR honest",
        "dNBR oracle",
    }
