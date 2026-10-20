"""The README may only quote numbers the pipeline produced."""

import json

from src.eval.check_readme import ROOT, mismatches


def _summary():
    return json.loads((ROOT / "outputs" / "regimes.json").read_text(encoding="utf-8"))["summary"]


def _cell(event, method, regime):
    return next(
        c for c in _summary()
        if (c["event"], c["method"], c["regime"]) == (event, method, regime)
    )


def _readme():
    return (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_table_matches_regimes_json():
    assert mismatches(_readme(), _summary()) == []


def test_edited_number_is_caught():
    readme = _readme()
    mean = f"{_cell('fontainebleau', 'GB pixel', 'unsupervised (otsu)')['iou']['mean']:.3f}"
    row = next(line for line in readme.splitlines() if line.startswith("| fontainebleau | GB pixel |"))
    tampered = readme.replace(row, row.replace(mean, f"{float(mean) + 0.02:.3f}", 1))
    assert tampered != readme
    assert any("Otsu" in e for e in mismatches(tampered, _summary()))


def test_dropping_the_seed_spread_is_caught():
    """A seeded mean without its sd reads as a single exact number."""
    spread = _cell("saumos", "U-Net", "frozen")["iou"]
    text = f"| Event | Method | frozen |\n|---|---|---|\n| saumos | U-Net | {spread['mean']:.3f} |\n"
    assert any("without its sd" in e for e in mismatches(text, _summary()))


def test_missing_table_is_an_error():
    assert mismatches("# no table here", _summary()) == ["no results table found in README"]
