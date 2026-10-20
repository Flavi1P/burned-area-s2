"""The README may only quote numbers the pipeline produced."""

import json

from src.eval.check_readme import ROOT, mismatches


def _rows():
    return json.loads((ROOT / "outputs" / "regimes.json").read_text(encoding="utf-8"))["rows"]


def test_readme_table_matches_regimes_json():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert mismatches(readme, _rows()) == []


def test_edited_number_is_caught():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tampered = readme.replace("| fontainebleau | GB pixel | 0.487 | 0.696 |", "| fontainebleau | GB pixel | 0.487 | 0.716 |")
    assert tampered != readme
    assert any("Otsu" in e for e in mismatches(tampered, _rows()))


def test_missing_table_is_an_error():
    assert mismatches("# no table here", _rows()) == ["no results table found in README"]
