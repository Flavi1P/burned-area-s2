"""Check that every number in the README results table matches outputs/regimes.json.

A README edited by hand drifts from the pipeline silently; this fails loudly
instead. A cell is either a bare number (a method with no seed) or
``mean ± sd`` with an optional ``(min–max)``; a method trained on several seeds
may not be quoted without its sd. Run: python -m src.eval.check_readme
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# README column -> (regime label in regimes.json, metric key)
COLUMNS = {
    "frozen": ("frozen", "iou"),
    "Otsu": ("unsupervised (otsu)", "iou"),
    "oracle": ("oracle", "iou"),
    "AP": ("frozen", "average_precision"),
}
CELL = re.compile(r"^(\d\.\d{3})(?: ± (\d\.\d{3}))?(?: \((\d\.\d{3})–(\d\.\d{3})\))?$")


def _cell_errors(value: str, spread: dict) -> list[str]:
    match = CELL.match(value)
    if match is None:
        return [f"unreadable cell {value!r}"]
    mean, sd, low, high = match.groups()
    errors = []
    if mean != f"{spread['mean']:.3f}":
        errors.append(f"mean {mean}, json {spread['mean']:.3f}")
    if spread["sd"] is None:
        if sd is not None or low is not None:
            errors.append("spread quoted for a method with one value")
    elif sd is None:
        errors.append(f"mean quoted without its sd over {spread['n']} seeds")
    elif sd != f"{spread['sd']:.3f}":
        errors.append(f"sd {sd}, json {spread['sd']:.3f}")
    if low is not None and (low, high) != (f"{spread['min']:.3f}", f"{spread['max']:.3f}"):
        errors.append(f"range {low}–{high}, json {spread['min']:.3f}–{spread['max']:.3f}")
    return errors


def mismatches(readme: str, summary: list[dict]) -> list[str]:
    """Every README table cell that disagrees with the regimes.json summary."""
    index = {(c["event"], c["method"], c["regime"]): c for c in summary}
    events = {c["event"] for c in summary}
    header: list[str] | None = None
    errors, checked = [], 0
    for line in readme.splitlines():
        cells = [c.strip().strip("*") for c in line.strip().strip("|").split("|")]
        if cells[:2] == ["Event", "Method"]:
            header = cells
            continue
        if header is None or len(cells) != len(header) or cells[0] not in events:
            continue
        for name, value in zip(header[2:], cells[2:]):
            regime, metric = COLUMNS[name]
            cell = index.get((cells[0], cells[1], regime))
            if cell is None:
                errors.append(f"{cells[0]} / {cells[1]} / {regime}: not in regimes.json")
            else:
                spread = {**cell[metric], "n": cell["n_seeds"]}
                for e in _cell_errors(value, spread):
                    errors.append(f"{cells[0]} / {cells[1]} / {name}: {e}")
            checked += 1
    if checked == 0:
        errors.append("no results table found in README")
    return errors


def main() -> None:
    payload = json.loads((ROOT / "outputs" / "regimes.json").read_text(encoding="utf-8"))
    errors = mismatches((ROOT / "README.md").read_text(encoding="utf-8"), payload["summary"])
    for e in errors:
        print("MISMATCH", e)
    print("README consistent with outputs/regimes.json" if not errors else f"{len(errors)} mismatches")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":  # pragma: no cover
    main()
