"""Check that every number in the README results table matches outputs/regimes.json.

A README edited by hand drifts from the pipeline silently; this fails loudly
instead. Run: python -m src.eval.check_readme
"""

from __future__ import annotations

import json
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


def mismatches(readme: str, rows: list[dict]) -> list[str]:
    """Every README table cell that disagrees with regimes.json at 3 decimals."""
    index = {(r["event"], r["method"], r["regime"]): r for r in rows}
    events = {r["event"] for r in rows}
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
            row = index.get((cells[0], cells[1], regime))
            if row is None:
                errors.append(f"{cells[0]} / {cells[1]} / {regime}: not in regimes.json")
            elif f"{row[metric]:.3f}" != value:
                errors.append(f"{cells[0]} / {cells[1]} / {name}: README {value}, json {row[metric]:.3f}")
            checked += 1
    if checked == 0:
        errors.append("no results table found in README")
    return errors


def main() -> None:
    rows = json.loads((ROOT / "outputs" / "regimes.json").read_text(encoding="utf-8"))["rows"]
    errors = mismatches((ROOT / "README.md").read_text(encoding="utf-8"), rows)
    for e in errors:
        print("MISMATCH", e)
    print("README consistent with outputs/regimes.json" if not errors else f"{len(errors)} mismatches")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":  # pragma: no cover
    main()
