"""Shared fixtures.

Rasters are git-ignored, so on a clean clone (and in CI) the guards that read
them cannot run. They skip with a reason that `pytest -rs` prints, rather than
failing on a missing file -- a red build for want of data would hide a red
build for a broken protocol.
"""

from __future__ import annotations

import pytest

from src.config import load_config
from src.data import ems, stack


@pytest.fixture(scope="session")
def rasters():
    cfg = load_config()
    missing = [
        path
        for event in cfg.events.values()
        for path in (ems.label_raster_path(cfg, event), stack.stack_path(cfg, event))
        if not path.exists()
    ]
    if missing:
        pytest.skip(
            f"{len(missing)} raster(s) missing, e.g. {missing[0].name}: run the "
            "data chain in README *Running it* first"
        )
