"""Sentinel-2 L2A access over STAC.

Scenes come from the Element 84 earth-search catalogue: public COGs on S3, no
account, no token, so `git clone && conda env create && run` actually works for
whoever reads this repository.

Three things this module is careful about.

**Scene choice.** The post-fire scene is the acquisition closest to the date of
the imagery the EMS analyst delineated on -- not closest to the event, and not
the prettiest one in the window. A pre/post pair that disagrees with its own
label produces a systematic error no metric recovers from; residual smoke only
produces noise.

**Cloud screens, it never ranks.** A candidate too clouded *over the event
footprint* is dropped, because pixels under cloud carry no burn information for
either method. Among the survivors, the date alone decides. Scene-level
`eo:cloud_cover` is a pre-filter only: it is computed over a 110 km MGRS tile
and can be wrong by an order of magnitude about a 20 km footprint inside it.

**Resampling.** Every band lands on the event's 20 m grid. B8A/B11/B12 are
already 20 m and, because the grid is snapped to the same modulus as the
Sentinel-2 UTM grid, they map through as an identity. B04 is 10 m and is
averaged, not sampled: keeping one 10 m pixel in four would throw away three
quarters of the radiometry for no reason. SCL is categorical and is resampled
nearest -- averaging class codes would invent classes that do not exist.
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from ..config import Config, Event, load_config
from .grid import Grid, grid_for_event

# Categorical layers must never be interpolated.
CATEGORICAL_ASSETS = {"scl"}

# Scene Classification Layer codes that make a pixel unusable. Shadows are in
# here for a reason beyond bookkeeping: a cloud shadow over vegetation darkens
# NIR and SWIR much the way a burn scar does, and it is the textbook false
# positive of every index-based burned-area method.
SCL_CLOUD_SHADOW = (3, 8, 9, 10)  # shadow, cloud medium, cloud high, cirrus
SCL_NO_DATA = 0

# vsicurl settings that make reading a remote COG bearable: do not list the
# bucket directory, and keep a chunk cache big enough for a 20 m window.
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": str(64 * 1024 * 1024),
}


@dataclass(frozen=True)
class Scene:
    """One Sentinel-2 acquisition, resolved and justified."""

    item_id: str
    datetime: dt.datetime
    cloud_cover: float  # scene level, over the whole 110 km MGRS tile
    epsg: int
    phase: str  # "pre" or "post"
    assets: dict[str, str]
    reason: str
    footprint_cloud: float | None = None  # SCL cloud+shadow over the event grid

    @property
    def date(self) -> dt.date:
        return self.datetime.date()


def _client(cfg: Config) -> Client:
    return Client.open(cfg.stac["endpoint"])


def _epsg(props: dict) -> int:
    """STAC `proj` v1 spells it `proj:epsg`, v2 spells it `proj:code`."""
    if "proj:epsg" in props:
        return int(props["proj:epsg"])
    code = str(props.get("proj:code", ""))
    if code.upper().startswith("EPSG:"):
        return int(code.split(":")[1])
    raise KeyError("item carries no projection code")


def _with(scene: Scene, **changes) -> Scene:
    return replace(scene, **changes)


def search_scenes(cfg: Config, event: Event, phase: str) -> list[Scene]:
    """Candidate acquisitions for one phase of one event, most recent first."""
    window = event.imagery.pre_window if phase == "pre" else event.imagery.post_window
    search = _client(cfg).search(
        collections=[cfg.stac["collection"]],
        bbox=list(event.aoi_bbox),
        datetime=f"{window[0].isoformat()}/{window[1].isoformat()}",
        query={"eo:cloud_cover": {"lt": cfg.stac["cloud_cover_max"]}},
    )
    scenes = []
    for item in search.items():
        props = item.properties
        scenes.append(
            Scene(
                item_id=item.id,
                datetime=dt.datetime.fromisoformat(
                    props["datetime"].replace("Z", "+00:00")
                ),
                cloud_cover=float(props.get("eo:cloud_cover", float("nan"))),
                epsg=_epsg(props),
                phase=phase,
                assets={k: a.href for k, a in item.assets.items()},
                reason="",
            )
        )
    return sorted(scenes, key=lambda s: s.datetime, reverse=True)


def cloud_shadow_fraction(scl: np.ndarray) -> float:
    """Fraction of a footprint the SCL flags as cloud or cloud shadow."""
    return float(np.isin(scl, SCL_CLOUD_SHADOW).mean())


def footprint_cloud(cfg: Config, event: Event, scene: Scene, grid: Grid) -> float:
    """Cloud and shadow fraction of a candidate scene over the event grid.

    The Fontainebleau window is the case in point: its candidates run from 0.4%
    to 20% at scene level, and it is the 20% one that turns out to be 45%
    clouded directly over the burn.
    """
    scl = read_band(fetch_band(cfg, event, scene, cfg.project["qa_band"], grid))
    return cloud_shadow_fraction(scl)


def select_scene(
    cfg: Config, event: Event, phase: str, grid: Grid | None = None
) -> Scene:
    """Apply the configured selection rule and return the chosen scene.

    Post-fire: the acquisition closest to the imagery EMS delineated on.
    Pre-fire: the acquisition closest to ignition, to keep the phenological gap
    between the two dates short.

    Without a ``grid`` the footprint cloud cannot be measured and the choice
    falls back to the date alone; the scene's ``reason`` says so.
    """
    candidates = search_scenes(cfg, event, phase)
    if not candidates:
        raise LookupError(
            f"{event.id}/{phase}: no scene under "
            f"{cfg.stac['cloud_cover_max']}% scene cloud in the configured window"
        )

    pinned = event.imagery.post_date if phase == "post" else event.imagery.pre_date
    if pinned is not None:
        for scene in candidates:
            if scene.date == pinned:
                return _with(scene, reason=f"pinned in config.yaml as {phase}_date")
        raise LookupError(
            f"{event.id}/{phase}: pinned date {pinned} is not in the catalogue"
        )

    if phase == "post":
        rule = cfg.stac["post_scene_rule"]
        if rule != "nearest_to_label_reference_date":
            raise ValueError(f"unknown stac.post_scene_rule {rule!r}")
        target = event.label.reference_date
        anchor = f"the EMS label reference date {target}"
    else:
        target = event.event_datetime.date()
        anchor = f"the ignition date {target}"

    ranked = sorted(
        candidates, key=lambda s: (abs((s.date - target).days), s.cloud_cover)
    )

    if grid is None:
        chosen = ranked[0]
        gap = (chosen.date - target).days
        return _with(
            chosen,
            reason=(
                f"closest acquisition to {anchor} ({gap:+d} d), "
                f"{chosen.cloud_cover:.1f}% scene cloud, footprint cloud not checked"
            ),
        )

    limit = float(cfg.stac["aoi_cloud_max"])
    screened: list[Scene] = []
    for scene in ranked:
        scene = _with(scene, footprint_cloud=footprint_cloud(cfg, event, scene, grid))
        screened.append(scene)
        if scene.footprint_cloud <= limit:
            gap = (scene.date - target).days
            return _with(
                scene,
                reason=(
                    f"closest acquisition to {anchor} ({gap:+d} d) clearing the "
                    f"footprint-cloud screen: {scene.footprint_cloud:.1%} "
                    f"cloud/shadow, limit {limit:.0%}"
                ),
            )

    # Nothing clears the bar. Keep the clearest and say so out loud, rather than
    # quietly evaluating a mostly-masked footprint.
    chosen = min(screened, key=lambda s: s.footprint_cloud)
    gap = (chosen.date - target).days
    return _with(
        chosen,
        reason=(
            f"FALLBACK: no candidate under {limit:.0%} footprint cloud; kept the "
            f"clearest at {chosen.footprint_cloud:.1%} ({gap:+d} d from {anchor})"
        ),
    )


# --------------------------------------------------------------------------- #
# band retrieval
# --------------------------------------------------------------------------- #


def band_path(cfg: Config, event: Event, scene: Scene, asset: str) -> Path:
    return cfg.path_for(
        "data_raw",
        "s2",
        event.id,
        f"{scene.phase}_{scene.date:%Y%m%d}_{asset}.tif",
    )


def fetch_band(
    cfg: Config,
    event: Event,
    scene: Scene,
    asset: str,
    grid: Grid,
    force: bool = False,
) -> Path:
    """Clip one band to the event grid and write it as a local COG. Idempotent."""
    dest = band_path(cfg, event, scene, asset)
    if dest.exists() and not force:
        return dest
    if asset not in scene.assets:
        raise KeyError(f"scene {scene.item_id} has no asset {asset!r}")

    resampling = (
        Resampling.nearest if asset in CATEGORICAL_ASSETS else Resampling.average
    )
    dest.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(scene.assets[asset]) as src:
            with WarpedVRT(
                src,
                crs=grid.crs,
                transform=grid.transform,
                width=grid.width,
                height=grid.height,
                resampling=resampling,
            ) as vrt:
                data = vrt.read(1)
                dtype = vrt.dtypes[0]
                nodata = src.nodata

    profile = grid.profile(str(dtype), nodata=nodata)
    with rasterio.open(dest, "w", **profile) as dst:
        dst.write(data, 1)
        dst.update_tags(
            stac_item=scene.item_id,
            asset=asset,
            band=cfg.project["bands"].get(asset, asset),
            acquisition=scene.datetime.isoformat(),
            scene_cloud_cover=f"{scene.cloud_cover:.2f}",
            phase=scene.phase,
            selection_reason=scene.reason,
            resampling=resampling.name,
        )
    return dest


def fetch_scene(
    cfg: Config,
    event: Event,
    phase: str,
    grid: Grid | None = None,
    with_qa: bool = True,
    force: bool = False,
) -> tuple[Scene, dict[str, Path]]:
    """Select and download every configured band of one phase."""
    grid = grid or grid_for_event(cfg, event)
    scene = select_scene(cfg, event, phase, grid)
    assets = list(cfg.band_assets)
    if with_qa:
        assets.append(cfg.project["qa_band"])

    paths = {a: fetch_band(cfg, event, scene, a, grid, force=force) for a in assets}
    return scene, paths


def read_band(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--event", required=True, help="event identifier from config.yaml"
    )
    parser.add_argument("--phase", default="post", choices=["pre", "post"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--list",
        action="store_true",
        help="list the candidate scenes with their footprint cloud, and stop",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    event = cfg.event(args.event)
    grid = grid_for_event(cfg, event)

    if args.list:
        print(f"{event.id} / {args.phase} candidates   (grid: {grid})")
        for scene in search_scenes(cfg, event, args.phase):
            frac = footprint_cloud(cfg, event, scene, grid)
            print(
                f"  {scene.date}  scene cloud {scene.cloud_cover:5.1f}%  "
                f"footprint cloud/shadow {frac:6.1%}  {scene.item_id}"
            )
        return

    scene, paths = fetch_scene(cfg, event, args.phase, grid, force=args.force)
    print(f"event   : {event.id}")
    print(f"grid    : {grid}")
    print(f"scene   : {scene.item_id}  {scene.date}")
    print(f"chosen  : {scene.reason}")
    for asset, path in paths.items():
        band = cfg.project["bands"].get(asset, asset)
        print(f"  {asset:<7} {band:<5} {path.relative_to(path.parents[4])}")


if __name__ == "__main__":  # pragma: no cover
    main()
