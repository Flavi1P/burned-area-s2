"""Copernicus EMS Rapid Mapping labels.

Downloads a delineation package, reads the burned perimeter, records how the
perimeter was actually produced, and rasterises it onto the event grid.

Why the provenance matters enough to have its own code path: EMS delineations
are not produced the same way twice. Each polygon carries a ``det_method``
attribute -- "Visual interpretation", "Semi-automatic extraction", ... -- and
when it is semi-automatic, "the U-Net beats dNBR" no longer measures agreement
with ground truth but agreement with whatever the operator's classifier decided.
The result stays valid; it simply does not say what a hurried reader assumes it
says. So the method is read from the data, written into config.yaml, and
reported per event in the README.

The labels are never derived from a dNBR threshold. dNBR is the baseline under
test; using it as truth would make the whole evaluation circular.
"""

from __future__ import annotations

import argparse
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pyogrio
import requests
from rasterio.features import rasterize

from ..config import Config, Event, load_config
from .grid import Grid, grid_for_event

TIMEOUT = 300


# --------------------------------------------------------------------------- #
# package retrieval
# --------------------------------------------------------------------------- #


def package_dir(cfg: Config, event: Event) -> Path:
    return cfg.path_for("data_raw", "ems", event.label.product_id)


def fetch_package(cfg: Config, event: Event, force: bool = False) -> Path:
    """Download and unpack the EMS delineation package. Idempotent."""
    dest = package_dir(cfg, event)
    if dest.exists() and any(dest.iterdir()) and not force:
        return dest

    url = event.label.package_url
    dest.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(dest)
    return dest


def _geopackage(pkg: Path) -> Path:
    gpkgs = sorted(pkg.glob("*.gpkg"))
    if not gpkgs:
        raise FileNotFoundError(f"no .gpkg in EMS package {pkg}")
    return gpkgs[0]


def _resolve_layer(gpkg: Path, prefix: str) -> str:
    """EMS suffixes layer names with the product version (observedEventA_v2)."""
    names = [str(row[0]) for row in pyogrio.list_layers(gpkg)]
    matches = [n for n in names if n.startswith(prefix)]
    if not matches:
        raise KeyError(f"no layer starting with {prefix!r} in {gpkg.name}: {names}")
    return sorted(matches)[-1]


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Provenance:
    """What EMS says about how this delineation was made."""

    det_methods: dict[str, int]  # detection method -> polygon count
    source_imagery: tuple[str, ...]  # post-event sources actually cited
    analysis_scale: str | None
    mmu_m2: float | None
    geometric_rmse_m: float | None
    lineage_statement: str | None

    @property
    def dominant_method(self) -> str:
        return max(self.det_methods.items(), key=lambda kv: kv[1])[0]

    def as_config_block(self) -> dict[str, Any]:
        """The mapping to paste under ``label.production`` in config.yaml."""
        return {
            "method": self.dominant_method,
            "analysis_scale": self.analysis_scale,
            "mmu_m2": self.mmu_m2,
            "geometric_rmse_m": self.geometric_rmse_m,
        }


_SCALE_RE = re.compile(r"scale of analysis is\s*(1:\s*\d+)", re.I)
# The metadata writes the minimum mapping unit as "576 m"; it is an area.
_MMU_RE = re.compile(r"MMU is\s*([\d.]+)\s*m", re.I)
_RMSE_RE = re.compile(r"RMSE is\s*([\d.]+)\s*m", re.I)


def _lineage_statement(pkg: Path, prefix: str) -> str | None:
    xmls = sorted(pkg.glob(f"*{prefix}*.xml"))
    if not xmls:
        return None
    text = xmls[0].read_text(encoding="utf-8", errors="replace")
    match = re.search(r"<statement>\s*<gco:CharacterString>(.*?)</gco:CharacterString>",
                      text, re.S)
    return " ".join(match.group(1).split()) if match else None


def read_provenance(cfg: Config, event: Event) -> Provenance:
    pkg = fetch_package(cfg, event)
    gpkg = _geopackage(pkg)
    prefix = cfg.ems["observed_event_layer"]

    gdf = gpd.read_file(gpkg, layer=_resolve_layer(gpkg, prefix))
    methods: dict[str, int] = {}
    if "det_method" in gdf.columns:
        methods = {str(k): int(v) for k, v in gdf["det_method"].value_counts().items()}
    if not methods:
        methods = {"not stated in the EMS attributes": len(gdf)}

    sources: list[str] = []
    try:
        src = gpd.read_file(gpkg, layer=_resolve_layer(gpkg, "source"))
        for _, row in src.iterrows():
            if str(row.get("eventphase", "")).lower().startswith("post"):
                sources.append(
                    f"{row.get('source_nam')} {row.get('src_date')} "
                    f"({row.get('sensor_res')})".strip()
                )
    except (KeyError, ValueError):
        pass

    statement = _lineage_statement(pkg, prefix)
    scale = mmu = rmse = None
    if statement:
        if m := _SCALE_RE.search(statement):
            scale = m.group(1).replace(" ", "")
        if m := _MMU_RE.search(statement):
            mmu = float(m.group(1))
        if m := _RMSE_RE.search(statement):
            rmse = float(m.group(1))

    return Provenance(
        det_methods=methods,
        source_imagery=tuple(sources),
        analysis_scale=scale,
        mmu_m2=mmu,
        geometric_rmse_m=rmse,
        lineage_statement=statement,
    )


# --------------------------------------------------------------------------- #
# geometry and rasterisation
# --------------------------------------------------------------------------- #


def read_observed_event(cfg: Config, event: Event, crs: Any = None) -> gpd.GeoDataFrame:
    """The burned perimeter of one event, as delivered by EMS."""
    pkg = fetch_package(cfg, event)
    gpkg = _geopackage(pkg)
    layer = _resolve_layer(gpkg, cfg.ems["observed_event_layer"])
    gdf = gpd.read_file(gpkg, layer=layer)
    return gdf.to_crs(crs) if crs is not None else gdf


def read_area_of_interest(cfg: Config, event: Event, crs: Any = None) -> gpd.GeoDataFrame:
    pkg = fetch_package(cfg, event)
    gpkg = _geopackage(pkg)
    layer = _resolve_layer(gpkg, cfg.ems["aoi_layer"])
    gdf = gpd.read_file(gpkg, layer=layer)
    return gdf.to_crs(crs) if crs is not None else gdf


def rasterize_label(gdf: gpd.GeoDataFrame, grid: Grid) -> np.ndarray:
    """Burn the perimeter onto the grid as a uint8 mask.

    ``all_touched=False``: a pixel is burned when its centre falls inside the
    polygon. ``all_touched=True`` would systematically dilate every scar by one
    pixel, which at 20 m is a 20 m outward bias on a boundary whose own RMSE is
    8 m -- a bias applied identically to both methods, but a bias nonetheless,
    and one that inflates the positive class.
    """
    if gdf.crs is None:
        raise ValueError("label geometry has no CRS")
    gdf = gdf.to_crs(grid.crs)
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        raise ValueError("label layer contains no usable geometry")
    return rasterize(
        shapes,
        out_shape=grid.shape,
        transform=grid.transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )


def label_raster_path(cfg: Config, event: Event) -> Path:
    return cfg.path_for("data_interim", event.id, f"label_{event.label.product_id}.tif")


def write_label_raster(cfg: Config, event: Event, grid: Grid) -> tuple[Path, dict[str, float]]:
    """Rasterise the label onto the event grid and write it next to the imagery.

    Returns the path and the two areas that section 6.4 requires be compared:
    the polygon area and the area actually rasterised. Their difference is the
    quantisation cost of the 20 m grid, and it is the floor under any area error
    the models can be blamed for.
    """
    import rasterio

    gdf = read_observed_event(cfg, event, crs=grid.crs)
    mask = rasterize_label(gdf, grid)

    path = label_raster_path(cfg, event)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **grid.profile("uint8", nodata=255)) as dst:
        dst.write(mask, 1)
        dst.update_tags(
            ems_product=event.label.product_id,
            ems_status=event.label.status,
            ems_reference_date=event.label.reference_date.isoformat(),
        )

    areas = {
        "polygon_area_ha": float(gdf.area.sum() / 1e4),
        "rasterised_area_ha": float(mask.sum() * grid.pixel_area_ha),
        "reported_area_ha": event.label.reported_burnt_area_ha,
    }
    return path, areas


# --------------------------------------------------------------------------- #
# CLI -- T0.1: read the production method of every event and report the exact
# YAML to paste into config.yaml.
# --------------------------------------------------------------------------- #


def _report(cfg: Config, event_ids: list[str]) -> None:
    for eid in event_ids:
        event = cfg.event(eid)
        prov = read_provenance(cfg, event)
        grid = grid_for_event(cfg, event)
        gdf = read_observed_event(cfg, event, crs=grid.crs)
        mask = rasterize_label(gdf, grid)

        print(f"\n=== {event.id}  {event.label.product_id}")
        print(f"    status            : {event.label.status}")
        print(f"    reference date    : {event.label.reference_date}")
        print(f"    detection methods : {prov.det_methods}")
        print(f"    post-event sources: {list(prov.source_imagery)}")
        print(f"    analysis scale    : {prov.analysis_scale}")
        print(f"    MMU / RMSE        : {prov.mmu_m2} m2 / {prov.geometric_rmse_m} m")
        print(f"    polygons          : {len(gdf)}")
        print(f"    polygon area      : {gdf.area.sum() / 1e4:10.1f} ha")
        print(f"    rasterised @{grid.resolution:g} m : {mask.sum() * grid.pixel_area_ha:10.1f} ha")
        print(f"    EMS reported area : {event.label.reported_burnt_area_ha:10.1f} ha")
        print(f"    grid              : {grid}")
        print("    --- paste under label.production in config.yaml ---")
        for key, value in prov.as_config_block().items():
            rendered = "null" if value is None else (
                f'"{value}"' if isinstance(value, str) else value
            )
            print(f"        {key}: {rendered}")
        if prov.lineage_statement:
            print(f"    lineage: {prov.lineage_statement}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--event",
        action="append",
        dest="events",
        help="event identifier from config.yaml; repeatable, default all",
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    _report(cfg, args.events or list(cfg.events))


if __name__ == "__main__":  # pragma: no cover
    main()
