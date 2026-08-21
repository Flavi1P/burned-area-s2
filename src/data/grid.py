"""Target analysis grid.

One grid per event, shared by every raster the event produces: the eight image
channels, the SCL validity masks, the rasterised label, the dNBR, the model
score. "Same preprocessing, same information, same splits" is the backbone of
the comparison table, and it is only literally true if every layer lands on the
same pixels.

The grid is defined at 20 m and snapped outward to whole multiples of the
resolution in the target UTM CRS, which is also how the Sentinel-2 20 m bands
are laid out -- so B8A/B11/B12 resample onto it as an identity, and only the
10 m red band is actually aggregated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

from ..config import Config, Event

WGS84 = CRS.from_epsg(4326)


@dataclass(frozen=True)
class Grid:
    crs: CRS
    transform: Affine
    width: int
    height: int

    @property
    def resolution(self) -> float:
        return abs(self.transform.a)

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        minx = self.transform.c
        maxy = self.transform.f
        maxx = minx + self.width * self.transform.a
        miny = maxy + self.height * self.transform.e
        return minx, miny, maxx, maxy

    @property
    def pixel_area_ha(self) -> float:
        return self.resolution**2 / 10_000.0

    def profile(self, dtype: str, nodata: Any = None, count: int = 1) -> dict[str, Any]:
        """A rasterio creation profile for a raster on this grid (tiled COG)."""
        return {
            "driver": "GTiff",
            "dtype": dtype,
            "nodata": nodata,
            "count": count,
            "width": self.width,
            "height": self.height,
            "crs": self.crs,
            "transform": self.transform,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "compress": "deflate",
            "predictor": 2 if dtype.startswith(("int", "uint")) else 3,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        minx, miny, maxx, maxy = self.bounds
        return (
            f"Grid({self.crs.to_string()}, {self.width}x{self.height} px @ "
            f"{self.resolution:g} m, "
            f"{(maxx - minx) / 1000:.1f}x{(maxy - miny) / 1000:.1f} km)"
        )


def utm_crs_for_bbox(bbox: tuple[float, float, float, float]) -> CRS:
    """UTM zone of the bbox centre. Northern hemisphere only, as all three
    events are in metropolitan France."""
    lon = (bbox[0] + bbox[2]) / 2
    lat = (bbox[1] + bbox[3]) / 2
    if lat < 0:
        raise ValueError("southern hemisphere not handled; no event needs it")
    zone = int(math.floor((lon + 180) / 6) + 1)
    return CRS.from_epsg(32600 + zone)


def snap_bounds(
    bounds: tuple[float, float, float, float], resolution: float
) -> tuple[float, float, float, float]:
    """Expand bounds outward to whole multiples of the resolution."""
    minx, miny, maxx, maxy = bounds
    return (
        math.floor(minx / resolution) * resolution,
        math.floor(miny / resolution) * resolution,
        math.ceil(maxx / resolution) * resolution,
        math.ceil(maxy / resolution) * resolution,
    )


def grid_from_bounds(
    bounds: tuple[float, float, float, float], crs: CRS, resolution: float
) -> Grid:
    minx, miny, maxx, maxy = snap_bounds(bounds, resolution)
    width = int(round((maxx - minx) / resolution))
    height = int(round((maxy - miny) / resolution))
    return Grid(
        crs=crs,
        transform=Affine(resolution, 0.0, minx, 0.0, -resolution, maxy),
        width=width,
        height=height,
    )


def grid_for_event(cfg: Config, event: Event, crs: CRS | None = None) -> Grid:
    """The analysis grid of one event.

    The footprint is the EMS area of interest plus a configured buffer, and it
    is purely geometric: in test, every tile falling inside it is evaluated,
    including the ones that are entirely unburned. Filtering negatives would
    inflate precision by an arbitrary factor and make the number comparable to
    nothing.
    """
    if crs is None:
        configured = cfg.project.get("target_crs")
        crs = CRS.from_user_input(configured) if configured else utm_crs_for_bbox(event.aoi_bbox)

    buffer_m = float(cfg.evaluation.get("test_footprint_buffer_m", 0.0))
    bounds = transform_bounds(WGS84, crs, *event.aoi_bbox, densify_pts=21)
    bounds = (
        bounds[0] - buffer_m,
        bounds[1] - buffer_m,
        bounds[2] + buffer_m,
        bounds[3] + buffer_m,
    )
    return grid_from_bounds(bounds, crs, cfg.resolution_m)
