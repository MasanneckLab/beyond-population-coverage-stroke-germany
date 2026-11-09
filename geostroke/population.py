"""Population coverage helpers.

Wrapper utilities around *rasterio* for masking the GHS population raster
with polygons and computing population counts + percentages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Any

import numpy as np
import rasterio
import rasterio.mask
from rasterio.io import DatasetReader  # for typing
from shapely.geometry import Polygon, MultiPolygon, mapping

from . import config

__all__ = [
    "population_within",
]

Geometry = Polygon | MultiPolygon


def _safe_mask(src: DatasetReader, geom: Geometry) -> np.ndarray:
    """Return masked raster *array* (single band)."""

    try:
        out_image, _ = rasterio.mask.mask(src, [mapping(geom)], crop=True)
    except Exception:
        # fallback: pass geometry list directly
        out_image, _ = rasterio.mask.mask(src, geom, crop=True)  # type: ignore[arg-type]

    arr = out_image[0]
    # Replace nodata values (usually -200) by zero
    nodata_val = src.nodata if src.nodata is not None else -200
    arr[arr == nodata_val] = 0
    return arr


def population_within(geom: Geometry, raster_path: Path | str | None = None) -> int:
    """Return integer population *within* the given geometry."""

    if geom is None or geom.is_empty:
        return 0
    raster_path = Path(raster_path or config.POP_RASTER)
    with rasterio.open(raster_path) as src:
        arr = _safe_mask(src, geom)
    return int(arr.sum())
