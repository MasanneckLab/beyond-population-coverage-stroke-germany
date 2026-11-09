"""Data ingestion utilities.

All raw tabular and spatial data for the GeoStroke publication are
loaded through helper functions here.  Having a single module makes it
simple to switch data sources or preprocess steps without touching the
analysis logic.
"""

from __future__ import annotations

import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Literal, Any

from . import config

__all__ = [
    "load_stroke_units",
    "load_extended_stroke_units",
    "load_hospitals_ct",
    "load_germany_outline",
    "load_states",
    "load_counties",
]

# ---------------------------------------------------------------------------
# Tabular data loaders
# ---------------------------------------------------------------------------

def load_stroke_units(csv_path: Path | str | None = None) -> pd.DataFrame:
    """Return DataFrame with stroke-unit facility information.

    By default this expects a file called ``stroke_units_geocoded.csv`` in
    :pydata:`config.DATA_DIR`.
    """

    path = Path(csv_path or config.DATA_DIR / "stroke_units_geocoded.csv")
    if not path.exists():
        alt = config.ROOT / "stroke_units_geocoded.csv"
        if alt.exists():
            path = alt
            import warnings
            warnings.warn(
                f"stroke_units_geocoded.csv found in project root; consider moving it to {config.DATA_DIR}",
                RuntimeWarning,
            )
    df = pd.read_csv(path)

    required = {
        "longitude",
        "latitude",
        "name",
        "level",
        "is_thrombectomy_center",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Stroke-units CSV missing columns: {missing}")

    return df.dropna(subset=["longitude", "latitude"]).copy()


def load_extended_stroke_units(csv_path: Path | str | None = None) -> pd.DataFrame:
    """Return DataFrame with extended stroke-unit facility information.

    This includes the regular stroke units plus additional hospitals that
    perform frequent stroke care procedures (OPS codes 8-98b and 8-981).
    By default this expects a file called ``stroke_units_extended_geocoded.csv`` in
    :pydata:`config.DATA_DIR`.
    """

    path = Path(csv_path or config.DATA_DIR / "stroke_units_extended_geocoded.csv")
    if not path.exists():
        alt = config.ROOT / "stroke_units_extended_geocoded.csv"
        if alt.exists():
            path = alt
            import warnings
            warnings.warn(
                f"stroke_units_extended_geocoded.csv found in project root; consider moving it to {config.DATA_DIR}",
                RuntimeWarning,
            )
    
    if not path.exists():
        raise FileNotFoundError(
            f"Extended stroke units file not found at {path}. "
            f"Please run the additional_stroke_centers.ipynb notebook to generate this file."
        )
    
    df = pd.read_csv(path)

    required = {
        "longitude",
        "latitude", 
        "name",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Extended stroke-units CSV missing columns: {missing}")

    return df.dropna(subset=["longitude", "latitude"]).copy()


def load_hospitals_ct(xlsx_path: Path | str | None = None) -> pd.DataFrame:
    """Return DataFrame of all German hospitals with a CT scanner.

    The XLSX is cleaned in the same way as the original notebook:
      1. Duplicate *name*s are dropped.
      2. Only rows with *zIndex* > 5 are kept (quality filter).
      3. Columns renamed to ``longitude`` / ``latitude``.
    """

    path = Path(xlsx_path or config.DATA_DIR / "Hospitals_with_CT.xlsx")
    if not path.exists():
        alt = config.ROOT / "Hospitals_with_CT.xlsx"
        if alt.exists():
            path = alt
            import warnings
            warnings.warn(
                f"Hospitals_with_CT.xlsx found in project root; consider moving it to {config.DATA_DIR}",
                RuntimeWarning,
            )
    df = pd.read_excel(path)

    df = (
        df.drop_duplicates(subset="name")
          .query("zIndex > 5")
          .rename(columns={"lng": "longitude", "lat": "latitude"})
          .loc[:, ["longitude", "latitude", "name"]]
          .dropna(subset=["longitude", "latitude"])
          .reset_index(drop=True)
    )
    return df

# ---------------------------------------------------------------------------
# Spatial data loaders
# ---------------------------------------------------------------------------

def _load_geojson(path: Path, layer_name: str | None = None) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return gpd.read_file(path, layer=layer_name)


def load_germany_outline() -> gpd.GeoDataFrame:
    """High-resolution Germany outline (EPSG:4326)."""
    return _load_geojson(config.GERMANY_OUTLINE).set_crs(4326)


def load_states() -> gpd.GeoDataFrame:
    """Return GeoDataFrame of German states."""
    return _load_geojson(config.GERMANY_STATES).set_crs(4326)


def load_counties() -> gpd.GeoDataFrame:
    """Return GeoDataFrame of German Kreise (counties)."""
    return _load_geojson(config.GERMANY_COUNTIES).set_crs(4326)

# ---------------------------------------------------------------------------
# Convenience masks used frequently by analysis modules
# ---------------------------------------------------------------------------

def stroke_unit_masks(df_stroke: pd.DataFrame):  # type: ignore
    """Return boolean indexers (überregional, regional/telemed, thrombectomy)."""

    uber = df_stroke["level"].str.contains("überregional", case=False, na=False)
    reg = df_stroke["level"].str.contains(r"Regionale |Telemed", case=True, na=False)
    thromb = df_stroke["is_thrombectomy_center"].fillna(False).astype(bool)
    return uber, reg, thromb 

def telemed_mask(df_stroke: pd.DataFrame):
    """Return boolean indexer for telemed stroke units."""
    return df_stroke["level"].str.contains(r"Telemed", case=True, na=False)
