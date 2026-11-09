"""Raster coverage computation used by reproduce_2024; no document editing."""
from pathlib import Path
import json
import math
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.windows import Window, from_bounds
from rasterio.enums import Resampling
from rasterio.warp import reproject
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import geostroke as gs
from geostroke import config

PRIVATE_DIR = config.DATA_DIR
RAW_DATA = config.CACHE_DIR
AUDEBERT_DIR = config.ANALYSIS_DIR
EXTERNAL_DIR = config.EXTERNAL_DIR

URBAN_CODES = {21,22,23,30}
RURAL_CODES = {11,12,13}
CHUNK_SIZE = 1024
COUNTY_TABLE = AUDEBERT_DIR / "county_binary_comparison.csv"

def pick_smod_raster() -> Path:
    candidates = sorted(EXTERNAL_DIR.glob("GHS_SMOD*.tif"))
    if not candidates:
        raise FileNotFoundError("No GHS-SMOD raster found in raw_data.")
    non_ea = [p for p in candidates if "_EA" not in p.name]
    return non_ea[0] if non_ea else candidates[0]


def germany_geom():
    germany = gs.data.load_germany_outline()
    if hasattr(germany.geometry, "union_all"):
        return germany.geometry.union_all()
    return germany.unary_union


def load_pickle(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def valid_polys(polys):
    return [p for p in polys if p is not None and not p.is_empty]


def clipped_union(polys, germany):
    vals = valid_polys(polys)
    if not vals:
        return None
    return unary_union(vals).intersection(germany)


def build_scenario_geometries(germany):
    stroke_df = gs.data.load_stroke_units()
    tele_mask = gs.data.telemed_mask(stroke_df)
    non_tele_idx = stroke_df.loc[~tele_mask].index.tolist()

    stroke_30 = load_pickle(RAW_DATA / "poly30.pkl")
    extended_30 = load_pickle(RAW_DATA / "poly30_extended_stroke.pkl")
    ivt_30 = load_pickle(RAW_DATA / "poly30_lysis_2024.pkl")
    mt_60 = load_pickle(RAW_DATA / "poly60_thrombectomy_2024.pkl")

    geoms = {
        "Non-telemedical certified stroke units": clipped_union([stroke_30[i] for i in non_tele_idx], germany),
        "All certified stroke units": clipped_union(stroke_30, germany),
        "Certified + stroke-ready hospitals (>100 OPS 8-98b/8-981 per year)": clipped_union(extended_30, germany),
        "IVT centers >=25/year": clipped_union(ivt_30, germany),
        "MT centers >=25/year": clipped_union(mt_60, germany),
    }
    facilities_n = {
        "Non-telemedical certified stroke units": int((~tele_mask).sum()),
        "All certified stroke units": int(len(stroke_df)),
        "Certified + stroke-ready hospitals (>100 OPS 8-98b/8-981 per year)": int(len(pd.read_csv(PRIVATE_DIR / "stroke_units_extended_geocoded.csv"))),
        "IVT centers >=25/year": int(len(pd.read_csv(PRIVATE_DIR / "lysis_centers_geocoded.csv"))),
        "MT centers >=25/year": int(len(pd.read_csv(PRIVATE_DIR / "thrombectomy_centers_geocoded.csv"))),
    }
    return geoms, facilities_n


def iter_windows(base_window: Window, chunk_size: int):
    row_start = int(math.floor(base_window.row_off))
    col_start = int(math.floor(base_window.col_off))
    row_stop = int(math.ceil(base_window.row_off + base_window.height))
    col_stop = int(math.ceil(base_window.col_off + base_window.width))
    for row_off in range(row_start, row_stop, chunk_size):
        height = min(chunk_size, row_stop - row_off)
        for col_off in range(col_start, col_stop, chunk_size):
            width = min(chunk_size, col_stop - col_off)
            yield Window(col_off=col_off, row_off=row_off, width=width, height=height)


def load_county_geometries_with_rates() -> list[tuple[object, float]]:
    counties_df = pd.read_csv(AUDEBERT_DIR / "county_STR_analysis.csv", usecols=["krs_code_str", "total_pop", "expected_strokes"])
    counties_df["krs_code_str"] = counties_df["krs_code_str"].astype(str).str.zfill(5)
    counties_df["expected_rate"] = counties_df["expected_strokes"] / counties_df["total_pop"]
    rate_map = dict(zip(counties_df["krs_code_str"], counties_df["expected_rate"]))

    raw_geo = json.loads(Path(gs.config.GERMANY_COUNTIES).read_text())
    shapes = []
    for feat in raw_geo["features"]:
        props = feat["properties"]
        code = props.get("krs_code")
        if isinstance(code, list):
            code = code[0]
        code = str(code).zfill(5)
        rate = float(rate_map.get(code, 0.0))
        shapes.append((shape(feat["geometry"]), rate))
    return shapes


def compute_full_raster_table():
    germany = germany_geom()
    smod_path = pick_smod_raster()
    scenario_geoms, facilities_n = build_scenario_geometries(germany)
    scenario_bounds = {label: geom.bounds if geom is not None and not geom.is_empty else None for label, geom in scenario_geoms.items()}
    county_rate_shapes = load_county_geometries_with_rates()

    totals = {
        "population": 0.0,
        "expected": 0.0,
        "urban_population": 0.0,
        "rural_population": 0.0,
        "other_population": 0.0,
    }
    covered = {
        label: {
            "population": 0.0,
            "expected": 0.0,
            "urban_population": 0.0,
            "rural_population": 0.0,
            "other_population": 0.0,
        }
        for label in scenario_geoms
    }

    with rasterio.open(gs.config.POP_RASTER) as pop_src, rasterio.open(smod_path) as smod_src:
        germany_window = from_bounds(*germany.bounds, transform=pop_src.transform)
        germany_window = germany_window.round_offsets().round_lengths()
        nodata = pop_src.nodata

        for i, window in enumerate(iter_windows(germany_window, CHUNK_SIZE), start=1):
            pop = pop_src.read(1, window=window).astype(np.float64)
            if nodata is not None:
                pop[pop == nodata] = 0.0
            if not np.any(pop):
                continue

            transform = pop_src.window_transform(window)
            out_shape = pop.shape
            germany_mask = geometry_mask([mapping(germany)], out_shape=out_shape, transform=transform, invert=True)
            if not germany_mask.any():
                continue
            pop[~germany_mask] = 0.0
            if not np.any(pop):
                continue

            smod_arr = np.zeros(out_shape, dtype=np.int16)
            reproject(
                source=rasterio.band(smod_src, 1),
                destination=smod_arr,
                src_transform=smod_src.transform,
                src_crs=smod_src.crs,
                dst_transform=transform,
                dst_crs=pop_src.crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )

            rate_arr = rasterize(
                county_rate_shapes,
                out_shape=out_shape,
                transform=transform,
                fill=0.0,
                dtype="float32",
                all_touched=False,
            ).astype(np.float64)
            rate_arr[~germany_mask] = 0.0
            expected_arr = pop * rate_arr

            urban_mask = germany_mask & np.isin(smod_arr, list(URBAN_CODES))
            rural_mask = germany_mask & np.isin(smod_arr, list(RURAL_CODES))
            other_mask = germany_mask & ~(urban_mask | rural_mask)

            totals["population"] += pop[germany_mask].sum()
            totals["expected"] += expected_arr[germany_mask].sum()
            totals["urban_population"] += pop[urban_mask].sum()
            totals["rural_population"] += pop[rural_mask].sum()
            totals["other_population"] += pop[other_mask].sum()

            window_bounds = rasterio.windows.bounds(window, pop_src.transform)
            for label, geom in scenario_geoms.items():
                if geom is None or geom.is_empty:
                    continue
                bounds = scenario_bounds[label]
                if bounds is not None:
                    if bounds[2] < window_bounds[0] or bounds[0] > window_bounds[2] or bounds[3] < window_bounds[1] or bounds[1] > window_bounds[3]:
                        continue
                cov_mask = geometry_mask([mapping(geom)], out_shape=out_shape, transform=transform, invert=True)
                cov_mask &= germany_mask
                if not cov_mask.any():
                    continue
                covered[label]["population"] += pop[cov_mask].sum()
                covered[label]["expected"] += expected_arr[cov_mask].sum()
                covered[label]["urban_population"] += pop[urban_mask & cov_mask].sum()
                covered[label]["rural_population"] += pop[rural_mask & cov_mask].sum()
                covered[label]["other_population"] += pop[other_mask & cov_mask].sum()

            if i % 10 == 0:
                print(f"  raster chunk {i}")

    rows = []
    county_df = pd.read_csv(COUNTY_TABLE) if COUNTY_TABLE.exists() else None
    compare_rows = []
    for label, n_fac in facilities_n.items():
        pop_pct = round(covered[label]["population"] / totals["population"] * 100, 2) if totals["population"] else np.nan
        age_pct = round(covered[label]["expected"] / totals["expected"] * 100, 2) if totals["expected"] else np.nan
        diff_pp = round(pop_pct - age_pct, 2) if not np.isnan(pop_pct) and not np.isnan(age_pct) else np.nan
        urban_pct = round(covered[label]["urban_population"] / totals["urban_population"] * 100, 2) if totals["urban_population"] else np.nan
        rural_pct = round(covered[label]["rural_population"] / totals["rural_population"] * 100, 2) if totals["rural_population"] else np.nan

        rows.append(
            {
                "Facility type": label,
                "Facilities (n)": n_fac,
                "Population Coverage (%)": pop_pct,
                "Age-Adjusted Coverage (%)": age_pct,
                "Diff (pp)": diff_pp,
                "Urban Coverage (%)": urban_pct,
                "Rural Coverage (%)": rural_pct,
            }
        )

        if county_df is not None:
            match = county_df[county_df["Facility type"] == label]
            if not match.empty:
                compare_rows.append(
                    {
                        "Facility type": label,
                        "County Population Coverage (%)": float(match.iloc[0]["County Population Coverage (%)"]),
                        "Raster Population Coverage (%)": pop_pct,
                        "Population Delta Raster-County (pp)": round(pop_pct - float(match.iloc[0]["County Population Coverage (%)"]), 2),
                        "County Age-Adjusted Coverage (%)": float(match.iloc[0]["Age-Adjusted Coverage (%)"]),
                        "Raster Age-Adjusted Coverage (%)": age_pct,
                        "Age-Adjusted Delta Raster-County (pp)": round(age_pct - float(match.iloc[0]["Age-Adjusted Coverage (%)"]), 2),
                        "County Urban Coverage (%)": float(match.iloc[0]["Urban Coverage (%)"]),
                        "Raster Urban Coverage (%)": urban_pct,
                        "Urban Delta Raster-County (pp)": round(urban_pct - float(match.iloc[0]["Urban Coverage (%)"]), 2),
                        "County Rural Coverage (%)": float(match.iloc[0]["Rural Coverage (%)"]),
                        "Raster Rural Coverage (%)": rural_pct,
                        "Rural Delta Raster-County (pp)": round(rural_pct - float(match.iloc[0]["Rural Coverage (%)"]), 2),
                    }
                )

    meta = {
        "population_raster": str(gs.config.POP_RASTER),
        "smod_raster": str(smod_path),
        "county_expected_rate_source": str(AUDEBERT_DIR / "county_STR_analysis.csv"),
        "chunk_size": CHUNK_SIZE,
        "urban_codes": sorted(URBAN_CODES),
        "rural_codes": sorted(RURAL_CODES),
        "total_population_raster": round(totals["population"]),
        "total_expected_strokes_rasterized": round(totals["expected"], 2),
        "total_urban_population_raster": round(totals["urban_population"]),
        "total_rural_population_raster": round(totals["rural_population"]),
        "total_other_population_raster": round(totals["other_population"]),
        "exact_raster_totals": {key: float(value) for key, value in totals.items()},
        "exact_covered_totals": {label: {key: float(value) for key, value in counts.items()}
            for label, counts in covered.items()},
    }
    return pd.DataFrame(rows), pd.DataFrame(compare_rows), meta
