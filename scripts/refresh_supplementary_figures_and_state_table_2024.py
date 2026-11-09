"""State-level raster coverage, shared by the submission workflow."""
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

CHUNK_SIZE = 1024
STATE_TABLE_OUT = AUDEBERT_DIR / "bundesland_coverage.csv"

def load_pickle(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def germany_geom():
    germany = gs.data.load_germany_outline()
    if hasattr(germany.geometry, "union_all"):
        return germany.geometry.union_all()
    return germany.unary_union


def valid_polys(polys):
    return [p for p in polys if p is not None and not p.is_empty]


def clipped_union(polys, germany):
    vals = valid_polys(polys)
    if not vals:
        return None
    return unary_union(vals).intersection(germany)


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


def build_scenario_geometries(germany):
    stroke_30 = load_pickle(RAW_DATA / "poly30.pkl")
    ivt_30 = load_pickle(RAW_DATA / "poly30_lysis_2024.pkl")
    mt_60 = load_pickle(RAW_DATA / "poly60_thrombectomy_2024.pkl")
    return {
        "SU_30_pct": clipped_union(stroke_30, germany),
        "Lysis_30_pct": clipped_union(ivt_30, germany),
        "MT_60_pct": clipped_union(mt_60, germany),
    }


def compute_state_raster_coverage() -> tuple[pd.DataFrame, pd.DataFrame]:
    germany = germany_geom()
    states = gs.data.load_states()[["NAME_1", "geometry"]].copy()
    states = states.rename(columns={"NAME_1": "Bundesland"}).reset_index(drop=True)
    states["state_id"] = np.arange(1, len(states) + 1)

    county_df = pd.read_csv(
        AUDEBERT_DIR / "county_STR_analysis.csv",
        usecols=["lan_name_str", "total_pop", "pop_65plus", "expected_strokes"],
    )
    county_agg = (
        county_df.groupby("lan_name_str", as_index=False)
        .agg(
            N_counties=("lan_name_str", "size"),
            Pop=("total_pop", "sum"),
            Pop_65plus=("pop_65plus", "sum"),
            Expected_strokes=("expected_strokes", "sum"),
        )
        .rename(columns={"lan_name_str": "Bundesland"})
    )
    county_agg["Pct_65plus"] = county_agg["Pop_65plus"] / county_agg["Pop"] * 100

    existing_state = None
    if STATE_TABLE_OUT.exists():
        existing_state = pd.read_csv(STATE_TABLE_OUT)

    scenario_geoms = build_scenario_geometries(germany)
    totals = {name: 0.0 for name in states["Bundesland"]}
    covered = {
        scen: {name: 0.0 for name in states["Bundesland"]}
        for scen in scenario_geoms
    }

    with rasterio.open(gs.config.POP_RASTER) as pop_src:
        germany_window = from_bounds(*germany.bounds, transform=pop_src.transform)
        germany_window = germany_window.round_offsets().round_lengths()
        nodata = pop_src.nodata

        state_shapes = [(geom, int(sid)) for geom, sid in zip(states.geometry, states.state_id)]
        state_lookup = dict(zip(states.state_id, states.Bundesland))

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

            state_ids = rasterize(
                state_shapes,
                out_shape=out_shape,
                transform=transform,
                fill=0,
                dtype="int16",
                all_touched=False,
            )
            state_ids = np.where(germany_mask, state_ids, 0)

            ids = state_ids[germany_mask].astype(np.int64)
            weights = pop[germany_mask]
            state_sums = np.bincount(ids, weights=weights, minlength=len(states) + 1)
            for sid in np.nonzero(state_sums)[0]:
                if sid == 0:
                    continue
                totals[state_lookup[sid]] += float(state_sums[sid])

            for scen_name, geom in scenario_geoms.items():
                cov_mask = geometry_mask([mapping(geom)], out_shape=out_shape, transform=transform, invert=True)
                cov_mask &= germany_mask
                if not cov_mask.any():
                    continue
                cov_ids = state_ids[cov_mask].astype(np.int64)
                cov_weights = pop[cov_mask]
                cov_sums = np.bincount(cov_ids, weights=cov_weights, minlength=len(states) + 1)
                for sid in np.nonzero(cov_sums)[0]:
                    if sid == 0:
                        continue
                    covered[scen_name][state_lookup[sid]] += float(cov_sums[sid])

            if i % 10 == 0:
                print(f"  state raster chunk {i}")

    rows = []
    for _, state in states.iterrows():
        name = state["Bundesland"]
        agg = county_agg[county_agg["Bundesland"] == name]
        n_counties = int(agg["N_counties"].iloc[0]) if not agg.empty else 0
        pop_county = int(round(agg["Pop"].iloc[0])) if not agg.empty else 0
        pct_65plus = round(float(agg["Pct_65plus"].iloc[0]), 1) if not agg.empty else np.nan
        exp_strokes = int(round(float(agg["Expected_strokes"].iloc[0]))) if not agg.empty else 0

        total_pop = totals[name]
        row = {
            "Bundesland": name,
            "N_counties": n_counties,
            "Pop": pop_county,
            "Pct_65plus": pct_65plus,
            "Expected_strokes": exp_strokes,
        }
        for scen_name in ["SU_30_pct", "Lysis_30_pct", "MT_60_pct"]:
            row[scen_name] = round(covered[scen_name][name] / total_pop * 100, 1) if total_pop else np.nan
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("SU_30_pct").reset_index(drop=True)

    compare = df.copy()
    if existing_state is not None:
        compare = df.merge(
            existing_state[["Bundesland", "SU_30_pct", "Lysis_30_pct", "MT_60_pct"]],
            on="Bundesland",
            how="left",
            suffixes=("_raster", "_county_old"),
        )
        for col in ["SU_30_pct", "Lysis_30_pct", "MT_60_pct"]:
            compare[f"{col}_delta_pp"] = compare[f"{col}_raster"] - compare[f"{col}_county_old"]

    return df, compare
