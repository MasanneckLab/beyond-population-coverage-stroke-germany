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

from geostroke.str_models import gravity_alloc
from scripts.compute_str_travel_time_sparse_sensitivity_2024 import add_sparse_str_columns

def prepare_hospitals(frame, prefix):
    frame = frame.copy().reset_index(drop=True)
    frame["facility_id"] = [f"{prefix}_{i:04d}" for i in range(len(frame))]
    return frame


def county_geometry(counties):
    # GDAL drops list-valued GeoJSON properties with some installed versions.
    features = json.loads(config.GERMANY_COUNTIES.read_text())["features"]
    rows = []
    for feature in features:
        code = feature["properties"]["krs_code"]
        rows.append({"krs_code_str": str(code[0] if isinstance(code, list) else code).zfill(5),
                     "geometry": shape(feature["geometry"])})
    geo = gpd.GeoDataFrame(rows, crs=4326)
    geo = geo.set_index("krs_code_str").loc[counties.krs_code_str]
    centroids = gpd.GeoSeries(geo.to_crs(3035).geometry.centroid, crs=3035).to_crs(4326)
    out = counties.copy()
    out["centroid_lon"], out["centroid_lat"] = centroids.x.to_numpy(), centroids.y.to_numpy()
    return out


def assert_mass(counties, hospitals, column):
    if not np.isclose(counties[column].sum(), hospitals.total_cases.sum(), rtol=1e-12):
        raise ValueError(f"Procedure mass is not conserved in {column}")


def run_models(counties, ivt, mt, nodes, pairs, i63):
    out = county_geometry(counties)
    km_beta = []
    for therapy, hospitals, beta, radius, cutoff in [
            ("IVT", ivt, 2.0, 100, 90), ("MT", mt, 1.5, 150, 120)]:
        out[f"exp_{therapy}"] = out.expected_strokes * hospitals.total_cases.sum() / i63
        out[f"alloc_{therapy}"] = gravity_alloc(hospitals, out, beta, radius)
        out[f"STR_{therapy}"] = out[f"alloc_{therapy}"] / out[f"exp_{therapy}"]
        assert_mass(out, hospitals, f"alloc_{therapy}")
        for b in [1.0, 1.5, 2.0, 2.5, 3.0]:
            vals = gravity_alloc(hospitals, out, b, radius) / out[f"exp_{therapy}"].to_numpy()
            urban, rural = out.urban_rural.eq("urban"), out.urban_rural.eq("rural")
            km_beta.append({"Therapy": therapy, "Beta": b,
                "Mean_STR_urban": round(float(vals[urban].mean()), 2),
                "Mean_STR_rural": round(float(vals[rural].mean()), 2),
                "Pct_underserved_rural": round(float((vals[rural] < 0.5).mean()*100), 1)})
        add_sparse_str_columns(out, nodes, hospitals, pairs[therapy], therapy, beta,
            {"primary": float(cutoff), "narrow": float(cutoff-30), "candidate_envelope": None})
        for variant in ("primary", "narrow", "candidate_envelope"):
            assert_mass(out, hospitals, f"alloc_{therapy}_time_{variant}")
    return out.drop(columns=["centroid_lat", "centroid_lon"]), pd.DataFrame(km_beta)


def procedure_summary(raw, ten, main, i63):
    total = int(raw.total_cases.sum())
    covered = int(main.total_cases.sum())
    return dict(raw_sites=len(raw), total_proc=total, national_rate_pct=total/i63*100,
        centers_25=len(main), proc_thresh=covered, proc_thresh_pct=covered/total*100,
        median=float(main.total_cases.median()), q1=float(main.total_cases.quantile(.25)),
        q3=float(main.total_cases.quantile(.75)), centers_100=int(main.total_cases.ge(100).sum()),
        centers_10=len(ten), proc_10=int(ten.total_cases.sum()))


def county_fractions(counties, geoms):
    import geostroke as gs
    import rasterio
    from rasterio.mask import mask
    from rasterio.features import geometry_mask
    from shapely.geometry import shape, mapping
    features = json.loads(Path(gs.config.GERMANY_COUNTIES).read_text())["features"]
    shapes = {str(f["properties"]["krs_code"][0]): shape(f["geometry"]) for f in features}
    result = {key: [] for key in geoms}
    with rasterio.open(gs.config.POP_RASTER) as src:
        for n, code in enumerate(counties.krs_code_str, 1):
            geom = shapes[code]
            values, transform = mask(src, [mapping(geom)], crop=True, filled=True, nodata=0)
            population = values[0].astype(float)
            if not np.isfinite(population).all() or (population < 0).any() or population.sum() <= 0:
                raise ValueError(f"Invalid county raster population: {code}")
            for key, coverage in geoms.items():
                inside = geometry_mask([mapping(coverage)], out_shape=population.shape,
                    transform=transform, invert=True)
                result[key].append(float(population[inside].sum() / population.sum()))
            if n % 100 == 0:
                print("County fractions", n, "/400", flush=True)
    return result


def legacy_county_outputs(counties, summary):
    """Keep legacy comparison CSVs coherent, but do not substitute them for Table 2."""
    ur_rows, binary, continuous, sensitivity = [], [], [], []
    scenarios = [("Stroke Units (30 min)", "allsu", "SU_30"),
        ("Lysis \u226525/yr (30 min)", "ivt", "Lysis_30"),
        ("Thrombectomy \u226525/yr (60 min)", "mt", "MT_60")]
    for group in ["urban", "rural"]:
        sub = counties[counties.urban_rural.eq(group)]
        row = dict(Type=group, N_counties=len(sub), Pop=int(sub.total_pop.sum()),
            Pct_65plus=round(sub.pop_65plus.sum() / sub.total_pop.sum() * 100, 1),
            Stroke_rate=round(sub.expected_strokes.sum() / sub.total_pop.sum() * 100000), Expected_strokes=round(sub.expected_strokes.sum()))
        for _, key, prefix in scenarios:
            row[prefix + "_pop_pct"] = round((sub.total_pop * sub[f"pop_frac_{key}"]).sum() / sub.total_pop.sum() * 100, 1)
            row[prefix + "_stroke_pct"] = round((sub.expected_strokes * sub[f"pop_frac_{key}"]).sum() / sub.expected_strokes.sum() * 100, 1)
        ur_rows.append(row)
        summary["urban_rural"][group] = row
    for label, key, _ in scenarios:
        mask = counties[f"pop_frac_{key}"].gt(.5)
        cov, unc = counties[mask], counties[~mask]
        row = dict(Scenario=label, Counties_covered=int(mask.sum()), Counties_uncovered=int((~mask).sum()),
            Pop_covered_pct=round(cov.total_pop.sum() / counties.total_pop.sum() * 100, 1),
            Strokes_covered_pct=round(cov.expected_strokes.sum() / counties.expected_strokes.sum() * 100, 1))
        for name, sub in [("covered", cov), ("uncovered", unc)]:
            row["Pct65_" + name] = round(sub.pop_65plus.sum() / sub.total_pop.sum() * 100, 1)
            row["StrokeRate_" + name] = round(sub.expected_strokes.sum() / sub.total_pop.sum() * 100000)
        binary.append(row)
        for threshold in [.1, .25, .5, .75, .9]:
            mask = counties[f"pop_frac_{key}"].gt(threshold)
            pop = counties.loc[mask, "total_pop"].sum() / counties.total_pop.sum() * 100
            exp = counties.loc[mask, "expected_strokes"].sum() / counties.expected_strokes.sum() * 100
            sensitivity.append(dict(Scenario=label, Threshold=f">{threshold * 100:g}%", Threshold_num=threshold,
                N_covered=int(mask.sum()), N_uncovered=int((~mask).sum()), Pop_covered_pct=round(pop, 1),
                Strokes_covered_pct=round(exp, 1), Diff_pp=round(pop - exp, 1)))
    for label, key in [("Non-tele SU (30 min)", "nontel"), ("All SU (30 min)", "allsu"),
            ("IVT (30 min)", "ivt"), ("MT (60 min)", "mt")]:
        pop = (counties.total_pop * counties[f"pop_frac_{key}"]).sum() / counties.total_pop.sum() * 100
        exp = (counties.expected_strokes * counties[f"pop_frac_{key}"]).sum() / counties.expected_strokes.sum() * 100
        row = dict(crude_pct=round(pop, 1), age_adj_pct=round(exp, 1), diff_pp=round(pop - exp, 1), Scenario=label)
        for group in ["urban", "rural"]:
            sub = counties[counties.urban_rural.eq(group)]
            row[group + "_pct"] = round((sub.total_pop * sub[f"pop_frac_{key}"]).sum() / sub.total_pop.sum() * 100, 1)
        continuous.append(row)
    for name, rows in [("urban_rural_coverage.csv", ur_rows), ("age_adjusted_coverage.csv", binary),
            ("age_adjusted_coverage_continuous.csv", continuous), ("sensitivity_pop_threshold.csv", sensitivity)]:
        pd.DataFrame(rows).to_csv(config.ANALYSIS_DIR / name, index=False)
