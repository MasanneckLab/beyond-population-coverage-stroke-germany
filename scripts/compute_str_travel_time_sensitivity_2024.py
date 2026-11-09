"""Population medoids and ORS requests for the primary travel-time STR model."""
from __future__ import annotations
import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.mask import mask
from scipy import stats
from shapely.geometry import mapping, shape
from geostroke import config

CACHE_DIR = config.CACHE_DIR / "travel_time_cache_2024"
COUNTY_STR = config.ANALYSIS_DIR / "county_STR_analysis.csv"
OUT_NODES = config.CACHE_DIR / "demand_nodes_population_medoids_2024.csv"
COUNTIES_GEOJSON = config.GERMANY_COUNTIES
POP_RASTER = config.POP_RASTER
ORS_URL = config.ORS_BASE_URL
AGG_FACTOR = 12  # 12 x 3-arc-second cells, approximately 1 km in Germany.
MAX_ROUTES = 500
DEST_CHUNK = 25
ZERO_SECONDS_EPS = 1.0
BETAS = [1.0, 1.5, 2.0, 2.5, 3.0]
KNOWN_DEMAND_NODE_REPAIRS = {
    # The original Uckermark medoid snaps to a disconnected ORS graph component.
    # Use the nearest routable populated raster cell in the same county demand cluster.
    "12073_2": {
        "lon": 14.274583299191546,
        "lat": 53.08374998926142,
        "repair_distance_km": 0.668,
        "repair_reason": "nearest_routable_populated_cell_same_cluster",
    }
}


def list_value(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value


def load_county_geometries(counties_df: pd.DataFrame) -> list[dict]:
    wanted = set(counties_df["krs_code_str"].astype(str).str.zfill(5))
    data = json.loads(COUNTIES_GEOJSON.read_text())
    rows = []
    for feat in data["features"]:
        props = feat["properties"]
        code = str(list_value(props.get("krs_code"))).zfill(5)
        if code not in wanted:
            continue
        rows.append(
            {
                "krs_code_str": code,
                "krs_name_geo": list_value(props.get("krs_name")),
                "geometry": shape(feat["geometry"]),
            }
        )
    if len(rows) != len(wanted):
        found = {r["krs_code_str"] for r in rows}
        missing = sorted(wanted - found)
        raise RuntimeError(f"County geometry mismatch: found {len(rows)}, missing {missing[:10]}")
    return rows


def xy_km(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    mean_lat = np.nanmean(lat)
    x = lon * 111.32 * math.cos(math.radians(mean_lat))
    y = lat * 110.57
    return np.column_stack([x, y]).astype(np.float64)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    threshold = q * cw[-1]
    return float(v[np.searchsorted(cw, threshold, side="left")])


def weighted_medoid_idx(xy: np.ndarray, weights: np.ndarray, candidates: np.ndarray | None = None) -> int:
    n = len(xy)
    if n == 1:
        return 0
    weight_sum = float(weights.sum())
    if weight_sum <= 0 or not np.isfinite(weight_sum):
        norm_weights = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        norm_weights = weights.astype(np.float64) / weight_sum
    if candidates is None:
        if n <= 2500:
            candidates = np.arange(n)
        else:
            centroid = np.average(xy, axis=0, weights=norm_weights)
            d_centroid = np.linalg.norm(xy - centroid, axis=1)
            near = np.argsort(d_centroid)[:1250]
            high_pop = np.argsort(norm_weights)[-1250:]
            candidates = np.unique(np.concatenate([near, high_pop]))
    candidates = np.asarray(candidates, dtype=int)
    best_idx = int(candidates[0])
    best_score = np.inf
    for start in range(0, len(candidates), 128):
        cand = candidates[start : start + 128]
        dx = xy[cand, 0][:, None] - xy[:, 0][None, :]
        dy = xy[cand, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        scores = (dist * norm_weights[None, :]).sum(axis=1)
        local = int(np.argmin(scores))
        if float(scores[local]) < best_score:
            best_score = float(scores[local])
            best_idx = int(cand[local])
    return best_idx


def weighted_k_medoids(xy: np.ndarray, weights: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(xy)
    k = min(k, n)
    if k == 1:
        medoids = np.array([weighted_medoid_idx(xy, weights)], dtype=int)
        return medoids, np.zeros(n, dtype=int)

    medoids = [weighted_medoid_idx(xy, weights)]
    nearest = np.linalg.norm(xy - xy[medoids[0]], axis=1)
    for _ in range(1, k):
        score = nearest * np.sqrt(weights / max(float(weights.mean()), 1e-9))
        score[medoids] = -1.0
        next_idx = int(np.argmax(score))
        medoids.append(next_idx)
        nearest = np.minimum(nearest, np.linalg.norm(xy - xy[next_idx], axis=1))

    medoids_arr = np.array(medoids, dtype=int)
    assignment = np.zeros(n, dtype=int)
    for _ in range(12):
        dist = np.sqrt(((xy[:, None, :] - xy[medoids_arr][None, :, :]) ** 2).sum(axis=2))
        assignment = np.argmin(dist, axis=1)
        new_medoids = []
        for cluster in range(k):
            idx = np.where(assignment == cluster)[0]
            if len(idx) == 0:
                new_medoids.append(int(medoids_arr[cluster]))
                continue
            cluster_medoid_local = weighted_medoid_idx(xy[idx], weights[idx])
            new_medoids.append(int(idx[cluster_medoid_local]))
        new_arr = np.array(new_medoids, dtype=int)
        if np.array_equal(new_arr, medoids_arr):
            break
        medoids_arr = new_arr
    dist = np.sqrt(((xy[:, None, :] - xy[medoids_arr][None, :, :]) ** 2).sum(axis=2))
    assignment = np.argmin(dist, axis=1)
    return medoids_arr, assignment


def aggregate_population_cells(src: rasterio.DatasetReader, geom) -> pd.DataFrame:
    arr, transform = mask(src, [mapping(geom)], crop=True, filled=True, nodata=0)
    pop = arr[0].astype(np.float64)
    pop[~np.isfinite(pop)] = 0.0
    pop[pop < 0] = 0.0
    if float(pop.sum()) <= 0:
        return pd.DataFrame(columns=["lon", "lat", "pop"])

    h, w = pop.shape
    pad_h = int(math.ceil(h / AGG_FACTOR) * AGG_FACTOR - h)
    pad_w = int(math.ceil(w / AGG_FACTOR) * AGG_FACTOR - w)
    if pad_h or pad_w:
        pop = np.pad(pop, ((0, pad_h), (0, pad_w)), constant_values=0.0)
    gh = pop.shape[0] // AGG_FACTOR
    gw = pop.shape[1] // AGG_FACTOR
    block_pop = pop.reshape(gh, AGG_FACTOR, gw, AGG_FACTOR).sum(axis=(1, 3))
    rr, cc = np.where(block_pop > 0)
    if len(rr) == 0:
        return pd.DataFrame(columns=["lon", "lat", "pop"])

    # Coordinates are centers of the aggregated raster support.
    rows = rr * AGG_FACTOR + AGG_FACTOR / 2.0
    cols = cc * AGG_FACTOR + AGG_FACTOR / 2.0
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="ul")
    return pd.DataFrame({"lon": np.asarray(xs), "lat": np.asarray(ys), "pop": block_pop[rr, cc]})


def apply_known_demand_node_repairs(nodes: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    nodes = nodes.copy()
    if "node_id" not in nodes.columns:
        return nodes, 0
    if "node_repaired" not in nodes.columns:
        nodes["node_repaired"] = False
    if "original_lon" not in nodes.columns:
        nodes["original_lon"] = nodes["lon"]
    if "original_lat" not in nodes.columns:
        nodes["original_lat"] = nodes["lat"]
    if "repair_distance_km" not in nodes.columns:
        nodes["repair_distance_km"] = np.nan
    if "repair_reason" not in nodes.columns:
        nodes["repair_reason"] = ""

    n_repaired = 0
    for node_id, repair in KNOWN_DEMAND_NODE_REPAIRS.items():
        mask = nodes["node_id"].astype(str) == node_id
        if not mask.any():
            continue
        was_repaired = nodes.loc[mask, "node_repaired"].fillna(False).astype(bool)
        if not was_repaired.any():
            nodes.loc[mask, "original_lon"] = nodes.loc[mask, "lon"]
            nodes.loc[mask, "original_lat"] = nodes.loc[mask, "lat"]
        changed = (
            (nodes.loc[mask, "lon"].astype(float) != float(repair["lon"]))
            | (nodes.loc[mask, "lat"].astype(float) != float(repair["lat"]))
            | (~was_repaired)
        )
        nodes.loc[mask, "lon"] = float(repair["lon"])
        nodes.loc[mask, "lat"] = float(repair["lat"])
        nodes.loc[mask, "node_repaired"] = True
        nodes.loc[mask, "repair_distance_km"] = float(repair["repair_distance_km"])
        nodes.loc[mask, "repair_reason"] = str(repair["repair_reason"])
        n_repaired += int(changed.any())
    return nodes, n_repaired


def create_demand_nodes(force: bool = False) -> pd.DataFrame:
    if OUT_NODES.exists() and not force:
        nodes = pd.read_csv(OUT_NODES, dtype={"krs_code_str": str})
        nodes["krs_code_str"] = nodes["krs_code_str"].astype(str).str.zfill(5)
        nodes, n_repaired = apply_known_demand_node_repairs(nodes)
        if n_repaired:
            nodes.to_csv(OUT_NODES, index=False)
            print(f"Applied {n_repaired} routability repair(s) to cached demand nodes.")
        print(f"Loaded cached demand nodes: {len(nodes):,} from {OUT_NODES}")
        return nodes

    counties_df = pd.read_csv(COUNTY_STR, dtype={"krs_code_str": str})
    counties_df["krs_code_str"] = counties_df["krs_code_str"].astype(str).str.zfill(5)
    county_lookup = counties_df.set_index("krs_code_str")
    county_geoms = load_county_geometries(counties_df)

    node_rows = []
    with rasterio.open(POP_RASTER) as src:
        for i, row in enumerate(county_geoms, start=1):
            code = row["krs_code_str"]
            cells = aggregate_population_cells(src, row["geometry"])
            if cells.empty:
                rp = row["geometry"].representative_point()
                total_pop = float(county_lookup.loc[code, "total_pop"])
                node_rows.append(
                    {
                        "node_id": f"{code}_0",
                        "krs_code_str": code,
                        "krs_name_str": county_lookup.loc[code, "krs_name_str"],
                        "lon": rp.x,
                        "lat": rp.y,
                        "node_pop": total_pop,
                        "node_pop_frac_county": 1.0,
                        "county_raster_pop": 0.0,
                        "county_table_pop": total_pop,
                        "k_selected": 1,
                        "pop90_radius_km": np.nan,
                        "fallback_no_raster_pop": True,
                    }
                )
                continue

            lon = cells["lon"].to_numpy(dtype=float)
            lat = cells["lat"].to_numpy(dtype=float)
            weights = cells["pop"].to_numpy(dtype=float)
            xy = xy_km(lon, lat)
            first = weighted_medoid_idx(xy, weights)
            d_first = np.linalg.norm(xy - xy[first], axis=1)
            radius90 = weighted_quantile(d_first, weights, 0.90)
            if radius90 <= 10:
                k = 1
            elif radius90 <= 25:
                k = 3
            else:
                k = 5
            medoids, assignment = weighted_k_medoids(xy, weights, k)
            county_raster_pop = float(weights.sum())
            county_table_pop = float(county_lookup.loc[code, "total_pop"])
            for j, medoid_idx in enumerate(medoids):
                cluster_mask = assignment == j
                cluster_pop_raster = float(weights[cluster_mask].sum())
                node_pop = county_table_pop * cluster_pop_raster / county_raster_pop
                node_rows.append(
                    {
                        "node_id": f"{code}_{j}",
                        "krs_code_str": code,
                        "krs_name_str": county_lookup.loc[code, "krs_name_str"],
                        "lon": float(lon[medoid_idx]),
                        "lat": float(lat[medoid_idx]),
                        "node_pop": node_pop,
                        "node_pop_frac_county": node_pop / county_table_pop if county_table_pop else np.nan,
                        "county_raster_pop": county_raster_pop,
                        "county_table_pop": county_table_pop,
                        "k_selected": k,
                        "pop90_radius_km": radius90,
                        "fallback_no_raster_pop": False,
                    }
                )
            if i % 25 == 0:
                print(f"Demand nodes: processed {i}/{len(county_geoms)} counties; nodes={len(node_rows):,}")

    nodes = pd.DataFrame(node_rows)
    nodes = nodes.sort_values(["krs_code_str", "node_id"]).reset_index(drop=True)
    nodes, n_repaired = apply_known_demand_node_repairs(nodes)
    if n_repaired:
        print(f"Applied {n_repaired} routability repair(s) to demand nodes.")
    nodes.to_csv(OUT_NODES, index=False)
    print(f"Wrote demand nodes: {len(nodes):,} to {OUT_NODES}")
    return nodes


def load_hospitals(path: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["latitude", "longitude", "total_cases"]).copy()
    df = df.reset_index(drop=True)
    df["facility_id"] = [f"{prefix}_{i:04d}" for i in range(len(df))]
    return df


def wait_for_ors(url: str, timeout_s: int = 180) -> None:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/v2/health", timeout=5)
            last = r.text
            if r.ok and "ready" in r.text:
                return
        except requests.RequestException as exc:
            last = str(exc)
        time.sleep(2)
    raise RuntimeError(f"ORS did not become ready within {timeout_s}s. Last response: {last}")


def post_matrix(url: str, source_locs: list[list[float]], dest_locs: list[list[float]], retries: int = 5) -> tuple[np.ndarray, np.ndarray]:
    n_sources = len(source_locs)
    n_dest = len(dest_locs)
    body = {
        "locations": source_locs + dest_locs,
        "sources": list(range(n_sources)),
        "destinations": list(range(n_sources, n_sources + n_dest)),
        "metrics": ["duration", "distance"],
        "units": "km",
    }
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(f"{url}/v2/matrix/driving-car", json=body, timeout=300)
            if r.ok:
                data = r.json()
                durations = np.array(data.get("durations"), dtype=np.float64) / 60.0
                distances = np.array(data.get("distances"), dtype=np.float64)
                return durations, distances
            last_error = f"HTTP {r.status_code}: {r.text[:500]}"
        except Exception as exc:  # requests JSON errors included for retry visibility.
            last_error = repr(exc)
        if attempt < retries:
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"ORS matrix request failed after {retries} attempts: {last_error}")


def summarize_model(counties: pd.DataFrame, model: str, therapy: str, str_col: str, access_col: str, access_unit: str) -> list[dict]:
    rows = []
    for group, sub in [("All counties", counties), ("Urban counties", counties[counties["urban_rural"] == "urban"]), ("Rural counties", counties[counties["urban_rural"] == "rural"] )]:
        vals = sub[str_col].dropna()
        access = sub.loc[vals.index, access_col].dropna()
        rows.append(
            {
                "Model": model,
                "Therapy": therapy,
                "Group": group,
                "N_counties": int(vals.size),
                "Mean_STR_unweighted": round(float(vals.mean()), 2),
                "Median_STR": round(float(vals.median()), 2),
                "IQR_STR": f"{vals.quantile(0.25):.2f}-{vals.quantile(0.75):.2f}",
                "Counties_STR_lt_0_5": int((vals < 0.5).sum()),
                "Pct_counties_STR_lt_0_5": round(float((vals < 0.5).mean() * 100), 1),
                "Mean_nearest_access": round(float(access.mean()), 1) if len(access) else np.nan,
                "Access_unit": access_unit,
            }
        )
    return rows
