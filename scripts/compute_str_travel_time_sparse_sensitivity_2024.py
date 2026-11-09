"""Sparse ORS allocation and sensitivities; invoke via reproduce_2024."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from geostroke.procedure_inputs import validate_sparse_pairs
from scripts.compute_str_travel_time_sensitivity_2024 import (
    CACHE_DIR, BETAS, ZERO_SECONDS_EPS, post_matrix, wait_for_ors)

IVT_HAVERSINE_PREFILTER_KM = 180.0
MT_HAVERSINE_PREFILTER_KM = 300.0
DEST_CHUNK = 25
PROGRESS_EVERY_NODES = 10

def haversine_matrix(nodes: pd.DataFrame, hospitals: pd.DataFrame) -> np.ndarray:
    r = 6371.0
    lat1 = np.radians(nodes["lat"].to_numpy(dtype=float))[:, None]
    lon1 = np.radians(nodes["lon"].to_numpy(dtype=float))[:, None]
    lat2 = np.radians(hospitals["latitude"].to_numpy(dtype=float))[None, :]
    lon2 = np.radians(hospitals["longitude"].to_numpy(dtype=float))[None, :]
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))


def sparse_path(label: str) -> Path:
    return CACHE_DIR / f"{label}_sparse_pairs_2024.csv"


def meta_path(label: str) -> Path:
    return CACHE_DIR / f"{label}_sparse_pairs_2024.json"


def load_completed_nodes(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        return set(pd.read_csv(path, usecols=["node_id"])["node_id"].astype(str).unique())
    except Exception:
        return set()


def validate_completed_nodes(
    path: Path,
    nodes: pd.DataFrame,
    hdist: np.ndarray,
    haversine_cutoff_km: float,
    n_hospitals: int,
) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    existing = pd.read_csv(path, dtype={"node_id": str})
    if existing.empty:
        return set()
    expected_counts: dict[str, int] = {}
    for i, node in nodes.iterrows():
        n_expected = int((hdist[i] <= haversine_cutoff_km).sum())
        if n_expected == 0:
            n_expected = min(5, n_hospitals)
        expected_counts[str(node["node_id"])] = n_expected
    observed = existing.groupby("node_id").size().to_dict()
    complete = {
        node_id
        for node_id, expected in expected_counts.items()
        if int(observed.get(node_id, 0)) >= expected
    }
    incomplete_present = set(observed) - complete
    if incomplete_present:
        cleaned = existing[existing["node_id"].isin(complete)].copy()
        cleaned.to_csv(path, index=False)
        print(
            f"Removed {len(incomplete_present)} incomplete checkpoint node(s) from {path.name}; "
            "they will be recomputed.",
            flush=True,
        )
    return complete


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    if path.exists() and path.stat().st_size > 0:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        missing = [col for col in df.columns if col not in header]
        if missing:
            existing = pd.read_csv(path, dtype={"node_id": str, "krs_code_str": str, "facility_id": str})
            for col in missing:
                existing[col] = "ok" if col == "ors_status" else np.nan
            existing.to_csv(path, index=False)
    header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="a", header=header, index=False)


def pair_row(
    node: pd.Series,
    hosp: pd.Series,
    facility_idx: int,
    hdist_km: float,
    duration_min: float = np.nan,
    distance_km: float = np.nan,
    ors_status: str = "ok",
) -> dict:
    return {
        "node_id": node["node_id"],
        "node_idx": int(node.name),
        "krs_code_str": str(node["krs_code_str"]).zfill(5),
        "facility_id": hosp["facility_id"],
        "facility_idx": int(facility_idx),
        "duration_min": duration_min,
        "distance_km": distance_km,
        "haversine_km": float(hdist_km),
        "ors_status": ors_status,
    }


def request_node_pairs(
    ors_url: str,
    node: pd.Series,
    hospitals: pd.DataFrame,
    hospital_indices: np.ndarray,
    hdist_row: np.ndarray,
) -> list[dict]:
    rows: list[dict] = []
    source = [[float(node["lon"]), float(node["lat"])]]
    for start in range(0, len(hospital_indices), DEST_CHUNK):
        idx = hospital_indices[start : start + DEST_CHUNK]
        dest_locs = hospitals.iloc[idx][["longitude", "latitude"]].to_numpy(dtype=float).tolist()
        try:
            dur, dist = post_matrix(ors_url, source, dest_locs, retries=4)
        except RuntimeError:
            # Split once to isolate occasional ORS search failures without losing the node.
            for single in idx:
                dest_one = hospitals.iloc[[single]][["longitude", "latitude"]].to_numpy(dtype=float).tolist()
                hosp = hospitals.iloc[int(single)]
                try:
                    dur_one, dist_one = post_matrix(ors_url, source, dest_one, retries=1)
                    rows.append(
                        pair_row(
                            node,
                            hosp,
                            int(single),
                            hdist_row[int(single)],
                            float(dur_one[0, 0]) if np.isfinite(dur_one[0, 0]) else np.nan,
                            float(dist_one[0, 0]) if np.isfinite(dist_one[0, 0]) else np.nan,
                        )
                    )
                except RuntimeError as exc:
                    print(
                        f"WARNING: ORS failed for node {node['node_id']} -> facility {hosp['facility_id']}; "
                        "recording pair as unreachable. "
                        f"{str(exc)[:240]}",
                        flush=True,
                    )
                    rows.append(pair_row(node, hosp, int(single), hdist_row[int(single)], ors_status="ors_matrix_failed"))
            continue
        for offset, hosp_idx in enumerate(idx):
            hosp = hospitals.iloc[int(hosp_idx)]
            rows.append(
                pair_row(
                    node,
                    hosp,
                    int(hosp_idx),
                    hdist_row[int(hosp_idx)],
                    float(dur[0, offset]) if np.isfinite(dur[0, offset]) else np.nan,
                    float(dist[0, offset]) if np.isfinite(dist[0, offset]) else np.nan,
                )
            )
    return rows


def compute_or_load_sparse_pairs(
    nodes: pd.DataFrame,
    hospitals: pd.DataFrame,
    label: str,
    haversine_cutoff_km: float,
    force: bool,
    ors_url: str,
) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = sparse_path(label)
    meta = meta_path(label)
    if force and out.exists():
        out.unlink()
    if out.exists() and out.stat().st_size:
        cached = pd.read_csv(out, dtype={"krs_code_str": str, "node_id": str, "facility_id": str})
        check = validate_sparse_pairs(nodes, hospitals, cached, haversine_cutoff_km, require_complete=False)
        if check["complete"]:
            print(f"{label}: reused {len(cached):,} validated cached pairs (no ORS request).", flush=True)
            return cached
    wait_for_ors(ors_url)
    hdist = haversine_matrix(nodes, hospitals)
    total_candidate_pairs = int((hdist <= haversine_cutoff_km).sum())
    completed = validate_completed_nodes(out, nodes, hdist, haversine_cutoff_km, len(hospitals))
    print(
        f"{label}: haversine prefilter <= {haversine_cutoff_km:.0f} km yields "
        f"{total_candidate_pairs:,} candidate pairs across {len(nodes):,} demand nodes.",
        flush=True,
    )
    t0 = time.time()
    for i, node in nodes.iterrows():
        node_id = str(node["node_id"])
        if node_id in completed:
            continue
        candidate_idx = np.where(hdist[i] <= haversine_cutoff_km)[0]
        if candidate_idx.size == 0:
            candidate_idx = np.argsort(hdist[i])[: min(5, len(hospitals))]
        rows = request_node_pairs(ors_url, node, hospitals, candidate_idx, hdist[i])
        append_rows(out, rows)
        completed.add(node_id)
        if (len(completed) % PROGRESS_EVERY_NODES == 0) or len(completed) == len(nodes):
            elapsed = time.time() - t0
            rate = len(completed) / elapsed if elapsed > 0 else float("nan")
            remaining = (len(nodes) - len(completed)) / rate if rate and np.isfinite(rate) else float("nan")
            print(
                f"{label}: completed {len(completed):,}/{len(nodes):,} nodes; "
                f"elapsed={elapsed/3600:.2f}h; eta={remaining/3600:.2f}h",
                flush=True,
            )
            meta.write_text(
                json.dumps(
                    {
                        "label": label,
                        "haversine_cutoff_km": haversine_cutoff_km,
                        "completed_nodes": len(completed),
                        "n_nodes": len(nodes),
                        "n_hospitals": len(hospitals),
                        "total_candidate_pairs": total_candidate_pairs,
                        "elapsed_seconds_current_session": elapsed,
                        "eta_seconds_current_rate": remaining,
                        "ors_url": ors_url,
                    },
                    indent=2,
                )
            )
    pairs = pd.read_csv(out, dtype={"krs_code_str": str, "node_id": str, "facility_id": str})
    pairs["krs_code_str"] = pairs["krs_code_str"].str.zfill(5)
    validate_sparse_pairs(nodes, hospitals, pairs, haversine_cutoff_km)
    return pairs


def gravity_alloc_sparse(
    hospitals: pd.DataFrame,
    nodes: pd.DataFrame,
    pairs: pd.DataFrame,
    beta: float,
    max_time_min: float | None,
) -> np.ndarray:
    alloc = np.zeros(len(nodes), dtype=np.float64)
    node_pop = nodes["node_pop"].to_numpy(dtype=np.float64)
    grouped = {int(k): v for k, v in pairs.dropna(subset=["duration_min"]).groupby("facility_idx")}
    for h_idx, hosp in hospitals.iterrows():
        p = grouped.get(int(h_idx))
        if p is None or p.empty:
            continue
        t = p["duration_min"].to_numpy(dtype=np.float64)
        node_idx = p["node_idx"].to_numpy(dtype=int)
        valid = np.isfinite(t)
        zero = valid & ((t * 60.0) == 0.0)
        if np.any(zero):
            t[zero] = ZERO_SECONDS_EPS / 60.0
        if max_time_min is not None:
            valid &= t <= max_time_min
        if not np.any(valid):
            valid[np.nanargmin(t)] = True
        use_nodes = node_idx[valid]
        use_t = t[valid]
        weights = node_pop[use_nodes] / np.power(use_t, beta)
        s = weights.sum()
        if s <= 0 or not np.isfinite(s):
            continue
        alloc[use_nodes] += float(hosp["total_cases"]) * weights / s
    return alloc


def aggregate_alloc(nodes: pd.DataFrame, alloc: np.ndarray) -> pd.Series:
    tmp = nodes[["krs_code_str"]].copy()
    tmp["alloc"] = alloc
    return tmp.groupby("krs_code_str")["alloc"].sum()


def add_sparse_str_columns(
    counties: pd.DataFrame,
    nodes: pd.DataFrame,
    hospitals: pd.DataFrame,
    pairs: pd.DataFrame,
    therapy: str,
    beta: float,
    variants: dict[str, float | None],
) -> None:
    nearest = pairs.groupby("node_idx")["duration_min"].min()
    node_nearest = nodes[["krs_code_str"]].copy()
    node_nearest["nearest"] = node_nearest.index.map(nearest)
    counties[f"nearest_{therapy}_time_min"] = counties["krs_code_str"].map(node_nearest.groupby("krs_code_str")["nearest"].min())
    exp_col = f"exp_{therapy}"
    for suffix, max_time in variants.items():
        alloc_nodes = gravity_alloc_sparse(hospitals, nodes, pairs, beta=beta, max_time_min=max_time)
        alloc_county = aggregate_alloc(nodes, alloc_nodes)
        alloc_col = f"alloc_{therapy}_time_{suffix}"
        str_col = f"STR_{therapy}_time_{suffix}"
        counties[alloc_col] = counties["krs_code_str"].map(alloc_county).fillna(0.0)
        counties[str_col] = np.where(counties[exp_col] > 0, counties[alloc_col] / counties[exp_col], np.nan)


def compare_models(counties: pd.DataFrame, therapy: str) -> dict:
    old_col = f"STR_{therapy}"
    new_col = f"STR_{therapy}_time_primary"
    old_under = counties[old_col] < 0.5
    new_under = counties[new_col] < 0.5
    urban = counties["urban_rural"] == "urban"
    rural = counties["urban_rural"] == "rural"
    corr = stats.spearmanr(counties[old_col], counties[new_col], nan_policy="omit")
    return {
        "Therapy": therapy,
        "N_counties": int(counties[new_col].notna().sum()),
        "Current_km_underserved_n": int(old_under.sum()),
        "Travel_time_underserved_n": int(new_under.sum()),
        "Net_change_underserved_n": int(new_under.sum() - old_under.sum()),
        "Switched_to_underserved_n": int((~old_under & new_under).sum()),
        "Switched_from_underserved_n": int((old_under & ~new_under).sum()),
        "Spearman_current_vs_time_STR_rho": round(float(corr.statistic), 3),
        "Spearman_current_vs_time_STR_p": float(corr.pvalue),
        "Current_urban_minus_rural_mean_STR": round(float(counties.loc[urban, old_col].mean() - counties.loc[rural, old_col].mean()), 2),
        "Travel_time_urban_minus_rural_mean_STR": round(float(counties.loc[urban, new_col].mean() - counties.loc[rural, new_col].mean()), 2),
        "Current_all_mean_STR": round(float(counties[old_col].mean()), 2),
        "Travel_time_all_mean_STR": round(float(counties[new_col].mean()), 2),
    }


def beta_sensitivity_sparse(counties, nodes, hospitals, pairs, therapy, max_time):
    rows = []
    exp_col = f"exp_{therapy}"
    for beta in BETAS:
        alloc_nodes = gravity_alloc_sparse(hospitals, nodes, pairs, beta=beta, max_time_min=max_time)
        alloc_county = aggregate_alloc(nodes, alloc_nodes)
        vals = counties["krs_code_str"].map(alloc_county).fillna(0.0).to_numpy() / counties[exp_col].to_numpy()
        urban = counties["urban_rural"].to_numpy() == "urban"
        rural = counties["urban_rural"].to_numpy() == "rural"
        rows.append(
            {
                "Therapy": therapy,
                "Beta": beta,
                "Max_time_min": max_time,
                "Mean_STR_all": round(float(np.nanmean(vals)), 2),
                "Mean_STR_urban": round(float(np.nanmean(vals[urban])), 2),
                "Mean_STR_rural": round(float(np.nanmean(vals[rural])), 2),
                "Pct_underserved_rural": round(float(np.nanmean(vals[rural] < 0.5) * 100), 1),
            }
        )
    return rows
