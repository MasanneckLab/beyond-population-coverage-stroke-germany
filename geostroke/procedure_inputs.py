"""Source-checked procedure counts for the cross-sectional 2024 analysis.

Report filenames are the join key. A count refresh must never re-sort geocodes
or implicitly change a facility's location, linkage fields, or cohort membership.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

THERAPIES = {"lysis": "8-020.8", "thrombectomy": "8-836.80"}
RAW_NAMES = {"lysis": "lysis_hospitals_8-020-8.csv",
             "thrombectomy": "thrombectomy_hospitals_8-836-80.csv"}
MANIFEST = "procedure_source_manifest_2024.json"


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        digest = sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(folder: Path) -> dict[str, str]:
    files = sorted(folder.glob("*-2024-xml.xml"))
    if not files:
        raise ValueError(f"No 2024 source XMLs in {folder}")
    return {p.name: file_hash(p) for p in files}


def numeric_count(element, tag: str, path: Path) -> int | None:
    value = (element.findtext(tag) or "").strip()
    if not value:
        # Suppression is missing information, not evidence of zero activity.
        return None
    if not value.isdigit():
        raise ValueError(f"Invalid numeric {tag}={value!r} in {path.name}")
    return int(value)


def scan_sources(folder: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    rows, hashes = [], {}
    for path in sorted(folder.glob("*-2024-xml.xml")):
        blob = path.read_bytes()
        hashes[path.name] = sha256(blob).hexdigest()
        root = ET.fromstring(blob)  # Fail rather than silently skip malformed reports.
        row = {"file_name": path.name, "sha256": hashes[path.name], "i63": 0}
        for name, prefix in THERAPIES.items():
            details, suppressed = [], 0
            for proc in root.findall(".//Prozedur"):
                code = (proc.findtext("OPS_301") or "").strip()
                if code.startswith(prefix):
                    value = numeric_count(proc, "Anzahl", path)
                    suppressed += value is None
                    details.append({"ops_code": code, "case_count": value or 0})
            row[name] = sum(p["case_count"] for p in details)
            row[f"{name}_suppressed_cells"] = suppressed
            row[f"{name}_details"] = details
        for diagnosis in root.findall(".//Hauptdiagnose"):
            if (diagnosis.findtext("ICD_10") or "").strip().startswith("I63"):
                row["i63"] += numeric_count(diagnosis, "Fallzahl", path) or 0
        rows.append(row)
    if not rows:
        raise ValueError(f"No 2024 source XMLs in {folder}")
    return pd.DataFrame(rows), hashes


def plan_input_refresh(folder: Path, sources: pd.DataFrame):
    """Return proposed CSVs and deltas; write nothing until every cohort passes."""
    source = sources.set_index("file_name", verify_integrity=True)
    outputs, deltas = {}, []
    for therapy in THERAPIES:
        raw_name = RAW_NAMES[therapy]
        old = pd.read_csv(folder / raw_name)
        if not old.file_name.is_unique:
            raise ValueError(f"Duplicate source filenames in {raw_name}")
        required = set(source.index[source[therapy].gt(0)])
        if set(old.file_name) != required:
            raise ValueError(f"Reporting-site membership changed for {therapy}; "
                             "review site intake/removal and geocoding before a count-only refresh.")
        new = old.copy()
        new["total_cases"] = new.file_name.map(source[therapy]).astype(int)
        new["procedure_details"] = new.file_name.map(source[f"{therapy}_details"]).map(str)
        new["procedures_found"] = new.file_name.map(source[f"{therapy}_details"]).map(len)
        for i in old.index[old.total_cases.ne(new.total_cases)]:
            deltas.append({"therapy": therapy, "file_name": old.loc[i, "file_name"],
                           "old": int(old.loc[i, "total_cases"]),
                           "new": int(new.loc[i, "total_cases"])})
        outputs[raw_name] = new
        keyed = new.set_index("file_name")
        for suffix, threshold in [("10plus", 10), ("centers", 25)]:
            name = f"{therapy}_{suffix}_geocoded.csv"
            geo = pd.read_csv(folder / name)
            expected = set(new.loc[new.total_cases.ge(threshold), "file_name"])
            if not geo.file_name.is_unique or set(geo.file_name) != expected:
                raise ValueError(f"{name}: >= {threshold} cohort changed; rebuild and bind spatial caches first.")
            result = geo.copy()
            for col in ("total_cases", "procedure_details", "procedures_found"):
                result[col] = result.file_name.map(keyed[col])
            outputs[name] = result
    return outputs, deltas


def preserve_cached_order(current: pd.DataFrame, cached: pd.DataFrame) -> pd.DataFrame:
    if not current.file_name.is_unique or not cached.file_name.is_unique:
        raise ValueError("Cannot align duplicate report filenames")
    if set(current.file_name) != set(cached.file_name):
        raise ValueError("Facility membership changed: rebuild spatial caches, do not reuse by row index")
    return current.set_index("file_name").loc[cached.file_name].reset_index()


def reuse_verified_geocodes(current: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Keep an unchanged geocode snapshot byte-for-byte during legacy reruns."""
    cached = pd.read_csv(path)
    aligned = preserve_cached_order(current, cached)
    for column in ["total_cases", "I63_total", "TOTAL", "match_method"]:
        if column in aligned and column in cached:
            try:
                pd.testing.assert_series_equal(aligned[column], cached[column],
                    check_dtype=False, check_exact=True)
            except AssertionError as exc:
                raise ValueError(f"{path.name}: changed {column}; review linkage/count inputs before reusing geocodes") from exc
    return cached


def assert_current_inputs(folder: Path, xml_folder: Path | None = None) -> dict:
    path = folder / MANIFEST
    instruction = "Run python -m scripts.reproduce_2024 counts after reviewing the local input cohorts"
    if not path.exists():
        raise RuntimeError(f"Missing procedure source manifest. {instruction}")
    manifest = json.loads(path.read_text())
    if xml_folder is not None and source_hashes(xml_folder) != manifest["sources"]:
        raise RuntimeError(f"2024 source XMLs changed; procedure caches are stale. {instruction}")
    for name, expected in manifest["inputs"].items():
        if file_hash(folder / name) != expected:
            raise RuntimeError(f"Unverified procedure cache: {name}. {instruction}")
    return manifest


def threshold_polygon_indices(folder: Path, raw_data: Path, therapy: str, threshold: int):
    path = folder / "threshold_cache_binding_2024.json"
    if not path.exists():
        raise RuntimeError("Missing polygon-volume binding; run the cross-sectional count refresh")
    binding = json.loads(path.read_text())[therapy]
    for name, digest in binding["hashes"].items():
        if file_hash(raw_data / name) != digest:
            raise ValueError("Isochrone cache changed; revalidate the polygon-volume binding")
    volumes = np.asarray(binding["poly_volumes"], dtype=int)
    cohort = pd.read_csv(folder / f"{therapy}_centers_geocoded.csv")
    if sorted(volumes) != sorted(cohort.total_cases):
        raise ValueError("Polygon volumes are stale relative to the hospital cohort")
    return np.flatnonzero(volumes >= threshold).tolist()


def validate_sparse_pairs(nodes, hospitals, pairs, cutoff_km: float, require_complete=True) -> dict:
    """Verify every legacy pair against identities, geometry and candidate coverage.

This validates alignment, not the historical ORS graph/version (not recorded in
the legacy cache). Counts deliberately do not enter this route-cache contract.
"""
    ni = pd.to_numeric(pairs.node_idx, errors="raise").to_numpy()
    hi = pd.to_numeric(pairs.facility_idx, errors="raise").to_numpy()
    if (not np.isfinite(ni).all() or not np.isfinite(hi).all()
            or (ni != np.floor(ni)).any() or (hi != np.floor(hi)).any()
            or (ni < 0).any() or (ni >= len(nodes)).any()
            or (hi < 0).any() or (hi >= len(hospitals)).any()):
        raise ValueError("Invalid cached route indices")
    ni, hi = ni.astype(int), hi.astype(int)
    if pairs.duplicated(["node_idx", "facility_idx"]).any():
        raise ValueError("Duplicate cached route pairs")
    if not np.array_equal(pairs.node_id.astype(str), nodes.node_id.astype(str).to_numpy()[ni]):
        raise ValueError("Cached demand-node identity/order mismatch")
    if not np.array_equal(pairs.facility_id.astype(str), hospitals.facility_id.astype(str).to_numpy()[hi]):
        raise ValueError("Cached facility identity/order mismatch")
    if not np.array_equal(pairs.krs_code_str.astype(str).str.zfill(5),
                          nodes.krs_code_str.astype(str).str.zfill(5).to_numpy()[ni]):
        raise ValueError("Cached county identity mismatch")
    lat1, lon1 = np.radians(nodes.lat.to_numpy())[:, None], np.radians(nodes.lon.to_numpy())[:, None]
    lat2, lon2 = np.radians(hospitals.latitude.to_numpy())[None, :], np.radians(hospitals.longitude.to_numpy())[None, :]
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    distances = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    if not np.allclose(pairs.haversine_km, distances[ni, hi], rtol=1e-10, atol=1e-8):
        raise ValueError("Cached route coordinates/order mismatch")
    expected = distances <= cutoff_km
    for i in np.flatnonzero(~expected.any(axis=1)):
        expected[i, np.argsort(distances[i])[:min(5, len(hospitals))]] = True
    observed = np.zeros_like(expected)
    observed[ni, hi] = True
    complete = np.array_equal(expected, observed)
    if (observed & ~expected).any() or (require_complete and not complete):
        raise ValueError("Incomplete or unexpected cached candidate pairs")
    for col in ["duration_min", "distance_km"]:
        vals = pairs[col].to_numpy(dtype=float)
        if np.isinf(vals).any() or (vals[np.isfinite(vals)] < 0).any():
            raise ValueError(f"Invalid cached {col}")
    if complete and pairs.duration_min.notna().groupby(pairs.facility_idx).sum().reindex(
            range(len(hospitals)), fill_value=0).eq(0).any():
        raise ValueError("At least one facility has no routable demand node")
    return {"pairs": len(pairs), "nodes": len(nodes), "facilities": len(hospitals),
            "null_durations": int(pairs.duration_min.isna().sum()),
            "alignment_verified": True, "complete": bool(complete),
            "historical_ors_graph_provenance": "not recorded in legacy cache"}
