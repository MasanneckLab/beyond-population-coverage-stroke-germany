"""Isochrone management – compute once, reuse forever.

`ensure_polygons(df, suffix='')` will return a dictionary of travel-time
bands (15/30/45/60) to lists of Shapely polygons **aligned with the
DataFrame order**.  If pickles already exist they are loaded instantly;
otherwise OpenRouteService is queried and the responses cached to
`raw_data/poly{minutes}{suffix}.pkl`.

The function is purposely side-effecting (writes cache) so that
subsequent runs are fast and deterministic.  Use ``force_recalc=True``
if you need fresh polygons.
"""

from __future__ import annotations

import pickle
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from shapely.geometry import Polygon

import openrouteservice as ors
import requests
import json

from . import config

__all__ = ["ensure_polygons", "ensure_polygons_parallel", "ensure_polygons_for_scenarios", "test_ors_connectivity", "batch_generate_isochrones"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLIENT: ors.Client | None = None


def _client() -> ors.Client:
    global CLIENT
    if CLIENT is None:
        CLIENT = ors.Client(base_url=config.ORS_BASE_URL, timeout=config.ORS_TIMEOUT)
    return CLIENT


def _pickle_path(minutes: int, suffix: str = "") -> Path:
    return config.DATA_DIR / f"poly{minutes}{suffix}.pkl"


def _load_pickle(path: Path) -> List[Polygon]:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _save_pickle(path: Path, data: List[Polygon]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _compute_isochrones(lon: float, lat: float, max_retries: int = 3, custom_model: dict | None = None, speed_factor: float = 1.0) -> List[Polygon]:
    """Return list[Polygon] for all TIME_BINS with retry logic.
    
    Parameters
    ----------
    lon : float
        Longitude coordinate
    lat : float  
        Latitude coordinate
    max_retries : int, optional
        Maximum retry attempts for failed requests
    custom_model : dict, optional
        Custom model parameters for ORS (e.g., speed modifications)
    speed_factor : float, optional
        Speed multiplier for scenario simulation (1.0 = normal, 1.2 = emergency, 0.8 = bad traffic)
    """
    # Adjust time ranges based on speed factor
    # If speed_factor > 1.0 (faster), we request longer time ranges for larger areas
    # If speed_factor < 1.0 (slower), we request shorter time ranges for smaller areas
    # Question: "What area can I reach in X minutes at this speed?"
    adjusted_time_bins = [int(t * speed_factor) for t in config.TIME_BINS]
    ranges_seconds = [t * 60 for t in adjusted_time_bins]
    
    # ORS has a maximum of 10 isochrones per request
    MAX_ISOCHRONES_PER_REQUEST = 10
    all_polys: list[Polygon] = []
    
    # Split into chunks if we have more than 10 time bins
    for i in range(0, len(ranges_seconds), MAX_ISOCHRONES_PER_REQUEST):
        chunk_ranges = ranges_seconds[i:i + MAX_ISOCHRONES_PER_REQUEST]
        
        body = {
            "locations": [[lon, lat]],
            "range_type": "time",
            "range": chunk_ranges,  # Use adjusted ranges for speed simulation
            "attributes": ["area", "reachfactor", "total_pop"],
        }
        
        # Try custom model first if provided and if we're using normal speed factor
        try_custom_model = custom_model and speed_factor == 1.0
        if try_custom_model:
            body["options"] = custom_model
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{config.ORS_BASE_URL}/v2/isochrones/{config.ORS_PROFILE}",
                    json=body,
                    timeout=config.ORS_TIMEOUT,
                )
                response.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                # If custom model fails with 400 error, try without it using speed factor simulation
                if (try_custom_model and response.status_code == 400 and attempt == 0):
                    print(f"⚠️  Custom model not supported by ORS server, using speed factor simulation")
                    body.pop("options", None)  # Remove custom model
                    try_custom_model = False
                    # Continue with the retry (don't increment attempt)
                    continue
                elif attempt == max_retries - 1:
                    speed_desc = f" (speed factor: {speed_factor})" if speed_factor != 1.0 else ""
                    error_msg = f"❌ Failed to get isochrones after {max_retries} attempts{speed_desc}: {e}"
                    print(error_msg)
                    raise Exception(error_msg)
                else:
                    print(f"⚠️  Attempt {attempt + 1} failed, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    speed_desc = f" (speed factor: {speed_factor})" if speed_factor != 1.0 else ""
                    error_msg = f"❌ Failed to get isochrones after {max_retries} attempts{speed_desc}: {e}"
                    print(error_msg)
                    raise Exception(error_msg)
                print(f"⚠️  Attempt {attempt + 1} failed, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff

        try:
            data = response.json()
            
            if "features" not in data:
                error_msg = f"❌ No 'features' in ORS response for chunk {i//MAX_ISOCHRONES_PER_REQUEST + 1}"
                print(error_msg)
                raise Exception(error_msg)
                
            # Convert each feature to a Shapely polygon
            for feature in data["features"]:
                if "geometry" not in feature:
                    continue
                    
                geom = feature["geometry"]
                if geom["type"] == "Polygon":
                    # Handle single polygon
                    coords = geom["coordinates"][0]  # exterior ring
                    if len(coords) >= 4:  # Valid polygon needs at least 4 points
                        poly = Polygon(coords)
                        if poly.is_valid:
                            all_polys.append(poly)
                elif geom["type"] == "MultiPolygon":
                    # Handle multipolygon - take the largest polygon
                    largest_poly = None
                    largest_area = 0
                    
                    for poly_coords in geom["coordinates"]:
                        exterior_ring = poly_coords[0]
                        if len(exterior_ring) >= 4:
                            poly = Polygon(exterior_ring)
                            if poly.is_valid and poly.area > largest_area:
                                largest_area = poly.area
                                largest_poly = poly
                    
                    if largest_poly:
                        all_polys.append(largest_poly)
                        
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            error_msg = f"❌ Error parsing ORS response for chunk {i//MAX_ISOCHRONES_PER_REQUEST + 1}: {e}"
            print(error_msg)
            raise Exception(error_msg)
    
    if len(all_polys) != len(config.TIME_BINS):
        error_msg = f"⚠️  Expected {len(config.TIME_BINS)} polygons, got {len(all_polys)}"
        print(error_msg)
        print(f"🔄 Attempting individual time bin recovery for missing polygons...")
        
        # Try to recover missing polygons by making individual requests
        original_count = len(all_polys)
        missing_count = len(config.TIME_BINS) - original_count
        
        # For facilities that fail with batch requests, try individual time bin requests
        # This is slower but more reliable for edge cases
        recovered_polys = _recover_missing_polygons(lon, lat, len(all_polys), speed_factor, max_retries)
        all_polys.extend(recovered_polys)
        
        if len(all_polys) == len(config.TIME_BINS):
            print(f"✅ Successfully recovered {len(recovered_polys)} missing polygons via individual requests")
        else:
            # Still incomplete - pad with empty polygons and raise exception
            while len(all_polys) < len(config.TIME_BINS):
                all_polys.append(Polygon())
            raise Exception(f"Incomplete polygon set: {error_msg}")
    
    return all_polys


def _recover_missing_polygons(lon: float, lat: float, current_count: int, speed_factor: float = 1.0, max_retries: int = 3) -> List[Polygon]:
    """Attempt to recover missing polygons by making individual time bin requests.
    
    This function is called when batch requests fail to return the expected number of polygons.
    It tries to get the missing polygons by making individual API calls for each missing time bin.
    
    Parameters
    ----------
    lon : float
        Longitude coordinate
    lat : float
        Latitude coordinate  
    current_count : int
        Number of polygons already successfully retrieved
    speed_factor : float, optional
        Speed multiplier for scenario simulation
    max_retries : int, optional
        Maximum retry attempts for failed requests
    
    Returns
    -------
    List[Polygon]
        List of recovered polygons (may include empty polygons for failed attempts)
    """
    recovered_polys = []
    missing_time_bins = config.TIME_BINS[current_count:]  # Get the time bins we're missing
    
    print(f"   Attempting to recover {len(missing_time_bins)} missing time bins: {missing_time_bins}")
    
    for time_bin in missing_time_bins:
        try:
            # Calculate adjusted time for this specific bin
            adjusted_time = int(time_bin * speed_factor)
            time_seconds = adjusted_time * 60
            
            body = {
                "locations": [[lon, lat]],
                "range_type": "time", 
                "range": [time_seconds],  # Single time bin request
                "attributes": ["area", "reachfactor", "total_pop"],
            }
            
            print(f"   Recovering {time_bin}min ({time_seconds}s): ", end="", flush=True)
            
            success = False
            response = None
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        f"{config.ORS_BASE_URL}/v2/isochrones/{config.ORS_PROFILE}",
                        json=body,
                        timeout=180,  # Even longer timeout for individual requests
                    )
                    response.raise_for_status()
                    success = True
                    break
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        print(f"❌ Failed: {str(e)[:50]}")
                        recovered_polys.append(Polygon())  # Add empty polygon
                        break
                    else:
                        print(f"⚠️", end="", flush=True)
                        time.sleep(1)  # Brief pause between retries
            
            if success and response:
                # Successful request - process response
                try:
                    data = response.json()
                    if "features" in data and len(data["features"]) > 0:
                        feature = data["features"][0]
                        if "geometry" in feature and feature["geometry"]:
                            geom = feature["geometry"]
                            if geom["type"] == "Polygon":
                                coords = geom["coordinates"][0]
                                if len(coords) >= 4:
                                    poly = Polygon(coords)
                                    if poly.is_valid:
                                        recovered_polys.append(poly)
                                        print(f"✅ Success")
                                        continue
                            elif geom["type"] == "MultiPolygon":
                                # Handle multipolygon - take the largest
                                largest_poly = None
                                largest_area = 0
                                for poly_coords in geom["coordinates"]:
                                    exterior_ring = poly_coords[0]
                                    if len(exterior_ring) >= 4:
                                        poly = Polygon(exterior_ring)
                                        if poly.is_valid and poly.area > largest_area:
                                            largest_area = poly.area
                                            largest_poly = poly
                                if largest_poly:
                                    recovered_polys.append(largest_poly)
                                    print(f"✅ Success (MultiPolygon)")
                                    continue
                    
                    # If we reach here, the response was invalid
                    print(f"❌ Invalid response")
                    recovered_polys.append(Polygon())
                    
                except (KeyError, ValueError, json.JSONDecodeError) as e:
                    print(f"❌ Parse error: {str(e)[:30]}")
                    recovered_polys.append(Polygon())
                    
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
            recovered_polys.append(Polygon())
    
    return recovered_polys


def test_ors_connectivity() -> bool:
    """Test if OpenRouteService is accessible and working."""
    try:
        # Test with Berlin coordinates using first few time bins
        test_coords = [13.3888, 52.5170]
        test_ranges = [t * 60 for t in config.TIME_BINS[:3]]  # Test first 3 time bins
        test_params = {
            'profile': config.ORS_PROFILE,
            'range': test_ranges,
            'locations': [test_coords],
        }
        
        client = _client()
        result = client.isochrones(**test_params)  # type: ignore[attr-defined]
        
        return (result and 'features' in result and len(result['features']) > 0)
        
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_polygons(
    df: pd.DataFrame,
    *,
    force_recalc: bool = False,
    suffix: str = "",
    max_retries: int = 3,
    progress_interval: int = 10,
    scenario: str = "normal",
) -> Dict[int, List[Polygon]]:
    """Ensure pickled isochrone polygons exist and return them.

    Parameters
    ----------
    df : DataFrame
        Must have ``longitude`` and ``latitude`` columns.
    force_recalc : bool, optional
        If ``True``, ignore any existing pickles and re-query ORS.
    suffix : str, optional
        Added between *minutes* and ``.pkl`` so you can keep multiple
        polygon sets (e.g. ``_all_CTs``). If empty and scenario is not 'normal',
        the scenario suffix will be used automatically.
    max_retries : int, optional
        Maximum retry attempts for failed requests.
    progress_interval : int, optional
        Print progress every N facilities.
    scenario : str, optional
        Scenario to use for isochrone generation. Options: 'normal', 'emergency', 'bad_traffic'.
        Defaults to 'normal' for backward compatibility.
    """

    missing_cols = {"longitude", "latitude"}.difference(df.columns)
    if missing_cols:
        raise ValueError(f"DataFrame missing columns: {missing_cols}")

    # Validate scenario
    if scenario not in config.SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Available scenarios: {list(config.SCENARIOS.keys())}")
    
    scenario_config = config.SCENARIOS[scenario]
    
    # Determine suffix to use - combine scenario suffix with provided suffix
    if scenario != "normal":
        scenario_suffix = scenario_config["suffix"]
        if suffix:
            actual_suffix = f"{suffix}{scenario_suffix}"  # e.g., "_all_CTs_emergency"
        else:
            actual_suffix = scenario_suffix  # e.g., "_emergency"
    else:
        actual_suffix = suffix  # Normal scenario uses provided suffix as-is
    
    print(f"🔄 Processing scenario: {scenario_config['description']}")
    if actual_suffix:
        print(f"   Cache suffix: {actual_suffix}")

    # Try to read all pickle files first ------------------------------------------------
    polygons: Dict[int, List[Polygon]] = {}
    if not force_recalc:
        try:
            for t in config.TIME_BINS:
                polygons[t] = _load_pickle(_pickle_path(t, actual_suffix))
            print(f"✅ Loaded cached polygons{actual_suffix}: {len(polygons[config.TIME_BINS[0]])} facilities")
            return polygons  # cache hit
        except FileNotFoundError:
            pass  # at least one file missing – fall through to computation

    # Compute for each point (could be hundreds → keep user informed) ------------------------
    print(f"Computing isochrones{actual_suffix} – this may take a while…")
    start_time = time.time()
    progress = 0
    failed_facilities = []
    per_band: Dict[int, List[Polygon]] = {t: [] for t in config.TIME_BINS}

    for idx, row in df.iterrows():
        try:
            lon, lat = float(row["longitude"]), float(row["latitude"])
            # Pass both custom model and speed factor from scenario configuration
            polys = _compute_isochrones(
                lon, lat, 
                max_retries=max_retries, 
                custom_model=scenario_config.get("custom_model"),
                speed_factor=scenario_config.get("speed_factor", 1.0)
            )
            for minutes, poly in zip(config.TIME_BINS, polys, strict=False):
                per_band[minutes].append(poly)
            
        except Exception as e:
            # Record failed facility but continue processing
            facility_name = row.get('name', f'Index {idx}')
            failed_facilities.append((idx, facility_name, str(e)))
            print(f"❌ Failed to process {facility_name}: {e}")
            
            # Add empty polygons to maintain alignment
            for minutes in config.TIME_BINS:
                per_band[minutes].append(Polygon())  # Empty polygon
        
        progress += 1
        if progress % progress_interval == 0:
            elapsed = time.time() - start_time
            rate = progress / elapsed
            remaining = len(df) - progress
            eta = remaining / rate if rate > 0 else 0
            print(f"  {progress}/{len(df)} locations processed… "
                  f"({rate:.1f}/min, ETA: {eta/60:.1f} min)")

    # Final timing
    total_time = time.time() - start_time
    print(f"✅ Computation complete in {total_time:.1f} seconds")
    
    if failed_facilities:
        print(f"⚠️ {len(failed_facilities)} facilities failed:")
        for idx, name, error in failed_facilities[:5]:  # Show first 5
            print(f"   {name}: {error}")
        if len(failed_facilities) > 5:
            print(f"   ... and {len(failed_facilities) - 5} more")

    # Save ------------------------------------------------------------------
    print("💾 Saving polygon cache files...")
    for t, plist in per_band.items():
        _save_pickle(_pickle_path(t, actual_suffix), plist)

    print("✓ Isochrones ready and cached.")
    return per_band


# ---------------------------------------------------------------------------
# Parallel processing helpers
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Thread-safe progress tracker for parallel processing."""
    
    def __init__(self, total: int, interval: int = 10):
        self.total = total
        self.interval = interval
        self.completed = 0
        self.failed = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.failed_facilities = []
    
    def update(self, success: bool = True, facility_info: tuple | None = None):
        with self.lock:
            if success:
                self.completed += 1
            else:
                self.failed += 1
                if facility_info:
                    self.failed_facilities.append(facility_info)
            
            total_processed = self.completed + self.failed
            if total_processed % self.interval == 0 or total_processed == self.total:
                elapsed = time.time() - self.start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                remaining = self.total - total_processed
                eta = remaining / rate if rate > 0 else 0
                
                print(f"  {total_processed}/{self.total} facilities processed… "
                      f"({rate:.1f}/min, ETA: {eta/60:.1f} min)")
                if self.failed > 0:
                    print(f"     ⚠️ {self.failed} failures so far")
    
    def get_failed_facilities(self):
        with self.lock:
            return self.failed_facilities.copy()


def _compute_isochrones_with_index(args: tuple) -> tuple:
    """Wrapper for _compute_isochrones that includes the DataFrame index."""
    idx, row, max_retries, custom_model, speed_factor = args
    
    try:
        lon, lat = float(row["longitude"]), float(row["latitude"])
        polys = _compute_isochrones(lon, lat, max_retries=max_retries, custom_model=custom_model, speed_factor=speed_factor)
        return idx, polys, None
    except Exception as e:
        facility_name = row.get('name', f'Index {idx}')
        # Add facility coordinates to error message for debugging
        lon, lat = float(row["longitude"]), float(row["latitude"])
        enhanced_error = f"{str(e)} [Facility: {facility_name}, Coords: ({lon:.6f}, {lat:.6f})]"
        return idx, None, (idx, facility_name, enhanced_error)


def ensure_polygons_parallel(
    df: pd.DataFrame,
    *,
    force_recalc: bool = False,
    suffix: str = "",
    max_retries: int = 3,
    progress_interval: int = 10,
    scenario: str = "normal",
    max_workers: int | None = None,
) -> Dict[int, List[Polygon]]:
    """Parallelized version of ensure_polygons for faster isochrone generation.

    This version uses ThreadPoolExecutor to make concurrent API requests,
    significantly speeding up generation on multi-core systems.

    Parameters
    ----------
    df : DataFrame
        Must have ``longitude`` and ``latitude`` columns.
    force_recalc : bool, optional
        If ``True``, ignore any existing pickles and re-query ORS.
    suffix : str, optional
        Added between *minutes* and ``.pkl`` so you can keep multiple
        polygon sets (e.g. ``_all_CTs``). If empty and scenario is not 'normal',
        the scenario suffix will be used automatically.
    max_retries : int, optional
        Maximum retry attempts for failed requests.
    progress_interval : int, optional
        Print progress every N facilities.
    scenario : str, optional
        Scenario to use for isochrone generation. Options: 'normal', 'emergency', 'bad_traffic'.
        Defaults to 'normal' for backward compatibility.
    max_workers : int, optional
        Maximum number of concurrent threads. If None, defaults to min(8, core_count).
        For local ORS servers, you can increase this. For public APIs, keep it lower.
    """

    missing_cols = {"longitude", "latitude"}.difference(df.columns)
    if missing_cols:
        raise ValueError(f"DataFrame missing columns: {missing_cols}")

    # Validate scenario
    if scenario not in config.SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Available scenarios: {list(config.SCENARIOS.keys())}")
    
    scenario_config = config.SCENARIOS[scenario]
    
    # Determine suffix to use - combine scenario suffix with provided suffix
    if scenario != "normal":
        scenario_suffix = scenario_config["suffix"]
        if suffix:
            actual_suffix = f"{suffix}{scenario_suffix}"  # e.g., "_all_CTs_emergency"
        else:
            actual_suffix = scenario_suffix  # e.g., "_emergency"
    else:
        actual_suffix = suffix  # Normal scenario uses provided suffix as-is
    
    print(f"🔄 Processing scenario: {scenario_config['description']}")
    if actual_suffix:
        print(f"   Cache suffix: {actual_suffix}")

    # Auto-determine max_workers based on system and server type
    if max_workers is None:
        import os
        cpu_count = os.cpu_count() or 4
        # Conservative default: use up to 8 workers for public APIs, more for local
        if 'localhost' in config.ORS_BASE_URL or '127.0.0.1' in config.ORS_BASE_URL:
            max_workers = min(cpu_count, 12)  # More aggressive for local ORS
        else:
            max_workers = min(cpu_count, 8)   # Conservative for public APIs
    
    print(f"🚀 Using {max_workers} parallel workers for isochrone generation")

    # Try to read all pickle files first ------------------------------------------------
    polygons: Dict[int, List[Polygon]] = {}
    if not force_recalc:
        try:
            for t in config.TIME_BINS:
                polygons[t] = _load_pickle(_pickle_path(t, actual_suffix))
            print(f"✅ Loaded cached polygons{actual_suffix}: {len(polygons[config.TIME_BINS[0]])} facilities")
            return polygons  # cache hit
        except FileNotFoundError:
            pass  # at least one file missing – fall through to computation

    # Parallel computation ----------------------------------------------------------------
    print(f"Computing isochrones{actual_suffix} with {max_workers} workers – this should be much faster!")
    
    # Initialize progress tracker
    tracker = ProgressTracker(len(df), progress_interval)
    
    # Prepare per-time-bin polygon lists
    per_band: Dict[int, List[Polygon]] = {t: [Polygon()] * len(df) for t in config.TIME_BINS}
    
    # Prepare arguments for parallel processing
    args_list = [
        (idx, row, max_retries, scenario_config.get("custom_model"), scenario_config.get("speed_factor", 1.0))
        for idx, row in df.iterrows()
    ]
    
    # Execute in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(_compute_isochrones_with_index, args): args[0] 
            for args in args_list
        }
        
        # Process completed tasks
        for future in as_completed(future_to_idx):
            idx, polys, error_info = future.result()
            
            if error_info:
                # Handle failed facility
                tracker.update(success=False, facility_info=error_info)
                print(f"\n🚨 FACILITY FAILED: {error_info[1]}")
                print(f"   Error: {error_info[2]}")
                print(f"   This will create a gap in coverage for this facility\n")
                
                # Add empty polygons to maintain alignment
                for minutes in config.TIME_BINS:
                    per_band[minutes][idx] = Polygon()  # Empty polygon
            else:
                # Handle successful facility
                tracker.update(success=True)
                
                # Assign polygons to time bands
                for minutes, poly in zip(config.TIME_BINS, polys, strict=False):
                    per_band[minutes][idx] = poly

    # Final statistics
    elapsed = time.time() - tracker.start_time
    print(f"✅ Parallel computation complete in {elapsed:.1f} seconds")
    print(f"   Speedup: ~{max_workers:.1f}x theoretical (actual depends on ORS server capacity)")
    
    failed_facilities = tracker.get_failed_facilities()
    if failed_facilities:
        print(f"\n🚨 CRITICAL: {len(failed_facilities)} facilities failed!")
        print("=" * 60)
        for idx, name, error in failed_facilities:  # Show all failed facilities
            print(f"❌ {name}: {error}")
        if len(failed_facilities) > 0:
            print("=" * 60)
            print(f"💡 Tip: Check your ORS server status and network connectivity")
            print(f"   These failures will result in missing coverage areas in your analysis")

    # Save ------------------------------------------------------------------
    print("💾 Saving polygon cache files...")
    for t, plist in per_band.items():
        _save_pickle(_pickle_path(t, actual_suffix), plist)

    print("✓ Isochrones ready and cached.")
    return per_band


def ensure_polygons_for_scenarios(
    df: pd.DataFrame,
    scenarios: list[str] | None = None,
    *,
    force_recalc: bool = False,
    suffix: str = "",
    max_retries: int = 3,
    progress_interval: int = 10,
) -> Dict[str, Dict[int, List[Polygon]]]:
    """Generate isochrones for multiple scenarios.
    
    Parameters
    ----------
    df : DataFrame
        Must have ``longitude`` and ``latitude`` columns.
    scenarios : list[str], optional
        List of scenarios to generate. If None, generates all available scenarios.
    force_recalc : bool, optional
        If ``True``, ignore any existing pickles and re-query ORS.
    suffix : str, optional
        Additional suffix for cache files (added after scenario suffix).
    max_retries : int, optional
        Maximum retry attempts for failed requests.
        
    Returns
    -------
    dict
        Mapping of scenario names to polygon dictionaries.
    """
    
    if scenarios is None:
        scenarios = list(config.SCENARIOS.keys())
    
    # Validate all scenarios first
    for scenario in scenarios:
        if scenario not in config.SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'. Available scenarios: {list(config.SCENARIOS.keys())}")
    
    if not test_ors_connectivity():
        print("❌ OpenRouteService not available - cannot generate new isochrones")
        return {}
    
    results = {}
    
    for scenario in scenarios:
        print(f"\n📊 Processing scenario: {scenario}")
        try:
            polygons = ensure_polygons(
                df,
                force_recalc=force_recalc,
                suffix=suffix,
                max_retries=max_retries,
                progress_interval=progress_interval,
                scenario=scenario
            )
            results[scenario] = polygons
            print(f"✅ Scenario '{scenario}' complete")
        except Exception as e:
            print(f"❌ Scenario '{scenario}' failed: {e}")
            results[scenario] = {}
    
    return results 


def batch_generate_isochrones(
    datasets: Dict[str, tuple[pd.DataFrame, str]],
    force_recalc: bool = False,
    max_retries: int = 3
) -> Dict[str, Dict[int, List[Polygon]]]:
    """Generate isochrones for multiple datasets in batch.
    
    Parameters
    ----------
    datasets : dict
        Mapping of dataset names to (DataFrame, suffix) tuples.
    force_recalc : bool, optional
        If True, regenerate even if cached.
    max_retries : int, optional
        Maximum retry attempts for failed requests.
        
    Returns
    -------
    dict
        Mapping of dataset names to polygon dictionaries.
    """
    
    if not test_ors_connectivity():
        print("❌ OpenRouteService not available - cannot generate new isochrones")
        return {}
    
    results = {}
    
    for name, (df, suffix) in datasets.items():
        print(f"\n📊 Processing {name}...")
        try:
            polygons = ensure_polygons(
                df, 
                force_recalc=force_recalc, 
                suffix=suffix,
                max_retries=max_retries
            )
            results[name] = polygons
            print(f"✅ {name} complete")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results[name] = {}
    
    return results 