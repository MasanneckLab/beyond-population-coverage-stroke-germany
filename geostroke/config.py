"""Package-wide configuration and constants.

The module centralises path handling so that other sub-modules don't need
hard-coded absolute paths.  All filesystem locations can be overridden
via environment variables, making the package portable and CI-friendly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Directory layout (overridable via env vars)
# ---------------------------------------------------------------------------

ROOT: Final[Path] = Path(os.getenv("GEOSTROKE_ROOT", ".")).resolve()

DATA_ROOT: Final[Path] = Path(os.getenv("GEOSTROKE_DATA_ROOT", ROOT / "data"))
DATA_DIR: Final[Path] = Path(os.getenv("GEOSTROKE_PRIVATE_DATA", DATA_ROOT / "private"))
AGGREGATE_DIR: Final[Path] = Path(os.getenv("GEOSTROKE_AGGREGATE_DATA", DATA_ROOT / "aggregate"))
ANALYSIS_DIR: Final[Path] = Path(os.getenv("GEOSTROKE_ANALYSIS_DATA", ROOT / "outputs" / "tables"))
CACHE_DIR: Final[Path] = Path(os.getenv("GEOSTROKE_CACHE", DATA_DIR / "cache"))
EXTERNAL_DIR: Final[Path] = Path(os.getenv("GEOSTROKE_EXTERNAL_DATA", DATA_ROOT / "external"))
GRAPH_DIR: Final[Path] = Path(os.getenv("GEOSTROKE_FIGURES", ROOT / "outputs" / "figures"))
SHAPE_DIR: Final[Path] = Path(os.getenv("GEOSTROKE_SHAPE", EXTERNAL_DIR))
RESULTS_DIR: Final[Path] = Path(os.getenv("GEOSTROKE_RESULTS", ROOT / "outputs"))

# Make sure common output dirs exist  ------------------------------------------------
for _p in (GRAPH_DIR, RESULTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data files (relative to the above directories)
# ---------------------------------------------------------------------------

POP_RASTER: Final[Path] = Path(
    os.getenv(
        "GEOSTROKE_POP_RASTER",
        EXTERNAL_DIR
        / "GHS_POP_E2025_GLOBE_R2023A_4326_3ss_V1_0.tif",
    )
)
GERMANY_OUTLINE: Final[Path] = SHAPE_DIR / "germany-detailed-boundary_917.geojson"
GERMANY_STATES: Final[Path] = SHAPE_DIR / "germany-states.geojson"
GERMANY_COUNTIES: Final[Path] = SHAPE_DIR / "georef-germany-kreis@public.geojson"

# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------

import matplotlib.pyplot as _plt

# All available time bins for isochrone generation
TIME_BINS: Final[list[int]] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]  # minutes

# Default time bins for backward compatibility (all available time bins)
DEFAULT_TIME_BINS: Final[list[int]] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]  # minutes

DISPLAY_TIME_BINS: Final[list[int]] = [15, 30, 45, 60]  # minutes

CMAP = _plt.get_cmap("RdYlBu_r")  # Original notebook colormap

# Create colors ensuring backward compatibility for default time bins
TIME_COLOURS: Final[dict[int, tuple[float, float, float, float]]] = {}

# Use the original notebook's exact formula: CMAP(0.2 + 0.8 * i/(len(TIME_BINS)-1))
for i, t in enumerate(sorted(TIME_BINS)):
    TIME_COLOURS[t] = CMAP(0.2 + 0.8 * i / (len(TIME_BINS) - 1))

# County population baseline; raster coverage uses its own measured denominator.
POP_TOTAL: Final[int] = 83_577_140  # Destatis, 31 December 2024; not the raster denominator.

# ---------------------------------------------------------------------------
# ORS client settings
# ---------------------------------------------------------------------------

ORS_BASE_URL: Final[str] = os.getenv("GEOSTROKE_ORS_URL", "http://localhost:8080/ors")
ORS_TIMEOUT: Final[int] = int(os.getenv("GEOSTROKE_ORS_TIMEOUT", "5000"))
ORS_PROFILE: Final[str] = os.getenv("GEOSTROKE_ORS_PROFILE", "driving-car")

ISO_RANGE: Final[list[int]] = [3600]  # seconds (outer radius)
ISO_INTERVAL: Final[int] = 900        # seconds

# ---------------------------------------------------------------------------
# Scenario configurations for isochrone generation
# ---------------------------------------------------------------------------

# Define scenarios with their speed adjustments
# Since ORS custom models aren't widely supported, we simulate speed changes
# by adjusting the time ranges sent to ORS
SCENARIOS: Final[dict[str, dict]] = {
    "normal": {
        "suffix": "",
        "description": "Normal driving conditions",
        "speed_factor": 1.0,  # No speed adjustment
        "custom_model": None  # Keep for backward compatibility
    },
    "emergency": {
        "suffix": "_emergency", 
        "description": "Emergency vehicle conditions (+20% speed)",
        "speed_factor": 1.20,  # 20% faster = equivalent areas with 20% less time
        "custom_model": {
            "speed": [{"if": True, "multiply_by": 1.20}]
        }
    },
    "bad_traffic": {
        "suffix": "_bad_traffic",
        "description": "Bad traffic conditions (0.8× speed)", 
        "speed_factor": 0.80,  # 20% slower = equivalent areas with 25% more time
        "custom_model": {
            "speed": [{"if": True, "multiply_by": 0.80}]
        }
    }
}

# Backward compatibility: default scenario
DEFAULT_SCENARIO: Final[str] = "normal"

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def resolve(path: str | Path) -> Path:
    """Return *path* as absolute :class:`~pathlib.Path`.  Accepts str/Path."""

    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def get_all_time_bins() -> list[int]:
    """Return all available time bins for comprehensive analysis.
    
    Returns
    -------
    list[int]
        All time bins: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        
    Examples
    --------
    # Use all time bins for detailed analysis
    gs.run_publication_figures(time_bins=gs.config.get_all_time_bins())
    """
    return list(TIME_BINS)


def get_display_time_bins() -> list[int]:
    """Return display time bins for publication figures.
    
    Returns
    -------
    list[int]
        Display time bins: [15, 30, 45, 60]
        
    Examples
    --------
    # Use display time bins for cleaner publication figures
    gs.run_publication_figures(time_bins=gs.config.get_display_time_bins())
    """
    return list(DISPLAY_TIME_BINS)


def debug_dump() -> None:
    """Print current config to stdout – handy for troubleshooting."""

    import pprint as _pp

    _pp.pprint({
        "ROOT": ROOT,
        "DATA_ROOT": DATA_ROOT,
        "DATA_DIR": DATA_DIR,
        "AGGREGATE_DIR": AGGREGATE_DIR,
        "EXTERNAL_DIR": EXTERNAL_DIR,
        "GRAPH_DIR": GRAPH_DIR,
        "SHAPE_DIR": SHAPE_DIR,
        "RESULTS_DIR": RESULTS_DIR,
        "POP_RASTER": POP_RASTER,
        "ORS_BASE_URL": ORS_BASE_URL,
        "TIME_BINS": TIME_BINS,
        "DEFAULT_TIME_BINS": DEFAULT_TIME_BINS,
        "DISPLAY_TIME_BINS": DISPLAY_TIME_BINS,
    })
