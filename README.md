# Urban-rural disparities in access to acute stroke care in Germany: a nationwide geospatial analysis

Analysis code accompanying the manuscript prepared for the **Journal of Neurology**.
This is the corrected cross-sectional 2024 release, dated 9 November 2026, under the
MIT License. The repository URL is retained so that the manuscript's code link
continues to work. This release does not contain the separate longitudinal study.

## What Is Included

- Code for source-count validation, 2024 age calibration, population coverage,
  standardized treatment ratios (STRs), and the reported sensitivity analyses.
- Code for the three main figures, three main tables, six supplementary figures,
  and six supplementary tables, with the current submission numbering.
- County/state/national aggregate reference results and header-only input schemas.
- Synthetic regression tests and an explicit public-release data boundary.

**No hospital datasets are included:** no hospital rows, names, addresses,
coordinates, report identifiers, XML reports, individual procedure/diagnosis
counts, geocoding decisions, route matrices, or hospital-linked isochrone caches.
Manuscript files, the Supplement PDF, correspondence and author forms are also
excluded. The Supplement PDF remains in the authors' submission package only.

## Analysis Snapshot

The hospital reports and county age populations refer to **2024**. Other inputs
have their own dates: GHS-POP and GHS-SMOD use the **2025 epoch**, and the certified
stroke-unit registry was accessed on **22 April 2026**. The routing snapshot is
described below; these inputs must not all be labelled "2024 data".

The corrected source totals are **241,251 I63 admissions, 47,220 IVT procedures
and 26,965 MT procedures** across 2,220 report files. These are aggregate
procedure-to-admission ratios, not patient-linked treatment rates.

The primary STR analysis uses ORS driving times and centres with at least 10
annual procedures. The coverage analysis uses at least 25 procedures. The
Haversine-distance STR model is a comparator, not the primary model. Age rates
are recalibrated from the 2024 I63 total and 2024 age denominators at full
precision; rounded rates are for display only.

## Setup

Use a local Python environment on macOS or Linux. The submission was computed
with Python **3.13.3**; the observed library versions are recorded in
[`docs/environment_reference.json`](docs/environment_reference.json). This is an
environment record, not a claim that every platform supplies the same wheels.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The geospatial dependencies require GDAL/PROJ-compatible installations. Figure
exports additionally use Ghostscript (`gs` on `PATH`) for EPS. Arial was used in
the submission; the renderer falls back to Helvetica or DejaVu Sans if Arial is
not installed. A font substitution can change layout without changing results.

## Reproduction

Full reconstruction requires locally supplied, reviewed hospital cohorts and
external geographic inputs. **A public clone alone cannot reproduce the hospital
geography or the full analyses.** Input contracts, cache provenance checks,
commands, and limitations are in [`docs/reproduction.md`](docs/reproduction.md).
The completed numerical and figure checks are recorded in
[`docs/validation.md`](docs/validation.md).

```bash
python -m scripts.reproduce_2024 check-inputs
python -m scripts.reproduce_2024 all
```

The `all` command validates the source counts, recalibrates age-specific need,
generates or checks demand nodes and routes, recomputes the models and coverage,
and exports the tables and figures. New ORS calculations can be lengthy.
Generated files go to `outputs/tables/` and `outputs/figures/`; they are ignored
by Git. Approved aggregate reference results are in `data/aggregate/`, separate
from newly computed outputs. No command modifies manuscript or Supplement files.

For figure regeneration from the supplied aggregate reference results, the
reviewed hospital/cohort caches and boundary files are still required for maps:

```bash
GEOSTROKE_ANALYSIS_DATA=data/aggregate python -m scripts.reproduce_2024 figures
```

## OpenRouteService

The recorded service used **ORS 9.1.0**, profile **`driving-car`**, graph version
1, OSM timestamp **2025-04-15T20:21:30Z**, and graph build timestamp
**2025-07-20T13:27:55Z**. Merely using ORS 9.1.0 does not recreate that road graph.
Retain the original graph/configuration or use the reviewed local caches for
the closest reproduction. Legacy cached routes did not originally record full
graph provenance; identity/coordinate checks and unchanged-route controls do
not prove historical graph identity.

- Default endpoint: `http://localhost:8080/ors`; override with `GEOSTROKE_ORS_URL`.
- Required APIs: status/health, matrix and isochrones, with Germany coverage.
- Matrix requests travel from population demand nodes **to hospitals**. Durations
  are converted from seconds to minutes; distances are requested in kilometres.
- The sparse route client uses one origin and at most 25 destinations per batch.
  Isochrones cover 5 to 60 minutes in 5-minute increments, in batches of at most
  10 ranges. Do not substitute a different isolated-range request pattern.
- The recorded service allowed routes up to 1,000 km. Routing limits, available
  isochrone attributes and the supplied road graph must support these requests.

## Interpretation And Limits

STRs are modelled spatial allocations, not observed county treatment delivery.
Their association with driving time is partly structural because the allocation
model itself uses travel time. Hospital reports can pool campuses or shared
procedure tables; address validation does not resolve patient-level attribution.
The existing sensitivity analyses do not eliminate these limitations.

Time bins are mutually exclusive: IVT `<10`, `10 to <20`, `20 to <30`, `>=30`
minutes; MT `<15`, `15 to <30`, `30 to <45`, `>=45` minutes. The final open-ended
bin is not a cumulative threshold.

No interactive hospital map is published. The optional `docs/` page is static
documentation only.

## License And Citation

Code is MIT-licensed; third-party data retain their own licensing and access
conditions. Data exclusion here is not a statement that all underlying sources
are legally confidential.

Masanneck L, Pawlitzki M, Meuth SG, Schreiber S, Gliem M, Audebert HJ.
*Urban-rural disparities in access to acute stroke care in Germany: a nationwide
geospatial analysis.* Manuscript prepared for the Journal of Neurology. See
[`CITATION.cff`](CITATION.cff). No acceptance or publication is implied.
