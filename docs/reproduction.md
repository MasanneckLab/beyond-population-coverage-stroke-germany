# Reproduction Contract

## Inputs And Provenance

Supply the following files locally. Do not commit them. Do not download or open
untrusted pickle caches: Python pickle can execute code.

| Location | Required input |
| --- | --- |
| `data/private/xml_2024/` | Frozen 2024 quality reports, named `*-2024-xml.xml` |
| `data/private/` | Reviewed IVT/MT raw extracts, >=10 and >=25 geocoded cohorts |
| `data/private/` | Certified SU roster, extended stroke-ready roster, deduplicated >=100 selected-diagnosis cohort |
| `data/private/cache/` | Local demand nodes, isochrones, their bindings and `travel_time_cache_2024/` sparse pairs, if reusing caches |
| `data/external/` | Destatis `12411-02-03-4.csv`, semicolon/Latin-1 export, six header lines, reference date 31 December 2024 |
| `data/external/` | `germany-detailed-boundary_917.geojson`, `germany-states.geojson`, `georef-germany-kreis@public.geojson` |
| `data/external/` | `GHS_POP_E2025_GLOBE_R2023A_4326_3ss_V1_0.tif` and the GHS-SMOD 2025 raster |

Exact CSV filenames/columns are in `data/templates/`. Coordinates use WGS84;
`file_name` is the report-level key, not a hospital-group name or IK alone.
The supplied aggregate `county_metadata.csv` freezes the analysis classification
of 400 counties (139 urban, 261 rural) and geographic names. That centroid-based
classification is distinct from the pixel-level SMOD classification in Table 2.

Obtain reports through the G-BA reference database, age populations through
Destatis Regionalstatistik, population/settlement rasters through the European
Commission GHSL, and the certified roster through DSG/DGN, subject to the source
terms. National counts depend on the exact report-file vintage, not just the year.
Administrative-boundary snapshots also affect raster/centroid calculations.

The public workflow starts from reviewed hospital geocodes and cohort membership.
It does **not** claim to recreate the authors' manual site adjudication from
public code alone. `geostroke/site_contacts.py` contains the reporting-site-first
XML address parser. A replacement input set requires evidence-backed site review,
including report-location versus group headquarters, campus linkage, duplicate
certified sites, and pooled/shared procedure reporting. Do not geocode an entire
group to its headquarters or infer patient deduplication from matching addresses.

The additional stroke-ready criterion is **>100** combined OPS 8-981/8-98b cases,
excluding certified-site overlap. The broad comparator is **>=100** selected
diagnoses (I60-I64 and G45 excluding G45.4). The fourth coverage scenario is this
ICD cohort **alone**, not its cumulative union with the previous scenarios.

## Execution

Run commands from the repository root with the local environment activated.

```bash
python -m scripts.reproduce_2024 check-inputs
python -m scripts.reproduce_2024 counts
python -m scripts.reproduce_2024 age
python -m scripts.reproduce_2024 nodes
# Only needed when the reviewed isochrone caches are unavailable:
python -m scripts.reproduce_2024 isochrones
python -m scripts.reproduce_2024 routes
python -m scripts.reproduce_2024 models
python -m scripts.reproduce_2024 coverage
python -m scripts.reproduce_2024 tables
python -m scripts.reproduce_2024 figures
```

`counts` fails on changed cohort membership/counts instead of silently changing
clinical sites. It stores per-source hashes only in the local private directory.
Suppressed cells are not assigned invented values; observed numeric totals omit
them. The count stage reports the IVT/MT number of suppressed cells separately.

The age model fixes case shares at 3%, 20%, 30% and 47% for ages <45, 45-64,
65-74 and >=75. It divides the 2024 cases in each band by the 2024 population;
expected county counts sum to the 2024 national I63 total before rasterization.
Rasterized need uses county-specific per-capita rates on the GHS population
surface, so its national denominator is not forced to equal the county total.

Demand uses one, three or five population-weighted medoids per county. The fixed
Uckermark repair is a population node, not a hospital coordinate. Two island
nodes remain unroutable; their county need is not dropped. County nearest access
is the minimum among routable node-to-hospital pairs.

The primary IVT/MT gravity models use beta 2.0/1.5 and 90/120-minute cutoffs.
Sensitivity models use beta 1.0-3.0 in 0.5 steps, 60/90-minute cutoffs, and no
time cutoff within the finite 180/300-km candidate envelopes. The last variant
is not an unrestricted nationwide all-pairs model. A hospital without an eligible
node within the time cutoff allocates to its nearest routable candidate. Exact
zero-second routes use one second for numerical stability. All model variants
must conserve the included national procedure totals. The km comparator uses
county centroids, beta 2.0/1.5, 100/150-km radii and a 1-km distance floor.

Sparse cache reuse validates node/hospital identity, row order, coordinates,
candidate completeness, duplicate pairs, finite nonnegative durations/distances,
and a routable node for every facility. This does not independently establish
the original ORS graph version. The submitted correction reused unchanged routes
after exact checks on 10 control routes per therapy and rerouted changed sites;
full historical graph provenance was not recorded in the original caches.
Use `routes --force` to recompute routes, but expect differences if the graph or
snapping settings change. Full cache regeneration is not a substitute for input
curation. Save the graph, OSM timestamp, engine status and request configuration.

Isochrone generation uses the original 5-60-minute request sequence, including
its 10-range batching. Its semantics must not be changed while comparing to the
submission. The original certified-unit polygons were retained in the corrected
snapshot; legacy bindings have weaker provenance than newly generated bindings.
Newly generated caches record cohort hashes, coordinates and polygon hashes.

Override paths using `GEOSTROKE_PRIVATE_DATA`, `GEOSTROKE_EXTERNAL_DATA`,
`GEOSTROKE_CACHE`, `GEOSTROKE_ANALYSIS_DATA`, `GEOSTROKE_FIGURES`, and
`GEOSTROKE_POP_RASTER`. `GEOSTROKE_ORS_URL` selects the local routing service.
Keep output directories separate from private source inputs and reference results.

## Outputs And Checks

| Submission item | Output |
| --- | --- |
| Main Tables 1-3 | `outputs/tables/Table1.csv` through `Table3.csv` |
| Supplementary Tables 1-6 | `outputs/tables/TableS1.csv` through `TableS6.csv` |
| Main Figures 1-3 | `outputs/figures/Fig1` through `Fig3` (PDF, EPS, 600-dpi RGB TIFF) |
| Supplementary Figures 1-6 | `outputs/figures/Supplementary/FigS1` through `FigS6` (PDF, 600-dpi RGB TIFF) |

Table 1 exports underlying summary quantities in long format; the manuscript
presentation is transposed and display-rounded. Table S5 is the **ORS beta
sensitivity**; Table S6 is the **km-based STR comparator**. Figure S4 retains the
km-based beta sensitivity. Figure 1 includes cumulative driving-time coverage
as panel e below the four maps. No extra historical Figure 2 is exported.

Synthetic tests require no hospital data or ORS server. Reference aggregate CSVs
permit numerical comparisons, not reconstruction of excluded hospital inputs.
The release verification records which computations were actually rerun. Matching
software and input snapshots is necessary for exact reproduction; changes in
road networks, raster grids, source reports or fonts must be reported rather
than silently accepted as the same analysis.
