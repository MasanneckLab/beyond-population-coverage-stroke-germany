# Release Verification: 9 November 2026

The public workflow was executed in the existing scientific Python environment
against the reviewed local 2024 inputs, with outputs in a separate test directory.
The private analysis inputs and the submission documents were not overwritten.

- 26 offline synthetic regression tests passed, including year and duplicate-cell
  checks, source-hash invalidation, routing identity/coordinate validation, mass
  conservation, mutually exclusive bins, request direction/units and data exclusion.
- Re-extraction of all 2,220 XML reports reproduced the submitted national counts:
  241,251 I63 admissions, 47,220 IVT procedures and 26,965 MT procedures. The
  stroke/TIA denominator and stroke-care procedure summaries also matched Table 1.
- Age calibration, all primary and comparator STR variants, both beta sensitivity
  series, time bins and model-comparison tables matched the corrected reference
  outputs, using numerical tolerance `rtol=atol=1e-12` where appropriate.
- All 400 counties matched in the 27 reference columns, including coverage
  fractions. Main Table 2 matched exactly in every exported cell. State coverage,
  progressive scenarios, county-level coverage sensitivities, volume thresholds
  and cumulative driving-time curves matched the corrected reference outputs.
- All three main and six supplementary figures were regenerated. Their 600-dpi
  RGB TIFF pixels were **identical** to the approved JON submission artwork.
  Matching PDF/EPS exports were generated where specified in the output contract.
- The staged release was checked for unexpected data, hospital-level CSV/JSON
  fields, populated schema templates, hospital report identifiers, local absolute
  paths, possible credentials, binaries and submission documents.

## Boundaries Of This Check

This was a numerical and figure-reproduction check, not a new clinical validity
study. Reviewed isochrones and sparse routes were reused: 124,371 IVT and 121,420 MT
candidate pairs passed the identity/coordinate/completeness checks. No new full
national routing graph, geocoding review, or alternative hospital attribution
study was created for this release. Cold ORS requests were unit-tested with
synthetic responses, not reissued nationwide. See the graph-provenance and
model-interpretation limitations in the README and reproduction contract.

No fresh dependency installation on a second machine was tested. The recorded
environment is the environment in which the numerical and pixel comparisons
passed; other compatible versions/platforms should repeat these checks.
