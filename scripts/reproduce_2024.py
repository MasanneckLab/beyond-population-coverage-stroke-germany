"""Reproduce the cross-sectional submission from local, reviewed inputs.

Use `python -m scripts.reproduce_2024 --help` from the repository root.
Hospital inputs and all routing caches remain local. This never edits Word files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
from scipy import stats
from shapely.ops import unary_union

from geostroke import config, data, population
from geostroke.age_calibration import calibrate_2024
from geostroke.analysis_2024 import (prepare_hospitals, county_geometry, run_models,
    procedure_summary, county_fractions, legacy_county_outputs)
from geostroke.procedure_inputs import (MANIFEST, RAW_NAMES, file_hash, scan_sources,
    assert_current_inputs, validate_sparse_pairs)
from geostroke.str_models import haversine_km
from scripts import compute_str_travel_time_sensitivity_2024 as travel
from scripts import compute_str_travel_time_sparse_sensitivity_2024 as sparse

P, C, A, E = config.DATA_DIR, config.CACHE_DIR, config.ANALYSIS_DIR, config.EXTERNAL_DIR
THERAPIES = [('IVT', 'lysis', 'ivt', 180, 90), ('MT', 'thrombectomy', 'mt', 300, 120)]
COHORTS = {
    '': 'stroke_units_geocoded.csv',
    '_extended_stroke': 'stroke_units_extended_geocoded.csv',
    '_lysis_2024': 'lysis_centers_geocoded.csv',
    '_thrombectomy_2024': 'thrombectomy_centers_geocoded.csv',
    '_lysis10_2024': 'lysis_10plus_geocoded.csv',
    '_thrombectomy10_2024': 'thrombectomy_10plus_geocoded.csv',
    '_icd_hospitals_2024': 'icd_hospitals_100plus_geocoded_2024_dedup.csv',
}


def read(name):
    return pd.read_csv(A / name, dtype={'krs_code_str': str})


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def check_inputs():
    required = [P / name for name in set(COHORTS.values()) | set(RAW_NAMES.values())]
    required += [E / '12411-02-03-4.csv', config.POP_RASTER, config.GERMANY_COUNTIES,
        config.GERMANY_OUTLINE, config.GERMANY_STATES, config.AGGREGATE_DIR / 'county_metadata.csv']
    missing = [str(p) for p in required if not p.is_file()]
    if not list((P / 'xml_2024').glob('*-2024-xml.xml')):
        missing.append(str(P / 'xml_2024/*-2024-xml.xml'))
    if not list(E.glob('GHS_SMOD*.tif')):
        missing.append(str(E / 'GHS_SMOD*.tif'))
    if missing:
        raise ValueError('Missing local inputs (see docs/reproduction.md):\n' + '\n'.join(missing))
    print('Required local input files are present; this is not a scientific validation.')


def counts():
    sources, hashes = scan_sources(P / 'xml_2024')
    keyed = sources.set_index('file_name', verify_integrity=True)
    inputs, rows = {}, []
    i63 = int(sources.i63.sum())
    for therapy, name, prefix, envelope, cutoff in THERAPIES:
        frames = {}
        for label, filename, threshold in [('raw', RAW_NAMES[name], 1),
                ('ten', f'{name}_10plus_geocoded.csv', 10), ('main', f'{name}_centers_geocoded.csv', 25)]:
            frame = pd.read_csv(P / filename)
            expected = set(keyed.index[keyed[name].ge(threshold)])
            if not frame.file_name.is_unique or set(frame.file_name) != expected:
                raise ValueError(f'{filename}: site membership differs from the XML extraction; review the roster.')
            if not np.array_equal(frame.total_cases, frame.file_name.map(keyed[name])):
                raise ValueError(f'{filename}: stale counts; refresh the curated input without reordering it.')
            if label != 'raw' and not np.isfinite(frame[['longitude','latitude']]).all().all():
                raise ValueError(f'{filename}: missing or non-finite coordinates')
            inputs[filename] = file_hash(P / filename)
            frames[label] = frame
        summary = procedure_summary(frames['raw'], frames['ten'], frames['main'], i63)
        rows.append(dict(Therapy=therapy, diagnostic_denominator=i63, **summary))
    # Preserve site-level report grain; suppressed counts contribute no invented value.
    stroke_care, selected_diagnoses = [], 0
    for path in sorted((P / 'xml_2024').glob('*-2024-xml.xml')):
        root = ET.parse(path).getroot()
        relevant = [x for x in root.findall('.//Prozedur')
            if (x.findtext('OPS_301') or '').startswith(('8-981', '8-98b'))]
        if relevant:
            values = [(x.findtext('Anzahl') or '').strip() for x in relevant]
            if any(x and not x.isdigit() for x in values):
                raise ValueError('Invalid stroke-care procedure count')
            stroke_care.append(sum(int(x) for x in values if x))
        for diagnosis in root.findall('.//Hauptdiagnose'):
            code = (diagnosis.findtext('ICD_10') or '').strip()
            if code[:3] in {'I60','I61','I62','I63','I64'} or (code.startswith('G45') and not code.startswith('G45.4')):
                value = (diagnosis.findtext('Fallzahl') or '').strip()
                if value and not value.isdigit():
                    raise ValueError('Invalid diagnosis count')
                selected_diagnoses += int(value) if value else 0
    frame = pd.DataFrame({'total_cases':stroke_care})
    su = procedure_summary(frame, frame[frame.total_cases.ge(10)], frame[frame.total_cases.ge(25)], selected_diagnoses)
    rows.insert(0, dict(Therapy='SU care', diagnostic_denominator=selected_diagnoses, **su))
    pd.DataFrame(rows).to_csv(A / 'national_procedures_2024.csv', index=False)
    write_json(A / 'national_counts_2024.json', dict(report_files=len(sources), i63=i63,
        stroke_tia=selected_diagnoses, ivt=int(sources.lysis.sum()), mt=int(sources.thrombectomy.sum()),
        suppressed_ivt_cells=int(sources.lysis_suppressed_cells.sum()),
        suppressed_mt_cells=int(sources.thrombectomy_suppressed_cells.sum())))
    write_json(P / MANIFEST, {'sources':hashes, 'inputs':inputs})
    print(f'Validated {len(sources)} source reports; I63={i63}, IVT={sources.lysis.sum()}, MT={sources.thrombectomy.sum()}')


def age():
    i63 = json.loads((A / 'national_counts_2024.json').read_text())['i63']
    county, calibration = calibrate_2024(E / '12411-02-03-4.csv', i63)
    metadata = pd.read_csv(config.AGGREGATE_DIR / 'county_metadata.csv', dtype={'krs_code_str':str})
    if set(metadata.krs_code_str) != set(county.krs_code_str) or not metadata.krs_code_str.is_unique:
        raise ValueError('County classification and 2024 population do not have the same support')
    county = metadata.merge(county, on='krs_code_str', validate='one_to_one')
    county.to_csv(A / 'county_STR_analysis.csv', index=False)
    calibration.to_csv(A / 'age_calibration_2024.csv', index=False)


def nodes(force=False):
    frame = travel.create_demand_nodes(force=force)
    county = read('county_STR_analysis.csv').set_index('krs_code_str')
    total = frame.groupby('krs_code_str').node_pop.sum().reindex(county.index)
    if not np.allclose(total, county.total_pop, rtol=1e-10):
        raise ValueError('Demand-node weights do not conserve current county populations')
    return frame


def routes(force=False):
    frame = nodes()
    for therapy, name, prefix, envelope, cutoff in THERAPIES:
        hospitals = prepare_hospitals(pd.read_csv(P / f'{name}_10plus_geocoded.csv'), prefix)
        pairs = sparse.compute_or_load_sparse_pairs(frame, hospitals, f'{prefix}_10plus', envelope,
            force, config.ORS_BASE_URL)
        validate_sparse_pairs(frame, hospitals, pairs, envelope)
    print('Both sparse route tables are complete and identity/coordinate checked.')


def isochrones():
    from geostroke.iso_manager import _compute_isochrones
    status = requests.get(config.ORS_BASE_URL + '/v2/status', timeout=30)
    status.raise_for_status()
    engine = status.json()
    binding = {}
    unique = {}
    for suffix, filename in COHORTS.items():
        frame = pd.read_csv(P / filename)
        if not np.isfinite(frame[['longitude','latitude']]).all().all():
            raise ValueError(f'Invalid coordinates in {filename}')
        polygons = []
        for row in frame.itertuples():
            key = (row.longitude, row.latitude)
            if key not in unique:
                values = _compute_isochrones(*key)
                if len(values) != 12 or any(p.is_empty or not p.is_valid for p in values):
                    raise ValueError('ORS returned incomplete/invalid isochrones')
                unique[key] = values
            polygons.append(unique[key])
        hashes = {}
        for i, minute in enumerate(range(5,61,5)):
            path = C / f'poly{minute}{suffix}.pkl'
            path.write_bytes(pickle.dumps([p[i] for p in polygons], protocol=pickle.HIGHEST_PROTOCOL))
            hashes[path.name] = file_hash(path)
        binding[suffix] = {'input_file':filename, 'input_sha256':file_hash(P / filename),
            'coordinates':frame[['longitude','latitude']].values.tolist(), 'polygon_hashes':hashes}
    write_json(C / 'isochrone_site_binding_2024.json', binding)
    write_json(C / 'ors_status.json', engine)


def validate_polygons():
    binding_path = C / 'isochrone_site_binding_2024.json'
    if not binding_path.exists():
        raise ValueError('Missing local isochrone binding; generate isochrones or supply the reviewed cache with its provenance.')
    bindings = json.loads(binding_path.read_text())
    for suffix, filename in COHORTS.items():
        frame = pd.read_csv(P / filename)
        binding = bindings.get(suffix)
        if binding is None and suffix not in {'', '_extended_stroke'}:
            raise ValueError(f'Missing binding: {suffix}')
        if binding is not None:
            if binding['input_sha256'] != file_hash(P / filename):
                raise ValueError(f'Stale polygon input binding: {filename}')
            if not np.array_equal(frame[['longitude','latitude']].values, binding['coordinates']):
                raise ValueError(f'Polygon coordinate ordering changed: {filename}')
            for name, digest in binding.get('polygon_hashes', {}).items():
                if file_hash(C / name) != digest:
                    raise ValueError(f'Changed polygon geometry: {name}')
        for minute in range(5,61,5):
            polygons = pickle.loads((C / f'poly{minute}{suffix}.pkl').read_bytes())
            if len(polygons) != len(frame) or any(p is None or p.is_empty or not p.is_valid for p in polygons):
                raise ValueError(f'Invalid or misaligned polygons: {suffix}, {minute}')


def models():
    assert_current_inputs(P, P / 'xml_2024')
    counties = read('county_STR_analysis.csv')
    demand = nodes()
    hospitals, pairs = {}, {}
    geo = county_geometry(counties)
    for therapy, name, prefix, envelope, cutoff in THERAPIES:
        hospital = prepare_hospitals(pd.read_csv(P / f'{name}_10plus_geocoded.csv'), prefix)
        pair = pd.read_csv(sparse.sparse_path(f'{prefix}_10plus'), dtype={'krs_code_str':str,'node_id':str,'facility_id':str})
        validate_sparse_pairs(demand, hospital, pair, envelope)
        hospitals[therapy], pairs[therapy] = hospital, pair
        counties[f'dist_{therapy}'] = haversine_km(geo.centroid_lat.to_numpy()[:,None],
            geo.centroid_lon.to_numpy()[:,None], hospital.latitude.to_numpy()[None,:], hospital.longitude.to_numpy()[None,:]).min(axis=1)
    i63 = json.loads((A / 'national_counts_2024.json').read_text())['i63']
    result, km_beta = run_models(counties, hospitals['IVT'], hospitals['MT'], demand, pairs, i63)
    result.to_csv(A / 'county_STR_analysis.csv', index=False)
    result.to_csv(A / 'county_STR_travel_time_sensitivity_2024.csv', index=False)
    km_beta.to_csv(A / 'gravity_sensitivity.csv', index=False)
    summaries, betas, bins = [], [], []
    for therapy, name, prefix, envelope, cutoff in THERAPIES:
        summaries += travel.summarize_model(result, 'Current km-based STR', therapy, f'STR_{therapy}', f'dist_{therapy}', 'km')
        for suffix, label in [('primary','primary'),('narrow','narrow cutoff'),('candidate_envelope','candidate envelope, no time cutoff')]:
            summaries += travel.summarize_model(result, f'ORS travel-time STR ({label})', therapy,
                f'STR_{therapy}_time_{suffix}', f'nearest_{therapy}_time_min', 'min')
        betas += sparse.beta_sensitivity_sparse(result, demand, hospitals[therapy], pairs[therapy], therapy, cutoff)
        bounds = [0,10,20,30,np.inf] if therapy == 'IVT' else [0,15,30,45,np.inf]
        groups = pd.cut(result[f'nearest_{therapy}_time_min'], bounds, right=False)
        for group in groups.cat.categories:
            values = result.loc[groups == group, f'STR_{therapy}_time_primary']
            bins.append(dict(Therapy=therapy, Time_bin=str(group), N=len(values), Mean_STR=float(values.mean())))
    pd.DataFrame(summaries).to_csv(A / 'str_travel_time_summary_2024.csv', index=False)
    pd.DataFrame(betas).to_csv(A / 'gravity_travel_time_beta_sensitivity_2024.csv', index=False)
    pd.DataFrame([sparse.compare_models(result,t) for t in ['IVT','MT']]).to_csv(A / 'str_travel_time_model_comparison_2024.csv', index=False)
    pd.DataFrame(bins).to_csv(A / 'str_time_bins_2024.csv', index=False)
    print(pd.DataFrame(bins).to_string(index=False))


def coverage():
    from scripts import compute_table2_full_raster_2024 as table2
    from scripts import refresh_supplementary_figures_and_state_table_2024 as states
    validate_polygons()
    table, _, metadata = table2.compute_full_raster_table()
    table.to_csv(A / 'table2_full_raster_2024.csv', index=False)
    write_json(A / 'table2_full_raster_2024.json', metadata)
    germany = data.load_germany_outline().geometry.union_all()
    geometries, _ = table2.build_scenario_geometries(germany)
    counties = read('county_STR_analysis.csv')
    fractions = county_fractions(counties, dict(zip(['nontel','allsu','extended','ivt','mt'], geometries.values())))
    for key, values in fractions.items():
        counties[f'pop_frac_{key}'] = values
        counties[f'cov_{key}'] = np.asarray(values) > .5
    counties.to_csv(A / 'county_STR_analysis.csv', index=False)
    if 'STR_IVT_time_primary' in counties:
        counties.to_csv(A / 'county_STR_travel_time_sensitivity_2024.csv', index=False)
    state, _ = states.compute_state_raster_coverage()
    state.to_csv(A / 'bundesland_coverage.csv', index=False)
    denominator = metadata['exact_raster_totals']['population']
    summary = dict(coverage_denominator=denominator, urban_rural={}, supp_threshold_coverages={})

    def pct(polygons):
        return round(population.population_within(unary_union(polygons).intersection(germany)) / denominator * 100, 2)

    def polys(minute, suffix):
        return pickle.loads((C / f'poly{minute}{suffix}.pkl').read_bytes())

    for suffix, key in [('', 'stroke'),('_lysis_2024','lysis'),('_thrombectomy_2024','mt')]:
        summary[key + '_curve_full'] = [dict(time_min=m, pct=pct(polys(m,suffix))) for m in range(5,61,5)]
    for therapy, name, prefix, envelope, cutoff in THERAPIES:
        minute = 30 if therapy == 'IVT' else 60
        frame = pd.read_csv(P / f'{name}_centers_geocoded.csv')
        values = {}
        for threshold in [10,25,50,100]:
            if threshold == 10:
                selected = pd.read_csv(P / f'{name}_10plus_geocoded.csv')
                selected_polys = polys(minute, f'_{name}10_2024')
            else:
                selected = frame[frame.total_cases.ge(threshold)]
                full = polys(minute, f'_{name}_2024')
                selected_polys = [full[i] for i in selected.index]
            values[f'>={threshold}_n'], values[f'>={threshold}_pct'] = len(selected), pct(selected_polys)
        summary['supp_threshold_coverages'][prefix] = values
    icd_pct = pct(polys(30, '_icd_hospitals_2024'))
    coverages = table['Population Coverage (%)'].to_list()[:3] + [icd_pct]
    ns = table['Facilities (n)'].to_list()[:3] + [len(pd.read_csv(P / COHORTS['_icd_hospitals_2024']))]
    progressive = pd.DataFrame({'Step':[1,2,3,4], 'Facilities (n)':ns, 'Coverage (%)':coverages,
        'Added %':[np.nan] + [round(coverages[i]-coverages[i-1],2) for i in range(1,4)]})
    progressive.to_csv(A / 'supplementary_table2_progressive_coverage_2024.csv', index=False)
    summary.update(stroke_units_total=ns[1], stroke_units_nontele=ns[0],
        stroke_ready_additional_n=ns[2]-ns[1], table2_full_raster=table.to_dict('records'))
    national = json.loads((A / 'national_counts_2024.json').read_text())
    summary.update(icd_total=national['i63'], reporting_facilities_2024=national['report_files'])
    for row in read('national_procedures_2024.csv').to_dict('records'):
        if row['Therapy'] in {'IVT','MT'}:
            summary[row.pop('Therapy').lower() + '_proc'] = row
    legacy_county_outputs(counties, summary)
    write_json(A / 'manuscript_update_2024_summary.json', summary)
    print(table.to_string(index=False))


def tables():
    # Submission numbering, distinct from historical script names.
    names = {'national_procedures_2024.csv':'Table1', 'table2_full_raster_2024.csv':'Table2',
        'age_calibration_2024.csv':'TableS1', 'supplementary_table2_progressive_coverage_2024.csv':'TableS2',
        'bundesland_coverage.csv':'TableS4', 'gravity_travel_time_beta_sensitivity_2024.csv':'TableS5'}
    for name, label in names.items():
        read(name).to_csv(A / f'{label}.csv', index=False)
    summary = read('str_travel_time_summary_2024.csv')
    for model, label in [('ORS travel-time STR (primary)','Table3'), ('Current km-based STR','TableS6')]:
        summary[summary.Model.eq(model)].to_csv(A / f'{label}.csv', index=False)
    threshold = json.loads((A / 'manuscript_update_2024_summary.json').read_text())['supp_threshold_coverages']
    rows = [dict(Therapy=t.upper(), Minimum_procedures=n, N=threshold[t][f'>={n}_n'], Coverage_pct=threshold[t][f'>={n}_pct'])
        for n in [10,25,50,100] for t in ['ivt','mt']]
    pd.DataFrame(rows).to_csv(A / 'TableS3.csv', index=False)


def figures():
    from scripts import format_jon_figures_2024 as export
    export.make_all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['check-inputs','counts','age','nodes','isochrones','routes','models','coverage','tables','figures','all'])
    parser.add_argument('--force', action='store_true', help='Rebuild demand nodes or sparse routes rather than reuse reviewed caches')
    args = parser.parse_args()
    for folder in [A,C,travel.CACHE_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
    actions = {'check-inputs':check_inputs, 'counts':counts, 'age':age, 'nodes':lambda:nodes(args.force),
        'isochrones':isochrones, 'routes':lambda:routes(args.force), 'models':models, 'coverage':coverage, 'tables':tables, 'figures':figures}
    try:
        if args.step == 'all':
            # Expensive ORS queries occur only if no caches exist. Caches are validated later.
            for name in ['check-inputs','counts','age','nodes']:
                actions[name]()
            if not (C / 'isochrone_site_binding_2024.json').exists():
                isochrones()
            for name in ['routes','models','coverage','tables','figures']:
                actions[name]()
        else:
            actions[args.step]()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
