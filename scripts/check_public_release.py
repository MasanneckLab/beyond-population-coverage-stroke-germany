"""Fail closed on unexpected staged data or publication documents."""
import csv
import io
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAFE_AGGREGATES = {
    'age_adjusted_coverage.csv', 'age_adjusted_coverage_continuous.csv',
    'age_calibration_2024.csv', 'bundesland_coverage.csv', 'county_metadata.csv',
    'county_STR_analysis.csv', 'county_STR_travel_time_sensitivity_2024.csv',
    'coverage_results.csv', 'gravity_sensitivity.csv',
    'gravity_travel_time_beta_sensitivity_2024.csv', 'sensitivity_pop_threshold.csv',
    'str_time_bins_2024.csv', 'str_travel_time_model_comparison_2024.csv',
    'str_travel_time_summary_2024.csv', 'supplementary_table2_progressive_coverage_2024.csv',
    'table2_full_raster_2024.csv', 'urban_rural_coverage.csv',
    'manuscript_update_2024_summary.json', 'national_counts_2024.json',
    'national_procedures_2024.csv',
}


def audit_file(name, blob):
    path = Path(name)
    if path.suffix.lower() not in {'.py','.md','.txt','.csv','.json','.toml','.cff','.yml','.yaml',''} and name not in {'.gitignore'}:
        raise ValueError(f'Unexpected file format: {name}')
    if name.startswith(('data/private/','data/external/','outputs/')) and path.name not in {'.gitkeep','README.md'}:
        raise ValueError(f'Private/generated file staged: {name}')
    if len(blob) > 2_000_000 or b'\0' in blob:
        raise ValueError(f'Unexpected binary or large file: {name}')
    text = blob.decode('utf-8')
    if re.search(r'/Users/[A-Za-z]|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----', text):
        raise ValueError(f'Local path or possible secret: {name}')
    if re.search(r'\b\d{9}-\d{9}-2024-xml\.xml\b', text):
        raise ValueError(f'Real-form hospital report identifier: {name}')
    if name.startswith('data/aggregate/') and path.name not in SAFE_AGGREGATES:
        raise ValueError(f'Aggregate not explicitly approved: {name}')
    if name.startswith('data/aggregate/') and path.suffix == '.json':
        def inspect(value):
            if isinstance(value, dict):
                forbidden = {'hospital_records','file_names','coordinates','sources','facility_id','latitude','longitude','full_address'}
                if forbidden.intersection(value):
                    raise ValueError(f'Hospital-level fields in aggregate JSON: {name}')
                for item in value.values():
                    inspect(item)
            elif isinstance(value, list):
                for item in value:
                    inspect(item)
        inspect(json.loads(text))
    if path.suffix == '.csv':
        rows = list(csv.reader(io.StringIO(text)))
        if name.startswith('data/templates/'):
            if len(rows) != 1:
                raise ValueError(f'Template contains records: {name}')
        elif name.startswith('data/aggregate/'):
            forbidden = {'file_name','facility_id','hospital_id','latitude','longitude','lat','lon','ik',
                'standortnummer','address','full_address','procedure_details','node_id','facility_idx'}
            if not rows or forbidden.intersection(x.lower() for x in rows[0]):
                raise ValueError(f'Hospital/routing columns in aggregate: {name}')
        else:
            raise ValueError(f'CSV outside approved data directories: {name}')


def main():
    records = subprocess.check_output(['git','ls-files','--stage','-z'], cwd=ROOT).decode().split('\0')
    count = 0
    for record in filter(None, records):
        mode, oid, rest = record.split(' ', 2)
        stage, name = rest.split('\t', 1)
        if mode not in {'100644','100755'} or stage != '0':
            raise ValueError(f'Symlink, submodule or unresolved index entry: {name}')
        blob = subprocess.check_output(['git','cat-file','blob',oid], cwd=ROOT)
        audit_file(name, blob)
        count += 1
    print(f'Public-release boundary passed for {count} staged files. Review aggregate semantics separately.')


if __name__ == '__main__':
    main()
