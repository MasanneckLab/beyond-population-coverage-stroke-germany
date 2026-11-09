"""Single-year age calibration for the 2024 cross-sectional analysis."""
from pathlib import Path

import numpy as np
import pandas as pd

SHARES = {"<45": 0.03, "45-64": 0.20, "65-74": 0.30, "75+": 0.47}
GROUPS = {
    "<45": ["unter 3 Jahre", "3 bis unter 6 Jahre", "6 bis unter 10 Jahre",
        "10 bis unter 15 Jahre", "15 bis unter 18 Jahre", "18 bis unter 20 Jahre",
        "20 bis unter 25 Jahre", "25 bis unter 30 Jahre", "30 bis unter 35 Jahre",
        "35 bis unter 40 Jahre", "40 bis unter 45 Jahre"],
    "45-64": ["45 bis unter 50 Jahre", "50 bis unter 55 Jahre",
        "55 bis unter 60 Jahre", "60 bis unter 65 Jahre"],
    "65-74": ["65 bis unter 75 Jahre"],
    "75+": ["75 Jahre und mehr"],
}


def calibrate_2024(path: Path, i63: int):
    """Return county expectations and audit table; rounding is display-only."""
    if i63 <= 0:
        raise ValueError("A positive 2024 diagnosis total is required")
    age = pd.read_csv(path, sep=";", skiprows=6, header=None, encoding="latin-1",
        names=["date", "code", "name", "age_group", "total", "male", "female"],
        dtype={"code": str})
    age["code"] = age.code.str.strip().replace({"02": "02000", "11": "11000"})
    age = age[age.code.str.len().eq(5) & age.age_group.ne("Insgesamt")].copy()
    if set(age.date) != {"31.12.2024"}:
        raise ValueError("Age populations must refer only to 31 December 2024")
    # The download also lists abolished counties with all 17 cells marked '-'.
    obsolete = age.groupby("code").total.apply(lambda values: values.eq("-").all())
    age = age[~age.code.isin(obsolete.index[obsolete])].copy()
    lookup = {label: group for group, labels in GROUPS.items() for label in labels}
    age["coarse"] = age.age_group.map(lookup)
    age["total"] = pd.to_numeric(age.total, errors="raise")
    if age.coarse.isna().any() or age.total.isna().any() or age.duplicated(["code", "age_group"]).any():
        raise ValueError("Unknown, missing, or duplicated age populations")
    if len(age.code.unique()) != 400 or not age.groupby("code").size().eq(17).all():
        raise ValueError("Expected 17 age groups in each of 400 counties")
    population = age.groupby("coarse").total.sum().reindex(SHARES)
    audit = pd.DataFrame({"population": population, "case_share": pd.Series(SHARES)})
    audit["estimated_cases"] = audit.case_share * i63
    audit["rate_per_100000"] = audit.estimated_cases / audit.population * 100000
    age["expected"] = age.total * age.coarse.map(audit.rate_per_100000) / 100000
    county = age.groupby("code").agg(total_pop=("total", "sum"), expected_strokes=("expected", "sum"))
    county["pop_65plus"] = age[age.coarse.isin(["65-74", "75+"])].groupby("code").total.sum()
    county["pct_65plus"] = (county.pop_65plus / county.total_pop * 100).round(1)
    county["stroke_rate"] = (county.expected_strokes / county.total_pop * 100000).round(0)
    if not np.isclose(county.expected_strokes.sum(), i63, rtol=1e-12):
        raise ValueError("Age-calibrated national expected count does not match 2024 I63")
    return county.rename_axis("krs_code_str").reset_index(), audit.rename_axis("age_group").reset_index()
