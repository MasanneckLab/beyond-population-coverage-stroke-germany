#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image
from scipy import stats
from shapely.geometry import shape
from shapely.ops import unary_union


import geostroke as gs
from geostroke import config
RAW_DATA = config.CACHE_DIR
PRIVATE_DIR = config.DATA_DIR
AUDEBERT_DIR = config.ANALYSIS_DIR
FIG_DIR = config.GRAPH_DIR
COUNTIES_GEOJSON = config.GERMANY_COUNTIES
GERMANY = STATES = None

MM_PER_INCH = 25.4
LANCET_FULL_WIDTH_MM = 190
LANCET_FULL_WIDTH_IN = LANCET_FULL_WIDTH_MM / MM_PER_INCH
PUBLICATION_DPI = 500
PUBLICATION_PIXEL_WIDTH = round(LANCET_FULL_WIDTH_IN * PUBLICATION_DPI)

plt.rcParams["figure.dpi"] = PUBLICATION_DPI
plt.rcParams["savefig.dpi"] = PUBLICATION_DPI
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COL_URBAN = "#5f9ec9"
COL_URBAN_LIGHT = "#cfe4f2"
COL_RURAL = "#d55e00"
COL_RURAL_LIGHT = "#f6c99f"
COL_OTHER = "#d9d9d9"
ORANGE_BLUE_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "orange_white_blue_cb",
    ["#d55e00", "#f7f7f7", "#0072b2"],
)


def load_pickle(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def flatten_counties_geojson(path: Path) -> gpd.GeoDataFrame:
    raw = json.loads(path.read_text())
    rows = []
    geoms = []
    for feat in raw["features"]:
        props = {}
        for key, value in feat["properties"].items():
            if isinstance(value, list):
                props[key] = value[0] if value else None
            elif isinstance(value, dict):
                props[key] = json.dumps(value, sort_keys=True)
            else:
                props[key] = value
        rows.append(props)
        geoms.append(shape(feat["geometry"]))
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=4326)


def county_gdf(counties: pd.DataFrame) -> gpd.GeoDataFrame:
    geo = flatten_counties_geojson(COUNTIES_GEOJSON)
    geo["krs_code"] = geo["krs_code"].astype(str).str.zfill(5)
    counties = counties.copy()
    counties["krs_code_str"] = counties["krs_code_str"].astype(str).str.zfill(5)
    merged = geo.merge(counties, left_on="krs_code", right_on="krs_code_str", how="left")
    missing = int(merged["krs_code_str"].isna().sum())
    if missing:
        raise ValueError(f"County geometry merge left {missing} counties without analysis rows")
    return merged


def set_map_extent(ax) -> None:
    ax.set_xlim(5.5, 15.5)
    ax.set_ylim(47.0, 55.5)
    ax.set_axis_off()


def base_map(ax) -> None:
    GERMANY.plot(ax=ax, color="#f0f0f0", edgecolor="#333333", linewidth=0.55)
    STATES.plot(ax=ax, facecolor="none", edgecolor="#777777", linewidth=0.45)
    set_map_extent(ax)


def overlay_state_borders(ax) -> None:
    STATES.boundary.plot(ax=ax, color="#666666", linewidth=0.65, zorder=8)
    GERMANY.boundary.plot(ax=ax, color="#333333", linewidth=0.85, zorder=8.1)


def publication_size(width_mm: float = LANCET_FULL_WIDTH_MM, height_mm: float = 150) -> tuple[float, float]:
    return width_mm / MM_PER_INCH, height_mm / MM_PER_INCH


def add_panel_label(ax, label: str, fontsize: float = 9.5) -> None:
    ax.text(
        0.015,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        zorder=20,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )


def save_existing_formats(fig, stem: str, formats: tuple[str, ...]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    base = FIG_DIR / stem
    for suffix in formats:
        if suffix == ".png":
            fig.savefig(base.with_suffix(suffix), dpi=PUBLICATION_DPI)
        elif suffix == ".tiff":
            fig.savefig(base.with_suffix(suffix), dpi=PUBLICATION_DPI, pil_kwargs={"compression": "tiff_lzw"})
        elif suffix == ".eps":
            fig.savefig(base.with_suffix(suffix))
        elif suffix == ".pdf":
            fig.savefig(base.with_suffix(suffix))
        else:
            raise ValueError(f"Unsupported suffix: {suffix}")


def valid_polys(polys):
    return [p for p in polys if p is not None and not p.is_empty]


def plot_union(ax, polys, color: str, alpha: float = 0.35) -> None:
    vals = valid_polys(polys)
    if not vals:
        return
    gpd.GeoSeries([unary_union(vals)], crs=4326).plot(
        ax=ax, color=color, alpha=alpha, edgecolor="none", zorder=3
    )


def make_figure_3_age_coverage() -> None:
    counties = pd.read_csv(AUDEBERT_DIR / "county_STR_analysis.csv")
    gdf = county_gdf(counties)
    gdf["pop_frac_ivt"] = gdf["pop_frac_ivt"].clip(0, 1)

    fig, axes = plt.subplots(1, 3, figsize=publication_size(height_mm=82))

    panels = [
        (
            axes[0],
            "A)",
            "pct_65plus",
            "YlOrRd",
            dict(vmin=17, vmax=34),
            r"% aged $\geq$65",
            r"Population aged $\geq$65 years",
        ),
        (
            axes[1],
            "B)",
            "stroke_rate",
            "Blues",
            dict(vmin=250, vmax=560),
            "per 100,000/yr",
            "Ischaemic stroke incidence",
        ),
    ]
    for ax, label, col, cmap, scale, cbar_label, title in panels:
        base_map(ax)
        gdf.plot(
            column=col,
            ax=ax,
            cmap=cmap,
            edgecolor="white",
            linewidth=0.1,
            alpha=0.95,
            legend=True,
            legend_kwds=dict(label=cbar_label, orientation="vertical", shrink=0.62, pad=0.02, format="%.0f"),
            **scale,
        )
        overlay_state_borders(ax)
        add_panel_label(ax, label)
        ax.set_title(title, fontsize=7.5, fontweight="bold")
        set_map_extent(ax)

    ax = axes[2]
    base_map(ax)
    cmap_c = ORANGE_BLUE_CMAP
    norm_c = mpl.colors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
    gdf.plot(
        column="pop_frac_ivt",
        ax=ax,
        cmap=cmap_c,
        norm=norm_c,
        edgecolor="white",
        linewidth=0.1,
        alpha=0.95,
        legend=False,
    )
    overlay_state_borders(ax)
    sm = plt.cm.ScalarMappable(cmap=cmap_c, norm=norm_c)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.62, pad=0.02, format="%.0%%")
    cbar.set_label("Population coverage\nfraction (IVT 30 min)", fontsize=6.5)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    cbar.ax.tick_params(labelsize=6)
    ax.set_title("IVT coverage (30 min)\nBlue = covered, orange = gap", fontsize=7.5, fontweight="bold")
    add_panel_label(ax, "C)")
    set_map_extent(ax)

    panel_axes = set(axes)
    for colorbar_ax in [candidate for candidate in fig.axes if candidate not in panel_axes]:
        colorbar_ax.tick_params(labelsize=6)
        colorbar_ax.yaxis.label.set_size(6.5)
    fig.tight_layout()
    save_existing_formats(fig, "Figure_3_new", (".png", ".eps", ".pdf", ".tiff"))
    plt.close(fig)


def make_s2_volume_threshold_sensitivity() -> None:
    summary = json.loads((AUDEBERT_DIR / "manuscript_update_2024_summary.json").read_text())
    labels = [">=10", ">=25", ">=50", ">=100"]
    specs = [
        ("A)", "IV thrombolysis centres", summary["supp_threshold_coverages"]["ivt"], COL_URBAN, "30-min coverage"),
        ("B)", "Mechanical thrombectomy centres", summary["supp_threshold_coverages"]["mt"], COL_RURAL, "60-min coverage"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=publication_size(height_mm=88), sharey=True)
    for ax, (panel, title, values, color, coverage_label) in zip(axes, specs):
        pct = [values[f"{label}_pct"] for label in labels]
        n_vals = [values[f"{label}_n"] for label in labels]
        x = np.arange(len(labels))
        bars = ax.bar(x, pct, color=color, width=0.62, edgecolor="white", linewidth=0.5)
        for bar, pc, n in zip(bars, pct, n_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                pc + 1.0,
                f"{pc:.1f}%\nn={n}",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Minimum annual procedure volume", fontsize=7)
        ax.set_title(f"{title}\n{coverage_label}", fontsize=8, fontweight="bold")
        ax.tick_params(axis="y", labelsize=6.5)
        ax.grid(axis="y", linestyle=":", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        add_panel_label(ax, panel, fontsize=9)
    axes[0].set_ylabel("Population coverage (%)", fontsize=7)
    fig.tight_layout()
    save_existing_formats(fig, "Supplementary_Figure_S2_new", (".png", ".eps", ".pdf", ".tiff"))
    plt.close(fig)


def make_s3_state_coverage() -> None:
    df = pd.read_csv(AUDEBERT_DIR / "bundesland_coverage.csv").sort_values("SU_30_pct")
    fig, ax = plt.subplots(figsize=publication_size(height_mm=158))
    y = np.arange(len(df))
    height = 0.22
    bars_su = ax.barh(y + height, df["SU_30_pct"], height, label="Stroke units, 30 min", color="#4c97c9")
    bars_ivt = ax.barh(y, df["Lysis_30_pct"], height, label="IVT >=25/yr, 30 min", color="#2d6db7")
    bars_mt = ax.barh(y - height, df["MT_60_pct"], height, label="MT >=25/yr, 60 min", color="#c61c33")
    ax.set_yticks(y)
    ax.set_yticklabels(df["Bundesland"], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("Population coverage (%)", fontsize=7)
    ax.set_xlim(0, 105)
    ax.tick_params(axis="x", labelsize=6.5)
    fig.suptitle("Coverage by federal state", fontsize=8.5, fontweight="bold", y=0.985)
    fig.legend(
        handles=[bars_su, bars_ivt, bars_mt],
        labels=["Stroke units, 30 min", "IVT >=25/yr, 30 min", "MT >=25/yr, 60 min"],
        loc="upper center",
        bbox_to_anchor=(0.58, 0.955),
        ncol=3,
        fontsize=6.5,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.2,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(top=0.89, bottom=0.08, left=0.19, right=0.985)
    save_existing_formats(fig, "Supplementary_Figure_S3_new", (".png", ".eps", ".pdf", ".tiff"))
    plt.close(fig)


def make_s4_gravity_sensitivity() -> None:
    df = pd.read_csv(AUDEBERT_DIR / "gravity_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=publication_size(height_mm=88), sharex=True)
    specs = [("A)", "IVT", 2.0), ("B)", "MT", 1.5)]
    for ax, (panel, therapy, ref_beta) in zip(axes, specs):
        sub = df[df["Therapy"] == therapy].sort_values("Beta")
        ax.plot(sub["Beta"], sub["Mean_STR_urban"], marker="o", color=COL_URBAN, linewidth=1.3, markersize=3.5, label="Urban mean STR")
        ax.plot(sub["Beta"], sub["Mean_STR_rural"], marker="s", color=COL_RURAL, linewidth=1.3, markersize=3.5, label="Rural mean STR")
        ax.axvline(ref_beta, color="#555555", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(therapy, fontsize=8, fontweight="bold")
        ax.set_xlabel("Distance-decay beta", fontsize=7)
        ax.set_ylabel("Mean STR", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.set_ylim(0, max(sub["Mean_STR_urban"].max(), sub["Mean_STR_rural"].max()) * 1.15)
        ax.grid(axis="y", linestyle=":", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax2 = ax.twinx()
        ax2.plot(
            sub["Beta"],
            sub["Pct_underserved_rural"],
            marker="^",
            linestyle="--",
            color="#444444",
            linewidth=1.1,
            markersize=3.5,
            label="Rural STR <0.5 (%)",
        )
        ax2.set_ylabel("Rural STR <0.5 (%)", fontsize=7)
        ax2.tick_params(labelsize=6.5)
        ax2.spines["top"].set_visible(False)
        lines = [line for line in ax.get_lines() + ax2.get_lines() if not line.get_label().startswith("_")]
        ax.legend(lines, [line.get_label() for line in lines], fontsize=6.2, loc="upper left", frameon=False)
        add_panel_label(ax, panel, fontsize=9)
    fig.tight_layout()
    save_existing_formats(fig, "Supplementary_Figure_S4_new", (".png", ".eps", ".pdf", ".tiff"))
    plt.close(fig)


def make_progressive_s1() -> None:
    stroke_df = gs.data.load_stroke_units()
    tele_mask = gs.data.telemed_mask(stroke_df)
    non_tele_idx = stroke_df.loc[~tele_mask].index.tolist()
    stroke_30 = load_pickle(RAW_DATA / "poly30.pkl")
    extended_30 = load_pickle(RAW_DATA / "poly30_extended_stroke.pkl")
    icd_30 = load_pickle(RAW_DATA / "poly30_icd_hospitals_2024.pkl")

    scenarios = [
        ("A)", "Certified Non-telemedical\nStroke Units (30 min)", [stroke_30[i] for i in non_tele_idx], "#4393c3"),
        ("B)", "All Certified\nStroke Units (30 min)", stroke_30, "#2166ac"),
        ("C)", "Certified + Stroke-ready\nHospitals (30 min)", extended_30, "#b2182b"),
        ("D)", "All Hospitals >=100\nStroke Diagnoses (30 min)", icd_30, "#762a83"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=publication_size(height_mm=246))
    for ax, (label, title, polys, color) in zip(axes.flatten(), scenarios):
        base_map(ax)
        plot_union(ax, polys, color, alpha=0.35)
        overlay_state_borders(ax)
        add_panel_label(ax, label)
        ax.set_title(title, fontsize=8, fontweight="bold")
        set_map_extent(ax)
    fig.tight_layout()
    save_existing_formats(fig, "Supplementary_Figure_S1_new", (".png", ".eps", ".pdf", ".tiff"))
    plt.close(fig)


def _scatter_decay(ax, df: pd.DataFrame, therapy: str, str_col: str, dist_col: str, title: str, ylabel: str, bins: list[float], x_label: str) -> None:
    colors = {"urban": COL_URBAN, "rural": COL_RURAL}
    for group, sub in df.groupby("urban_rural"):
        label = f"{group.capitalize()} counties" if x_label.startswith("Distance") else group.capitalize()
        ax.scatter(sub[dist_col], sub[str_col], s=10, alpha=0.45, color=colors.get(group, "#777777"), edgecolors="none", label=label)
    labels = []
    centers_by_label = {}
    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1]
        if hi >= 999:
            label = f">={lo:g}"
            previous_width = lo - bins[i - 1] if i else 10
            center = lo + previous_width / 2
        elif lo == 0:
            label = f"<{hi:g}"
            center = (lo + hi) / 2
        else:
            label = f"{lo:g}-<{hi:g}"
            center = (lo + hi) / 2
        labels.append(label)
        centers_by_label[label] = center
    tmp = df[[dist_col, str_col]].dropna().copy()
    tmp["_bin"] = pd.cut(tmp[dist_col], bins=bins, labels=labels, include_lowest=True, right=False)
    means = tmp.groupby("_bin", observed=True)[str_col].mean()
    centers = [centers_by_label[lab] for lab in means.index.astype(str)]
    ax.plot(centers, means.to_numpy(), color="black", marker="o", linewidth=1.3, markersize=3.5, label="Binned mean")
    corr = stats.spearmanr(df[dist_col], df[str_col], nan_policy="omit")
    ax.axhline(1.0, color="#777777", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(0.5, color="#c44e52", linestyle=":", linewidth=0.8, alpha=0.7)
    p_label = "p<0.001" if corr.pvalue < .001 else f"p={corr.pvalue:.3f}"
    ax.set_title(f"{title}\nSpearman rho={corr.statistic:.2f}, {p_label}", fontsize=7.5, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_str_map_scatter(stems: tuple[str, ...], travel_time: bool, shared_horizontal_colorbar: bool = False) -> None:
    if travel_time:
        counties = pd.read_csv(AUDEBERT_DIR / "county_STR_travel_time_sensitivity_2024.csv")
        str_cols = {"IVT": "STR_IVT_time_primary", "MT": "STR_MT_time_primary"}
        dist_cols = {"IVT": "nearest_IVT_time_min", "MT": "nearest_MT_time_min"}
        x_label = "Nearest qualifying centre (ORS driving time, min)"
        map_titles = {"IVT": "IVT travel-time STR", "MT": "MT travel-time STR"}
        scatter_titles = {"IVT": "IVT: driving time vs STR", "MT": "MT: driving time vs STR"}
        map_bins = {"IVT": [0, 10, 20, 30, 999], "MT": [0, 15, 30, 45, 999]}
    else:
        counties = pd.read_csv(AUDEBERT_DIR / "county_STR_analysis.csv")
        str_cols = {"IVT": "STR_IVT", "MT": "STR_MT"}
        dist_cols = {"IVT": "dist_IVT", "MT": "dist_MT"}
        x_label = "Distance to nearest centre (km)"
        map_titles = {"IVT": "IVT Standardised Treatment Ratio (30 min)", "MT": "MT Standardised Treatment Ratio (60 min)"}
        scatter_titles = {"IVT": "IVT: Distance vs. Treatment Ratio", "MT": "MT: Distance vs. Treatment Ratio"}
        map_bins = {"IVT": [0, 10, 20, 30, 50, 999], "MT": [0, 10, 20, 30, 50, 999]}

    gdf = county_gdf(counties)
    hospitals = {
        "IVT": pd.read_csv(PRIVATE_DIR / ("lysis_10plus_geocoded.csv" if travel_time else "lysis_centers_geocoded.csv")),
        "MT": pd.read_csv(PRIVATE_DIR / ("thrombectomy_10plus_geocoded.csv" if travel_time else "thrombectomy_centers_geocoded.csv")),
    }
    cmap = ORANGE_BLUE_CMAP
    vmax = 2.5 if travel_time else 2.0
    norm = mpl.colors.TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=vmax)
    if shared_horizontal_colorbar:
        fig = plt.figure(figsize=publication_size(height_mm=218))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.32, 0.06, 1.0], hspace=0.34, wspace=0.15)
        axes = np.array(
            [
                [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
            ],
            dtype=object,
        )
        cax = fig.add_subplot(gs[1, :])
    else:
        fig, axes = plt.subplots(2, 2, figsize=publication_size(height_mm=210), gridspec_kw={"height_ratios": [1.3, 1]})
        cax = None

    for col_idx, therapy in enumerate(["IVT", "MT"]):
        ax = axes[0, col_idx]
        base_map(ax)
        gdf.plot(
            column=str_cols[therapy],
            ax=ax,
            cmap=cmap,
            norm=norm,
            edgecolor="white",
            linewidth=0.08,
            alpha=0.92,
            missing_kwds=dict(color="#cccccc", edgecolor="white", linewidth=0.08),
        )
        overlay_state_borders(ax)
        ax.scatter(
            hospitals[therapy]["longitude"],
            hospitals[therapy]["latitude"],
            c="black",
            s=4,
            zorder=10,
            alpha=0.6,
            marker=".",
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        if not shared_horizontal_colorbar:
            cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.7, pad=0.02)
            cbar.set_label("STR (1.0 = expected)", fontsize=6.8)
            cbar.ax.tick_params(labelsize=6.5)
        ax.legend(
            handles=[Line2D([0], [0], marker=".", color="w", markerfacecolor="black", markersize=8, label="Hospitals")],
            loc="lower left",
            fontsize=6.5,
            framealpha=0.85,
            edgecolor="#aaaaaa",
        )
        add_panel_label(ax, "A)" if col_idx == 0 else "B)")
        ax.set_title(map_titles[therapy], fontsize=8, fontweight="bold")
        set_map_extent(ax)

        ax_sc = axes[1, col_idx]
        panel = "C)" if col_idx == 0 else "D)"
        _scatter_decay(
            ax_sc,
            counties,
            therapy,
            str_cols[therapy],
            dist_cols[therapy],
            scatter_titles[therapy],
            "Travel-time STR" if travel_time else "Standardised Treatment Ratio",
            map_bins[therapy],
            x_label,
        )
        add_panel_label(ax_sc, panel)

    if shared_horizontal_colorbar:
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap, norm=norm),
            cax=cax,
            orientation="horizontal",
        )
        cbar.ax.set_title("Standardised Treatment Ratio (STR)", fontsize=7.5, pad=2.5)
        cbar.ax.tick_params(labelsize=6.5)
        fig.subplots_adjust(top=0.975, bottom=0.07, left=0.07, right=0.98)
    else:
        fig.tight_layout()
    for stem in stems:
        save_existing_formats(fig, stem, (".png", ".eps", ".pdf", ".tiff"))
    plt.close(fig)


def make_s5_scatter_only() -> None:
    counties = pd.read_csv(AUDEBERT_DIR / "county_STR_travel_time_sensitivity_2024.csv")
    colors = {"urban": COL_URBAN, "rural": COL_RURAL}
    fig, axes = plt.subplots(2, 2, figsize=publication_size(height_mm=142))
    panels = [
        ("A)", axes[0, 0], "IVT", "STR_IVT", "STR_IVT_time_primary", "IVT: km-based vs travel-time STR", "Current STR (Haversine km impedance)", "Travel-time STR (ORS minutes impedance)"),
        ("B)", axes[0, 1], "MT", "STR_MT", "STR_MT_time_primary", "MT: km-based vs travel-time STR", "Current STR (Haversine km impedance)", "Travel-time STR (ORS minutes impedance)"),
    ]
    for label, ax, therapy, xcol, ycol, title, xlabel, ylabel in panels:
        for group, sub in counties.groupby("urban_rural"):
            ax.scatter(sub[xcol], sub[ycol], s=10, alpha=0.65, color=colors.get(group, "#777777"), label=group.capitalize(), edgecolor="none")
        lim = max(float(np.nanquantile(counties[[xcol, ycol]].to_numpy(), 0.98)), 2.5)
        ax.plot([0, lim], [0, lim], color="#444444", linewidth=0.8, linestyle="--")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        corr = stats.spearmanr(counties[xcol], counties[ycol], nan_policy="omit")
        ax.set_title(f"{title}\nSpearman rho={corr.statistic:.2f}", fontsize=7.5, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.legend(frameon=False, fontsize=6.5)
        add_panel_label(ax, label)

    for label, ax, therapy, bins in [
        ("C)", axes[1, 0], "IVT", [0, 10, 20, 30, 999]),
        ("D)", axes[1, 1], "MT", [0, 15, 30, 45, 999]),
    ]:
        _scatter_decay(
            ax,
            counties,
            therapy,
            f"STR_{therapy}_time_primary",
            f"nearest_{therapy}_time_min",
            f"{therapy}: travel time vs STR",
            "Travel-time STR",
            bins,
            "Nearest qualifying centre (ORS driving time, min)",
        )
        add_panel_label(ax, label)

    fig.tight_layout()
    save_existing_formats(fig, "Supplementary_Figure_S5_travel_time_STR_sensitivity_new", (".png", ".eps", ".pdf", ".tiff"))
    plt.close(fig)
