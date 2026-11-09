"""Final JON figure numbering and approved typography; no Word files required."""
from pathlib import Path
from hashlib import sha256
import json
import pickle
import re
import shutil
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
from shapely.ops import unary_union
from geostroke import config
from scripts import update_publication_figure_labels_borders as plot

FIGS = config.GRAPH_DIR
SUPP = FIGS / "Supplementary"
WORK = FIGS / "previews"
RECORDS = []
INPUTS = {}
SUPPLEMENT_DISPLAY_MM = {'FigS1': (140.43838888888888, 195.73602777777776), 'FigS2': (142.31025, 63.28080555555555), 'FigS3': (155.50636111111112, 130.16927777777778), 'FigS4': (161.29, 72.01027777777777), 'FigS5': (161.29, 119.25683333333333), 'FigS6': (161.29, 175.6725277777778)}

def record(path):
    INPUTS[str(path)] = sha256(path.read_bytes()).hexdigest()
    return path


def read_pickle(name):
    return pickle.loads(record(config.CACHE_DIR / name).read_bytes())


def values_digest(fig):
    digest = sha256()
    for ax in fig.axes:
        for patch in ax.patches:
            digest.update(patch.get_path().vertices.tobytes())
            digest.update(patch.get_patch_transform().get_matrix().tobytes())
        for line in ax.lines:
            digest.update(np.asarray(line.get_xydata()).tobytes())
        for collection in ax.collections:
            digest.update(np.asarray(collection.get_offsets()).tobytes())
            arr = collection.get_array()
            if arr is not None:
                digest.update(np.asarray(arr).tobytes())
            for path in collection.get_paths():
                digest.update(path.vertices.tobytes())
    return digest.hexdigest()


def save(fig, stem, formats=None):
    names = {
        "Figure_3_new": "Fig2",
        "Figure_4_comp_new": "Fig3",
        "Supplementary_Figure_S1_new": "FigS1",
        "Supplementary_Figure_S2_new": "FigS2",
        "Supplementary_Figure_S3_new": "FigS3",
        "Supplementary_Figure_S4_new": "FigS4",
        "Supplementary_Figure_S5_travel_time_STR_sensitivity_new": "FigS5",
        "Supplementary_Figure_S6_km_based_STR_new": "FigS6",
    }
    name = names.get(stem, stem)
    target = (SUPP if name.startswith("FigS") else FIGS) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    before = values_digest(fig)
    min_size=8.0
    if name.startswith("FigS"):
        old_width, old_height = SUPPLEMENT_DISPLAY_MM[name]
        width=fig.get_figwidth()*25.4;height=fig.get_figheight()*25.4
        display_scale=min(old_width/width,old_height/height,1)
        min_size=8.05/display_scale
    for text in fig.findobj(Text):
        if re.fullmatch(r"[A-E]\)", text.get_text()):
            text.set_text(text.get_text()[0].lower())
        text.set_fontsize(max(text.get_fontsize(), min_size))
    if name == "FigS1":
        for ax,title in zip(fig.axes,["Non-telemedical\nstroke units", "All certified\nstroke units",
                "Certified units +\nstroke-ready hospitals", "All hospitals with\n>=100 stroke diagnoses"]):
            ax.set_title(title,fontsize=min_size,fontweight="bold")
        fig.tight_layout(pad=1.0)
    if name == "Fig2":
        for ax in fig.axes:
            if "Ischaemic stroke incidence" in ax.get_title():
                ax.set_title("Expected stroke incidence", fontsize=8, fontweight="bold")
            if "IVT coverage (30 min)" in ax.get_title():
                ax.set_title("IVT coverage (30 min)", fontsize=8, fontweight="bold")
            if "Population aged" in ax.get_title():
                ax.set_title("Population aged >=65 years", fontsize=8, fontweight="bold")
        fig.tight_layout(pad=0.7, w_pad=0.6)
    if name in {"Fig3", "FigS5", "FigS6"}:
        for ax in fig.axes:
            if ax.get_xlabel() == "Nearest qualifying centre (ORS driving time, min)":
                ax.set_xlabel("Nearest qualifying centre\n(ORS driving time, min)", fontsize=min_size)
            if name == "FigS5":
                if "Current STR" in ax.get_xlabel():
                    ax.set_xlabel("Km-based STR (Haversine)", fontsize=min_size)
                if "ORS minutes impedance" in ax.get_ylabel():
                    ax.set_ylabel("Travel-time STR (ORS)", fontsize=min_size)
            if name == "FigS6" and "Standardised Treatment Ratio (" in ax.get_title():
                therapy=ax.get_title().split()[0]
                ax.set_title(therapy+" km-based STR", fontsize=min_size, fontweight="bold")
        if name != "Fig3":
            fig.tight_layout(pad=1.2, h_pad=2.0, w_pad=1.6)
        else:
            fig.subplots_adjust(left=0.09, right=0.985, bottom=0.085, top=0.975, wspace=0.30)
    if name in {"FigS2", "FigS4"}:
        fig.tight_layout(pad=1.1, w_pad=1.4)
    if name == "FigS2":
        for ax in fig.axes:
            ax.set_ylim(0,120)
            ax.set_yticks([0,20,40,60,80,100])
    if name == "FigS4":
        shared_handles=None
        for ax in fig.axes:
            legend=ax.get_legend()
            if legend is not None:
                if shared_handles is None:
                    shared_handles=(legend.legend_handles,[t.get_text() for t in legend.get_texts()])
                legend.remove()
                ax.title.set_y(1.0)
        fig.legend(*shared_handles,loc="upper center",bbox_to_anchor=(.5,.995),ncol=3,
            frameon=False,fontsize=min_size,handlelength=1.5,columnspacing=1.1)
        fig.subplots_adjust(top=.82)
    if name == "FigS3":
        fig._suptitle.set_text("")
        fig.subplots_adjust(top=0.89, bottom=0.08, left=0.27, right=0.985)
    after = values_digest(fig)
    assert before == after, f"Plot values changed: {name}"
    fig.savefig(target.with_suffix(".pdf"), metadata={"Title": name, "Creator": "Matplotlib; harmonized 2024 JON export"})
    fig.savefig(target.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"}, facecolor="white", transparent=False)
    # RGB TIFF without alpha is required for straightforward print ingestion.
    with Image.open(target.with_suffix(".tiff")) as image:
        rgb = image.convert("RGB")
    rgb.save(target.with_suffix(".tiff"), dpi=(600, 600), compression="tiff_lzw")
    fig.savefig(WORK / f"{name}_preview.png", dpi=150)
    if not name.startswith("FigS"):
        subprocess.run([shutil.which("gs") or "gs", "-q", "-dBATCH", "-dNOPAUSE", "-sDEVICE=eps2write",
            "-r600", "-dEmbedAllFonts=true", "-dEPSCrop", f"-sOutputFile={target.with_suffix('.eps')}",
            str(target.with_suffix(".pdf"))], check=True)
    RECORDS.append(dict(figure=name, width_mm=round(fig.get_figwidth()*25.4, 2),
        height_mm=round(fig.get_figheight()*25.4, 2), unchanged_plot_values=before == after,
        values_sha256=before, minimum_text_pt=min(t.get_fontsize() for t in fig.findobj(Text) if t.get_text()),
        pdf=str(target.with_suffix(".pdf"))))


def make_first():
    summary = json.loads(record(config.ANALYSIS_DIR / "manuscript_update_2024_summary.json").read_text())
    facilities = plot.gs.data.load_stroke_units()
    assert len(facilities) == summary["stroke_units_total"] == 353
    nontele = facilities.loc[~plot.gs.data.telemed_mask(facilities)]
    assert len(nontele) == 322
    su = read_pickle("poly30.pkl")
    ivt = pd.read_csv(record(config.DATA_DIR / "lysis_centers_geocoded.csv"))
    mt = pd.read_csv(record(config.DATA_DIR / "thrombectomy_centers_geocoded.csv"))
    assert len(ivt) == 424 and len(mt) == 198
    specs = [
        ("a", [su[i] for i in nontele.index], nontele, "#9ecae1", "Certified non-telemedical stroke units\nn=322; 30-min coverage: 83.0%"),
        ("b", su, facilities, "#6baed6", "All certified stroke units\nn=353; 30-min coverage: 85.4%"),
        ("c", read_pickle("poly30_lysis_2024.pkl"), ivt, "#4a79c3", "IVT centres (>=25/year)"),
        ("d", read_pickle("poly60_thrombectomy_2024.pkl"), mt, "#d98b99", "MT centres (>=25/year)"),
    ]
    fig = plt.figure(figsize=(174/25.4, 210/25.4))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, .46], hspace=.17, wspace=.08)
    outline = plot.GERMANY.geometry.union_all()
    for index, (label, polys, points, color, title) in enumerate(specs):
        ax = fig.add_subplot(grid[index//2, index%2])
        plot.base_map(ax)
        geom = unary_union([p for p in polys if p is not None and not p.is_empty]).intersection(outline)
        gpd.GeoSeries([geom], crs=4326).plot(ax=ax, color=color, alpha=.55, edgecolor="none", zorder=3)
        plot.overlay_state_borders(ax)
        ax.scatter(points.longitude, points.latitude, c="#cc2f3c" if label != "d" else "#b53a4c",
            s=8, alpha=.7, edgecolors="darkred", linewidths=.25, zorder=5)
        plot.add_panel_label(ax, label)
        ax.set_title(title, fontsize=8, fontweight="bold")
    ax = fig.add_subplot(grid[2, :])
    for key, color, marker, style, label in [
        ("stroke_curve_full", "#4c97c9", "s", "-", "Certified stroke units"),
        ("lysis_curve_full", "#2d6db7", "o", "--", "IVT centres >=25/yr"),
        ("mt_curve_full", "#c61c33", "D", ":", "MT centres >=25/yr")]:
        vals = summary[key]
        ax.plot([v["time_min"] for v in vals], [v["pct"] for v in vals], color=color,
            marker=marker, linestyle=style, linewidth=1.15, markersize=3.1, label=label)
    ax.axhline(95, color="#bdbdbd", linestyle="-.", linewidth=.85, label="95% threshold")
    ax.axvline(30, color="#9fc5e8", linestyle=":", linewidth=1)
    ax.axvline(60, color="#f4cccc", linestyle=":", linewidth=1)
    ax.set(xlim=(4, 62), ylim=(0, 102), xticks=[5,15,30,45,60], yticks=[0,50,100],
        xlabel="Driving time (minutes)", ylabel="Population coverage (%)")
    ax.legend(loc="lower right", frameon=True, framealpha=.95, fontsize=8, ncol=2,
        columnspacing=1.0, handlelength=1.8, borderpad=.35)
    ax.grid(True, alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0,1.06,"e",transform=ax.transAxes,ha="left",va="bottom",fontsize=9.5,fontweight="bold")
    fig.subplots_adjust(top=.955, bottom=.065, left=.09, right=.985)
    metrics = pd.read_csv(config.ANALYSIS_DIR / "table2_full_raster_2024.csv")
    labels = ["Certified non-telemedical stroke units", "All certified stroke units", "IVT centres (>=25/year)", "MT centres (>=25/year)"]
    for ax, label, row_index, minutes in zip(fig.axes[:4], labels, [0,1,3,4], [30,30,30,60]):
        row = metrics.iloc[row_index]
        ax.set_title(f"{label}\nn={int(row['Facilities (n)'])}; {minutes}-min coverage: {row['Population Coverage (%)']:.1f}%", fontsize=8, fontweight="bold")
    save(fig,"Fig1")
    plt.close(fig)


def make_all():
    for folder in [FIGS, SUPP, WORK]:
        folder.mkdir(parents=True, exist_ok=True)
    plot.GERMANY = plot.gs.data.load_germany_outline()
    plot.STATES = plot.gs.data.load_states()
    plot.publication_size = lambda width_mm=174, height_mm=150: (174/25.4, min(height_mm,230)/25.4)
    plot.save_existing_formats = save
    make_first()
    plot.make_figure_3_age_coverage()
    plot.make_str_map_scatter(("Figure_4_comp_new",), travel_time=True, shared_horizontal_colorbar=True)
    plot.make_progressive_s1()
    plot.make_s2_volume_threshold_sensitivity()
    plot.make_s3_state_coverage()
    plot.make_s4_gravity_sensitivity()
    plot.make_s5_scatter_only()
    plot.make_str_map_scatter(("Supplementary_Figure_S6_km_based_STR_new",), travel_time=False)
    if not all(sha256(Path(p).read_bytes()).hexdigest() == digest for p,digest in INPUTS.items()):
        raise ValueError("Figure inputs changed while exporting")
    (WORK / "figure_validation.json").write_text(json.dumps(RECORDS, indent=2))
