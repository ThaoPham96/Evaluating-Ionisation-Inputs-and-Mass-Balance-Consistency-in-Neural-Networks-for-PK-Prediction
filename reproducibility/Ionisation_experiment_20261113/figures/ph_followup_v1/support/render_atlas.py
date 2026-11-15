"""Render all endpoint metrics and paired pH contrasts from frozen scores."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ionisation_style import AXES, FONT, GRID, INK, MUTED, ENDPOINT_COLORS
EXPERIMENT = HERE.parents[2]
SCORING = EXPERIMENT / "outputs/ph_followup/scoring"
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909)
MODELS = {"GraphConv": "GraphConv", "Chemprop": "Chemprop", "ChemBERTa_head": "Frozen ChemBERTa + MLP"}
ENDPOINTS = {
    "t_half": ("Half-life", ENDPOINT_COLORS["t_half"], 316, 316),
    "Vd": ("Vd", ENDPOINT_COLORS["Vd"], 259, 259),
    "CL": ("CL", ENDPOINT_COLORS["CL"], 246, 246),
    "Cmax": ("Cmax", ENDPOINT_COLORS["Cmax"], 190, 190),
    "Tmax": ("Tmax", ENDPOINT_COLORS["Tmax"], 249, 249),
    "F": ("F", ENDPOINT_COLORS["F"], 172, 172),
    "PPB": ("PPB", ENDPOINT_COLORS["PPB"], 292, 284),
}
CONDITIONS = {
    ("primary", "REF0"): "Structure only",
    ("primary", "REF74"): "pH 7.4 category",
    ("primary", "REP2"): "pH 7.4 category ×2",
    ("primary", "REP4"): "pH 7.4 category ×4",
    ("primary", "ADD2"): "pH 7.4 + 2.0 categories",
    ("primary", "ADD65"): "pH 7.4 + 6.5 categories",
    ("primary", "ADD5"): "pH 7.4 + 5.0 categories",
    ("primary", "ALL4"): "Four-pH categories",
    ("tuned", "ALL4"): "Four-pH categories",
    ("tuned", "REP4"): "pH 7.4 category ×4",
    ("historical", "P2"): "Raw pKa pair",
    ("historical", "P3"): "pH 7.4 propensity pair",
    ("historical", "P4"): "Four-pH propensity profile",
    ("historical", "P5"): "pH 7.4 propensity pair ×4",
}
PROCEDURES = {"primary": "Shared recipe", "tuned": "Separate tuning", "historical": "Historical numerical inputs"}
KEYS = ["cohort", "procedure", "backbone", "arm", "endpoint"]
BETTER, WORSE = "#176E7B", "#AF4D36"


@dataclass(frozen=True)
class Metric:
    key: str
    title: str
    short: str
    relative: bool
    colour_limit: float
    detail: str

    @property
    def colour_unit(self):
        return "% change" if self.relative else "Score change"

    @property
    def direction(self):
        return "Negative values indicate lower error" if self.relative else "Positive values indicate higher scores"


METRICS = (
    Metric("mae", "Mean absolute error", "MAE", True, 15,
           "MAE uses log10 units except bioavailability (F) and protein binding (PPB), which use percentage points."),
    Metric("aafe", "Average absolute fold error", "AAFE", True, 20,
           "AAFE is a fold error on positive observed labels. PPB has 284 positive labels among 292 observed values."),
    Metric("spearman", "Spearman rank correlation", "Spearman ρ", False, 0.15,
           "Spearman ρ measures rank agreement between predictions and observations."),
    Metric("r2", "Coefficient of determination", "R²", False, 0.20,
           "R² is 1 − residual sum of squares / total sum of squares. Negative values indicate error above the mean-label reference."),
)


def digest(path):
    with path.open("rb") as handle:
        return sha256(handle.read()).hexdigest()


def read_inputs():
    manifest_path = SCORING / "scoring_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest["status"] != "report_ready" or manifest["cohort"] != "numerical"
        or manifest["n_structures"] != 327 or manifest["n_groups"] != 238
        or tuple(manifest["seeds"]) != SEEDS or manifest["outer_folds"] != [1, 2, 3, 4, 5]
    ):
        raise ValueError("Expected completed scoring on the frozen numerical cohort")
    sources = {manifest_path.name: digest(manifest_path)}
    tables = {}
    for name in ("endpoint_summary", "endpoint_metrics", "contrasts", "contrast_seeds"):
        path = SCORING / f"{name}.csv"
        sources[path.name] = digest(path)
        if sources[path.name] != manifest["files"][path.name]:
            raise ValueError(f"Score checksum mismatch for {path.name}")
        tables[name] = pd.read_csv(path, float_precision="round_trip")
    return manifest, sources, tables


def validate_metrics(summary, seeds):
    expected = {
        ("numerical", procedure, model, arm, endpoint)
        for procedure, arm in CONDITIONS for model in MODELS for endpoint in ENDPOINTS
    }
    if set(summary[KEYS].itertuples(index=False, name=None)) != expected or len(summary) != len(expected):
        raise ValueError("Expected 14 conditions × 3 models × 7 endpoints in the summary")
    expected_seeds = {(*row, seed) for row in expected for seed in SEEDS}
    actual_seeds = set(seeds[[*KEYS, "seed"]].itertuples(index=False, name=None))
    if actual_seeds != expected_seeds or len(seeds) != len(expected_seeds):
        raise ValueError("Expected nine unique scored seeds for each condition/model/endpoint")
    for table in (summary, seeds):
        for key, i in (("n_structures", 2), ("n_aafe", 3)):
            np.testing.assert_array_equal(table[key], table.endpoint.map({ep: row[i] for ep, row in ENDPOINTS.items()}))
    indexed = summary.set_index(KEYS).sort_index()
    grouped = seeds.groupby(KEYS, sort=True)
    for metric in METRICS:
        column = seeds[metric.key].to_numpy()
        if not np.isfinite(column).all():
            raise ValueError(f"Nonfinite {metric.short} values in the seed metrics")
        np.testing.assert_allclose(indexed[f"{metric.key}_mean"], grouped[metric.key].mean(), atol=1e-12)
        np.testing.assert_allclose(indexed[f"{metric.key}_seed_sd"], grouped[metric.key].std(), atol=1e-12)
        np.testing.assert_array_equal(indexed[f"{metric.key}_n_defined_seeds"], len(SEEDS))


def heatmap_values(summary):
    values = []
    common = [*KEYS, "n_structures", "n_aafe", "n_groups"]
    for metric in METRICS:
        column = f"{metric.key}_mean"
        table = summary[common + [column]].rename(columns={column: "absolute_mean"})
        baseline = table.loc[table.procedure.eq("primary") & table.arm.eq("REF74"), ["backbone", "endpoint", "absolute_mean"]]
        table = table.merge(baseline.rename(columns={"absolute_mean": "ref74_mean"}), on=["backbone", "endpoint"], validate="many_to_one")
        change = table.absolute_mean - table.ref74_mean
        if metric.relative:
            if not table.ref74_mean.gt(0).all():
                raise ValueError(f"Invalid {metric.short} reference denominator")
            change = 100 * change / table.ref74_mean
        if not change.abs().le(metric.colour_limit).all():
            raise ValueError(f"Change outside the {metric.short} heatmap scale")
        table["colour_value"] = change
        table["metric"] = metric.key
        table["colour_unit"] = metric.colour_unit
        values.append(table)
    return pd.concat(values, ignore_index=True)


def paired_values(tables):
    source = tables["endpoint_metrics"]
    keys = [key for key in KEYS if key != "arm"] + ["seed"]
    mask = source.procedure.isin(("primary", "tuned"))
    columns = [*keys, "n_structures", "n_aafe", *(metric.key for metric in METRICS)]
    a = source.loc[mask & source.arm.eq("ALL4"), columns]
    b = source.loc[mask & source.arm.eq("REP4"), columns]
    paired = a.merge(b, on=keys, suffixes=("_all4", "_rep4"), validate="one_to_one")
    if len(paired) != 2 * len(MODELS) * len(ENDPOINTS) * len(SEEDS):
        raise ValueError("Missing paired ALL4 and REP4 predictions")
    for key in ("n_structures", "n_aafe"):
        np.testing.assert_array_equal(paired[f"{key}_all4"], paired[f"{key}_rep4"])
    groups = ["cohort", "procedure", "backbone", "endpoint"]
    values = []
    for metric in METRICS:
        tab = paired[[*keys, "n_structures_all4", "n_aafe_all4", f"{metric.key}_all4", f"{metric.key}_rep4"]].rename(columns={
            "n_structures_all4": "n_structures", "n_aafe_all4": "n_aafe",
            f"{metric.key}_all4": "all4_value", f"{metric.key}_rep4": "rep4_value",
        })
        tab["mean_rep4"] = tab.groupby(groups).rep4_value.transform("mean")
        tab["absolute_delta"] = tab.all4_value - tab.rep4_value
        tab["plotted_delta"] = tab.absolute_delta
        if metric.relative:
            tab["plotted_delta"] = 100 * tab.absolute_delta / tab.mean_rep4
        tab["metric"] = metric.key
        tab["plot_unit"] = metric.colour_unit
        values.append(tab)
    values = pd.concat(values, ignore_index=True)
    stats = values.groupby([*groups, "metric", "plot_unit"], as_index=False).agg(
        mean=("plotted_delta", "mean"), sd=("plotted_delta", "std"), n_seeds=("plotted_delta", "size"),
        n_structures=("n_structures", "first"), n_aafe=("n_aafe", "first"),
    )
    expected = tables["contrast_seeds"].query("arm=='ALL4' and comparator_arm=='REP4' and scope=='all' and stratum=='all'")
    check = values.loc[values.metric.eq("mae")].merge(expected, on=keys, validate="one_to_one")
    if len(check) != len(paired):
        raise ValueError("Scorer output is missing paired MAE seed values")
    np.testing.assert_allclose(check.absolute_delta, check.delta, atol=1e-12)
    summary = tables["contrasts"].query("arm=='ALL4' and comparator_arm=='REP4' and scope=='all' and stratum=='all'")
    check = stats.loc[stats.metric.eq("mae")].merge(summary, on=groups, validate="one_to_one")
    if len(check) != len(stats) // len(METRICS):
        raise ValueError("Scorer output is missing paired MAE summaries")
    np.testing.assert_allclose(check["mean"], check.relative_delta_pct, atol=1e-12)
    np.testing.assert_allclose(check.sd, 100 * check.seed_sd / check.comparator_mae, atol=1e-12)
    return values, stats


def make_page(title, subtitle, detail, height=10.4):
    fig = plt.figure(figsize=(14.6, height), facecolor="white")
    fig.text(0.035, 0.949, title, fontsize=23, fontweight="bold", color=INK)
    fig.text(0.035, 0.911, subtitle, fontsize=13, color=MUTED)
    fig.text(0.035, 0.046, detail, fontsize=11.8, color=MUTED)
    fig.text(0.035, 0.025, "Frozen numerical cohort  •  327 structures  •  238 groups  •  Five grouped outer folds  •  Nine training seeds", fontsize=11.8, color=MUTED)
    return fig


def plot_heatmap(metric, values):
    fig = make_page(
        metric.title,
        f"Cell values are nine-seed means. Colour shows {'percentage change' if metric.relative else 'the score difference'} from the same model’s shared-recipe pH 7.4 category (REF74).",
        metric.detail,
    )
    palette = (BETTER, "#F4F6F5", WORSE) if metric.relative else (WORSE, "#F4F6F5", BETTER)
    cmap = LinearSegmentedColormap.from_list("reference_change", palette)
    norm = TwoSlopeNorm(vmin=-metric.colour_limit, vcenter=0, vmax=metric.colour_limit)
    grid = fig.add_gridspec(1, 3, left=0.247, right=0.971, bottom=0.167, top=0.823, wspace=0.08)
    separators = {"primary": 0, "tuned": 8, "historical": 10}
    n_cols, n_rows = len(ENDPOINTS), len(CONDITIONS)
    entries = list(CONDITIONS)
    for col, (model, name) in enumerate(MODELS.items()):
        ax = fig.add_subplot(grid[0, col])
        sub = values.loc[values.backbone.eq(model)].set_index(["procedure", "arm", "endpoint"])
        ordered = sub.loc[[(procedure, arm, ep) for procedure, arm in entries for ep in ENDPOINTS]]
        colour = ordered.colour_value.to_numpy().reshape(n_rows, n_cols)
        scores = ordered.absolute_mean.to_numpy().reshape(n_rows, n_cols)
        mesh = ax.pcolormesh(np.arange(n_cols+1), np.arange(n_rows+1), colour, cmap=cmap, norm=norm,
                             edgecolors="white", linewidths=0.65, shading="flat")
        for i, j in np.ndindex(colour.shape):
            rgb = cmap(norm(colour[i, j]))[:3]
            luminance = 0.2126*rgb[0] + 0.7152*rgb[1] + 0.0722*rgb[2]
            ax.text(j+0.5, i+0.5, f"{scores[i, j]:.3f}", ha="center", va="center",
                    color="white" if luminance < 0.53 else INK, fontsize=10.4,
                    fontweight="bold" if i == 1 else "normal")
        ax.set(xlim=(0, n_cols), ylim=(n_rows, 0), xticks=np.arange(n_cols)+0.5, yticks=[])
        tick_labels = [f"{label}\n{aafe_n if metric.key=='aafe' else n}" for label, _, n, aafe_n in ENDPOINTS.values()]
        ax.set_xticklabels(tick_labels, fontsize=10.4, linespacing=1.0)
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=5)
        ax.set_title(name, fontsize=13.5, fontweight="bold", color=INK, pad=33)
        for tick, (_, color, _, _) in zip(ax.get_xticklabels(), ENDPOINTS.values()):
            tick.set_color(color)
        for i in (8, 10):
            ax.axhline(i, color=MUTED, lw=1.1)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if col == 0:
            for i, ((procedure, arm), label) in enumerate(CONDITIONS.items()):
                y = i+0.5
                style = "bold" if (procedure, arm) == ("primary", "REF74") else "normal"
                ax.text(-0.13, y, label, ha="right", va="center", color=INK, fontsize=10.8, fontweight=style)
                ax.text(-4.1, y, arm, ha="left", va="center", color=MUTED, fontsize=10.4, fontweight=style)
            for procedure, i in separators.items():
                ax.text(-6.0, i+0.5, PROCEDURES[procedure].replace(" ", "\n"),
                        ha="left", va="center", fontsize=11, fontweight="bold", color=INK, linespacing=1.05)
    bar_ax = fig.add_axes([0.392, 0.123, 0.384, 0.018])
    ticks = np.linspace(-metric.colour_limit, metric.colour_limit, 7 if metric.key == "spearman" else 5)
    def format_tick(value, _):
        if metric.relative:
            return "0%" if value == 0 else f"{value:+g}%".replace("-", "−")
        return "0.00" if value == 0 else f"{value:+.2f}".replace("-", "−")

    fmt = FuncFormatter(format_tick)
    bar = fig.colorbar(mesh, cax=bar_ax, orientation="horizontal", ticks=ticks, format=fmt)
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor("face")
    bar.ax.tick_params(labelsize=10.5, length=3, pad=4)
    bar.outline.set_visible(False)
    fig.text(0.379, 0.132, "Lower error" if metric.relative else "Lower score", fontsize=11.8, ha="right", va="center", color=MUTED)
    fig.text(0.789, 0.132, "Higher error" if metric.relative else "Higher score", fontsize=11.8, ha="left", va="center", color=MUTED)
    fig.text(0.035, 0.083, "Four-pH profiles use pH 2.0, 5.0, 6.5 and 7.4. Numbers below endpoint names give labelled structures used for each metric.", fontsize=11.8, color=MUTED)
    fig.text(0.035, 0.065, "Separate tuning uses the same cohort and folds. Some fitted predictions were reused.", fontsize=11.8, color=MUTED)
    return fig


def plot_paired(metric, values, summary):
    fig = make_page(
        f"Four-pH categories versus repeated pH 7.4 · {metric.short}",
        "Small dots show nine paired seeds. Diamonds show the mean. Bars show ±1 seed standard deviation.",
        "Seed SD describes variation across fits. Confidence intervals for the prespecified MAE tests are shown in the main results figure.",
        height=9.85,
    )
    fig.text(0.035, 0.079, metric.detail, fontsize=11.8, color=MUTED)
    fig.text(0.035, 0.063, "Separate tuning uses the same cohort and folds. Some fitted predictions were reused.", fontsize=11.8, color=MUTED)
    grid = fig.add_gridspec(2, 3, left=0.187, right=0.971, bottom=0.164, top=0.817, hspace=0.27, wspace=0.15)
    low = min(values.plotted_delta.min(), (summary["mean"]-summary.sd).min())
    high = max(values.plotted_delta.max(), (summary["mean"]+summary.sd).max())
    padding = 0.045 * (high - low)
    ticks = MaxNLocator(nbins=5, min_n_ticks=4).tick_values(min(0, low)-padding, max(0, high)+padding)
    bounds = (ticks[0], ticks[-1])
    jitter = np.array([-0.24, 0.18, -0.12, 0.24, 0.06, -0.18, 0.12, -0.06, 0.0])
    positions = np.arange(len(ENDPOINTS))[::-1]
    for row, procedure in enumerate(("primary", "tuned")):
        fig.text(0.035, (0.850, 0.468)[row], "AB"[row], fontweight="bold", fontsize=14, color=INK)
        fig.text(0.055, (0.850, 0.468)[row], PROCEDURES[procedure], fontweight="bold", fontsize=13.5, color=INK)
        for col, (model, name) in enumerate(MODELS.items()):
            ax = fig.add_subplot(grid[row, col])
            if row == 0:
                ax.set_title(name, fontsize=13.5, fontweight="bold", color=INK, pad=9)
            sample = values.loc[values.procedure.eq(procedure) & values.backbone.eq(model)].set_index(["endpoint", "seed"])
            means = summary.loc[summary.procedure.eq(procedure) & summary.backbone.eq(model)].set_index("endpoint")
            for (endpoint, (_, color, _, _)), y in zip(ENDPOINTS.items(), positions):
                by_seed = sample.loc[endpoint, "plotted_delta"].loc[list(SEEDS)]
                stat = means.loc[endpoint]
                ax.scatter(by_seed, y + jitter, s=22, facecolor=color, edgecolor="white", linewidth=0.48, alpha=0.72, zorder=4)
                ax.errorbar(stat["mean"], y, xerr=stat.sd, fmt="D", markersize=6.0, markeredgewidth=0.55,
                            markeredgecolor=INK, markerfacecolor=color, color=color, linewidth=1.6, capsize=2.4, zorder=5)
            ax.set(xlim=bounds, ylim=(-0.56, len(ENDPOINTS)-0.44), xticks=ticks, yticks=positions)
            ax.set_axisbelow(True)
            ax.grid(axis="x", color=GRID, lw=0.65)
            ax.axvline(0, color=AXES, ls=(0, (4, 3)), lw=0.9, zorder=1)
            ax.tick_params(axis="y", length=0, pad=9, labelsize=13, labelleft=col == 0)
            ax.tick_params(axis="x", length=3, color=MUTED, labelsize=12, labelbottom=row == 1)
            if row == 0:
                ax.tick_params(axis="x", length=0)
                ax.spines["bottom"].set_visible(False)
            ax.spines[["left", "right", "top"]].set_visible(False)
            ax.spines["bottom"].set_color("#ADBABC")
            if col == 0:
                ax.set_yticklabels([f"{label}  ({aafe_n if metric.key=='aafe' else n})" for label, _, n, aafe_n in ENDPOINTS.values()])
                for tick, color in zip(ax.get_yticklabels(), ENDPOINT_COLORS.values()):
                    tick.set_color(color)
    label = f"Change in {metric.short} (% of mean REP4)" if metric.relative else f"Change in {metric.short} (ALL4 − REP4)"
    fig.text(0.579, 0.111, f"{label}   •   {metric.direction}", ha="center", fontsize=11.5, color=INK)
    return fig


CAPTIONS = """# Categorical pH follow-up metric atlas

**Pages 1–4.** Mean absolute error (MAE), average absolute fold error (AAFE), Spearman rank correlation and coefficient of determination (R²) for 14 run conditions across three backbones and seven PK endpoints. Each printed value is the arithmetic mean of nine per-seed scores. Each seed pools held-out predictions from five grouped outer folds. Cell colour compares the run with the shared-recipe pH 7.4 category (REF74) for the same backbone and endpoint. MAE and AAFE use percentage changes with REF74 mean error as the fixed denominator. Spearman and R² use raw score differences. Teal indicates better performance on each metric. Colour scales cover all observations. Printed values are rounded to three decimal places. The CSV preserves full precision.

Shared-recipe rows compare pH categories with structure-only input and repeated copies of the pH 7.4 category. Separate-tuning rows show the full four-pH category and its width-matched four-copy control, each tuned on the same cohort and folds as the shared-recipe rows. Exact matches reused fitted predictions, including 36 of 45 outer fits per arm for frozen ChemBERTa. Historical inputs are the same numerical-cohort runs with raw pKas, pH 7.4 propensities, the four-pH propensity profile, or the pH 7.4 propensity pair repeated four times. The four-pH representations use pH 2.0, 5.0, 6.5 and 7.4. Both historical and new runs use the same frozen labels and held-out folds.

**Pages 5–8.** Paired ALL4 minus REP4 differences. Each small dot is one seed with predictions pooled across the five held-out folds. Vertical offsets improve visibility. Diamonds show the nine-seed mean and bars show ±1 sample SD (ddof = 1). SD describes paired seed variability. The main results figure reports the bootstrap intervals for MAE. For MAE and AAFE, each seed’s absolute difference is divided by the fixed REP4 MAE or AAFE averaged across the nine seeds for that backbone, tuning procedure and endpoint, then multiplied by 100. Spearman and R² are raw differences. Higher Spearman and R² and lower MAE and AAFE favour ALL4. Counts in parentheses give observed labels used for the plotted metric.

MAE uses log₁₀ units for half-life, Vd, CL, Cmax and Tmax, and percentage points for bioavailability (F) and plasma protein binding (PPB). AAFE uses strictly positive labels. The 292 PPB MAE labels include eight zeros, leaving 284 labels for PPB AAFE. Spearman and R² use log₁₀-scale values for the five positive endpoints and original percentage-scale values for F and PPB. All models use the 327-structure numerical cohort and 238 frozen parent/scaffold groups, with zero mass-balance loss and equal supervised endpoint weights.
"""


def main():
    scoring, sources, tables = read_inputs()
    validate_metrics(tables["endpoint_summary"], tables["endpoint_metrics"])
    heat = heatmap_values(tables["endpoint_summary"])
    paired, estimates = paired_values(tables)
    plt.rcParams.update({
        "font.family": FONT, "font.size": 11, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "pdf.fonttype": 42,
        "svg.fonttype": "none", "svg.hashsalt": "ph-followup-atlas-v1",
    })
    pages, outputs = [], []
    pdf_path = HERE / "metric_atlas.pdf"
    with PdfPages(pdf_path, metadata={"Title": "Categorical pH follow-up metric atlas", "CreationDate": None}) as pdf:
        for group in ("absolute", "paired"):
            for metric in METRICS:
                if group == "absolute":
                    fig = plot_heatmap(metric, heat.loc[heat.metric.eq(metric.key)])
                else:
                    fig = plot_paired(metric, paired.loc[paired.metric.eq(metric.key)], estimates.loc[estimates.metric.eq(metric.key)])
                stem = f"ph_metric_comparison_{group}_{metric.key}"
                pdf.savefig(fig)
                png, svg = HERE / f"{stem}.png", HERE / f"{stem}.svg"
                fig.savefig(png, dpi=210)
                fig.savefig(svg, metadata={"Date": None})
                plt.close(fig)
                pages.append({"page": len(pages)+1, "metric": metric.key, "display": group, "png": png.name, "svg": svg.name})
                outputs.extend((png, svg))
    outputs.append(pdf_path)
    for filename, table in (
        ("heatmap_values.csv", heat), ("paired_seed_values.csv", paired), ("paired_summary.csv", estimates),
    ):
        path = HERE / filename
        table.to_csv(path, index=False)
        outputs.append(path)
    captions = HERE / "captions.md"
    captions.write_text(CAPTIONS, encoding="utf-8")
    outputs.append(captions)
    manifest = {
        "campaign_id": scoring["campaign_id"], "release_fingerprints": scoring["release_fingerprints"],
        "inputs_directory": SCORING.relative_to(EXPERIMENT).as_posix(), "inputs_sha256": sources,
        "script_sha256": digest(Path(__file__)), "outputs_sha256": {p.name: digest(p) for p in outputs},
        "pages": pages, "heatmap_cells": len(heat), "paired_seed_values": len(paired), "paired_summaries": len(estimates),
        "seed_ids": SEEDS, "sd_ddof": 1, "heatmap_reference": "Same-model, same-endpoint primary REF74",
        "absolute_annotations_decimal_places": 3, "nonfinite_metrics": 0, "colour_clipping": False,
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(f"Rendered {len(pages)} atlas pages, {len(heat)} heatmap cells and {len(paired)} paired seed values.")


if __name__ == "__main__":
    main()
