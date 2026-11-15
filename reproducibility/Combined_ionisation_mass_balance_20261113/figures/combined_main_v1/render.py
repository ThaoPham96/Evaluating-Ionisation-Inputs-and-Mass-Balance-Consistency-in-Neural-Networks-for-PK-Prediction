"""Render the combined study's schematic, joint outcomes and paired effects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "combined_results_v1"))
import render as review
from combined_style import (AXES, BAND, HEADING_X, LABEL_SIZE, REFERENCE, RIGHT, TICK_SIZE,
                            TITLE_SIZE)


ACCURACY = review.ACCURACY
RESIDUAL = review.CONSISTENCY
CONDITIONS = review.CONDITIONS
MODELS = review.MODELS
SEEDS = review.SEEDS
INK = review.INK
SECONDARY = review.SECONDARY_INK
PURPLE = review.PURPLE
GRID = review.GRID
ALL_ENDPOINTS = "all_seven_equal_weight_pct_mae"


def load_all_endpoints(scores: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((HERE / "all_endpoint_manifest.json").read_text(encoding="utf-8"))
    if manifest["score_manifest_sha256"] != review.sha256(scores / "manifest.json"):
        raise ValueError("All-endpoint averages must use the same frozen score tables")
    for name, expected in manifest["files"].items():
        if review.sha256(HERE / name) != expected:
            raise ValueError(f"All-endpoint summary checksum mismatch: {name}")
    means = pd.read_csv(HERE / "all_endpoint_summary.csv").set_index("backbone", verify_integrity=True)
    seeds = pd.read_csv(HERE / "all_endpoint_seeds.csv").set_index(["backbone", "seed"], verify_integrity=True)
    expected = pd.MultiIndex.from_product([MODELS, SEEDS], names=["backbone", "seed"])
    if set(means.index) != set(MODELS) or set(seeds.index) != set(expected):
        raise ValueError("All-endpoint summaries must contain all three models and nine paired seeds")
    means, seeds = means.reindex(MODELS), seeds.reindex(expected)
    if not np.allclose(seeds["value_pct"].groupby(level="backbone").mean().reindex(MODELS),
                       means["mean_pct"], rtol=0, atol=1e-12):
        raise ValueError("All-endpoint mean must average the paired seed changes")
    return manifest, means, seeds


def load_triad_display(scores: Path, residual_seeds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((HERE / "triad_display_manifest.json").read_text(encoding="utf-8"))
    if manifest["score_manifest_sha256"] != review.sha256(scores / "manifest.json"):
        raise ValueError("Normalised triad display must use the same frozen score tables")
    for name, expected in manifest["files"].items():
        if review.sha256(HERE / name) != expected:
            raise ValueError(f"Triad display checksum mismatch: {name}")
    data = pd.read_csv(HERE / "triad_display_seed_metrics.csv", dtype={"cell": str}).set_index(
        ["backbone", "cell", "seed"], verify_integrity=True)
    if set(data.index) != set(residual_seeds.index):
        raise ValueError("Triad display must contain all 108 model/condition/seed coordinates")
    data = data.reindex(residual_seeds.index)
    if not np.allclose(data[["t_half_mae", "Vd_mae", "CL_mae"]].mean(axis=1),
                       residual_seeds["triad_mae_log10"], rtol=0, atol=1e-12):
        raise ValueError("The raw triad scores must retain the original primary metric")
    if not np.allclose(data["all_structures_mean_abs_residual_log10"],
                       residual_seeds["mean_abs_residual_log10"], rtol=0, atol=1e-12):
        raise ValueError("The reference residual must match the original 327-structure result")
    if not np.isfinite(data["mean_abs_residual_log10"]).all():
        raise ValueError("The matched 242-structure residual must be defined for every seed")
    return data.groupby(level=["backbone", "cell"], sort=False).mean(), data


def joint_coordinates(
    source: pd.DataFrame, *, value: str, keys: list[str], expected: pd.MultiIndex,
) -> pd.DataFrame:
    filter_rows = source["cell"].isin(CONDITIONS) & source["metric"].isin((ACCURACY, RESIDUAL))
    source = source.loc[filter_rows]
    counts = source["metric"].map({ACCURACY: 242, RESIDUAL: 327})
    if not source["n_structures"].eq(counts).all():
        raise ValueError("Joint accuracy and consistency use 242 and 327 structures, respectively")
    coords = source.pivot(index=keys, columns="metric", values=value)
    if set(coords.index) != set(expected) or not np.isfinite(coords.to_numpy()).all():
        raise ValueError("Joint outcomes must cover all plotted backbone/condition combinations")
    return coords.reindex(expected)[[ACCURACY, RESIDUAL]].rename(
        columns={ACCURACY: "triad_mae_log10", RESIDUAL: "mean_abs_residual_log10"},
    )


def all_endpoint_coordinates(source: pd.DataFrame, residual_seeds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = {"t_half_mae": 316, "Vd_mae": 259, "CL_mae": 246, "Cmax_mae": 190,
              "Tmax_mae": 249, "F_mae": 172, "PPB_mae": 292}
    rows = source.loc[source["cell"].isin(CONDITIONS) & source["metric"].isin(counts)]
    if not rows["n_structures"].eq(rows["metric"].map(counts)).all():
        raise ValueError("All-endpoint coordinates require every endpoint's full observed-label mask")
    values = rows.pivot(index=["backbone", "cell", "seed"], columns="metric", values="value")
    if set(values.index) != set(residual_seeds.index) or set(values.columns) != set(counts):
        raise ValueError("All-endpoint coordinates require seven endpoints for all 108 model/cell/seeds")
    values = values.reindex(residual_seeds.index)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("All seven endpoint scores must be defined for every seed")
    result = residual_seeds[["mean_abs_residual_log10"]].copy()
    result["all_endpoint_error_change_pct"] = np.nan
    for backbone in MODELS:
        model = values.xs(backbone, level="backbone")
        baseline_mean = model.xs("00", level="cell").mean(axis=0)
        if not baseline_mean.gt(0).all():
            raise ValueError("Normalising endpoint baseline MAEs must be positive")
        # Normalise absolute per-seed errors to fixed baseline means, preserving
        # seed variability in the baseline rather than forcing every seed to zero.
        normalised = 100 * (model.div(baseline_mean).mean(axis=1) - 1)
        model_mask = result.index.get_level_values("backbone") == backbone
        result.loc[model_mask, "all_endpoint_error_change_pct"] = normalised.to_numpy()
    summary = result.groupby(level=["backbone", "cell"], sort=False).mean()
    if not np.allclose(summary.xs("00", level="cell")["all_endpoint_error_change_pct"], 0, atol=1e-12):
        raise ValueError("The mean structure-only point must have zero normalised error change")
    return summary, result


def load_data(scores: Path) -> tuple:
    manifest, frames = review.load_scores(scores)
    _, _, _, means, seeds = review.extract(frames)
    means, seeds = review.percent_changes(means, seeds)
    joint_keys = ["backbone", "cell"]
    cells = pd.MultiIndex.from_product([MODELS, CONDITIONS], names=joint_keys)
    joint_means = joint_coordinates(frames["metrics"], value="mean", keys=joint_keys, expected=cells)
    seed_keys = [*joint_keys, "seed"]
    seed_cells = pd.MultiIndex.from_product([MODELS, CONDITIONS, SEEDS], names=seed_keys)
    joint_seeds = joint_coordinates(frames["seed_metrics"], value="value", keys=seed_keys, expected=seed_cells)
    pooled = joint_seeds.groupby(level=joint_keys).mean().reindex(cells)
    if not np.allclose(pooled, joint_means, rtol=0, atol=1e-12):
        raise ValueError("Joint means must average the same matched seed scores in both metrics")
    for metric, coordinate in ((ACCURACY, "triad_mae_log10"), (RESIDUAL, "mean_abs_residual_log10")):
        for cell in ("10", "11"):
            plotted = joint_means.xs(cell, level="cell")[coordinate].reindex(MODELS)
            baseline = means.xs(metric, level="metric")[f"mean_{cell}"].reindex(MODELS)
            if not np.allclose(plotted, baseline, rtol=0, atol=1e-12):
                raise ValueError("Absolute outcomes and primary contrasts require the same cell means")
    all_joint_means, all_joint_seeds = all_endpoint_coordinates(frames["seed_metrics"], joint_seeds)
    for backbone in MODELS:
        contrasts = frames["contrasts"]
        endpoints = ["t_half_mae", "Vd_mae", "CL_mae", "Cmax_mae", "Tmax_mae", "F_mae", "PPB_mae"]
        archived = contrasts.loc[contrasts["backbone"].eq(backbone)
                                 & contrasts["contrast"].eq("11_minus_00")
                                 & contrasts["metric"].isin(endpoints), "relative_change_pct"]
        if len(archived) != 7 or not np.isclose(archived.mean(),
                all_joint_means.loc[(backbone, "11"), "all_endpoint_error_change_pct"], rtol=0, atol=1e-12):
            raise ValueError("The seven-endpoint joint mean must match the archived endpoint contrasts")
    return manifest, joint_means, joint_seeds, means, seeds, all_joint_means, all_joint_seeds


def draw_panel_heading(fig: plt.Figure, letter: str, label: str, x: float, y: float, size: int = 13) -> None:
    fig.text(x, y, letter, fontsize=size, fontweight="bold", va="baseline")
    fig.text(x + .026, y, label, fontsize=size, fontweight="bold", va="baseline")


def draw_paired_supplement(
    fig: plt.Figure, means: pd.DataFrame, seeds: pd.DataFrame,
    all_means: pd.DataFrame, all_seeds: pd.DataFrame, cohort_summary: pd.DataFrame,
) -> None:
    fig.text(HEADING_X, .94, "S5  Adding the loss to ionisation-informed models", fontsize=11, fontweight="bold")
    fig.text(HEADING_X, .888, "Ionisation + loss versus ionisation", fontsize=LABEL_SIZE, color=SECONDARY)
    handles = [
        Line2D([], [], linestyle="none", marker="o", color=AXES, markersize=3, label="Paired seed"),
        Line2D([], [], linestyle="none", marker="D", color=SECONDARY, markersize=4.5, label="Mean"),
        Line2D([], [], color=SECONDARY, linewidth=2.5, label="Exploratory 95% CI"),
        Line2D([], [], color=SECONDARY, linewidth=.85, label="Adjusted 99.17% CI (primary)"),
    ]
    fig.legend(handles=handles, ncols=4, frameon=False, fontsize=TICK_SIZE,
               loc="center", bbox_to_anchor=(.55, .825), columnspacing=1.4, handlelength=1.6)
    settings = (
        ("A", "Prediction error", ACCURACY, (-5.7, 6.7), [-5, -2.5, 0, 2.5, 5]),
        ("B", "Internal disagreement (n = 327)", RESIDUAL, (-71, 4), [-60, -40, -20, 0]),
    )
    left, right = .235, .918
    gs = fig.add_gridspec(1, 2, left=left, right=right, bottom=.234, top=.735, wspace=.24)
    offsets = np.linspace(-.14, .14, len(SEEDS))
    shared = None
    for column, (letter, label, metric, limits, ticks) in enumerate(settings):
        ax = fig.add_subplot(gs[column], sharey=shared)
        shared = ax if shared is None else shared
        heading_x = HEADING_X if column == 0 else ax.get_position().x0
        fig.text(heading_x, .762, letter, fontsize=10.5, fontweight="bold")
        fig.text(heading_x + .032, .762, label, fontsize=TITLE_SIZE, fontweight="bold")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID, linewidth=.75)
        ax.axvline(0, color=AXES, linewidth=.75, linestyle=REFERENCE, zorder=1)
        ax.spines["left"].set_visible(False)
        ax.set_xlim(*limits)
        ax.set_ylim(2.48, -.52)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}%"))
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=TICK_SIZE)
        ax.axhspan(.48, 1.48, color=BAND, linewidth=0, zorder=0)
        for y, backbone in enumerate(MODELS):
            if column == 0:
                row_labels = (
                    (y - .33, MODELS[backbone], "bold"),
                    (y - .09, "Seven endpoints", "normal"),
                    (y + .25, "Triad (n = 242)", "normal"),
                )
                for row_y, row_label, weight in row_labels:
                    ax.text(-.025, row_y, row_label, transform=ax.get_yaxis_transform(), ha="right",
                            va="center", fontsize=TICK_SIZE, fontweight=weight)
            series = [(means.loc[backbone, metric],
                       seeds["value_pct"].xs((backbone, metric), level=("backbone", "metric")).reindex(SEEDS),
                       y + (.25 if column == 0 else 0), True)]
            if column == 0:
                series.insert(0, (all_means.loc[backbone],
                                  all_seeds["value_pct"].xs(backbone, level="backbone").reindex(SEEDS),
                                  y - .09, False))
            for stats, pairs, position, primary in series:
                spread = offsets * (.40 if column == 0 else 1)
                ax.scatter(pairs, position + spread, color=PURPLE, alpha=.50, s=8, linewidths=0, zorder=2)
                if primary:
                    ax.hlines(position, stats["adjusted_low_pct"], stats["adjusted_high_pct"],
                              color=PURPLE, lw=.85, zorder=3)
                ax.hlines(position, stats["crossed95_low_pct"], stats["crossed95_high_pct"],
                          color=PURPLE, lw=2.5, zorder=4)
                ax.scatter(stats["mean_pct"], position, marker="D", s=25, color=PURPLE,
                           edgecolor="white", lw=.6, zorder=5)
                value = f"{stats['mean_pct']:+.1f}%".replace("-", "−")
                ax.text(1.025, position, value, transform=ax.get_yaxis_transform(), color=PURPLE,
                        ha="left", va="center", fontsize=TICK_SIZE)
    fig.text((left + right) / 2, .127, "Change relative to ionisation only (%)", ha="center",
             fontsize=LABEL_SIZE)
    subset = cohort_summary.xs(242, level="n_structures").reindex(MODELS)
    full = cohort_summary.xs(327, level="n_structures").reindex(MODELS)
    reductions = lambda frame: ", ".join(f"{-v:.1f}%" for v in frame["both_minus_ion_pct"])
    footer = ("Negative changes indicate lower error or closer agreement.\n"
              f"Disagreement reductions in model order: {reductions(subset)} (242 complete triads)\n"
              f"and {reductions(full)} (all 327 structures).")
    fig.text(HEADING_X, .023, footer, fontsize=TICK_SIZE, color=SECONDARY, ha="left")


def draw_normalised_joint(
    fig: plt.Figure, summary: pd.DataFrame, seeds: pd.DataFrame, *,
    letter: str, title: str, xcoord: str, limits: tuple[float, float], note: str,
    consistency_n: int,
) -> None:
    draw_panel_heading(fig, letter, title, .045, .93)
    handles = [Line2D([], [], linewidth=2.1, color=color, label=label)
               for label, color in CONDITIONS.values()]
    handles += [Line2D([], [], linestyle="none", marker="o", markersize=3.8,
                       color="#93A0A9", label="Seed"),
                Line2D([], [], linestyle="none", marker="D", markersize=6.3,
                       color=SECONDARY, label="Mean")]
    fig.legend(handles=handles, ncols=6, frameon=False, fontsize=9, loc="center",
               bbox_to_anchor=(.553, .83), columnspacing=1.55, handletextpad=.44, handlelength=1.4)
    gs = fig.add_gridspec(1, 3, left=.106, right=.978, bottom=.18, top=.68, wspace=.19)
    shared = None
    for column, (backbone, label) in enumerate(MODELS.items()):
        ax = fig.add_subplot(gs[column], sharex=shared, sharey=shared)
        shared = ax if shared is None else shared
        ax.set_title(label, fontweight="bold", fontsize=11, pad=8)
        ax.set_axisbelow(True)
        ax.grid(color=GRID, linewidth=.75)
        ax.axvline(0, color="#A1ABB3", linewidth=.9, linestyle=(0, (4, 3)), zorder=1)
        ax.set_xlim(*limits)
        ax.set_xticks(np.arange(5 * np.ceil(limits[0] / 5), limits[1] + .01, 5))
        ax.set_ylim(.045, .173)
        ax.set_yticks([.06, .09, .12, .15])
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}%"))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
        ax.tick_params(axis="both", labelsize=9)
        if column == 0:
            ax.set_ylabel(f"Mean absolute log₁₀ residual\n({consistency_n} predicted triads)", fontsize=10, labelpad=7)
        else:
            ax.tick_params(axis="y", labelleft=False)
        model_summary = summary.xs(backbone, level="backbone")
        model_seeds = seeds.xs(backbone, level="backbone")
        for condition, (_, color) in CONDITIONS.items():
            points = model_seeds.xs(condition, level="cell")
            ax.scatter(points[xcoord], points["mean_abs_residual_log10"], s=16, color=color,
                       alpha=.62, edgecolor="white", linewidth=.3, zorder=2)
        for condition, (_, color) in CONDITIONS.items():
            point = model_summary.loc[condition]
            ax.scatter(point[xcoord], point["mean_abs_residual_log10"], s=57, marker="D",
                       color=color, edgecolor="white", linewidth=.9, zorder=4)
    fig.text(.543, .075, "Average endpoint error (% change from structure-only mean)",
             fontsize=10, ha="center")
    fig.text(.543, .025, note + " Lower left is better.",
             fontsize=9, ha="center", color=SECONDARY)


def export_paired_data(
    destination: Path,
    effects: pd.DataFrame, effect_seeds: pd.DataFrame,
    all_means: pd.DataFrame, all_seeds: pd.DataFrame,
) -> None:
    cols = ["mean_10", "mean", "mean_pct", "crossed95_low", "crossed95_high", "adjusted_level",
            "adjusted_low", "adjusted_high", "crossed95_low_pct", "crossed95_high_pct",
            "adjusted_low_pct", "adjusted_high_pct"]
    effects = effects[cols].reset_index()
    effect_seeds = effect_seeds[["mean_10", "value", "value_pct"]].reset_index()
    panels = {ACCURACY: "S5A", RESIDUAL: "S5B"}
    all_data = pd.concat([
        effects.assign(panel=effects["metric"].map(panels), plot_kind="summary"),
        effect_seeds.assign(panel=effect_seeds["metric"].map(panels), plot_kind="seed"),
        all_means.reset_index().assign(panel="S5A", plot_kind="summary", metric=ALL_ENDPOINTS),
        all_seeds.reset_index().assign(panel="S5A", plot_kind="seed", metric=ALL_ENDPOINTS),
    ], ignore_index=True)
    first = ["panel", "plot_kind", "backbone"]
    all_data[[*first, *(col for col in all_data.columns if col not in first)]].to_csv(
        destination / "S5_plot_data.csv", index=False,
    )


CAPTION = """Combining ionisation inputs with a mass-balance loss

a. Each model predicts half-life, volume of distribution, clearance, peak concentration (Cmax), time to peak concentration (Tmax), bioavailability (F) and plasma protein binding (PPB). Paracetamol illustrates the structure input. The ionisation condition adds one of four categories defined by the pH 7.4 thresholds on acidic and basic pKa. Acidic pKa below 7.4 meets the acidic criterion, and basic pKa above 7.4 meets the basic criterion. The four empty boxes show the one-hot input dimensions. The loss adds a squared penalty on the log10 mass-balance residual for predicted half-life, volume of distribution and clearance. The two loss-enabled conditions use the same positive penalty strength within each backbone and outer fold, selected using only inner validation predictions. The ChemBERTa encoder is frozen and its multilayer prediction head is fitted.

b,c. Four-condition ablation of ionisation input and mass-balance loss across GraphConv, Chemprop and frozen ChemBERTa + MLP. Each point pairs prediction error and internal disagreement from one fitted seed pooled across five grouped held-out outer folds. Small points show nine matched seeds and diamonds their means. Structure only uses neither addition. All six plots share their x and y limits.

The horizontal axis uses the same construction in both rows: for each model, condition and seed, divide each endpoint MAE by that endpoint's fixed nine-seed mean structure-only MAE, average these ratios equally over endpoints, subtract one and multiply by 100. Zero is the mean structure-only error and negative values indicate lower error. Individual baseline seeds retain their variability; this display is centred on the fixed baseline mean rather than subtracting each seed's baseline. Because each model has its own baseline, horizontal coordinates compare relative effects within each model rather than absolute accuracy between architectures.

b. Both axes use the same 242 structures with complete source-defined observed triads. The accuracy summary averages three normalised endpoint errors: half-life, Vd and CL. Each endpoint's MAE is measured in log10 units before normalisation. This display gives equal weight to relative endpoint errors. Internal disagreement is evaluated on the predictions for those same 242 structures. The prespecified raw triad macro MAE, which averages the three raw log10 MAEs before normalisation, and the original full-cohort consistency analysis remain in Supplementary Figures S4-S5.

c. The seven-endpoint summary gives equal weight to relative endpoint errors using all observed labels per endpoint: half-life 316, Vd 259, CL 246, Cmax 190, Tmax 249, F 172 and PPB 292. Five endpoints use log10 MAE; F and PPB use percentage-point MAE with zero labels retained. The endpoint-specific masks differ from the complete-triad mask in b.

Both rows use the same definition of internal disagreement: mean absolute log10 triad residual, calculated as abs(log10(half-life in hours) - log10(Vd in litres) + log10(CL in litres per hour) - log10(ln 2)). This measures internal agreement of the three coupled predictions. The population is the matched 242-structure subset in b and the full 327-structure cohort in c. Paired effects of adding the loss to ionisation-informed models, including the original adjusted primary intervals, the exploratory seven-endpoint interval and a comparison of consistency reductions on 242 versus 327 structures, are provided in Supplementary Figure S5. Training, data splits and original primary analyses remain unchanged.
"""

S5_CAPTION = """S5. Paired effects of adding the mass-balance loss to ionisation-informed models.

A. “Seven endpoints” shows the equal-weight average of seven endpoint percentage MAE changes. Each endpoint uses all observed labels: half-life 316, Vd 259, CL 246, Cmax 190, Tmax 249, F 172 and PPB 292, retaining zero F/PPB labels. Five endpoints use log10 MAE and F/PPB use percentage-point MAE. Each endpoint's paired MAE difference (both minus ionisation only) is divided by its fixed ionisation-only nine-seed mean MAE, multiplied by 100, and the seven changes are averaged within each seed. “Triad” shows the prespecified percentage change in raw triad macro log10 MAE on the 242 complete same-row triads, divided by the fixed ionisation-only triad mean. This triad macro metric averages raw log10 endpoint MAEs before normalisation; it therefore retains the original primary weighting, whereas the main figure's triad display averages normalised endpoint MAEs.

B. Percentage change in mean absolute log10 triad residual on all 327 structures when adding the loss to ionisation-informed models. Estimates and interval bounds use the fixed ionisation-only mean across seeds as the denominator.

Dots show nine paired seed changes; diamonds show means. Thick lines show exploratory 95% percentile intervals. Thin lines show 99.1667% Bonferroni-adjusted intervals for the six prespecified primary comparisons (triad accuracy and disagreement across three models). The seven-endpoint summary is exploratory and has 95% intervals only. The 10,000 crossed bootstrap draws jointly resample whole structure groups and whole fitted seed trajectories across endpoints, preserving endpoint dependence and using each endpoint's resampled observed-label count. Normalising denominators remain fixed. Training data, folds, recipes and selection procedure remain fixed.
"""


def render(scores: Path, output: Path) -> None:
    from schematic import draw_schematic

    manifest, joint_means, joint_seeds, effects, effect_seeds, all_joint_means, all_joint_seeds = load_data(scores)
    all_manifest, all_means, all_seeds = load_all_endpoints(scores)
    triad_means, triad_seeds = load_triad_display(scores, joint_seeds)
    cohort_summary = pd.read_csv(HERE / "residual_cohort_summary.csv").set_index(
        ["backbone", "n_structures"], verify_integrity=True)
    if all_manifest["campaign_id"] != manifest["campaign_id"]:
        raise ValueError("All plotted results must belong to the same campaign")
    output.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.3, 10.6), facecolor="white")
    top, middle, bottom = fig.subfigures(3, 1, height_ratios=[3.4, 3.6, 3.6], hspace=0)
    draw_panel_heading(top, "a", "Combining ionisation inputs with a mass-balance loss", .045, .91, size=15)
    schematic = top.add_axes([.073, .06, .905, .82])
    draw_schematic(schematic)
    schematic.set(xlim=(0, 112), ylim=(0, 31))
    schematic.set_axis_off()
    all_x = np.r_[triad_seeds["triad_error_change_pct"], all_joint_seeds["all_endpoint_error_change_pct"]]
    limits = (np.floor(all_x.min() - .6), np.ceil(all_x.max() + .6))
    draw_normalised_joint(middle, triad_means, triad_seeds, letter="b",
                          title="Triad accuracy and internal disagreement", xcoord="triad_error_change_pct",
                          limits=limits, note="Half-life, Vd and CL; the same 242 structures on both axes.",
                          consistency_n=242)
    draw_normalised_joint(bottom, all_joint_means, all_joint_seeds, letter="c",
                          title="Seven-endpoint accuracy and internal disagreement", xcoord="all_endpoint_error_change_pct",
                          limits=limits, note="All seven endpoints; all available labels per endpoint.",
                          consistency_n=327)
    review.save(fig, output, "combined_main")
    pd.concat([
        triad_means.reset_index().assign(panel="b", plot_kind="summary"),
        triad_seeds.reset_index().assign(panel="b", plot_kind="seed"),
        all_joint_means.reset_index().assign(panel="c", plot_kind="summary"),
        all_joint_seeds.reset_index().assign(panel="c", plot_kind="seed"),
    ], ignore_index=True).to_csv(output / "plot_data.csv", index=False)
    supplement = output / "supplementary"
    supplement.mkdir(parents=True, exist_ok=True)
    paired_fig = plt.figure(figsize=(11.3, 4.1), facecolor="white")
    draw_paired_supplement(paired_fig, effects, effect_seeds, all_means, all_seeds, cohort_summary)
    review.save(paired_fig, supplement, "S5_paired_effects")
    export_paired_data(supplement, effects, effect_seeds, all_means, all_seeds)
    cohort_summary.reset_index().to_csv(supplement / "S5_cohort_sensitivity.csv", index=False)
    sensitivity_lines = ["\nComplete-triad sensitivity (mean residual, both minus ionisation only):"]
    for backbone, label in MODELS.items():
        matched = cohort_summary.loc[(backbone, 242), "both_minus_ion_pct"]
        full = cohort_summary.loc[(backbone, 327), "both_minus_ion_pct"]
        sensitivity_lines.append(f"{label}: {matched:+.2f}% on the 242 complete-triad structures; {full:+.2f}% on all 327.")
    sensitivity_lines.append("These are descriptive sensitivity estimates; the plotted primary intervals above remain the full-cohort analyses. Each population uses its own fixed ionisation-only mean as denominator.")
    (supplement / "S5_caption.txt").write_text(S5_CAPTION + "\n".join(sensitivity_lines) + "\n", encoding="utf-8")
    (output / "caption.txt").write_text(CAPTION, encoding="utf-8")
    paths = ["combined_main.png", "combined_main.pdf", "combined_main.svg", "plot_data.csv", "caption.txt"]
    source_manifest = {
        "campaign_id": manifest["campaign_id"],
        "score_manifest_sha256": review.sha256(scores / "manifest.json"),
        "verified_score_files": manifest["files"],
        "source_files": {
            "render.py": review.sha256(Path(__file__)),
            "schematic.py": review.sha256(HERE / "schematic.py"),
            "all_endpoints.py": review.sha256(HERE / "all_endpoints.py"),
            "all_endpoint_manifest.json": review.sha256(HERE / "all_endpoint_manifest.json"),
            "triad_display.py": review.sha256(HERE / "triad_display.py"),
            "triad_display_manifest.json": review.sha256(HERE / "triad_display_manifest.json"),
            "../combined_results_v1/render.py": review.sha256(Path(review.__file__)),
        },
        "files": {name: review.sha256(output / name) for name in paths},
        "display_x_limits": [float(v) for v in limits],
        "supplementary_paired_files": {name: review.sha256(supplement / name) for name in
                                      ("S5_paired_effects.png", "S5_paired_effects.pdf", "S5_paired_effects.svg",
                                       "S5_plot_data.csv", "S5_caption.txt", "S5_cohort_sensitivity.csv")},
    }
    (output / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=review.SCORES)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    render(args.scores, args.output)
