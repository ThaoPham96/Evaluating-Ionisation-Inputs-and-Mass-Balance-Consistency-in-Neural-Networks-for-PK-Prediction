"""Reproduce the combined-study overview and the primary paired trade-off."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator, PercentFormatter, StrMethodFormatter
import numpy as np
import pandas as pd

from combined_style import (CDF_STYLES, CONDITIONS, GRID, HEADING_X, INK, MODELS, MUTED,
                            RIGHT, TICK_SIZE, WIDTH, WSPACE, configure, panel_heading)

HERE = Path(__file__).resolve().parent
SCORES = HERE.parents[1] / "outputs" / "scores"
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909)
ACCURACY = "triad_macro_log10_mae"
CONSISTENCY = "mean_abs_log10_residual"
SECONDARY_INK = MUTED
PURPLE = CONDITIONS["11"][1]
X_CDF_MAX = 5.0

configure()


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def load_scores(directory: Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    required = ("metrics", "seed_metrics", "disagreement_cdf", "contrasts", "seed_contrasts")
    if (manifest.get("status") != "complete" or manifest.get("final_rows") != 35_316
            or manifest.get("fit_counts") != {"inner": 1050, "final": 540}
            or manifest.get("bootstrap_draws") != 10_000
            or any(f"{name}.csv" not in manifest["files"] for name in required)):
        raise ValueError("A verified, complete combined-study score export is required")
    for filename, expected in manifest["files"].items():
        if sha256(directory / filename) != expected:
            raise ValueError(f"Score-table checksum mismatch for {filename}")
    frames = {name: pd.read_csv(directory / f"{name}.csv", dtype={"cell": str}, float_precision="round_trip")
              for name in required}
    return manifest, frames


def extract(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, ...]:
    metrics = frames["metrics"]
    seed_metrics = frames["seed_metrics"]
    cdf = frames["disagreement_cdf"]
    contrasts = frames["contrasts"]
    seed_contrasts = frames["seed_contrasts"]
    cells = pd.MultiIndex.from_product([MODELS, CONDITIONS], names=["backbone", "cell"])
    seeds = pd.MultiIndex.from_product([MODELS, CONDITIONS, SEEDS], names=["backbone", "cell", "seed"])
    primary = pd.MultiIndex.from_product([MODELS, (ACCURACY, CONSISTENCY)], names=["backbone", "metric"])
    primary_seeds = pd.MultiIndex.from_product([MODELS, (ACCURACY, CONSISTENCY), SEEDS],
                                               names=["backbone", "metric", "seed"])
    metrics = metrics.loc[metrics["metric"].eq(ACCURACY) & metrics["cell"].isin(CONDITIONS)]
    seed_metrics = seed_metrics.loc[seed_metrics["metric"].eq(ACCURACY) & seed_metrics["cell"].isin(CONDITIONS)]
    cdf = cdf.loc[cdf["cell"].isin(CONDITIONS)].copy()
    contrasts = contrasts.loc[contrasts["family"].eq("primary") & contrasts["contrast"].eq("11_minus_10")]
    seed_contrasts = seed_contrasts.loc[seed_contrasts["family"].eq("primary")
                                        & seed_contrasts["contrast"].eq("11_minus_10")]
    selected = []
    for data, index in ((metrics, cells), (seed_metrics, seeds), (contrasts, primary),
                        (seed_contrasts, primary_seeds)):
        data = data.set_index(index.names, verify_integrity=True)
        if set(data.index) != set(index):
            raise ValueError("Score tables lack complete data for the plotted cells and seeds")
        selected.append(data.reindex(index))
    metrics, seed_metrics, contrasts, seed_contrasts = selected
    mean_seed_accuracy = seed_metrics["value"].groupby(level=["backbone", "cell"]).mean().reindex(cells)
    mean_seed_delta = seed_contrasts["value"].groupby(level=["backbone", "metric"]).mean().reindex(primary)
    if (not metrics["n_structures"].eq(242).all() or not metrics["n_seeds_defined"].eq(9).all()
            or not np.allclose(metrics["mean"], mean_seed_accuracy, rtol=0, atol=1e-12)
            or not np.allclose(contrasts["mean"], mean_seed_delta, rtol=0, atol=1e-12)
            or not np.allclose(contrasts["mean_11"] - contrasts["mean_10"], contrasts["mean"], rtol=0, atol=1e-12)
            or not np.allclose(contrasts["adjusted_level"], 1 - 0.05 / 6, rtol=0, atol=1e-12)
            or not contrasts["valid_draws"].eq(10_000).all()
            or not contrasts["mean_10"].gt(0).all()):
        raise ValueError("Plotted means, population counts or primary interval specifications disagree")
    if (set(cdf.groupby(["backbone", "cell"]).groups) != set(cells)
            or not cdf["n_structures_per_seed"].eq(327).all() or not cdf["n_seeds"].eq(9).all()):
        raise ValueError("Disagreement curves lack the expected structures, models, cells or seeds")
    cdf = cdf.sort_values(["backbone", "cell", "disagreement_fold"])
    for _, curve in cdf.groupby(["backbone", "cell"]):
        support = curve["disagreement_fold"].to_numpy()
        probabilities = curve["mean_seed_cdf"].to_numpy()
        if (not np.isfinite(support).all() or support[0] < 1 or np.any(np.diff(support) <= 0)
                or probabilities[0] <= 0 or probabilities[-1] != 1 or np.any(np.diff(probabilities) <= 0)):
            raise ValueError("Invalid disagreement CDF")
    return metrics, seed_metrics, cdf, contrasts, seed_contrasts


def save(fig: plt.Figure, destination: Path, stem: str) -> None:
    fig.savefig(destination / f"{stem}.png", dpi=300, facecolor="white")
    for fmt in ("pdf", "svg"):
        fig.savefig(destination / f"{stem}.{fmt}", facecolor="white")
    plt.close(fig)


def overview(metrics: pd.DataFrame, seeds: pd.DataFrame, cdf: pd.DataFrame, destination: Path) -> tuple[int, int]:
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.65), sharey="row")
    fig.subplots_adjust(left=.127, right=RIGHT, top=.755, bottom=.18, hspace=.61, wspace=WSPACE)
    fig.text(HEADING_X, .958, "Accuracy and internal consistency across four conditions", fontsize=11, weight="bold")
    fig.text(HEADING_X, .913, "Nine matched seeds and five grouped outer folds per seed", color=SECONDARY_INK)
    legend = [Line2D([0], [0], color=color, lw=1.6, linestyle=CDF_STYLES[cell],
                     label=label) for cell, (label, color) in CONDITIONS.items()]
    fig.legend(handles=legend, loc="center", bbox_to_anchor=(.55, .865), frameon=False,
               ncols=4, handlelength=2.4, columnspacing=1.4, fontsize=TICK_SIZE)
    panel_heading(fig, "A", "Triad accuracy · 242 structures", .828)
    panel_heading(fig, "B", "Triad consistency · 327 structures", .441)
    x = np.arange(len(CONDITIONS))
    offsets = np.linspace(-.13, .13, len(SEEDS))
    upper = max(metrics["crossed95_high"].max(), seeds["value"].max())
    lower = min(metrics["crossed95_low"].min(), seeds["value"].min())
    limits = (np.floor((lower - 0.008) / 0.02) * .02, np.ceil((upper + 0.004) / 0.02) * .02)
    max_omitted = 0
    denominator = -1
    for col, (backbone, name) in enumerate(MODELS.items()):
        ax = axes[0, col]
        ax.set_title(name, fontweight="bold", pad=8)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=.75)
        values = seeds["value"].xs(backbone, level="backbone").unstack("cell").reindex(index=SEEDS, columns=CONDITIONS)
        for offset, seed_values in zip(offsets, values.to_numpy()):
            ax.plot(x + offset, seed_values, color="#B1BAC1", linewidth=.65, alpha=.58, zorder=1)
        for idx, (cell, (_, color)) in enumerate(CONDITIONS.items()):
            entry = metrics.loc[backbone, cell]
            ax.scatter(idx + offsets, values[cell], s=9, color=color, alpha=.68, linewidths=0, zorder=3)
            ax.vlines(idx, entry["crossed95_low"], entry["crossed95_high"], color=color, linewidth=1.4, zorder=4)
            ax.scatter(idx, entry["mean"], color=color, marker="D", s=28, linewidths=.6, edgecolors="white", zorder=5)
        ax.set_xlim(-.35, 3.35)
        ax.set_ylim(*limits)
        ax.set_xticks(x, ["Structure\nonly", "Ionisation", "Loss", "Both"])
        ax.tick_params(axis="x", labelsize=TICK_SIZE, length=0, pad=5)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
        if col == 0:
            ax.set_ylabel("Triad macro MAE (log₁₀)")

        ax = axes[1, col]
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=.75)
        for cell, (_, color) in CONDITIONS.items():
            curve = cdf.loc[cdf["backbone"].eq(backbone) & cdf["cell"].eq(cell)]
            support = curve["disagreement_fold"].to_numpy()
            probability = curve["mean_seed_cdf"].to_numpy()
            displayed = support <= X_CDF_MAX
            cumulative = probability[displayed][-1]
            denominator = int(curve["n_structures_per_seed"].iat[0] * curve["n_seeds"].iat[0])
            max_omitted = max(max_omitted, round((1 - cumulative) * denominator))
            ax.step(np.r_[1.0, support[displayed], X_CDF_MAX], np.r_[0.0, probability[displayed], cumulative],
                    where="post", color=color, linewidth=1.6, linestyle=CDF_STYLES[cell])
        ax.set_xscale("log")
        ax.set_xlim(1.0, X_CDF_MAX)
        ax.set_ylim(0, 1.035)
        ax.xaxis.set_major_locator(FixedLocator([1, 1.5, 2, 3, 5]))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_xlabel("Fold disagreement")
        if col == 0:
            ax.set_ylabel("Predictions within\nthreshold")
    fig.text(HEADING_X, .067, "A: dots show paired seeds, diamonds show means and bars show exploratory 95% intervals.",
             fontsize=TICK_SIZE, color=SECONDARY_INK)
    fig.text(HEADING_X, .026, f"B: curves show 1–{X_CDF_MAX:g}-fold. At most {max_omitted} of {denominator:,} "
             f"predictions per curve ({max_omitted / denominator:.2%}) lie above {X_CDF_MAX:g}-fold.",
             fontsize=TICK_SIZE, color=SECONDARY_INK)
    save(fig, destination, "combined_overview")
    return max_omitted, denominator


def percent_changes(means: pd.DataFrame, seeds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    means = means.copy()
    seeds = seeds.join(means["mean_10"], on=["backbone", "metric"])
    for field in ("mean", "crossed95_low", "crossed95_high", "adjusted_low", "adjusted_high"):
        means[f"{field}_pct"] = 100 * means[field] / means["mean_10"]
    seeds["value_pct"] = 100 * seeds["value"] / seeds["mean_10"]
    return means, seeds


def tradeoff(means: pd.DataFrame, seeds: pd.DataFrame, destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.35), sharey=True)
    fig.subplots_adjust(left=.212, right=.977, top=.661, bottom=.216, wspace=.27)
    fig.text(.044, .915, "Adding the loss to ionisation-informed models", fontsize=17, weight="bold")
    fig.text(.044, .843, "Paired percent change from the ionisation-only mean", color=SECONDARY_INK)
    legend = [
        Line2D([0], [0], marker="o", linestyle="none", color="#A291B2", markersize=4, label="Seed"),
        Line2D([0], [0], marker="D", linestyle="none", color=PURPLE, markersize=6, label="Mean"),
        Line2D([0], [0], color=PURPLE, linewidth=3, label="Exploratory 95% CI"),
        Line2D([0], [0], color=PURPLE, linewidth=1, label="Adjusted 99.17% CI"),
    ]
    fig.legend(handles=legend, loc="center right", bbox_to_anchor=(.977, .802), frameon=False, ncols=4,
               fontsize=9, columnspacing=1.3, handlelength=1.7, borderaxespad=0)
    offsets = np.linspace(-.13, .13, len(SEEDS))
    positions = np.arange(len(MODELS))
    settings = (
        (ACCURACY, "a   Triad prediction error", "Mean absolute log₁₀ error · n = 242", (-5.7, 5.7), [-5, -2.5, 0, 2.5, 5]),
        (CONSISTENCY, "b   Internal disagreement", "Mean absolute log₁₀ residual · n = 327", (-71, 4), [-60, -40, -20, 0]),
    )
    for col, (ax, (metric, heading, count, limits, ticks)) in enumerate(zip(axes, settings)):
        ax.text(0, 1.16, heading, transform=ax.transAxes, fontsize=11, fontweight="bold")
        ax.text(0, 1.04, count, transform=ax.transAxes, fontsize=9, color=SECONDARY_INK)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID, linewidth=.75)
        ax.axvline(0, color="#7B8790", linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ax.spines["left"].set_visible(False)
        ax.set_xlim(*limits)
        ax.set_ylim(2.48, -.45)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}%"))
        ax.set_yticks(positions, ["GraphConv", "Chemprop", "Frozen ChemBERTa\n+ MLP"])
        ax.tick_params(axis="y", length=0, pad=11, labelsize=10)
        ax.set_xlabel("Change in error (%)" if col == 0 else "Change in disagreement (%)", labelpad=8)
        for y, backbone in enumerate(MODELS):
            stats = means.loc[backbone, metric]
            seed_values = seeds["value_pct"].xs((backbone, metric), level=("backbone", "metric")).reindex(SEEDS)
            ax.scatter(seed_values, y + offsets, color="#9880AA", alpha=.66, s=17,
                       linewidths=0, zorder=2)
            ax.hlines(y, stats["adjusted_low_pct"], stats["adjusted_high_pct"],
                      linewidth=1, color=PURPLE, zorder=3)
            ax.hlines(y, stats["crossed95_low_pct"], stats["crossed95_high_pct"],
                      linewidth=3.4, color=PURPLE, zorder=4)
            ax.scatter(stats["mean_pct"], y, marker="D", color=PURPLE, s=48, edgecolors="white",
                       linewidths=.9, zorder=5)
            label = f"{stats['mean_pct']:+.1f}%".replace("-", "−")
            ax.text(.986, y - .13, label, transform=ax.get_yaxis_transform(), va="bottom", ha="right",
                    color=PURPLE, fontsize=10, fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    fig.text(.044, .063, "Negative values indicate lower error or lower disagreement. Intervals are crossed group-and-seed bootstrap percentiles.",
             fontsize=9, color=SECONDARY_INK)
    save(fig, destination, "adding_loss_tradeoff")


def export_plot_data(
    metrics: pd.DataFrame, accuracy_seeds: pd.DataFrame, cdf: pd.DataFrame,
    contrasts: pd.DataFrame, contrast_seeds: pd.DataFrame, destination: Path,
) -> None:
    data = []
    for frame, figure, panel, kind in (
        (metrics.reset_index(), "combined_overview", "a", "summary"),
        (accuracy_seeds.reset_index(), "combined_overview", "a", "seed"),
        (cdf.assign(metric="disagreement_fold", visible_in_plot=cdf["disagreement_fold"] <= X_CDF_MAX),
         "combined_overview", "b", "cdf"),
        (contrasts.reset_index(), "adding_loss_tradeoff", None, "summary"),
        (contrast_seeds.reset_index(), "adding_loss_tradeoff", None, "seed"),
    ):
        frame = frame.copy()
        if panel is None:
            frame = frame[["backbone", "metric", "contrast", "mean_10",
                           *(["mean", "mean_pct", "crossed95_low", "crossed95_high", "adjusted_level",
                              "adjusted_low", "adjusted_high", "crossed95_low_pct", "crossed95_high_pct",
                              "adjusted_low_pct", "adjusted_high_pct"] if kind == "summary"
                             else ["seed", "value", "value_pct"])]]
        frame.insert(0, "plot_kind", kind)
        frame.insert(0, "panel", panel if panel is not None else frame["metric"].map({ACCURACY: "a", CONSISTENCY: "b"}))
        frame.insert(0, "figure", figure)
        data.append(frame)
    pd.concat(data, ignore_index=True).to_csv(destination / "plot_data.csv", index=False)


def write_caption(destination: Path, omitted: int, denominator: int) -> None:
    caption = f"""Accuracy and internal consistency across four conditions

a. The triad macro MAE averages absolute log10 errors equally across half-life, volume of distribution and clearance on 242 observed same-row triads. Each dot pools the five held-out outer folds of one seed. Lines connect the same seed across conditions. Diamonds show the mean of nine seeds with 95% crossed bootstrap intervals. Structure only (00) has neither ionisation input nor a mass-balance penalty. Ionisation (10) adds the pH 7.4 category. Loss (01) adds the mass-balance penalty. Both (11) uses the ionisation category and the same fold-specific positive penalty as 01.

b. Fold disagreement is 10 raised to the absolute log10 residual, or the larger divided by the smaller of predicted half-life and the half-life implied by ln(2) times predicted volume divided by predicted clearance. A value of one indicates exact agreement. Each curve averages empirical CDFs computed separately for nine fitted seeds on all 327 held-out structures per seed. Higher curves indicate more consistent predictions. The panel shows 1–{X_CDF_MAX:g}-fold. At most {omitted} of {denominator:,} predictions per curve ({omitted / denominator:.2%}) lie above {X_CDF_MAX:g}-fold. The full CDF support is included in plot_data.csv.

Adding the loss to ionisation-informed models

a. Change in triad macro log10 MAE on the 242 same-row observed triads. b. Change in mean absolute log10 triad residual across all 327 predicted triads. Each contrast subtracts ionisation only (10) from the combined condition (11). For each backbone and metric, all percentage estimates and interval bounds use the corresponding fixed nine-seed mean for condition 10 as the denominator. Dots give the nine paired seed differences on that scale. Diamonds show the mean. Thick intervals show exploratory 95% percentile coverage and thin intervals show 99.1667% Bonferroni-adjusted percentile coverage for the six primary comparisons. Negative values indicate lower error or lower disagreement.

Intervals cross 10,000 paired bootstrap draws of whole structure groups and whole fitted seed trajectories. They describe variation over groups and training seeds conditional on the training dataset, splits, recipes and selection procedure.\n"""
    (destination / "caption.txt").write_text(caption, encoding="utf-8")


def render(scores: Path, destination: Path) -> None:
    manifest, frames = load_scores(scores)
    metrics, accuracy_seeds, cdf, contrasts, contrast_seeds = extract(frames)
    contrasts, contrast_seeds = percent_changes(contrasts, contrast_seeds)
    destination.mkdir(parents=True, exist_ok=True)
    omitted, denominator = overview(metrics, accuracy_seeds, cdf, destination)
    tradeoff(contrasts, contrast_seeds, destination)
    export_plot_data(metrics, accuracy_seeds, cdf, contrasts, contrast_seeds, destination)
    write_caption(destination, omitted, denominator)
    source_manifest = {
        "campaign_id": manifest["campaign_id"],
        "score_manifest_sha256": sha256(scores / "manifest.json"),
        "verified_score_files": manifest["files"],
        "render_sha256": sha256(Path(__file__)),
        "cdf_view_max_fold": X_CDF_MAX,
        "cdf_max_count_above_view": omitted,
        "cdf_denominator_per_curve": denominator,
    }
    outputs = ["plot_data.csv", "caption.txt", *(f"{stem}.{fmt}"
               for stem in ("combined_overview", "adding_loss_tradeoff") for fmt in ("png", "pdf", "svg"))]
    source_manifest["files"] = {filename: sha256(destination / filename) for filename in outputs}
    (destination / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=SCORES)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    render(args.scores, args.output)
