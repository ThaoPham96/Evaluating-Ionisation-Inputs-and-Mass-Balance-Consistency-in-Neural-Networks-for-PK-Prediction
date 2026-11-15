"""Draw the paired endpoint-accuracy panel from verified score tables."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from matplotlib.figure import SubFigure
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "combined_results_v1"))
from combined_style import (AXES, BAND, CONDITIONS as ALL_CONDITIONS, ENDPOINT_COLORS, GRID, HEADING_X, INK,
                            LABEL_SIZE, LEFT, MODELS, MUTED, REFERENCE, RIGHT, TICK_SIZE,
                            TITLE_SIZE, TITLE_X, WSPACE, panel_heading)

ENDPOINTS = {
    "t_half_mae": ("Half-life", 316),
    "Vd_mae": ("Volume of distribution", 259),
    "CL_mae": ("Clearance", 246),
    "Cmax_mae": ("Peak concentration", 190),
    "Tmax_mae": ("Time to peak", 249),
    "F_mae": ("Bioavailability", 172),
    "PPB_mae": ("Protein binding", 292),
}
CONDITIONS = {cell: ALL_CONDITIONS[cell] for cell in ("10", "01", "11")}
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909)
CONTRASTS = {cell: f"{cell}_minus_00" for cell in CONDITIONS}
KEYS = ["backbone", "metric", "contrast"]
ATOL = 1e-12
AVG_METRIC = "average_7_endpoint_mae_pct"
SUMMARY_FILE = "average_summary.csv"
SEEDS_FILE = "average_seeds.csv"
MANIFEST_FILE = "average_manifest.json"


def panel_data(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = {metric: n for metric, (_, n) in ENDPOINTS.items()}
    expected = pd.MultiIndex.from_product((MODELS, ENDPOINTS, CONTRASTS.values()), names=KEYS)
    expected_seeds = pd.MultiIndex.from_product((MODELS, ENDPOINTS, CONTRASTS.values(), SEEDS), names=[*KEYS, "seed"])
    paired_tables = []
    for key, index in (("contrasts", expected), ("seed_contrasts", expected_seeds)):
        source = frames[key]
        paired = source.loc[source["metric"].isin(ENDPOINTS) & source["contrast"].isin(CONTRASTS.values())]
        paired = paired.set_index(index.names, verify_integrity=True)
        if set(paired.index) != set(index):
            raise ValueError(f"Incomplete paired endpoint coverage in {key}")
        paired = paired.reindex(index)
        n_structures = paired.index.get_level_values("metric").map(counts)
        if not paired["n_structures"].eq(n_structures).all():
            raise ValueError(f"Observed-label counts disagree in {key}")
        paired_tables.append(paired)
    means, seeds = paired_tables
    if (not means["n_seeds_defined"].eq(len(SEEDS)).all() or not means["valid_draws"].eq(10_000).all()
            or not means["baseline_00_mean"].gt(0).all()
            or not np.isfinite(means[["baseline_00_mean", "mean", "crossed95_low", "crossed95_high"]]).all().all()
            or not np.isfinite(seeds["value"]).all()):
        raise ValueError("Endpoint effects require finite paired results, nine seeds and 10,000 valid draws")
    means = means.reset_index()
    means["denominator00"] = means["baseline_00_mean"]
    scale = 100 / means["denominator00"]
    means["mean_pct"] = scale * means["mean"]
    means["low95_pct"] = scale * means["crossed95_low"]
    means["high95_pct"] = scale * means["crossed95_high"]
    means = means.set_index(KEYS)
    seeds = seeds.join(means["denominator00"], on=KEYS)
    seeds["value_pct"] = 100 * seeds["value"] / seeds["denominator00"]
    pooled_seed = seeds["value_pct"].groupby(level=KEYS).mean().reindex(means.index)
    np.testing.assert_allclose(pooled_seed, means["mean_pct"], rtol=0, atol=ATOL)

    # The joint panel centres on fixed baseline means. Its condition averages
    # agree with the paired averages here, whereas individual seed effects differ.
    absolute = frames["seed_metrics"]
    absolute = absolute.loc[absolute["metric"].isin(ENDPOINTS) & absolute["cell"].isin(["00", *CONDITIONS])]
    if not absolute["n_structures"].eq(absolute["metric"].map(counts)).all():
        raise ValueError("Joint endpoint scores must use the full observed-label count")
    absolute_mae = absolute.pivot(index=["backbone", "metric", "seed"], columns="cell", values="value")
    absolute_index = pd.MultiIndex.from_product((MODELS, ENDPOINTS, SEEDS), names=["backbone", "metric", "seed"])
    if set(absolute_mae.index) != set(absolute_index) or not np.isfinite(absolute_mae).all().all():
        raise ValueError("Missing absolute endpoint seed values")
    absolute_means = absolute_mae.groupby(level=["backbone", "metric"]).mean()
    denominator = means["denominator00"]
    for cell, contrast in CONTRASTS.items():
        expected = means.xs(contrast, level="contrast").index
        fixed_base = absolute_means["00"].reindex(expected)
        np.testing.assert_allclose(denominator.xs(contrast, level="contrast"), fixed_base, rtol=0, atol=ATOL)
        joint_pct = 100 * (absolute_means[cell].reindex(expected) / fixed_base - 1)
        np.testing.assert_allclose(means.xs(contrast, level="contrast")["mean_pct"], joint_pct, rtol=0, atol=ATOL)
    return means, seeds


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def average_data(means: pd.DataFrame, seeds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((HERE / MANIFEST_FILE).read_text(encoding="utf-8"))
    scores = HERE.parents[1] / "outputs" / "scores" / "manifest.json"
    score_record = json.loads(scores.read_text(encoding="utf-8"))
    if (manifest["campaign_id"] != score_record["campaign_id"]
            or manifest["score_manifest_sha256"] != sha256(scores)
            or manifest["bootstrap_draws"] != 10_000 or manifest["ci_level"] != .95):
        raise ValueError("The average-row bootstrap must match the frozen score export")
    tables = []
    for filename, expected in (
        (SUMMARY_FILE, pd.MultiIndex.from_product((MODELS, [AVG_METRIC], CONTRASTS.values()), names=KEYS)),
        (SEEDS_FILE, pd.MultiIndex.from_product((MODELS, [AVG_METRIC], CONTRASTS.values(), SEEDS),
                                              names=[*KEYS, "seed"])),
    ):
        path = HERE / filename
        if sha256(path) != manifest["files"][filename]:
            raise ValueError(f"Average-row table checksum mismatch for {filename}")
        table = pd.read_csv(path, float_precision="round_trip").set_index(expected.names, verify_integrity=True)
        if set(table.index) != set(expected):
            raise ValueError(f"Average-row table has incomplete coverage for {filename}")
        tables.append(table.reindex(expected))
    avg_means, avg_seeds = tables
    mean_order = avg_means.droplevel("metric").index
    seed_order = avg_seeds.droplevel("metric").index
    if not np.isfinite(avg_means[["mean_pct", "low95_pct", "high95_pct"]]).all().all():
        raise ValueError("The average row requires finite means and interval bounds")
    np.testing.assert_allclose(
        avg_means["mean_pct"],
        means["mean_pct"].groupby(level=["backbone", "contrast"]).mean().reindex(mean_order),
        rtol=0, atol=ATOL,
    )
    np.testing.assert_allclose(
        avg_seeds["value_pct"],
        seeds["value_pct"].groupby(level=["backbone", "contrast", "seed"]).mean().reindex(seed_order),
        rtol=0, atol=ATOL,
    )
    return avg_means, avg_seeds


def draw_endpoint_panel(
    fig: SubFigure, frames: dict[str, pd.DataFrame], left: float = LEFT, right: float = RIGHT,
    *, descriptive_text: bool = True,
) -> pd.DataFrame:
    """Draw panel B and return exact plotted estimates and paired seed values."""
    means, seeds = panel_data(frames)
    avg_means, avg_seeds = average_data(means, seeds)
    means = pd.concat([means, avg_means])
    seeds = pd.concat([seeds, avg_seeds])
    panel_heading(fig, "B", "Accuracy across all seven endpoints" if descriptive_text else "", .974)
    if descriptive_text:
        fig.text(TITLE_X, .888, "Each addition compared with structure only",
                 fontsize=TICK_SIZE, va="center", color=MUTED)
    legend = [Line2D([], [], color=color, marker="D", lw=1.5, markersize=4.7, label=label)
              for label, color in CONDITIONS.values()]
    fig.legend(handles=legend, loc="center", bbox_to_anchor=((left + right) / 2, .888 if descriptive_text else .928),
               ncols=3, frameon=False, fontsize=TICK_SIZE, columnspacing=1.55, handlelength=1.45, handletextpad=.55)
    gs = fig.add_gridspec(1, 3, left=left, right=right,
                         top=.802 if descriptive_text else .83,
                         bottom=.164 if descriptive_text else .12, wspace=WSPACE)
    minimum = min(seeds["value_pct"].min(), means["low95_pct"].min())
    maximum = max(seeds["value_pct"].max(), means["high95_pct"].max())
    low, high = min(-31, np.floor(minimum / 5) * 5 - 1), max(21, np.ceil(maximum / 5) * 5 + 1)
    x_ticks = np.arange(np.ceil(low / 20) * 20, high, 20)
    jitter = np.linspace(-.05 if descriptive_text else -.035,
                         .05 if descriptive_text else .035, len(SEEDS))
    shared = None
    for i, (backbone, model_title) in enumerate(MODELS.items()):
        ax = fig.add_subplot(gs[i], sharex=shared, sharey=shared)
        shared = ax if shared is None else shared
        title = "ChemBERTa" if not descriptive_text and backbone == "ChemBERTa_head" else model_title
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", color=INK, pad=6)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID, linewidth=.65)
        ax.axvline(0, color=AXES, linewidth=.75, linestyle=REFERENCE, zorder=1)
        ax.spines[["top", "left", "right"]].set_visible(False)
        boundary = 6.54 if descriptive_text else 6.5
        ax.axhspan(boundary, 7.5, facecolor=BAND, edgecolor="none", zorder=0)
        if descriptive_text:
            ax.axhline(boundary, color="#D0D8DE", linewidth=.7, zorder=1)
        else:
            for boundary in np.arange(.5, 7, 1):
                ax.axhline(boundary, color="#D0D8DE", linewidth=.55, zorder=1)
        for j, metric in enumerate([*ENDPOINTS, AVG_METRIC]):
            offsets = (-.22, 0, .22) if descriptive_text else (-.26, 0, .26)
            for (cell, (_, color)), offset in zip(CONDITIONS.items(), offsets, strict=True):
                key = (backbone, metric, CONTRASTS[cell])
                estimate = means.loc[key]
                points = seeds["value_pct"].xs(key, level=KEYS).reindex(SEEDS)
                y = j + offset
                ax.scatter(points, y + jitter, color=color, s=7, alpha=.43, linewidths=0, zorder=2)
                ax.hlines(y, estimate["low95_pct"], estimate["high95_pct"], color=color, lw=1.2, zorder=3)
                ax.scatter(estimate["mean_pct"], y, color=color, s=19, marker="D",
                           edgecolors="white", linewidths=.45, zorder=4)
        ax.set(xlim=(low, high), ylim=(len(ENDPOINTS) + .5, -.55))
        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}%"))
        labels = [f"{label} (n = {n})" if descriptive_text else label for label, n in ENDPOINTS.values()]
        ax.set_yticks(list(range(len(labels) + 1)), [*labels, "Average (7 endpoints)" if descriptive_text else "Average"])
        ax.tick_params(axis="y", length=0, pad=7, labelsize=LABEL_SIZE, labelleft=i == 0, colors=INK)
        if i == 0:
            ticks = ax.get_yticklabels()
            for tick, metric in zip(ticks[:-1], ENDPOINTS, strict=True):
                tick.set_color(ENDPOINT_COLORS[metric.removesuffix("_mae")])
            ticks[-1].set_fontweight("bold")
        ax.tick_params(axis="x", labelsize=TICK_SIZE, colors=MUTED)
    fig.text((left + right) / 2, .071 if descriptive_text else .035, "Change in MAE relative to structure only (%)",
             ha="center", fontsize=LABEL_SIZE, color=INK)
    if descriptive_text:
        fig.text(HEADING_X, .022, "Dots show nine paired seeds. Diamonds show means. Bars show exploratory 95% crossed bootstrap intervals.",
                 fontsize=TICK_SIZE, color=MUTED)
    columns = ["denominator00", "n_structures", "n_groups"]
    plotted_means = means[[*columns, "mean_pct", "low95_pct", "high95_pct"]].reset_index().assign(plot_kind="summary")
    plotted_seeds = seeds[[*columns, "value_pct"]].reset_index().assign(plot_kind="seed")
    plotted = pd.concat([plotted_means, plotted_seeds], ignore_index=True)
    plotted.insert(0, "panel", "b")
    plotted["seed"] = plotted["seed"].astype("Int64")
    return plotted
