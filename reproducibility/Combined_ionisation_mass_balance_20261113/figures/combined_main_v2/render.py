"""Build the combined main figure and supplementary figures from frozen results."""
from __future__ import annotations

import argparse
import importlib.util
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
ROOT = HERE.parents[1]
OLD = HERE.parent / "combined_main_v1"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


legacy = module("legacy_combined_main", OLD / "render.py")
schematic = module("combined_schematic", HERE / "schematic.py")
endpoint = module("combined_endpoints", HERE / "endpoint_panel.py")
from combined_style import (AXES, CONDITIONS, GRID, HEADING_X, LABEL_SIZE, LEFT, MODELS,
                            MUTED, REFERENCE, RIGHT, TICK_SIZE, TITLE_SIZE, WIDTH, WSPACE,
                            configure, panel_heading)

review = legacy.review

JOINT_HEIGHT = 2.22
MAIN_LABEL_COLOUR = "#161616"


def joint_panel(fig, means, seeds, *, letter="C", title="Seven-endpoint accuracy and internal disagreement",
                xcoord="all_endpoint_error_change_pct", n=327, descriptive_text=True):
    panel_heading(fig, letter, title if descriptive_text else "", .967)
    left = LEFT if descriptive_text else .105
    bottom = .265 if descriptive_text else .22
    top = .73 if descriptive_text else .745
    handles = [
        Line2D([], [], marker="D", linestyle="none", markersize=4.7,
               markerfacecolor="none" if cell == "11" and descriptive_text else color,
               markeredgecolor=color, label=label)
        for cell, (label, color) in CONDITIONS.items()
    ]
    if descriptive_text:
        handles += [Line2D([], [], marker="o", linestyle="none", color=AXES, markersize=3.1, label="Seed"),
                    Line2D([], [], marker="D", linestyle="none", color=MUTED, markersize=4.7, label="Mean")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=((left + RIGHT) / 2, .851 if descriptive_text else .865),
               frameon=False, ncols=len(handles), fontsize=TICK_SIZE, columnspacing=1.0, handlelength=1.2, handletextpad=.4)
    gs = fig.add_gridspec(1, 3, left=left, right=RIGHT, top=top, bottom=bottom, wspace=WSPACE)
    limits = (np.floor(seeds[xcoord].min() - .6), np.ceil(seeds[xcoord].max() + .6))
    ticks = np.arange(5 * np.ceil(limits[0] / 5), limits[1] + .01, 5)
    shared = None
    for i, (backbone, label) in enumerate(MODELS.items()):
        ax = fig.add_subplot(gs[i], sharex=shared, sharey=shared)
        shared = ax if shared is None else shared
        name = "ChemBERTa" if not descriptive_text and backbone == "ChemBERTa_head" else label
        ax.set_title(name, fontsize=TITLE_SIZE, fontweight="bold", pad=6)
        ax.set_axisbelow(True)
        ax.grid(color=GRID, linewidth=.65)
        ax.axvline(0, color=AXES, linewidth=.75, linestyle=REFERENCE)
        ax.set(xlim=limits, ylim=(.045, .173))
        ax.set_xticks(ticks)
        ax.set_yticks([.06, .09, .12, .15])
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}%"))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
        ax.tick_params(labelsize=TICK_SIZE, labelleft=i == 0)
        if i == 0:
            ylabel = (f"Mean absolute log₁₀ residual\n({n} predicted triads)" if descriptive_text
                      else "Mass-balance consistency\nMean absolute residual (log₁₀ units)")
            ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, labelpad=7)
        for cell, (_, color) in CONDITIONS.items():
            points = seeds.xs((backbone, cell), level=("backbone", "cell"))
            mean = means.loc[backbone, cell]
            joint = cell == "11" and descriptive_text
            ax.scatter(points[xcoord], points["mean_abs_residual_log10"], s=11,
                       facecolors="none" if joint else color,
                       alpha=.82 if joint else .60, edgecolors=color if joint else "white",
                       linewidths=.65 if joint else .25, zorder=2)
            ax.scatter(mean[xcoord], mean["mean_abs_residual_log10"], s=30, marker="D",
                       facecolors="none" if joint else color, edgecolors=color if joint else "white",
                       linewidths=.95 if joint else .6, zorder=3)
    xlabel = ("Average error relative to structure-only mean (%)" if descriptive_text
              else "Change in MAE relative to structure only (%)")
    fig.text((left + RIGHT) / 2, .102 if descriptive_text else .055, xlabel,
             fontsize=LABEL_SIZE, ha="center")
    if descriptive_text:
        note = ("All seven endpoints with available labels. Hollow marks show ionisation + loss."
                if n == 327 else "Same 242 complete triads on both axes. Hollow marks show ionisation + loss.")
        fig.text(HEADING_X, .022, note, fontsize=TICK_SIZE, color=MUTED)
    return list(map(float, limits))


CAPTION = """Combining ionisation inputs with a mass-balance loss.

All comparisons use the same 327-structure numerical-pKa cohort, 238 scaffold/parent groups, five grouped outer folds and nine matched training seeds. Each seed pools held-out predictions across its five outer folds.

A. A structural encoder feeds a multitask prediction head. The ionisation input is a four-dimensional one-hot category derived from acidic and basic pKa at pH 7.4 and concatenated with the structural representation. Acidic pKa below 7.4 and basic pKa above 7.4 define the two activity criteria. The four categories are acid-only, base-only, neither-active and dual-active. Paracetamol illustrates the structure input; the four blank boxes illustrate input dimensions. The mass-balance penalty is the mean squared residual between predicted log10 half-life and the value implied by predicted volume and clearance. It is applied in unstandardised log10 space. Coloured output symbols identify the seven PK endpoints. Dashed orange arrows connect the three coupled predictions to the training loss and its feedback to the prediction head. The loss-enabled conditions share the same positive penalty weight within a backbone and outer fold, selected using inner-validation predictions. The frozen ChemBERTa model trains its MLP head.

B. Endpoint MAE changes for ionisation only, loss only and ionisation + loss, each compared with the matched structure-only model. Each paired seed difference is divided by the endpoint's fixed nine-seed mean structure-only MAE and multiplied by 100. Negative values indicate lower error. Black row labels identify the endpoints and the estimates use condition colours. The Average row averages these seven normalised changes equally within each seed. Five endpoints use log10 MAE; bioavailability and protein binding use percentage-point MAE, retaining zero labels. All available labels are used separately for each endpoint. Label counts are 316 for half-life, 259 for volume of distribution, 246 for clearance, 190 for peak concentration, 249 for time to peak, 172 for bioavailability and 292 for protein binding. Small dots show nine paired seed differences, diamonds their means and bars exploratory 95% crossed-bootstrap percentile intervals. The 10,000 resamples jointly draw whole structure groups and whole fitted seed trajectories. The average row is calculated within each joint draw, preserving dependence across endpoints. Normalising denominators remain fixed. Intervals are not adjusted for multiple endpoint comparisons and condition on the observed training data, saved folds and selected recipes.

C. Overall accuracy and internal disagreement across all four conditions. Each small circle pairs two outcomes for one fitted seed; diamonds average nine seeds. Filled purple marks show ionisation + loss. The horizontal axis uses all available labels for each of the seven endpoints. Horizontal values divide each endpoint's seed MAE by its fixed mean structure-only MAE, average the seven ratios equally, subtract one and multiply by 100. The condition means match the Average row in B. Individual seeds here are centred on the fixed baseline mean, whereas B subtracts matched seed baselines. Thus baseline seed variability remains visible in C. Each architecture has its own baseline, so the horizontal axis compares relative effects within architectures. Vertical values are mean absolute log10 residuals over all 327 predicted triads: abs(log10(t_half in h) - log10(Vd in L) + log10(CL in L/h) - log10(ln 2)). This measures agreement among predictions, independently of observed endpoint labels. Lower left indicates lower average error and closer internal agreement. Seed spread in C is not a confidence interval.

"""


def supplementary(frames, data):
    dest = HERE / "supplementary"
    dest.mkdir(exist_ok=True)
    source = module("combined_supplement_source", HERE.parent / "combined_supplementary_v1" / "render.py")
    source.PLOTTED = []
    fig = source.interaction_plot(frames["contrasts"], frames["seed_contrasts"])
    fig.texts[0].set_text("S1  Accuracy interactions between ionisation and loss")
    review.save(fig, dest, "S1_interactions")
    fig = source.control_plot(frames["contrasts"], frames["seed_contrasts"], frames["metrics"], frames["seed_metrics"])
    fig.texts[0].set_text("S2  Training loss and post-hoc correction controls")
    review.save(fig, dest, "S2_posthoc_controls")
    pd.DataFrame(source.PLOTTED).to_csv(dest / "S1_S2_plot_data.csv", index=False)

    metrics, seeds, cdf, _, _ = review.extract(frames)
    save = review.save
    def renamed_save(fig, destination, stem):
        fig.texts[0].set_text("S3  Absolute accuracy and continuous disagreement")
        save(fig, destination, "S3_accuracy_and_disagreement")
    review.save = renamed_save
    try:
        omitted, denominator = review.overview(metrics, seeds, cdf, dest)
    finally:
        review.save = save
    pd.concat([metrics.reset_index().assign(panel="S3A", plot_kind="summary"),
               seeds.reset_index().assign(panel="S3A", plot_kind="seed"),
               cdf.assign(panel="S3B", plot_kind="cdf")], ignore_index=True).to_csv(dest / "S3_plot_data.csv", index=False)

    _, _, joint_seeds, effects, effect_seeds, _, _ = data
    _, all_means, all_seeds = legacy.load_all_endpoints(review.SCORES)
    sensitivity = pd.read_csv(OLD / "residual_cohort_summary.csv").set_index(["backbone", "n_structures"])
    fig = plt.figure(figsize=(WIDTH, 3.35), facecolor="white")
    legacy.draw_paired_supplement(fig, effects, effect_seeds, all_means, all_seeds, sensitivity)
    fig.texts[0].set_text("S4  Adding the loss to ionisation-informed models")
    review.save(fig, dest, "S4_paired_effects")
    paired = pd.read_csv(OLD / "supplementary" / "S5_plot_data.csv")
    paired["panel"] = paired["panel"].str.replace("S5", "S4", regex=False)
    paired.to_csv(dest / "S4_plot_data.csv", index=False)
    sensitivity.to_csv(dest / "S4_cohort_sensitivity.csv")
    triad_means, triad_seeds = legacy.load_triad_display(review.SCORES, joint_seeds)
    fig = plt.figure(figsize=(WIDTH, JOINT_HEIGHT), facecolor="white")
    joint_panel(fig, triad_means, triad_seeds, letter="S5", title="Accuracy and agreement on complete triads",
                xcoord="triad_error_change_pct", n=242)
    review.save(fig, dest, "S5_matched_triads")
    pd.concat([triad_means.reset_index().assign(plot_kind="summary"),
               triad_seeds.reset_index().assign(plot_kind="seed")], ignore_index=True).to_csv(dest / "S5_plot_data.csv", index=False)
    old_captions = (HERE.parent / "combined_supplementary_v1" / "captions.txt").read_text(encoding="utf-8")
    supplement_caption = old_captions[old_captions.index("S2:"):].replace("S2:", "S1:").replace("S3:", "S2:")
    supplement_caption = supplement_caption.replace(
        "Thick intervals are exploratory 95%;",
        "The Vd and clearance row labels use their endpoint colours. Purple identifies the interaction estimates. "
        "Thick intervals are exploratory 95%;")
    supplement_caption = supplement_caption.replace(
        "as a dashed reference line", "as a blue dashed reference line")
    supplement_caption = supplement_caption.replace(
        "The exact controls use alpha=1, producing algebraic consistency by construction.",
        "Charcoal intervals are dashed for orthogonal correction and dotted for half-life-only correction. "
        "Open diamonds show exact controls (α = 1), which produce algebraic consistency by construction.")
    overview_caption = (HERE.parent / "combined_results_v1" / "caption.txt").read_text(encoding="utf-8").split("Adding the loss to ionisation-informed models")[0]
    overview_caption = overview_caption.replace("included in plot_data.csv", "included in S3_plot_data.csv")
    overview_caption = overview_caption.replace("\na. ", "\nA. ").replace("\nb. ", "\nB. ")
    overview_caption = overview_caption.replace("Both (11)", "Ionisation + loss (11), labelled Both on the axis,")
    paired_caption = legacy.S5_CAPTION.replace("S5.", "S4.").replace("the main figure's triad display", "Supplementary Figure S5")
    matched_caption = ("S5. Accuracy and internal agreement on the matched 242-structure complete-triad subset. "
        "Each endpoint MAE (half-life, Vd and CL) is divided by its fixed nine-seed structure-only mean, "
        "and the three ratios are averaged equally, centred at one and multiplied by 100. "
        "The vertical axis uses predicted triad residuals on the same 242 structures. "
        "Circles pair outcomes for nine fitted seeds and diamonds show means. Hollow marks show ionisation + loss. "
        "This relative endpoint average differs from the prespecified raw triad macro MAE in S3 and S4.\n")
    (dest / "captions.txt").write_text(supplement_caption + "\nS3. " + overview_caption + "\n" + paired_caption + "\n" + matched_caption, encoding="utf-8")
    return {"pages": 5, "cdf_view_max_fold": 5, "max_predictions_above_view": omitted,
            "seed_predictions_per_curve": denominator}


def main(*, main_only=False):
    configure()
    manifest, frames = review.load_scores(review.SCORES)
    data = legacy.load_data(review.SCORES)
    all_means, all_seeds = data[-2:]
    heights = (1.80, 4.25, 2.30)
    fig = plt.figure(figsize=(WIDTH, sum(heights)), facecolor="white")
    top, middle, bottom = fig.subfigures(3, 1, height_ratios=heights, hspace=0)
    panel_heading(top, "A", "", .967)
    ax = top.add_axes([.035, .0, .943, .86])
    schematic.draw_schematic(ax)
    b_data = endpoint.draw_endpoint_panel(middle, frames, left=LEFT, right=RIGHT, descriptive_text=False)
    for label in middle.axes[0].get_yticklabels()[:-1]:
        label.set_color(MAIN_LABEL_COLOUR)
    limits = joint_panel(bottom, all_means, all_seeds, descriptive_text=False)
    b_summaries = b_data.loc[b_data["plot_kind"].eq("summary")]
    for (backbone, cell), row in all_means.iterrows():
        if cell == "00":
            continue
        observed = b_summaries.loc[b_summaries["backbone"].eq(backbone) & b_summaries["contrast"].eq(f"{cell}_minus_00")]
        endpoint_rows = observed.loc[observed["metric"].isin(endpoint.ENDPOINTS)]
        assert len(endpoint_rows) == 7
        np.testing.assert_allclose(endpoint_rows["mean_pct"].mean(), row["all_endpoint_error_change_pct"], rtol=0, atol=1e-12)
        average_rows = observed.loc[~observed["metric"].isin(endpoint.ENDPOINTS)]
        assert len(average_rows) == 1, "Average row required by user"
        np.testing.assert_allclose(average_rows["mean_pct"].iloc[0], row["all_endpoint_error_change_pct"], rtol=0, atol=1e-12)
    review.save(fig, HERE, "combined_main")
    plotted = pd.concat([b_data, all_means.reset_index().assign(panel="c", plot_kind="summary"),
                         all_seeds.reset_index().assign(panel="c", plot_kind="seed")], ignore_index=True)
    assert len(plotted) == 840
    plotted.to_csv(HERE / "plot_data.csv", index=False)
    (HERE / "caption.txt").write_text(CAPTION, encoding="utf-8")
    manifest_path = HERE / "source_manifest.json"
    if main_only:
        supplement = json.loads(manifest_path.read_text(encoding="utf-8"))["supplementary"]
    else:
        supplement = supplementary(frames, data)
    dependencies = [Path(__file__), HERE / "schematic.py", HERE / "endpoint_panel.py", HERE / "average_endpoints.py", HERE / "average_manifest.json",
                    OLD / "render.py", OLD / "schematic.py", OLD / "triad_display_manifest.json",
                    OLD / "all_endpoint_manifest.json", Path(review.__file__),
                    Path(review.__file__).with_name("combined_style.py"),
                    HERE.parent / "combined_supplementary_v1" / "render.py"]
    record = {"campaign_id": manifest["campaign_id"], "score_manifest_sha256": review.sha256(review.SCORES / "manifest.json"),
              "verified_score_files": manifest["files"], "main_plot_rows": len(plotted),
              "joint_x_limits": limits, "main_panel_counts": {"b": 720, "c": 120},
              "cross_panel_average_verified_atol": 1e-12, "supplementary": supplement,
              "supplementary_rendered_by_this_build": not main_only,
              "presentation": "Panel letters, endpoint-coloured schematic symbols, black endpoint labels and condition-coloured estimates",
              "main_layout": {"figure_inches": [WIDTH, sum(heights)], "panel_heights_inches": list(heights),
                              "simplified_input_and_loss_boxes": True, "endpoint_separator_lines": True,
                              "endpoint_condition_offsets": [-.26, 0, .26], "joint_panel_left": .105,
                              "combined_condition_markers": "Filled purple", "display_average_label": "Average",
                              "display_transformer_label": "ChemBERTa"},
              "source_files": {str(p.relative_to(ROOT)): review.sha256(p) for p in dependencies},
              "files": {p.name: review.sha256(p) for p in HERE.iterdir() if p.suffix in (".pdf", ".png", ".svg", ".csv", ".txt")}}
    manifest_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"main": str(HERE / "combined_main.pdf"), "plotted_rows": len(plotted), "supplementary": supplement}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-only", action="store_true", help="Render the main figure and retain existing supplements")
    main(main_only=parser.parse_args().main_only)
