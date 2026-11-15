"""Render the selected mass-balance supplementary panels from frozen tables."""
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import pandas as pd
from aggregate_single_task import aggregate, METRIC as AVERAGE

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "code"))
import figstyle as fs
from fig_s2 import CELLS
from fig_s3 import forest
from fig_s4 import STRATA

STEM = "mass_balance_supplementary_selected"
MODELS = [(key, "ChemBERTA" if key == "ChemBERTa_head" else title) for key, title in fs.MODELS]
SIZE = (7.0, 8.85)
SEEDS = (101, 202, 303)
STRATUM_COUNTS = (363, 177, 101, 33)
CONSISTENCY, ABSOLUTE, RELATIVE = "within_1_25_fold_percent", "triad_mae_log10", "triad_mae_relative_log10"
ENCODER_CONTRAST = "loss - baseline (same encoder/depth)"
SINGLE_CONTRASTS = ("single_task - multitask_baseline", "single_task - multitask_loss")
CURVES = [
    ("multitask_baseline", "Multitask, no loss", fs.BASELINE_GREY, fs.BASELINE_DASH, 1.2),
    ("multitask_loss", "Multitask, with loss", fs.LOSS, "-", 1.6),
    ("single_task", "Seven single-task models", fs.CONTROL, (0, (1, 1.6)), 1.4),
]
SINGLE_ROWS = [(RELATIVE, "Triad (three endpoints)", fs.CONTROL)] + [
    (f"{key}_mae_relative_{fs.ENDPOINT_UNIT[key]}", label, fs.ENDPOINT_COLOURS[key])
    for key, (label, _) in fs.ENDPOINTS.items()
]


def selection():
    specs = [
        ("encoder_seed_metrics", "sup04_encoder_depth_seed_metrics.csv", 96,
         lambda d: d.metric.isin((CONSISTENCY, ABSOLUTE))),
        ("encoder_contrasts", "sup04_encoder_depth_contrasts.csv", 8,
         lambda d: d.contrast.eq(ENCODER_CONTRAST) & d.metric.eq(RELATIVE)),
        ("single_task_contrasts", "sup06_single_task_contrasts.csv", 16,
         lambda d: d.contrast.isin(SINGLE_CONTRASTS) & d.metric.isin([key for key, _, _ in SINGLE_ROWS])),
        ("single_task_ecdf", "sup06_predicted_disagreement_ecdf.csv", 1803,
         lambda d: d.model.isin([key for key, *_ in CURVES])),
        ("training_loss_strata", "sup07_compatibility_strata_contrasts.csv", 12,
         lambda d: d.arm_a.eq("selected_positive") & d.arm_b.eq("baseline") & d.metric.eq(ABSOLUTE)),
    ]
    data, records = {}, {}
    for key, name, count, select in specs:
        source = fs.TABLES / name
        frame = pd.read_csv(source)
        mask = select(frame)
        assert mask.sum() == count, (name, mask.sum())
        data[key] = frame.loc[mask].copy()
        export = ROOT / f"{key}.csv"
        # Retain the exact source field strings, including floating-point precision.
        with source.open(newline="", encoding="utf-8-sig") as original, export.open(
                "w", newline="", encoding="utf-8") as out:
            reader = csv.reader(original)
            writer = csv.writer(out, lineterminator="\n")
            writer.writerow(next(reader))
            writer.writerows(row for keep, row in zip(mask, reader, strict=True) if keep)
        pd.testing.assert_frame_equal(pd.read_csv(export), data[key].reset_index(drop=True), check_exact=True)
        records[key] = {"source": f"../tables/{name}", "source_sha256": digest(source), "rows": count,
                        "export": export.name, "export_sha256": digest(export)}
    return data, records


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def check_range(values, limits):
    lower, upper = limits
    assert np.all(np.isfinite(values)) and np.all((values >= lower) & (values <= upper)), limits


def draw_forest_axis(ax, xlim, ticks, y, labels, xlabel):
    ax.set(xlim=xlim, xticks=ticks, ylim=(min(y) - .6, max(y) + .6), yticks=y, xlabel=xlabel)
    ax.set_yticklabels(labels, fontsize=fs.BASE)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.grid(axis="x", color=fs.GRID, lw=.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)


def encoder_panels(fig, seeds, contrasts):
    y = np.arange(len(CELLS))[::-1].astype(float)
    y[:5] += .5
    offsets = np.array([-.16, 0, .16])
    configs = [
        (CONSISTENCY, "Predictions within\n1.25-fold disagreement (%)", (0, 100), [0, 25, 50, 75, 100]),
        (RELATIVE, "Change in triad MAE with loss\n(% of baseline MAE), 95% interval", (-2.2, 2.7), [-2, -1, 0, 1, 2]),
        (ABSOLUTE, "Triad MAE\n(log$_{10}$ units)", (.450, .487), [.46, .47, .48]),
    ]
    left, width, gap = .277, .211, .024
    for column, (metric, label, limits, ticks) in enumerate(configs):
        x0 = left + (width + gap) * column
        ax = fig.add_axes([x0, .700, width, .239], label="ABC"[column])
        fs.letter(fig, "ABC"[column], .012 if column == 0 else x0 - .013, .961)
        for (model, depth, _), yy in zip(CELLS, y):
            if metric == RELATIVE:
                rows = contrasts[contrasts.backbone.eq(model) & contrasts.depth.eq(depth)]
                assert len(rows) == 1 and rows.n_seeds.eq(3).all()
                check_range(rows[["ci95_lower", "estimate", "ci95_upper"]].to_numpy(), limits)
                forest(ax, rows.iloc[0], yy, fs.LOSS)
                continue
            mean = {}
            for arm, face, edge in (("baseline", "white", fs.BASELINE_GREY), ("loss", fs.LOSS, fs.LOSS)):
                rows = seeds[seeds.backbone.eq(model) & seeds.depth.eq(depth) & seeds.arm.eq(arm)
                             & seeds.metric.eq(metric)]
                assert tuple(rows.seed) == SEEDS
                check_range(rows.value, limits)
                ax.scatter(rows.value, yy + offsets, s=9, facecolor=face, edgecolor=edge, linewidth=.7, zorder=3)
                mean[arm] = rows.value.mean()
            ax.plot([mean["baseline"], mean["loss"]], [yy, yy], color="#B8C2C9", lw=.8, zorder=1)
        if metric == RELATIVE:
            ax.axvline(0, color=fs.REFERENCE, lw=.7, ls="--", zorder=1)
        draw_forest_axis(ax, limits, ticks, y, [name for *_, name in CELLS] if column == 0 else [], label)
    handles = [
        Line2D([], [], ls="none", marker="o", ms=3.4, mfc="white", mec=fs.BASELINE_GREY, mew=.7),
        Line2D([], [], ls="none", marker="o", ms=3.4, mfc=fs.LOSS, mec=fs.LOSS, mew=.7),
    ]
    fig.legend(handles, ["Baseline, no loss", "With mass-balance loss"], loc="upper right",
               bbox_to_anchor=(.980, .987), ncol=2, handletextpad=.3, columnspacing=1.3, borderaxespad=0)


def single_task_panels(fig, single, ecdf):
    fs.letter(fig, "D", .012, .639)
    fs.letter(fig, "E", .611, .639)
    ax = fig.add_axes([.253, .364, .337, .241], label="D")
    display_rows = [*SINGLE_ROWS, (AVERAGE, "Average (7 endpoints)", fs.CONTROL)]
    y = np.arange(len(display_rows))[::-1].astype(float)
    y[0] += .4
    y[-1] -= .35
    limits = (-16, 9)
    ax.axhspan(y[-1] - .55, .24, color=fs.BAND, zorder=0)
    ax.axhline(.43, color="#CBD5DC", lw=.6)
    ax.axvline(0, color=fs.REFERENCE, lw=.7, ls="--", zorder=1)
    for (metric, _, colour), yy in zip(display_rows, y):
        for contrast, offset, filled in zip(SINGLE_CONTRASTS, (.15, -.15), (True, False)):
            rows = single[single.contrast.eq(contrast) & single.metric.eq(metric)]
            assert len(rows) == 1 and rows.n_seeds.eq(3).all()
            check_range(rows[["ci95_lower", "estimate", "ci95_upper"]].to_numpy(), limits)
            forest(ax, rows.iloc[0], yy + offset, colour, filled=filled)
    draw_forest_axis(ax, limits, [-15, -10, -5, 0, 5], y, [name for _, name, _ in display_rows],
                     "Single-task minus multitask MAE\n(% of multitask MAE), 95% interval")
    ax.get_yticklabels()[-1].set_fontweight("bold")
    handles = [
        Line2D([], [], ls="none", marker="D", ms=3.8, mfc=fs.CONTROL, mec=fs.CONTROL),
        Line2D([], [], ls="none", marker="D", ms=3.8, mfc="white", mec=fs.CONTROL, mew=.9),
    ]
    ax.legend(handles, ["Versus multitask, no loss", "Versus multitask, with loss"],
              loc="lower left", bbox_to_anchor=(-.015, 1.012), handletextpad=.3, borderaxespad=0, labelspacing=.3)

    ax = fig.add_axes([.675, .364, .296, .241], label="E")
    limits = (.92, 5)
    for model, label, colour, style, lw in CURVES:
        rows = ecdf[ecdf.model.eq(model)]
        assert len(rows) == 601 and rows.n_seeds.eq(3).all() and rows.n_structures.eq(952).all()
        assert rows.fold_disagreement.iloc[[0, -1]].tolist() == [1.0, 10.0]
        assert rows.mean_cumulative_percent.is_monotonic_increasing
        check_range(rows.fold_disagreement, (1.0, 10.0))
        check_range(rows.mean_cumulative_percent, (0, 100))
        ax.plot(rows.fold_disagreement, rows.mean_cumulative_percent, color=colour, ls=style, lw=lw, label=label)
    ax.set(xlim=limits, ylim=(-2.5, 102.5), xticks=[1, 2, 3, 4, 5], yticks=[0, 25, 50, 75, 100],
           xlabel="Predicted fold disagreement", ylabel="Predictions within fold (%)")
    ax.spines["bottom"].set_bounds(1, 5)
    ax.spines["left"].set_bounds(0, 100)
    ax.grid(axis="y", color=fs.GRID, lw=.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", handlelength=2.4, labelspacing=.4, borderaxespad=.2)


def strata_panel(fig, strata):
    fs.letter(fig, "F", .012, .285)
    left, width, gap = .104, .275, .024
    x = np.arange(len(STRATA))
    limits = (-.014, .021)
    for column, (model, title) in enumerate(MODELS):
        ax = fig.add_axes([left + (width + gap) * column, .087, width, .167], label=f"F-{title}")
        d = strata[strata.backbone.eq(model)].set_index("observed_stratum").loc[[key for key, _ in STRATA]]
        assert tuple(d.n_triads_in_stratum) == STRATUM_COUNTS and d.n_seeds.eq(9).all() and len(d) == len(x)
        check_range(d[["ci95_lower", "estimate", "ci95_upper"]].to_numpy(), limits)
        ax.axhline(0, color=fs.REFERENCE, lw=.7, ls="--", zorder=1)
        ax.errorbar(x, d.estimate, yerr=[d.estimate - d.ci95_lower, d.ci95_upper - d.estimate], fmt="D",
                    color=fs.LOSS, ms=4, lw=1.0, capsize=1.6)
        ax.set(xlim=(-.5, 3.5), xticks=x, ylim=limits, yticks=[-.01, 0, .01, .02])
        ax.set_xticklabels([label for _, label in STRATA])
        ax.set_title(title, fontweight="bold", pad=5)
        ax.grid(axis="y", color=fs.GRID, lw=.5)
        ax.set_axisbelow(True)
        if column == 0:
            ax.set_ylabel("Change in triad MAE\n(log$_{10}$ units), 95% interval", labelpad=3)
        else:
            ax.tick_params(axis="y", labelleft=False, length=0)
    centre = left + (3 * width + 2 * gap) / 2
    fig.text(centre, .036, "Observed fold disagreement of the triad", ha="center", va="center")


def main():
    family = fs.configure()
    data, records = selection()
    average, average_record = aggregate(data["single_task_contrasts"])
    average_csv = ROOT / "single_task_average.csv"
    average.to_csv(average_csv, index=False)
    average_record.update({"export": average_csv.name, "export_sha256": digest(average_csv), "rows": len(average)})
    fig = plt.figure(figsize=SIZE, facecolor="white")
    encoder_panels(fig, data["encoder_seed_metrics"], data["encoder_contrasts"])
    single_task_panels(fig, pd.concat([data["single_task_contrasts"], average], ignore_index=True),
                       data["single_task_ecdf"])
    strata_panel(fig, data["training_loss_strata"])
    for label in fig.findobj(Text):
        label.set_color("black")
    problems = fs.check_layout(fig)
    assert not problems, problems
    outputs = [ROOT / f"{STEM}.{ext}" for ext in ("pdf", "png", "svg")]
    for output in outputs:
        fig.savefig(output, dpi=300)
    manifest = {
        "dimensions_inches": list(SIZE), "font": family, "preview_dpi": 300,
        "selection": {"A–C": "former S2 A–C", "D–E": "former S3 B–C", "F": "former S4 D, training loss"},
        "seeds_A_to_E": list(SEEDS), "n_seeds_F": 9, "stratum_counts_F": list(STRATUM_COUNTS),
        "source_tables": records, "single_task_average": average_record,
        "figure_sha256": {output.name: digest(output) for output in outputs},
        "caption_sha256": digest(ROOT / "caption.txt"), "renderer_sha256": digest(Path(__file__)),
        "dependencies_sha256": {name: digest(ROOT / name)
                                 for name in ("../code/figstyle.py", "../code/fig_s2.py", "../code/fig_s3.py",
                                              "../code/fig_s4.py", "../code/mbsupp.py", "aggregate_single_task.py")},
        "axes": {ax.get_label(): {"x": list(ax.get_xlim()), "y": list(ax.get_ylim())} for ax in fig.axes},
        "range_checks": "All displayed estimates and intervals fit their axes. E retains the original displayed x-range.",
        "ecdf_source_range": [1.0, 10.0],
        "text_layout_problems": problems,
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)
    print(f"Rendered {STEM}; exported {sum(record['rows'] for record in records.values())} source rows.")


if __name__ == "__main__":
    main()
