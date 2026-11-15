"""Figure S3: task-weight sensitivity and single-task comparison (GraphConv, three seeds)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

WEIGHT_ROWS = [("loss effect, equal weights", "Loss effect, equal weights"),
               ("loss effect, Vd/CL weights 5", r"Loss effect, $V_d$ and $CL$ weights $\times$5"),
               ("weight effect, baseline", r"Weights $\times$5 effect, no loss"),
               ("weight effect, with loss", r"Weights $\times$5 effect, with loss")]
WEIGHT_COLUMNS = [("triad_mae_relative_log10", "Triad", fs.INK),
                  ("t_half_mae_relative_log10", "Half-life", fs.ENDPOINT_COLOURS["t_half"]),
                  ("Vd_mae_relative_log10", "Volume of\ndistribution", fs.ENDPOINT_COLOURS["Vd"]),
                  ("CL_mae_relative_log10", "Clearance", fs.ENDPOINT_COLOURS["CL"])]


def forest(ax, r, y, colour, filled=True):
    ax.errorbar(r.estimate, y, xerr=[[r.estimate - r.ci95_lower], [r.ci95_upper - r.estimate]], fmt="D", ms=3.8,
                mfc=colour if filled else "white", mec=colour, mew=.9, color=colour, lw=1.1, capsize=1.8, zorder=4)


def main():
    fs.configure()
    weights = pd.read_csv(fs.TABLES / "sup05_task_weight_contrasts.csv")
    single = pd.read_csv(fs.TABLES / "sup06_single_task_contrasts.csv")
    ecdf = pd.read_csv(fs.TABLES / "sup06_predicted_disagreement_ecdf.csv")
    fig = plt.figure(figsize=(fs.WIDTH, 5.55), facecolor="white")
    # ---- A: task weights -------------------------------------------------
    fs.letter(fig, "A", .012, .985)
    left, width, gap, bottom, height = .300, .15925, .014, .662, .247
    y = np.arange(len(WEIGHT_ROWS))[::-1]
    for column, (metric, title, colour) in enumerate(WEIGHT_COLUMNS):
        ax = fig.add_axes([left + (width + gap) * column, bottom, width, height])
        ax.set_title(title, fontweight="bold", pad=5, va="bottom", color=colour)
        ax.axvline(0, color=fs.REFERENCE, lw=.7, ls="--", zorder=1)
        for (contrast, _), yy in zip(WEIGHT_ROWS, y):
            loss_effect = contrast.startswith("loss effect")
            forest(ax, weights[(weights.contrast == contrast) & (weights.metric == metric)].iloc[0],
                   yy, fs.LOSS if loss_effect else fs.CONTROL, filled=loss_effect)
        ax.set(xlim=(-3.6, 5.4), xticks=[-2, 0, 2, 4], ylim=(-.6, len(WEIGHT_ROWS) - .4), yticks=y)
        ax.set_yticklabels([label for _, label in WEIGHT_ROWS] if column == 0 else [])
        ax.tick_params(axis="y", length=0, pad=5)
        ax.grid(axis="x", color=fs.GRID, lw=.5)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
    fig.text(left + (4 * width + 3 * gap) / 2, bottom - .061, "Change in MAE (% of comparator MAE), 95% interval",
             ha="center", va="center", fontsize=fs.BASE)
    # ---- B: single-task versus multitask ---------------------------------
    fs.letter(fig, "B", .012, .5495)
    ax = fig.add_axes([.252, .0934, .333, .429])
    rows = [("triad_mae_relative_log10", "Triad (three endpoints)", fs.INK)]
    rows += [(f"{name}_mae_relative_{fs.ENDPOINT_UNIT[name]}", label, fs.ENDPOINT_COLOURS[name])
             for name, (label, _) in fs.ENDPOINTS.items()]
    y = np.arange(len(rows))[::-1].astype(float)
    y[0] += .4
    ax.axvline(0, color=fs.REFERENCE, lw=.7, ls="--", zorder=1)
    for (metric, _, _), yy in zip(rows, y):
        forest(ax, single[(single.contrast == "single_task - multitask_baseline") & (single.metric == metric)].iloc[0],
               yy + .15, fs.CONTROL)
        forest(ax, single[(single.contrast == "single_task - multitask_loss") & (single.metric == metric)].iloc[0],
               yy - .15, fs.CONTROL, filled=False)
    ax.set(xlim=(-16, 9), xticks=[-15, -10, -5, 0, 5], ylim=(y.min() - .6, y.max() + .6), yticks=y)
    labels = ax.set_yticklabels([label for _, label, _ in rows], fontsize=fs.BASE)
    for label, (_, _, colour) in zip(labels, rows):
        label.set_color(colour)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.grid(axis="x", color=fs.GRID, lw=.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Single-task minus multitask MAE\n(% of multitask MAE), 95% interval", labelpad=3)
    handles = [Line2D([], [], ls="none", marker="D", ms=3.8, mfc=fs.CONTROL, mec=fs.CONTROL),
               Line2D([], [], ls="none", marker="D", ms=3.8, mfc="white", mec=fs.CONTROL, mew=.9)]
    ax.legend(handles, ["versus multitask, no loss", "versus multitask, with loss"], loc="lower left",
              bbox_to_anchor=(-.02, 1.0), ncol=1, handletextpad=.3, borderaxespad=0, labelspacing=.3)
    # ---- C: consistency of single-task versus multitask predictions -------
    fs.letter(fig, "C", .611, .5495)
    ax = fig.add_axes([.682, .0934, .293, .429])
    styles = [("multitask_baseline", "Multitask, no loss", fs.BASELINE_GREY, fs.BASELINE_DASH, 1.2),
              ("multitask_loss", "Multitask, with loss", fs.LOSS, "-", 1.6),
              ("single_task", "Seven single-task models", fs.CONTROL, (0, (1, 1.6)), 1.4)]
    for model, label, colour, style, lw in styles:
        d = ecdf[ecdf.model == model]
        ax.plot(d.fold_disagreement, d.mean_cumulative_percent, color=colour, ls=style, lw=lw, label=label)
    ax.set(xlim=(.92, 5), ylim=(-2.5, 102.5), xticks=[1, 2, 3, 4, 5], yticks=[0, 25, 50, 75, 100])
    ax.spines["bottom"].set_bounds(1, 5)
    ax.spines["left"].set_bounds(0, 100)
    ax.grid(axis="y", color=fs.GRID, lw=.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Predicted fold disagreement", labelpad=3)
    ax.set_ylabel("Predictions within fold (%)", labelpad=3)
    ax.legend(loc="lower right", handlelength=2.4, labelspacing=.4, borderaxespad=.2)
    return fs.save(fig, "figure_S3_task_weights_and_single_task")


if __name__ == "__main__":
    main()
