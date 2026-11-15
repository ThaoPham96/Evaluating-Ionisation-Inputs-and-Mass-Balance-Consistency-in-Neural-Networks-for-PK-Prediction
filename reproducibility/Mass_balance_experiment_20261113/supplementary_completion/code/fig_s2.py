"""Figure S2: alternative graph encoders and GraphConv depth (three seeds, transferred settings)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

CELLS = [("GraphConv", 1, "GraphConv, 1 layer"), ("GraphConv", 2, "GraphConv, 2 layers (reference)"),
         ("GraphConv", 3, "GraphConv, 3 layers"), ("GraphConv", 4, "GraphConv, 4 layers"),
         ("GraphConv", 5, "GraphConv, 5 layers"), ("GIN", 2, "GIN, 2 layers"), ("GAT", 2, "GAT, 2 layers"),
         ("AttentiveFP", 2, "AttentiveFP, 2 layers")]


def main():
    fs.configure()
    seeds = pd.read_csv(fs.TABLES / "sup04_encoder_depth_seed_metrics.csv")
    contrasts = pd.read_csv(fs.TABLES / "sup04_encoder_depth_contrasts.csv")
    contrasts = contrasts[contrasts.contrast.str.startswith("loss - baseline")]
    fig = plt.figure(figsize=(fs.WIDTH, 4.0), facecolor="white")
    y = np.arange(len(CELLS))[::-1].astype(float)
    y[:5] += .5                                           # gap between the depth and encoder groups
    left, width, gap, bottom, height = .285, .205, .030, .150, .715
    panels = [("within_1_25_fold_percent", "Predictions within\n1.25-fold disagreement (%)", (0, 100), [0, 25, 50, 75, 100]),
              (None, "Change in triad MAE with loss\n(% of baseline MAE), 95% interval", (-2.2, 2.7), [-2, -1, 0, 1, 2]),
              ("triad_mae_log10", "Triad MAE\n(log$_{10}$ units)", (.450, .487), [.46, .47, .48])]
    offsets = np.array([-.16, 0, .16])
    for column, (metric, label, xlim, ticks) in enumerate(panels):
        ax = fig.add_axes([left + (width + gap) * column, bottom, width, height])
        fs.letter(fig, "ABC"[column], left + (width + gap) * column - (.270 if column == 0 else .012), bottom + height + .062)
        for (backbone, depth, _), yy in zip(CELLS, y):
            if metric is None:
                r = contrasts[(contrasts.backbone == backbone) & (contrasts.depth == depth)
                              & (contrasts.metric == "triad_mae_relative_log10")].iloc[0]
                ax.errorbar(r.estimate, yy, xerr=[[r.estimate - r.ci95_lower], [r.ci95_upper - r.estimate]], fmt="D",
                            ms=3.8, mfc=fs.LOSS, mec=fs.LOSS, color=fs.LOSS, lw=1.1, capsize=1.8, zorder=4)
                continue
            means = {}
            for arm, face, edge in (("baseline", "white", fs.BASELINE_GREY), ("loss", fs.LOSS, fs.LOSS)):
                v = seeds[(seeds.backbone == backbone) & (seeds.depth == depth) & (seeds.arm == arm) & (seeds.metric == metric)]
                assert len(v) == 3
                ax.scatter(v.value, yy + offsets, s=9, facecolor=face, edgecolor=edge, linewidth=.7, zorder=3)
                means[arm] = v.value.mean()
            ax.plot([means["baseline"], means["loss"]], [yy, yy], color="#B8C2C9", lw=.8, zorder=1)
        if metric is None:
            ax.axvline(0, color=fs.REFERENCE, lw=.7, ls="--", zorder=1)
        ax.set(xlim=xlim, xticks=ticks, ylim=(y.min() - .6, y.max() + .6), yticks=y)
        ax.set_yticklabels([label_ for *_, label_ in CELLS] if column == 0 else [])
        ax.tick_params(axis="y", length=0, pad=5)
        ax.grid(axis="x", color=fs.GRID, lw=.5)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(label, labelpad=3)
    handles = [Line2D([], [], ls="none", marker="o", ms=3.4, mfc="white", mec=fs.BASELINE_GREY, mew=.7),
               Line2D([], [], ls="none", marker="o", ms=3.4, mfc=fs.LOSS, mec=fs.LOSS, mew=.7)]
    fig.legend(handles, ["No loss (baseline), one dot per seed", "With mass-balance loss, one dot per seed"],
               loc="upper right", bbox_to_anchor=(left + 3 * width + 2 * gap, .995), ncol=2, handletextpad=.3,
               columnspacing=1.4, borderaxespad=0)
    return fs.save(fig, "figure_S2_encoders_and_depth")


if __name__ == "__main__":
    main()
