"""Figure S4: endpoint shifts and error changes by observed-label compatibility."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

TRIAD = ("t_half", "Vd", "CL")
STRATA = [("within 1.25-fold", "≤1.25"), ("1.25- to 2-fold", "1.25–2"), ("2- to 5-fold", "2–5"),
          ("above 5-fold", ">5")]
LEFT, WIDTH, GAP = .216, .242, .016
RIGHT = LEFT + 3 * WIDTH + 2 * GAP
CENTRE = (LEFT + RIGHT) / 2


def point(ax, r, y, colour, marker="D", filled=True, size=4.0):
    ax.errorbar(r.estimate, y, xerr=[[r.estimate - r.ci95_lower], [r.ci95_upper - r.estimate]], fmt=marker, ms=size,
                mfc=colour if filled else "white", mec=colour, mew=.9, color=colour, lw=1.1, capsize=1.8, zorder=4)


def interval_key(colour, marker="D", filled=True):
    return Line2D([], [], ls="none", marker=marker, ms=4.0, mfc=colour if filled else "white", mec=colour, mew=.9)


def forest_axes(fig, column, bottom, height, xlim, ticks, rows, row_labels):
    ax = fig.add_axes([LEFT + (WIDTH + GAP) * column, bottom, WIDTH, height])
    ax.set(xlim=xlim, xticks=ticks, ylim=(min(rows) - .55, max(rows) + .55), yticks=rows)
    ax.set_yticklabels(row_labels if column == 0 else [], fontsize=fs.BASE)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.grid(axis="x", color=fs.GRID, lw=.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    return ax


def main():
    fs.configure()
    shifts = pd.read_csv(fs.TABLES / "sup08_endpoint_shifts_and_spread.csv")
    shifts = shifts[shifts.policy == "selected_positive"]
    strata = pd.read_csv(fs.TABLES / "sup07_compatibility_strata_contrasts.csv")
    triads = pd.read_csv(fs.ROOT / "data" / "same_row_complete_triads.csv")
    observed_mean = float(triads.signed_residual_log10.mean())
    assert len(triads) == 674
    fig = plt.figure(figsize=(fs.WIDTH, 7.45), facecolor="white")
    letters = {"A": .969, "B": .746, "C": .500, "D": .360}
    for letter, ypos in letters.items():
        fs.letter(fig, letter, .012, ypos)
    labels = [fs.ENDPOINTS[endpoint][0] for endpoint in TRIAD]
    endpoint_colours = [fs.ENDPOINT_COLOURS[endpoint] for endpoint in TRIAD]
    counts = strata.groupby("observed_stratum").n_triads_in_stratum.first()
    x = np.arange(len(STRATA))

    def get(model, endpoint, statistic):
        return shifts[(shifts.backbone == model) & (shifts.endpoint == endpoint)
                      & (shifts.statistic == statistic)].iloc[0]

    for column, (model, title) in enumerate(fs.MODELS):
        # A and B: plot each endpoint on a labelled row using the same model columns.
        ax = forest_axes(fig, column, .7756, .1385, (-.08, 1.04), [0, .25, .5, .75, 1],
                         [3, 2, 1, -.25], labels + ["Total (three endpoints)"])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        fs.model_title(ax, title)
        ax.axhspan(-.79, .27, color=fs.BAND, zorder=0)
        ax.axhline(.38, color="#CBD5DC", lw=.6)
        ax.axvline(1 / 3, color=fs.REFERENCE, lw=.7, ls=":", zorder=1)
        for label, colour in zip(ax.get_yticklabels(), endpoint_colours):
            label.set_color(colour)
        for endpoint, yy, colour in zip(TRIAD, [3, 2, 1], endpoint_colours):
            point(ax, get(model, endpoint, "absorption_share_of_baseline_residual"), yy, colour)
        point(ax, get(model, "triad", "total_fraction_of_baseline_residual_removed"), -.25, fs.CONTROL)

        ax = forest_axes(fig, column, .5627, .1332, (-.045, .062), [-.04, 0, .04],
                         [2, 1, 0], labels)
        ax.axvline(0, color=fs.REFERENCE, lw=.7, ls="--", zorder=1)
        for label, colour in zip(ax.get_yticklabels(), endpoint_colours):
            label.set_color(colour)
        for endpoint, yy, colour in zip(TRIAD, [2, 1, 0], endpoint_colours):
            point(ax, get(model, endpoint, "mean_signed_shift"), yy, colour)

        # C: connected estimates before and after training with the loss.
        ax = forest_axes(fig, column, .3903, .0785, (-.01, .135), [0, .04, .08, .12], [0], [""])
        before = get(model, "triad", "mean_signed_residual_baseline")
        after = get(model, "triad", "mean_signed_residual_policy")
        ax.axvline(0, color=fs.REFERENCE, lw=.7, ls="--", zorder=1)
        ax.axvline(observed_mean, color=fs.REFERENCE, lw=.7, ls=":", zorder=1)
        ax.plot([before.estimate, after.estimate], [0, 0], color="#B8C2C9", lw=.8, zorder=1)
        point(ax, before, 0, fs.BASELINE_GREY, filled=False)
        point(ax, after, 0, fs.LOSS)

        # D: error contrasts by discrete observed compatibility strata.
        ax = fig.add_axes([LEFT + (WIDTH + GAP) * column, .1095, WIDTH, .1901])
        ax.axhline(0, color=fs.REFERENCE, lw=.7, ls="--", zorder=1)
        for policy, colour, marker, filled, shift in (
            ("selected_positive", fs.LOSS, "D", True, -.14),
            ("exact_orthogonal", fs.CONTROL, "s", False, .14),
        ):
            d = strata[(strata.backbone == model) & (strata.arm_a == policy) & (strata.metric == "triad_mae_log10")]
            d = d.set_index("observed_stratum").loc[[key for key, _ in STRATA]]
            ax.errorbar(x + shift, d.estimate, yerr=[d.estimate - d.ci95_lower, d.ci95_upper - d.estimate], fmt=marker,
                        ms=4.0, mfc=colour if filled else "white", mec=colour, mew=.9, color=colour, lw=1.0, capsize=1.6)
        ax.set(xlim=(-.5, len(STRATA) - .5), xticks=x, ylim=(-.014, .021), yticks=[-.01, 0, .01, .02])
        ax.set_xticklabels([f"{label}\n(n = {counts[key]})" for key, label in STRATA])
        ax.grid(axis="y", color=fs.GRID, lw=.5)
        ax.set_axisbelow(True)
        if column == 0:
            ax.set_ylabel("Change in triad MAE\n(log$_{10}$ units), 95% interval", labelpad=3)
        else:
            ax.tick_params(axis="y", labelleft=False, length=0)

    for ypos, label in [
        (.732, "Fraction of baseline residual removed through each endpoint"),
        (.518, "Mean shift in prediction with loss (log$_{10}$ units)"),
        (.346, "Mean signed triad residual (log$_{10}$ units)"),
        (.043, "Observed fold disagreement of the triad (number of triads)"),
    ]:
        fig.text(CENTRE, ypos, label, ha="center", va="center")

    dotted = Line2D([], [], color=fs.REFERENCE, lw=.7, ls=":")
    fig.legend([dotted], ["One third"], loc="upper left", bbox_to_anchor=(.051, .966),
               handletextpad=.3, borderaxespad=0)
    fig.legend([interval_key(fs.BASELINE_GREY, filled=False), interval_key(fs.LOSS), dotted],
               ["No loss", "With loss", "Observed triads"], loc="center left", bbox_to_anchor=(.051, .430),
               handletextpad=.3, borderaxespad=0)
    fig.legend([interval_key(fs.LOSS), interval_key(fs.CONTROL, "s", False)],
               ["Training loss", "Exact orthogonal correction"], loc="upper center",
               bbox_to_anchor=(CENTRE, .323), ncol=2, handletextpad=.3, borderaxespad=0)
    print("observed mean signed residual (log10):", observed_mean)
    return fs.save(fig, "figure_S4_shifts_and_compatibility")


if __name__ == "__main__":
    main()
