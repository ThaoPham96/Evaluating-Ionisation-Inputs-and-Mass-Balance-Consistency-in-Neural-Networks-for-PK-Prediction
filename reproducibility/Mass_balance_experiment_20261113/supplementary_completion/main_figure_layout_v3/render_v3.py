"""Main mass-balance figure (panels A-E), rendered from frozen v3 inputs."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
STEM = "mass_balance_main_v3"
sys.path.insert(0, str(ROOT.parent / "code"))
import figstyle as fs
sys.path.insert(0, str(ROOT))
from schematic import draw_schematic

WIDTH, HEIGHT = fs.WIDTH, 8.85
BODY_HEIGHT = 8.35
BASE, SMALL = fs.BASE, fs.SMALL
INK, MUTED, GRID, BOX = fs.INK, fs.MUTED, fs.GRID, fs.BOX
MODELS = [(key, "ChemBERTA" if key == "ChemBERTa_head" else title) for key, title in fs.MODELS]
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909]
METRICS = [
    ("t_half_log10_mae", "t_half"),
    ("Vd_log10_mae", "Vd"),
    ("CL_log10_mae", "CL"),
    ("Cmax_log10_mae", "Cmax"),
    ("Tmax_log10_mae", "Tmax"),
    ("F_percentage_point_mae", "F"),
    ("PPB_percentage_point_mae", "PPB"),
]
ENDPOINTS = [(metric, *fs.ENDPOINTS[key], fs.ENDPOINT_COLOURS[key]) for metric, key in METRICS]
THRESHOLDS = (1.5, 2.0, 5.0)
LEFT, COL, GAP = .356, .197, .0165
RIGHT = LEFT + 3 * COL + 2 * GAP
ENDPOINT_LEFT, ENDPOINT_GAP = .240, .020
ENDPOINT_WIDTH = (RIGHT - ENDPOINT_LEFT - 2 * ENDPOINT_GAP) / 3


def model_axes(fig, column, bottom, height, title):
    ax = fig.add_axes([LEFT + (COL + GAP) * column, bottom, COL, height])
    fs.model_title(ax, title)
    return ax


def schematic(fig):
    # Extra height accommodates both losses while B-E retain their physical size.
    return draw_schematic(fig, [.035, 6.4 / HEIGHT, .94, 2.3205 / HEIGHT],
                          letter_position=(.025, 8.635 / HEIGHT), letter_size=fs.LETTER)


def observed_panel(fig, triads):
    fs.letter(fig, "B", .025, .752)
    ax = fig.add_axes([.081, .569, .197, .161])
    r = (np.log10(triads["t1/2(hour)"]) - np.log10(triads["VD(liter)"])
         + np.log10(triads["Cl(liter/hour)"]) - np.log10(np.log(2)))
    fold = 10 ** r.abs()
    np.testing.assert_allclose(fold, triads.disagreement_fold, rtol=1e-12)
    assert len(triads) == triads.structure_id.nunique() == 674
    within = {t: int((fold <= t).sum()) for t in THRESHOLDS}
    assert within == {1.5: 451, 2.0: 540, 5.0: 641}, within
    upper = 90.0
    assert 1 < fold.min() and fold.max() < upper
    # Exact empirical cumulative distribution, using the same step convention as C.
    ordered = np.sort(fold.to_numpy())
    curve_x = np.r_[1.0, ordered, upper]
    curve_y = np.r_[0.0, 100 * np.arange(1, len(ordered) + 1) / len(ordered), 100.0]
    colour = "#33424D"
    ax.step(curve_x, curve_y, where="post", color=colour, lw=1.6, zorder=3)
    for threshold in THRESHOLDS:
        percent = 100 * within[threshold] / len(ordered)
        ax.plot([threshold, threshold], [0, percent], color=fs.REFERENCE,
                lw=.6, ls=(0, (3, 2.5)), zorder=2)
        ax.plot(threshold, percent, "o", ms=3.0, color=colour, mec="white", mew=.4, zorder=4)
    ax.set(xscale="log", xlim=(1 / 1.03, upper), ylim=(-2.5, 102.5), yticks=[0, 25, 50, 75, 100])
    ax.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
    ax.minorticks_off()
    fig.text(.081 + .197 / 2, .536, "Observed mass-balance\nfold disagreement", ha="center", va="top")
    ax.set_ylabel("Structures within fold (%)", labelpad=3)
    ax.grid(axis="y", color=GRID, lw=.5)
    ax.set_axisbelow(True)
    ax.text(.43, .66, "Within fold", transform=ax.transAxes, fontsize=SMALL,
            fontweight="bold", va="top")
    for y, threshold in zip([.50, .37, .24], THRESHOLDS):
        ax.text(.43, y, f"\u2264{threshold:g}×", transform=ax.transAxes, va="center", fontsize=SMALL)
        ax.text(.99, y, f"{100 * within[threshold] / 674:.1f}%", transform=ax.transAxes, ha="right", va="center",
                fontsize=SMALL, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_bounds(1, upper)
    ax.spines["left"].set_bounds(0, 100)
    pd.DataFrame({"fold_disagreement": curve_x, "cumulative_percent": curve_y}).to_csv(
        ROOT / "observed_ecdf_v3.csv", index=False)
    return {"n": 674, **{f"within_{t:g}_n": n for t, n in within.items()}, "beyond_5_n": 674 - within[5.0],
            "median_fold_disagreement": float(fold.median()), "maximum_fold_disagreement": float(fold.max()),
            "x_axis_upper_limit": upper, "x_axis_scale": "logarithmic",
            "display": "Exact empirical cumulative distribution of all 674 structures; post steps; "
                       "reference points at 1.5-, 2- and 5-fold; display endpoints at 1 and 90"}


def continuous_panel(fig, curves):
    fs.letter(fig, "C", .298, .752)
    assert curves.groupby(["backbone", "policy"]).ngroups == 6
    styles = [
        ("baseline", fs.BASELINE_GREY, fs.BASELINE_DASH, 1.2, "No loss"),
        ("selected_positive", fs.LOSS, "-", 1.6, "With loss"),
    ]
    for column, (model, title) in enumerate(MODELS):
        ax = model_axes(fig, column, .569, .161, title)
        for policy, colour, linestyle, linewidth, _ in styles:
            d = curves[(curves.backbone == model) & (curves.policy == policy)].sort_values("fold_disagreement")
            assert d.mean_cumulative_percent.is_monotonic_increasing
            d = d[d.fold_disagreement <= 3.05]
            ax.step(d.fold_disagreement, d.mean_cumulative_percent, where="post", color=colour,
                    lw=linewidth, ls=linestyle, zorder=2 if policy == "baseline" else 3)
        ax.set(xlim=(.94, 3), ylim=(-2.5, 102.5), yticks=[0, 25, 50, 75, 100])
        ax.set_xticks([1, 1.5, 2, 2.5, 3], ["1", "1.5", "2", "2.5", "3"])
        ax.grid(axis="y", color=GRID, lw=.5)
        ax.set_axisbelow(True)
        ax.spines["bottom"].set_bounds(1, 3)
        ax.spines["left"].set_bounds(0, 100)
        if column == 0:
            ax.set_ylabel("Predictions within fold (%)", labelpad=3)
        else:
            ax.tick_params(axis="y", labelleft=False, length=0)
    fig.text((LEFT + RIGHT) / 2, .536, "Predicted mass-balance\nfold disagreement", ha="center", va="top")
    handles = [Line2D([], [], color=colour, lw=linewidth, ls=linestyle)
               for _, colour, linestyle, linewidth, _ in styles]
    ax.legend(handles, [label for *_, label in styles], loc="lower right",
              fontsize=SMALL, handlelength=2.3, frameon=False, borderaxespad=.5)


def split_panel(fig, split):
    """Change in triad MAE with the loss by observed disagreement (95% intervals)."""
    fs.letter(fig, "D", .025, .496)
    left, gap = .098, .026
    width = (RIGHT - left - 2 * gap) / 3
    groups = ["within 1.5-fold", "above 1.5-fold"]
    assert len(split) == 6 and set(split.metric) == {"triad_mae_log10"} and set(split.policy) == {"selected_positive"}
    for column, (model, title) in enumerate(MODELS):
        ax = fig.add_axes([left + (width + gap) * column, .334, width, .140])
        fs.model_title(ax, title)
        ax.axhline(0, color=fs.REFERENCE, lw=.7, ls="--", zorder=1)
        d = split[split.backbone == model].set_index("stratum").loc[groups]
        assert d.n_triads.tolist() == [451, 223] and (d.n_seeds == 9).all()
        np.testing.assert_allclose(d.relative_percent, 100 * d.estimate / d.baseline_mean, atol=1e-9)
        ax.errorbar(np.arange(2), d.relative_percent, yerr=[d.relative_percent - d.relative_ci95_lower,
                                                           d.relative_ci95_upper - d.relative_percent],
                    color=fs.LOSS, fmt="D", ms=4.0, elinewidth=.9, capsize=1.6, zorder=3)
        ax.set(xlim=(-.5, 1.5), xticks=[0, 1], ylim=(-2.3, 2.6), yticks=[-2, -1, 0, 1, 2])
        ax.set_xticklabels(["≤1.5-fold", ">1.5-fold"])
        ax.grid(axis="y", color=GRID, lw=.5)
        ax.set_axisbelow(True)
        if column == 0:
            ax.set_ylabel("Change in MAE vs\nstructure only (%)", labelpad=3)
        else:
            ax.tick_params(axis="y", labelleft=False, length=0)
    fig.text((left + RIGHT) / 2, .285, "Observed mass-balance\nfold disagreement", ha="center", va="center")


def endpoint_panel(fig, rows, macro, estimates):
    fs.letter(fig, "E", .025, .249)
    all_values = pd.concat([rows, macro], ignore_index=True)
    assert len(rows) == 189 and len(macro) == 27
    np.testing.assert_allclose(rows.normalized_change_percent, 100 * rows.delta / rows.mean_baseline_mae, atol=1e-10)
    for (model, seed), g in rows.groupby(["backbone", "seed"]):
        value = macro[(macro.backbone == model) & (macro.seed == seed)].normalized_change_percent.iloc[0]
        np.testing.assert_allclose(value, g.normalized_change_percent.mean(), atol=1e-11)
    display = [(metric, name, colour) for metric, name, _, colour in ENDPOINTS]
    display.append(("macro_seven_endpoint", "Average (7 endpoints)", fs.CONTROL))
    positions = [7, 6, 5, 4, 3, 2, 1, -.35]
    jitter = np.array([-.17, .13, -.09, .18, .04, -.13, .09, -.03, .15])
    bottom, height = .0714, .1556
    axes = []
    for column, (model, title) in enumerate(MODELS):
        ax = fig.add_axes([ENDPOINT_LEFT + (ENDPOINT_WIDTH + ENDPOINT_GAP) * column,
                           bottom, ENDPOINT_WIDTH, height])
        fs.model_title(ax, title)
        axes.append(ax)
        ax.axvline(0, color=fs.REFERENCE, lw=.7, ls="--", zorder=1)
        ax.axhspan(-.8, .1, color=fs.BAND, zorder=0)
        ax.axhline(.35, color="#CBD5DC", lw=.6)
        for (metric, _, colour), y in zip(display, positions):
            v = all_values[(all_values.backbone == model) & (all_values.metric == metric)].set_index("seed").loc[SEEDS]
            stat = estimates[(estimates.backbone == model) & (estimates.metric == metric)].iloc[0]
            np.testing.assert_allclose([v.normalized_change_percent.mean(), v.normalized_change_percent.std(ddof=1)],
                                       [stat["mean"], stat.sd], atol=1e-11)
            assert v.normalized_change_percent.between(-8, 6.5).all()
            ax.scatter(v.normalized_change_percent, y + jitter, s=8, facecolor=colour, edgecolor="white",
                       linewidth=.25, alpha=.74, zorder=3)
            ax.errorbar(stat["mean"], y, xerr=stat.sd, fmt="D", ms=3.6, mfc=colour, mec=INK, mew=.45, color=colour,
                        lw=1.2, capsize=1.8, zorder=5)
            if metric == "macro_seven_endpoint":
                ax.text(.97, y, f"{stat['mean']:+.2f}%".replace("-", "\u2212"), transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=SMALL)
        ax.set(xlim=(-8, 6.5), ylim=(-1.15, 7.6), xticks=[-6, -4, -2, 0, 2, 4, 6], yticks=positions)
        ax.grid(axis="x", color=GRID, lw=.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=6)
        if column == 0:
            labels = ax.set_yticklabels([d[1] for d in display], fontsize=BASE)
            for label, (_, _, colour) in zip(labels, display):
                label.set_color(colour)
            labels[-1].set_fontweight("bold")
        else:
            ax.set_yticklabels([])
    centre = (ENDPOINT_LEFT + RIGHT) / 2
    fig.text(centre, .0341, "Change in MAE vs structure only (%)", ha="center",
             va="center", fontsize=BASE)
    # Two separate strings: SVG text collapses runs of spaces.
    fig.text(centre - .025, .0139, "\u2190 lower error", ha="right", va="center", fontsize=SMALL, color=MUTED)
    fig.text(centre + .025, .0139, "higher error \u2192", ha="left", va="center", fontsize=SMALL, color=MUTED)
    return axes


def main():
    family = fs.configure()
    triads = pd.read_csv(INPUTS / "same_row_complete_triads.csv")
    curves = pd.read_csv(INPUTS / "cdf_values.csv.gz")
    rows = pd.read_csv(INPUTS / "endpoint_seed_values.csv")
    macro = pd.read_csv(INPUTS / "average_seed_values.csv")
    estimates = pd.read_csv(INPUTS / "endpoint_summary.csv")
    split = pd.read_csv(INPUTS / "split_1_5_summary.csv")
    fig = plt.figure(figsize=(WIDTH, BODY_HEIGHT), facecolor="white")
    observed = observed_panel(fig, triads)
    continuous_panel(fig, curves)
    split_panel(fig, split)
    endpoint_panel(fig, rows, macro, estimates)
    for label in fig.findobj(Text):
        label.set_color("black")
    # Keep every data-panel dimension and text size unchanged in inches.
    ratio = BODY_HEIGHT / HEIGHT
    for ax in fig.axes:
        x, y, w, h = ax.get_position().bounds
        ax.set_position([x, y * ratio, w, h * ratio])
    for label in fig.texts:
        x, y = label.get_position()
        label.set_position((x, y * ratio))
    fig.set_size_inches(WIDTH, HEIGHT)
    schematic(fig)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(ROOT / f"{STEM}.{ext}", dpi=350)
    problems = fs.check_layout(fig)
    plt.close(fig)
    (ROOT / "observed_summary_v3.json").write_text(json.dumps(observed, indent=2) + "\n", encoding="utf-8")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    (ROOT / "manifest.json").write_text(json.dumps({
        "purpose": "Endpoint-coloured estimates and schematic symbols with black labels",
        "layout_revision": "Separate supervised MSE and mass-balance terms in A; B-E retain physical dimensions and positions from the lower edge",
        "new_training": False, "data_changed": False, "seeds": SEEDS, "dimensions_inches": [WIDTH, HEIGHT],
        "png_dpi": 350, "font": family, "text_sizes_pt": {"base": BASE, "small": SMALL, "panel_letter": fs.LETTER, "title": fs.TITLE, "panel_A_triad_symbols": 12,
                                                    "panel_A_other_symbols": 9.5, "panel_A_equation": 8.0},
        "uncertainty_D": "95% paired group-bootstrap intervals, nine seeds",
        "uncertainty_E": "sample SD across nine paired seeds",
        "inputs_sha256": {p.name: sha(p) for p in sorted(INPUTS.iterdir()) if p.is_file()},
        "script_sha256": sha(Path(__file__)), "schematic_sha256": sha(ROOT / "schematic.py"), "style_sha256": sha(Path(fs.__file__)),
        "outputs_sha256": {name: sha(ROOT / name) for name in
                           [f"{STEM}.png", f"{STEM}.pdf", f"{STEM}.svg", "observed_ecdf_v3.csv",
                            "observed_summary_v3.json", "caption.txt"]},
        "layout_check_problems": problems}, indent=2) + "\n", encoding="utf-8")
    print("font:", family)
    print("layout problems:", problems)
    print(json.dumps(observed))


if __name__ == "__main__":
    main()
