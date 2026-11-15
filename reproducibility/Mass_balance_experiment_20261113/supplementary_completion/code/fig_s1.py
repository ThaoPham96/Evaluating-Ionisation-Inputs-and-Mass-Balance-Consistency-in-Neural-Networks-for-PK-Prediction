"""Figure S1: training loss versus post-hoc correction controls, and matched strength trade-offs."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

POLICIES = [("baseline", "No loss (baseline)"), ("selected_positive", r"Loss, selected $\lambda$ > 0"),
            ("zero_eligible", r"Loss, $\lambda$ = 0 allowed"),
            ("selected_alpha_orthogonal", "Partial correction, orthogonal"),
            ("selected_alpha_half_life", "Partial correction, half-life only"),
            ("exact_orthogonal", "Exact correction, orthogonal"),
            ("exact_half_life", "Exact correction, half-life only")]
CURVES = [("training_loss", "Training loss, fixed $\\lambda$", "-", "o", True),
          ("correction_orthogonal", "Orthogonal correction, fixed $\\alpha$", (0, (4, 2)), "s", False),
          ("correction_half_life", "Half-life-only correction, fixed $\\alpha$", (0, (1, 1.6)), "^", False)]
LEFT, WIDTH, GAP = .300, .2165, .0125


def policy_colour(policy):
    if policy == "baseline":
        return fs.BASELINE_GREY
    if policy in {"selected_positive", "zero_eligible"}:
        return fs.LOSS
    return fs.CONTROL


def main():
    fs.configure()
    seeds = pd.read_csv(fs.TABLES / "sup02_policy_seed_metrics.csv")
    contrasts = pd.read_csv(fs.TABLES / "sup02_paired_contrasts.csv")
    curve_seeds = pd.read_csv(fs.TABLES / "sup03_curve_seed_metrics.csv")
    curves = pd.read_csv(fs.TABLES / "sup03_curve_contrasts_vs_baseline.csv")
    selection = pd.read_csv(fs.TABLES / "sup02_selection_by_fold.csv")
    fig = plt.figure(figsize=(fs.WIDTH, 8.7), facecolor="white")
    ypos = np.arange(len(POLICIES))[::-1]
    jitter = np.linspace(-.22, .22, 9)
    rows = {"A": (.762, .175), "B": (.528, .175), "C": (.262, .185), "D": (.050, .125)}
    for letter, (bottom, height) in rows.items():
        fs.letter(fig, letter, .012, bottom + height + (.056 if letter == "A" else .040 if letter == "C" else .030))
    for column, (model, title) in enumerate(fs.MODELS):
        x0 = LEFT + (WIDTH + GAP) * column
        # ---- A: consistency by policy (nine seeds) -----------------------
        ax = fig.add_axes([x0, rows["A"][0], WIDTH, rows["A"][1]])
        fs.model_title(ax, title)
        for (policy, _), y in zip(POLICIES, ypos):
            colour = policy_colour(policy)
            v = seeds[(seeds.backbone == model) & (seeds.policy == policy) & (seeds.metric == "within_1_25_fold_percent")]
            assert len(v) == 9
            ax.scatter(v.value, y + jitter, s=7, facecolor=colour, edgecolor="white", linewidth=.25, alpha=.75, zorder=3)
            ax.plot(v.value.mean(), y, marker="D", ms=4, mfc="white", mec=colour, mew=1.0, zorder=4)
            ax.text(121, y, f"{v.value.mean():.0f}", va="center", ha="right", fontsize=fs.SMALL)
        ax.set(xlim=(0, 123), ylim=(-.6, len(POLICIES) - .4), xticks=[0, 25, 50, 75, 100], yticks=ypos)
        ax.spines["bottom"].set_bounds(0, 100)
        ax.set_yticklabels([label for _, label in POLICIES] if column == 0 else [])
        ax.tick_params(axis="y", length=0, pad=5)
        ax.grid(axis="x", color=fs.GRID, lw=.5)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        if column == 1:
            ax.set_xlabel("Predictions within 1.25-fold disagreement (%)", labelpad=3)
        # ---- B: paired accuracy change versus baseline -------------------
        ax = fig.add_axes([x0, rows["B"][0], WIDTH, rows["B"][1]])
        ax.axvline(0, color=fs.REFERENCE, lw=.7, ls="--", zorder=1)
        for (policy, _), y in zip(POLICIES, ypos):
            colour = policy_colour(policy)
            if policy == "baseline":
                ax.text(0.06, y, "reference", va="center", ha="left", fontsize=fs.SMALL, color=fs.MUTED)
                continue
            r = contrasts[(contrasts.backbone == model) & (contrasts.arm_a == policy) & (contrasts.arm_b == "baseline")
                          & (contrasts.metric == "triad_mae_relative_log10")].iloc[0]
            if r.arms_identical:
                ax.plot(0, y, marker="D", ms=4, mfc="white", mec=colour, mew=1.0, zorder=4)
                ax.text(0.12, y, "identical to baseline", va="center", ha="left", fontsize=fs.SMALL, color=fs.MUTED)
                continue
            ax.errorbar(r.estimate, y, xerr=[[r.estimate - r.ci95_lower], [r.ci95_upper - r.estimate]], fmt="D", ms=4,
                        mfc=colour, mec=colour, color=colour, lw=1.2, capsize=1.8, zorder=4)
        ax.set(xlim=(-1.6, 2.1), ylim=(-.6, len(POLICIES) - .4), xticks=[-1, 0, 1, 2], yticks=ypos)
        ax.set_yticklabels([label for _, label in POLICIES] if column == 0 else [])
        ax.tick_params(axis="y", length=0, pad=5)
        ax.grid(axis="x", color=fs.GRID, lw=.5)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        if column == 1:
            ax.set_xlabel("Change in triad MAE versus baseline (% of baseline MAE), 95% interval", labelpad=3)
        # ---- C: matched three-seed trade-off curves ----------------------
        ax = fig.add_axes([x0, rows["C"][0], WIDTH, rows["C"][1]])
        ax.axhline(0, color=fs.REFERENCE, lw=.7, ls="--", zorder=1)
        for method, _, style, marker, filled in CURVES:
            colour = fs.LOSS if filled else fs.CONTROL
            c = curves[(curves.backbone == model) & (curves.method == method) & (curves.metric == "triad_mae_relative_log10")]
            c = c.sort_values("strength")
            x = [curve_seeds[(curve_seeds.backbone == model) & (curve_seeds.policy == p)
                             & (curve_seeds.metric == "within_1_25_fold_percent")].value.mean() for p in c.policy]
            ax.errorbar(x, c.estimate, yerr=[c.estimate - c.ci95_lower, c.ci95_upper - c.estimate], color=colour,
                        ls=style, lw=1.0, marker=marker, ms=3.6, mfc=colour if filled else "white", mec=colour,
                        mew=.8, elinewidth=.6, capsize=1.2, zorder=4 if filled else 3)
        ax.set(xlim=(30, 104), ylim=(-3.1, 2.6), xticks=[40, 60, 80, 100], yticks=[-3, -2, -1, 0, 1, 2])
        ax.grid(color=fs.GRID, lw=.5)
        ax.set_axisbelow(True)
        if column == 0:
            ax.set_ylabel("Change in triad MAE\n(% of baseline MAE)", labelpad=3)
        else:
            ax.set_yticklabels([])
        if column == 1:
            ax.set_xlabel("Predictions within 1.25-fold disagreement (%)", labelpad=3)
        # ---- D: inner-selected settings by outer fold --------------------
        ax = fig.add_axes([x0, rows["D"][0], WIDTH, rows["D"][1]])
        fields = [("selected_positive_lambda", r"Selected $\lambda$ (must be > 0)"),
                  ("zero_eligible_lambda", r"Selected $\lambda$ when 0 is allowed"),
                  ("selected_alpha_orthogonal", r"Selected $\alpha$, orthogonal"),
                  ("selected_alpha_half_life", r"Selected $\alpha$, half-life only")]
        s = selection[selection.backbone == model].sort_values("outer_fold")
        for i, (field, _) in enumerate(fields):
            for j, value in enumerate(s[field]):
                y = len(fields) - 1 - i
                if value == 0:
                    ax.add_patch(plt.Rectangle((j - .46, y - .42), .92, .84, fc="#DDE3E7", ec="none"))
                ax.text(j, y, f"{value:g}", ha="center", va="center", fontsize=fs.SMALL,
                        fontweight="bold" if value == 0 else "normal")
        ax.set(xlim=(-.5, 4.5), ylim=(-.5, len(fields) - .5), xticks=range(5), yticks=range(len(fields)))
        ax.set_xticklabels([1, 2, 3, 4, 5])
        ax.set_yticklabels([label for _, label in fields][::-1] if column == 0 else [])
        ax.tick_params(length=0, pad=4)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if column == 1:
            ax.set_xlabel("Outer fold", labelpad=3)
    fig.text(.051, rows["A"][0] + rows["A"][1] + .047, "Dots: nine seeds\nOpen diamond and number: mean",
             ha="left", va="center", fontsize=fs.SMALL, color=fs.MUTED)
    handles = []
    for _, _, style, marker, filled in CURVES:
        colour = fs.LOSS if filled else fs.CONTROL
        handles.append(Line2D([], [], color=colour, ls=style, lw=1.0, marker=marker, ms=3.6,
                              mfc=colour if filled else "white", mec=colour, mew=.8))
    fig.legend(handles, [label for _, label, *_ in CURVES], loc="lower right", ncol=3, handlelength=2.6,
               bbox_to_anchor=(LEFT + 3 * WIDTH + 2 * GAP, rows["C"][0] + rows["C"][1] + .012), columnspacing=1.2,
               handletextpad=.5, borderaxespad=0)
    fig.text(LEFT + 3 * WIDTH + 2 * GAP, rows["D"][0] + rows["D"][1] + .018, "Shaded: no intervention selected",
             ha="right", va="center", fontsize=fs.SMALL, color=fs.MUTED)
    return fs.save(fig, "figure_S1_controls_and_tradeoffs")


if __name__ == "__main__":
    main()
