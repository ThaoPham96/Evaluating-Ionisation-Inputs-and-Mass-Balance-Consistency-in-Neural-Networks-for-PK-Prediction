"""Render supporting ionisation figures from the completed scoring exports."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from ionisation_style import AXES, BLUE, CONTROL, FONT, GRID, INK, MODELS as MODEL_NAMES, MUTED, ROW, ENDPOINT_COLORS
EXPERIMENT = HERE.parents[2]
SCORING = EXPERIMENT / "outputs" / "scoring"
PLAN = EXPERIMENT / "runtime" / "plan.json"
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909)
MODELS = {model: (name, BLUE) for model, name in MODEL_NAMES.items()}
ENDPOINTS = {
    "t_half": ("Half-life", ENDPOINT_COLORS["t_half"]),
    "Vd": ("Volume of distribution", ENDPOINT_COLORS["Vd"]),
    "CL": ("Clearance", ENDPOINT_COLORS["CL"]),
    "Cmax": ("Peak concentration", ENDPOINT_COLORS["Cmax"]),
    "Tmax": ("Time to peak", ENDPOINT_COLORS["Tmax"]),
    "F": ("Bioavailability", ENDPOINT_COLORS["F"]),
    "PPB": ("Protein binding", ENDPOINT_COLORS["PPB"]),
}
COHORTS = {"class": "Class cohort", "numerical": "Numerical pKa cohort"}
COUNTS = {
    "class": dict(zip(ENDPOINTS, (474, 388, 374, 273, 361, 244, 433))),
    "numerical": dict(zip(ENDPOINTS, (316, 259, 246, 190, 249, 172, 292))),
}
AVERAGE = "seven_endpoint_average"
CONTEXT = "Grouped five-fold outer cross-validation  •  9 paired seeds"
KEY = [
    "cohort", "procedure", "backbone", "arm", "comparator_procedure",
    "comparator_backbone", "comparator_arm", "scope", "stratum", "endpoint",
]


@dataclass(frozen=True)
class Contrast:
    cohort: str
    arm: str
    comparator: str
    label: str
    procedure: str = "primary"
    comparator_procedure: str = "primary"

    def select(self, table, endpoints):
        return table.loc[
            table["cohort"].eq(self.cohort)
            & table["procedure"].eq(self.procedure)
            & table["arm"].eq(self.arm)
            & table["comparator_procedure"].eq(self.comparator_procedure)
            & table["comparator_arm"].eq(self.comparator)
            & table["backbone"].isin(MODELS)
            & table["comparator_backbone"].eq(table["backbone"])
            & table["scope"].eq("all")
            & table["stratum"].eq("all")
            & table["endpoint"].isin(endpoints)
        ].copy()


HEADLINE = {
    "class": Contrast("class", "C2", "C0", "One-hot class at pH 7.4 vs structure only"),
    "numerical": Contrast("numerical", "P4", "P3", "Four-pH profile vs pH 7.4 propensity pair"),
}
CONTROLS = {
    "class": [
        Contrast("class", "C1", "C0", "Charge-sign scalar vs\nstructure only"),
        Contrast("class", "C2", "C1", "One-hot class vs\ncharge-sign scalar"),
        Contrast("class", "C3", "C2", "Learned class embedding vs\none-hot class"),
    ],
    "numerical": [
        Contrast("numerical", "P3", "P1", "pH 7.4 propensity pair vs\none-hot class"),
        Contrast("numerical", "P4", "P2", "Four-pH profile vs\nraw pKa pair"),
        Contrast("numerical", "P4", "P5", "Four-pH profile vs\nrepeated pH 7.4 pair"),
    ],
}
TUNED = {
    "class": Contrast("class", "C2", "C0", HEADLINE["class"].label, "tuned"),
    "numerical": Contrast("numerical", "P4", "P3", HEADLINE["numerical"].label, "tuned", "tuned"),
}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read_inputs():
    manifest_path = SCORING / "scoring_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    identity = plan["identity"]
    if (
        manifest["status"] != "report_ready"
        or manifest["campaign_id"] != plan["campaign_id"]
        or manifest["release_fingerprints"] != {
            key: identity["datasets"][key]["fingerprint"] for key in COHORTS
        }
        or manifest["bootstrap_draws"] != 10000
        or tuple(identity["seeds"]) != SEEDS
        or identity["endpoint_weights"] != [1] * len(ENDPOINTS)
        or identity["lambda"] != 0
        or {unit["outer_fold"] for unit in identity["units"]} != set(range(1, 6))
    ):
        raise ValueError("Unexpected scoring manifest or campaign design")

    sources = {
        str(path.relative_to(EXPERIMENT).as_posix()): digest(path)
        for path in (manifest_path, PLAN)
    }
    tables = []
    for name in ("contrasts.csv", "contrast_seeds.csv", "endpoint_metrics.csv", "coverage.csv"):
        path = SCORING / name
        file_hash = digest(path)
        if file_hash != manifest["files"][name]:
            raise ValueError(f"Scoring checksum mismatch: {name}")
        sources[path.relative_to(EXPERIMENT).as_posix()] = file_hash
        tables.append(pd.read_csv(path))
    return tables, manifest, sources


def prepare_contrast(contrast, endpoints, scores, seed_scores, metrics, coverage):
    summary = contrast.select(scores, endpoints)
    seeds = contrast.select(seed_scores, endpoints)
    expected = len(MODELS) * len(endpoints)
    if (
        len(summary) != expected
        or len(seeds) != expected * len(SEEDS)
        or summary.duplicated(KEY).any()
        or seeds.duplicated(KEY + ["seed"]).any()
        or not seeds.groupby(KEY)["seed"].agg(lambda values: set(values) == set(SEEDS)).all()
        or not summary["n_seeds"].eq(len(SEEDS)).all()
        or not summary["comparator_mae"].gt(0).all()
    ):
        raise ValueError(f"Unexpected seed or contrast coverage for {contrast}")

    indexed_summary = summary.set_index(KEY).sort_index()
    seed_summary = seeds.groupby(KEY).agg(
        delta_mean=("delta", "mean"), seed_sd=("delta", "std"),
        arm_mae=("arm_mae", "mean"), comparator_mae=("comparator_mae", "mean"),
    )
    pd.testing.assert_index_equal(seed_summary.index, indexed_summary.index)
    np.testing.assert_allclose(
        seed_summary, indexed_summary[seed_summary.columns], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(seeds["delta"], seeds["arm_mae"] - seeds["comparator_mae"], atol=1e-12)
    seeds = seeds.merge(
        seed_summary["comparator_mae"].rename("mean_comparator_mae"),
        on=KEY, validate="many_to_one",
    )
    seeds["relative_delta_pct"] = 100 * seeds["delta"] / seeds["mean_comparator_mae"]
    relative_seed_means = seeds.groupby(KEY)["relative_delta_pct"].mean()
    np.testing.assert_allclose(relative_seed_means, indexed_summary["relative_delta_pct"], atol=1e-11)

    expected_counts = summary["endpoint"].map(COUNTS[contrast.cohort])
    np.testing.assert_array_equal(summary["n_structures"], expected_counts)
    cohort_coverage = coverage.loc[
        coverage["cohort"].eq(contrast.cohort)
        & coverage["outer_fold"].eq("all")
        & coverage["ionisation_class"].eq("all")
    ].set_index("endpoint")
    np.testing.assert_array_equal(summary["n_structures"], summary["endpoint"].map(cohort_coverage["n_labels"]))
    np.testing.assert_array_equal(summary["n_groups"], summary["endpoint"].map(cohort_coverage["n_label_groups"]))

    metric_keys = ["cohort", "procedure", "backbone", "arm", "seed", "endpoint"]
    for side in ("arm", "comparator"):
        match = seeds[metric_keys].copy()
        if side == "comparator":
            for name in ("procedure", "backbone", "arm"):
                match[name] = seeds[f"comparator_{name}"]
        matched = match.merge(
            metrics[metric_keys + ["mae", "n_structures"]], on=metric_keys,
            how="left", validate="one_to_one",
        )
        np.testing.assert_allclose(matched["mae"], seeds[f"{side}_mae"], atol=1e-12)
        np.testing.assert_array_equal(matched["n_structures"], seeds["endpoint"].map(COUNTS[contrast.cohort]))
    return summary, seeds


def endpoint_values(tables):
    summaries, seed_rows = [], []
    for contrast in HEADLINE.values():
        summary, seeds = prepare_contrast(contrast, ENDPOINTS, *tables)
        summaries.append(summary)
        seed_rows.append(seeds)
    seeds = pd.concat(seed_rows, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)

    contrast_keys = [key for key in KEY if key != "endpoint"]
    grouped = seeds.groupby(contrast_keys + ["seed"])["relative_delta_pct"]
    if not grouped.size().eq(len(ENDPOINTS)).all():
        raise ValueError("The descriptive average requires all seven endpoints in each seed")
    macro = grouped.mean().rename("relative_delta_pct").reset_index()
    macro["endpoint"] = AVERAGE
    display = pd.concat([seeds, macro], ignore_index=True)
    estimates = display.groupby(KEY)["relative_delta_pct"].agg(
        mean="mean", seed_sd="std", seed_min="min", seed_max="max", n_seeds="size",
    ).reset_index()
    estimates = estimates.merge(
        summary[KEY + ["n_structures", "n_groups", "comparator_mae"]],
        on=KEY, how="left", validate="one_to_one",
    )
    estimates = estimates.astype({"n_structures": "Int64", "n_groups": "Int64"})
    estimates["lower_sd"] = estimates["mean"] - estimates["seed_sd"]
    estimates["upper_sd"] = estimates["mean"] + estimates["seed_sd"]
    return display, estimates


def representation_values(tables):
    all_values = []
    for cohort, controls in CONTROLS.items():
        for section, contrasts in (
            ("Representation controls", controls),
            ("Tuning sensitivity", [HEADLINE[cohort], TUNED[cohort]]),
        ):
            for order, contrast in enumerate(contrasts):
                summary, _ = prepare_contrast(contrast, ["Vd"], *tables)
                summary["section"] = section
                summary["plot_order"] = order
                summary["comparison_label"] = contrast.label.replace("\n", " ")
                summary["lower_95_pct"] = 100 * summary["crossed95_low"] / summary["comparator_mae"]
                summary["upper_95_pct"] = 100 * summary["crossed95_high"] / summary["comparator_mae"]
                if not summary["valid_bootstrap_draws"].eq(10000).all():
                    raise ValueError(f"Incomplete bootstrap draws for {contrast}")
                all_values.append(summary)
    values = pd.concat(all_values, ignore_index=True)
    return values[KEY + [
        "section", "plot_order", "comparison_label", "n_structures", "n_groups", "n_seeds",
        "delta_mean", "comparator_mae", "crossed95_low", "crossed95_high", "valid_bootstrap_draws",
        "relative_delta_pct", "lower_95_pct", "upper_95_pct",
    ]]


def style_axis(ax, limits, ticks, tick_labels=True):
    ax.set(xlim=limits, xticks=ticks)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.65)
    ax.axvline(0, color=AXES, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.spines["bottom"].set_color("#A4AFB7")
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", width=0.65, length=3, color=AXES, labelbottom=tick_labels)
    if not tick_labels:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)


def save_figure(fig, stem):
    for ext, metadata in (
        ("png", {}),
        ("pdf", {"CreationDate": None}),
        ("svg", {"Date": None}),
    ):
        fig.savefig(HERE / f"{stem}.{ext}", dpi=250, metadata=metadata)
    plt.close(fig)


def render_endpoint_effects(seeds, summary):
    fig = plt.figure(figsize=(13.7, 9.0))
    grid = fig.add_gridspec(2, 3, left=0.225, right=0.97, bottom=0.115, top=0.805, hspace=0.40, wspace=0.16)
    fig.text(0.04, 0.949, "Ionisation inputs across seven PK endpoints", fontsize=20, fontweight="bold")
    fig.text(0.04, 0.902, "Small dots = paired seeds   •   Diamond = mean   •   Bar = ±1 seed SD", color=MUTED)

    endpoints = {**ENDPOINTS, AVERAGE: ("Seven-endpoint average", INK)}
    positions = [7, 6, 5, 4, 3, 2, 1, -0.25]
    jitter = np.array([-0.18, 0.14, -0.09, 0.18, 0.045, -0.135, 0.09, -0.045, 0.0])
    limits = (-20, 11.5)
    ticks = list(range(-20, 11, 5))
    if (
        min(seeds["relative_delta_pct"].min(), summary["lower_sd"].min()) <= limits[0]
        or max(seeds["relative_delta_pct"].max(), summary["upper_sd"].max()) >= limits[1]
    ):
        raise ValueError("An endpoint seed or SD range falls outside the plotting limits")

    for row, (cohort, cohort_label) in enumerate(COHORTS.items()):
        heading_y = (0.849, 0.453)[row]
        fig.text(0.04, heading_y, "AB"[row], fontsize=15, fontweight="bold")
        fig.text(0.063, heading_y, cohort_label, fontsize=13.5, fontweight="bold")
        label_start = (0.176, 0.222)[row]
        fig.text(label_start, heading_y, HEADLINE[cohort].label, fontsize=12, color=MUTED)
        for col, (model, (name, _)) in enumerate(MODELS.items()):
            ax = fig.add_subplot(grid[row, col])
            if row == 0:
                ax.set_title(name, fontweight="bold", fontsize=12, pad=7)
            ax.axhspan(-0.82, 0.30, color="#F1F4F6", zorder=0)
            ax.axhline(0.34, color=GRID, linewidth=0.65, zorder=1)
            stats = summary.loc[summary["cohort"].eq(cohort) & summary["backbone"].eq(model)].set_index("endpoint")
            points = seeds.loc[seeds["cohort"].eq(cohort) & seeds["backbone"].eq(model)].set_index(["endpoint", "seed"])
            for (endpoint, (_, color)), y in zip(endpoints.items(), positions):
                values = points.loc[endpoint, "relative_delta_pct"].loc[list(SEEDS)]
                stat = stats.loc[endpoint]
                ax.scatter(values, y + jitter, s=22, facecolor=color, edgecolor="white", linewidth=0.42, alpha=0.67, zorder=4)
                ax.errorbar(
                    stat["mean"], y, xerr=stat["seed_sd"], fmt="D", markersize=5.4, markerfacecolor=color,
                    markeredgecolor=INK, markeredgewidth=0.5, color=color, linewidth=1.5, capsize=2.5, zorder=5,
                )
                if endpoint == AVERAGE:
                    ax.text(0.97, y, f"{stat['mean']:+.2f}%", transform=ax.get_yaxis_transform(),
                            ha="right", va="center", color=INK, fontsize=10.5)
            style_axis(ax, limits, ticks, tick_labels=row == 1)
            ax.set(ylim=(-0.82, 7.57), yticks=positions)
            if col == 0:
                ax.set_yticklabels([
                    f"{name}  ({COUNTS[cohort][key]})" if key != AVERAGE else name
                    for key, (name, _) in endpoints.items()
                ])
                for tick, (_, color) in zip(ax.get_yticklabels(), endpoints.values()):
                    tick.set_color(color)
                ax.get_yticklabels()[-1].set_fontweight("bold")
            else:
                ax.tick_params(axis="y", labelleft=False)

    fig.text(0.5975, 0.066, "Change in MAE (% of mean comparator MAE)   •   Negative values indicate lower error",
             ha="center", fontsize=11.5)
    fig.text(0.04, 0.024, "Parentheses give endpoint label counts.  " + CONTEXT, fontsize=10.5, color=MUTED)
    save_figure(fig, "endpoint_effects")


def render_representation_checks(values):
    fig = plt.figure(figsize=(12.5, 10.5))
    fig.text(.04, .955, "Vd representation controls and tuning", fontsize=20, fontweight="bold")
    fig.text(.04, .923, "Dots: paired means  ·  Bars: exploratory 95% crossed-bootstrap intervals",
             fontsize=13, color=MUTED)

    limits = (-20.5, 10)
    ticks = [-20, -10, 0, 10]
    if values["lower_95_pct"].min() <= limits[0] or values["upper_95_pct"].max() >= limits[1]:
        raise ValueError("A crossed interval falls outside the plotting limits")
    if not values["relative_delta_pct"].between(values["lower_95_pct"], values["upper_95_pct"]).all():
        raise ValueError("A mean falls outside its displayed interval")

    for cohort_idx, (cohort, cohort_label) in enumerate(COHORTS.items()):
        group_top = (.87, .465)[cohort_idx]
        n = COUNTS[cohort]["Vd"]
        fig.text(.04, group_top, f"{cohort_label}  ·  {n} labelled structures",
                 fontsize=15, fontweight="bold")
        for row, section in enumerate(("Representation controls", "Tuning sensitivity")):
            panel = ("AB", "CD")[cohort_idx][row]
            bottom = group_top - (.21, .366)[row]
            height = (.140, .098)[row]
            y_title = bottom + height + (.03 if row == 0 else .017)
            fig.text(.04, y_title, f"{panel}   {section}", fontsize=13, fontweight="bold")
            if row == 1:
                fig.text(.26, y_title, HEADLINE[cohort].label, fontsize=12, color=MUTED)
            data = values.loc[values["cohort"].eq(cohort) & values["section"].eq(section)]
            n_rows = data["plot_order"].nunique()
            labels = ([c.label for c in CONTROLS[cohort]] if row == 0 else
                      ["Shared recipe", "Separate tuning"])
            for col, (model, (name, _)) in enumerate(MODELS.items()):
                ax = fig.add_axes([.26 + col*.238, bottom, .214, height])
                if row == 0:
                    ax.set_title(name, fontsize=13, fontweight="bold", pad=4)
                for order in range(n_rows):
                    y = n_rows - order - 1
                    if order % 2 == 0:
                        ax.axhspan(y - .47, y + .47, color=ROW, zorder=0)
                    r = data.loc[data.plot_order.eq(order) & data.backbone.eq(model)].iloc[0]
                    mean, low, high = r[["relative_delta_pct", "lower_95_pct", "upper_95_pct"]]
                    repeated_control = row == 0 and cohort == "numerical" and order == 2
                    color = CONTROL if repeated_control else BLUE
                    ax.errorbar(mean, y, xerr=[[mean - low], [high - mean]], fmt="o", markersize=5.6,
                                color=color, markerfacecolor="white" if repeated_control else color,
                                markeredgewidth=.8, linewidth=1.5, capsize=0, zorder=4)
                style_axis(ax, limits, ticks, tick_labels=row == 1)
                ax.set(ylim=(-.52, n_rows - .48), yticks=list(range(n_rows - 1, -1, -1)),
                       yticklabels=labels if col == 0 else [])
                ax.tick_params(axis="both", labelsize=12)

    fig.text(.26, .040, "Change in Vd MAE (% of comparator MAE)  ·  Negative values indicate lower error",
             fontsize=12, color=ENDPOINT_COLORS["Vd"])
    fig.text(.04, .017, "Open mark: four-pH profile versus repeated pH 7.4 control", fontsize=11, color=MUTED)
    save_figure(fig, "representation_checks")


CAPTIONS = """# Endpoint effects

Paired changes in held-out mean absolute error (MAE) for one-hot class at pH 7.4 versus structure only (A, class cohort) and the four-pH ionisation propensity profile versus the pH 7.4 propensity pair (B, numerical pKa cohort). Both contrasts use the primary shared recipe. Each dot represents one training seed with predictions pooled across the five grouped outer folds. Each arm-minus-comparator MAE difference is divided by the same nine-seed mean comparator MAE for its cohort, model and endpoint and multiplied by 100. Negative changes indicate lower error for the first arm. Diamonds show means and bars span one sample standard deviation across the nine paired seeds on either side of the mean. Dot offsets are for legibility. The seven-endpoint average gives equal weight to the seven normalized changes within each seed and then summarizes those nine seed averages. The number at the right of each average row gives its mean.

Parentheses give the number of labelled structures for each endpoint in the relevant cohort. Half-life, volume of distribution, clearance, peak concentration and time to peak use log10 MAE. Bioavailability and protein binding use percentage-point MAE. The comparisons pair models within each cohort and endpoint. All models use mass-balance weight λ = 0 and explicit supervised endpoint weights of 1.

# Representation checks

Paired changes in held-out volume-of-distribution log10 MAE for ionisation representation controls on the class (A) and numerical pKa (C) cohorts, and for the headline contrasts under the shared recipe and independent tuning in the class (B) and numerical pKa (D) cohorts. The charge-sign scalar encodes anion as −1, cation as +1 and neutral or zwitterionic class as 0. The four-class one-hot vector and the 16-dimensional learned embedding use the same audited classes at pH 7.4. The four-pH profile evaluates the acidic and basic propensities at pH 2.0, 5.0, 6.5 and 7.4, while the repeated-pair control repeats the pH 7.4 pair four times. The class tuning check reuses the already tuned structure-only comparator. The numerical tuning check tunes each representation independently. Comparisons use matched structures within each cohort.

Points give the mean paired arm-minus-comparator MAE change. Bars show approximate exploratory 95% equal-tailed percentile intervals from 10,000 paired crossed bootstrap draws over whole seed trajectories and frozen parent/scaffold groups, conditional on the observed training data, fixed folds and selected settings. The point estimate and the absolute-delta interval limits are scaled by 100 divided by the fixed nine-seed mean comparator MAE for that model and contrast. The denominator is held constant during scaling, so the bars are scaled absolute-delta intervals. Negative changes indicate lower error for the first arm. The shared-recipe headline analyses use separate multiplicity-adjusted intervals for the prespecified tests. Each seed pools predictions across five grouped outer folds, with nine seeds per arm. All models use mass-balance weight λ = 0 and explicit supervised endpoint weights of 1.
"""


def main():
    tables, scoring, sources = read_inputs()
    seeds, endpoints = endpoint_values(tables)
    representations = representation_values(tables)
    plt.rcParams.update({
        "font.family": FONT,
        "font.size": 11, "xtick.labelsize": 10.5, "ytick.labelsize": 11,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "ionisation_results_v1_support",
    })
    render_endpoint_effects(seeds, endpoints)
    render_representation_checks(representations)
    csv_outputs = {
        "endpoint_effects_seed_values.csv": seeds,
        "endpoint_effects_summary.csv": endpoints,
        "representation_checks_values.csv": representations,
    }
    for filename, data in csv_outputs.items():
        data.to_csv(HERE / filename, index=False)
    (HERE / "captions.md").write_text(CAPTIONS, encoding="utf-8")
    outputs = [
        f"{stem}.{ext}"
        for stem in ("endpoint_effects", "representation_checks")
        for ext in ("png", "pdf", "svg")
    ] + list(csv_outputs) + ["captions.md"]
    manifest = {
        "campaign_id": scoring["campaign_id"],
        "release_fingerprints": scoring["release_fingerprints"],
        "inputs_sha256": sources,
        "render_script_sha256": digest(Path(__file__)),
        "outputs_sha256": {name: digest(HERE / name) for name in outputs},
        "seeds": SEEDS,
        "outer_folds": 5,
        "lambda": 0,
        "endpoint_loss_weights": [1] * len(ENDPOINTS),
        "labelled_structure_counts": COUNTS,
        "endpoint_rows": len(seeds),
        "endpoint_summary_rows": len(endpoints),
        "representation_rows": len(representations),
        "endpoint_normalization": "100 × seed paired delta / mean comparator MAE over nine seeds",
        "endpoint_average": "Equal-weight mean of seven normalized changes within each seed",
        "endpoint_error_bars": "Mean ± sample SD across nine seeds (ddof=1)",
        "representation_error_bars": "Exploratory crossed 95% absolute-delta bounds × 100 / fixed mean comparator MAE",
        "bootstrap_draws": scoring["bootstrap_draws"],
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Rendered two figures from {len(seeds)} seed rows and {len(representations)} Vd representation rows.")


if __name__ == "__main__":
    main()
