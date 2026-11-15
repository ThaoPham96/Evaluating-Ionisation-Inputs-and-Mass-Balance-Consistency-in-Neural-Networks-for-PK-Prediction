"""Render the five single-pH inputs, including the completed pH 8.0 extension."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ionisation_style import AXES, DIVERGING, FONT, GRID, INK, MUTED, ENDPOINT_COLORS
EXPERIMENT = HERE.parents[1]
SCORING = EXPERIMENT / "outputs/single_ph/scoring"
PH8_SCORING = EXPERIMENT / "outputs/ph8/scoring"
RELEASE = "99e83f5ad3b3802ab2374777eff38f8d9d66e2178237c0e1ea5f23d143948cb8"
PRIOR_CAMPAIGN = "3ffed2efa5ca4a2bdfa3584cb0d1d3d2869d6edc17e1697e6476fa5a356b5e28"
MODELS = {"GraphConv": "GraphConv", "Chemprop": "Chemprop", "ChemBERTa_head": "Frozen ChemBERTa + MLP"}
ARMS = {"PH2": "pH 2.0 only", "PH5": "pH 5.0 only", "PH65": "pH 6.5 only", "REF74": "pH 7.4 only", "PH8": "pH 8.0 only"}
PAIRS = {"REF0": tuple(ARMS), "REF74": tuple(arm for arm in ARMS if arm != "REF74")}
ENDPOINTS = {
    "t_half": ("Half-life", ENDPOINT_COLORS["t_half"], 316, 230),
    "Vd": ("Vd", ENDPOINT_COLORS["Vd"], 259, 196),
    "CL": ("CL", ENDPOINT_COLORS["CL"], 246, 185),
    "Cmax": ("Cmax", ENDPOINT_COLORS["Cmax"], 190, 147),
    "Tmax": ("Tmax", ENDPOINT_COLORS["Tmax"], 249, 189),
    "F": ("F", ENDPOINT_COLORS["F"], 172, 135),
    "PPB": ("PPB", ENDPOINT_COLORS["PPB"], 292, 224),
}
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909)
KEYS = ["backbone", "endpoint", "arm", "comparator_arm"]
COHORT_NOTE = "Numerical-pKa cohort  •  327 structures in 238 groups  •  Five grouped outer folds  •  Nine paired seeds"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def checked_tables():
    """Join verified comparisons, retaining each shared reference contrast once."""
    old_est, old_seeds, old_fits, old_manifest, old_hashes = load_scores(
        SCORING, ("PH2", "PH5", "PH65"), 44145, 675, 405)
    new_est, new_seeds, new_fits, new_manifest, new_hashes = load_scores(
        PH8_SCORING, ("PH8",), 26487, 405, 135)
    for old, new, keys in ((old_est, new_est, KEYS),
                           (old_seeds, new_seeds, [*KEYS, "seed"]),
                           (old_fits, new_fits, ["backbone", "arm", "outer_fold", "seed"])):
        ref_arms = ("REF0", "REF74") if "endpoint" not in keys else ("REF74",)
        left = old.loc[old.arm.isin(ref_arms)].set_index(keys).sort_index()
        right = new.loc[new.arm.isin(ref_arms)].set_index(keys).sort_index()
        pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-12, atol=1e-12)
    estimates = pd.concat([old_est, new_est.loc[new_est.arm.eq("PH8")]], ignore_index=True)
    seeds = pd.concat([old_seeds, new_seeds.loc[new_seeds.arm.eq("PH8")]], ignore_index=True)
    fits = pd.concat([old_fits, new_fits.loc[new_fits.arm.eq("PH8")]], ignore_index=True)
    if len(fits) != 810 or fits.duplicated(["backbone", "arm", "outer_fold", "seed"]).any():
        raise ValueError("Expected 810 unique final source fits across six representations")
    validate_values(estimates, seeds)
    metadata = dict(
        campaign_ids={"single_ph": old_manifest["campaign_id"], "ph8": new_manifest["campaign_id"]},
        prior_campaign_id=PRIOR_CAMPAIGN, release_fingerprints={"numerical": RELEASE},
        prediction_fit_count=len(fits), n_structures=327, n_groups=238,
    )
    metadata["comparison_id"] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    return estimates, seeds, metadata, old_hashes | new_hashes


def load_scores(directory, new_arms, n_predictions, n_fits, n_new):
    manifest_path = directory / "scoring_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Scoring is incomplete: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "status": "report_ready", "cohort": "numerical", "n_structures": 327, "n_groups": 238,
        "release_fingerprints": {"numerical": RELEASE}, "prior_campaign_id": PRIOR_CAMPAIGN,
        "outer_folds": [1, 2, 3, 4, 5], "seeds": list(SEEDS), "bootstrap_draws": 10000,
        "interval_level": 0.95, "primary_family_size": 0,
        "predictions_rows": n_predictions, "prediction_fit_count": n_fits,
        "n_new_prediction_fits": n_new, "n_reused_prediction_fits": 270,
        "bootstrap_seed": 20261113,
    }
    if any(manifest.get(key) != value for key, value in expected.items()) or not manifest.get("campaign_id"):
        raise ValueError("Unexpected single-pH scoring identity")
    sources = {manifest_path.relative_to(EXPERIMENT).as_posix(): digest(manifest_path)}
    for name, expected_hash in manifest["files"].items():
        path = directory / name
        sources[path.relative_to(EXPERIMENT).as_posix()] = digest(path)
        if digest(path) != expected_hash:
            raise ValueError(f"Score checksum mismatch: {name}")
    table_names = ("contrasts.csv", "contrast_seeds.csv", "prediction_sources.csv")
    if not set(table_names).issubset(manifest["files"]):
        raise ValueError("Scoring manifest is missing required plot inputs")
    estimates, seeds, fits = (pd.read_csv(directory / name, float_precision="round_trip") for name in table_names)
    expected_keys = {(model, endpoint, arm, comparator) for model in MODELS for endpoint in ENDPOINTS
                     for comparator, arms in {"REF0": ("REF74", *new_arms), "REF74": new_arms}.items()
                     for arm in arms}
    if len(estimates) != len(expected_keys) or set(estimates[KEYS].itertuples(index=False, name=None)) != expected_keys:
        raise ValueError(f"Unexpected contrast coverage in {directory}")
    expected_seed_keys = {(*row, seed) for row in expected_keys for seed in SEEDS}
    if len(seeds) != len(expected_seed_keys) or set(seeds[[*KEYS, "seed"]].itertuples(index=False, name=None)) != expected_seed_keys:
        raise ValueError(f"Unexpected seed coverage in {directory}")
    validate_fits(fits, manifest["campaign_id"], new_arms, n_fits)
    return estimates, seeds, fits, manifest, sources


def validate_fits(fits, campaign_id, new_arms, n_fits):
    keys = ["backbone", "arm", "outer_fold", "seed"]
    expected = {(model, arm, fold, seed) for model in MODELS for arm in ("REF0", "REF74", *new_arms)
                for fold in range(1, 6) for seed in SEEDS}
    if len(fits) != n_fits or set(fits[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("Missing or duplicated original and single-pH fit references")
    reused = fits.arm.isin(("REF0", "REF74"))
    if (
        not fits.cohort.eq("numerical").all() or not fits.procedure.eq("primary").all()
        or not fits.loc[reused, "source_campaign_id"].eq(PRIOR_CAMPAIGN).all()
        or not fits.loc[~reused, "source_campaign_id"].eq(campaign_id).all()
        or fits.source_fit_id.isna().any() or fits.duplicated(["source_campaign_id", "source_fit_id"]).any()
    ):
        raise ValueError("Unexpected prediction source mapping")


def validate_values(estimates, seeds):
    expected = {(model, endpoint, arm, comparator) for model in MODELS for endpoint in ENDPOINTS
                for comparator, arms in PAIRS.items() for arm in arms}
    if len(estimates) != len(expected) or set(estimates[KEYS].itertuples(index=False, name=None)) != expected:
        raise ValueError("Expected 189 complete endpoint/model/arm/comparator estimates")
    expected_seeds = {(*row, seed) for row in expected for seed in SEEDS}
    if len(seeds) != len(expected_seeds) or set(seeds[[*KEYS, "seed"]].itertuples(index=False, name=None)) != expected_seeds:
        raise ValueError("Expected nine unique paired seed values for every contrast")
    for frame in (estimates, seeds):
        flags = {
            "cohort": "numerical", "procedure": "primary", "comparator_procedure": "primary",
            "scope": "all", "stratum": "all", "purpose": "exploratory", "primary_test": False,
        }
        if any(not frame[key].eq(value).all() for key, value in flags.items()) or not frame.backbone.eq(frame.comparator_backbone).all():
            raise ValueError("Expected only same-model exploratory contrasts")
    for field, index in (("n_structures", 2), ("n_groups", 3)):
        np.testing.assert_array_equal(estimates[field], estimates.endpoint.map({key: row[index] for key, row in ENDPOINTS.items()}))
    if not estimates.n_seeds.eq(9).all() or not estimates.valid_bootstrap_draws.eq(10000).all():
        raise ValueError("Incomplete paired seed or bootstrap counts")
    value_columns = ["relative_delta_pct", "crossed95_low_pct", "crossed95_high_pct", "comparator_mae"]
    if (
        not np.isfinite(estimates[value_columns]).all().all() or not np.isfinite(seeds.relative_delta_pct).all()
        or not estimates.comparator_mae.gt(0).all() or not estimates.crossed95_low_pct.le(estimates.crossed95_high_pct).all()
    ):
        raise ValueError("Undefined mean, interval, or fixed comparator MAE")
    table = estimates.set_index(KEYS).sort_index()
    grouped = seeds.groupby(KEYS, sort=True)
    np.testing.assert_allclose(table.relative_delta_pct, grouped.relative_delta_pct.mean(), atol=1e-12)
    np.testing.assert_allclose(table.comparator_mae, grouped.comparator_mae.mean(), atol=1e-12)
    np.testing.assert_allclose(seeds.relative_delta_pct, 100*seeds.delta/seeds.mean_comparator_mae, atol=1e-12)
    np.testing.assert_allclose(table.relative_delta_pct, 100*table.delta_mean/table.comparator_mae, atol=1e-12)
    for side in ("low", "high"):
        np.testing.assert_allclose(table[f"crossed95_{side}_pct"], 100*table[f"crossed95_{side}"]/table.comparator_mae, atol=1e-12)


def percent(value, digits):
    if round(value, digits) == 0:
        return f"{0:.{digits}f}%"
    return f"{value:+.{digits}f}%".replace("-", "−")


def heatmap(estimates):
    fig = plt.figure(figsize=(15.7, 6.2), facecolor="white")
    fig.text(0.035, 0.928, "Single-pH categories across seven PK endpoints", fontsize=22, fontweight="bold")
    fig.text(0.035, 0.863, "Paired mean MAE changes (%) from the same model’s structure-only reference (REF0).", fontsize=12, color=MUTED)
    table = estimates.loc[estimates.comparator_arm.eq("REF0")]
    limit = max(5, 5*np.ceil(table.relative_delta_pct.abs().max()/5))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    cmap = LinearSegmentedColormap.from_list("relative_error", DIVERGING)
    grid = fig.add_gridspec(1, 3, left=0.191, right=0.972, bottom=0.270, top=0.735, wspace=0.08)
    n_rows, n_cols = len(ARMS), len(ENDPOINTS)
    order = pd.MultiIndex.from_product([ARMS, ENDPOINTS], names=["arm", "endpoint"])
    for i, (model, label) in enumerate(MODELS.items()):
        ax = fig.add_subplot(grid[0, i])
        values = table.loc[table.backbone.eq(model)].set_index(["arm", "endpoint"]).loc[order, "relative_delta_pct"].to_numpy().reshape(n_rows, n_cols)
        mesh = ax.pcolormesh(np.arange(n_cols+1), np.arange(n_rows+1), values, cmap=cmap, norm=norm,
                             edgecolors="white", linewidths=0.7, shading="flat")
        for row, col in np.ndindex(values.shape):
            value = values[row, col]
            r, g, b, _ = cmap(norm(value))
            foreground = "white" if 0.2126*r + 0.7152*g + 0.0722*b < 0.53 else INK
            ax.text(col+0.5, row+0.5, percent(value, 1).removesuffix("%"), ha="center", va="center", fontsize=14.4, color=foreground)
        ax.set(xlim=(0, n_cols), ylim=(n_rows, 0), xticks=np.arange(n_cols)+0.5, yticks=np.arange(n_rows)+0.5)
        ax.set_xticklabels([f"{name}\n{count}" for name, _, count, _ in ENDPOINTS.values()], fontsize=13.5, linespacing=1.0)
        ax.set_yticklabels(ARMS.values() if i == 0 else [], fontsize=11)
        ax.xaxis.tick_top()
        for tick, color in zip(ax.get_xticklabels(), ENDPOINT_COLORS.values()):
            tick.set_color(color)
        ax.tick_params(length=0, pad=5)
        ax.set_title(label, fontsize=15, fontweight="bold", pad=34)
        for spine in ax.spines.values():
            spine.set_visible(False)
    colour_axis = fig.add_axes([0.337, 0.192, 0.426, 0.027])
    bar = fig.colorbar(mesh, cax=colour_axis, orientation="horizontal", ticks=[-limit, 0, limit],
                       format=FuncFormatter(lambda value, _: percent(value, 0)))
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor("face")
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=3, pad=4, labelsize=10.5)
    fig.text(0.325, 0.204, "Lower error", ha="right", va="center", color=MUTED, fontsize=12)
    fig.text(0.775, 0.204, "Higher error", ha="left", va="center", color=MUTED, fontsize=12)
    fig.text(0.035, 0.105, "Each model receives the four-category input at one pH. Numbers below endpoints give labelled structures.", fontsize=12, color=MUTED)
    fig.text(0.035, 0.054, COHORT_NOTE, fontsize=12, color=MUTED)
    return fig, float(limit)


def forest(estimates, seeds, comparator):
    baseline = "structure-only models" if comparator == "REF0" else "the pH 7.4 category"
    fig = plt.figure(figsize=(15.7, 7.95), facecolor="white")
    fig.text(0.035, 0.938, f"Vd and clearance compared with {baseline}", fontsize=22, fontweight="bold")
    fig.text(0.035, 0.887, "Small dots show nine paired seeds. Diamonds show means. Bars show exploratory crossed-bootstrap 95% intervals.", fontsize=11.8, color=MUTED)
    grid = fig.add_gridspec(2, 3, left=0.191, right=0.972, bottom=0.162, top=0.790, hspace=0.30, wspace=0.145)
    arms = PAIRS[comparator]
    row_positions = np.arange(len(arms))[::-1]
    jitter = np.array([-0.21, 0.158, -0.105, 0.21, 0.052, -0.158, 0.105, -0.052, 0.0])
    limits = {}
    for row, endpoint in enumerate(("Vd", "CL")):
        short, color, count, _ = ENDPOINTS[endpoint]
        title = "Volume of distribution" if endpoint == "Vd" else "Clearance"
        y = (0.825, 0.456)[row]
        fig.text(0.035, y, "AB"[row], fontsize=14, fontweight="bold")
        fig.text(0.057, y, f"{title} ({short}) · {count} labelled structures", fontsize=13, fontweight="bold", color=color)
        table = estimates.loc[estimates.comparator_arm.eq(comparator) & estimates.endpoint.eq(endpoint)]
        point_table = seeds.loc[seeds.comparator_arm.eq(comparator) & seeds.endpoint.eq(endpoint)]
        low = min(0, table.crossed95_low_pct.min(), table.relative_delta_pct.min(), point_table.relative_delta_pct.min())
        high = max(0, table.crossed95_high_pct.max(), table.relative_delta_pct.max(), point_table.relative_delta_pct.max())
        padding = max(0.25, 0.065*(high-low))
        ticks = MaxNLocator(nbins=5).tick_values(low-padding, high+padding)
        xmin, xmax = float(ticks[0]), float(ticks[-1])
        limits[endpoint] = [xmin, xmax]
        for col, (model, name) in enumerate(MODELS.items()):
            ax = fig.add_subplot(grid[row, col])
            if row == 0:
                ax.set_title(name, fontsize=14, fontweight="bold", pad=8)
            stats = table.loc[table.backbone.eq(model)].set_index("arm")
            points = point_table.loc[point_table.backbone.eq(model)].set_index(["arm", "seed"])
            for arm, ypos in zip(arms, row_positions):
                estimate = stats.loc[arm]
                x = points.loc[arm, "relative_delta_pct"].loc[list(SEEDS)].to_numpy()
                ax.scatter(x, ypos+jitter, s=24, facecolor=color, edgecolor="white", linewidth=0.45, alpha=0.62, zorder=4)
                ax.hlines(ypos, estimate.crossed95_low_pct, estimate.crossed95_high_pct, color=color, linewidth=1.8, zorder=3)
                ax.plot(estimate.relative_delta_pct, ypos, marker="D", ms=5.6, mfc=color, mec=INK, mew=0.5, zorder=5)
            ax.set(xlim=(xmin, xmax), ylim=(-0.50, len(arms)-0.50), xticks=ticks, yticks=row_positions)
            ax.tick_params(axis="y", length=0, pad=8, labelsize=13, labelleft=col == 0)
            ax.tick_params(axis="x", length=3, pad=3, labelsize=12)
            ax.set_axisbelow(True)
            ax.grid(axis="x", color=GRID, linewidth=0.7)
            ax.axvline(0, color=AXES, linewidth=0.95, linestyle=(0, (4, 3)), zorder=1)
            ax.spines[["left", "right", "top"]].set_visible(False)
            ax.spines["bottom"].set_color("#9CAAB0")
            if col == 0:
                ax.set_yticklabels([ARMS[arm] for arm in arms])
    fig.text(0.558, 0.095, f"Change in log10 MAE (% of mean {comparator} MAE)   •   Negative values indicate lower error",
             ha="center", fontsize=11.5)
    fig.text(0.035, 0.049, COHORT_NOTE, fontsize=12, color=MUTED)
    return fig, limits


CAPTIONS = """SINGLE-pH IONISATION RESULTS

Page 1. All seven endpoints. Each cell gives the mean of nine paired MAE changes from the structure-only reference (REF0) for the same backbone. Each ionisation arm receives a single four-channel category input at pH 2.0, 5.0, 6.5, 7.4 or 8.0. The colour scale is shared across all models and endpoints. MAE uses log10 units for the five positive endpoints and percentage points for F and PPB. The figure shows point estimates.

Page 2. Vd and clearance versus structure only. Each pH arm is compared with REF0 on the same structures. Vd has 259 observed labels in 196 groups. Clearance has 246 observed labels in 185 groups.

Page 3. Vd and clearance versus the pH 7.4 category. Each of the four other pHs replaces the pH 7.4 category block at the same four-channel width. The pH 7.4 reference is REF74.

For pages 2 and 3, small dots show all nine paired seed changes, diamonds show means, and bars show exploratory, unadjusted 95% intervals. Intervals resample whole seed trajectories and frozen parent/scaffold groups (10,000 draws), conditional on observed data, fixed folds and selected settings. The x-axis limits are common across backbones within each endpoint and comparison. Each paired absolute difference and interval bound is divided by the fixed nine-seed mean comparator MAE for the same model and endpoint and multiplied by 100. Negative values indicate lower error.

All comparisons use the 327-structure numerical-pKa subset with 238 frozen groups. Each seed pools held-out predictions from five grouped outer folds. REF0 and REF74 reuse the original reference predictions. The single-pH categories are based on archived acidic/basic pKa descriptors. The outcomes are the original observed PK endpoints. All fits use zero mass-balance penalty and equal endpoint weights.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate scoring inputs without writing figures")
    args = parser.parse_args()
    estimates, seeds, scoring, inputs = checked_tables()
    validate_values(estimates, seeds)
    if args.check:
        print(f"Verified {len(estimates)} complete paired estimates and {len(seeds)} seed values.")
        return

    plt.rcParams.update({
        "font.family": FONT, "font.size": 11, "text.color": INK, "xtick.color": INK,
        "ytick.color": INK, "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "single-ph-v2",
    })
    pages, outputs, scales = [], [], {}
    pdf_path = HERE / "single_ph_results.pdf"
    with PdfPages(pdf_path, metadata={"Title": "Single-pH ionisation results", "CreationDate": None}) as pdf:
        for stem, comparator in (("all_endpoint_mae", None), ("vd_cl_vs_structure", "REF0"), ("vd_cl_vs_ph74", "REF74")):
            if comparator is None:
                fig, scale = heatmap(estimates)
            else:
                fig, scale = forest(estimates, seeds, comparator)
            pdf.savefig(fig)
            png, svg = HERE / f"{stem}.png", HERE / f"{stem}.svg"
            fig.savefig(png, dpi=210)
            fig.savefig(svg, metadata={"Date": None})
            plt.close(fig)
            pages.append({"page": len(pages)+1, "png": png.name, "svg": svg.name})
            outputs.extend((png, svg))
            scales[stem] = scale
    outputs.append(pdf_path)
    for filename, table in (("plotted_estimates.csv", estimates), ("plotted_seeds.csv", seeds)):
        path = HERE / filename
        table.to_csv(path, index=False)
        outputs.append(path)
    captions = HERE / "captions.txt"
    captions.write_text(CAPTIONS, encoding="utf-8")
    outputs.append(captions)
    manifest = {
        "comparison_id": scoring["comparison_id"], "campaign_ids": scoring["campaign_ids"], "prior_campaign_id": scoring["prior_campaign_id"],
        "release_fingerprints": scoring["release_fingerprints"], "seed_ids": SEEDS,
        "input_directories": [p.relative_to(EXPERIMENT).as_posix() for p in (SCORING, PH8_SCORING)], "inputs_sha256": inputs,
        "render_script_sha256": digest(Path(__file__)), "outputs_sha256": {p.name: digest(p) for p in outputs},
        "pages": pages, "plotted_scales": scales, "estimate_rows": len(estimates), "seed_rows": len(seeds),
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(f"Rendered {len(pages)} pages from {len(estimates)} estimates and {len(seeds)} paired seed values.")


if __name__ == "__main__":
    main()
