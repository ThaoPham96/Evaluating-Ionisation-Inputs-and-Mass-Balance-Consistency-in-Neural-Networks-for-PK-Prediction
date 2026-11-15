"""Render the four selected supplementary panels from frozen, verified results.

Run with Python 3.11+ and Matplotlib, NumPy and pandas. The script writes all
outputs next to itself and does not fit or rescore any models.
"""
from hashlib import file_digest
import json
from pathlib import Path, PureWindowsPath
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent
EXPERIMENT = FIGURES.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(FIGURES))
from ionisation_style import BASELINE, BLUE, CATEGORIES, DIVERGING, ENDPOINT_COLORS, FONT


OBSERVED = FIGURES / "ionisation_observed_predicted_v1"
SCORES = EXPERIMENT / "outputs/scoring"
SINGLE_PH = FIGURES / "single_ph_v2"
MODELS = {"GraphConv": "GraphConv", "Chemprop": "Chemprop", "ChemBERTa_head": "ChemBERTa"}
SCATTER_ARMS = {"REF0": "Structure only", "REF74": "+ ionisation (pH 7.4)"}
CLASS_ARMS = {"C0": "Structure only", "C1": "Charge-sign code", "C2": "One-hot category", "C3": "Learned embedding"}
PH_ARMS = {"PH2": "2.0", "PH5": "5.0", "PH65": "6.5", "REF74": "7.4", "PH8": "8.0"}
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909)
ENDPOINTS = {
    "t_half": ("t½", "t1/2(hour)", 316, 230),
    "Vd": ("Vd", "VD(liter)", 259, 196),
    "CL": ("CL", "Cl(liter/hour)", 246, 185),
    "Cmax": ("Cmax", "Cmax_uM", 190, 147),
    "Tmax": ("Tmax", "Tmax", 249, 189),
    "F": ("F", "F(percentage)", 172, 135),
    "PPB": ("PPB", "PPB(percentage)", 292, 224),
}
SCATTERS = {"Vd": ("Volume of distribution", "L", (1, 2, 3, 4, 5)),
            "CL": ("Clearance", "L/h", (-2, -1, 0, 1, 2, 3))}
WIDTH, HEIGHT = 7.4, 9.40
SCALE = 15.0
STEM = "ionisation_supplementary_selected"
GRID = "#E6E6E6"
AXES = "#777777"
plt.rcParams.update({"font.family": FONT, "font.size": 7.5,
                     "text.color": "black", "axes.labelcolor": "black",
                     "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": AXES,
                     "axes.linewidth": 0.50, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "svg.hashsalt": STEM})

CAPTION = """Supplementary Figure. Ionisation encodings and single-pH inputs for pharmacokinetic prediction.

A, Volume of distribution (Vd; n = 259). B, Clearance (CL; n = 246). Each row shows one model trained on structure alone (left) or structure plus a four-channel one-hot ionisation category at pH 7.4 (right). Each dot represents one structure with an observed endpoint. Predictions on the horizontal axis are averaged in log10 space over nine out-of-fold predictions, equivalent to a geometric mean in physical units. The dashed line denotes equality. R² is calculated from the displayed predictions. Colours identify the pH 7.4 categories in both columns; in the structure-only column, the categories annotate the plots. The four categories in A, B and D are defined by acidic and basic pKa thresholds.

C, Vd mean absolute error (MAE) for four encodings in the audited class cohort (494 structures; 388 with Vd). The charge-sign code assigns −1 to the audited anionic class, +1 to the cationic class and 0 to the neutral and zwitterionic classes. The one-hot category and the learned 16-dimensional embedding encode the same four audited pH 7.4 categories. Circles show nine seed values; diamonds and bars show the mean ± one sample standard deviation.

D, Change in MAE for one-hot ionisation categories supplied at one pH at a time. Values are the mean paired MAE difference across nine seeds, divided by the same model and endpoint’s nine-seed mean structure-only MAE and multiplied by 100. Negative values indicate lower error. Label counts are 316 (half-life), 259 (Vd), 246 (CL), 190 (Cmax), 249 (Tmax), 172 (bioavailability, F) and 292 (plasma protein binding, PPB). MAE uses log10 units for half-life, Vd, CL, Cmax and Tmax, and percentage points for F and PPB.

A, B and D use the numerical-pKa subset (327 structures in 238 groups). All panels use five grouped outer folds and nine training seeds. All explicit endpoint loss weights are one and the mass-balance penalty is zero. ChemBERTa uses a frozen encoder with a trained multilayer perceptron head.
"""


def record_hash(path, hashes, expected=None):
    with path.open("rb") as handle:
        digest = file_digest(handle, "sha256").hexdigest()
    if expected is not None and expected != digest:
        raise ValueError(f"SHA-256 mismatch: {path}")
    hashes[path.relative_to(EXPERIMENT).as_posix()] = digest


def read_manifest(path, hashes):
    record_hash(path, hashes)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path, hashes, expected, **kwargs):
    record_hash(path, hashes, expected)
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def assert_keys(frame, columns, expected):
    assert len(frame) == len(expected)
    assert set(frame[columns].itertuples(index=False, name=None)) == expected


def load_and_verify():
    hashes, checks = {}, {}
    for path in (FIGURES / "ionisation_style.py", Path(__file__).resolve()):
        record_hash(path, hashes)
    observed_source = read_manifest(OBSERVED / "manifest.json", hashes)
    ph_source = read_manifest(SINGLE_PH / "manifest.json", hashes)
    class_source = read_manifest(SCORES / "scoring_manifest.json", hashes)
    releases = {}
    for cohort in ("class", "numerical"):
        base = EXPERIMENT / "data/release" / cohort
        release = read_manifest(base / "manifest.json", hashes)
        assert release["fingerprint"] == class_source["release_fingerprints"][cohort]
        if cohort == "numerical":
            assert release["fingerprint"] == ph_source["release_fingerprints"][cohort]
        fields = ["structure_id", "outer_fold", "group_id", "ionisation_class", *[v[1] for v in ENDPOINTS.values()]]
        releases[cohort] = read_csv(base / "model_dataset.csv", hashes, release["files"]["model_dataset.csv"],
                                    usecols=fields, dtype={"structure_id": str}).set_index("structure_id")

    numerical, classes = releases["numerical"], releases["class"]
    assert len(numerical) == 327 and numerical.index.is_unique and numerical.group_id.nunique() == 238
    assert len(classes) == 494 and classes.index.is_unique and numerical.index.isin(classes.index).all()
    assert numerical.ionisation_class.isin(CATEGORIES).all()
    for frame in releases.values():
        assert set(frame.outer_fold) == set(range(1, 6))
    for key, (_, field, count, groups) in ENDPOINTS.items():
        labelled = numerical.loc[numerical[field].notna()]
        assert len(labelled) == count and labelled.group_id.nunique() == groups, key
    class_labels = classes.loc[classes["VD(liter)"].notna()]
    assert len(class_labels) == 388

    for original_path, checksum in observed_source["sources"].items():
        path = PureWindowsPath(original_path)
        li = path.parts.index(EXPERIMENT.name)
        record_hash(EXPERIMENT.joinpath(*path.parts[li + 1:]), hashes, checksum)
    assert observed_source["n_seeds"] == 9
    points, metrics = [read_csv(OBSERVED / n, hashes, observed_source["outputs"][n], dtype={"structure_id": str})
                       for n in ("plotted_points.csv", "metrics.csv")]
    points = points.loc[points.arm.isin(SCATTER_ARMS) & points.endpoint.isin(SCATTERS)]
    metrics = metrics.loc[metrics.arm.isin(SCATTER_ARMS) & metrics.endpoint.isin(SCATTERS)]
    assert_keys(metrics, ["endpoint", "backbone", "arm"],
                {(e, m, a) for e in SCATTERS for m in MODELS for a in SCATTER_ARMS})
    predictions = pd.read_csv(EXPERIMENT / "outputs/ph_followup/outer_predictions.csv", float_precision="round_trip",
                              usecols=["structure_id", "backbone", "arm", "seed", "procedure", "outer_fold", "pred_Vd", "pred_CL"],
                              dtype={"structure_id": str})
    predictions = predictions.loc[predictions.procedure.eq("primary") & predictions.arm.isin(SCATTER_ARMS)]
    keys = ["structure_id", "backbone", "arm"]
    assert_keys(predictions, [*keys, "seed"], {(s, m, a, seed) for s in numerical.index for m in MODELS for a in SCATTER_ARMS for seed in SEEDS})
    np.testing.assert_array_equal(predictions.outer_fold, predictions.structure_id.map(numerical.outer_fold))
    means = predictions.groupby(keys)[["pred_Vd", "pred_CL"]].mean()
    checks["scatter_identity_and_metrics"] = []
    for (endpoint, model, arm), frame in points.groupby(["endpoint", "backbone", "arm"]):
        _, field, count, _ = ENDPOINTS[endpoint]
        labels = numerical.loc[numerical[field].notna()]
        assert len(frame) == count and set(frame.structure_id) == set(labels.index)
        y, x = frame.observed_log10.to_numpy(), frame.predicted_log10.to_numpy()
        lo, hi = observed_source["axis_limits_log10"][endpoint]
        assert np.all((lo < y) & (y < hi) & (lo < x) & (x < hi))
        assert frame.physical_unit.eq(SCATTERS[endpoint][1]).all()
        for column in ("outer_fold", "ionisation_class"):
            np.testing.assert_array_equal(frame[column], frame.structure_id.map(labels[column]))
        np.testing.assert_allclose(y, np.log10(frame.structure_id.map(labels[field])), rtol=0, atol=1e-12)
        np.testing.assert_allclose(x, means.reindex(pd.MultiIndex.from_frame(frame[keys]))[f"pred_{endpoint}"], rtol=0, atol=1e-12)
        r2 = 1 - np.sum((y - x) ** 2) / np.sum((y - y.mean()) ** 2)
        mae = np.abs(y - x).mean()
        source = metrics.loc[metrics.endpoint.eq(endpoint) & metrics.backbone.eq(model) & metrics.arm.eq(arm)].iloc[0]
        np.testing.assert_allclose([r2, mae], [source.r2_displayed_seed_mean, source.mae_log10_displayed_seed_mean], rtol=0, atol=1e-12)
        assert source.n == count
        checks["scatter_identity_and_metrics"].append(dict(endpoint=endpoint, model=model, arm=arm, n=count, r2=r2, mae=mae))
    assert len(points) == 3030 and len(checks["scatter_identity_and_metrics"]) == 12

    enc_seeds, enc_summary = [read_csv(SCORES / n, hashes, class_source["files"][n]).query(
        "cohort == 'class' and procedure == 'primary' and endpoint == 'Vd' and arm in @class_arms",
        local_dict={"class_arms": list(CLASS_ARMS)}) for n in ("endpoint_metrics.csv", "endpoint_summary.csv")]
    enc_keys = ["backbone", "arm"]
    assert_keys(enc_summary, enc_keys, {(m, a) for m in MODELS for a in CLASS_ARMS})
    assert_keys(enc_seeds, [*enc_keys, "seed"], {(m, a, s) for m in MODELS for a in CLASS_ARMS for s in SEEDS})
    for frame in (enc_seeds, enc_summary):
        assert frame.n_structures.eq(len(class_labels)).all()
        assert frame.n_groups.eq(class_labels.group_id.nunique()).all()
        assert frame.unit.eq("log10").all()
    grouped = enc_seeds.groupby(enc_keys).mae
    class_stats = enc_summary.set_index(enc_keys).sort_index()
    assert class_stats.n_seeds.eq(len(SEEDS)).all()
    np.testing.assert_allclose(grouped.mean(), class_stats.mae_mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(grouped.std(ddof=1), class_stats.mae_seed_sd, rtol=0, atol=1e-12)
    checks["class_encoding"] = {"structures": len(classes), "Vd_labels": len(class_labels),
                                "Vd_groups": int(class_labels.group_id.nunique()), "seed_rows": len(enc_seeds),
                                "mean_sd_rows_verified": len(enc_summary)}

    assert ph_source["seed_ids"] == list(SEEDS)
    ph_summary, ph_seeds = [read_csv(SINGLE_PH / n, hashes, ph_source["outputs_sha256"][n]).query("comparator_arm == 'REF0'")
                            for n in ("plotted_estimates.csv", "plotted_seeds.csv")]
    ph_keys = ["backbone", "arm", "endpoint"]
    assert_keys(ph_summary, ph_keys, {(m, a, e) for m in MODELS for a in PH_ARMS for e in ENDPOINTS})
    assert_keys(ph_seeds, [*ph_keys, "seed"], {(m, a, e, s) for m in MODELS for a in PH_ARMS for e in ENDPOINTS for s in SEEDS})
    for frame in (ph_summary, ph_seeds):
        assert frame.cohort.eq("numerical").all() and frame.procedure.eq("primary").all()
        assert frame.backbone.eq(frame.comparator_backbone).all() and frame.comparator_procedure.eq("primary").all()
        assert frame.scope.eq("all").all() and frame.stratum.eq("all").all()
    assert ph_summary.n_seeds.eq(len(SEEDS)).all()
    for name, i in (("n_structures", 2), ("n_groups", 3)):
        np.testing.assert_array_equal(ph_summary[name], ph_summary.endpoint.map({e: v[i] for e, v in ENDPOINTS.items()}))
    grouped = ph_seeds.groupby(ph_keys)
    ph_stats = ph_summary.set_index(ph_keys).sort_index()
    np.testing.assert_allclose(grouped.delta.mean(), ph_stats.delta_mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(grouped.comparator_mae.mean(), ph_stats.comparator_mae, rtol=0, atol=1e-12)
    np.testing.assert_allclose(grouped.relative_delta_pct.mean(), ph_stats.relative_delta_pct, rtol=0, atol=1e-12)
    np.testing.assert_allclose(100 * ph_stats.delta_mean / ph_stats.comparator_mae, ph_stats.relative_delta_pct, rtol=0, atol=1e-12)
    np.testing.assert_allclose(100 * ph_seeds.delta / ph_seeds.mean_comparator_mae, ph_seeds.relative_delta_pct, rtol=0, atol=1e-12)
    assert ph_summary.relative_delta_pct.abs().max() < SCALE
    checks["single_pH"] = {"structures": len(numerical), "groups": int(numerical.group_id.nunique()),
                           "estimates_verified": len(ph_summary), "seed_rows": len(ph_seeds),
                           "min_percent": ph_summary.relative_delta_pct.min(), "max_percent": ph_summary.relative_delta_pct.max(),
                           "label_counts": {e: v[2] for e, v in ENDPOINTS.items()}}
    tables = {"scatter_points.csv": points, "scatter_metrics.csv": metrics, "encoding_seed_values.csv": enc_seeds,
              "encoding_summary.csv": enc_summary, "single_ph_estimates.csv": ph_summary, "single_ph_seed_values.csv": ph_seeds}
    return tables, observed_source["axis_limits_log10"], hashes, checks


def axes(fig, x, y, width, height):
    return fig.add_axes([x / WIDTH, y / HEIGHT, width / WIDTH, height / HEIGHT])


def label(fig, x, y, value, **kwargs):
    return fig.text(x / WIDTH, y / HEIGHT, value, **kwargs)


def heading(fig, x, y, letter, title, count=None):
    label(fig, x, y, letter, fontsize=11.5, fontweight="bold", va="baseline")
    text = label(fig, x + 0.20, y, title, fontsize=10, fontweight="bold", va="baseline")
    if count:
        fig.canvas.draw()
        x_end = text.get_window_extent().x1 / fig.dpi
        label(fig, x_end + 0.09, y, f"n = {count}", fontsize=7.8, va="baseline")


def draw_scatter(fig, points, metrics, limits):
    x_positions = ((1.01, 2.48), (4.35, 5.82))
    y_positions = (7.30, 5.83, 4.36)
    side = 1.34
    for e_i, (endpoint, (title, unit, ticks)) in enumerate(SCATTERS.items()):
        left, right = x_positions[e_i]
        centre = (left + right + side) / 2
        label(fig, (0.07, 3.95)[e_i], 9.05, "AB"[e_i], fontsize=11.5,
              fontweight="bold", va="baseline")
        label(fig, centre, 9.05, title, fontsize=10, fontweight="bold", va="baseline", ha="center")
        fig.add_artist(Line2D([left / WIDTH, (right + side) / WIDTH],
                              [8.965 / HEIGHT, 8.965 / HEIGHT], transform=fig.transFigure,
                              color=ENDPOINT_COLORS[endpoint], linewidth=1.8,
                              solid_capstyle="butt"))
        label(fig, (left + right + side) / 2, 4.03, f"Predicted log₁₀ {endpoint} ({unit})", ha="center", fontsize=8)
        label(fig, left - (0.23 if e_i == 0 else 0.36), 6.50, f"Observed log₁₀ {endpoint} ({unit})",
              ha="center", va="center", rotation=90, fontsize=8)
        for a_i, (arm, arm_label) in enumerate(SCATTER_ARMS.items()):
            x = x_positions[e_i][a_i]
            label(fig, x + side / 2, 8.80, arm_label, ha="center", fontsize=8, fontweight="bold")
            for m_i, (model, model_name) in enumerate(MODELS.items()):
                y = y_positions[m_i]
                if e_i == a_i == 0:
                    label(fig, 0.07, y + side / 2, model_name, va="center", fontsize=7.8, fontweight="bold")
                ax = axes(fig, x, y, side, side)
                frame = points.loc[points.endpoint.eq(endpoint) & points.backbone.eq(model) & points.arm.eq(arm)]
                r = metrics.loc[metrics.endpoint.eq(endpoint) & metrics.backbone.eq(model) & metrics.arm.eq(arm)].iloc[0]
                lo, hi = limits[endpoint]
                ax.set(xlim=(lo, hi), ylim=(lo, hi), xticks=ticks, yticks=ticks)
                ax.set_aspect("equal")
                ax.set_axisbelow(True)
                ax.grid(color=GRID, linewidth=0.40)
                ax.spines[["right", "top"]].set_visible(False)
                ax.plot([lo, hi], [lo, hi], color=AXES, linewidth=0.60, linestyle=(0, (3, 2)), zorder=1)
                for category, (_, color, _) in CATEGORIES.items():
                    dots = frame.loc[frame.ionisation_class.eq(category)]
                    ax.scatter(dots.predicted_log10, dots.observed_log10, s=9, alpha=0.77, color=color,
                               marker="o", edgecolors="none", linewidths=0, zorder=3)
                ax.tick_params(length=2, width=0.45, pad=2, labelsize=7, labelleft=a_i == 0,
                               labelbottom=m_i == len(MODELS) - 1)
                ax.text(0.955, 0.060, f"R² = {r.r2_displayed_seed_mean:.3f}", transform=ax.transAxes,
                        fontsize=7.2, ha="right", va="bottom", zorder=5)
    handles = [Line2D([], [], color=color, marker="o", linestyle="none", markersize=4.7,
                      markeredgewidth=0, label=name) for name, color, _ in CATEGORIES.values()]
    fig.legend(handles=handles, bbox_to_anchor=(0.538, 3.79 / HEIGHT), loc="center", ncols=4, frameon=False,
               handletextpad=0.25, columnspacing=1.6, fontsize=7.6)


def draw_encodings(fig, seeds, summary):
    heading(fig, 0.07, 3.50, "C", "Categorical encodings for Vd: full class cohort")
    label(fig, 4.13, 1.95, "Vd MAE (log₁₀ units)", ha="center", fontsize=8)
    xlim = 0.37, 0.51
    jitter = np.array([-0.24, 0.18, -0.12, 0.24, 0.06, -0.18, 0.12, -0.06, 0.0])
    for i, (model, model_name) in enumerate(MODELS.items()):
        ax = axes(fig, 1.28 + i * 2.01, 2.28, 1.76, 0.95)
        ax.set_title(model_name, fontsize=8, fontweight="bold", pad=5)
        labels = range(len(CLASS_ARMS))
        ax.set(xlim=xlim, ylim=(3.52, -0.52), xticks=[.38, .42, .46, .50], yticks=list(labels))
        ax.set_yticklabels(CLASS_ARMS.values())
        ax.grid(axis="x", color=GRID, linewidth=0.50)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=2, width=.45, pad=2, labelsize=7)
        ax.tick_params(axis="y", length=0, pad=5, labelsize=7.7, labelleft=i == 0)
        ax.spines[["left", "right", "top"]].set_visible(False)
        for j, arm in enumerate(CLASS_ARMS):
            values = seeds.loc[seeds.backbone.eq(model) & seeds.arm.eq(arm)].set_index("seed").loc[list(SEEDS), "mae"]
            row = summary.loc[summary.backbone.eq(model) & summary.arm.eq(arm)].iloc[0]
            mean, sd = row.mae_mean, row.mae_seed_sd
            assert xlim[0] < min(values.min(), mean - sd) and max(values.max(), mean + sd) < xlim[1]
            color = BASELINE if arm == "C0" else BLUE
            ax.scatter(values, j + jitter, s=8, color=color, edgecolors="none", alpha=.49, zorder=3)
            ax.errorbar(mean, j, xerr=sd, fmt="D", ms=4.1, color=color, mec="white", mew=.55,
                        elinewidth=1.15, capsize=2.0, capthick=.85, zorder=5)
    return xlim


def draw_heatmaps(fig, estimates):
    heading(fig, 0.07, 1.70, "D", "Single-pH inputs: change in MAE from structure only")
    cmap = LinearSegmentedColormap.from_list("relative_error", DIVERGING)
    norm = TwoSlopeNorm(vmin=-SCALE, vcenter=0, vmax=SCALE)
    order = pd.MultiIndex.from_product([PH_ARMS, ENDPOINTS], names=["arm", "endpoint"])
    left, patch, gap = 0.48, 2.13, 0.177
    label(fig, 0.245, 1.248, "pH", fontsize=7.4, ha="center", fontweight="bold")
    for i, (model, model_name) in enumerate(MODELS.items()):
        ax = axes(fig, left + i * (patch + gap), .49, patch, .75)
        values = estimates.loc[estimates.backbone.eq(model)].set_index(["arm", "endpoint"]).loc[order, "relative_delta_pct"]
        matrix = values.to_numpy().reshape(5, 7)
        mesh = ax.pcolormesh(matrix, cmap=cmap, norm=norm, shading="flat", edgecolors="white", linewidth=.55)
        ax.set(xlim=(0, 7), ylim=(5, 0), yticks=np.arange(5) + .5, xticks=np.arange(7) + .5)
        ax.tick_params(axis="both", length=0, pad=5, labelsize=7)
        ax.set_xticklabels([v[0] for v in ENDPOINTS.values()])
        ax.set_yticklabels(PH_ARMS.values())
        ax.tick_params(axis="y", labelleft=i == 0)
        ax.xaxis.tick_top()
        ax.set_title(model_name, fontsize=8, fontweight="bold", pad=19)
        for idx, endpoint in enumerate(ENDPOINTS):
            ax.plot([idx + .21, idx + .79], [1.025, 1.025], color=ENDPOINT_COLORS[endpoint], linewidth=1.5,
                    solid_capstyle="butt", transform=ax.get_xaxis_transform(), clip_on=False)
        for row, col in np.ndindex(matrix.shape):
            value = matrix[row, col]
            string = f"{value:.1f}".replace("-", "−") if abs(value) >= .05 else "0.0"
            r, g, b, _ = cmap(norm(value))
            # A white halo gives dark numbers enough contrast at the deepest fill.
            halo = [pe.withStroke(linewidth=.65, foreground="white", alpha=.84)] if .2126*r + .7152*g + .0722*b < .45 else []
            ax.text(col + .5, row + .5, string, ha="center", va="center", fontsize=7, path_effects=halo)
        for spine in ax.spines.values():
            spine.set_visible(False)
    bar = fig.colorbar(mesh, cax=axes(fig, 2.36, .236, 2.66, .080), orientation="horizontal", ticks=[-15, 0, 15])
    bar.ax.set_xticklabels(["−15%", "0%", "+15%"])
    bar.ax.tick_params(length=2, width=.5, pad=2, labelsize=7)
    bar.outline.set_visible(False)
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor("face")
    label(fig, 2.27, .276, "Lower error", ha="right", va="center", fontsize=7.3)
    label(fig, 5.11, .276, "Higher error", ha="left", va="center", fontsize=7.3)


def main():
    tables, limits, input_hashes, checks = load_and_verify()
    fig = plt.figure(figsize=(WIDTH, HEIGHT), facecolor="white")
    draw_scatter(fig, tables["scatter_points.csv"], tables["scatter_metrics.csv"], limits)
    encoding_xlim = draw_encodings(fig, tables["encoding_seed_values.csv"], tables["encoding_summary.csv"])
    draw_heatmaps(fig, tables["single_ph_estimates.csv"])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = fig.get_tightbbox(renderer)
    assert bbox.x0 > 0 and bbox.y0 > 0 and bbox.x1 < WIDTH and bbox.y1 < HEIGHT, bbox.bounds
    outputs = {}
    for ext in ("pdf", "png", "svg"):
        name = f"{STEM}.{ext}"
        metadata = {"CreationDate": None, "ModDate": None} if ext == "pdf" else {"Date": None} if ext == "svg" else None
        fig.savefig(HERE / name, dpi=300, metadata=metadata)
        record_hash(HERE / name, outputs)
    plt.close(fig)
    (HERE / "caption.txt").write_text(CAPTION, encoding="utf-8")
    record_hash(HERE / "caption.txt", outputs)
    for name, frame in tables.items():
        dst = HERE / name
        frame.to_csv(dst, index=False)
        pd.testing.assert_frame_equal(frame.reset_index(drop=True), pd.read_csv(dst, float_precision="round_trip",
                                      dtype={"structure_id": str}), check_exact=True)
        record_hash(dst, outputs)
    metadata = {"inputs_sha256": input_hashes, "outputs_sha256": outputs, "panels": {
        "A": "Vd observed versus predicted: REF0/REF74", "B": "CL observed versus predicted: REF0/REF74",
        "C": "Audited class cohort Vd MAE: C0–C3", "D": "Numerical-pKa cohort MAE change: five single-pH arms versus REF0"},
        "seeds": list(SEEDS), "outer_folds": [1, 2, 3, 4, 5], "figure_size_inches": [WIDTH, HEIGHT],
        "artists_bbox_inches": list(bbox.bounds), "scatter_limits_log10": limits, "encoding_xlim_log10_mae": list(encoding_xlim),
        "heatmap_limits_percent": [-SCALE, SCALE], "matplotlib_version": matplotlib.__version__,
        "palette": {"text": "#000000", "baseline": BASELINE, "ionisation": BLUE, "heatmap": list(DIVERGING),
                    "endpoints": ENDPOINT_COLORS, "category_circles": {v[0]: v[1] for v in CATEGORIES.values()}}, "checks": checks}
    (HERE / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Verified and rendered {len(checks['scatter_identity_and_metrics'])} scatterplots, "
          f"{checks['class_encoding']['mean_sd_rows_verified']} encodings and {checks['single_pH']['estimates_verified']} heatmap cells.")


if __name__ == "__main__":
    main()
