"""Show paired MAE changes from structure-only models across all seven endpoints."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ionisation_style import CONTROL, DIVERGING, FONT, INK, MUTED, ENDPOINT_COLORS
MAIN = HERE.parents[1]
MODELS = {"GraphConv": "GraphConv", "Chemprop": "Chemprop", "ChemBERTa_head": "Frozen ChemBERTa + MLP"}
ARMS = {
    "REF74": "pH 7.4 category",
    "REP2": "pH 7.4 category ×2",
    "ADD2": "pH 7.4 + 2.0 categories",
    "ADD5": "pH 7.4 + 5.0 categories",
    "ADD65": "pH 7.4 + 6.5 categories",
    "REP4": "pH 7.4 category ×4",
    "ALL4": "Four-pH categories",
}
REPEATS = {"REP2", "REP4"}
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
KEYS = ["arm", "backbone", "endpoint"]
LIMIT = 15


CAPTION = (
    "Relative changes in held-out MAE for seven categorical input arms versus the same backbone's "
    "structure-only reference (REF0), on the 327-structure numerical-pKa subset with 238 parent/scaffold "
    "groups. Columns show the seven endpoints and their labelled structure counts. Each cell is the mean "
    "of nine paired MAE differences divided by the fixed nine-seed mean REF0 MAE for the same backbone "
    "and endpoint, multiplied by 100. Each seed pools its five held-out outer folds. Negative values indicate "
    "lower error. MAE uses log10 units for half-life, Vd, CL, Cmax and Tmax, and percentage points for F and "
    "PPB. The pH 6.5 comparison is descriptive. The plotted-data CSV includes exploratory 95% crossed "
    "seed/group bootstrap intervals from 10,000 draws. Grey row labels mark repeats of the pH 7.4 category, "
    "which change input parameterisation while preserving category information. Comparisons against REF0 "
    "describe the complete input representation. Figure S6 compares the four-pH input with a width-matched "
    "control. All models use zero mass-balance penalty and equal endpoint weights.\n"
)


def digest(path):
    with path.open("rb") as handle:
        return sha256(handle.read()).hexdigest()


def load_data():
    source_manifest = MAIN / "panel_ph_vs_baseline_manifest.json"
    metadata = json.loads(source_manifest.read_text(encoding="utf-8"))
    if (
        metadata["status"] != "report_ready" or metadata["cohort"] != "numerical"
        or metadata["procedure"] != "primary" or metadata["comparator"] != "REF0"
        or metadata["n_structures"] != 327 or metadata["n_groups"] != 238
        or metadata["bootstrap_draws"] != 10000 or metadata["interval_level"] != 0.95
        or tuple(metadata["seeds"]) != SEEDS
    ):
        raise ValueError("Expected completed paired REF0 contrasts for the numerical cohort")
    tables = []
    sources = {source_manifest.name: digest(source_manifest)}
    for name in ("estimates", "seeds"):
        path = MAIN / f"panel_ph_vs_baseline_{name}.csv"
        sources[path.name] = digest(path)
        if sources[path.name] != metadata["files"][path.name]:
            raise ValueError(f"Data checksum mismatch: {path.name}")
        tables.append(pd.read_csv(path, float_precision="round_trip"))
    estimates, seeds = tables
    order = pd.MultiIndex.from_product((ARMS, MODELS, ENDPOINTS), names=KEYS)
    expected = set(order.tolist())
    if len(estimates) != len(expected) or set(estimates[KEYS].itertuples(index=False, name=None)) != expected:
        raise ValueError("Incomplete arm, backbone or endpoint coverage")
    actual_seeds = set(seeds[[*KEYS, "seed"]].itertuples(index=False, name=None))
    if actual_seeds != {(*row, seed) for row in expected for seed in SEEDS} or len(seeds) != len(expected) * len(SEEDS):
        raise ValueError("Incomplete paired seed coverage")
    estimates = estimates.set_index(KEYS).loc[order].reset_index()
    for field, index in (("n_structures", 2), ("n_groups", 3)):
        np.testing.assert_array_equal(estimates[field], estimates.endpoint.map({key: value[index] for key, value in ENDPOINTS.items()}))
    np.testing.assert_allclose(estimates.set_index(KEYS).relative_delta_pct, seeds.groupby(KEYS).relative_delta_pct.mean().loc[order], atol=1e-12)
    low, high, means = (estimates[key] for key in ("crossed95_low_pct", "crossed95_high_pct", "relative_delta_pct"))
    if (
        not np.isfinite([low, high, means]).all() or not means.between(low, high).all()
        or not estimates.n_seeds.eq(9).all() or not estimates.valid_bootstrap_draws.eq(10000).all()
        or not means.abs().lt(LIMIT).all()
    ):
        raise ValueError("Invalid or out-of-scale overview values")
    return estimates.assign(comparator="REF0"), metadata, sources


def signed_pct(value, places):
    if round(value, places) == 0:
        return f"{0:.{places}f}%"
    return f"{value:+.{places}f}%".replace("-", "−")


def render(estimates):
    plt.rcParams.update({
        "font.family": FONT, "font.size": 11, "text.color": INK, "xtick.color": INK,
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(15.4, 6.7), facecolor="white")
    fig.text(0.035, 0.938, "MAE changes across all seven PK endpoints", fontsize=21, fontweight="bold")
    fig.text(0.035, 0.883, "Relative change in MAE (%) versus the same model’s structure-only reference (REF0).", fontsize=12, color=MUTED)
    cmap = LinearSegmentedColormap.from_list("error_change", DIVERGING)
    norm = TwoSlopeNorm(vmin=-LIMIT, vcenter=0, vmax=LIMIT)
    n_rows, n_cols = len(ARMS), len(ENDPOINTS)
    grid = fig.add_gridspec(1, 3, left=0.191, right=0.972, bottom=0.225, top=0.765, wspace=0.08)
    for panel, (model, title) in enumerate(MODELS.items()):
        ax = fig.add_subplot(grid[0, panel])
        model_values = estimates.loc[estimates.backbone.eq(model)].set_index(["arm", "endpoint"])
        values = model_values.loc[[(arm, ep) for arm in ARMS for ep in ENDPOINTS], "relative_delta_pct"].to_numpy().reshape(n_rows, n_cols)
        mesh = ax.pcolormesh(np.arange(n_cols+1), np.arange(n_rows+1), values, cmap=cmap, norm=norm,
                             edgecolors="white", linewidths=0.7, shading="flat")
        for row, col in np.ndindex(values.shape):
            value = values[row, col]
            rgb = cmap(norm(value))[:3]
            luminance = 0.2126*rgb[0] + 0.7152*rgb[1] + 0.0722*rgb[2]
            ax.text(col+0.5, row+0.5, signed_pct(value, 1).removesuffix("%"), fontsize=14.4, ha="center", va="center",
                    color="white" if luminance < 0.53 else INK)
        ax.set(xlim=(0, n_cols), ylim=(n_rows, 0), xticks=np.arange(n_cols)+0.5, yticks=[])
        ax.set_xticklabels([f"{label}\n{n}" for label, _, n, _ in ENDPOINTS.values()], fontsize=13.5, linespacing=1.0)
        ax.xaxis.tick_top()
        for tick, color in zip(ax.get_xticklabels(), ENDPOINT_COLORS.values()):
            tick.set_color(color)
        ax.tick_params(length=0, pad=5)
        ax.set_title(title, fontsize=15, fontweight="bold", pad=35)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if panel == 0:
            for row, (arm, label) in enumerate(ARMS.items()):
                box = {"facecolor": "#E9EEF0", "edgecolor": "none", "pad": 3} if arm in REPEATS else None
                ax.text(-0.16, row+0.5, f"{label}  ({arm})", ha="right", va="center", fontsize=12,
                        color=CONTROL if arm in REPEATS else INK, bbox=box, clip_on=False)
    bar = fig.add_axes([0.36, 0.151, 0.39, 0.023])
    legend = fig.colorbar(mesh, cax=bar, orientation="horizontal", ticks=np.arange(-LIMIT, LIMIT+1, 5),
                          format=FuncFormatter(lambda value, _: signed_pct(value, 0)))
    legend.solids.set_rasterized(False)
    legend.solids.set_edgecolor("face")
    legend.outline.set_visible(False)
    legend.ax.tick_params(length=3, pad=3, labelsize=10)
    fig.text(0.348, 0.162, "Lower error", ha="right", va="center", fontsize=12, color=MUTED)
    fig.text(0.762, 0.162, "Higher error", ha="left", va="center", fontsize=12, color=MUTED)
    fig.text(0.035, 0.075, "Grey labels mark repeated-category controls. Four-pH categories use pH 2.0, 5.0, 6.5 and 7.4.", fontsize=12, color=MUTED)
    fig.text(0.035, 0.043, "Numbers below endpoint names give labelled structures. Five grouped outer folds, nine paired seeds. pH 6.5 comparison is descriptive.", fontsize=12, color=MUTED)
    fig.savefig(HERE / "ph_vs_baseline_overview.pdf", metadata={"CreationDate": None})
    fig.savefig(HERE / "ph_vs_baseline_overview.png", dpi=210)
    plt.close(fig)


def main():
    values, metadata, sources = load_data()
    render(values)
    values.to_csv(HERE / "plotted_data.csv", index=False)
    (HERE / "caption.txt").write_text(CAPTION, encoding="utf-8")
    outputs = ("ph_vs_baseline_overview.pdf", "ph_vs_baseline_overview.png", "plotted_data.csv", "caption.txt")
    manifest = {
        "source_files_relative_to": "figures/ionisation_main_v2",
        "source_sha256": sources, "render_script_sha256": digest(Path(__file__)),
        "outputs": {name: digest(HERE / name) for name in outputs},
        "campaign_id": metadata["export_campaign_id"], "release_fingerprints": metadata["release_fingerprints"],
        "comparator": "REF0", "rows": len(values), "seed_ids": SEEDS, "colour_limit_pct": [-LIMIT, LIMIT],
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(f"Rendered {len(values)} paired mean MAE changes.")


if __name__ == "__main__":
    main()
