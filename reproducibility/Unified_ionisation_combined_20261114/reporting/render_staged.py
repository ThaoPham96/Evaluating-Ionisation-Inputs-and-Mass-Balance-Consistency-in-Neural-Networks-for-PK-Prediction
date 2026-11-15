"""Data adapters for the approved Figure 2, Figure 3 and Figure S2 drawing functions."""
from hashlib import file_digest
import importlib.util
from pathlib import Path, PureWindowsPath
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ION_MAIN_HASH = "3bf3d35420c36536ab687a011c87011cdbe461b0a49b325b20dc2c11c270130b"
ION_SUPP_HASH = "8f1f5546300ea5ea8508df4d267091cc713398e8191a6213abd60d60dd6b6131"
COMBINED_HASH = "5da5ff628ecb0adcb8195f5eee68d45ec19700321f7559e55b14fd849a1b5505"


def digest(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def checked(path, expected, source_hashes):
    actual = digest(path)
    if expected != actual:
        raise ValueError(f"Approved source checksum mismatch: {path}")
    source_hashes[path.as_posix()] = actual
    return path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def write_csv(directory, name, table):
    table.to_csv(directory / name, index=False)


def save(fig, path, stem):
    for ext, metadata in (("pdf", {"CreationDate": None}), ("png", None), ("svg", {"Date": None})):
        fig.savefig(path / f"{stem}.{ext}", dpi=300, metadata=metadata, facecolor="white")
    plt.close(fig)


def ion_tables(ph, ph_seeds, averages, average_seeds):
    baseline = ph.loc[ph.comparator_arm.eq("REF0")]
    baseline_seeds = ph_seeds.loc[ph_seeds.comparator_arm.eq("REF0")]
    b = baseline.loc[baseline.arm.eq("REF74")].assign(arm="P1", comparator_arm="P0").drop(columns="purpose")
    b_seeds = baseline_seeds.loc[baseline_seeds.arm.eq("REF74")].assign(arm="P1", comparator_arm="P0")
    b_average = averages.loc[averages.contrast.eq("10_minus_00")].rename(columns={
        "mean_pct": "relative_delta_pct", "low95_pct": "crossed95_low_pct", "high95_pct": "crossed95_high_pct",
    }).drop(columns=["metric", "contrast"]).assign(endpoint="Average", valid_bootstrap_draws=10_000,
                                                   requested_bootstrap_draws=10_000)
    b_average_seeds = average_seeds.loc[average_seeds.contrast.eq("10_minus_00")].rename(
        columns={"value_pct": "relative_delta_pct"},
    )[["backbone", "seed", "relative_delta_pct"]].assign(endpoint="Average")
    b_average["seeds_below_zero"] = b_average.backbone.map(
        b_average_seeds.groupby("backbone").relative_delta_pct.apply(lambda s: s.lt(0).sum()),
    )
    return baseline, baseline_seeds, b, b_seeds, b_average, b_average_seeds


def render_ion(output, src, ph, ph_seeds, averages, average_seeds, points, metrics, limits, hashes):
    directory = src / "figures/ionisation_main_v3"
    accepted = read_json(checked(directory / "manifest.json", ION_MAIN_HASH, hashes))
    for rel, value in accepted["sources"].items():
        if rel.replace("\\", "/").startswith(("runtime/ph_category_features/", "runtime/ph8_features/")) or rel.replace("\\", "/") in (
            "figures/ionisation_main_v3/schematic.py", "figures/ionisation_style.py",
            "data/release/numerical/model_dataset.csv", "data/release/numerical/manifest.json",
        ):
            checked(src.joinpath(*PureWindowsPath(rel).parts), value, hashes)
    mod = load_module("unified_render_ion", checked(directory / "render.py", accepted["render_script_sha256"], hashes))
    count_table = pd.DataFrame([dict(pH=k, changed_category=v, n=327) for k, v in mod.category_switch_counts().items()])
    output.mkdir()
    mod.HERE = output  # Accepted drawing functions write their plotted sidecars via HERE.
    mod.SCATTERS = output / "scatter_inputs"
    mod.SCATTERS.mkdir()
    mod.EXP = output
    write_csv(mod.SCATTERS, "plotted_points.csv", points)
    write_csv(mod.SCATTERS, "metrics.csv", metrics)
    scatter_manifest = dict(axis_limits_log10=limits, outputs={p.name: digest(p) for p in mod.SCATTERS.glob("*.csv")})
    (mod.SCATTERS / "manifest.json").write_text(json.dumps(scatter_manifest, indent=2) + "\n", encoding="utf-8")
    baseline, baseline_seeds, b, b_seeds, b_average, b_average_seeds = ion_tables(ph, ph_seeds, averages, average_seeds)
    d = baseline.loc[baseline.endpoint.isin(["Vd", "CL"])]
    d_seeds = baseline_seeds.loc[baseline_seeds.endpoint.isin(["Vd", "CL"])]
    fig = plt.figure(figsize=(mod.FIGURE_WIDTH, mod.FIGURE_HEIGHT), facecolor="white")
    mod.schematic(fig)
    mod.ionisation_performance(fig, b, b_seeds, b_average, b_average_seeds, baseline)
    # Reuse the exact panel D percent bounds and plotted seed values for panel B.
    # The accepted B function converts percent→absolute→percent (and uses
    # factor*delta), which can round by one ULP. Replace only the x coordinates
    # using the common table after checking artist↔row correspondence.
    for axis, model in zip(fig.axes[1:], mod.MODELS, strict=True):
        model_b = baseline.loc[baseline.backbone.eq(model) & baseline.arm.eq("REF74")].set_index("endpoint")
        model_s = baseline_seeds.loc[baseline_seeds.backbone.eq(model) & baseline_seeds.arm.eq("REF74")]
        for y, ep in enumerate(mod.ENDPOINTS):
            shared = model_b.loc[ep]
            interval = axis.lines[2*y + 2]  # A horizontal separator and zero reference precede the estimates.
            exact_bounds = [shared.crossed95_low_pct, shared.crossed95_high_pct]
            np.testing.assert_allclose(interval.get_xdata(), exact_bounds, rtol=0, atol=1e-12)
            interval.set_xdata(exact_bounds)
            dots = axis.collections[y]
            exact_seed = model_s.loc[model_s.endpoint.eq(ep)].sort_values("seed").relative_delta_pct.to_numpy()
            pos = dots.get_offsets().copy()
            np.testing.assert_allclose(pos[:, 0], exact_seed, rtol=0, atol=1e-12)
            pos[:, 0] = exact_seed
            dots.set_offsets(pos)
    # The displayed sidecars use the same source data as the updated artists.
    write_csv(output, "panel_b_displayed_estimates.csv", b)
    write_csv(output, "panel_b_seeds.csv", b_seeds[["backbone", "endpoint", "seed", "relative_delta_pct"]])
    write_csv(output, "panel_b_average_estimates.csv", b_average)
    write_csv(output, "panel_b_average_seeds.csv", b_average_seeds)
    write_csv(output, "panel_a_counts.csv", count_table)
    mod.vd_scatter_row(fig)
    d_limits = mod.single_ph_panel(fig, d, d_seeds)
    schematic_ax, *plots = fig.axes
    endpoint_colours = {mod.to_rgb(colour) for colour in mod.ENDPOINT_COLORS.values()}
    output_symbols = {label for label in schematic_ax.texts if mod.to_rgb(label.get_color()) in endpoint_colours}
    for label in fig.findobj(mod.Text):
        if label not in output_symbols:
            label.set_color("black")
    for ax in plots:
        ax.tick_params(colors="black")
        for spine in ax.spines.values():
            spine.set_color("black")
    save(fig, output, "ionisation_main")
    formatted = [f"{b_average.set_index('backbone').loc[model, 'relative_delta_pct']:.2f}%" for model in mod.MODELS]
    caption = mod.CAPTION.replace("-3.79%, -5.47% and -1.49%", f"{formatted[0]}, {formatted[1]} and {formatted[2]}")
    caption = caption.replace("The pH 7.4 and structure-only conditions reuse the original fitted predictions.",
        "The pH 7.4 and structure-only conditions use the same fitted predictions as the two zero-loss conditions in Figure 3.")
    (output / "caption.txt").write_text(caption, encoding="utf-8")
    return (b, b_seeds, b_average, b_average_seeds), d_limits


def render_s2(output, src, baseline, baseline_seeds, points, metrics, limits, hashes):
    directory = src / "figures/supplementary_selected_v1"
    accepted = read_json(checked(directory / "manifest.json", ION_SUPP_HASH, hashes))
    source = directory / "render.py"
    mod = load_module("unified_render_s2", checked(source, accepted["inputs_sha256"][source.relative_to(src).as_posix()], hashes))
    output.mkdir()
    tables = {"scatter_points.csv": points, "scatter_metrics.csv": metrics, "single_ph_estimates.csv": baseline,
              "single_ph_seed_values.csv": baseline_seeds}
    for name in ("encoding_seed_values.csv", "encoding_summary.csv"):
        path = directory / name
        data = checked(path, accepted["outputs_sha256"][path.relative_to(src).as_posix()], hashes).read_bytes()
        (output / name).write_bytes(data)
        tables[name] = pd.read_csv(output / name, float_precision="round_trip")
    fig = plt.figure(figsize=(mod.WIDTH, mod.HEIGHT), facecolor="white")
    mod.draw_scatter(fig, points, metrics, limits)
    mod.draw_encodings(fig, tables["encoding_seed_values.csv"], tables["encoding_summary.csv"])
    mod.draw_heatmaps(fig, baseline)
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    if not (bbox.x0 > 0 and bbox.y0 > 0 and bbox.x1 < mod.WIDTH and bbox.y1 < mod.HEIGHT):
        raise ValueError(f"Supplementary figure extends beyond its bounds: {bbox.bounds}")
    save(fig, output, mod.STEM)
    (output / "caption.txt").write_text(mod.CAPTION, encoding="utf-8")
    for name, table in tables.items():
        if not name.startswith("encoding_"):
            write_csv(output, name, table)
    # Keep the approved ±15% colour scale; cells outside it retain their exact
    # numeric labels but saturate. Expose each such cell in the staging receipt.
    cls = baseline.loc[baseline.relative_delta_pct.abs().gt(mod.SCALE), ["backbone", "arm", "endpoint", "relative_delta_pct"]]
    return dict(heatmap_outside_colour_scale=cls.to_dict("records"), artists_bbox_inches=list(bbox.bounds),
                class_encoding_checksums={n: digest(output / n) for n in tables if n.startswith("encoding_")})


def render_combined(output, src, factorial, averages, average_seeds, hashes):
    directory = src / "figures/combined_main_v2"
    accepted = read_json(checked(directory / "source_manifest.json", COMBINED_HASH, hashes))
    paths = ("combined_main_v2/render.py", "combined_main_v2/schematic.py", "combined_main_v2/endpoint_panel.py",
             "combined_main_v1/render.py", "combined_results_v1/render.py", "combined_results_v1/combined_style.py")
    for rel in paths:
        path = src / "figures" / rel
        checked(path, accepted["source_files"][str(PureWindowsPath(path.relative_to(src)))], hashes)
    mod = load_module("unified_render_combined", directory / "render.py")
    endpoint = mod.endpoint
    cells = pd.MultiIndex.from_product([mod.MODELS, mod.CONDITIONS, endpoint.SEEDS], names=["backbone", "cell", "seed"])
    residual_seeds = mod.legacy.joint_coordinates(factorial["seed_metrics"], value="value",
                                                 keys=list(cells.names), expected=cells)
    all_means, all_seeds = mod.legacy.all_endpoint_coordinates(factorial["seed_metrics"], residual_seeds)
    # The accepted reader verifies CSV/manifest hashes on disk. The staged input
    # tables are paired from the validated predictions in memory; retain its
    # arithmetic and key checks here while avoiding reads of the older run.
    def average_data(means, seeds):
        a = averages.set_index(endpoint.KEYS, verify_integrity=True)
        s = average_seeds.set_index([*endpoint.KEYS, "seed"], verify_integrity=True)
        for source, target, column, levels in (
            (means, a, "mean_pct", ["backbone", "contrast"]),
            (seeds, s, "value_pct", ["backbone", "contrast", "seed"]),
        ):
            np.testing.assert_allclose(source[column].groupby(level=levels).mean().reindex(target.droplevel("metric").index),
                                       target[column], rtol=0, atol=1e-12)
        return a, s

    endpoint.average_data = average_data
    mod.configure()
    heights = tuple(accepted["main_layout"]["panel_heights_inches"])
    fig = plt.figure(figsize=(mod.WIDTH, sum(heights)), facecolor="white")
    top, middle, bottom = fig.subfigures(3, 1, height_ratios=heights, hspace=0)
    mod.panel_heading(top, "A", "", .967)
    mod.schematic.draw_schematic(top.add_axes([.035, .0, .943, .86]))
    b = endpoint.draw_endpoint_panel(middle, factorial, left=mod.LEFT, right=mod.RIGHT, descriptive_text=False)
    for label in middle.axes[0].get_yticklabels()[:-1]:
        label.set_color(mod.MAIN_LABEL_COLOUR)
    x_limits = mod.joint_panel(bottom, all_means, all_seeds, descriptive_text=False)
    min_y, max_y = all_seeds.mean_abs_residual_log10.agg(["min", "max"])
    y_limits = [.045, .173]
    if min_y <= y_limits[0] or max_y >= y_limits[1]:
        # Preserve the approved scale whenever it includes all new points.
        y_limits = [min(y_limits[0], float(min_y-.005)), max(y_limits[1], float(max_y+.005))]
        for ax in bottom.axes:
            ax.set_ylim(y_limits)
    mean_check = averages.set_index(["backbone", "contrast"])["mean_pct"]
    for (model, cell), row in all_means.iterrows():
        expected = 0 if cell == "00" else mean_check[model, f"{cell}_minus_00"]
        np.testing.assert_allclose(row.all_endpoint_error_change_pct, expected, atol=1e-12, rtol=0)
    output.mkdir()
    save(fig, output, "combined_main")
    plotted = pd.concat([b, all_means.reset_index().assign(panel="c", plot_kind="summary"),
                         all_seeds.reset_index().assign(panel="c", plot_kind="seed")], ignore_index=True)
    assert len(plotted) == 840
    write_csv(output, "plot_data.csv", plotted)
    write_csv(output, "average_summary.csv", averages)
    write_csv(output, "average_seeds.csv", average_seeds)
    (output / "caption.txt").write_text(mod.CAPTION, encoding="utf-8")
    return b, dict(main_plot_rows=len(plotted), joint_x_limits=x_limits, joint_y_limits=y_limits)


def identity_table(ion, b3):
    b2, s2, a2, sa2 = ion
    b3 = b3.loc[b3.contrast.eq("10_minus_00")].copy()
    b3["endpoint"] = b3.metric.str.removesuffix("_mae").replace("average_7_endpoint_mae_pct", "Average")
    keys = ["backbone", "endpoint"]
    means = b3.loc[b3.plot_kind.eq("summary") & b3.endpoint.ne("Average")].set_index(keys).sort_index()
    all_b2 = b2.set_index(keys).sort_index()
    for ion_col, combined_col in (("relative_delta_pct", "mean_pct"), ("crossed95_low_pct", "low95_pct"),
                                  ("crossed95_high_pct", "high95_pct"), ("comparator_mae", "denominator00")):
        np.testing.assert_array_equal(all_b2[ion_col], means.loc[all_b2.index, combined_col])
    ends = b3.loc[b3.plot_kind.eq("seed") & b3.endpoint.ne("Average")].set_index([*keys, "seed"]).sort_index()
    np.testing.assert_array_equal(s2.set_index([*keys, "seed"]).sort_index().relative_delta_pct, ends.value_pct)
    for ion_values, kind, left, right, cols in (
        (a2, "summary", ["relative_delta_pct", "crossed95_low_pct", "crossed95_high_pct"],
         ["mean_pct", "low95_pct", "high95_pct"], keys),
        (sa2, "seed", ["relative_delta_pct"], ["value_pct"], [*keys, "seed"]),
    ):
        reference = b3.loc[b3.plot_kind.eq(kind) & b3.endpoint.eq("Average")].set_index(cols).sort_index()
        np.testing.assert_array_equal(ion_values.set_index(cols).sort_index()[left], reference[right])
    return b3.drop(columns=["panel", "metric", "contrast"])


def render_figures(output, ion_src, combined_src, campaign_id, factorial, ph, ph_seeds,
                   averages, average_seeds, points, metrics, limits):
    hashes = {}
    with plt.rc_context():
        ion, d_limits = render_ion(output / "fig2", ion_src, ph, ph_seeds, averages, average_seeds, points, metrics, limits, hashes)
    with plt.rc_context():
        supp = render_s2(output / "figS2", ion_src, ph.loc[ph.comparator_arm.eq("REF0")],
                         ph_seeds.loc[ph_seeds.comparator_arm.eq("REF0")], points, metrics, limits, hashes)
    with plt.rc_context():
        b3, combined = render_combined(output / "fig3", combined_src, factorial, averages, average_seeds, hashes)
    identical = identity_table(ion, b3)
    write_csv(output, "figure2_figure3_shared_ph74.csv", identical)
    hashes[Path(__file__).as_posix()] = digest(Path(__file__))
    return dict(campaign_id=campaign_id, source_sha256={Path(p).relative_to(ion_src.parent).as_posix(): h for p, h in hashes.items()},
                ph74_identity={"summary_rows": int(identical.plot_kind.eq("summary").sum()),
                               "seed_rows": int(identical.plot_kind.eq("seed").sum()),
                               "checks": ["same prediction values and fit IDs", "same fixed denominators", "same endpoint and average point estimates and CI bounds", "same seed differences"]},
                fig2=dict(panel_d_x_limits=list(d_limits), scatter_limits=limits), figS2=supp, fig3=combined,
                matplotlib_version=matplotlib.__version__)
