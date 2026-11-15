"""Verify and score a completed numerical campaign, then stage Figures 2, 3 and S2."""
from __future__ import annotations

import argparse
from hashlib import file_digest, sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKSPACE = ROOT.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "analysis"))
import combined_score as score

from render_staged import render_figures

ALT_PH = ("PH2", "PH5", "PH65", "PH8")
REF = {"REF0": "00", "REF74": "10"}
ALIASES = {"P0": "REF0", "P1": "REF74"}
KEYS = ["backbone", "arm", "seed", "structure_id"]
PRED = [f"{'pred' if j < 5 else 'physical'}_{e}" for j, e in enumerate(score.ENDPOINTS)]
METHOD = "Fixed unresampled mean comparator MAE; jointly resample full structure groups and full seed trajectories."
OLD_ION = WORKSPACE / "Ionisation_experiment_20261113"
OLD_COMBINED = WORKSPACE / "Combined_ionisation_mass_balance_20261113"
CAMPAIGN_ID = "84858a14dd64206333d9bc3851252293e659881a4643d4b26ee7d7d39af545a1"
SOURCE_TRIADS_HASH = "db51103863ca3abe0ff966a838a0715dc06e1657a599e9ce1d43af95e5d642bd"
FIT_COUNTS = {"units": 15, "inner_fits": 1350, "final_fits": 1080, "total_fits": 2430}


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, values: pd.DataFrame) -> None:
    values.to_csv(path, index=False)


def check_hash(path: Path, expected: str) -> str:
    if digest(path) != expected:
        raise ValueError(f"Checksum mismatch: {path}")
    return expected


def check_ion_table(path: Path, arms, cohort: score.Cohort, campaign_id: str) -> pd.DataFrame:
    table = score.read_csv(path)
    columns = [*KEYS, "outer_fold", "cohort", "procedure", "source_campaign_id", "source_fit_id",
               *[f"{prefix}_{e}" for prefix in ("pred", "physical") for e in score.ENDPOINTS]]
    score.prior.require_columns(table, columns, path.name)
    index = pd.MultiIndex.from_product([score.BACKBONES, arms, score.SEEDS, cohort.rows.index], names=KEYS)
    table = table.set_index(KEYS, verify_integrity=True)
    if len(table) != len(index) or not index.isin(table.index).all():
        raise ValueError(f"Incomplete numerical prediction coverage in {path}")
    table = table.reindex(index)[columns[len(KEYS):]]
    if (not table["cohort"].eq("numerical").all() or not table["procedure"].eq("primary").all()
            or not table["source_campaign_id"].eq(campaign_id).all() or table["source_fit_id"].isna().any()
            or not np.array_equal(table["outer_fold"], index.get_level_values("structure_id").map(cohort.rows.outer_fold))
            or not np.isfinite(table.filter(regex=r"^(pred_|physical_)").to_numpy(dtype=float)).all()
            or not table[["physical_F", "physical_PPB"]].apply(lambda c: c.between(0, 100)).all().all()):
        raise ValueError(f"Invalid metadata or predictions in {path}")
    fits = table.reset_index()[["backbone", "arm", "seed", "outer_fold", "source_fit_id"]].drop_duplicates()
    if (len(fits) != len(score.BACKBONES) * len(arms) * len(score.SEEDS) * len(score.FOLDS)
            or fits["source_fit_id"].duplicated().any() or fits.duplicated(fits.columns[:-1]).any()):
        raise ValueError(f"Unpaired or duplicate fit IDs in {path}")
    return table


def verify_exports(export: Path, cohort: score.Cohort):
    combined, ion = export / "combined", export / "ionisation"
    receipt = read_json(export / "manifest.json")
    for name in ("combined", "ionisation"):
        check_hash(export / name / "manifest.json", receipt["exports"][name])
    plan_path = export / "plan.json"
    check_hash(plan_path, receipt["plan_sha256"])
    plan = read_json(plan_path)
    identity = plan["identity"]
    identity_hash = sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if (not plan["campaign_id"] == identity_hash == receipt["campaign_id"] == CAMPAIGN_ID
            or receipt["release_fingerprint"] != "3e1de9c4f7641238fdb6603676c17d059c2ffce262496437c752340ee5208d2b"
            or receipt["unified_fit_counts"] != FIT_COUNTS or identity["counts"] != FIT_COUNTS
            or len(receipt["unit_markers"]) != FIT_COUNTS["units"]):
        raise ValueError("Expected the completed, approved unified campaign plan and collection receipt")
    manifest, final, zero, grid, records = score.read_export(combined)
    campaign_id = manifest["campaign_id"]
    ion_manifest = read_json(ion / "manifest.json")
    for component in (manifest, ion_manifest):
        for field in ("campaign_id", "release_fingerprint", "plan_sha256", "environment_hash", "unified_fit_counts"):
            if component[field] != receipt[field]:
                raise ValueError(f"Unified export receipt disagrees on {field}")
    if set(final.fit_id) & set(grid.fit_id):
        raise ValueError("Inner and final fits must have distinct fit IDs")
    inner = score.validate_inner(grid, zero, cohort)
    selections, lambda_audit = score.audit_selections(inner, records, campaign_id)
    predictions = score.validate_final(final, cohort, selections)
    for filename in ("outer_predictions.csv", "reference_and_ph_predictions.csv"):
        check_hash(ion / filename, ion_manifest["files"][filename])
    outer = check_ion_table(ion / "outer_predictions.csv", tuple(ALIASES), cohort, campaign_id)
    ph = check_ion_table(ion / "reference_and_ph_predictions.csv", (*REF, *ALT_PH), cohort, campaign_id)
    if (ion_manifest["rows"], ion_manifest["reference_rows"], ion_manifest["fits"], ion_manifest["reference_fits"]) != (len(ph), len(outer), 810, 270):
        raise ValueError("Ionisation export receipt disagrees with rows or fit counts")
    same = outer.reset_index().assign(arm=lambda rows: rows.arm.map(ALIASES)).set_index(KEYS).sort_index()
    ref_rows = ph.loc[ph.index.get_level_values("arm").isin(REF)].sort_index()
    pd.testing.assert_frame_equal(same, ref_rows, check_exact=True)
    combined_refs = final.loc[final.cell.isin(REF.values())].assign(
        arm=lambda rows: rows.cell.map({v: k for k, v in REF.items()}),
    ).set_index(KEYS).reindex(ref_rows.index)
    np.testing.assert_array_equal(ref_rows["source_fit_id"], combined_refs["fit_id"])
    np.testing.assert_array_equal(ref_rows[PRED].to_numpy(), combined_refs[score.PRED_COLUMNS].to_numpy())
    alternatives = ph.loc[ph.index.get_level_values("arm").isin(ALT_PH)]
    if set(alternatives["source_fit_id"]) & (set(final.fit_id) | set(grid.fit_id)):
        raise ValueError("Alternative pH fits must be distinct from the factorial fits")
    alt = {b: alternatives.loc[b, PRED].to_numpy().reshape(len(ALT_PH), len(score.SEEDS), len(cohort.rows), -1)
           for b in score.BACKBONES}
    return manifest, predictions, alt, inner, selections, pd.DataFrame(lambda_audit)


def score_ph(predictions, alt, cohort, draws, factorial):
    """Calculate pH contrasts and seven-endpoint averages using the common draws."""
    crossed = factorial["contrasts"].set_index(["backbone", "metric", "contrast"], verify_integrity=True).sort_index()
    crossed_seeds = factorial["seed_contrasts"].set_index(["backbone", "metric", "contrast", "seed"], verify_integrity=True).sort_index()
    names = (*score.CELLS, *ALT_PH)
    pairs = [(arm, "REF0") for arm in ("REF74", *ALT_PH)] + [(arm, "REF74") for arm in ALT_PH]
    summary, seeds, average_rows, average_seeds, average_contrasts = [], [], [], [], []
    for backbone, factorial_q in predictions.items():
        q = np.concatenate([factorial_q, alt[backbone]])
        outcomes, _, _ = score.metrics(q, cohort)
        normalised_deltas, normalised_draws = [], []
        for ep in score.ENDPOINTS:
            metric = f"{ep}_mae"
            errors, mask, unit = outcomes[metric]
            errors, groups = errors[..., mask], cohort.group[mask]
            by_seed = errors.mean(axis=-1)
            by_draw = score.crossed_means(errors, groups, draws)
            denominator00 = by_seed[0].mean()
            if not np.isfinite(denominator00) or denominator00 <= 0:
                raise ValueError(f"Nonpositive normalizer for {backbone}/{ep}")
            normalised_deltas.append((100 / denominator00) * (by_seed - by_seed[0]))
            normalised_draws.append((100 / denominator00) * (by_draw - by_draw[0]))
            for arm, reference in pairs:
                a, b = (names.index(REF.get(label, label)) for label in (arm, reference))
                meta = dict(cohort="numerical", procedure="primary", backbone=backbone, arm=arm,
                            comparator_procedure="primary", comparator_backbone=backbone, comparator_arm=reference,
                            scope="all", stratum="all", purpose="exploratory", primary_test=False,
                            endpoint=ep, unit=unit)
                stats, _ = score.prior.contrast_statistics(errors[a], errors[b], groups, draws, primary=False)
                delta = by_seed[a] - by_seed[b]
                low, high = score.prior.interval(by_draw[a] - by_draw[b])
                denominator = by_seed[b].mean()
                if not np.isfinite(denominator) or denominator <= 0:
                    raise ValueError(f"Nonpositive {reference} normalizer for {backbone}/{ep}")
                arm_mae = by_seed[a].mean()
                if (arm, reference) == ("REF74", "REF0"):
                    key = (backbone, metric, "10_minus_00")
                    row = crossed.loc[key]
                    authoritative_seeds = crossed_seeds.loc[key, "value"].reindex(score.SEEDS).to_numpy()
                    np.testing.assert_allclose(delta, authoritative_seeds, atol=1e-12, rtol=0)
                    np.testing.assert_allclose([denominator, low, high],
                        row[["baseline_00_mean", "crossed95_low", "crossed95_high"]].to_numpy(dtype=float),
                        atol=1e-12, rtol=0)
                    delta = authoritative_seeds
                    denominator, low, high = row[["baseline_00_mean", "crossed95_low", "crossed95_high"]]
                    arm_mae = row["mean_10"]
                factor = 100 / denominator
                stats.update(score.prior.seed_summary(delta))
                if (arm, reference) == ("REF74", "REF0"):
                    stats["delta_mean"] = row["mean"]
                stats.update(arm_mae=arm_mae, comparator_mae=denominator, crossed95_low=low, crossed95_high=high,
                             crossed95_direction=score.prior.interval_direction((low, high)),
                             relative_delta_pct=factor * stats["delta_mean"],
                             crossed95_low_pct=factor * low, crossed95_high_pct=factor * high)
                summary.append(dict(meta, **stats))
                # Match the accepted combined seed normalisation order exactly.
                seeds.extend(dict(meta, seed=s, delta=d, relative_delta_pct=100*d/denominator, arm_mae=av,
                                  comparator_mae=bv, mean_comparator_mae=denominator)
                             for s, d, av, bv in zip(score.SEEDS, delta, by_seed[a], by_seed[b], strict=True))
        joint_seed, joint_draw = np.mean(normalised_deltas, axis=0), np.mean(normalised_draws, axis=0)
        for i, cell in enumerate(score.CELLS[1:], start=1):
            meta = dict(backbone=backbone, metric="average_7_endpoint_mae_pct", contrast=f"{cell}_minus_00")
            low, high = score.prior.interval(joint_draw[i])
            average_rows.append(dict(meta, mean_pct=joint_seed[i].mean(), low95_pct=low, high95_pct=high,
                                     n_seeds=len(score.SEEDS), seed_sd_pct=joint_seed[i].std(ddof=1)))
            average_seeds.extend(dict(meta, seed=s, value_pct=v) for s, v in zip(score.SEEDS, joint_seed[i], strict=True))
        for a, b in (("11", "10"), ("11", "01")):
            i, j = names.index(a), names.index(b)
            values = joint_seed[i] - joint_seed[j]
            low, high = score.prior.interval(joint_draw[i] - joint_draw[j])
            average_contrasts.append(dict(backbone=backbone, contrast=f"{a}_minus_{b}", mean_pp=values.mean(),
                                          low95_pp=low, high95_pp=high, n_seeds=len(score.SEEDS)))
    return tuple(pd.DataFrame(value) for value in (summary, seeds, average_rows, average_seeds, average_contrasts))


def scatter_tables(predictions, cohort):
    points, metrics, limits = [], [], {}
    for ep, unit in (("Vd", "L"), ("CL", "L/h")):
        j = score.ENDPOINTS.index(ep)
        mask = np.isfinite(cohort.truth[:, j])
        y = cohort.truth[mask, j]
        selected = cohort.rows.loc[mask]
        base = selected.reset_index()[["structure_id", "outer_fold", "ionisation_class"]]
        extent = [y.min(), y.max()]
        for arm, cell in REF.items():
            for model, q in predictions.items():
                seed_pred = q[score.CELLS.index(cell)][:, mask, j]
                x = seed_pred.mean(axis=0)
                extent.extend([x.min(), x.max()])
                df = base.assign(arm=arm, backbone=model, endpoint=ep, physical_unit=unit,
                                 observed_log10=y, predicted_log10=x, predicted_geometric_mean=10**x,
                                 observed_physical=selected[score.RAW_ENDPOINTS[j]].to_numpy())
                points.append(df[["arm", "backbone", *base.columns, "endpoint", "physical_unit", "observed_log10",
                                  "predicted_log10", "predicted_geometric_mean", "observed_physical"]])
                desc = score.endpoint_descriptors(seed_pred, y, percentage=False)
                mean = score.endpoint_descriptors(x[None, :], y, percentage=False)
                metrics.append(dict(arm=arm, backbone=model, endpoint=ep, n=len(y),
                                    r2_displayed_seed_mean=mean["r2"][0], spearman_displayed_seed_mean=mean["spearman"][0],
                                    mae_log10_displayed_seed_mean=np.abs(y-x).mean(), r2_individual_seed_mean=desc["r2"].mean(),
                                    r2_individual_seed_sd=desc["r2"].std(ddof=1), spearman_individual_seed_mean=desc["spearman"].mean()))
        limits[ep] = [float(np.floor(min(extent)*2)/2-.15), float(np.ceil(max(extent)*2)/2+.15)]
    return pd.concat(points, ignore_index=True), pd.DataFrame(metrics), limits


def manuscript_claims(factorial, ph, average, average_contrasts, scatters, selections):
    contrasts = factorial["contrasts"]
    endpoint = contrasts.loc[contrasts.metric.isin([f"{e}_mae" for e in score.ENDPOINTS]) &
                             contrasts.contrast.isin(["10_minus_00", "01_minus_00", "11_minus_00", "11_minus_10", "11_minus_01"])]
    cols = ["backbone", "metric", "contrast", "relative_change_pct", "relative_crossed95_low_pct", "relative_crossed95_high_pct"]
    effects = endpoint[cols].rename(columns={"relative_change_pct": "change_pct_of_structure_only",
                                             "relative_crossed95_low_pct": "low95_pct_of_structure_only",
                                             "relative_crossed95_high_pct": "high95_pct_of_structure_only"})
    residual = factorial["metrics"].loc[lambda f: f.metric.eq(score.RESIDUAL) & f.cell.isin(score.CELLS)]
    residual = residual[["backbone", "cell", "mean", "crossed95_low", "crossed95_high"]].rename(columns={"mean": "mean_abs_residual_log10"})
    residual["geometric_mean_fold_disagreement"] = 10**residual.mean_abs_residual_log10
    residual_contrasts = contrasts.loc[contrasts.metric.eq(score.RESIDUAL) & ~contrasts.contrast.str.startswith("zero_eligible"),
                                      ["backbone", "contrast", "mean", "crossed95_low", "crossed95_high", *[f"mean_{cell}" for cell in score.CELLS]]].copy()
    residual_contrasts["geometric_mean_fold_ratio"] = 10**residual_contrasts["mean"]
    for idx, row in residual_contrasts.iterrows():
        if "_minus_" in row.contrast:
            _, reference = row.contrast.split("_minus_")
            residual_contrasts.loc[idx, "change_pct_of_comparator_mean"] = 100 * row["mean"] / row[f"mean_{reference}"]
    ph_subset = ph.loc[ph.arm.isin(["PH2", "PH8"]) & ph.endpoint.isin(["Vd", "CL"]),
                       ["backbone", "arm", "comparator_arm", "endpoint", "relative_delta_pct", "crossed95_low_pct", "crossed95_high_pct"]]
    nontriad = ph.loc[ph.arm.eq("REF74") & ph.comparator_arm.eq("REF0") & ~ph.endpoint.isin(["t_half", "Vd", "CL"])]
    non_vd_cl = ph.loc[ph.arm.eq("REF74") & ph.comparator_arm.eq("REF0") & ~ph.endpoint.isin(["Vd", "CL"])]
    def count_zero(rows):
        return dict(n_intervals=len(rows), n_including_zero=int((rows.crossed95_low.le(0) & rows.crossed95_high.ge(0)).sum()))
    return dict(
        endpoint_percent_effects=effects.to_dict("records"),
        endpoint_average_percent_effects=average.to_dict("records"),
        endpoint_average_percentage_point_comparisons=average_contrasts.to_dict("records"),
        residual_means=residual.to_dict("records"), residual_contrasts=residual_contrasts.replace({np.nan: None}).to_dict("records"),
        scatter_r2=scatters[["backbone", "arm", "endpoint", "n", "r2_displayed_seed_mean"]].to_dict("records"),
        ph2_ph8_volume_clearance=ph_subset.to_dict("records"),
        ref74_intervals_excluding_half_life_volume_and_clearance=count_zero(nontriad),
        ref74_intervals_excluding_volume_and_clearance=count_zero(non_vd_cl),
        ref74_seeds_below_zero=ph.loc[ph.arm.eq("REF74"), ["backbone", "endpoint", "seeds_below_zero"]].to_dict("records"),
        selected_lambda_by_outer_fold={b: [selections[b, f][0] for f in score.FOLDS] for b in score.BACKBONES},
    )


def build(export: Path, output: Path) -> None:
    release = ROOT / "data/release/numerical"
    triads = ROOT / "runtime/same_row_complete_triads.csv"
    check_hash(triads, SOURCE_TRIADS_HASH)
    cohort = score.load_cohort(release, triads)
    manifest, predictions, alt, inner, selections, lambda_audit = verify_exports(export, cohort)
    draws = score.prior.group_draws(len(np.unique(cohort.group)), len(score.SEEDS), score.N_BOOTSTRAP,
                                   np.random.default_rng(score.BOOTSTRAP_SEED))
    with threadpool_limits(limits=2, user_api="blas"):
        factorial = score.score_predictions(predictions, inner, cohort, selections, posthoc=False, draws=draws)
        ph, ph_seeds, average, average_seeds, average_contrasts = score_ph(predictions, alt, cohort, draws, factorial)
        points, metrics, limits = scatter_tables(predictions, cohort)
    factorial["lambda_selection"] = lambda_audit
    for name, table in factorial.items():
        write_csv(output / "scores" / f"{name}.csv", table)
    write_json(output / "scores" / "manifest.json", dict(
        status="complete", campaign_id=manifest["campaign_id"], release_fingerprint=score.FINGERPRINT,
        final_rows=manifest["final_rows"], fit_counts={"inner": manifest["inner_fits"], "final": manifest["final_fits"]},
        bootstrap_draws=score.N_BOOTSTRAP, bootstrap_seed=score.BOOTSTRAP_SEED,
        plan_sha256=manifest["plan_sha256"], export_manifest_sha256=digest(export / "combined" / "manifest.json"),
        source_triads_sha256=SOURCE_TRIADS_HASH, scorer_sha256=digest(Path(score.__file__)),
        files={f"{name}.csv": digest(output / "scores" / f"{name}.csv") for name in factorial},
    ))
    write_csv(output / "scores" / "single_ph_contrasts.csv", ph)
    write_csv(output / "scores" / "single_ph_seed_contrasts.csv", ph_seeds)
    write_csv(output / "scores" / "combined_average_loss_contrasts.csv", average_contrasts)
    write_json(output / "manuscript_claims.json", manuscript_claims(factorial, ph, average, average_contrasts, metrics, selections))
    render_receipt = render_figures(output, OLD_ION, OLD_COMBINED, manifest["campaign_id"], factorial,
                                    ph, ph_seeds, average, average_seeds, points, metrics, limits)
    pinned_runtime = (release / "manifest.json", release / "model_dataset.csv", release / "outer.csv",
                      release / "inner.csv", triads, Path(score.__file__), Path(score.prior.__file__), Path(__file__))
    input_records = {str(p.relative_to(WORKSPACE)): digest(p) for p in pinned_runtime}
    input_records.update({f"export/{name}": digest(export / name) for name in ("manifest.json", "plan.json")})
    for source in (export / "combined", export / "ionisation"):
        for filename in ("manifest.json", *read_json(source / "manifest.json")["files"]):
            path = source / filename
            input_records[f"export/{source.name}/{filename}"] = digest(path)
    write_json(output / "manifest.json", dict(
        campaign_id=manifest["campaign_id"], release_fingerprint=score.FINGERPRINT,
        plan_sha256=manifest["plan_sha256"], bootstrap_draws=score.N_BOOTSTRAP, bootstrap_seed=score.BOOTSTRAP_SEED,
        bootstrap_method=METHOD, seeds=list(score.SEEDS), numerically_scored_structures=len(cohort.rows),
        numerically_scored_groups=len(np.unique(cohort.group)), input_sha256=input_records,
        rendering=render_receipt, output_sha256={p.relative_to(output).as_posix(): digest(p)
                                                 for p in sorted(output.rglob("*")) if p.is_file()},
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True, help="Parent of the combined and ionisation export directories")
    parser.add_argument("--output-dir", type=Path, default=HERE / "output", help="New directory inside this reporting folder")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    export = args.export_dir.resolve()
    if not output.is_relative_to(HERE) or output == HERE or output.is_relative_to(export) or output.exists():
        parser.error("Choose an unused output directory inside reporting, separate from the prediction export")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".stage-", dir=output.parent, ignore_cleanup_errors=True) as scratch:
        staged = Path(scratch) / "reports"
        (staged / "scores").mkdir(parents=True)
        build(export, staged)
        staged.rename(output)
    print(f"Verified staged figures and score tables: {output}")


if __name__ == "__main__":
    main()
