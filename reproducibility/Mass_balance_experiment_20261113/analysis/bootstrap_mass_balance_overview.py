"""Pair the primary v6 policies in a connected-group bootstrap of outer predictions."""

import argparse
import platform
from pathlib import Path

import numpy as np

from mass_balance_campaign import ENDPOINTS, PRIMARY, SEEDS, read_csv, sha256, write_csv, write_json
from mass_balance_controls import residual_log10
from summarize_mass_balance_campaign import Cohort, PERCENT_COLUMNS, Predictions, scoring_units


BOOTSTRAP_SEED = 20261112
N_RESAMPLES = 10_000
N_GROUPS = 552
COHERENCE_LIMIT = np.log10(1.25)
POLICIES = ("selected_positive", "selected_alpha_orthogonal", "selected_alpha_half_life")


def group_membership(cohort):
    """Return sorted group IDs and group indices in frozen structure order."""
    group_by_structure = {}
    for row in read_csv(cohort.path / "outer.csv"):
        group_by_structure[row["structure_id"]] = row["group_id"]
    groups = np.array([group_by_structure[sid] for sid in cohort.index])
    group_ids, membership = np.unique(groups, return_inverse=True)
    if len(group_ids) != N_GROUPS or "" in group_ids:
        raise ValueError(f"Expected {N_GROUPS} nonempty connected groups")
    for group in range(len(group_ids)):
        if np.unique(cohort.folds[membership == group]).size != 1:
            raise ValueError(f"Connected group crosses outer folds: {group_ids[group]}")
    return group_ids, membership


def resample_group_counts(n_groups, n_resamples, seed):
    """Draw n_groups cluster IDs per resample and count each ID's multiplicity."""
    rng = np.random.Generator(np.random.PCG64(seed))
    group_draws = rng.integers(0, n_groups, size=(n_resamples, n_groups), dtype=np.int32)
    group_draws += (np.arange(n_resamples, dtype=np.int32) * n_groups)[:, None]
    return np.bincount(group_draws.ravel(), minlength=n_resamples * n_groups).reshape(n_resamples, n_groups)


def aggregate_groups(values, membership, n_groups):
    """Aggregate (seed, structure) values into a (group, seed) matrix."""
    sums = np.zeros((n_groups, len(values)), dtype=np.float64)
    np.add.at(sums, membership, values.T)
    return sums


def paired_bootstrap(base, policy, common, membership, draws):
    """Return point contrasts and seed-averaged draw contrasts with paired denominators."""
    n_groups = draws.shape[1]
    base_by_group = aggregate_groups(np.where(common, base, 0), membership, n_groups)
    policy_by_group = aggregate_groups(np.where(common, policy, 0), membership, n_groups)
    n_by_group = aggregate_groups(common, membership, n_groups)
    delta_by_group = policy_by_group - base_by_group

    n_point = n_by_group.sum(axis=0)
    n_draw = draws @ n_by_group
    draw_delta = draws @ delta_by_group
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "baseline_by_seed": base_by_group.sum(axis=0) / n_point,
            "policy_by_seed": policy_by_group.sum(axis=0) / n_point,
            "delta_by_seed": delta_by_group.sum(axis=0) / n_point,
            "n_scored_by_seed": n_point.astype(int),
            "delta_by_draw": (draw_delta / n_draw).mean(axis=1),
        }


def paired_errors(base, policy, truth):
    """Per-structure absolute errors and the same-row paired accuracy masks."""
    eligible = np.isfinite(truth)
    base_valid = np.isfinite(base)
    policy_valid = np.isfinite(policy)
    with np.errstate(invalid="ignore", over="ignore"):
        base_error = np.abs(base - truth)
        policy_error = np.abs(policy - truth)
    if truth.ndim == 2:
        eligible = eligible.all(axis=1)
        base_valid = base_valid.all(axis=2)
        policy_valid = policy_valid.all(axis=2)
        base_error = base_error.mean(axis=2)
        policy_error = policy_error.mean(axis=2)
    common = eligible & base_valid & policy_valid
    return base_error, policy_error, common, eligible, base_valid, policy_valid


def paired_coherence(base, policy):
    """Percentage-point outcomes with all structures in the threshold denominator."""
    def outcome(prediction):
        with np.errstate(invalid="ignore", over="ignore"):
            residual = np.abs(residual_log10(prediction))
        valid = np.isfinite(prediction).all(axis=-1) & np.isfinite(residual)
        return 100.0 * (valid & (residual <= COHERENCE_LIMIT)), valid

    base_hits, base_valid = outcome(base)
    policy_hits, policy_valid = outcome(policy)
    common = np.ones(base_hits.shape, dtype=bool)
    return base_hits, policy_hits, common, common[0], base_valid, policy_valid


def metric_rows(backbone, policy_name, metric, inputs, membership, draws):
    base, policy, common, eligible, base_valid, policy_valid = inputs
    score = paired_bootstrap(base, policy, common, membership, draws)
    seed_rows = []
    for i, seed in enumerate(SEEDS):
        if not np.isfinite(score["delta_by_seed"][i]):
            raise ValueError(f"Undefined paired point estimate for {backbone}/{policy_name}/{metric}/{seed}")
        seed_rows.append({
            "backbone": backbone, "policy": policy_name, "metric": metric, "seed": seed,
            "baseline_mean": float(score["baseline_by_seed"][i]),
            "policy_mean": float(score["policy_by_seed"][i]),
            "delta": float(score["delta_by_seed"][i]),
            "n_eligible": int(eligible.sum()), "n_scored": int(score["n_scored_by_seed"][i]),
            "n_baseline_invalid": int((eligible & ~base_valid[i]).sum()),
            "n_policy_invalid": int((eligible & ~policy_valid[i]).sum()),
        })
    finite_draws = score["delta_by_draw"][np.isfinite(score["delta_by_draw"])]
    if not finite_draws.size:
        raise ValueError(f"No defined bootstrap draws for {backbone}/{policy_name}/{metric}")
    lower, upper = np.quantile(finite_draws, [0.025, 0.975], method="linear")
    summary = {
        "backbone": backbone, "policy": policy_name, "metric": metric,
        "baseline_mean": float(score["baseline_by_seed"].mean()),
        "policy_mean": float(score["policy_by_seed"].mean()),
        "delta": float(score["delta_by_seed"].mean()), "ci95_lower": float(lower), "ci95_upper": float(upper),
        "seed_delta_sd": float(score["delta_by_seed"].std(ddof=1)),
        "seed_delta_min": float(score["delta_by_seed"].min()),
        "seed_delta_max": float(score["delta_by_seed"].max()),
        "n_seeds": len(SEEDS), "n_eligible": int(eligible.sum()),
        "min_n_scored": int(score["n_scored_by_seed"].min()),
        "max_n_scored": int(score["n_scored_by_seed"].max()),
        "n_baseline_invalid_across_seeds": sum(row["n_baseline_invalid"] for row in seed_rows),
        "n_policy_invalid_across_seeds": sum(row["n_policy_invalid"] for row in seed_rows),
        "n_bootstrap_draws": len(draws), "n_defined_draws": len(finite_draws),
    }
    return summary, seed_rows


def score_policies(source, cohort, membership, draws):
    def pooled(backbone, policy):
        arrays, missing = source.pooled(backbone, policy, SEEDS)
        if missing:
            raise ValueError(f"Incomplete {backbone}/{policy} predictions: {missing}")
        return np.stack([scoring_units(arrays[seed]) for seed in SEEDS])

    summary, per_seed = [], []
    for backbone in PRIMARY:
        base = pooled(backbone, "baseline")
        for policy_name in POLICIES:
            policy = pooled(backbone, policy_name)
            metrics = {
                "same_row_triad_macro_log10_mae": paired_errors(base[:, :, :3], policy[:, :, :3], cohort.triad_truth),
                "within_1_25_fold_percentage_points": paired_coherence(base[:, :, :3], policy[:, :, :3]),
            }
            for column, endpoint in enumerate(ENDPOINTS.values()):
                units = "percentage_point" if column in PERCENT_COLUMNS else "log10"
                metrics[f"{endpoint}_{units}_mae"] = paired_errors(
                    base[:, :, column], policy[:, :, column], cohort.endpoint_truth[:, column]
                )
            for metric, inputs in metrics.items():
                row, seeds = metric_rows(backbone, policy_name, metric, inputs, membership, draws)
                summary.append(row)
                per_seed.extend(seeds)
    return summary, per_seed


def write_report(output, source, cohort, counts, group_ids, membership, summary, seed_rows):
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("summary.csv", summary), ("seed_deltas.csv", seed_rows)):
        write_csv(output / filename, tuple(rows[0]), (tuple(row.values()) for row in rows))
    np.savez_compressed(output / "group_draw_counts.npz", group_ids=group_ids, counts=counts.astype(np.uint16))
    write_csv(output / "group_membership.csv", ("structure_id", "group_id"),
              zip(cohort.index, group_ids[membership], strict=True))
    write_json(output / "manifest.json", {
        "campaign_id": source.campaign_id, "campaign_dir": str(source.path), "release_dir": str(cohort.path),
        "campaign_artifacts_read": source.source_hashes, "release_artifacts_read": cohort.hashes,
        "code": {name: sha256(Path(__file__).with_name(name)) for name in
                 (Path(__file__).name, "summarize_mass_balance_campaign.py", "mass_balance_campaign.py",
                  "mass_balance_controls.py")},
        "outputs": {name: sha256(output / name) for name in
                    ("summary.csv", "seed_deltas.csv", "group_draw_counts.npz", "group_membership.csv")},
        "bootstrap": {
            "n_groups": len(group_ids), "n_structures": len(membership), "n_resamples": len(counts),
            "rng": "numpy.random.Generator(PCG64)", "rng_seed": BOOTSTRAP_SEED,
            "groups_per_resample": len(group_ids), "group_order": "ascending frozen group ID",
            "draws": "sample connected groups uniformly with replacement; retain every sampled group's structure multiplicity",
            "paired": "one shared draw matrix for every model, policy, seed and metric",
            "aggregation": "pool outer-test folds within each seed; recompute paired contrast within each draw and seed; average the nine seed contrasts",
            "ci": "linear-interpolated 2.5th and 97.5th percentiles of defined draws; conditional on fitted models",
            "seed_spread": "sample SD and range of the nine paired seed deltas on the original cohort",
            "accuracy_mask": "common valid predictions on the same-row complete triads or frozen endpoint labels",
            "coherence_mask": "all structures, invalid predictions counted as outside threshold",
            "contrast": "policy minus baseline",
            "fit_seeds": list(SEEDS),
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if any(output.is_relative_to(path.resolve()) for path in (args.campaign_dir, args.release_dir)):
        parser.error("Use an output directory outside the campaign and frozen release")
    cohort = Cohort(args.release_dir)
    source = Predictions(args.campaign_dir, cohort)
    group_ids, membership = group_membership(cohort)
    counts = resample_group_counts(len(group_ids), N_RESAMPLES, BOOTSTRAP_SEED)
    summary, per_seed = score_policies(source, cohort, membership, counts.astype(np.float64))
    write_report(output, source, cohort, counts, group_ids, membership, summary, per_seed)
    for row in summary:
        if row["metric"] in ("same_row_triad_macro_log10_mae", "within_1_25_fold_percentage_points"):
            print(f"{row['backbone']:15} {row['policy']:28} {row['metric']:34} "
                  f"{row['delta']:+.6f} [{row['ci95_lower']:+.6f}, {row['ci95_upper']:+.6f}]", flush=True)


if __name__ == "__main__":
    main()
