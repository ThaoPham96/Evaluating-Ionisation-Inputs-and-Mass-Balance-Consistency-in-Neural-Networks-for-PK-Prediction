"""Score the single-pH categorical inputs on the frozen numerical cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import ionisation_score as common
import ph_ionisation_score as categorical


COHORT = categorical.COHORT
PROCEDURE = "primary"
NEW_ARMS = ("PH2", "PH5", "PH65")
ARMS = ("REF0", "REF74", *NEW_ARMS)
N_STRUCTURES, N_GROUPS = 327, 238
N_PREDICTIONS = len(ARMS) * len(common.BACKBONES) * len(common.SEEDS) * N_STRUCTURES
N_FITS = len(ARMS) * len(common.BACKBONES) * len(common.SEEDS) * len(common.FOLDS)


def expected_runs() -> set[tuple[str, str, str, str]]:
    return {(COHORT, PROCEDURE, backbone, arm) for backbone in common.BACKBONES for arm in ARMS}


def validate_predictions(
    frame: pd.DataFrame, data: pd.DataFrame, campaign_id: str,
) -> tuple[dict[tuple[str, str, str, str], common.Predictions], pd.DataFrame]:
    """Align one complete held-out prediction per structure, run and seed."""
    columns = [*categorical.FIT_KEYS, "structure_id", *common.PRED_COLUMNS,
               *common.PHYSICAL_COLUMNS, *categorical.SOURCE_KEYS]
    common.require_columns(frame, columns, "outer predictions")
    rows = frame[columns].copy()
    if rows.empty or rows.isna().any().any():
        raise ValueError("Outer predictions must contain complete, nonempty rows")
    for key in ("seed", "outer_fold"):
        rows[key] = common.integer_column(rows[key])
    index_keys = ["seed", "structure_id"]
    if rows.duplicated([*common.RUN_COLUMNS, *index_keys]).any():
        raise ValueError("Duplicate held-out prediction for a seed and structure")
    observed = set(rows[list(common.RUN_COLUMNS)].itertuples(index=False, name=None))
    wanted = expected_runs()
    if observed != wanted:
        raise ValueError(f"Run mismatch. Missing {sorted(wanted - observed)}. Unexpected {sorted(observed - wanted)}")

    expected_index = pd.MultiIndex.from_product([common.SEEDS, data.index], names=index_keys)
    expected_folds = np.tile(data["outer_fold"].to_numpy(), len(common.SEEDS))
    shape = (len(common.SEEDS), len(data), len(common.ENDPOINTS))
    aligned = {}
    for run, block in rows.groupby(list(common.RUN_COLUMNS), sort=True):
        block = block.set_index(index_keys).sort_index()
        if not block.index.equals(expected_index):
            raise ValueError(f"{run} must predict every structure exactly once for each of the nine fixed seeds")
        if not np.array_equal(block["outer_fold"].to_numpy(), expected_folds):
            raise ValueError(f"{run} differs from the frozen held-out fold assignment")
        transformed = block[list(common.PRED_COLUMNS)].to_numpy(dtype=float).reshape(shape)
        physical = block[list(common.PHYSICAL_COLUMNS)].to_numpy(dtype=float).reshape(shape)
        if not (np.isfinite(transformed).all() and np.isfinite(physical).all()):
            raise ValueError(f"{run} contains nonfinite predictions")
        percentages = physical[..., common.PERCENT]
        if np.any(physical[..., common.POSITIVE] <= 0) or np.any((percentages < 0) | (percentages > 100)):
            raise ValueError(f"{run} predictions fall outside the endpoint domains")
        if not np.allclose(physical, common.inverse_physical(transformed), rtol=1e-6, atol=0):
            raise ValueError(f"{run} physical and transformed predictions disagree")
        aligned[run] = common.Predictions(common.SEEDS, transformed, physical)

    sources = categorical.prediction_sources(rows, campaign_id)
    reused = sources["arm"].isin(categorical.REFERENCES)
    if not sources.loc[reused, "source_campaign_id"].eq(categorical.PRIOR_CAMPAIGN).all():
        raise ValueError("REF0 and REF74 must come from the original frozen numerical campaign")
    return aligned, sources


def run_contrasts():
    """Yield the seven planned pairs for each model."""
    pairs = [(arm, "REF74") for arm in NEW_ARMS]
    pairs.extend((arm, "REF0") for arm in ("REF74", *NEW_ARMS))
    for backbone in common.BACKBONES:
        for arm, comparator in pairs:
            yield (COHORT, PROCEDURE, backbone, arm), (COHORT, PROCEDURE, backbone, comparator)


def score(
    data: pd.DataFrame, runs: dict, *, n_draws: int = common.N_BOOTSTRAP,
    bootstrap_seed: int = common.BOOTSTRAP_SEED,
) -> dict[str, pd.DataFrame]:
    """Calculate seven endpoint metrics and crossed intervals on common masks."""
    if set(runs) != expected_runs() or n_draws < 1:
        raise ValueError("Scoring requires all validated runs and at least one bootstrap draw")
    raw = data[list(common.RAW_ENDPOINTS)].to_numpy(dtype=float)
    transformed = common.transform_targets(raw)
    observed = np.isfinite(raw)
    if not observed.any(axis=0).all():
        raise ValueError("Scoring requires an observed label for each endpoint")
    errors = {run: common.errors(pred, raw, transformed) for run, pred in runs.items()}
    memberships, groups = pd.factorize(data["group_id"], sort=True)
    draws = common.group_draws(len(groups), len(common.SEEDS), n_draws, np.random.default_rng(bootstrap_seed))
    metrics, summary = categorical.endpoint_tables(runs, raw, transformed, errors, memberships)
    coverage, group_sizes = common.label_coverage({COHORT: data})
    folds = data["outer_fold"].to_numpy()
    contrasts, seed_rows, fold_rows = [], [], []
    for arm, comparator in run_contrasts():
        info = dict(zip(common.RUN_COLUMNS, arm), comparator_procedure=comparator[1],
                    comparator_backbone=comparator[2], comparator_arm=comparator[3],
                    scope="all", stratum="all", purpose="exploratory", primary_test=False)
        for j, endpoint in enumerate(common.ENDPOINTS):
            mask = observed[:, j]
            a, b = errors[arm][:, mask, j], errors[comparator][:, mask, j]
            stats, by_seed = common.contrast_statistics(a, b, memberships[mask], draws, primary=False)
            record = dict(**info, endpoint=endpoint, unit=categorical.unit_for(j))
            mean_comparator = stats["comparator_mae"]
            scale = 100 / mean_comparator if mean_comparator > 0 else np.nan
            contrasts.append(dict(
                **record, **stats, crossed95_low_pct=scale*stats["crossed95_low"],
                crossed95_high_pct=scale*stats["crossed95_high"],
            ))
            for i, seed in enumerate(common.SEEDS):
                seed_rows.append(dict(
                    **record, seed=seed, delta=by_seed[i], relative_delta_pct=scale*by_seed[i],
                    arm_mae=a[i].mean(), comparator_mae=b[i].mean(), mean_comparator_mae=mean_comparator,
                ))
            paired = a - b
            for fold in common.FOLDS:
                subset = folds[mask] == fold
                fold_rows.append(dict(
                    **record, outer_fold=fold, n_structures=int(subset.sum()),
                    n_groups=len(np.unique(memberships[mask][subset])),
                    delta_mean=paired[:, subset].mean() if subset.any() else np.nan,
                ))
    return {
        "endpoint_metrics": metrics, "endpoint_summary": summary, "contrasts": pd.DataFrame(contrasts),
        "contrast_seeds": pd.DataFrame(seed_rows), "contrast_outer_folds": pd.DataFrame(fold_rows),
        "coverage": coverage, "group_sizes": group_sizes,
    }


def score_export(release: Path, predictions: Path, out: Path) -> dict:
    """Validate a completed export and publish its scoring tables and checksums."""
    fingerprints = common.frozen_release_fingerprints(release, (COHORT,), diagnostic=False)
    if fingerprints[COHORT] != categorical.RELEASE_FINGERPRINT:
        raise ValueError("Unexpected frozen numerical release")
    data = common.load_cohort(release, COHORT)
    if len(data) != N_STRUCTURES or data["group_id"].nunique() != N_GROUPS:
        raise ValueError("Expected the 327-structure, 238-group numerical cohort")

    manifest_path = predictions.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise ValueError("Single-pH scoring requires a collector manifest")
    prediction_hash = categorical.sha256(predictions)
    collector_hash = categorical.sha256(manifest_path)
    collector = categorical.checked_campaign(predictions, prediction_hash, fingerprints[COHORT])
    if collector is None or collector.get("prior_campaign_id") != categorical.PRIOR_CAMPAIGN:
        raise ValueError("Collector manifest has an unexpected prior campaign ID")
    frame = common.read_csv(predictions)
    if len(frame) != N_PREDICTIONS or collector.get("rows") != len(frame):
        raise ValueError(f"Expected {N_PREDICTIONS:,} held-out prediction rows")
    runs, sources = validate_predictions(frame, data, collector["campaign_id"])
    if len(sources) != N_FITS or sources.duplicated(categorical.SOURCE_KEYS).any():
        raise ValueError(f"Expected {N_FITS} unique source fits")
    provenance_hash = categorical.check_provenance(predictions, collector, sources)
    if provenance_hash is None:
        raise ValueError("Single-pH scoring requires a hashed source-fit ledger")
    with threadpool_limits(limits=1, user_api="blas"):
        tables = score(data, runs)
    tables["prediction_sources"] = sources

    provenance_path = predictions.with_suffix(".provenance.json")
    input_hashes = {predictions: prediction_hash, manifest_path: collector_hash, provenance_path: provenance_hash}
    if any(categorical.sha256(path) != expected for path, expected in input_hashes.items()):
        raise ValueError("Collector export or sidecar changed during scoring")
    prior_manifest = out / "scoring_manifest.json"
    if prior_manifest.is_file():
        previous = json.loads(prior_manifest.read_text(encoding="utf-8"))
        if previous.get("campaign_id") != collector["campaign_id"]:
            raise ValueError("Scoring output directory belongs to a different campaign")
    names = [f"{name}.csv" for name in tables]
    if predictions.resolve() in {(out / name).resolve() for name in names}:
        raise ValueError("Scoring output would overwrite the predictions input")

    metadata = dict(
        status="report_ready", cohort=COHORT, campaign_id=collector["campaign_id"],
        prior_campaign_id=categorical.PRIOR_CAMPAIGN, release_fingerprints=fingerprints,
        predictions_sha256=prediction_hash, predictions_rows=len(frame),
        collector_manifest_sha256=collector_hash, source_fit_ledger_sha256=provenance_hash,
        prediction_fit_count=len(sources), n_new_prediction_fits=int(sources.arm.isin(NEW_ARMS).sum()),
        n_reused_prediction_fits=int(sources.arm.isin(categorical.REFERENCES).sum()),
        n_structures=len(data), n_groups=int(data["group_id"].nunique()),
        outer_folds=common.FOLDS, seeds=common.SEEDS,
        bootstrap_seed=common.BOOTSTRAP_SEED, bootstrap_draws=common.N_BOOTSTRAP,
        interval_level=0.95, primary_family_size=0,
        inference="Exploratory paired comparisons. No outer-test selection of pH.",
        delta="Arm minus comparator. Negative MAE deltas indicate lower error.",
        relative_delta="100 × paired MAE delta / fixed nine-seed mean comparator MAE; the same denominator scales both crossed interval bounds.",
        crossed_interval="Approximate equal-tailed percentile interval from resampling whole seed trajectories and frozen groups, conditional on observed training data, folds and selected settings.",
        conditional_group_interval="Equal-tailed percentile interval over frozen groups with all fitted seeds fixed.",
        seed_interval="Paired t interval over whole-seed deltas on the fixed observed labels; seed SD separately describes paired variation.",
        masks="All arms use the same observed endpoint labels. F and PPB MAE uses original percentage points, including zeros. The other five MAEs use log10 units.",
        aafe="10 ** mean absolute log10 error on strictly positive labels; zero predictions on those labels yield infinite AAFE.",
        correlations="Spearman rho and R-squared use the same log10 or original percentage values as MAE and are computed separately within each seed.",
        empty_draws="A draw with no observed labels for an endpoint is undefined. Valid bootstrap draw counts are recorded.",
    )
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".scoring-", dir=out) as directory:
        temporary = Path(directory)
        for name, table in tables.items():
            table.to_csv(temporary / f"{name}.csv", index=False)
        metadata["files"] = {name: categorical.sha256(temporary / name) for name in names}
        (temporary / prior_manifest.name).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        for name in (*names, prior_manifest.name):
            (temporary / name).replace(out / name)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    score_export(args.release, args.predictions, args.out)


if __name__ == "__main__":
    main()
