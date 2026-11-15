"""Validate and score held-out predictions from the categorical pH follow-up."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import ionisation_score as common


COHORT = "numerical"
RELEASE_FINGERPRINT = "99e83f5ad3b3802ab2374777eff38f8d9d66e2178237c0e1ea5f23d143948cb8"
PRIOR_CAMPAIGN = "3ffed2efa5ca4a2bdfa3584cb0d1d3d2869d6edc17e1697e6476fa5a356b5e28"
NEW_ARMS = ("ADD2", "ADD65", "ADD5", "ALL4", "REP2", "REP4")
REFERENCES = ("REF0", "REF74")
TUNED_ARMS = ("ALL4", "REP4")
HISTORICAL_ARMS = ("P2", "P3", "P4", "P5")
PRIMARY_ENDPOINTS = ("Vd", "F")
PH_VALUES = {"ADD2": 2.0, "ADD65": 6.5, "ADD5": 5.0}
FIT_KEYS = [*common.RUN_COLUMNS, "outer_fold", "seed"]
SOURCE_KEYS = ["source_campaign_id", "source_fit_id"]
MIN_LABELS, MIN_GROUPS = 10, 5


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def expected_runs() -> set[tuple[str, str, str, str]]:
    return {
        (COHORT, procedure, backbone, arm)
        for procedure, arms in (
            ("primary", (*REFERENCES, *NEW_ARMS)),
            ("tuned", TUNED_ARMS),
            ("historical", HISTORICAL_ARMS),
        )
        for backbone in common.BACKBONES
        for arm in arms
    }


def checked_campaign(path: Path, prediction_hash: str, release_fingerprint: str) -> dict | None:
    """Check the collector's optional export sidecar against the exact input."""
    sidecar = path.with_suffix(".manifest.json")
    if not sidecar.is_file():
        return None
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    if manifest.get("sha256") != prediction_hash:
        raise ValueError("Predictions differ from the collector export manifest")
    if manifest.get("release_fingerprints", {}).get(COHORT) != release_fingerprint:
        raise ValueError("Prediction manifest numerical release fingerprint differs")
    if not isinstance(manifest.get("campaign_id"), str) or not manifest["campaign_id"].strip():
        raise ValueError("Prediction manifest requires a nonempty campaign_id")
    return manifest


def check_provenance(path: Path, collector: dict | None, sources: pd.DataFrame) -> str | None:
    """Check the hashed source-fit ledger emitted by the campaign exporter."""
    if collector is None or "provenance_sha256" not in collector:
        return None
    sidecar = path.with_suffix(".provenance.json")
    if not sidecar.is_file() or sha256(sidecar) != collector["provenance_sha256"]:
        raise ValueError("Prediction source-fit ledger is missing or differs from the collector manifest")
    ledger = json.loads(sidecar.read_text(encoding="utf-8"))
    if ledger.get("campaign_id") != collector["campaign_id"] or ledger.get("prior_campaign_id") != PRIOR_CAMPAIGN:
        raise ValueError("Prediction source-fit ledger has an unexpected campaign ID")
    logged = pd.DataFrame(ledger.get("fits", [])).assign(cohort=COHORT)
    keys = [*FIT_KEYS, *SOURCE_KEYS]
    common.require_columns(logged, keys, "prediction source-fit ledger")
    if len(logged) != len(sources) or logged.duplicated(FIT_KEYS).any():
        raise ValueError("Prediction source-fit ledger has missing or duplicate fit requests")
    merged = sources.merge(logged[keys], on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(sources):
        raise ValueError("Prediction source-fit ledger differs from prediction rows")
    return collector["provenance_sha256"]


def prediction_sources(frame: pd.DataFrame, campaign_id: str | None = None) -> pd.DataFrame:
    """Verify one source fit per model, arm, procedure, outer fold and seed."""
    common.require_columns(frame, [*FIT_KEYS, *SOURCE_KEYS], "outer predictions")
    if frame[SOURCE_KEYS].isna().any().any():
        raise ValueError("Missing source campaign or fit identifier")
    if not all(frame[key].map(lambda value: isinstance(value, str) and bool(value.strip())).all() for key in SOURCE_KEYS):
        raise ValueError("Source campaign and fit identifiers must be nonempty strings")
    sources = frame[[*FIT_KEYS, *SOURCE_KEYS]].drop_duplicates().sort_values(FIT_KEYS)
    if sources.duplicated(FIT_KEYS).any():
        raise ValueError("A seed and fold has multiple source fits for the same arm and procedure")
    reused = sources["procedure"].eq("historical") | sources["arm"].isin(REFERENCES)
    prior_campaigns = set(sources.loc[reused, "source_campaign_id"])
    new_campaigns = set(sources.loc[~reused, "source_campaign_id"])
    if len(prior_campaigns) != 1 or len(new_campaigns) != 1 or prior_campaigns & new_campaigns:
        raise ValueError("Expected one prior source campaign and one distinct pH follow-up campaign")
    if campaign_id is not None and new_campaigns != {campaign_id}:
        raise ValueError("New prediction source campaign differs from the collector manifest")
    # The same final fit can appear under primary and tuned only for the same
    # backbone, arm, fold and seed. Its predictions must also agree exactly.
    identity = [key for key in FIT_KEYS if key != "procedure"]
    if sources[[*SOURCE_KEYS, *identity]].drop_duplicates().duplicated(SOURCE_KEYS).any():
        raise ValueError("A source fit ID is shared by different fit requests")
    shared = sources.loc[sources.duplicated(SOURCE_KEYS, keep=False), SOURCE_KEYS].drop_duplicates()
    if not shared.empty:
        rows = frame.merge(shared, on=SOURCE_KEYS, how="inner", validate="many_to_one")
        keys = [*SOURCE_KEYS, "structure_id"]
        values = [*common.PRED_COLUMNS, *common.PHYSICAL_COLUMNS]
        if rows[[*keys, *values]].drop_duplicates().duplicated(keys).any():
            raise ValueError("Predictions differ across procedures sharing the same source fit")
    return sources.reset_index(drop=True)


def validate_predictions(
    frame: pd.DataFrame, data: pd.DataFrame, campaign_id: str | None = None,
) -> dict[tuple[str, str, str, str], common.Predictions]:
    """Enforce complete nine-seed trajectories and the frozen held-out mapping."""
    cols = [*FIT_KEYS, "structure_id", *common.PRED_COLUMNS, *common.PHYSICAL_COLUMNS, *SOURCE_KEYS]
    common.require_columns(frame, cols, "outer predictions")
    rows = frame[cols].copy()
    if rows.empty or rows.isna().any().any():
        raise ValueError("Outer predictions must contain complete, nonempty rows")
    for key in ("seed", "outer_fold"):
        rows[key] = common.integer_column(rows[key])
    index_keys = ["seed", "structure_id"]
    if rows.duplicated([*common.RUN_COLUMNS, *index_keys]).any():
        raise ValueError("Duplicate held-out prediction for a seed and structure")
    actual_runs = set(rows[list(common.RUN_COLUMNS)].itertuples(index=False, name=None))
    wanted = expected_runs()
    if actual_runs != wanted:
        raise ValueError(f"Run mismatch. Missing {sorted(wanted - actual_runs)}. Unexpected {sorted(actual_runs - wanted)}")
    expected_index = pd.MultiIndex.from_product([common.SEEDS, data.index], names=index_keys)
    expected_folds = np.tile(data["outer_fold"].to_numpy(), len(common.SEEDS))
    shape = (len(common.SEEDS), len(data), len(common.ENDPOINTS))
    result = {}
    for run, block in rows.groupby(list(common.RUN_COLUMNS), sort=True):
        block = block.set_index(index_keys).sort_index()
        if not block.index.equals(expected_index):
            raise ValueError(f"{run} must predict every structure exactly once for each of the nine fixed seeds")
        if not np.array_equal(block["outer_fold"].to_numpy(), expected_folds):
            raise ValueError(f"{run} differs from the frozen held-out fold assignment")
        pred = block[list(common.PRED_COLUMNS)].to_numpy(dtype=float).reshape(shape)
        physical = block[list(common.PHYSICAL_COLUMNS)].to_numpy(dtype=float).reshape(shape)
        if not (np.isfinite(pred).all() and np.isfinite(physical).all()):
            raise ValueError(f"{run} contains nonfinite predictions")
        if np.any(physical[..., common.POSITIVE] <= 0) or np.any((physical[..., common.PERCENT] < 0) | (physical[..., common.PERCENT] > 100)):
            raise ValueError(f"{run} predictions fall outside the endpoint domains")
        if not np.allclose(physical, common.inverse_physical(pred), rtol=1e-6, atol=0):
            raise ValueError(f"{run} physical and transformed predictions disagree")
        result[run] = common.Predictions(common.SEEDS, pred, physical)
    prediction_sources(rows, campaign_id)
    return result


def run_contrasts():
    """Yield paired runs and their prespecified purpose."""
    within_primary = [
        ("ALL4", "REP4", "headline"),
        ("ADD2", "REP2", "exploratory"),
        ("ADD5", "REP2", "exploratory"),
        ("ADD65", "REP2", "descriptive"),
        *((arm, "REF74", "exploratory") for arm in ("ALL4", *PH_VALUES, "REP2", "REP4")),
        *((arm, "REF0", "context") for arm in ("REF74", *NEW_ARMS)),
    ]
    for backbone in common.BACKBONES:
        def run(procedure, arm):
            return (COHORT, procedure, backbone, arm)

        for arm, base, purpose in within_primary:
            yield run("primary", arm), run("primary", base), purpose
        yield run("tuned", "ALL4"), run("tuned", "REP4"), "tuning_sensitivity"
        yield run("primary", "ALL4"), run("historical", "P4"), "context"
        for arm, base in (("P4", "P3"), ("P4", "P5"), ("P5", "P3"), ("P4", "P2")):
            yield run("historical", arm), run("historical", base), "historical"
        yield run("historical", "P3"), run("primary", "REF74"), "historical"


def category_codes(data: pd.DataFrame, ph: float) -> np.ndarray:
    """Strict descriptor categories. Equality to pH leaves that side inactive."""
    return data["pka_acid"].lt(ph).to_numpy(dtype=int) + 2 * data["pka_base"].gt(ph).to_numpy(dtype=int)


def subgroup_masks(data: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    audited = {label: data["ionisation_class"].eq(label).to_numpy() for label in common.CLASSES}
    base = category_codes(data, 7.4)
    switches = {f"pH{ph:g}": category_codes(data, ph) != base for ph in PH_VALUES.values()}
    switches["any_added_pH"] = np.logical_or.reduce(list(switches.values()))
    return audited, switches


def contrast_scopes(purpose, arm, audited, switches):
    if purpose == "headline":
        yield from (("class", label, mask) for label, mask in audited.items())
        key = "any_added_pH"
    elif arm in PH_VALUES and purpose in ("exploratory", "descriptive"):
        key = f"pH{PH_VALUES[arm]:g}"
    else:
        return
    yield "switch_status", f"{key}/switcher", switches[key]
    yield "switch_status", f"{key}/stable", ~switches[key]


def coverage(data, audited, switches):
    """Count labelled structures and groups under identical frozen masks."""
    rows = []
    for fold in ("all", *common.FOLDS):
        partition = np.ones(len(data), dtype=bool) if fold == "all" else data["outer_fold"].eq(fold).to_numpy()
        scopes = [("all", "all", partition)]
        scopes.extend(("class", label, mask) for label, mask in audited.items())
        for key, mask in switches.items():
            scopes.extend((("switch_status", f"{key}/switcher", mask), ("switch_status", f"{key}/stable", ~mask)))
        for scope, stratum, mask in scopes:
            part = data.loc[partition & mask]
            for endpoint, column in zip(common.ENDPOINTS, common.RAW_ENDPOINTS):
                labels = part[column].notna()
                rows.append(dict(
                    cohort=COHORT, outer_fold=fold, scope=scope, stratum=stratum, endpoint=endpoint,
                    n_structures=len(part), n_groups=part["group_id"].nunique(),
                    n_labels=int(labels.sum()), n_label_groups=part.loc[labels, "group_id"].nunique(),
                    n_label_zeros=int(part[column].eq(0).sum()), n_aafe=int(part[column].gt(0).sum()),
                ))
    return pd.DataFrame(rows)


def endpoint_tables(runs, raw, transformed, errors, memberships):
    rows = []
    observed_by_endpoint = np.isfinite(raw)
    group_counts = [len(np.unique(memberships[mask])) for mask in observed_by_endpoint.T]
    for run, pred in runs.items():
        meta = dict(zip(common.RUN_COLUMNS, run))
        for i, seed in enumerate(common.SEEDS):
            for j, endpoint in enumerate(common.ENDPOINTS):
                stats = common.endpoint_metric(pred, raw, errors[run], i, j)
                observed = observed_by_endpoint[:, j]
                stats["n_groups"] = group_counts[j]
                if j in common.PERCENT:
                    target, estimate = raw[:, j], pred.physical[i, :, j]
                else:
                    target, estimate = transformed[:, j], pred.transformed[i, :, j]
                stats["spearman"], stats["r2"] = common.vd_correlations(
                    target[observed], estimate[observed], stats["n_groups"], subgroup=False, class_median=False,
                )
                rows.append(dict(**meta, seed=seed, endpoint=endpoint, unit=unit_for(j), **stats))
    table = pd.DataFrame(rows)
    summary = common.metric_summary(
        table, [*common.RUN_COLUMNS, "endpoint", "unit"],
        ["n_structures", "n_groups", "n_label_zeros", "n_aafe"], ["mae", "aafe", "spearman", "r2"],
    )
    return table, summary


def unit_for(endpoint_index: int) -> str:
    return "percentage_point" if endpoint_index in common.PERCENT else "log10"


def has_subgroup_precision(observed, memberships) -> bool:
    return int(observed.sum()) >= MIN_LABELS and len(np.unique(memberships[observed])) >= MIN_GROUPS


def interaction_statistics(delta, labelled, switchers, memberships, draws) -> dict:
    """Estimate switcher-minus-stable paired MAE effects with shared resamples."""
    masks = (labelled & switchers, labelled & ~switchers)
    result = {
        "n_switcher_labels": int(masks[0].sum()), "n_stable_labels": int(masks[1].sum()),
        "n_switcher_groups": len(np.unique(memberships[masks[0]])),
        "n_stable_groups": len(np.unique(memberships[masks[1]])),
        "contrast_of_deltas": np.nan, "seed_sd": np.nan,
        "crossed95_low": np.nan, "crossed95_high": np.nan, "valid_bootstrap_draws": 0,
    }
    if all(mask.any() for mask in masks):
        by_seed = delta[:, masks[0]].mean(axis=1) - delta[:, masks[1]].mean(axis=1)
        result.update(contrast_of_deltas=by_seed.mean(), seed_sd=by_seed.std(ddof=1))
    if all(has_subgroup_precision(mask, memberships) for mask in masks):
        a, b = (common.paired_bootstrap(delta[:, mask], memberships[mask], draws)[0] for mask in masks)
        lo, hi = common.interval(a - b)
        result.update(crossed95_low=lo, crossed95_high=hi, valid_bootstrap_draws=int(np.isfinite(a - b).sum()))
    return result


def score(
    data: pd.DataFrame, runs: dict, *, n_draws: int = common.N_BOOTSTRAP, bootstrap_seed: int = common.BOOTSTRAP_SEED,
) -> dict[str, pd.DataFrame]:
    """Score aligned predictions, reusing group/seed draws across comparisons."""
    if set(runs) != expected_runs() or n_draws < 1:
        raise ValueError("Scoring requires all validated runs and at least one bootstrap draw")
    raw = data[list(common.RAW_ENDPOINTS)].to_numpy(dtype=float)
    transformed = common.transform_targets(raw)
    errors = {run: common.errors(pred, raw, transformed) for run, pred in runs.items()}
    memberships, groups = pd.factorize(data["group_id"], sort=True)
    draws = common.group_draws(len(groups), len(common.SEEDS), n_draws, np.random.default_rng(bootstrap_seed))
    no_draws = common.BootstrapDraws(draws.group_counts[:0], draws.seed_counts[:0])
    audited, switches = subgroup_masks(data)
    all_mask = np.ones(len(data), dtype=bool)
    outer_folds = data["outer_fold"].to_numpy()
    contrasts, seeds, folds, interactions = [], [], [], []
    for arm, comparator, purpose in run_contrasts():
        meta = dict(zip(common.RUN_COLUMNS, arm), comparator_procedure=comparator[1],
                    comparator_backbone=comparator[2], comparator_arm=comparator[3], purpose=purpose)
        a, b = errors[arm], errors[comparator]
        scopes = [("all", "all", all_mask), *contrast_scopes(purpose, arm[3], audited, switches)]
        for j, endpoint in enumerate(common.ENDPOINTS):
            labelled = np.isfinite(raw[:, j])
            delta = a[:, :, j] - b[:, :, j]
            endpoint_meta = dict(**meta, endpoint=endpoint, unit=unit_for(j))
            for scope, stratum, mask in scopes:
                observed = labelled & mask
                primary = scope == "all" and purpose == "headline" and endpoint in PRIMARY_ENDPOINTS
                eligible = scope == "all" or has_subgroup_precision(observed, memberships)
                stats, by_seed = common.contrast_statistics(
                    a[:, observed, j], b[:, observed, j], memberships[observed], draws if eligible else no_draws,
                    primary=primary,
                )
                if not eligible:
                    stats.update(seed_t95_low=np.nan, seed_t95_high=np.nan, seed_t_df=np.nan,
                                 seed_t95_direction="unavailable", interval_direction_disagreement=False)
                record = dict(**endpoint_meta, scope=scope, stratum=stratum, primary_test=primary,
                              interval_eligible=eligible)
                contrasts.append(dict(**record, **stats))
                for i, seed in enumerate(common.SEEDS):
                    seeds.append(dict(**record, seed=seed, delta=by_seed[i],
                                      arm_mae=a[i, observed, j].mean() if observed.any() else np.nan,
                                      comparator_mae=b[i, observed, j].mean() if observed.any() else np.nan))
                if scope == "switch_status" and stratum.endswith("/switcher"):
                    interactions.append(dict(**endpoint_meta, contrast=stratum.split("/")[0],
                                             **interaction_statistics(delta, labelled, mask, memberships, draws)))
            for fold in common.FOLDS:
                observed = labelled & (outer_folds == fold)
                folds.append(dict(**endpoint_meta, outer_fold=fold, n_structures=int(observed.sum()),
                                  n_groups=len(np.unique(memberships[observed])),
                                  delta_mean=delta[:, observed].mean() if observed.any() else np.nan))
    metrics, summary = endpoint_tables(runs, raw, transformed, errors, memberships)
    contrast_table = pd.DataFrame(contrasts)
    if contrast_table["primary_test"].sum() != 6:
        raise RuntimeError("Expected exactly six adjusted primary endpoint/model comparisons")
    group_sizes = data.groupby(["group_id", "outer_fold"], sort=True).size().rename("n_structures").reset_index()
    return {
        "endpoint_metrics": metrics, "endpoint_summary": summary,
        "contrasts": contrast_table, "contrast_seeds": pd.DataFrame(seeds), "contrast_outer_folds": pd.DataFrame(folds),
        "switcher_interactions": pd.DataFrame(interactions), "coverage": coverage(data, audited, switches),
        "group_sizes": group_sizes.assign(cohort=COHORT),
    }


def score_export(release: Path, predictions: Path, out: Path) -> dict:
    """Validate the frozen release and write the postflight tables and hashes."""
    fingerprints = common.frozen_release_fingerprints(release, (COHORT,), diagnostic=False)
    if fingerprints[COHORT] != RELEASE_FINGERPRINT:
        raise ValueError("Unexpected frozen numerical release for this protocol")
    data = common.load_cohort(release, COHORT)
    if len(data) != 327 or data["group_id"].nunique() != 238:
        raise ValueError("Expected the 327-structure, 238-group numerical cohort")
    prediction_hash = sha256(predictions)
    collector = checked_campaign(predictions, prediction_hash, fingerprints[COHORT])
    frame = common.read_csv(predictions)
    if collector and "rows" in collector and collector["rows"] != len(frame):
        raise ValueError("Prediction row count differs from the collector manifest")
    campaign_id = collector["campaign_id"] if collector else None
    runs = validate_predictions(frame, data, campaign_id)
    sources = prediction_sources(frame, campaign_id)
    reused = sources["arm"].isin((*REFERENCES, *HISTORICAL_ARMS))
    if not sources.loc[reused, "source_campaign_id"].eq(PRIOR_CAMPAIGN).all():
        raise ValueError("Historical predictions must come from the frozen prior numerical campaign")
    provenance_hash = check_provenance(predictions, collector, sources)
    with threadpool_limits(limits=1, user_api="blas"):
        tables = score(data, runs)
    tables["prediction_sources"] = sources
    meta = dict(
        status="report_ready", cohort=COHORT, release_fingerprints=fingerprints,
        campaign_id=str(sources.loc[sources["arm"].eq("ADD2"), "source_campaign_id"].iloc[0]),
        predictions_sha256=prediction_hash, predictions_rows=len(frame),
        collector_manifest_sha256=sha256(predictions.with_suffix(".manifest.json")) if collector else None,
        source_fit_ledger_sha256=provenance_hash, prior_campaign_id=PRIOR_CAMPAIGN,
        prediction_fit_count=int(sources[SOURCE_KEYS].drop_duplicates().shape[0]),
        n_structures=len(data), n_groups=int(data["group_id"].nunique()),
        outer_folds=common.FOLDS, seeds=common.SEEDS,
        bootstrap_seed=common.BOOTSTRAP_SEED, bootstrap_draws=common.N_BOOTSTRAP,
        adjusted_level=common.ADJUSTED_LEVEL, primary_family_size=6,
        primary_rule="Shared-recipe ALL4 minus REP4 on Vd and F in three backbones; adjusted upper bound below zero.",
        delta="Arm minus comparator. Negative MAE deltas favour the first arm.",
        relative_delta="100 × paired mean delta / comparator MAE averaged over nine seeds; a fixed denominator.",
        crossed_interval="Approximate equal-tailed paired percentile interval over whole seed trajectories and frozen groups, conditional on observed training data, fixed folds and selected settings.",
        conditional_group_interval="Equal-tailed paired percentile interval over frozen groups with all trained seeds fixed.",
        seed_interval="Paired t interval over whole-trajectory deltas on the fixed observed structures. Seed SD separately describes dispersion of paired seed effects.",
        masks="Within each endpoint, both arms use the same observed labels. Invalid, missing and duplicate predictions fail validation before scoring.",
        aafe="10 ** mean absolute log10 error on strictly positive observed labels; zero predictions yield infinite AAFE.",
        subgroup_minimum={"labelled_structures": MIN_LABELS, "labelled_groups": MIN_GROUPS},
        switcher_interactions="Switcher minus stable difference of paired absolute MAE deltas. Bootstrap retains shared seed/group resamples across both strata.",
        empty_draws="Draws with no labelled structure in an endpoint/stratum are undefined. Valid draw counts are recorded.",
    )
    if sha256(predictions) != prediction_hash:
        raise ValueError("Predictions changed during scoring")
    if predictions.resolve().parent == out.resolve() and predictions.name in {f"{name}.csv" for name in tables}:
        raise ValueError("Scoring output would overwrite its predictions input")
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".scoring-", dir=out) as directory:
        temporary = Path(directory)
        for name, table in tables.items():
            table.to_csv(temporary / f"{name}.csv", index=False)
        meta["files"] = {f"{name}.csv": sha256(temporary / f"{name}.csv") for name in tables}
        meta_name = "scoring_manifest.json"
        (temporary / meta_name).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        for filename in (*meta["files"], meta_name):
            (temporary / filename).replace(out / filename)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    score_export(args.release, args.predictions, args.out)


if __name__ == "__main__":
    main()
