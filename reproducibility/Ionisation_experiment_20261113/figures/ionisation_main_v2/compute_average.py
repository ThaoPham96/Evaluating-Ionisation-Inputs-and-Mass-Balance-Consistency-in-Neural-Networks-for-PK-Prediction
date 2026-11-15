"""Reproduce the equal-endpoint average of the numerical P1 minus P0 MAE effects."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
ANALYSIS = ROOT / "analysis"
sys.path.insert(0, str(ANALYSIS))

import ionisation_score as common
import ph_ionisation_score as followup

AVERAGE = "average"
TOLERANCE = 1e-12


def compare(actual: pd.Series, expected: pd.Series, label: str) -> float:
    """Verify keyed component estimates and return the largest absolute difference."""
    if not actual.sort_index().index.equals(expected.sort_index().index):
        raise ValueError(f"Unexpected component keys in {label}")
    difference = float((actual - expected).abs().max())
    np.testing.assert_allclose(actual, expected.reindex(actual.index), rtol=0, atol=TOLERANCE,
                               equal_nan=False, err_msg=label)
    return difference


def original_rows(path: Path, keys: list[str]) -> pd.DataFrame:
    rows = common.read_csv(path)
    selected = rows.query(
        "cohort == 'numerical' and procedure == 'primary' and arm == 'P1' and "
        "comparator_procedure == 'primary' and comparator_arm == 'P0' and "
        "backbone == comparator_backbone and scope == 'all' and stratum == 'all'"
    )
    return selected.set_index(keys, verify_integrity=True)


def main() -> None:
    release = ROOT / "data" / "release"
    source = ROOT / "outputs" / "ph_followup" / "outer_predictions.csv"
    original = ROOT / "outputs" / "scoring"
    original_manifest = original / "scoring_manifest.json"
    source_paths = [
        Path(__file__).resolve(), ANALYSIS / "ionisation_score.py", ANALYSIS / "ph_ionisation_score.py",
        release / "numerical" / "manifest.json", release / "numerical" / "model_dataset.csv",
        release / "numerical" / "outer.csv", source, source.with_suffix(".manifest.json"),
        source.with_suffix(".provenance.json"), original_manifest, original / "contrasts.csv",
        original / "contrast_seeds.csv", OUT / "panel_b_estimates.csv", OUT / "panel_b_seeds.csv",
    ]
    source_hashes = {str(path.relative_to(ROOT).as_posix()): followup.sha256(path) for path in source_paths}
    fingerprint = common.frozen_release_fingerprints(release, (followup.COHORT,), diagnostic=False)[followup.COHORT]
    if fingerprint != followup.RELEASE_FINGERPRINT:
        raise ValueError("Unexpected numerical release fingerprint")
    data = common.load_cohort(release, followup.COHORT)
    groups, group_ids = pd.factorize(data["group_id"], sort=True)
    if len(data) != 327 or len(group_ids) != 238:
        raise ValueError("Expected the frozen 327-structure, 238-group numerical cohort")

    collector = followup.checked_campaign(source, followup.sha256(source), fingerprint)
    if collector is None or collector.get("prior_campaign_id") != followup.PRIOR_CAMPAIGN:
        raise ValueError("Expected a verified follow-up export with prior campaign references")
    frame = common.read_csv(source)
    if len(frame) != collector["rows"]:
        raise ValueError("Unexpected number of outer predictions")
    runs = followup.validate_predictions(frame, data, collector["campaign_id"])
    sources = followup.prediction_sources(frame, collector["campaign_id"])
    reused = sources["arm"].isin(followup.REFERENCES)
    if not sources.loc[reused, "source_campaign_id"].eq(followup.PRIOR_CAMPAIGN).all():
        raise ValueError("REF74 and REF0 must come from the frozen prior numerical campaign")
    if followup.check_provenance(source, collector, sources) is None:
        raise ValueError("A verified source-fit provenance ledger is required")

    original_meta = json.loads(original_manifest.read_text(encoding="utf-8"))
    if (original_meta["status"] != "report_ready" or original_meta["campaign_id"] != followup.PRIOR_CAMPAIGN
            or original_meta["release_fingerprints"][followup.COHORT] != fingerprint):
        raise ValueError("Original scoring manifest does not match the reference predictions")
    for name in ("contrasts.csv", "contrast_seeds.csv"):
        if original_meta["files"][name] != followup.sha256(original / name):
            raise ValueError(f"Original scoring table differs from its manifest: {name}")

    targets = data[list(common.RAW_ENDPOINTS)].to_numpy(dtype=float)
    transformed = common.transform_targets(targets)
    masks = np.isfinite(targets)
    # Reuse a single draw matrix across the seven endpoints and all backbones.
    draws = common.group_draws(len(group_ids), len(common.SEEDS), common.N_BOOTSTRAP,
                               np.random.default_rng(common.BOOTSTRAP_SEED))
    component_keys = ["backbone", "endpoint"]
    seed_keys = [*component_keys, "seed"]
    component_index = pd.MultiIndex.from_product([common.BACKBONES, common.ENDPOINTS], names=component_keys)
    seed_index = pd.MultiIndex.from_product([common.BACKBONES, common.ENDPOINTS, common.SEEDS], names=seed_keys)
    baseline = pd.Series(index=component_index, dtype=float)
    effects = pd.Series(index=seed_index, dtype=float)
    estimates = []
    seeds = []
    with threadpool_limits(limits=1, user_api="blas"):
        for backbone in common.BACKBONES:
            arm = common.errors(runs[(followup.COHORT, "primary", backbone, "REF74")], targets, transformed)
            base = common.errors(runs[(followup.COHORT, "primary", backbone, "REF0")], targets, transformed)
            endpoint_seed_effects = []
            endpoint_bootstrap_effects = []
            for j, endpoint in enumerate(common.ENDPOINTS):
                labelled = masks[:, j]
                baseline_mae = base[:, labelled, j].mean()
                if not np.isfinite(baseline_mae) or baseline_mae <= 0:
                    raise ValueError(f"Expected positive baseline MAE for {backbone} {endpoint}")
                scale = 100 / baseline_mae
                delta = arm[:, labelled, j] - base[:, labelled, j]
                per_seed = delta.mean(axis=1) * scale
                crossed, _ = common.paired_bootstrap(delta, groups[labelled], draws)
                endpoint_seed_effects.append(per_seed)
                endpoint_bootstrap_effects.append(crossed * scale)
                baseline.loc[backbone, endpoint] = baseline_mae
                effects.loc[backbone, endpoint, :] = per_seed

            seed_mean = np.mean(endpoint_seed_effects, axis=0)
            crossed_mean = np.mean(endpoint_bootstrap_effects, axis=0)
            valid = int(np.isfinite(crossed_mean).sum())
            if valid != common.N_BOOTSTRAP:
                raise ValueError(f"Some draws lack complete endpoint coverage for {backbone}")
            low, high = common.interval(crossed_mean)
            estimates.append(dict(backbone=backbone, endpoint=AVERAGE, relative_delta_pct=seed_mean.mean(),
                                  crossed95_low_pct=low, crossed95_high_pct=high,
                                  seed_sd_pct=seed_mean.std(ddof=1), seeds_below_zero=int((seed_mean < 0).sum()),
                                  n_seeds=len(common.SEEDS), valid_bootstrap_draws=valid,
                                  requested_bootstrap_draws=common.N_BOOTSTRAP))
            seeds.extend(dict(backbone=backbone, endpoint=AVERAGE, seed=seed, relative_delta_pct=value)
                         for seed, value in zip(common.SEEDS, seed_mean, strict=True))

    main_point = common.read_csv(OUT / "panel_b_estimates.csv")
    main_point = main_point[main_point["endpoint"].isin(common.ENDPOINTS)].set_index(component_keys, verify_integrity=True)
    main_seed = common.read_csv(OUT / "panel_b_seeds.csv")
    main_seed = main_seed[main_seed["endpoint"].isin(common.ENDPOINTS)].set_index(seed_keys, verify_integrity=True)
    original_point = original_rows(original / "contrasts.csv", component_keys)
    original_seed = original_rows(original / "contrast_seeds.csv", seed_keys)
    original_seed_delta = 100 * original_seed["delta"].div(original_point["comparator_mae"])
    means = effects.groupby(level=component_keys).mean()
    differences = {
        "original_comparator_mae": compare(baseline, original_point["comparator_mae"], "original comparator MAE"),
        "original_seed_relative_delta_pct": compare(effects, original_seed_delta, "original seed contrasts"),
        "original_mean_relative_delta_pct": compare(means, original_point["relative_delta_pct"], "original estimates"),
        "panel_b_seed_relative_delta_pct": compare(effects, main_seed["relative_delta_pct"], "panel B seed contrasts"),
        "panel_b_mean_relative_delta_pct": compare(means, main_point["relative_delta_pct"], "panel B estimates"),
    }
    for path in source_paths:
        key = path.relative_to(ROOT).as_posix()
        if followup.sha256(path) != source_hashes[key]:
            raise ValueError(f"Source changed during calculation: {key}")

    metadata = dict(
        status="report_ready", cohort=followup.COHORT, procedure="primary", arm="P1", comparator="P0",
        reference_predictions={"P1": "REF74", "P0": "REF0"}, prior_campaign_id=followup.PRIOR_CAMPAIGN,
        export_campaign_id=collector["campaign_id"], release_fingerprints={followup.COHORT: fingerprint},
        n_structures=len(data), n_groups=len(group_ids), seeds=common.SEEDS, endpoints=common.ENDPOINTS,
        labelled_structures={endpoint: int(masks[:, j].sum()) for j, endpoint in enumerate(common.ENDPOINTS)},
        labelled_groups={endpoint: int(len(np.unique(groups[masks[:, j]]))) for j, endpoint in enumerate(common.ENDPOINTS)},
        endpoint_weight=1 / len(common.ENDPOINTS),
        estimand="Within each seed, average the seven endpoint values of 100 × (P1 MAE − P0 MAE) / original nine-seed mean P0 endpoint MAE. Average the nine seed values for the point estimate.",
        error_units="Absolute log10 errors for t_half, Vd, CL, Cmax and Tmax. Absolute physical percentage-point errors for F and PPB, including observed zeros.",
        bootstrap_seed=common.BOOTSTRAP_SEED, bootstrap_draws=common.N_BOOTSTRAP,
        bootstrap="Joint crossed percentile interval using the numerical-cohort RNG stream. Each draw resamples 238 frozen groups and nine entire seed trajectories with replacement. Shared draw counts preserve pairing across arms, endpoints and backbones. Each endpoint divides by its own resampled observed label count and uses its fixed original P0 MAE normalization. The seven endpoint effects are then averaged within the draw.",
        interval_level=0.95, inference="Exploratory, unadjusted.",
        checks=dict(n_endpoint_estimates=len(means), n_endpoint_seed_effects=len(effects),
                    absolute_tolerance=TOLERANCE, max_absolute_differences=differences),
        comparator_mae={backbone: baseline.loc[backbone].to_dict() for backbone in common.BACKBONES},
        source_sha256=source_hashes,
    )
    estimate_name = "panel_b_average_estimates.csv"
    seed_name = "panel_b_average_seeds.csv"
    meta_name = "panel_b_average_manifest.json"
    with TemporaryDirectory(prefix=".average-", dir=OUT) as directory:
        temp = Path(directory)
        pd.DataFrame(estimates).to_csv(temp / estimate_name, index=False)
        pd.DataFrame(seeds).to_csv(temp / seed_name, index=False)
        metadata["files"] = {name: followup.sha256(temp / name) for name in (estimate_name, seed_name)}
        (temp / meta_name).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        for name in (estimate_name, seed_name, meta_name):
            (temp / name).replace(OUT / name)


if __name__ == "__main__":
    main()
