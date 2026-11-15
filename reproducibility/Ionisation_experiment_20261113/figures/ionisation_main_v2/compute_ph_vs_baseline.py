"""Score each shared-recipe pH representation directly against REF0 for all endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from compute_average import ANALYSIS, OUT, ROOT, common, compare, followup, original_rows

ARMS = ("REF74", "REP2", "ADD2", "ADD5", "ADD65", "REP4", "ALL4")
ENDPOINTS = common.ENDPOINTS


def main() -> None:
    release = ROOT / "data" / "release"
    numerical = release / followup.COHORT
    source = ROOT / "outputs" / "ph_followup" / "outer_predictions.csv"
    original = ROOT / "outputs" / "scoring"
    original_manifest = original / "scoring_manifest.json"
    panel_estimates = OUT / "panel_b_estimates.csv"
    panel_seeds = OUT / "panel_b_seeds.csv"
    source_paths = [
        Path(__file__).resolve(), OUT / "compute_average.py", ANALYSIS / "ionisation_score.py",
        ANALYSIS / "ph_ionisation_score.py", numerical / "manifest.json", numerical / "model_dataset.csv",
        numerical / "outer.csv", source, source.with_suffix(".manifest.json"),
        source.with_suffix(".provenance.json"), original_manifest, original / "contrasts.csv",
        original / "contrast_seeds.csv", panel_estimates, panel_seeds,
    ]
    source_hashes = {path.relative_to(ROOT).as_posix(): followup.sha256(path) for path in source_paths}
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

    raw = data[list(common.RAW_ENDPOINTS)].to_numpy(dtype=float)
    transformed = common.transform_targets(raw)
    draws = common.group_draws(len(group_ids), len(common.SEEDS), common.N_BOOTSTRAP,
                               np.random.default_rng(common.BOOTSTRAP_SEED))
    estimates = []
    seeds = []
    with threadpool_limits(limits=1, user_api="blas"):
        for backbone in common.BACKBONES:
            base_error = common.errors(runs[(followup.COHORT, "primary", backbone, "REF0")], raw, transformed)
            arm_errors = {arm: common.errors(runs[(followup.COHORT, "primary", backbone, arm)], raw, transformed)
                          for arm in ARMS}
            for j, endpoint in enumerate(ENDPOINTS):
                labelled = np.isfinite(raw[:, j])
                base = base_error[:, labelled, j]
                for arm, arm_error in arm_errors.items():
                    summary, by_seed = common.contrast_statistics(arm_error[:, labelled, j], base,
                                                                  groups[labelled], draws, primary=False)
                    valid = summary["valid_bootstrap_draws"]
                    if valid != common.N_BOOTSTRAP or not np.isfinite(summary["relative_delta_pct"]):
                        raise ValueError(f"Invalid paired contrast for {backbone} {endpoint} {arm}")
                    denominator = summary["comparator_mae"]
                    estimates.append(dict(arm=arm, backbone=backbone, endpoint=endpoint,
                                          relative_delta_pct=summary["relative_delta_pct"],
                                          crossed95_low_pct=100 * summary["crossed95_low"] / denominator,
                                          crossed95_high_pct=100 * summary["crossed95_high"] / denominator,
                                          n_structures=summary["n_structures"], n_groups=summary["n_groups"],
                                          unit="percentage_point" if j in common.PERCENT else "log10",
                                          comparator_mae=denominator, n_seeds=summary["n_seeds"],
                                          valid_bootstrap_draws=valid))
                    seeds.extend(dict(arm=arm, backbone=backbone, endpoint=endpoint, seed=seed,
                                      relative_delta_pct=100 * delta / denominator)
                                 for seed, delta in zip(common.SEEDS, by_seed, strict=True))

    estimate_table = pd.DataFrame(estimates)
    seed_table = pd.DataFrame(seeds)
    keys = ["backbone", "endpoint"]
    seed_keys = [*keys, "seed"]
    reference = estimate_table.query("arm == 'REF74'").set_index(keys, verify_integrity=True)
    reference_seed = seed_table.query("arm == 'REF74'").set_index(seed_keys, verify_integrity=True)
    old_point = original_rows(original / "contrasts.csv", keys)
    old_seed = original_rows(original / "contrast_seeds.csv", seed_keys)
    old_seed_relative = 100 * old_seed["delta"].div(old_point["comparator_mae"])
    panel_point = common.read_csv(panel_estimates).set_index(keys, verify_integrity=True)
    panel_seed = common.read_csv(panel_seeds).set_index(seed_keys, verify_integrity=True)
    differences = {
        "original_REF74_comparator_mae": compare(reference["comparator_mae"],
                                               old_point.loc[reference.index, "comparator_mae"], "original REF0 MAE"),
        "original_REF74_estimates_pct": compare(reference["relative_delta_pct"],
                                              old_point.loc[reference.index, "relative_delta_pct"], "original REF74 estimates"),
        "original_REF74_seed_effects_pct": compare(reference_seed["relative_delta_pct"],
                                                 old_seed_relative.loc[reference_seed.index], "original REF74 seed effects"),
        "panel_b_REF74_estimates_pct": compare(reference["relative_delta_pct"],
                                             panel_point.loc[reference.index, "relative_delta_pct"], "panel B REF74 estimates"),
        "panel_b_REF74_seed_effects_pct": compare(reference_seed["relative_delta_pct"],
                                                panel_seed.loc[reference_seed.index, "relative_delta_pct"], "panel B REF74 seed effects"),
    }
    for path in source_paths:
        key = path.relative_to(ROOT).as_posix()
        if followup.sha256(path) != source_hashes[key]:
            raise ValueError(f"Source changed during calculation: {key}")

    metadata = dict(
        status="report_ready", cohort=followup.COHORT, procedure="primary", comparator="REF0", arms=ARMS,
        endpoints=ENDPOINTS, n_structures=len(data), n_groups=len(group_ids), seeds=common.SEEDS,
        prior_campaign_id=followup.PRIOR_CAMPAIGN, export_campaign_id=collector["campaign_id"],
        release_fingerprints={followup.COHORT: fingerprint},
        estimand="100 × paired mean MAE difference versus REF0 / fixed original nine-seed mean REF0 endpoint MAE.",
        error_units="Absolute log10 errors for t_half, Vd, CL, Cmax and Tmax. Absolute physical percentage-point errors for F and PPB, including observed zeros.",
        bootstrap_seed=common.BOOTSTRAP_SEED, bootstrap_draws=common.N_BOOTSTRAP,
        bootstrap="Resample whole frozen groups and nine complete seed trajectories. Use the same group and seed draw counts for all arms, endpoints and backbones. Each endpoint divides by its own resampled observed label count. REF0 normalization stays fixed.",
        interval_level=0.95, inference="Exploratory, unadjusted. ADD65 is descriptive because few structures change category at pH 6.5.",
        checks=dict(n_REF74_estimates=len(reference), n_REF74_seed_effects=len(reference_seed),
                    max_absolute_differences=differences),
        source_sha256=source_hashes,
    )
    estimate_name = "panel_ph_vs_baseline_estimates.csv"
    seed_name = "panel_ph_vs_baseline_seeds.csv"
    meta_name = "panel_ph_vs_baseline_manifest.json"
    with TemporaryDirectory(prefix=".ph-baseline-", dir=OUT) as directory:
        temp = Path(directory)
        estimate_table.to_csv(temp / estimate_name, index=False)
        seed_table.to_csv(temp / seed_name, index=False)
        metadata["files"] = {name: followup.sha256(temp / name) for name in (estimate_name, seed_name)}
        (temp / meta_name).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        for name in (estimate_name, seed_name, meta_name):
            (temp / name).replace(OUT / name)


if __name__ == "__main__":
    main()
