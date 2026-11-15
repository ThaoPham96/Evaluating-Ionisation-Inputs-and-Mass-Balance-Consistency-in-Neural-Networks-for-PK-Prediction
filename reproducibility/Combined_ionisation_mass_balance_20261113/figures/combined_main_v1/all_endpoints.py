"""Score the exploratory average percent MAE change across seven endpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "analysis"))
import combined_score as score


CELLS = ("10", "11")
CONTRAST = "11_minus_10"
CI_LEVEL = 0.95
ATOL = 1e-12


def load_predictions(
    export: Path, scores: Path, release: Path, triads: Path,
) -> tuple[dict, score.Cohort, dict[str, np.ndarray]]:
    cohort = score.load_cohort(release, triads)
    manifest, final, p1_zero, grid, declared = score.read_export(export)
    scored = json.loads((scores / "manifest.json").read_text(encoding="utf-8"))
    if (scored.get("status") != "complete"
            or scored["campaign_id"] != manifest["campaign_id"]
            or scored["plan_sha256"] != manifest["plan_sha256"]
            or scored["export_manifest_sha256"] != score.sha256(export / "manifest.json")
            or scored["source_triads_sha256"] != score.sha256(triads)
            or scored["scorer_sha256"] != score.sha256(Path(score.__file__))
            or scored["release_fingerprint"] != score.FINGERPRINT
            or scored["bootstrap_seed"] != score.BOOTSTRAP_SEED
            or scored["bootstrap_draws"] != score.N_BOOTSTRAP):
        raise ValueError("Prediction export, scorer and bootstrap settings must match the completed score manifest")
    for filename, expected in scored["files"].items():
        if score.sha256(scores / filename) != expected:
            raise ValueError(f"Score-table checksum mismatch for {filename}")
    validated_inner = score.validate_inner(grid, p1_zero, cohort)
    selection, _ = score.audit_selections(validated_inner, declared, manifest["campaign_id"])
    return scored, cohort, score.validate_final(final, cohort, selection)


def compare(label: str, actual, archived) -> None:
    if not np.allclose(actual, archived, rtol=0, atol=ATOL):
        raise ValueError(f"Endpoint calculation differs from the archived scores for {label}")


def aggregate(
    predictions: dict[str, np.ndarray], cohort: score.Cohort, scores: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    archived_metrics = score.read_csv(scores / "seed_metrics.csv").set_index(["backbone", "metric", "cell", "seed"])
    archived_contrasts = score.read_csv(scores / "contrasts.csv").set_index(["backbone", "metric", "contrast"])
    archived_seed_contrasts = score.read_csv(scores / "seed_contrasts.csv").set_index(
        ["backbone", "metric", "contrast", "seed"],
    )
    for frame in (archived_metrics, archived_contrasts, archived_seed_contrasts):
        frame.sort_index(inplace=True)
    n_seeds = len(score.SEEDS)
    rng = np.random.default_rng(score.BOOTSTRAP_SEED)
    draws = score.prior.group_draws(len(np.unique(cohort.group)), n_seeds, score.N_BOOTSTRAP, rng)
    summary, seed_rows, denominators = [], [], {}
    for backbone, all_cells in predictions.items():
        selected = all_cells[[score.CELLS.index(cell) for cell in CELLS]]
        outcomes, _, _ = score.metrics(selected, cohort)
        seed_effects, draw_effects, normalization = [], [], {}
        for endpoint, count in zip(score.ENDPOINTS, score.LABEL_COUNTS, strict=True):
            metric = f"{endpoint}_mae"
            key = (backbone, metric, CONTRAST)
            errors, mask, _ = outcomes[metric]
            if mask.sum() != count:
                raise ValueError(f"Observed-label count differs for {endpoint}")
            absolute_errors = errors[..., mask]
            seed_mae = absolute_errors.mean(axis=-1)
            mean_mae = seed_mae.mean(axis=-1)
            paired = seed_mae[1] - seed_mae[0]
            for i, cell in enumerate(CELLS):
                archived = archived_metrics.loc[(backbone, metric, cell), "value"].reindex(score.SEEDS)
                compare(f"{backbone}/{metric}/{cell}/seed", seed_mae[i], archived)
            contrast = archived_contrasts.loc[key]
            seed_contrast = archived_seed_contrasts.loc[key, "value"].reindex(score.SEEDS)
            compare(f"{backbone}/{metric}/seed contrast", paired, seed_contrast)
            compare(f"{backbone}/{metric}/mean cells", mean_mae, contrast[["mean_10", "mean_11"]].to_numpy(dtype=float))
            compare(f"{backbone}/{metric}/mean contrast", paired.mean(), contrast["mean"])
            boot_mae = score.crossed_means(absolute_errors, cohort.group[mask], draws)
            boot_delta = boot_mae[1] - boot_mae[0]
            compare(f"{backbone}/{metric}/95% bounds", score.prior.interval(boot_delta, CI_LEVEL),
                    contrast[["crossed95_low", "crossed95_high"]].to_numpy(dtype=float))
            baseline = float(mean_mae[0])
            if baseline <= 0 or not np.isfinite(baseline):
                raise ValueError(f"Expected a positive fixed ionisation-only MAE for {backbone}/{endpoint}")
            normalization[endpoint] = baseline
            seed_effects.append(100 * paired / baseline)
            draw_effects.append(100 * boot_delta / baseline)
        pooled_seeds = np.mean(seed_effects, axis=0)
        pooled_draws = np.mean(draw_effects, axis=0)
        lo, hi = score.prior.interval(pooled_draws, CI_LEVEL)
        summary.append(dict(backbone=backbone, mean_pct=pooled_seeds.mean(), crossed95_low_pct=lo,
                            crossed95_high_pct=hi, n_seeds=n_seeds, seed_sd_pct=pooled_seeds.std(ddof=1)))
        seed_rows.extend(dict(backbone=backbone, seed=seed, value_pct=value)
                         for seed, value in zip(score.SEEDS, pooled_seeds, strict=True))
        denominators[backbone] = normalization
    return pd.DataFrame(summary), pd.DataFrame(seed_rows), denominators


def run(export: Path, scores: Path, release: Path, triads: Path, output: Path) -> None:
    source_manifest, cohort, predictions = load_predictions(export, scores, release, triads)
    with threadpool_limits(limits=2, user_api="blas"):
        summary, seeds, denominators = aggregate(predictions, cohort, scores)
    output.mkdir(parents=True, exist_ok=True)
    summary_path, seeds_path = output / "all_endpoint_summary.csv", output / "all_endpoint_seeds.csv"
    summary.to_csv(summary_path, index=False)
    seeds.to_csv(seeds_path, index=False)
    record = {
        "campaign_id": source_manifest["campaign_id"],
        "score_manifest_sha256": score.sha256(scores / "manifest.json"),
        "method": "Equal-weight mean of seven endpoint percent MAE changes, cell 11 minus cell 10.",
        "normalization": "Each endpoint uses its fixed cell 10 mean MAE across nine seeds within backbone.",
        "labels": "All observed labels per endpoint. Zero labels are included.",
        "label_counts": dict(zip(score.ENDPOINTS, score.LABEL_COUNTS, strict=True)),
        "endpoint_baseline_10_mae": denominators,
        "bootstrap": "Exploratory equal-tailed crossed percentile intervals. The joint draws resample groups and seeds "
                     "across all endpoints and backbones. Endpoint MAEs use resampled observed-label counts.",
        "bootstrap_seed": score.BOOTSTRAP_SEED,
        "bootstrap_draws": score.N_BOOTSTRAP,
        "ci_level": CI_LEVEL,
        "n_structures": len(cohort.rows),
        "n_groups": len(np.unique(cohort.group)),
        "seeds": score.SEEDS,
        "verified_endpoint_checks": ["cell seed MAEs", "cell means", "paired seed effects",
                                     "mean contrasts", "95% crossed interval bounds"],
        "verification_absolute_tolerance": ATOL,
        "source_sha256": {
            "all_endpoints.py": score.sha256(Path(__file__)),
            "analysis/combined_score.py": score.sha256(Path(score.__file__)),
            "analysis/vendor_ionisation/ionisation_score.py": score.sha256(Path(score.prior.__file__)),
            "outputs/scores/manifest.json": score.sha256(scores / "manifest.json"),
            "outputs/scores/seed_metrics.csv": score.sha256(scores / "seed_metrics.csv"),
            "outputs/scores/contrasts.csv": score.sha256(scores / "contrasts.csv"),
            "outputs/scores/seed_contrasts.csv": score.sha256(scores / "seed_contrasts.csv"),
            "outputs/export/manifest.json": score.sha256(export / "manifest.json"),
            "outputs/export/final_predictions.csv": score.sha256(export / "final_predictions.csv"),
            "data/release/numerical/manifest.json": score.sha256(release / "manifest.json"),
            "runtime/same_row_complete_triads.csv": score.sha256(triads),
        },
        "files": {path.name: score.sha256(path) for path in (summary_path, seeds_path)},
    }
    destination = output / "all_endpoint_manifest.json"
    temp = destination.with_suffix(".json.tmp")
    temp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temp.replace(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, default=score.ROOT / "outputs" / "export")
    parser.add_argument("--scores", type=Path, default=score.ROOT / "outputs" / "scores")
    parser.add_argument("--release", type=Path, default=score.ROOT / "data" / "release" / "numerical")
    parser.add_argument("--triads", type=Path, default=score.ROOT / "runtime" / "same_row_complete_triads.csv")
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    run(args.export, args.scores, args.release, args.triads, args.output)
