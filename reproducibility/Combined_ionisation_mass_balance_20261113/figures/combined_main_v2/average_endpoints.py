"""Score paired seven-endpoint average changes relative to structure only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "combined_main_v1"))
from all_endpoints import ATOL, CI_LEVEL, compare, load_predictions, score
from endpoint_panel import (
    AVG_METRIC, CONDITIONS, CONTRASTS, ENDPOINTS, KEYS, MANIFEST_FILE, SEEDS_FILE, SUMMARY_FILE, panel_data,
)


def aggregate(scores: Path, predictions: dict, cohort: score.Cohort) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = {name: score.read_csv(scores / f"{name}.csv")
              for name in ("contrasts", "seed_contrasts", "seed_metrics")}
    expected_means, expected_seeds = panel_data(frames)
    draws = score.prior.group_draws(len(np.unique(cohort.group)), len(score.SEEDS), score.N_BOOTSTRAP,
                                   np.random.default_rng(score.BOOTSTRAP_SEED))
    summary, seeds = [], []
    baseline_index = score.CELLS.index("00")
    for backbone, q in predictions.items():
        outcomes, _, _ = score.metrics(q, cohort)
        paired_by_ep, boot_by_ep, centred_means_by_ep = [], [], []
        for metric in ENDPOINTS:
            errors, observed, _ = outcomes[metric]
            seed_mae = errors[..., observed].mean(axis=-1)
            draw_mae = score.crossed_means(errors[..., observed], cohort.group[observed], draws)
            denom00 = seed_mae[baseline_index].mean()
            if not np.isfinite(denom00) or denom00 <= 0:
                raise ValueError(f"Invalid fixed structure-only normalizer for {backbone}/{metric}")
            paired = 100 * (seed_mae - seed_mae[baseline_index]) / denom00
            boot = 100 * (draw_mae - draw_mae[baseline_index]) / denom00
            centred = 100 * (seed_mae / denom00 - 1)
            for cell, contrast in CONTRASTS.items():
                index = score.CELLS.index(cell)
                key = (backbone, metric, contrast)
                expected = expected_means.loc[key]
                compare(f"{key} seed values", paired[index], expected_seeds["value_pct"].xs(key, level=KEYS))
                compare(f"{key} mean", paired[index].mean(), expected["mean_pct"])
                # The stored interval bounds are in native MAE units. Check
                # there before dividing by the fixed structure-only baseline.
                compare(f"{key} 95% interval",
                        score.prior.interval(draw_mae[index] - draw_mae[baseline_index], CI_LEVEL),
                        expected[["crossed95_low", "crossed95_high"]].to_numpy(dtype=float))
            paired_by_ep.append(paired)
            boot_by_ep.append(boot)
            centred_means_by_ep.append(centred.mean(axis=-1))
        seed_effect = np.mean(paired_by_ep, axis=0)
        bootstrap_effect = np.mean(boot_by_ep, axis=0)
        compare(f"{backbone} panel C means", seed_effect.mean(axis=-1), np.mean(centred_means_by_ep, axis=0))
        for cell in CONDITIONS:
            index = score.CELLS.index(cell)
            low, high = score.prior.interval(bootstrap_effect[index], CI_LEVEL)
            row = dict(backbone=backbone, metric=AVG_METRIC, contrast=CONTRASTS[cell])
            summary.append(dict(row, mean_pct=seed_effect[index].mean(), low95_pct=low, high95_pct=high,
                                n_seeds=len(score.SEEDS), seed_sd_pct=seed_effect[index].std(ddof=1)))
            seeds.extend(dict(row, seed=seed, value_pct=value)
                         for seed, value in zip(score.SEEDS, seed_effect[index], strict=True))
    return pd.DataFrame(summary), pd.DataFrame(seeds)


def run(export: Path, scores: Path, release: Path, triads: Path, output: Path) -> None:
    manifest, cohort, predictions = load_predictions(export, scores, release, triads)
    with threadpool_limits(limits=2, user_api="blas"):
        summary, seeds = aggregate(scores, predictions, cohort)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / SUMMARY_FILE, index=False)
    seeds.to_csv(output / SEEDS_FILE, index=False)
    record = {
        "campaign_id": manifest["campaign_id"],
        "score_manifest_sha256": score.sha256(scores / "manifest.json"),
        "export_manifest_sha256": score.sha256(export / "manifest.json"),
        "source_triads_sha256": score.sha256(triads),
        "release_fingerprint": manifest["release_fingerprint"],
        "method": "Average the seven paired endpoint percent MAE changes against structure only with equal weights.",
        "normalization": "Fixed nine-seed structure-only mean MAE for each endpoint and backbone.",
        "bootstrap": "Within each crossed group-and-seed draw, recompute each endpoint MAE on its own observed-label "
                     "mask, difference the matched conditions, normalize and average, then take percentiles across draws.",
        "bootstrap_seed": score.BOOTSTRAP_SEED,
        "bootstrap_draws": score.N_BOOTSTRAP,
        "ci_level": CI_LEVEL,
        "label_counts": {metric: count for metric, (_, count) in ENDPOINTS.items()},
        "verification_absolute_tolerance": ATOL,
        "source_sha256": {
            "average_endpoints.py": score.sha256(Path(__file__)),
            "analysis/combined_score.py": score.sha256(Path(score.__file__)),
            "analysis/vendor_ionisation/ionisation_score.py": score.sha256(Path(score.prior.__file__)),
            "../combined_main_v1/all_endpoints.py": score.sha256(HERE.parent / "combined_main_v1" / "all_endpoints.py"),
        },
        "files": {name: score.sha256(output / name) for name in (SUMMARY_FILE, SEEDS_FILE)},
    }
    temporary = output / f"{MANIFEST_FILE}.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / MANIFEST_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, default=score.ROOT / "outputs" / "export")
    parser.add_argument("--scores", type=Path, default=score.ROOT / "outputs" / "scores")
    parser.add_argument("--release", type=Path, default=score.ROOT / "data" / "release" / "numerical")
    parser.add_argument("--triads", type=Path, default=score.ROOT / "runtime" / "same_row_complete_triads.csv")
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    run(args.export, args.scores, args.release, args.triads, args.output)
