"""Score endpoint-normalized triad errors for the joint outcome display."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from all_endpoints import ATOL, compare, load_predictions, score

HERE = Path(__file__).resolve().parent
TRIAD_ENDPOINTS = score.ENDPOINTS[:3]
MAE_COLUMNS = [f"{endpoint}_mae" for endpoint in TRIAD_ENDPOINTS]


def run(export: Path, scores: Path, release: Path, triads: Path, output: Path) -> None:
    scored, cohort, predictions = load_predictions(export, scores, release, triads)
    n_triads = int(cohort.triads.sum())
    if n_triads != 242:
        raise ValueError("The joint triad display requires 242 frozen same-row triads")
    seed_table = score.read_csv(scores / "seed_metrics.csv")
    archived = seed_table.loc[seed_table["metric"].eq(score.TRIAD) & seed_table["cell"].isin(score.CELLS)]
    keys = ["backbone", "cell", "seed"]
    expected = pd.MultiIndex.from_product((score.BACKBONES, score.CELLS, score.SEEDS), names=keys)
    archived = archived.set_index(keys, verify_integrity=True)
    if set(archived.index) != set(expected) or not archived["n_structures"].eq(n_triads).all():
        raise ValueError("The archived triad seed table does not cover all models, conditions and seeds")

    results, denominators, residual_summaries = [], {}, []
    truth = cohort.truth[cohort.triads, :len(TRIAD_ENDPOINTS)]
    for backbone, q in predictions.items():
        endpoint_mae = np.abs(q[..., cohort.triads, :len(TRIAD_ENDPOINTS)] - truth).mean(axis=-2)
        recorded = archived.xs(backbone, level="backbone")["value"]
        order = pd.MultiIndex.from_product((score.CELLS, score.SEEDS), names=["cell", "seed"])
        compare(f"{backbone}/triad raw macro MAE", endpoint_mae.mean(axis=-1).ravel(), recorded.reindex(order))
        baseline = endpoint_mae[score.CELLS.index("00")].mean(axis=0)
        if not np.isfinite(baseline).all() or np.any(baseline <= 0):
            raise ValueError(f"Invalid structure-only denominators for {backbone}")
        denominators[backbone] = dict(zip(TRIAD_ENDPOINTS, baseline.tolist(), strict=True))
        change_pct = 100 * ((endpoint_mae / baseline).mean(axis=-1) - 1)
        residual = np.abs(score.residual_log10(q[..., :len(TRIAD_ENDPOINTS)]))
        matched_residual = residual[..., cohort.triads].mean(axis=-1)
        full_residual = residual.mean(axis=-1)
        archived_residual = seed_table.loc[seed_table["backbone"].eq(backbone)
                                          & seed_table["metric"].eq(score.RESIDUAL)
                                          & seed_table["cell"].isin(score.CELLS)].set_index(["cell", "seed"])
        compare(f"{backbone}/full-cohort residual", full_residual.ravel(),
                archived_residual["value"].reindex(order))
        for n_structures, per_seed in ((327, full_residual), (242, matched_residual)):
            cell_means = per_seed.mean(axis=1)
            ion = score.CELLS.index("10")
            both = score.CELLS.index("11")
            changes = 100 * (per_seed[both] - per_seed[ion]) / cell_means[ion]
            residual_summaries.append(dict(
                backbone=backbone, n_structures=n_structures,
                **{f"mean_{cell}": float(value) for cell, value in zip(score.CELLS, cell_means, strict=True)},
                both_minus_ion_pct=float(changes.mean()), seed_sd_pct=float(changes.std(ddof=1)),
                n_seeds=len(score.SEEDS),
            ))
        data = pd.DataFrame(endpoint_mae.reshape(len(order), len(TRIAD_ENDPOINTS)),
                            index=order, columns=MAE_COLUMNS).reset_index()
        data["backbone"] = backbone
        data["triad_error_change_pct"] = change_pct.ravel()
        data["mean_abs_residual_log10"] = matched_residual.ravel()
        data["all_structures_mean_abs_residual_log10"] = full_residual.ravel()
        results.append(data[[*keys, "triad_error_change_pct", *MAE_COLUMNS,
                             "mean_abs_residual_log10", "all_structures_mean_abs_residual_log10"]])

    table = pd.concat(results, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "triad_display_seed_metrics.csv"
    table.to_csv(path, index=False)
    summary_path = output / "residual_cohort_summary.csv"
    pd.DataFrame(residual_summaries).to_csv(summary_path, index=False)
    record = {
        "campaign_id": scored["campaign_id"],
        "score_manifest_sha256": score.sha256(scores / "manifest.json"),
        "method": "Equal-weight mean of three normalized endpoint MAEs on the frozen same-row triads.",
        "formula": "100 * (mean_e[MAE(backbone, cell, seed, endpoint) / "
                   "mean_s(MAE(backbone, 00, s, endpoint))] - 1)",
        "n_triads": n_triads,
        "residual_population": "mean_abs_residual_log10 uses the same 242 complete triads as accuracy; "
                               "all_structures_mean_abs_residual_log10 retains the 327-structure reference.",
        "cohort_sensitivity": "Both-minus-ionisation residual changes use each cohort's fixed ionisation-only "
                              "mean across nine seeds as denominator. Seed SD is descriptive; no new intervals.",
        "n_groups": int(len(np.unique(cohort.group[cohort.triads]))),
        "seeds": score.SEEDS,
        "raw_endpoint_mae_unit": "log10",
        "endpoint_baseline_00_mae": denominators,
        "verification_absolute_tolerance": ATOL,
        "source_sha256": {
            "triad_display.py": score.sha256(Path(__file__)),
            "all_endpoints.py": score.sha256(HERE / "all_endpoints.py"),
            "analysis/combined_score.py": score.sha256(Path(score.__file__)),
            "outputs/scores/seed_metrics.csv": score.sha256(scores / "seed_metrics.csv"),
            "outputs/export/manifest.json": score.sha256(export / "manifest.json"),
            "outputs/export/final_predictions.csv": score.sha256(export / "final_predictions.csv"),
            "data/release/numerical/manifest.json": score.sha256(release / "manifest.json"),
            "runtime/same_row_complete_triads.csv": score.sha256(triads),
        },
        "files": {p.name: score.sha256(p) for p in (path, summary_path)},
    }
    destination = output / "triad_display_manifest.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, default=score.ROOT / "outputs" / "export")
    parser.add_argument("--scores", type=Path, default=score.ROOT / "outputs" / "scores")
    parser.add_argument("--release", type=Path, default=score.ROOT / "data" / "release" / "numerical")
    parser.add_argument("--triads", type=Path, default=score.ROOT / "runtime" / "same_row_complete_triads.csv")
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    run(args.export, args.scores, args.release, args.triads, args.output)
