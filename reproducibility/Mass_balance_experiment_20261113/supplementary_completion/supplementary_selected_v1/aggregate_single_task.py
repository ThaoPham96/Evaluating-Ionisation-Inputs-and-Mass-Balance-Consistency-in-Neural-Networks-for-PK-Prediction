"""Calculate the equal-weight endpoint average for the two SUP06 contrasts."""

from hashlib import sha256
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "code"))
import mbsupp as mb

METRIC = "macro_seven_endpoint"
CONTRASTS = {
    "single_task - multitask_baseline": "baseline",
    "single_task - multitask_loss": "selected_positive",
}
DENOMINATOR = (
    "For each endpoint, 100 × mean across paired seeds of (single-task MAE − comparator MAE) "
    "/ mean comparator MAE across those seeds. Recompute numerator and denominator within every draw."
)


def endpoint_scores(boot, cohort, predictions):
    """Compute endpoint error from the original endpoint-specific labelled cohorts."""
    assert predictions.shape == (len(mb.SEEDS3), *cohort.endpoint_truth.shape)
    assert np.isfinite(predictions).all()
    scores = {}
    for column, endpoint in enumerate(mb.NAMES):
        truth = cohort.endpoint_truth[:, column]
        unit = "percentage_point" if column in mb.PERCENT_COLUMNS else "log10"
        metric = f"{endpoint}_mae_relative_{unit}"
        scores[metric] = boot.mean(np.abs(predictions[:, :, column] - truth), np.isfinite(truth))
    return scores


def aggregate(frozen):
    source = mb.Source()
    draws_path = mb.ROOT / "results" / "bootstrap" / "group_draw_counts.npz"
    boot = mb.Bootstrap(source.cohort, saved=draws_path)
    single = endpoint_scores(boot, source.cohort, mb.units(source.single_task(mb.SEEDS3), mb.SEEDS3))
    rows, checked = [], 0
    n_labelled = {}
    for contrast, policy in CONTRASTS.items():
        raw = source.primary("GraphConv", policy, mb.SEEDS3)
        comparator = endpoint_scores(boot, source.cohort, mb.units(raw, mb.SEEDS3))
        points, draws = [], []
        for metric, (mae_a, draw_a, n_a) in single.items():
            mae_b, draw_b, n_b = comparator[metric]
            np.testing.assert_array_equal(n_a, n_b)
            n_labelled[metric] = int(n_a.min())
            point = 100 * (mae_a - mae_b) / mae_b.mean()
            draw = 100 * (draw_a - draw_b) / draw_b.mean(axis=1, keepdims=True)
            assert np.isfinite(draw).all()
            summary = boot.interval(point, draw)
            reference = frozen.loc[frozen.contrast.eq(contrast) & frozen.metric.eq(metric)]
            assert len(reference) == 1 and reference.n_scored.iat[0] == n_labelled[metric]
            actual = [*summary.values(), mae_a.mean(), mae_b.mean()]
            expected = reference[[*summary, "mean_a", "mean_b"]].iloc[0].to_numpy(dtype=float)
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12, err_msg=f"{contrast}: {metric}")
            checked += 1
            points.append(point)
            draws.append(draw)

        # Average the endpoint ratios within each seed and shared draw before taking percentiles.
        summary = boot.interval(np.mean(points, axis=0), np.mean(draws, axis=0))
        assert summary["n_defined_draws"] == len(boot.draws) == 10_000
        rows.append({"contrast": contrast, "metric": METRIC, "unit": "mean of endpoint MAE percentage changes",
                     **summary, "n_endpoints": len(points)})

    records = {
        "estimand": "Equal arithmetic mean of the seven endpoint MAE percentage changes, excluding the triad row.",
        "denominator": DENOMINATOR,
        "interval": "2.5th and 97.5th percentiles of paired group-draw means, conditional on fitted models.",
        "draws_file": draws_path.relative_to(mb.ROOT).as_posix(),
        "draws_sha256": sha256(draws_path.read_bytes()).hexdigest(),
        "group_count": boot.n_groups,
        "draw_count": len(boot.draws),
        "seeds": list(mb.SEEDS3),
        "labelled_structures_by_endpoint_metric": n_labelled,
        "frozen_endpoint_contrasts_reproduced": checked,
        "verified_campaign_inputs_sha256": {
            (source.path / path).relative_to(mb.ROOT).as_posix(): value
            for path, value in sorted(source.source_hashes.items())
        },
    }
    return pd.DataFrame(rows), records
