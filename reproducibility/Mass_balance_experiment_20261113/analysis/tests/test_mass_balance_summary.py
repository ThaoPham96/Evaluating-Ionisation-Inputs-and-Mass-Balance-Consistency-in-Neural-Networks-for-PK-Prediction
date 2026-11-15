"""Check frozen outer coverage, paired scores and result verification without training."""

from pathlib import Path

import numpy as np
import pytest

import summarize_mass_balance_campaign as summary
from mass_balance_campaign import digest, sha256, write_csv, write_json
from mass_balance_controls import LOG10_LN2


RELEASE = Path(__file__).resolve().parents[2] / "data"


@pytest.fixture(scope="module")
def cohort():
    return summary.Cohort(RELEASE)


def test_exact_coherence_and_invalid_failures():
    predictions = np.array([[LOG10_LN2, 0, 0], [LOG10_LN2 + 0.2, 0, 0], [np.nan, 0, 0]])
    measured = summary.coherence_metrics(predictions)
    assert measured["median_abs_log10_residual"] == pytest.approx(0.1)
    assert measured["median_disagreement_fold"] == pytest.approx((1 + 10**0.2) / 2)
    assert measured["within_1_25"] == pytest.approx(1 / 3)
    assert measured["within_2"] == pytest.approx(2 / 3)
    assert measured["n_invalid"] == 1


def test_ppb_zero_is_retained_and_pairing_is_explicit(cohort):
    baseline = np.zeros_like(cohort.endpoint_truth)
    intervention = np.zeros_like(baseline)
    baseline[0, 0] = np.nan
    ppb = summary.seed_metrics(cohort, intervention, baseline)["PPB_percentage_point"]
    observed = cohort.endpoint_truth[:, 4]
    assert ppb["n_eligible"] == ppb["n_scored"] == 805
    assert ppb["mae"] == pytest.approx(np.abs(50 - observed[np.isfinite(observed)]).mean())
    assert cohort.ppb_zero_count == 24
    paired = summary.triad_metrics(np.array([[1, 0, 0], [np.nan, 0, 0]]),
                                   np.array([[2, 0, 0], [9, 0, 0]]), np.zeros((2, 3)), np.ones(2, bool))
    assert paired["mae"] == pytest.approx(1 / 3)
    assert paired["baseline_common_mae"] == pytest.approx(2 / 3)
    assert paired["n_scored"] == paired["n_invalid"] == 1


def campaign(cohort, path):
    identity = {"dataset_fingerprint": summary.FINGERPRINT,
                "release": {name: cohort.hashes[name] for name in summary.RELEASE_NAMES}}
    write_json(path / "campaign.json", {"protocol": summary.PROTOCOL, "logical_fit_count": 4240,
                                       "identity": identity, "campaign_id": digest(identity)})
    return summary.Predictions(path, cohort)


def write_selection(source, fold, stage, **values):
    name = f"GraphConv-outer{fold}-{stage}"
    path = source.path / "selections" / name
    path.mkdir(parents=True)
    if stage == "baseline":
        values["candidate_scores"] = {f"GraphConv_{candidate:02d}": candidate / 10 for candidate in range(1, 21)}
    else:
        values["lambda_scores"] = {str(value): {"macro_log10_mae": 0.1 if value == 0 else 1 + abs(value - 0.1)}
                                   for value in summary.LAMBDAS}
    write_json(path / "selection.json", {"name": name, "kind": stage, "campaign_id": source.campaign_id,
                                        "candidate_id": "GraphConv_01", **values})


def write_fit(source, fold, seed, logical_id):
    directory = source.path / "results" / logical_id
    directory.mkdir(parents=True)
    expected = source.cohort.ids_by_fold[fold]
    ids = sorted(expected)
    request = {
        "campaign_id": source.campaign_id, "logical_id": logical_id, "backbone": "GraphConv", "outer_fold": fold,
        "block": "primary_final", "endpoint": None, "inner_fold": None, "seed": seed, "lambda_value": 0.0,
        "candidate_id": "GraphConv_01", "fixed_epochs": 23, "train_ids": sorted(source.cohort.index.keys() - expected),
        "val_ids": [], "predict_ids": ids,
    }
    request["fit_id"] = f"{logical_id}-{digest(request)[:20]}"
    write_json(directory / "request.json", {"request": request, "request_hash": digest(request),
               "resolved_hash": "test-resolved", "effective_ids": {name: request[name] for name in
                                                                 ("train_ids", "val_ids", "predict_ids")}})
    write_csv(directory / "predictions.csv", ("structure_id", *summary.PRED_COLUMNS),
              ((sid, LOG10_LN2 + 0.3, 0, 0, 0, 0, 0, 0) for sid in reversed(ids)))
    write_json(directory / "complete.json", {"fit_id": request["fit_id"], "resolved_hash": "test-resolved",
                "files": {name: sha256(directory / name) for name in ("request.json", "predictions.csv")}})
    write_json(directory / "published.json", {"fit_id": request["fit_id"], "campaign_id": source.campaign_id,
               "trainer_marker_sha256": sha256(directory / "complete.json"),
               "sizes": {"predictions.csv": (directory / "predictions.csv").stat().st_size}})
    return directory


def test_fold_specific_controls_and_incomplete_coverage(cohort, tmp_path):
    source = campaign(cohort, tmp_path)
    seed = summary.SEEDS[0]
    for fold in summary.FOLDS:
        logical_id = f"MB5-{fold:05d}"
        alpha = float(fold % 2)
        write_selection(source, fold, "baseline", alpha_controls={method: {
            "selected_alpha": alpha,
            "scores": {str(value): {"macro_log10_mae": abs(value - alpha)} for value in summary.ALPHA_GRID},
        } for method in summary.METHODS})
        write_selection(source, fold, "lambda", baseline_selection=f"GraphConv-outer{fold}-baseline",
                        selected_lambda=0.1, selected_zero_eligible_lambda=0.0,
                        all_positives_worse_than_zero=True,
                        aliases={"baseline": {str(seed): logical_id}}, epochs={"0.0": 23})
        write_fit(source, fold, seed, logical_id)
    arrays, missing = source.pooled("GraphConv", "selected_alpha_half_life", (seed,))
    assert not missing
    predictions = arrays[seed]
    assert np.allclose(predictions[cohort.folds % 2 == 1, 0], LOG10_LN2)
    assert np.allclose(predictions[cohort.folds % 2 == 0, 0], LOG10_LN2 + 0.3)
    assert np.all(predictions[:, 1:] == 0)

    arrays, missing = source.pooled("GraphConv", "baseline", summary.SEEDS)
    assert not arrays
    assert len(missing) == 5 * 8
    report = summary.summarize(source, cohort, include_curves=False)
    assert not report["metrics"]
    assert not any(row["available"] for row in report["coverage"])


def test_checksum_and_outer_partition_rejection(cohort, tmp_path):
    source = campaign(cohort, tmp_path)
    seed, logical_id = summary.SEEDS[0], "MB5-00001"
    directory = write_fit(source, 1, seed, logical_id)
    with pytest.raises(ValueError, match="Outer fit assignment differs"):
        source.fit(logical_id, "GraphConv", 2, seed, 0.0, "GraphConv_01", 23)
    with (directory / "predictions.csv").open("a", encoding="utf-8") as handle:
        handle.write("corruption")
    with pytest.raises(ValueError, match="Prediction checksum or size mismatch"):
        source.fit(logical_id, "GraphConv", 1, seed, 0.0, "GraphConv_01", 23)


def test_compute_seed_metrics_before_averaging(cohort):
    shape = cohort.endpoint_truth.shape
    baseline = np.zeros(shape)
    seeds = (101, 202)
    predictions = np.array([np.full(shape, -1), np.full(shape, 1)])
    rows = [{"backbone": "GraphConv", "policy": "selected_positive", "seed": seed,
             "domain": "same_row_triad_log10", "metric": "mae", "value":
             summary.seed_metrics(cohort, values, baseline)["same_row_triad_log10"]["mae"]}
            for seed, values in zip(seeds, predictions)]
    aggregate = summary.summarize_seed_rows(rows, {"selected_positive": seeds})[0]
    pooled_mae = summary.seed_metrics(cohort, predictions.mean(axis=0), baseline)["same_row_triad_log10"]["mae"]
    assert aggregate["mean"] == pytest.approx(sum(row["value"] for row in rows) / 2)
    assert aggregate["mean"] > pooled_mae
    assert aggregate["n_seeds_defined"] == 2
