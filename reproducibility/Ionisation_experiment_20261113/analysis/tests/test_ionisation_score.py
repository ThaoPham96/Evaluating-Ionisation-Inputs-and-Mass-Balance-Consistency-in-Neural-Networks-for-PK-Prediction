import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ionisation_score import (
    ADJUSTED_LEVEL, BootstrapDraws, ENDPOINTS, PHYSICAL_COLUMNS, PRED_COLUMNS, RAW_ENDPOINTS,
    SEEDS, Predictions, coherence_delta, contrast_statistics, endpoint_metric, errors,
    evidence_strata, expected_runs, inverse_physical, load_cohort, paired_bootstrap, score,
    transform_targets, validate_predictions, vd_correlations,
)


def make_cohort(n=10):
    data = pd.DataFrame(np.repeat(np.arange(1, n + 1)[:, None], 7, axis=1), columns=RAW_ENDPOINTS)
    data[RAW_ENDPOINTS[4]] = [0, *range(2, n + 1)]
    data["group_id"] = pd.array([f"group_{i // 2}" for i in range(n)], dtype="string")
    data["outer_fold"] = [(i // 2) % 5 + 1 for i in range(n)]
    data["ionisation_class"] = "neutral"
    data.index = pd.Index([f"s{i:03}" for i in range(n)], name="structure_id", dtype="string")
    return data


def make_predictions(data, runs, seeds):
    truth = transform_targets(data[list(RAW_ENDPOINTS)].to_numpy())
    frames = []
    for cohort, procedure, backbone, arm in sorted(runs):
        for seed in seeds:
            pred = truth.copy()
            pred[:, 1] += {"C0": 0.5, "C2": 0.1}.get(arm, 0.2)
            both = pd.DataFrame(np.column_stack((pred, inverse_physical(pred))), columns=(*PRED_COLUMNS, *PHYSICAL_COLUMNS))
            frames.append(both.assign(cohort=cohort, procedure=procedure, backbone=backbone, arm=arm,
                                     seed=seed, structure_id=data.index, outer_fold=data["outer_fold"].to_numpy()))
    return pd.concat(frames, ignore_index=True)


def test_endpoint_metrics_include_ppb_zero_once_and_aafe_excludes_it():
    raw = np.array([[2, 10, 1, 0, 0, 2, 1], [1, 100, 1, 50, 20, 1, 2], [2, 10, 1, 0, np.nan, 2, 1]])
    physical = np.array([[[4, 100, 1, 0, 5, 2, 1], [1, 10, 1, 50, 10, 1, 2], [2, 10, 1, 0, 10, 2, 1]]])
    pred = Predictions((1,), transform_targets(physical[0])[None], physical)
    error = errors(pred, raw, transform_targets(raw))
    ppb = endpoint_metric(pred, raw, error, 0, 4)
    assert ppb == pytest.approx(dict(n_structures=2, n_label_zeros=1, n_aafe=1, n_aafe_zero_predictions=0, mae=7.5, aafe=2))
    vd = endpoint_metric(pred, raw, error, 0, 1)
    assert vd["mae"] == pytest.approx(2 / 3)
    assert vd["aafe"] == pytest.approx(10 ** (2 / 3))


def test_crossed_bootstrap_reuses_whole_seed_and_group_counts():
    delta = np.array([[1., 3., 4.], [-2., -2., -4.]])
    membership = np.array([0, 0, 1])
    draws = BootstrapDraws(np.array([[2, 0], [0, 2], [1, 1]]), np.array([[2, 0], [0, 2], [1, 1]]))
    crossed, group_only = paired_bootstrap(delta, membership, draws)
    np.testing.assert_allclose(crossed, [2, -4, 0])
    np.testing.assert_allclose(group_only, [0, 0, 0])
    rare_crossed, _ = paired_bootstrap(delta[:, :2], membership[:2], draws)
    np.testing.assert_allclose(rare_crossed, [2, np.nan, 0], equal_nan=True)


def test_contrast_direction_adjusted_interval_and_denominator():
    arm = np.array([[1, 3, 1], [1, 4, 2]], dtype=float)
    base = arm + np.array([[2], [1]])
    draws = BootstrapDraws(np.array([[1, 1], [2, 0], [0, 2]]), np.array([[1, 1], [2, 0], [0, 2]]))
    result, deltas = contrast_statistics(arm, base, np.array([0, 0, 1]), draws, primary=True)
    np.testing.assert_allclose(deltas, [-2, -1])
    assert result["delta_mean"] == -1.5
    assert result["relative_delta_pct"] == pytest.approx(-150 / base.mean())
    assert result["seed_sd"] == pytest.approx(np.sqrt(0.5))
    assert result["seeds_below_zero"] == 2
    assert result["primary_lower_error"]
    assert result["crossed_adjusted_high"] == pytest.approx(np.quantile([-2, -1.5, -1], (1 + ADJUSTED_LEVEL) / 2))


def test_correlation_gates_and_class_median_exclusion():
    values = np.arange(10.)
    assert vd_correlations(values, values, 5, subgroup=True, class_median=False) == pytest.approx((1, 1))
    assert np.isnan(vd_correlations(values, values, 4, subgroup=True, class_median=False)).all()
    assert np.isnan(vd_correlations(values[:9], values[:9], 5, subgroup=True, class_median=False)).all()
    assert np.isnan(vd_correlations(values, np.zeros(10), 5, subgroup=True, class_median=False)).all()
    assert np.isnan(vd_correlations(values, values, 5, subgroup=True, class_median=True)[0])


def test_reject_incomplete_duplicate_nonfinite_wrong_fold_and_scale():
    data = make_cohort()
    runs = {("class", "primary", "GraphConv", arm) for arm in ("C0", "C2")}
    frame = make_predictions(data, runs, (101, 202))
    assert len(validate_predictions(frame, {"class": data}, diagnostic=True)) == 2
    for broken, message in (
        (frame.iloc[:-1], "every cohort structure"),
        (pd.concat([frame, frame.iloc[[0]]]), "Duplicate"),
        (frame.assign(pred_Vd=np.inf), "nonfinite"),
        (frame.assign(outer_fold=99), "frozen held-out"),
        (frame.assign(physical_Vd=1.0), "predictions disagree"),
        (frame[~(frame.arm.eq("C2") & frame.seed.eq(202))], "different seed trajectories"),
        (frame.assign(cohort="numerical"), "run mismatch"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_predictions(broken, {"class": data}, diagnostic=True)
    with pytest.raises(ValueError, match="Missing"):
        validate_predictions(frame, {"class": data})


def test_production_requires_all_nine_exact_seed_ids():
    data = make_cohort()
    runs = expected_runs(("class",), control=False)
    frame = make_predictions(data, runs, SEEDS)
    assert len(validate_predictions(frame, {"class": data})) == len(runs)
    frame.loc[frame.seed == 909, "seed"] = 999
    with pytest.raises(ValueError, match="exactly"):
        validate_predictions(frame, {"class": data})


def test_evidence_axes_not_inferred_from_missing_data_and_scored_separately():
    data = make_cohort()
    data["class_support"] = ["complete_descriptors"] * 5 + ["missing_side"] * 5
    data["pka_acid_kind"] = ["experimental"] * 5 + [None] * 5
    data["pka_base_kind"] = ["predicted"] * 10
    strata = evidence_strata(data)
    assert strata["complete_descriptors/mixed_experimental_predicted"].sum() == 5
    assert strata["missing_side/predicted_only"].sum() == 5
    runs = {("class", "primary", "GraphConv", arm) for arm in ("C0", "C2")}
    frame = make_predictions(data, runs, (101, 202))
    verified = validate_predictions(frame, {"class": data}, diagnostic=True)
    tables = score({"class": data}, verified, n_draws=100, confirmatory=False)
    contrasts = tables["contrasts"]
    row = contrasts[contrasts.primary_test].iloc[0]
    assert row["delta_mean"] == pytest.approx(-0.4)
    assert row["primary_lower_error"] is None
    assert len(contrasts[contrasts.scope == "evidence"]) == len(strata)
    assert len(tables["endpoint_summary"]) == 2 * len(ENDPOINTS)
    assert len(tables["contrast_outer_folds"]) == 5 * len(ENDPOINTS)


def test_coherence_decomposition_keeps_vd_sign():
    base = Predictions((1,), np.zeros((1, 2, 7)), np.ones((1, 2, 7)))
    arm_log = np.zeros_like(base.transformed)
    arm_log[..., :3] = (1, 2, 3)
    arm = Predictions((1,), arm_log, inverse_physical(arm_log))
    result = coherence_delta(arm, base, np.array([True, True]))
    assert result["minus_delta_log_Vd"][0] == -2
    assert result["delta_signed_log_residual"][0] == 2
    assert result["delta_median_fold_disagreement"][0] > 100


def test_single_diagnostic_neural_seed_survives_a_cpu_control_contrast():
    data = make_cohort()
    neural = make_predictions(data, {("class", "primary", "GraphConv", "C2")}, (101,))
    control = make_predictions(data, {("class", "control", "CPU", "class_median")}, (0,))
    control.loc[:, list(PHYSICAL_COLUMNS)] = data[list(RAW_ENDPOINTS)].to_numpy()
    control.loc[:, list(PRED_COLUMNS)] = transform_targets(data[list(RAW_ENDPOINTS)].to_numpy())
    cohorts = {"class": data}
    runs = validate_predictions(neural, cohorts, diagnostic=True)
    runs.update(validate_predictions(control, cohorts, diagnostic=True, control=True))
    tables = score(cohorts, runs, n_draws=10, confirmatory=False)
    assert set(tables["contrast_seeds"]["seed"]) == {101}


def test_frozen_group_cannot_cross_outer_folds(tmp_path):
    folder = tmp_path / "class"
    folder.mkdir()
    data = make_cohort()
    data.reset_index().to_csv(folder / "model_dataset.csv", index=False)
    outer = data[["group_id", "outer_fold"]].copy()
    outer.iloc[0, 1] = 4
    outer.reset_index().to_csv(folder / "outer.csv", index=False)
    with pytest.raises(ValueError, match="one outer fold"):
        load_cohort(tmp_path, "class")
