import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ionisation_controls import (
    ALPHAS, build_controls, descriptor_robustness, fit_ridge_control, inner_folds, median_predictions, numerical_features,
    ridge_grid, standardise_profile,
)
from ionisation_score import (
    ENDPOINTS, PRED_COLUMNS, RAW_ENDPOINTS, inverse_physical, validate_predictions,
)


def fixture_data():
    n = 20
    index = pd.Index([f"s{i:02}" for i in range(n)], name="structure_id", dtype="string")
    data = pd.DataFrame(np.repeat(np.arange(1, n + 1)[:, None], 7, axis=1), columns=RAW_ENDPOINTS, index=index)
    data["group_id"] = pd.array([f"g{i // 2}" for i in range(n)], dtype="string")
    data["outer_fold"] = [i // 4 + 1 for i in range(n)]
    data["ionisation_class"] = np.resize(["anionic", "neutral", "cationic", "zwitterionic"], n)
    data["pka_acid"] = np.linspace(1, 11, n)
    data["pka_base"] = np.linspace(5, 9, n)
    return data


def fixture_inner(data):
    frames = []
    for outer_fold in sorted(data["outer_fold"].unique()):
        train = data[data["outer_fold"].ne(outer_fold)]
        inner = train[["group_id"]].copy()
        inner["inner_fold"] = [int(group[1:]) % 2 + 1 for group in inner["group_id"]]
        inner["outer_fold"] = outer_fold
        frames.append(inner.reset_index())
    return pd.concat(frames, ignore_index=True)


def test_class_median_uses_training_raw_values_and_endpoint_fallback():
    raw = np.repeat([[1.], [3.], [100.], [4.]], 7, axis=1)
    raw[:2, 1] = np.nan
    raw[2, 4] = 0
    overall, conditional, fallback = median_predictions(
        raw, np.array(["anionic", "anionic", "cationic", "cationic"]), np.array(["anionic", "neutral", "cationic"])
    )
    assert overall[0, 0] == 3.5
    assert conditional[0, 0] == 2
    assert conditional[0, 1] == 52
    assert fallback["anionic"] == ["Vd"]
    np.testing.assert_array_equal(conditional[1], overall[1])
    assert conditional[2, 4] == 2
    assert fallback["neutral"] == list(ENDPOINTS)


def test_inner_folds_exclude_outer_test_and_keep_groups_intact(tmp_path):
    data = fixture_data()
    inner = fixture_inner(data)
    path = tmp_path / "inner.csv"
    inner.to_csv(path, index=False)
    validated = inner_folds(path, data, diagnostic=True)
    assert len(validated) == 5 and len(validated[1]) == 16
    outer_test_row = inner.iloc[[0]].assign(structure_id=data.index[0], group_id=data["group_id"].iloc[0])
    pd.concat([inner, outer_test_row], ignore_index=True).to_csv(path, index=False)
    with pytest.raises(ValueError, match="exactly its outer-training IDs"):
        inner_folds(path, data, diagnostic=True)
    inner.loc[0, "inner_fold"] = 3
    inner.to_csv(path, index=False)
    with pytest.raises(ValueError, match="splits a group"):
        inner_folds(path, data, diagnostic=True)


def test_profile_channels_and_training_only_scaler():
    data = pd.DataFrame({"pka_acid": [2., 3.], "pka_base": [7.4, 7.4]})
    features = numerical_features(data)
    assert features.shape == (2, 8)
    assert features[0, 0] == 0.5
    assert features[0, 7] == 0.5
    train, held, audit = standardise_profile(np.array([[0., 1.], [2., 1.]]), np.array([[5., 100.]]))
    np.testing.assert_allclose(train, [[-0.5, 0], [0.5, 0]])
    np.testing.assert_allclose(held, [[2, 0]])
    assert audit["feature_mean"] == [1., 1.]
    assert audit["constant_features"] == [False, True]


def test_stress_shift_crosses_only_the_changed_threshold_and_sweep_is_sorted():
    data = fixture_data().iloc[:1].copy()
    data["pka_acid"], data["pka_base"] = 7.5, 7.0
    stress, sweep = descriptor_robustness(data)
    overall = stress[stress.audited_class.eq("all")]
    assert len(overall) == 24
    acid_only = overall[overall.acid_pka_shift.eq(-0.5) & overall.base_pka_shift.eq(0)].iloc[0]
    assert acid_only.threshold_transition_counts == '{"neutral->anionic": 1}'
    assert acid_only["mean_abs_delta_acid_pH_7.4"] > 0
    assert acid_only["mean_abs_delta_base_pH_7.4"] == 0
    both = overall[overall.acid_pka_shift.eq(-0.5) & overall.base_pka_shift.eq(0.5)].iloc[0]
    assert both.threshold_transition_counts == '{"neutral->zwitterionic": 1}'
    assert sweep.pH.to_numpy() == pytest.approx(np.linspace(2, 8, 25))
    assert sweep.acid_propensity.is_monotonic_increasing
    assert sweep.base_propensity.is_monotonic_decreasing


def test_ridge_with_intercept_and_label_mask_matches_normal_equation():
    train = np.arange(8.).reshape(4, 2)
    test = np.array([[3., 4.], [1., 2.]])
    target = np.repeat(np.array([[1.], [np.nan], [3.], [9.]]), len(ENDPOINTS), axis=1)
    grid = ridge_grid(train, test, target, ALPHAS)
    mask = np.isfinite(target[:, 0])
    x = np.column_stack((np.ones(mask.sum()), train[mask]))
    z = np.column_stack((np.ones(len(test)), test))
    for i, alpha in enumerate(ALPHAS):
        beta = np.linalg.solve(x.T @ x + alpha * np.diag([0., 1., 1.]), x.T @ target[mask])
        np.testing.assert_allclose(grid[i], z @ beta)


def test_ridge_selection_is_independent_of_outer_test_features():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(12, 8))
    y = rng.normal(size=(12, 7))
    y[0, 4] = np.nan
    inner = np.tile([1, 2, 3], 4)
    a = fit_ridge_control(x, np.zeros((2, 8)), y, inner, scale_profile=True, diagnostic=True)
    b = fit_ridge_control(x, np.full((2, 8), 999), y, inner, scale_profile=True, diagnostic=True)
    assert a[1:] == b[1:]
    assert not np.allclose(a[0], b[0])
    first_inner = a[1][0]
    np.testing.assert_allclose(first_inner["feature_mean"], x[inner != 1].mean(axis=0))
    assert sum(row.get("selected", False) for row in a[1]) == 1


def test_outer_heldout_label_cannot_change_controls_for_its_fold(tmp_path):
    data = fixture_data()
    path = tmp_path / "inner.csv"
    fixture_inner(data).to_csv(path, index=False)
    splits = {"numerical": inner_folds(path, data, diagnostic=True)}
    cohorts = {"numerical": data}
    predictions, audit = build_controls(cohorts, splits, diagnostic=True)
    assert len(validate_predictions(predictions, cohorts, control=True)) == 3
    assert len(predictions) == len(data) * 3
    altered = data.copy()
    altered.loc[altered.outer_fold.eq(1), RAW_ENDPOINTS[1]] *= 100
    predictions2, audit2 = build_controls({"numerical": altered}, splits, diagnostic=True)
    np.testing.assert_allclose(predictions.loc[predictions.outer_fold.eq(1), list(PRED_COLUMNS)],
                               predictions2.loc[predictions2.outer_fold.eq(1), list(PRED_COLUMNS)])
    pd.testing.assert_frame_equal(audit[audit.outer_fold.eq(1)], audit2[audit2.outer_fold.eq(1)])
    assert np.isfinite(inverse_physical(predictions[list(PRED_COLUMNS)].to_numpy())).all()
