import numpy as np

from bootstrap_mass_balance_overview import paired_bootstrap, paired_coherence, paired_errors, resample_group_counts
from mass_balance_controls import LOG10_LN2


def test_group_bootstrap_matches_explicit_cluster_resampling_with_seed_missingness():
    # Group sizes 3, 1 and 2. The first draw samples the largest group twice.
    membership = np.array([0, 0, 0, 1, 2, 2])
    draw_ids = [[0, 0, 2], [0, 1, 1], [1, 1, 1]]
    counts = np.array([np.bincount(draw, minlength=3) for draw in draw_ids])
    truth = np.array([2., 4., np.nan, 10., 2., 1.])
    base = np.array([[1., 5., 8., 11., 4., 1.], [1., 5., 8., np.nan, 4., 1.]])
    arm = np.array([[2., 9., 8., 9., 8., 1.], [3., np.nan, 8., 9., 2., 4.]])
    base_error, arm_error, common, eligible, _, _ = paired_errors(base, arm, truth)
    result = paired_bootstrap(base_error, arm_error, common, membership, counts)

    def expanded_deltas(indices):
        deltas = []
        for seed in range(len(base)):
            masked = [
                i for i in indices if np.isfinite([truth[i], base[seed, i], arm[seed, i]]).all()
            ]
            errors = [abs(arm[seed, i] - truth[i]) - abs(base[seed, i] - truth[i]) for i in masked]
            deltas.append(np.mean(errors) if errors else np.nan)
        return deltas

    expected = []
    for draw in draw_ids:
        indices = [i for group in draw for i, assigned in enumerate(membership) if group == assigned]
        expected.append(np.mean(expanded_deltas(indices)))

    np.testing.assert_allclose(result["delta_by_draw"], expected, equal_nan=True)
    np.testing.assert_allclose(result["delta_by_seed"], expanded_deltas(range(len(truth))))
    np.testing.assert_array_equal(result["n_scored_by_seed"], [5, 3])
    assert eligible.sum() == 5
    # Mean of per-seed ratios differs from pooling available structures across seeds.
    assert not np.isclose(result["delta_by_seed"].mean(),
                          np.average(result["delta_by_seed"], weights=result["n_scored_by_seed"]))


def test_complete_triad_accuracy_uses_one_common_mask_per_seed():
    truth = np.array([[0., 1., 2.], [3., 4., np.nan], [4., 3., 2.], [1., 0., 1.]])
    base = np.array([[[1., 2., 3.], [3., 4., 2.], [4., 3., 2.], [1., 0., 4.]]])
    arm = np.array([[[0., 1., 4.], [5., 6., 7.], [4., 3., np.nan], [2., 0., 1.]]])
    base_error, arm_error, common, eligible, _, _ = paired_errors(base, arm, truth)
    result = paired_bootstrap(base_error, arm_error, common, np.array([0, 0, 1, 2]), np.ones((1, 3)))

    np.testing.assert_array_equal(eligible, [True, False, True, True])
    np.testing.assert_array_equal(common, [[True, False, False, True]])
    np.testing.assert_allclose(result["baseline_by_seed"], [1.0])
    np.testing.assert_allclose(result["policy_by_seed"], [0.5])
    np.testing.assert_allclose(result["delta_by_draw"], [-0.5])


def test_coherence_retains_invalid_and_outside_threshold_rows():
    base = np.array([[[LOG10_LN2, 0., 0.], [np.nan, 0., 0.], [1., 0., 0.]]])
    arm = np.array([[[LOG10_LN2, 0., 0.], [LOG10_LN2, 0., 0.], [np.inf, 0., 0.]]])
    base_hits, arm_hits, common, eligible, base_valid, arm_valid = paired_coherence(base, arm)
    counts = np.array([[1, 1], [0, 2]])
    result = paired_bootstrap(base_hits, arm_hits, common, np.array([0, 0, 1]), counts)

    np.testing.assert_array_equal(eligible, [True, True, True])
    np.testing.assert_array_equal(base_valid, [[True, False, True]])
    np.testing.assert_array_equal(arm_valid, [[True, True, False]])
    np.testing.assert_array_equal(result["n_scored_by_seed"], [3])
    np.testing.assert_allclose(result["delta_by_draw"], [100 / 3, 0.0])


def test_draw_counts_reproduce_the_seeded_cluster_draws():
    counts = resample_group_counts(5, 20, 20261112)
    raw = np.random.default_rng(20261112).integers(0, 5, size=(20, 5), dtype=np.int32)
    expected = np.stack([np.bincount(row, minlength=5) for row in raw])
    np.testing.assert_array_equal(counts, expected)
    assert (counts > 1).any()
