"""Tests for calpit.coverage."""

import numpy as np

from calpit import coverage


def test_rearrange_leaves_monotone_curves_unchanged() -> None:
    alpha = np.linspace(0, 1, 11)
    pit_cdf = np.tile(alpha**2, (3, 1))
    np.testing.assert_array_equal(coverage.rearrange(pit_cdf, alpha), pit_cdf)


def test_rearrange_sorts_along_increasing_alpha() -> None:
    alpha = np.array([[0.5, 0.1, 0.9], [0.2, 0.3, 0.1]])
    pit_cdf = np.array([[0.2, 0.6, 0.4], [0.1, 0.9, 0.5]])
    np.testing.assert_array_equal(coverage.rearrange(pit_cdf, alpha), [[0.4, 0.2, 0.6], [0.5, 0.9, 0.1]])


def test_rearrange_is_never_further_from_a_monotone_truth() -> None:
    rng = np.random.default_rng(0)
    alpha = np.linspace(0, 1, 101)
    truth = alpha**0.5
    noisy = truth + rng.normal(0, 0.05, (200, len(alpha)))
    error_before = np.abs(noisy - truth).sum(axis=1)
    error_after = np.abs(coverage.rearrange(noisy, alpha) - truth).sum(axis=1)
    assert (error_after <= error_before + 1e-12).all()


def test_train_val_split_is_the_seeded_permutation() -> None:
    x = np.arange(20.0).reshape(10, 2)
    pit = np.arange(10.0) / 10
    x_train, pit_train, x_val, pit_val = coverage.train_val_split(x, pit, 0.3, random_state=7)
    order = np.random.default_rng(7).permutation(10)
    np.testing.assert_array_equal(pit_train, pit[order[:7]])
    np.testing.assert_array_equal(pit_val, pit[order[7:]])
    np.testing.assert_array_equal(x_val, x[order[7:]])
    assert len(x_train) == 7


def test_stratified_alpha_has_one_draw_per_stratum() -> None:
    alpha = coverage.stratified_alpha(1000, 20, random_state=0)
    assert alpha.shape == (1000, 20)
    stratum = np.floor(alpha * 20)
    np.testing.assert_array_equal(stratum, np.tile(np.arange(20), (1000, 1)))


def test_expand_coverage_rows_and_targets() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    pit = np.array([0.3, 0.7])
    features, targets = coverage.expand_coverage(x, pit, np.array([0.5, 0.8]))
    np.testing.assert_array_equal(features, [[0.5, 1, 2], [0.8, 1, 2], [0.5, 3, 4], [0.8, 3, 4]])
    np.testing.assert_array_equal(targets, [True, True, False, True])
