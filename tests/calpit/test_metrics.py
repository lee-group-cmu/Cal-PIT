"""Tests for calpit.metrics and the PyTorch cde_loss in calpit.nn.utils."""

import numpy as np
import pytest
import torch
from scipy import integrate, stats

from calpit import metrics
from calpit.nn import utils as nn_utils


def _masked_pit(cde: np.ndarray, y_grid: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """The PIT as calpit <= 0.1.2 computed it: integrate the grid points <= y."""
    pit = np.zeros(len(y_true))
    for i, y in enumerate(y_true):
        below = y_grid <= y
        if below.sum() > 1:
            pit[i] = integrate.trapezoid(cde[i, below], y_grid[below])
    return pit


def test_pit_of_calibrated_cdes_is_uniform(gaussian_cdes) -> None:
    pit = metrics.probability_integral_transform(
        gaussian_cdes.cde, gaussian_cdes.y_grid, gaussian_cdes.y_true
    )
    assert stats.kstest(pit, "uniform").pvalue > 0.01


def test_pit_matches_previous_implementation(gaussian_cdes, rng) -> None:
    y_grid = gaussian_cdes.y_grid[::40]  # coarse, so truncation at grid points matters
    cde = gaussian_cdes.cde[:, ::40]
    y_true = gaussian_cdes.y_true.copy()
    y_true[:3] = [y_grid[0] - 1.0, y_grid[-1] + 1.0, y_grid[25]]  # below, above, on a node
    pit = metrics.probability_integral_transform(cde, y_grid, y_true)
    np.testing.assert_allclose(pit, _masked_pit(cde, y_grid, y_true), rtol=0, atol=1e-12)
    assert pit[0] == 0.0
    np.testing.assert_allclose(pit[1], integrate.trapezoid(cde[1], y_grid))


def test_pit_rejects_mismatched_shapes(gaussian_cdes) -> None:
    with pytest.raises(ValueError, match="Number of samples"):
        metrics.probability_integral_transform(
            gaussian_cdes.cde, gaussian_cdes.y_grid, gaussian_cdes.y_true[:-1]
        )
    with pytest.raises(ValueError, match="Number of grid points"):
        metrics.probability_integral_transform(
            gaussian_cdes.cde, gaussian_cdes.y_grid[:-1], gaussian_cdes.y_true
        )


def test_cde_loss_prefers_the_true_density(gaussian_cdes) -> None:
    too_wide = stats.norm.pdf(gaussian_cdes.y_grid[None, :], loc=gaussian_cdes.means[:, None], scale=3.0)
    true_loss, true_se = metrics.cde_loss(gaussian_cdes.cde, gaussian_cdes.y_grid, gaussian_cdes.y_true)
    wide_loss, _ = metrics.cde_loss(too_wide, gaussian_cdes.y_grid, gaussian_cdes.y_true)
    assert true_loss < wide_loss
    assert true_se > 0
    # E[integral p^2 - 2 p(Y)] for a unit Gaussian is 1 / (2 sqrt(pi)) - 2 / (2 sqrt(pi)).
    np.testing.assert_allclose(true_loss, -1 / (2 * np.sqrt(np.pi)), atol=4 * true_se)


def test_torch_cde_loss_matches_numpy(gaussian_cdes) -> None:
    expected = metrics.cde_loss(gaussian_cdes.cde, gaussian_cdes.y_grid, gaussian_cdes.y_true)
    loss, se = nn_utils.cde_loss(
        torch.tensor(gaussian_cdes.cde),
        torch.tensor(gaussian_cdes.y_grid),
        torch.tensor(gaussian_cdes.y_true),
    )
    np.testing.assert_allclose([loss.item(), se.item()], expected, rtol=1e-10)


def test_distance_statistics_vanish_for_identical_cdfs() -> None:
    cdf = np.linspace(0.01, 0.99, 50)
    assert metrics.kolmogorov_smirnov_statistic(cdf, cdf) == 0.0
    assert metrics.cramer_von_mises_statistic(cdf, cdf) == 0.0
    assert metrics.anderson_darling_statistic(cdf, cdf, n_tot=10) == 0.0


def test_distance_statistics_of_a_shifted_cdf() -> None:
    cdf_ref = np.linspace(0.01, 0.99, 50)
    cdf_test = cdf_ref + 0.01
    np.testing.assert_allclose(metrics.kolmogorov_smirnov_statistic(cdf_test, cdf_ref), 0.01)
    # sqrt(integral of 0.01^2 over [0.01, 0.99]).
    np.testing.assert_allclose(metrics.cramer_von_mises_statistic(cdf_test, cdf_ref), 0.01 * np.sqrt(0.98))
    assert metrics.anderson_darling_statistic(cdf_test, cdf_ref, n_tot=10) > 0
