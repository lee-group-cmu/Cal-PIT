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
    pit = metrics.probability_integral_transform(cde, y_grid, y_true, method="step")
    np.testing.assert_allclose(pit, _masked_pit(cde, y_grid, y_true), rtol=0, atol=1e-12)
    assert pit[0] == 0.0
    np.testing.assert_allclose(pit[1], integrate.trapezoid(cde[1], y_grid))


@pytest.mark.parametrize("y_true", [-11.0, -10.0, -3.37, 0.0, 2.5, 10.0, 11.0])
def test_linear_pit_integrates_the_trapezoid_density_exactly(y_true) -> None:
    y_grid = np.linspace(-10.0, 10.0, 41)
    pdf = stats.norm.pdf(y_grid, 0.5, 2.0)
    pit = metrics.probability_integral_transform(pdf[None, :], y_grid, np.array([y_true]))[0]
    # The trapezoid CDF treats the density as linear between grid points; integrate that exactly.
    fine = np.linspace(-10.0, np.clip(y_true, -10.0, 10.0), 200001)
    expected = integrate.trapezoid(np.interp(fine, y_grid, pdf), fine) if y_true >= -10.0 else 0.0
    np.testing.assert_allclose(pit, expected, rtol=0, atol=1e-9)


def test_linear_pit_error_falls_with_the_square_of_the_spacing(gaussian_cdes) -> None:
    means, y_true = gaussian_cdes.means, gaussian_cdes.y_true
    exact = stats.norm.cdf(y_true, loc=means)
    errors = {}
    for method in ("linear", "step"):
        for spacing in (0.2, 0.1):
            y_grid = np.arange(-10.0, 10.0 + spacing / 2, spacing)
            cde = stats.norm.pdf(y_grid[None, :], loc=means[:, None])
            pit = metrics.probability_integral_transform(cde, y_grid, y_true, method=method)
            errors[method, spacing] = np.abs(pit - exact).max()
    assert 3.5 < errors["linear", 0.2] / errors["linear", 0.1] < 4.5
    assert 1.5 < errors["step", 0.2] / errors["step", 0.1] < 2.5
    assert errors["linear", 0.1] < errors["step", 0.1] / 50


def test_pit_methods_agree_on_grid_points(gaussian_cdes) -> None:
    y_true = gaussian_cdes.y_grid[[100, 900, 1500]]
    cde = gaussian_cdes.cde[:3]
    np.testing.assert_array_equal(
        metrics.probability_integral_transform(cde, gaussian_cdes.y_grid, y_true),
        metrics.probability_integral_transform(cde, gaussian_cdes.y_grid, y_true, method="step"),
    )
    with pytest.raises(ValueError, match="method must be one of"):
        metrics.probability_integral_transform(cde, gaussian_cdes.y_grid, y_true, method="nearest")


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
