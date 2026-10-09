"""Losses, PIT values and distances between CDFs for assessing CDEs on a grid."""

import numpy as np
from scipy import integrate


def cde_loss(cde_estimates: np.ndarray, y_grid: np.ndarray, y_test: np.ndarray) -> tuple:
    """
    Calculates conditional density estimation loss on holdout data.

    Args:
        cde_estimates (numpy.array): An array where each row is a density estimate on y_grid.
        y_grid (numpy.array): An array of the grid points at which cde_estimates is evaluated.
        y_test (numpy.array): An array of the true y values corresponding to the rows of cde_estimates.

    Returns:
        tuple: A tuple containing the loss and the standard error of the loss.

    Raises:
        ValueError: If the dimensions of the input tensors are not compatible.
    """

    if len(y_test.shape) == 1:
        y_test = y_test.reshape(-1, 1)
    if len(y_grid.shape) == 1:
        y_grid = y_grid.reshape(-1, 1)

    n_obs, n_grid = cde_estimates.shape
    n_samples, feats_samples = y_test.shape
    n_grid_points, feats_grid = y_grid.shape

    if n_obs != n_samples:
        raise ValueError(
            f"Number of samples in CDEs should be the same as in y_test. Currently {n_obs} and {n_samples}."
        )
    if n_grid != n_grid_points:
        raise ValueError(
            "Number of grid points in CDEs should be the same as in y_grid. "
            f"Currently {n_grid} and {n_grid_points}."
        )

    if feats_samples != feats_grid:
        raise ValueError(
            "Dimensionality of test points and grid points need to coincide. "
            f"Currently {feats_samples} and {feats_grid}."
        )

    integrals = integrate.trapezoid(cde_estimates**2, np.squeeze(y_grid), axis=1)

    nn_ids = np.argmin(np.abs(y_grid - y_test.T), axis=0)
    likeli = cde_estimates[(tuple(np.arange(n_samples)), tuple(nn_ids))]

    losses = integrals - 2 * likeli
    loss = np.mean(losses)
    se_error = np.std(losses, axis=0) / (n_obs**0.5)

    return loss, se_error


def kolmogorov_smirnov_statistic(cdf_test: np.ndarray, cdf_ref: np.ndarray) -> np.ndarray:
    """
    Calculate the Kolmogorov-Smirnov statistic between two cumulative distribution functions (CDFs).

    Args:
        cdf_test (np.ndarray): CDF of the test distribution.
        cdf_ref (np.ndarray): CDF of the reference distribution on the same grid.

    Returns:
        np.ndarray: The Kolmogorov-Smirnov statistic.

    """
    ks = np.max(np.abs(cdf_test - cdf_ref), axis=-1)

    return ks


def cramer_von_mises_statistic(cdf_test: np.ndarray, cdf_ref: np.ndarray) -> np.ndarray:
    """
    Calculates the Cramer-von Mises statistic between two cumulative distribution functions (CDFs).

    Args:
        cdf_test (np.ndarray): CDF of the test distribution.
        cdf_ref (np.ndarray): CDF of the reference distribution on the same grid.

    Returns:
        np.ndarray: The Cramer-von Mises statistic.

    """
    diff = (cdf_test - cdf_ref) ** 2

    cvm2 = integrate.trapezoid(diff, cdf_ref, axis=-1)
    return np.sqrt(cvm2)


def anderson_darling_statistic(cdf_test: np.ndarray, cdf_ref: np.ndarray, n_tot: int = 1) -> np.ndarray:
    """
    Calculates the Anderson-Darling statistic between two cumulative distribution functions (CDFs).

    Args:
        cdf_test (np.ndarray): CDF of the test distribution (1D array).
        cdf_ref (np.ndarray): CDF of the reference distribution on the same grid (1D array).
        n_tot (int): Scaling factor equal to the number of PDFs used to construct ECDF.

    Returns:
        np.ndarray: The Anderson-Darling statistic.

    """
    num = (cdf_test - cdf_ref) ** 2
    den = cdf_ref * (1 - cdf_ref)

    ad2 = n_tot * integrate.trapezoid((num / den), cdf_ref, axis=-1)
    return np.sqrt(ad2)


PIT_METHODS = ("linear", "step")


def probability_integral_transform(
    cde: np.ndarray, y_grid: np.ndarray, y_test: np.ndarray, method: str = "linear"
) -> np.ndarray:
    """
    Calculates the Probability Integral Transform (PIT) based on Conditional Density Estimates (CDE).

    The CDF is the trapezoid-rule integral of the density, which treats the density as linear
    between grid points. With method "linear" (the default) the PIT integrates that piecewise
    linear density exactly up to each true value, so its error falls with the square of the grid
    spacing. Method "step" stops at the last grid point at or below the true value, as calpit
    did before version 0.2; it is biased low by up to the density times the grid spacing, and is
    kept to reproduce earlier results. Either way the PIT is 0 below the grid and the integral
    over the whole grid above it.

    Args:
        cde (np.ndarray): A numpy array of conditional density estimates.
            Each row corresponds to an observation, each column corresponds to a grid point.
        y_grid (np.ndarray): A numpy array of the grid points at which cde is evaluated.
        y_test (np.ndarray): A numpy array of the true y values corresponding to the rows of cde.
        method (str, optional): "linear" or "step", as described above. Defaults to "linear".

    Returns:
        np.ndarray: A numpy array of PIT values.

    Raises:
        ValueError: If the number of samples in cde is not the same as in y_test,
            or if the number of grid points in cde is not the same as in y_grid.

    """
    # flatten the input arrays to 1D
    y_grid = np.ravel(y_grid)
    y_test = np.ravel(y_test)

    # Sanity checks
    nrow_cde, ncol_cde = cde.shape
    n_samples = y_test.shape[0]
    n_grid_points = y_grid.shape[0]

    if nrow_cde != n_samples:
        raise ValueError(
            "Number of samples in CDEs should be the same as in y_test. "
            f"Currently {nrow_cde} and {n_samples}."
        )
    if ncol_cde != n_grid_points:
        raise ValueError(
            "Number of grid points in CDEs should be the same as in y_grid. "
            f"Currently {ncol_cde} and {n_grid_points}."
        )

    if method not in PIT_METHODS:
        raise ValueError(f"method must be one of {PIT_METHODS}: {method=}")

    # The CDF integrated up to the last grid point at or below y_test.
    cdf = integrate.cumulative_trapezoid(cde, y_grid, axis=1, initial=0)
    last_below = np.searchsorted(y_grid, y_test, side="right") - 1
    rows = np.arange(n_samples)
    left = np.clip(last_below, 0, n_grid_points - 1)
    pit = cdf[rows, left]

    if method == "linear" and n_grid_points > 1:
        # Add the integral of the linearly interpolated density from that grid point to y_test.
        right = np.minimum(left + 1, n_grid_points - 1)
        spacing = y_grid[right] - y_grid[left]
        inside = (last_below >= 0) & (last_below < n_grid_points - 1)
        step = np.where(inside, y_test - y_grid[left], 0.0)
        slope = np.where(inside, (cde[rows, right] - cde[rows, left]) / np.where(inside, spacing, 1.0), 0.0)
        density_at_y = cde[rows, left] + slope * step
        pit = pit + step * (cde[rows, left] + density_at_y) / 2

    return np.where(last_below >= 0, pit, 0.0)
