from scipy.stats import binom
from matplotlib import pyplot as plt
import numpy as np


def cde_loss(cde_estimates: np.ndarray, y_grid: np.ndarray, y_test: np.ndarray) -> tuple:
    """
    Calculates conditional density estimation loss on holdout data.

    Args:
        cde_estimates (numpy.array): An array where each row is a density estimate on y_grid.
        z_grid (numpy.array): An array of the grid points at which cde_estimates is evaluated.
        z_test (numpy.array): An array of the true y values corresponding to the rows of cde_estimates.

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
            f"Number of samples in CDEs should be the same as in z_test.Currently {n_obs} and {n_samples}."
        )
    if n_grid != n_grid_points:
        raise ValueError(
            f"Number of grid points in CDEs should be the same as in z_grid. Currently {n_grid} and {n_grid_points}."
        )

    if feats_samples != feats_grid:
        raise ValueError(
            f"Dimensionality of test points and grid points need to coincise. Currently {feats_samples} and {feats_grid}."
        )

    integrals = np.trapz(cde_estimates**2, np.squeeze(y_grid), axis=1)

    nn_ids = np.argmin(np.abs(y_grid - y_test.T), axis=0)
    likeli = cde_estimates[(tuple(np.arange(n_samples)), tuple(nn_ids))]

    losses = integrals - 2 * likeli
    loss = np.mean(losses)
    se_error = np.std(losses, axis=0) / (n_obs**0.5)

    return loss, se_error


def normalize(
    cde_estimates: np.ndarray, y_grid: np.ndarray, tol: float = 1e-6, max_iter: int = 200
) -> np.ndarray:
    """
    Normalizes conditional density estimates to be non-negative and integrate to one.

    Args:
        cde_estimates (numpy.ndarray): A numpy array or matrix of conditional density estimates.
        x_grid (numpy.ndarray): The array of grid points.
        tol (float): The tolerance to accept for abs(area - 1).
        max_iter (int): The maximal number of search iterations.

    Returns:
        numpy.ndarray: The normalized conditional density estimates.

    """
    if cde_estimates.ndim == 1:
        normalized_cde = _normalize(cde_estimates, y_grid, tol, max_iter)
    else:
        normalized_cde = np.apply_along_axis(_normalize, 1, cde_estimates, y_grid, tol=tol, max_iter=max_iter)
    return normalized_cde


def _normalize(density, y_grid, tol=1e-6, max_iter=500):
    hi = np.max(density)
    lo = 0.0

    area = np.trapz(np.maximum(density, 0.0), y_grid)
    if area == 0.0:
        # replace with uniform if all negative density
        density[:] = 1 / (y_grid.max() - y_grid.min())
    elif area < 1:
        density /= area
        density[density < 0.0] = 0.0
        return density

    for _ in range(max_iter):
        mid = (hi + lo) / 2
        area = np.trapz(np.maximum(density - mid, 0.0), y_grid)
        if abs(1.0 - area) <= tol:
            break
        if area < 1.0:
            hi = mid
        else:
            lo = mid

    # update in place
    density -= mid
    density[density < 0.0] = 0.0

    return density


def kolmogorov_smirnov_statistic(cdf_test: np.ndarray, cdf_ref: np.ndarray) -> np.ndarray:
    """
    Calculate the Kolmogorov-Smirnov statistic between two cumulative distribution functions (CDFs).

    Parameters:
    cdf_test (np.ndarray): CDF of the test distribution.
    cdf_ref (np.ndarray): CDF of the reference distribution on the same grid.

    Returns:
    np.ndarray: The Kolmogorov-Smirnov statistic.

    """
    ks = np.max(np.abs(cdf_test - cdf_ref), axis=-1)

    return ks


def cramer_von_mises(cdf_test: np.ndarray, cdf_ref: np.ndarray) -> np.ndarray:
    """
    Calculates the Cramer-von Mises statistic between two cumulative distribution functions (CDFs).

    Args:
        cdf_test (np.ndarray): CDF of the test distribution.
        cdf_ref (np.ndarray): CDF of the reference distribution on the same grid.

    Returns:
        np.ndarray: The Cramer-von Mises statistic.

    """
    diff = (cdf_test - cdf_ref) ** 2

    cvm2 = np.trapz(diff, cdf_ref, axis=-1)
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

    ad2 = n_tot * np.trapz((num / den), cdf_ref, axis=-1)
    return np.sqrt(ad2)


def probability_integral_transform(cde: np.ndarray, y_grid: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """
    Calculates the Probability Integral Transform (PIT) based on Conditional Density Estimates (CDE).

    Args:
        cde (np.ndarray): A numpy array of conditional density estimates.
            Each row corresponds to an observation, each column corresponds to a grid point.
        y_grid (np.ndarray): A numpy array of the grid points at which cde is evaluated.
        y_test (np.ndarray): A numpy array of the true y values corresponding to the rows of cde.

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
            f"Number of samples in CDEs should be the same as in z_test. Currently {nrow_cde} and {n_samples}."
        )
    if ncol_cde != n_grid_points:
        raise ValueError(
            f"Number of grid points in CDEs should be the same as in z_grid. Currently {nrow_cde} and {n_grid_points}."
        )

    # Vectorized implementation using masked arrays
    pit = np.ma.masked_array(cde, (y_grid > y_test[:, np.newaxis]))
    pit = np.trapz(pit, y_grid)

    return np.array(pit)


def trapz_grid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Does trapezoid integration between the same limits as the grid.

    Args:
        y (np.ndarray): The array of values to integrate.
        x (np.ndarray): The array of grid points.

    Returns:
        np.ndarray: The integrated values.

    """
    dx = np.diff(x)
    trapz_area = dx * (y[:, 1:] + y[:, :-1]) / 2
    integral = np.cumsum(trapz_area, axis=-1)
    return np.hstack((np.zeros(len(integral))[:, None], integral))


def plot_pit(pit_values, ci_level, n_bins=30, y_true=None, ax=None, **fig_kw):
    """
    Plots the PIT/HPD histogram and calculates the confidence interval for the bin values,
    were the PIT/HPD values follow an uniform distribution

    @param values: a numpy array with PIT/HPD values
    @param ci_level: a float between 0 and 1 indicating the size of the confidence level
    @param x_label: a string, populates the x_label of the plot
    @param n_bins: an integer, the number of bins in the histogram
    @param figsize: a tuple, the plot size (width, height)
    @param ylim: a list of two elements, including the lower and upper limit for the y axis
    @returns The matplotlib figure object with the histogram of the PIT/HPD values
    and the CI for the uniform distribution
    """

    # Extract the number of CDEs
    n = pit_values.shape[0]

    # Creating upper and lower limit for selected uniform band
    ci_quantity = (1 - ci_level) / 2
    low_lim = binom.ppf(q=ci_quantity, n=n, p=1 / n_bins)
    upp_lim = binom.ppf(q=ci_level + ci_quantity, n=n, p=1 / n_bins)

    # Creating figure

    if ax is None:
        fig, ax = plt.subplots(1, 2, **fig_kw)

    # plot PIT histogram
    ax[0].hist(pit_values, bins=n_bins)
    ax[0].axhline(y=low_lim, color="grey")
    ax[0].axhline(y=upp_lim, color="grey")
    ax[0].axhline(y=n / n_bins, label="Uniform Average", color="red")
    ax[0].fill_between(
        x=np.linspace(0, 1, 100),
        y1=np.repeat(low_lim, 100),
        y2=np.repeat(upp_lim, 100),
        color="grey",
        alpha=0.2,
    )
    ax[0].set_xlabel("PIT Values")
    ax[0].legend(loc="best")

    # plot P-P plot
    prob_theory = np.linspace(0.01, 0.99, 100)
    prob_data = [np.sum(pit_values < i) / len(pit_values) for i in prob_theory]
    # # plot Q-Q
    # quants = np.linspace(0, 100, 100)
    # quant_theory = quants/100.
    # quant_data = np.percentile(pit_values,quants)

    ax[1].scatter(prob_theory, prob_data, marker=".")
    ax[1].plot(prob_theory, prob_theory, c="k", ls="--")
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel("Expected Cumulative Probability")
    ax[1].set_ylabel("Empirical Cumulative Probability")
    xlabels = np.linspace(0, 1, 6)[1:]
    ax[1].set_xticks(xlabels)
    ax[1].set_aspect("equal")
    if y_true is not None:
        ks = kolmogorov_smirnov_statistic(prob_data, prob_theory)
        ad = anderson_darling_statistic(prob_data, prob_theory, len(y_true))
        cvm = cramer_von_mises(prob_data, prob_theory)
        ax[1].text(0.05, 0.9, f"KS:  ${ks:.3f} $", fontsize=15)
        ax[1].text(0.05, 0.84, f"CvM:  ${cvm:.3f} $", fontsize=15)
        ax[1].text(0.05, 0.78, f"AD:  ${ad:.2f} $", fontsize=15)

    return fig, ax
