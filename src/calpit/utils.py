from scipy.stats import binom
from matplotlib import pyplot as plt
import numpy as np


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
    # TODO: Use an alternate root finding method to vectorize this
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
