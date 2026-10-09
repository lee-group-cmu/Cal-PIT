"""Helpers for CDEs on a grid: normalization, integration and the PIT histogram plot."""

import numpy as np
from scipy import integrate, stats

from calpit import metrics


def normalize(
    cde_estimates: np.ndarray, y_grid: np.ndarray, tol: float = 1e-6, max_iter: int = 200
) -> np.ndarray:
    """
    Normalizes conditional density estimates to be non-negative and integrate to one.

    Args:
        cde_estimates (numpy.ndarray): A numpy array or matrix of conditional density estimates.
        y_grid (numpy.ndarray): The array of grid points.
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

    area = integrate.trapezoid(np.maximum(density, 0.0), y_grid)
    if area == 0.0:
        # replace with uniform if all negative density
        density[:] = 1 / (y_grid.max() - y_grid.min())
    elif area < 1:
        density /= area
        density[density < 0.0] = 0.0
        return density

    for _ in range(max_iter):
        mid = (hi + lo) / 2
        area = integrate.trapezoid(np.maximum(density - mid, 0.0), y_grid)
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
    Plots the PIT histogram and the P-P plot of PIT values against a uniform distribution.

    The histogram is drawn with the band that a uniform distribution would fall in with
    probability ci_level. When y_true is given, the P-P plot is annotated with the
    Kolmogorov-Smirnov, Cramer-von Mises and Anderson-Darling statistics.

    Args:
        pit_values (np.ndarray): The PIT values, shape (n_samples,).
        ci_level (float): The confidence level of the uniform band, between 0 and 1.
        n_bins (int, optional): The number of histogram bins. Defaults to 30.
        y_true (np.ndarray, optional): The true values; only its length is used, to scale the
            Anderson-Darling statistic. Defaults to None, which omits the statistics.
        ax (sequence of matplotlib.axes.Axes, optional): Two axes to draw the histogram and the
            P-P plot on. Defaults to None, which creates a new figure.
        **fig_kw: Keyword arguments passed to matplotlib.pyplot.subplots when ax is None.

    Returns:
        tuple: The matplotlib figure and the array of two axes, as a tuple (fig, ax).

    Raises:
        ImportError: If matplotlib is not installed.
    """
    try:
        from matplotlib import pyplot as plt  # noqa: PLC0415 - optional dependency.
    except ImportError as error:
        raise ImportError(
            "plot_pit requires the optional dependency matplotlib. "
            "Install it with: pip install 'calpit[plot]'"
        ) from error

    # Extract the number of CDEs
    n = pit_values.shape[0]

    # Creating upper and lower limit for selected uniform band
    ci_quantity = (1 - ci_level) / 2
    low_lim = stats.binom.ppf(q=ci_quantity, n=n, p=1 / n_bins)
    upp_lim = stats.binom.ppf(q=ci_level + ci_quantity, n=n, p=1 / n_bins)

    # Creating figure

    if ax is None:
        fig, ax = plt.subplots(1, 2, **fig_kw)
    else:
        fig = ax[0].figure

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
    prob_data = np.array([np.sum(pit_values < i) / len(pit_values) for i in prob_theory])
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
        ks = metrics.kolmogorov_smirnov_statistic(prob_data, prob_theory)
        ad = metrics.anderson_darling_statistic(prob_data, prob_theory, len(y_true))
        cvm = metrics.cramer_von_mises_statistic(prob_data, prob_theory)
        ax[1].text(0.05, 0.9, f"KS:  ${ks:.3f} $", fontsize=15)
        ax[1].text(0.05, 0.84, f"CvM:  ${cvm:.3f} $", fontsize=15)
        ax[1].text(0.05, 0.78, f"AD:  ${ad:.2f} $", fontsize=15)

    return fig, ax
