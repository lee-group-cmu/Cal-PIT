"""Local diagnostics of conditional density estimates (Zhao et al. 2021).

For a well calibrated CDE at x, the PIT is uniform, so r(alpha; x) =
P(PIT <= alpha | x) equals alpha: the local P-P plot is the diagonal. How far
the predicted r strays from the diagonal measures the miscalibration at x.
"""

import dataclasses
from typing import Any

import numpy as np
import numpy.typing as npt

from calpit import _optional, metrics

FloatArray = npt.NDArray[np.floating]


@dataclasses.dataclass(frozen=True)
class LocalCalibration:
    """Predicted local P-P curves of a set of CDEs.

    Attributes:
        alpha: The coverage levels, increasing, shape (n_alpha,).
        pit_cdf: The predicted r(alpha; x) of each object,
            shape (n_objects, n_alpha).
    """

    alpha: FloatArray
    pit_cdf: FloatArray

    @property
    def ks(self) -> FloatArray:
        """The largest distance of each local P-P curve from the diagonal, shape (n_objects,)."""
        return metrics.kolmogorov_smirnov_statistic(self.pit_cdf, self.alpha)

    @property
    def cvm(self) -> FloatArray:
        """The root mean squared distance from the diagonal over alpha, shape (n_objects,)."""
        return metrics.cramer_von_mises_statistic(self.pit_cdf, self.alpha)

    def coverage(self, level: float) -> FloatArray:
        """Returns the coverage of the central interval at a nominal level.

        Args:
            level: The nominal coverage of the interval between the
                (1 - level) / 2 and (1 + level) / 2 quantiles, in (0, 1).

        Returns:
            The predicted probability that the true value falls in that
            interval, for each object, shape (n_objects,).
        """
        return self._pit_cdf_at((1 + level) / 2) - self._pit_cdf_at((1 - level) / 2)

    def _pit_cdf_at(self, alpha: float) -> FloatArray:
        """Interpolates every local P-P curve linearly at one alpha."""
        upper = int(np.clip(np.searchsorted(self.alpha, alpha), 1, len(self.alpha) - 1))
        alpha_lo, alpha_hi = self.alpha[upper - 1], self.alpha[upper]
        weight = np.clip((alpha - alpha_lo) / (alpha_hi - alpha_lo), 0.0, 1.0)
        return (1 - weight) * self.pit_cdf[:, upper - 1] + weight * self.pit_cdf[:, upper]


def plot_local_pp(diagnostics: LocalCalibration, indices: npt.ArrayLike, ax: Any = None) -> tuple[Any, Any]:
    """Plots the local P-P curves of some objects against the diagonal.

    Args:
        diagnostics: The local P-P curves.
        indices: The objects to plot.
        ax: The matplotlib Axes to draw on, or None to create a figure.

    Returns:
        The figure and the axes, as a tuple (fig, ax).

    Raises:
        ImportError: If matplotlib is not installed.
    """
    pyplot = _optional.import_optional("matplotlib.pyplot", "plot", "plot_local_pp")
    if ax is None:
        fig, ax = pyplot.subplots()
    else:
        fig = ax.figure
    for index in np.atleast_1d(indices):
        ax.plot(diagnostics.alpha, diagnostics.pit_cdf[index], label=f"object {index}")
    ax.plot([0, 1], [0, 1], color="k", linestyle="--")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\hat{r}(\alpha; x) = \hat{P}(\mathrm{PIT} \leq \alpha \mid x)$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    return fig, ax
