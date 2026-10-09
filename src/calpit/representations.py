"""Conditional density estimates (CDEs) on a grid, as quantiles or as samples.

Cal-PIT learns r(alpha; x) = P(PIT <= alpha | x), a monotone map of [0, 1]
onto itself. A CDE with CDF F(y | x) is recalibrated by composing it with that
map, F_new(y | x) = r(F(y | x); x), and its quantile function Q(tau | x) by
composing with the inverse, Q_new(tau | x) = Q(r^-1(tau; x) | x). So every
representation needs only two things, which the classes here provide:

- pit(y_true): the PIT of the true values, to fit Cal-PIT;
- recalibration_alpha() and recalibrate(pit_cdf): the alpha at which r must
  be predicted, and the recalibrated CDEs built from those predictions.

The PIT and the recalibrated CDEs are the same quantities whichever
representation the CDEs come in; only the interpolation error differs.
"""

from typing import Protocol, Self

import numpy as np
import numpy.typing as npt
from scipy import interpolate

from calpit import metrics, utils

FloatArray = npt.NDArray[np.floating]

_MAX_COMPARISONS = 2**24


class ConditionalDensities(Protocol):
    """A set of CDEs, one per object, that Cal-PIT can fit and recalibrate."""

    def __len__(self) -> int:
        """Returns the number of objects."""
        ...

    def pit(self, y_true: npt.ArrayLike) -> FloatArray:
        """Returns the PIT of the true values, shape (n_objects,)."""
        ...

    def recalibration_alpha(self) -> FloatArray:
        """Returns the alpha to predict r at, shape (n_alpha,) or (n_objects, n_alpha)."""
        ...

    def recalibrate(self, pit_cdf: FloatArray) -> Self:
        """Returns the CDEs recalibrated with r predicted at recalibration_alpha()."""
        ...


def _interp_rows(x_new: FloatArray, xp: FloatArray, fp: FloatArray) -> FloatArray:
    """Interpolates each row linearly, like np.interp applied row by row.

    Within a run of equal xp the smallest matching fp is returned, which makes
    this the generalized inverse when fp is a quantile function. Outside the
    range of xp the end values of fp are returned.

    Args:
        x_new: The points to interpolate at, shape (n_rows, n_new).
        xp: The non-decreasing abscissae of each row, shape (n_rows, n_points).
        fp: The values at xp, shape (n_rows, n_points) or (n_points,).

    Returns:
        The interpolated values, shape (n_rows, n_new).
    """
    fp = np.broadcast_to(fp, xp.shape)
    n_points = xp.shape[1]
    # Count the xp below each new point a block of rows at a time, which bounds
    # the (rows, n_new, n_points) comparison to about _MAX_COMPARISONS values.
    rows_per_block = max(1, _MAX_COMPARISONS // (x_new.shape[1] * n_points))
    hi = np.concatenate(
        [
            (
                xp[start : start + rows_per_block, None, :] < x_new[start : start + rows_per_block, :, None]
            ).sum(axis=-1)
            for start in range(0, len(xp), rows_per_block)
        ]
    )
    hi = np.clip(hi, 1, n_points - 1)
    lo = hi - 1
    xp_lo = np.take_along_axis(xp, lo, axis=1)
    xp_hi = np.take_along_axis(xp, hi, axis=1)
    fp_lo = np.take_along_axis(fp, lo, axis=1)
    fp_hi = np.take_along_axis(fp, hi, axis=1)
    width = xp_hi - xp_lo
    safe_width = np.where(width > 0, width, 1.0)
    weight = np.clip(np.where(width > 0, (x_new - xp_lo) / safe_width, 0.0), 0.0, 1.0)
    return fp_lo + weight * (fp_hi - fp_lo)


class GridCDE:
    """CDEs given as densities on a common grid of y.

    Args:
        pdf: The densities, shape (n_objects, n_grid).
        y_grid: The increasing grid, shape (n_grid,).

    Attributes:
        pdf: The densities, shape (n_objects, n_grid).
        y_grid: The grid, shape (n_grid,).
    """

    def __init__(self, pdf: npt.ArrayLike, y_grid: npt.ArrayLike) -> None:
        self.pdf = np.asarray(pdf, dtype=float)
        self.y_grid = np.ravel(np.asarray(y_grid, dtype=float))
        if self.pdf.ndim != 2 or self.pdf.shape[1] != len(self.y_grid):
            raise ValueError(
                f"pdf must have shape (n_objects, len(y_grid)): {self.pdf.shape=}, {self.y_grid.shape=}"
            )

    def __len__(self) -> int:
        """Returns the number of objects."""
        return len(self.pdf)

    def cdf(self) -> FloatArray:
        """Returns the CDFs on the grid by the trapezoid rule, shape (n_objects, n_grid)."""
        return utils.trapz_grid(self.pdf, self.y_grid)

    def pit(self, y_true: npt.ArrayLike) -> FloatArray:
        """Returns the PIT of the true values.

        The CDF is integrated with the trapezoid rule up to the last grid point
        at or below each true value, as calpit.metrics.probability_integral_transform
        does.

        Args:
            y_true: The true values, shape (n_objects,).

        Returns:
            The PIT values, shape (n_objects,).
        """
        return metrics.probability_integral_transform(self.pdf, self.y_grid, np.asarray(y_true))

    def recalibration_alpha(self) -> FloatArray:
        """Returns the CDFs on the grid, where r is needed, shape (n_objects, n_grid)."""
        return self.cdf()

    def recalibrate(self, pit_cdf: FloatArray) -> "GridCDE":
        """Builds the recalibrated densities on the same grid.

        The recalibrated CDF r(F(y | x); x) on the grid is interpolated with a
        PCHIP spline and differentiated.

        Args:
            pit_cdf: r predicted at recalibration_alpha(), shape (n_objects, n_grid).

        Returns:
            The recalibrated CDEs. They can be slightly negative or fail to
            integrate to one; calpit.utils.normalize fixes both.
        """
        cdf = interpolate.PchipInterpolator(self.y_grid, pit_cdf, extrapolate=True, axis=1)
        return GridCDE(cdf.derivative(1)(self.y_grid), self.y_grid)


class QuantileCDE:
    """CDEs given as quantiles at common levels.

    The CDF is linearly interpolated between the quantiles. Outside the
    outermost quantiles it is taken as the outermost level, so include the
    levels 0 and 1, at the ends of the support, to cover every true value.

    Args:
        levels: The increasing quantile levels in [0, 1], shape (n_levels,).
        locations: The quantiles, non-decreasing along each row,
            shape (n_objects, n_levels).

    Attributes:
        levels: The quantile levels, shape (n_levels,).
        locations: The quantiles, shape (n_objects, n_levels).
    """

    def __init__(self, levels: npt.ArrayLike, locations: npt.ArrayLike) -> None:
        self.levels = np.ravel(np.asarray(levels, dtype=float))
        self.locations = np.asarray(locations, dtype=float)
        if self.locations.ndim != 2 or self.locations.shape[1] != len(self.levels):
            raise ValueError(
                "locations must have shape (n_objects, len(levels)): "
                f"{self.locations.shape=}, {self.levels.shape=}"
            )
        if np.any(np.diff(self.levels) <= 0) or self.levels[0] < 0 or self.levels[-1] > 1:
            raise ValueError(f"levels must increase within [0, 1]: {self.levels=}")

    def __len__(self) -> int:
        """Returns the number of objects."""
        return len(self.locations)

    def pit(self, y_true: npt.ArrayLike) -> FloatArray:
        """Returns the PIT of the true values.

        Args:
            y_true: The true values, shape (n_objects,).

        Returns:
            The PIT values, shape (n_objects,).
        """
        y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)
        return _interp_rows(y_true, self.locations, self.levels)[:, 0]

    def recalibration_alpha(self) -> FloatArray:
        """Returns the quantile levels, where r is needed, shape (n_levels,)."""
        return self.levels

    def recalibrate(self, pit_cdf: FloatArray) -> "QuantileCDE":
        """Builds the recalibrated quantiles at the same levels.

        The recalibrated CDF takes the value r(tau; x) at the old quantile
        Q(tau | x), so the new quantile at each level is interpolated between
        the old quantiles.

        Args:
            pit_cdf: r predicted at the levels, non-decreasing along each row,
                shape (n_objects, n_levels).

        Returns:
            The recalibrated CDEs. A level outside the range of r(levels; x)
            gets the outermost old quantile.
        """
        new_levels = np.broadcast_to(self.levels, self.locations.shape)
        pit_cdf = np.broadcast_to(pit_cdf, self.locations.shape)
        return QuantileCDE(self.levels, _interp_rows(new_levels, pit_cdf, self.locations))


class SampleCDE:
    """CDEs given as samples from each predictive distribution.

    Args:
        samples: The samples, shape (n_objects, n_samples).
        random_state: The seed or generator of the randomized PIT.

    Attributes:
        samples: The samples, shape (n_objects, n_samples).
        random_state: The seed or generator of the randomized PIT.
    """

    def __init__(self, samples: npt.ArrayLike, random_state: int | np.random.Generator | None = None) -> None:
        self.samples = np.asarray(samples, dtype=float)
        self.random_state = random_state
        if self.samples.ndim != 2:
            raise ValueError(f"samples must have shape (n_objects, n_samples): {self.samples.shape=}")

    def __len__(self) -> int:
        """Returns the number of objects."""
        return len(self.samples)

    def pit(self, y_true: npt.ArrayLike) -> FloatArray:
        """Returns the randomized PIT of the true values.

        With k of the m samples below y, the PIT is (k + U) / (m + 1) with U
        uniform on [0, 1], spread over any samples equal to y. It is exactly
        uniform when y is drawn from the same distribution as the samples,
        whereas the empirical CDF k / m only takes m + 1 values.

        Args:
            y_true: The true values, shape (n_objects,).

        Returns:
            The PIT values, shape (n_objects,).
        """
        y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)
        below = (self.samples < y_true).sum(axis=1)
        ties = (self.samples == y_true).sum(axis=1)
        uniform = np.random.default_rng(self.random_state).uniform(size=len(below))
        return (below + uniform * (ties + 1)) / (self.samples.shape[1] + 1)

    def to_quantiles(self) -> QuantileCDE:
        """Returns the sorted samples as quantiles at the levels i / (m + 1), i = 1, ..., m."""
        n_samples = self.samples.shape[1]
        levels = np.arange(1, n_samples + 1) / (n_samples + 1)
        return QuantileCDE(levels, np.sort(self.samples, axis=1))

    def recalibration_alpha(self) -> FloatArray:
        """Returns the levels of the sorted samples, where r is needed, shape (n_samples,)."""
        return self.to_quantiles().levels

    def recalibrate(self, pit_cdf: FloatArray) -> "SampleCDE":
        """Moves each sorted sample to the recalibrated quantile at its level.

        Args:
            pit_cdf: r predicted at recalibration_alpha(), non-decreasing along
                each row, shape (n_objects, n_samples).

        Returns:
            The recalibrated CDEs, as sorted samples.
        """
        quantiles = self.to_quantiles().recalibrate(pit_cdf)
        return SampleCDE(quantiles.locations, self.random_state)
