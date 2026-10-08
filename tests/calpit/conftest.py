"""Shared fixtures: Gaussian conditional density estimates on a grid."""

import dataclasses

import numpy as np
import numpy.typing as npt
import pytest
from scipy import stats

FloatArray = npt.NDArray[np.floating]


@dataclasses.dataclass(frozen=True)
class GaussianCdes:
    """Gaussian CDEs on a grid with targets drawn from a known distribution.

    Attributes:
        y_grid: The grid, shape (n_grid,).
        means: The mean of each CDE, shape (n_samples,).
        cde: The CDEs N(means, 1) on y_grid, shape (n_samples, n_grid).
        y_true: Targets drawn from N(means, 1), so the CDEs are calibrated,
            shape (n_samples,).
    """

    y_grid: FloatArray
    means: FloatArray
    cde: FloatArray
    y_true: FloatArray


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def gaussian_cdes(rng: np.random.Generator) -> GaussianCdes:
    """2000 calibrated unit-variance Gaussian CDEs on a fine grid."""
    y_grid = np.linspace(-10.0, 10.0, 2001)
    means = rng.uniform(-3.0, 3.0, size=2000)
    cde = stats.norm.pdf(y_grid[None, :], loc=means[:, None])
    y_true = rng.normal(means, 1.0)
    return GaussianCdes(y_grid=y_grid, means=means, cde=cde, y_true=y_true)
