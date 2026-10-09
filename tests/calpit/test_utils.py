"""Tests for calpit.utils."""

import numpy as np
import pytest
from scipy import integrate

from calpit import utils


def test_trapz_grid_matches_cumulative_trapezoid(gaussian_cdes) -> None:
    expected = integrate.cumulative_trapezoid(gaussian_cdes.cde, gaussian_cdes.y_grid, axis=1, initial=0)
    np.testing.assert_allclose(utils.trapz_grid(gaussian_cdes.cde, gaussian_cdes.y_grid), expected)


@pytest.mark.parametrize("n_rows", [None, 50])
def test_normalize_gives_nonnegative_unit_densities(gaussian_cdes, rng, n_rows) -> None:
    y_grid = gaussian_cdes.y_grid
    noisy = gaussian_cdes.cde[:50] * 1.7 + rng.normal(0, 0.02, size=(50, len(y_grid)))
    if n_rows is None:
        noisy = noisy[0]
    normalized = utils.normalize(noisy.copy(), y_grid)
    assert normalized.shape == noisy.shape
    assert (normalized >= 0).all()
    np.testing.assert_allclose(integrate.trapezoid(normalized, y_grid, axis=-1), 1.0, atol=1e-5)


def test_plot_pit(gaussian_cdes, rng) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib  # noqa: PLC0415 - optional dependency.

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt  # noqa: PLC0415 - optional dependency.

    pit = rng.uniform(size=500)
    fig, ax = utils.plot_pit(pit, ci_level=0.95, y_true=gaussian_cdes.y_true[:500])
    assert len(ax) == 2
    plt.close(fig)

    fig, ax = plt.subplots(1, 2)
    returned_fig, _ = utils.plot_pit(pit, ci_level=0.95, ax=ax)
    assert returned_fig is fig
    plt.close(fig)
