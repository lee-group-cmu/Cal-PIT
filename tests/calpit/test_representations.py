"""Grid, quantile and sample CDEs give the same PIT and the same recalibration."""

import numpy as np
import pytest
from scipy import stats

import calpit
from calpit import representations

N_OBJECTS = 400
Y_GRID = np.linspace(-8.0, 8.0, 801)
LEVELS = np.linspace(0.0, 1.0, 401)


def _squared(alpha: np.ndarray) -> np.ndarray:
    """r(alpha) = alpha**2: the recalibrated CDF is Phi(y - mean)**2."""
    return alpha**2


@pytest.fixture(scope="module")
def means() -> np.ndarray:
    return np.random.default_rng(0).uniform(-2.0, 2.0, N_OBJECTS)


@pytest.fixture(scope="module")
def y_true(means) -> np.ndarray:
    return np.random.default_rng(1).normal(means, 1.0)


def _grid(means: np.ndarray) -> calpit.GridCDE:
    return calpit.GridCDE(stats.norm.pdf(Y_GRID[None, :], loc=means[:, None]), Y_GRID)


def _quantiles(means: np.ndarray) -> calpit.QuantileCDE:
    # Levels 0 and 1 sit at the ends of the grid, standing in for the support.
    locations = stats.norm.ppf(np.clip(LEVELS, 1e-12, 1 - 1e-12), loc=means[:, None])
    return calpit.QuantileCDE(LEVELS, np.clip(locations, Y_GRID[0], Y_GRID[-1]))


def _samples(means: np.ndarray) -> calpit.SampleCDE:
    return calpit.SampleCDE(
        np.random.default_rng(2).normal(means[:, None], 1.0, (N_OBJECTS, 2000)), random_state=3
    )


# The grid PIT stops at the last grid point at or below y, as calpit always has,
# which biases it low by up to max(pdf) * grid spacing; test_metrics checks its
# uniformity on a finer grid.
@pytest.mark.parametrize("make", [_quantiles, _samples], ids=["quantiles", "samples"])
def test_pit_of_calibrated_cdes_is_uniform(make, means, y_true) -> None:
    pit = make(means).pit(y_true)
    assert pit.shape == (N_OBJECTS,)
    assert stats.kstest(pit, "uniform").pvalue > 0.01


def test_pit_agrees_across_representations(means, y_true) -> None:
    exact = stats.norm.cdf(y_true, loc=means)
    spacing = Y_GRID[1] - Y_GRID[0]
    grid_pit = _grid(means).pit(y_true)
    assert ((grid_pit <= exact + 1e-9) & (grid_pit >= exact - stats.norm.pdf(0) * spacing)).all()
    np.testing.assert_allclose(_quantiles(means).pit(y_true), exact, atol=2e-3)
    np.testing.assert_allclose(_samples(means).pit(y_true), exact, atol=0.05)


@pytest.mark.parametrize("make", [_grid, _quantiles, _samples], ids=["grid", "quantiles", "samples"])
def test_recalibration_agrees_with_the_exact_answer(make, means) -> None:
    cde = make(means)
    alpha = cde.recalibration_alpha()
    new = cde.recalibrate(_squared(np.broadcast_to(alpha, (N_OBJECTS, alpha.shape[-1]))))
    assert type(new) is type(cde)
    tau = np.linspace(0.05, 0.95, 19)
    exact = stats.norm.ppf(np.sqrt(tau))[None, :] + means[:, None]
    if isinstance(new, calpit.GridCDE):
        quantiles = representations._interp_rows(
            np.tile(tau, (N_OBJECTS, 1)), new.cdf(), np.broadcast_to(Y_GRID, new.pdf.shape)
        )
        atol = 0.01
    else:
        quantile_cde = new.to_quantiles() if isinstance(new, calpit.SampleCDE) else new
        quantiles = representations._interp_rows(
            np.tile(tau, (N_OBJECTS, 1)),
            np.broadcast_to(quantile_cde.levels, quantile_cde.locations.shape),
            quantile_cde.locations,
        )
        # 2000 samples pin the sqrt(0.95) quantile to about 0.06 (one standard error).
        atol = 0.01 if isinstance(new, calpit.QuantileCDE) else 0.3
    np.testing.assert_allclose(quantiles, exact, atol=atol)


def test_identity_recalibration_keeps_quantiles(means) -> None:
    cde = _quantiles(means)
    np.testing.assert_allclose(cde.recalibrate(np.tile(LEVELS, (N_OBJECTS, 1))).locations, cde.locations)


def test_interp_rows_matches_numpy_in_blocks(monkeypatch) -> None:
    rng = np.random.default_rng(4)
    xp = np.sort(rng.uniform(0, 1, (30, 12)), axis=1)
    xp[:, 3] = xp[:, 4]  # A tie.
    fp = np.sort(rng.normal(size=(30, 12)), axis=1)
    x_new = rng.uniform(-0.1, 1.1, (30, 7))
    monkeypatch.setattr(representations, "_MAX_COMPARISONS", 100)
    result = representations._interp_rows(x_new, xp, fp)
    for row in range(30):
        np.testing.assert_allclose(result[row], np.interp(x_new[row], xp[row], fp[row]))


def test_invalid_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="pdf must have shape"):
        calpit.GridCDE(np.zeros((3, 4)), np.zeros(5))
    with pytest.raises(ValueError, match="levels must increase"):
        calpit.QuantileCDE([0.5, 0.2], np.zeros((3, 2)))


def test_qp_round_trip(means) -> None:
    pytest.importorskip("qp")
    from calpit import qp_io  # noqa: PLC0415 - optional dependency.

    grid = _grid(means[:5])
    back = qp_io.from_qp(qp_io.to_qp(grid))
    assert isinstance(back, calpit.GridCDE)
    np.testing.assert_allclose(back.pdf, grid.pdf)

    quantiles = calpit.QuantileCDE(LEVELS[1:-1], stats.norm.ppf(LEVELS[1:-1], loc=means[:5, None]))
    back_quantiles = qp_io.from_qp(qp_io.to_qp(quantiles))
    assert isinstance(back_quantiles, calpit.QuantileCDE)
    np.testing.assert_allclose(
        back_quantiles.locations[:, 1:-1], quantiles.locations
    )  # qp adds levels 0 and 1.

    on_grid = qp_io.from_qp(qp_io.to_qp(quantiles), y_grid=Y_GRID)
    assert isinstance(on_grid, calpit.GridCDE)
    assert on_grid.pdf.shape == (5, len(Y_GRID))
    with pytest.raises(ValueError, match="pass y_grid or quantile_levels"):
        import qp  # noqa: PLC0415 - optional dependency.

        qp_io.from_qp(qp.Ensemble(qp.stats.norm, data={"loc": means[:5, None], "scale": np.ones((5, 1))}))
