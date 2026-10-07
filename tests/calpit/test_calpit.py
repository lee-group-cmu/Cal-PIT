"""End-to-end tests of CalPit: fit, predict and transform."""

import dataclasses

import numpy as np
import pytest
import torch
from scipy import integrate, stats

import calpit
from calpit import datasets, metrics
from calpit.nn import models

N_FEATURES = 3


@dataclasses.dataclass(frozen=True)
class MiscalibratedData:
    """Tuning-fork data with one N(0, 3) CDE for every sample, which ignores x."""

    x_calib: np.ndarray
    y_calib: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    y_grid: np.ndarray
    cde_calib: np.ndarray
    cde_test: np.ndarray


@pytest.fixture
def data() -> MiscalibratedData:
    torch.manual_seed(0)
    x, y = datasets.TuningFork(dims=N_FEATURES).generate_data(4000)
    y_grid = np.linspace(-15.0, 15.0, 301)
    cde = np.tile(stats.norm.pdf(y_grid, 0.0, 3.0), (len(y), 1))
    return MiscalibratedData(
        x_calib=x[:3000],
        y_calib=y[:3000],
        x_test=x[3000:],
        y_test=y[3000:],
        y_grid=y_grid,
        cde_calib=cde[:3000],
        cde_test=cde[3000:],
    )


def _fit(model: torch.nn.Module, data: MiscalibratedData, tmp_path, **kwargs) -> calpit.CalPit:
    calpit_model = calpit.CalPit(model)
    calpit_model.fit(
        data.x_calib,
        data.y_calib,
        data.cde_calib,
        data.y_grid,
        trace_func=lambda message: None,
        num_workers=0,
        checkpt_path=tmp_path / "checkpoint.pt",
        **kwargs,
    )
    return calpit_model


def _ks_distance(pit: np.ndarray) -> float:
    return stats.kstest(pit, "uniform").statistic


def test_fit_requires_pit_or_cdes(data, tmp_path) -> None:
    calpit_model = calpit.CalPit(models.MLP(N_FEATURES + 1, [8]))
    with pytest.raises(ValueError, match="pit_calib"):
        calpit_model.fit(data.x_calib, data.y_calib, checkpt_path=tmp_path / "checkpoint.pt")


def test_mlp_fit_predict_transform_shapes(data, tmp_path) -> None:
    calpit_model = _fit(models.MLP(N_FEATURES + 1, [16, 16]), data, tmp_path, n_epochs=2)
    assert calpit_model.training_loss is not None
    assert len(calpit_model.training_loss) == 2
    assert (tmp_path / "checkpoint.pt").exists()

    cov_grid = np.linspace(0, 1, 11)
    local_pit_cdf = calpit_model.predict(data.x_test, cov_grid)
    assert local_pit_cdf.shape == (len(data.x_test), len(cov_grid))
    assert ((local_pit_cdf >= 0) & (local_pit_cdf <= 1)).all()

    cde_new = calpit_model.transform(data.x_test, data.cde_test, data.y_grid)
    assert cde_new.shape == data.cde_test.shape


def test_fit_from_pit_values(data, tmp_path) -> None:
    pit_calib = metrics.probability_integral_transform(data.cde_calib, data.y_grid, data.y_calib)
    calpit_model = calpit.CalPit(models.MLP(N_FEATURES + 1, [8]))
    calpit_model.fit(
        data.x_calib,
        pit_calib=pit_calib,
        n_epochs=1,
        trace_func=lambda message: None,
        num_workers=0,
        checkpt_path=tmp_path / "checkpoint.pt",
    )
    assert calpit_model.predict(data.x_test, np.linspace(0, 1, 5)).shape == (len(data.x_test), 5)


def test_monotonic_nn_runs(data, tmp_path) -> None:
    model = calpit.nn.MonotonicNN(N_FEATURES + 1, [16, 16], sigmoid=True)
    calpit_model = _fit(model, data, tmp_path, n_epochs=1)
    local_pit_cdf = calpit_model.predict(data.x_test, np.linspace(0, 1, 21))
    assert (np.diff(local_pit_cdf, axis=1) >= -1e-5).all()  # float32 rounding


def test_ispline_recalibrates_miscalibrated_cdes(data, tmp_path) -> None:
    pytest.importorskip("splinebasis")
    model = calpit.nn.IsplineNN(N_FEATURES, [64, 64], dropout_p=0.0, num_basis=10)
    calpit_model = _fit(model, data, tmp_path, n_epochs=30, patience=10)

    cde_new = calpit_model.transform(data.x_test, data.cde_test, data.y_grid)
    assert cde_new.shape == data.cde_test.shape
    cde_new = np.clip(cde_new, 0, None)
    cde_new /= integrate.trapezoid(cde_new, data.y_grid, axis=1)[:, None]
    pit_before = metrics.probability_integral_transform(data.cde_test, data.y_grid, data.y_test)
    pit_after = metrics.probability_integral_transform(cde_new, data.y_grid, data.y_test)
    # Over torch seeds 0-3 the KS ratio was 0.32-0.51 and the loss fell by 0.09-0.12.
    assert _ks_distance(pit_after) < 0.7 * _ks_distance(pit_before)
    loss_before, _ = metrics.cde_loss(data.cde_test, data.y_grid, data.y_test)
    loss_after, _ = metrics.cde_loss(cde_new, data.y_grid, data.y_test)
    assert loss_after < loss_before - 0.05


@pytest.mark.xfail(
    strict=True,
    reason=(
        "IsplineNN feeds alpha to the network that sets the I-spline weights, so a trained "
        "model can decrease in alpha (12% of test rows after 30 epochs on the tuning fork)."
    ),
)
def test_ispline_weights_do_not_depend_on_alpha() -> None:
    pytest.importorskip("splinebasis")
    torch.manual_seed(0)
    model = calpit.nn.IsplineNN(N_FEATURES, [16], dropout_p=0.0).eval()
    features = torch.randn(10, N_FEATURES)
    with torch.no_grad():
        weights = [
            model.spline_layer.coefs(model.mlp_layers(torch.hstack([torch.full((10, 1), alpha), features])))
            for alpha in (0.1, 0.9)
        ]
    torch.testing.assert_close(weights[0], weights[1])
