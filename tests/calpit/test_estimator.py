"""CalPIT: the scikit-learn API, the PyTorch and scikit-learn backends, recalibration."""

import dataclasses
import functools
import pickle

import numpy as np
import pytest
from scipy import stats

import calpit

N_FEATURES = 3


@dataclasses.dataclass(frozen=True)
class MiscalibratedData:
    """Tuning-fork data with one N(0, 3) CDE for every sample, which ignores x."""

    x_calib: np.ndarray
    y_calib: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    y_grid: np.ndarray

    def cde(self, n_objects: int) -> calpit.GridCDE:
        """Returns the first n_objects CDEs."""
        return calpit.GridCDE(np.tile(stats.norm.pdf(self.y_grid, 0.0, 3.0), (n_objects, 1)), self.y_grid)


@pytest.fixture(scope="module")
def data() -> MiscalibratedData:
    x, y = calpit.datasets.TuningFork(dims=N_FEATURES).generate_data(4000)
    return MiscalibratedData(
        x_calib=x[:3000],
        y_calib=y[:3000],
        x_test=x[3000:],
        y_test=y[3000:],
        y_grid=np.linspace(-15.0, 15.0, 301),
    )


def _ks_distance(pit: np.ndarray) -> float:
    return stats.kstest(pit, "uniform").statistic


def test_constructor_only_stores_arguments() -> None:
    estimator = calpit.CalPIT(rearrange=False, max_epochs=3)
    assert estimator.get_params()["rearrange"] is False
    assert estimator.get_params()["max_epochs"] == 3
    assert not hasattr(estimator, "model_")
    assert repr(estimator) == "CalPIT(rearrange=False, max_epochs=3)"


def test_set_params_and_sklearn_clone() -> None:
    base = pytest.importorskip("sklearn.base")
    estimator = calpit.CalPIT(patience=5).set_params(lr=0.01)
    clone = base.clone(estimator)
    assert clone.get_params() == estimator.get_params()
    with pytest.raises(ValueError, match="invalid parameter"):
        estimator.set_params(learning_rate=0.1)


def test_predict_before_fit_raises() -> None:
    with pytest.raises(calpit.NotFittedError):
        calpit.CalPIT().predict(np.zeros((2, N_FEATURES)))


def test_fit_requires_pit_or_cdes(data) -> None:
    with pytest.raises(ValueError, match="pass either pit"):
        calpit.CalPIT().fit(data.x_calib, data.y_calib)


def test_torch_fit_predict_transform(data) -> None:
    nn = pytest.importorskip("calpit.nn")
    model = nn.MLP(N_FEATURES, [16, 16])
    estimator = calpit.CalPIT(model, max_epochs=2)
    assert estimator.fit(data.x_calib, data.y_calib, data.cde(3000)) is estimator
    assert estimator.backend_ == "torch"
    assert estimator.n_features_in_ == N_FEATURES
    assert len(estimator.train_loss_) == len(estimator.val_bce_) == 2
    assert estimator.best_val_bce_ == estimator.val_bce_.min()
    assert estimator.model_ is not model  # The model passed in is copied, not trained.

    alpha = np.linspace(0, 1, 11)
    pit_cdf = estimator.predict(data.x_test, alpha)
    assert pit_cdf.shape == (len(data.x_test), len(alpha))
    assert ((pit_cdf >= 0) & (pit_cdf <= 1)).all()
    assert (np.diff(pit_cdf, axis=1) >= 0).all()  # Rearranged by default.
    assert estimator.transform(data.x_test, data.cde(1000)).pdf.shape == (1000, len(data.y_grid))


def test_fit_from_pit_values_without_validation(data) -> None:
    nn = pytest.importorskip("calpit.nn")
    pit = data.cde(3000).pit(data.y_calib)
    estimator = calpit.CalPIT(nn.MLP(N_FEATURES, [8]), max_epochs=2, val_fraction=0).fit(
        data.x_calib, pit=pit
    )
    assert len(estimator.val_bce_) == 0
    assert estimator.best_val_bce_ is None
    assert estimator.predict(data.x_test, np.linspace(0, 1, 5)).shape == (len(data.x_test), 5)


def test_random_state_makes_training_reproducible(data) -> None:
    nn = pytest.importorskip("calpit.nn")
    factory = functools.partial(nn.MLP, hidden_layers=[8])
    fits = [
        calpit.CalPIT(factory, max_epochs=2).fit(data.x_calib, data.y_calib, data.cde(3000)) for _ in range(2)
    ]
    np.testing.assert_array_equal(fits[0].train_loss_, fits[1].train_loss_)
    np.testing.assert_array_equal(fits[0].predict(data.x_test), fits[1].predict(data.x_test))


def test_rearrange_switch(data) -> None:
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("calpit.nn")

    class Wiggly(torch.nn.Module):  # type: ignore[name-defined] # torch comes from importorskip.
        def forward(self, alpha: torch.Tensor, x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
            return 4 * torch.sin(8 * alpha) + 0 * x[:, 0]

    alpha = np.linspace(0, 1, 51)
    raw = calpit.CalPIT.from_fitted(Wiggly(), rearrange=False).predict(data.x_test[:10], alpha)
    sorted_ = calpit.CalPIT.from_fitted(Wiggly()).predict(data.x_test[:10], alpha)
    assert (np.diff(raw, axis=1) < 0).any()
    np.testing.assert_array_equal(sorted_, np.sort(raw, axis=1))
    expected = 1 / (1 + np.exp(-4 * np.sin(8 * alpha.astype(np.float32))))
    np.testing.assert_allclose(raw[0], expected, rtol=1e-6)
    assert nn.output_type(Wiggly()) == "logit"


def test_ispline_is_monotone_in_alpha() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("splinebasis")
    nn = pytest.importorskip("calpit.nn")
    torch.manual_seed(0)
    model = nn.IsplineNN(N_FEATURES, [16], dropout_p=0.0).eval()
    features = torch.randn(10, N_FEATURES)
    with torch.no_grad():
        weights = [model.spline_layer.coefs(model.mlp_layers(features)) for _ in (0.1, 0.9)]
        curve = torch.stack([model(torch.full((10,), alpha), features) for alpha in np.linspace(0, 1, 101)])
    torch.testing.assert_close(weights[0], weights[1])
    assert (torch.diff(curve, dim=0) >= 0).all()
    torch.testing.assert_close(curve[0], torch.zeros(10), atol=1e-6, rtol=0)
    torch.testing.assert_close(curve[-1], torch.ones(10), atol=1e-6, rtol=0)


def test_default_ispline_recalibrates_miscalibrated_cdes(data) -> None:
    pytest.importorskip("splinebasis")
    nn = pytest.importorskip("calpit.nn")
    estimator = calpit.CalPIT(
        functools.partial(nn.IsplineNN, hidden_layers=[64, 64], dropout_p=0.0), max_epochs=30, patience=10
    ).fit(data.x_calib, data.y_calib, data.cde(3000))
    new = estimator.transform(data.x_test, data.cde(1000))
    new_pdf = calpit.utils.normalize(np.clip(new.pdf, 0, None), data.y_grid)
    pit_before = data.cde(1000).pit(data.y_test)
    pit_after = calpit.GridCDE(new_pdf, data.y_grid).pit(data.y_test)
    assert _ks_distance(pit_after) < 0.7 * _ks_distance(pit_before)
    loss_before, _ = calpit.metrics.cde_loss(data.cde(1000).pdf, data.y_grid, data.y_test)
    loss_after, _ = calpit.metrics.cde_loss(new_pdf, data.y_grid, data.y_test)
    assert loss_after < loss_before - 0.05


def test_sklearn_backend_recalibrates_and_is_monotone(data) -> None:
    ensemble = pytest.importorskip("sklearn.ensemble")
    classifier = ensemble.HistGradientBoostingClassifier(max_iter=100)
    estimator = calpit.CalPIT(classifier, n_alpha=20).fit(data.x_calib, data.y_calib, data.cde(3000))
    assert estimator.backend_ == "sklearn"
    assert estimator.model_.monotonic_cst == [1, 0, 0, 0]
    assert classifier.monotonic_cst is None  # The classifier passed in is cloned, not changed.
    raw = calpit.CalPIT.from_fitted(estimator.model_, rearrange=False, n_alpha=20).predict(data.x_test)
    assert (np.diff(raw, axis=1) >= -1e-12).all()  # Monotone up to rounding.
    pit_before = data.cde(1000).pit(data.y_test)
    pit_after = estimator.transform(data.x_test, data.cde(1000)).pit(data.y_test)
    assert _ks_distance(pit_after) < 0.5 * _ks_distance(pit_before)


def test_pickle_round_trip(data) -> None:
    nn = pytest.importorskip("calpit.nn")
    estimator = calpit.CalPIT(nn.MLP(N_FEATURES, [8]), max_epochs=1).fit(
        data.x_calib, data.y_calib, data.cde(3000)
    )
    restored = pickle.loads(pickle.dumps(estimator))
    np.testing.assert_array_equal(restored.predict(data.x_test), estimator.predict(data.x_test))


def test_diagnose(data) -> None:
    nn = pytest.importorskip("calpit.nn")
    estimator = calpit.CalPIT(nn.MLP(N_FEATURES, [8]), max_epochs=1).fit(
        data.x_calib, data.y_calib, data.cde(3000)
    )
    local = estimator.diagnose(data.x_test[:7])
    assert local.pit_cdf.shape == (7, 101)
    assert local.ks.shape == local.cvm.shape == local.coverage(0.9).shape == (7,)
    perfect = calpit.diagnostics.LocalCalibration(alpha=local.alpha, pit_cdf=np.tile(local.alpha, (3, 1)))
    np.testing.assert_allclose(perfect.ks, 0)
    np.testing.assert_allclose(perfect.coverage(0.9), 0.9)
