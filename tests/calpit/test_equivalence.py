"""CalPIT reproduces calpit 0.2 exactly, for every model the old contract allowed.

The old engine is frozen in legacy_calpit. Each model is built once with the
old single-tensor contract and wrapped with ConcatAlpha for CalPIT, so both
start from the same weights and draw the same random numbers.
"""

import copy

import numpy as np
import pytest
import torch
from scipy import stats

import calpit

nn = pytest.importorskip("calpit.nn")
legacy_calpit = pytest.importorskip("legacy_calpit")

SEED = 299792458
N_FEATURES = 3


class _OldMLP(torch.nn.Module):
    """calpit.nn.models.MLP of calpit 0.2, with sigmoid=True and a dropout layer."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(N_FEATURES + 1, 16),
            torch.nn.PReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.layers(feature)


@pytest.fixture(scope="module")
def data() -> dict[str, np.ndarray]:
    x, y = calpit.datasets.TuningFork(dims=N_FEATURES).generate_data(1200)
    y_grid = np.linspace(-15.0, 15.0, 151)
    cde = np.tile(stats.norm.pdf(y_grid, 0.0, 3.0), (len(y), 1))
    return {"x": x, "y": y, "y_grid": y_grid, "cde": cde}


def _old_and_new(old_model: torch.nn.Module, data: dict[str, np.ndarray], **fit_kwargs: int) -> tuple:
    new_model = nn.ConcatAlpha(copy.deepcopy(old_model), output="probability")
    x, y, y_grid, cde = data["x"][:900], data["y"][:900], data["y_grid"], data["cde"][:900]
    pit = calpit.GridCDE(cde, y_grid).pit(y)
    torch.manual_seed(SEED)
    old_curves = legacy_calpit.fit(old_model, x, pit, seed=SEED, **fit_kwargs)
    new = calpit.CalPIT(
        new_model,
        max_epochs=fit_kwargs["n_epochs"],
        patience=fit_kwargs["patience"],
        batch_size=fit_kwargs["batch_size"],
        random_state=SEED,
        rearrange=False,
        trainer_kwargs={"accelerator": "cpu"},
    ).fit(x, y, calpit.GridCDE(cde, y_grid))
    return old_model, old_curves, new


@pytest.mark.parametrize(
    "build",
    [
        _OldMLP,
        lambda: legacy_calpit.MLP(N_FEATURES + 1, [16, 16], sigmoid=True),
        lambda: nn.models.umnn.MonotonicNN(N_FEATURES + 1, [8], sigmoid=True),
    ],
    ids=["mlp-with-dropout", "mlp", "monotonic-nn"],
)
def test_training_matches_calpit_0_2(build, data) -> None:
    torch.manual_seed(0)
    model = build()
    old_model, (old_loss, old_bce), new = _old_and_new(model, data, n_epochs=12, patience=2, batch_size=256)
    np.testing.assert_array_equal(new.train_loss_, old_loss)
    np.testing.assert_array_equal(new.val_bce_, old_bce)
    for old_weight, new_weight in zip(
        old_model.state_dict().values(), new.model_.network.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(new_weight.cpu(), old_weight, rtol=0, atol=0)


def test_predict_and_transform_match_calpit_0_2(data) -> None:
    torch.manual_seed(0)
    old_model = _OldMLP()
    new = calpit.CalPIT.from_fitted(
        nn.ConcatAlpha(copy.deepcopy(old_model), output="probability"), rearrange=False
    )
    x, cde, y_grid = data["x"][900:], data["cde"][900:], data["y_grid"]
    alpha = np.linspace(0.0, 1.0, 21)
    np.testing.assert_array_equal(new.predict(x, alpha), legacy_calpit.predict(old_model, x, alpha))
    np.testing.assert_array_equal(
        new.transform(x, calpit.GridCDE(cde, y_grid)).pdf, legacy_calpit.transform(old_model, x, cde, y_grid)
    )


def test_new_mlp_starts_from_the_old_mlp_weights() -> None:
    torch.manual_seed(0)
    old = legacy_calpit.MLP(N_FEATURES + 1, [16, 8], sigmoid=True)
    torch.manual_seed(0)
    new = nn.MLP(N_FEATURES, [16, 8], output="probability")
    for old_weight, new_weight in zip(old.state_dict().values(), new.state_dict().values(), strict=True):
        torch.testing.assert_close(new_weight, old_weight, rtol=0, atol=0)
    alpha, x = torch.rand(50), torch.randn(50, N_FEATURES)
    torch.testing.assert_close(
        new(alpha, x), old(torch.cat([alpha[:, None], x], dim=1))[:, 0], rtol=0, atol=0
    )
