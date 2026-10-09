"""The PyTorch components, used on their own as in a hand-written training loop."""

import numpy as np
import pytest
from scipy import stats

import calpit

torch = pytest.importorskip("torch")
nn = pytest.importorskip("calpit.nn")

N_FEATURES = 3


def test_coverage_dataset_items() -> None:
    torch.manual_seed(0)
    dataset = nn.CoverageDataset(np.ones((5, N_FEATURES)), np.linspace(0, 1, 5), oversample=2)
    assert len(dataset) == 10
    alpha, x, target = dataset[7]
    assert alpha.shape == target.shape == ()
    assert x.shape == (N_FEATURES,)
    assert target.item() == float(alpha.item() >= 0.5)


def test_coverage_grid_dataset_matches_the_precomputed_tensors() -> None:
    rng = np.random.default_rng(0)
    x, pit = rng.normal(size=(37, N_FEATURES)), rng.uniform(size=37)
    alpha = np.linspace(0.001, 0.999, 201)
    dataset = nn.CoverageGridDataset(x, pit, alpha, batch_size=100)
    batches = list(dataset)
    assert len(batches) == len(dataset) == -(-37 * 201 // 100)
    # calpit 0.2 built every row up front, alpha-major.
    alpha_rows = torch.Tensor(np.repeat(alpha, 37))
    x_rows = torch.Tensor(np.tile(x, (201, 1)))
    target_rows = torch.Tensor(np.tile(pit, 201) <= np.repeat(alpha, 37))
    for rows, expected in zip(zip(*batches, strict=True), (alpha_rows, x_rows, target_rows), strict=True):
        torch.testing.assert_close(torch.cat(rows), expected, rtol=0, atol=0)


def test_coverage_loss_of_logits_and_probabilities_agree() -> None:
    logits = torch.randn(64)
    target = (torch.rand(64) < 0.5).float()
    torch.testing.assert_close(
        nn.coverage_loss(logits, target, "logit"),
        nn.coverage_loss(torch.sigmoid(logits), target, "probability"),
    )
    assert nn.coverage_loss(logits, target, reduction="none").shape == (64,)
    with pytest.raises(ValueError, match="output_type"):
        nn.coverage_loss(logits, target, "quantile")


def test_concat_alpha_and_output_type() -> None:
    network = torch.nn.Linear(N_FEATURES + 1, 1)
    model = nn.ConcatAlpha(network, output="probability")
    alpha, x = torch.rand(8), torch.randn(8, N_FEATURES)
    torch.testing.assert_close(model(alpha, x), network(torch.cat([alpha[:, None], x], dim=1))[:, 0])
    assert nn.output_type(model) == "probability"
    assert nn.output_type(nn.MLP(N_FEATURES, [4])) == "logit"
    with pytest.raises(ValueError, match="output must be one of"):
        nn.ConcatAlpha(network, output="odds")


def test_predict_pit_cdf_shapes_and_mode() -> None:
    model = nn.MLP(N_FEATURES, [8]).train()
    x = np.random.default_rng(0).normal(size=(10, N_FEATURES))
    shared = nn.predict_pit_cdf(model, x, np.linspace(0, 1, 5), batch_size=3)
    per_object = nn.predict_pit_cdf(model, x, np.tile(np.linspace(0, 1, 5), (10, 1)), batch_size=4)
    assert shared.shape == (10, 5)
    np.testing.assert_allclose(shared, per_object, rtol=1e-6)
    assert model.training  # The mode is restored.


def test_trapz_grid_torch_matches_numpy(gaussian_cdes) -> None:
    expected = calpit.utils.trapz_grid(gaussian_cdes.cde, gaussian_cdes.y_grid)
    result = nn.trapz_grid_torch(torch.tensor(gaussian_cdes.cde), torch.tensor(gaussian_cdes.y_grid))
    assert result.dtype == torch.float64
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-12)


def test_hand_written_training_loop() -> None:
    """The example of docs/training_loop.rst."""
    x_all, y_all = calpit.datasets.TuningFork(dims=N_FEATURES).generate_data(4000)
    y_grid = np.linspace(-15.0, 15.0, 301)
    cde_all = np.tile(stats.norm.pdf(y_grid, 0.0, 3.0), (len(y_all), 1))
    x_calib, y_calib, cde_calib = x_all[:3000], y_all[:3000], cde_all[:3000]
    x_test, y_test, cde_test = x_all[3000:], y_all[3000:], cde_all[3000:]
    torch.manual_seed(0)

    pit = calpit.GridCDE(cde_calib, y_grid).pit(y_calib)
    x_train, pit_train, x_val, pit_val = calpit.coverage.train_val_split(
        x_calib, pit, val_fraction=0.1, random_state=0
    )
    train_loader = nn.CoverageDataset(x_train, pit_train).batches(batch_size=256)
    val_set = nn.CoverageGridDataset(x_val, pit_val, alpha=np.linspace(0.001, 0.999, 101))

    model = nn.MonotonicNN(N_FEATURES, [32, 32])
    output_type = nn.output_type(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    val_losses = []
    for _ in range(8):
        model.train()
        for alpha, x, target in train_loader:
            optimizer.zero_grad()
            loss = nn.coverage_loss(model(alpha, x), target, output_type)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = sum(
                nn.coverage_loss(model(alpha, x), target, output_type, reduction="sum").item()
                for alpha, x, target in val_set
            )
        val_losses.append(val_loss / val_set.n_rows())

    recalibrator = calpit.CalPIT.from_fitted(model)
    cde_new = recalibrator.transform(x_test, calpit.GridCDE(cde_test, y_grid))

    assert val_losses[-1] < val_losses[0]
    pit_before = calpit.GridCDE(cde_test, y_grid).pit(y_test)
    pit_after = cde_new.pit(y_test)
    assert stats.kstest(pit_after, "uniform").statistic < 0.7 * stats.kstest(pit_before, "uniform").statistic


def test_lightning_module_with_own_trainer() -> None:
    lightning = pytest.importorskip("lightning")
    from torch.utils import data  # noqa: PLC0415 - kept next to the example it mirrors.

    from calpit.nn import lightning as calpit_lightning  # noqa: PLC0415 - needs lightning.

    rng = np.random.default_rng(0)
    x, pit = rng.normal(size=(500, N_FEATURES)), rng.uniform(size=500)
    module = calpit_lightning.CalPITModule(nn.MLP(N_FEATURES, [8]), lr=1e-2)
    early_stopping = calpit_lightning.BestWeightsEarlyStopping(patience=1)
    trainer = lightning.Trainer(
        max_epochs=20,
        accelerator="cpu",
        callbacks=[early_stopping],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(
        module,
        data.DataLoader(nn.CoverageDataset(x[:400], pit[:400]), batch_size=64, shuffle=True),
        data.DataLoader(
            nn.CoverageGridDataset(x[400:], pit[400:], np.linspace(0.01, 0.99, 11)), batch_size=None
        ),
    )
    assert len(module.val_bce_history) == len(module.train_loss_history) >= 2
    assert early_stopping.best_score == min(module.val_bce_history)
    assert calpit.CalPIT.from_fitted(module.model).predict(x[:3]).shape == (3, 101)


def test_photometry_dataset(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    features = np.arange(12.0).reshape(4, N_FEATURES)
    with h5py.File(tmp_path / "photometry.hdf5", "w") as catalog:
        catalog["dered_color_features"] = features
    dataset = nn.PhotometryDataset(tmp_path / "photometry.hdf5", pit=np.full(4, 0.5))
    assert len(dataset) == 4
    _, x, _ = dataset[2]
    np.testing.assert_array_equal(x.numpy(), features[2])
    dataset.close()


def test_probability_loss_works_under_autocast() -> None:
    probability, target = torch.rand(16), (torch.rand(16) < 0.5).float()
    expected = nn.coverage_loss(probability, target, "probability")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = nn.coverage_loss(probability.bfloat16(), target, "probability")
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, expected, rtol=1e-2, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_default_model_trains_with_mixed_precision() -> None:
    pytest.importorskip("splinebasis")
    x, y = calpit.datasets.TuningFork(dims=N_FEATURES).generate_data(500)
    y_grid = np.linspace(-15.0, 15.0, 101)
    cde = calpit.GridCDE(np.tile(stats.norm.pdf(y_grid, 0.0, 3.0), (len(y), 1)), y_grid)
    estimator = calpit.CalPIT(
        max_epochs=1, trainer_kwargs={"accelerator": "gpu", "precision": "16-mixed"}
    ).fit(x, y, cde)
    assert np.isfinite(estimator.train_loss_).all()


def test_ispline_forward_curves_matches_forward() -> None:
    pytest.importorskip("splinebasis")
    torch.manual_seed(0)
    model = nn.IsplineNN(N_FEATURES, [16])
    x = np.random.default_rng(0).normal(size=(9, N_FEATURES))
    alpha = np.random.default_rng(1).uniform(size=(9, 7))
    curves = nn.predict_pit_cdf(model, x, alpha, batch_size=4)
    model.eval()
    with torch.no_grad():
        rows = model(
            torch.as_tensor(alpha.ravel(), dtype=torch.float32),
            torch.as_tensor(np.repeat(x, 7, axis=0), dtype=torch.float32),
        )
    # Equal up to float32 rounding: the I-spline sums run over differently shaped tensors.
    np.testing.assert_allclose(curves, rows.numpy().astype(np.float64).reshape(9, 7), rtol=1e-6)


def test_batched_loader_matches_the_item_loader() -> None:
    from torch.utils import data  # noqa: PLC0415 - kept next to the comparison it makes.

    rng = np.random.default_rng(0)
    dataset = nn.CoverageDataset(rng.normal(size=(300, N_FEATURES)), rng.uniform(size=300), oversample=2)
    torch.manual_seed(5)
    items = list(data.DataLoader(dataset, batch_size=64, shuffle=True))
    torch.manual_seed(5)
    batches = list(dataset.batches(64))
    assert len(items) == len(batches)
    for item_batch, batch in zip(items, batches, strict=True):
        for expected, actual in zip(item_batch, batch, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_distributed_training_stops_together(tmp_path) -> None:
    """Two CPU processes must agree on the validation loss and the epoch to stop at."""
    pytest.importorskip("lightning")
    import subprocess  # noqa: PLC0415 - only this test launches processes.
    import sys  # noqa: PLC0415 - only this test launches processes.

    script = tmp_path / "ddp.py"
    script.write_text(
        "import os\n"
        "import numpy as np\n"
        "from scipy import stats\n"
        "import calpit, calpit.nn\n"
        "if __name__ == '__main__':\n"
        "    x, y = calpit.datasets.TuningFork(dims=3).generate_data(1000)\n"
        "    g = np.linspace(-15, 15, 101)\n"
        "    cde = calpit.GridCDE(np.tile(stats.norm.pdf(g, 0, 3), (len(y), 1)), g)\n"
        "    est = calpit.CalPIT(calpit.nn.MLP(3, [8]), max_epochs=200, patience=2, batch_size=128,\n"
        "        lr=0.05, lr_decay=0.9, n_alpha_val=21,\n"
        "        trainer_kwargs={'accelerator': 'cpu', 'devices': 2, 'strategy': 'ddp'}).fit(x, y, cde)\n"
        "    print('RESULT', len(est.val_bce_), repr(est.val_bce_[-1]), flush=True)\n"
    )
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=300, check=True, cwd=tmp_path
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("RESULT")]
    assert len(lines) == 2
    assert lines[0] == lines[1]
    assert int(lines[0].split()[1]) < 200
