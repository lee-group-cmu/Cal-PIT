"""The training and recalibration loop of calpit 0.2 (commit b4deaab), frozen.

The equivalence tests check that CalPIT reproduces it exactly. Only printing
and checkpoint files were removed; the arithmetic, the order of random draws
and the batching are as they were. Models take [alpha, x] in one tensor.
"""

import copy

import numpy as np
import torch
from scipy import interpolate
from torch.utils import data

from calpit import utils


class MLP(torch.nn.Module):
    """calpit.nn.models.MLP of calpit 0.2."""

    def __init__(
        self, input_dim: int, hidden_layers: list[int], output_dim: int = 1, sigmoid: bool = True
    ) -> None:
        super().__init__()
        self.all_layers = [input_dim]
        self.all_layers.extend(hidden_layers)
        self.all_layers.append(output_dim)
        self.layer_list: list[torch.nn.Module] = []
        for i in range(len(self.all_layers) - 1):
            self.layer_list.append(torch.nn.Linear(self.all_layers[i], self.all_layers[i + 1]))
            self.layer_list.append(torch.nn.PReLU())
        self.layer_list.pop()
        if sigmoid:
            self.layer_list.append(torch.nn.Sigmoid())
        self.layers = torch.nn.Sequential(*self.layer_list)
        self.layers.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the network on [alpha, x]."""
        return self.layers(x)


def _init_weights(module: torch.nn.Module) -> None:
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(module.weight)
        module.bias.data.fill_(0.01)


class RandomDataset(data.Dataset):
    """calpit.nn.utils.RandomDataset of calpit 0.2."""

    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, oversample: float = 1) -> None:
        self.x_data = x_data
        self.y_data = y_data
        self.len_x = len(x_data)
        self.oversample = oversample

    def __len__(self) -> int:
        return int(len(self.x_data) * self.oversample)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.rand(1)
        feature = torch.hstack((alpha, torch.Tensor(self.x_data[idx % self.len_x])))
        target = (self.y_data[idx % self.len_x] <= alpha).float()
        return feature, target


def fit(
    model: torch.nn.Module,
    x_calib: np.ndarray,
    pit_calib: np.ndarray,
    *,
    oversample: float = 1,
    n_cov_val: int = 201,
    patience: int = 20,
    n_epochs: int = 1000,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    batch_size: int = 2048,
    frac_train: float = 0.9,
    lr_decay: float = 0.99,
    seed: int = 299792458,
) -> tuple[list[float], list[float]]:
    """CalPit.fit of calpit 0.2; trains model in place and returns the loss curves."""
    device = next(model.parameters()).device
    cov_grid = np.linspace(0.001, 0.999, n_cov_val)
    train_size = int(frac_train * len(x_calib))
    valid_size = len(x_calib) - train_size
    rnd_idx = np.random.default_rng(seed=seed).permutation(len(x_calib))
    x_train = x_calib[rnd_idx[:train_size]]
    x_val = x_calib[rnd_idx[train_size:]]
    pit_train = pit_calib[rnd_idx[:train_size]]
    pit_val = pit_calib[rnd_idx[train_size:]]
    trainset = RandomDataset(x_train, pit_train, oversample=oversample)
    feature_val = torch.cat(
        [
            torch.Tensor(np.repeat(cov_grid, len(x_val)))[:, None],
            torch.Tensor(np.tile(x_val, (n_cov_val, 1))),
        ],
        dim=-1,
    )
    target_val = torch.Tensor(np.tile(pit_val, n_cov_val) <= np.repeat(cov_grid, len(x_val))).float()[:, None]
    validset = data.TensorDataset(feature_val, target_val)
    train_dataloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_dataloader = data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_decay**epoch)
    training_loss, validation_bce = [], []
    best_score, best_state, counter = None, None, 0
    for _ in range(n_epochs):
        training_loss_batch, validation_bce_batch = [], []
        model.train()
        for feature, target in train_dataloader:
            feature, target = feature.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(feature.float())
            loss_fn = torch.nn.BCELoss(reduction="sum")
            loss = loss_fn(torch.clamp(torch.squeeze(output), min=0.0, max=1.0), torch.squeeze(target))
            loss.backward()
            optimizer.step()
            training_loss_batch.append(loss.item())
        model.eval()
        for feature, target in valid_dataloader:
            feature, target = feature.to(device), target.to(device)
            output = model(feature.float())
            criterion = torch.nn.BCELoss(reduction="sum")
            bce = criterion(torch.clamp(torch.squeeze(output), min=0, max=1), torch.squeeze(target))
            validation_bce_batch.append(bce.item())
        training_loss.append(np.sum(training_loss_batch) / (train_size * oversample))
        validation_bce.append(np.sum(validation_bce_batch) / (valid_size * n_cov_val))
        scheduler.step()
        score = -validation_bce[-1]
        if best_score is None or score >= best_score:
            best_score, best_state, counter = score, copy.deepcopy(model.state_dict()), 0
        else:
            counter += 1
            if counter >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return training_loss, validation_bce


def predict(
    model: torch.nn.Module, x_test: np.ndarray, cov_grid: np.ndarray, batch_size: int = 2048
) -> np.ndarray:
    """CalPit.predict of calpit 0.2."""
    model.eval()
    device = next(model.parameters()).device
    pred_pit = []
    n_cov = cov_grid.shape[-1]
    for i in range((len(x_test) - 1) // batch_size + 1):
        x = x_test[i * batch_size : (i + 1) * batch_size]
        with torch.no_grad():
            if cov_grid.ndim == 1:
                features = np.hstack([np.repeat(cov_grid, len(x))[:, None], np.tile(x, (n_cov, 1))])
                batch = model(torch.Tensor(features).to(device)).cpu().numpy().reshape(n_cov, -1).T
            else:
                c = cov_grid[i * batch_size : (i + 1) * batch_size]
                features = np.hstack([np.ravel(c)[:, None], np.repeat(x, c.shape[1], axis=0)])
                batch = model(torch.Tensor(features).to(device)).cpu().numpy().reshape(len(x), -1)
        batch[batch < 0] = 0
        batch[batch > 1] = 1
        pred_pit.extend(batch)
    return np.array(pred_pit)


def transform(
    model: torch.nn.Module, x_test: np.ndarray, cde_test: np.ndarray, y_grid: np.ndarray
) -> np.ndarray:
    """CalPit.transform of calpit 0.2."""
    cdf_test = utils.trapz_grid(cde_test, y_grid)
    cdf_test_new = predict(model, x_test, cov_grid=cdf_test)
    cdf_funct = interpolate.PchipInterpolator(y_grid, cdf_test_new, extrapolate=True, axis=1)
    return cdf_funct.derivative(1)(y_grid)
