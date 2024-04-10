import torch
import numpy as np
from torch.utils.data import Dataset
from prettytable import PrettyTable


class RandomDataset(Dataset):
    """
    A custom dataset class to randomly select a data point.
    The data point is prepended with a random value between
    0 and 1 from a Uniform distribution (coverage parameter).
    The target value is 0 if Y value is less than or equal to
    the coverage parameter and 1 otherwise.
    The data set can be oversampled by a given factor.

    Args:
        X (list or array-like): The input features.
        Y (list or array-like): The target values.
        oversample (float, optional): The oversampling factor. Defaults to 1.

    Returns:
        tuple: A tuple containing the input feature and target value.

    """

    def __init__(self, x_data, y_data, oversample=1):
        self.x_data = x_data
        self.y_data = y_data
        self.len_x = len(x_data)
        self.oversample = oversample

    def __len__(self):
        return int(len(self.x_data) * self.oversample)

    def __getitem__(self, idx):
        alpha = torch.rand(1)
        feature = torch.hstack((alpha, torch.Tensor(self.x_data[idx % self.len_x])))
        target = (self.y_data[idx % self.len_x] <= alpha).float()
        return feature, target


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    Args:
        patience (int): How long to wait after last time validation loss improved.
                        Default: 7
        verbose (bool): If True, prints a message for each validation loss improvement.
                        Default: False
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                        Default: 0
        path (str): Path for the checkpoint to be saved to.
                        Default: 'checkpoint.pt'
        trace_func (function): trace print function.
                        Default: print

    """

    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def cde_loss(cde_estimates: torch.Tensor, y_grid: torch.Tensor, y_test: torch.Tensor) -> tuple:
    """
    Calculates conditional density estimation loss on holdout data.
    This is a PyTorch version of the original function.

    Args:
        cde_estimates (torch.Tensor): An array where each row is a density estimate on y_grid
        y_grid (torch.Tensor): An array of the grid points at which cde_estimates is evaluated.
        y_test (torch.Tensor): An array of the true y values corresponding to the rows of cde_estimates

    Returns:
        tuple: A tuple containing the loss and the standard error of the loss.

    Raises:
        ValueError: If the dimensions of the input tensors are not compatible.

    """
    if len(z_test.shape) == 1:
        z_test = z_test.reshape(-1, 1)
    if len(z_grid.shape) == 1:
        z_grid = z_grid.reshape(-1, 1)

    n_obs, n_grid = cde_estimates.shape
    n_samples, feats_samples = z_test.shape
    n_grid_points, feats_grid = z_grid.shape

    if n_obs != n_samples:
        raise ValueError(
            f"Number of samples in CDEs should be the same as in z_test.Currently {n_obs} and {n_samples}."
        )
    if n_grid != n_grid_points:
        raise ValueError(
            f"Number of grid points in CDEs should be the same as in z_grid. Currently {n_grid} and {n_grid_points}."
        )

    if feats_samples != feats_grid:
        raise ValueError(
            f"Dimensionality of test points and grid points need to coincise. Currently {feats_samples} and {feats_grid}."
        )

    integrals = torch.trapz(cde_estimates**2, torch.squeeze(y_grid), axis=1)

    nn_ids = torch.argmin(torch.abs(y_grid - y_test.T), axis=0)
    likeli = cde_estimates[(tuple(torch.arange(n_samples)), tuple(nn_ids))]

    losses = integrals - 2 * likeli
    loss = torch.mean(losses)
    se_error = torch.std(losses, axis=0) / (n_obs**0.5)

    return loss, se_error
