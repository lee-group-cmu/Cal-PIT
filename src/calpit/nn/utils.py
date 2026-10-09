"""PyTorch versions of calpit's numerical helpers."""

import torch


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
    if len(y_test.shape) == 1:
        y_test = y_test.reshape(-1, 1)
    if len(y_grid.shape) == 1:
        y_grid = y_grid.reshape(-1, 1)

    n_obs, n_grid = cde_estimates.shape
    n_samples, feats_samples = y_test.shape
    n_grid_points, feats_grid = y_grid.shape

    if n_obs != n_samples:
        raise ValueError(
            f"Number of samples in CDEs should be the same as in y_test. Currently {n_obs} and {n_samples}."
        )
    if n_grid != n_grid_points:
        raise ValueError(
            "Number of grid points in CDEs should be the same as in y_grid. "
            f"Currently {n_grid} and {n_grid_points}."
        )

    if feats_samples != feats_grid:
        raise ValueError(
            "Dimensionality of test points and grid points need to coincide. "
            f"Currently {feats_samples} and {feats_grid}."
        )

    integrals = torch.trapezoid(cde_estimates**2, torch.squeeze(y_grid), dim=1)

    nn_ids = torch.argmin(torch.abs(y_grid - y_test.T), dim=0)
    likeli = cde_estimates[torch.arange(n_samples), nn_ids]

    losses = integrals - 2 * likeli
    loss = torch.mean(losses)
    # correction=0 matches np.std in calpit.metrics.cde_loss.
    se_error = torch.std(losses, dim=0, correction=0) / (n_obs**0.5)

    return loss, se_error


def trapz_grid_torch(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Does trapezoid integration between the same limits as the grid, in PyTorch.

    Args:
        y (torch.Tensor): The values to integrate, shape (n_rows, n_grid).
        x (torch.Tensor): The grid points, shape (n_grid,).

    Returns:
        torch.Tensor: The integrals from x[0] to each grid point, shape (n_rows, n_grid).
    """
    dx = torch.diff(x)
    trapz_area = dx * (y[:, 1:] + y[:, :-1]) / 2
    integral = torch.cumsum(trapz_area, dim=-1)
    zeros = torch.zeros(len(integral), dtype=integral.dtype, device=x.device)
    return torch.hstack((zeros[:, None], integral))
