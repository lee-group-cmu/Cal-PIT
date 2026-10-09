"""Loss and prediction for Cal-PIT networks, usable in any training loop."""

import numpy as np
import numpy.typing as npt
import torch
from torch.nn import functional

from calpit.nn import models


def coverage_loss(
    output: torch.Tensor, target: torch.Tensor, output_type: str = "logit", reduction: str = "mean"
) -> torch.Tensor:
    """Binary cross entropy of a Cal-PIT network's output against 1{PIT <= alpha}.

    Args:
        output: The network's output, shape (batch,).
        target: The targets, 1.0 if the PIT is at most alpha and 0.0 otherwise,
            shape (batch,).
        output_type: What the network returns, "logit" or "probability"
            (calpit.nn.output_type gives it). Probabilities are clamped to
            [0, 1] first.
        reduction: "mean", "sum" or "none", as in torch.nn.BCELoss.

    Returns:
        The loss, a scalar unless reduction is "none".
    """
    output = output.reshape(-1)
    target = target.reshape(-1).to(output.dtype)
    if output_type == "logit":
        return functional.binary_cross_entropy_with_logits(output, target, reduction=reduction)
    if output_type == "probability":
        return functional.binary_cross_entropy(
            torch.clamp(output, min=0.0, max=1.0), target, reduction=reduction
        )
    raise ValueError(f"output_type must be one of {models.OUTPUT_TYPES}: {output_type=}")


def _device_of(model: torch.nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    return torch.device("cpu") if parameter is None else parameter.device


def predict_pit_cdf(
    model: torch.nn.Module,
    x: npt.ArrayLike,
    alpha: npt.ArrayLike,
    batch_size: int = 2048,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Predicts r(alpha; x) = P(PIT <= alpha | x) with a Cal-PIT network.

    The network is evaluated in eval mode without gradients, and restored to
    the mode it was in. Logits go through a sigmoid; the results are clipped
    to [0, 1]. They are not rearranged (see calpit.coverage.rearrange).

    Args:
        model: The network, which follows the contract in calpit.nn.models.
        x: The features, shape (n_objects, n_features).
        alpha: The coverage levels, shape (n_alpha,) to use the same levels for
            every object, or (n_objects, n_alpha).
        batch_size: The number of objects per forward pass; each pass has
            batch_size * n_alpha rows.
        device: The device to evaluate on; the model is moved there. None
            uses the device of the model's parameters.

    Returns:
        The predicted PIT CDF, shape (n_objects, n_alpha).
    """
    x = np.asarray(x)
    alpha = np.asarray(alpha)
    device = _device_of(model) if device is None else torch.device(device)
    model.to(device)
    output_type = models.output_type(model)
    was_training = model.training
    model.eval()
    batches = []
    try:
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                x_batch = x[start : start + batch_size]
                if alpha.ndim == 1:
                    # Rows run alpha-major: every object at alpha[0], then at alpha[1], ...
                    alpha_rows = np.repeat(alpha, len(x_batch))
                    x_rows = np.tile(x_batch, (len(alpha), 1))
                else:
                    alpha_batch = alpha[start : start + batch_size]
                    alpha_rows = np.ravel(alpha_batch)
                    x_rows = np.repeat(x_batch, alpha_batch.shape[1], axis=0)
                output = model(
                    torch.as_tensor(alpha_rows, dtype=torch.float32, device=device),
                    torch.as_tensor(x_rows, dtype=torch.float32, device=device),
                )
                if output_type == "logit":
                    output = torch.sigmoid(output)
                output = output.cpu().numpy().astype(np.float64)
                if alpha.ndim == 1:
                    batches.append(output.reshape(len(alpha), -1).T)
                else:
                    batches.append(output.reshape(len(x_batch), -1))
    finally:
        model.train(was_training)
    pit_cdf = np.concatenate(batches) if batches else np.empty((0, alpha.shape[-1]))
    return np.clip(pit_cdf, 0.0, 1.0)
