"""Networks for Cal-PIT and the contract every Cal-PIT network follows.

A Cal-PIT network models r(alpha; x) = P(PIT <= alpha | x). Its forward takes
the coverage levels and the features as two tensors and returns one value per
row:

    forward(alpha, x) -> output
        alpha: shape (batch,), the coverage levels in [0, 1].
        x: shape (batch, n_features), the features.
        output: shape (batch,), the logit of r, or r itself.

The network says which it returns in its `output` attribute, "logit" (the
default when the attribute is absent) or "probability". A logit is trained
with BCEWithLogitsLoss and can be any real number; a probability must lie in
[0, 1]. A network can also define forward_curves(alpha, x), with alpha of
shape (batch, n_alpha) and output of shape (batch, n_alpha), to predict many
alpha per object at once; calpit.nn.predict_pit_cdf then uses it.
CalPIT.transform needs r to be non-decreasing in alpha. Networks that
are monotone by construction, like IsplineNN and MonotonicNN, give that
exactly; for any other network CalPIT rearranges the predictions
(calpit.coverage.rearrange).
"""

import torch
from torch import nn

from calpit.nn import umnn

OUTPUT_TYPES = ("logit", "probability")


def output_type(model: nn.Module) -> str:
    """Returns what a Cal-PIT network outputs, "logit" or "probability".

    Args:
        model: The network. Its `output` attribute, if any, must be one of
            OUTPUT_TYPES.

    Returns:
        The output type, "logit" when the network has no `output` attribute.
    """
    output = getattr(model, "output", "logit")
    if output not in OUTPUT_TYPES:
        raise ValueError(f"model.output must be one of {OUTPUT_TYPES}: {output=}")
    return output


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        module.bias.data.fill_(0.01)


class ConcatAlpha(nn.Module):
    """Adapts a network that takes one input tensor to the Cal-PIT contract.

    The wrapped network gets alpha and the features joined into one tensor,
    shape (batch, n_features + 1), with alpha in column 0.

    Args:
        network: The network to wrap. Its output must have batch elements,
            such as shape (batch,) or (batch, 1).
        output: What the network returns, "logit" or "probability".

    Attributes:
        network: The wrapped network.
        output: What the network returns.
    """

    def __init__(self, network: nn.Module, output: str = "logit") -> None:
        super().__init__()
        if output not in OUTPUT_TYPES:
            raise ValueError(f"output must be one of {OUTPUT_TYPES}: {output=}")
        self.network = network
        self.output = output

    def forward(self, alpha: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the network on [alpha, x].

        Args:
            alpha: The coverage levels, shape (batch,).
            x: The features, shape (batch, n_features).

        Returns:
            The network's output, shape (batch,).
        """
        return self.network(torch.cat([alpha[:, None], x], dim=1)).reshape(alpha.shape)


class MLP(nn.Module):
    """Multi-layer perceptron of [alpha, x] with PReLU activations.

    The MLP is not monotone in alpha, so CalPIT rearranges its predictions.

    Args:
        n_features: The number of features, not counting alpha.
        hidden_layers: The widths of the hidden layers.
        output: "logit" for a linear output layer, or "probability" to end
            with a sigmoid.

    Attributes:
        layers: The layers, from [alpha, x] to the output.
        output: What the network returns.
    """

    def __init__(self, n_features: int, hidden_layers: list[int], output: str = "logit") -> None:
        super().__init__()
        if output not in OUTPUT_TYPES:
            raise ValueError(f"output must be one of {OUTPUT_TYPES}: {output=}")
        self.output = output
        widths = [n_features + 1, *hidden_layers, 1]
        layers: list[nn.Module] = []
        for width_in, width_out in zip(widths[:-1], widths[1:], strict=True):
            layers.extend([nn.Linear(width_in, width_out), nn.PReLU()])
        layers.pop()
        if output == "probability":
            layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
        self.layers.apply(_init_weights)

    def forward(self, alpha: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the network.

        Args:
            alpha: The coverage levels, shape (batch,).
            x: The features, shape (batch, n_features).

        Returns:
            The logit or probability that the PIT is at most alpha, shape (batch,).
        """
        return self.layers(torch.cat([alpha[:, None], x], dim=1)).reshape(alpha.shape)


class MonotonicNN(nn.Module):
    """Unconstrained monotonic neural network, monotone in alpha by construction.

    The output is offset(x) + scale(x) * integral from 0 to alpha of a positive
    function of (t, x), learned with the UMNN of Wehenkel & Louppe (2019),
    https://arxiv.org/abs/1908.05164.

    Args:
        n_features: The number of features, not counting alpha.
        hidden_layers: The widths of the hidden layers of the integrand and of
            the network that sets the offset and scale.
        nb_steps: The number of integration steps.
        output: "logit" for the integral itself, or "probability" to pass it
            through a sigmoid.

    Attributes:
        network: The UMNN, which takes [alpha, x].
        output: What the network returns.
    """

    def __init__(
        self, n_features: int, hidden_layers: list[int], nb_steps: int = 50, output: str = "logit"
    ) -> None:
        super().__init__()
        if output not in OUTPUT_TYPES:
            raise ValueError(f"output must be one of {OUTPUT_TYPES}: {output=}")
        self.output = output
        self.network = umnn.MonotonicNN(
            n_features + 1, hidden_layers, nb_steps=nb_steps, sigmoid=output == "probability"
        )

    def forward(self, alpha: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the network.

        Args:
            alpha: The coverage levels, shape (batch,).
            x: The features, shape (batch, n_features).

        Returns:
            The logit or probability that the PIT is at most alpha, shape (batch,).
        """
        return self.network(torch.cat([alpha[:, None], x], dim=1)).reshape(alpha.shape)
