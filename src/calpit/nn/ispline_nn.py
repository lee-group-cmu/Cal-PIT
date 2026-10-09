"""IsplineNN: a Cal-PIT network that is monotone in alpha by construction."""

import numpy as np
import torch
import torch.nn as nn


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        module.bias.data.fill_(0.01)


class ISplineLayer(nn.Module):
    """
    Output layer that is a monotone function of alpha built from I-spline basis functions.

    The output is a convex combination of cubic I-spline basis functions of alpha on [0, 1],
    with weights from a softmax over a linear map of the input. Each basis function increases
    from 0 to 1, so the output is non-decreasing in alpha and lies in [0, 1].

    Args:
        in_features (int): The number of input features.
        num_basis (int): The number of I-spline basis functions.
        dropout_p (float, optional): The dropout probability applied to the weights.
            Defaults to 0.

    Raises:
        ImportError: If spline-basis is not installed.
    """

    def __init__(self, in_features, num_basis, dropout_p=0):
        super().__init__()
        self.in_features = in_features
        self.num_basis = num_basis
        self.coefs = nn.Sequential(
            nn.Linear(in_features, num_basis), nn.Softmax(dim=-1), nn.Dropout(p=dropout_p)
        )
        # Imported here so that calpit installs and imports without the optional
        # spline-basis dependency; only IsplineNN needs it.
        try:
            import splinebasis  # noqa: PLC0415 - optional dependency.
        except ImportError as error:
            raise ImportError(
                "IsplineNN requires the optional dependency spline-basis. "
                "Install it with: pip install 'calpit[spline]'"
            ) from error
        self.grid = torch.linspace(0, 1, 1000)
        # spline-basis 0.1 returns a NumPy array; 0.2 follows the grid's backend and may
        # return a view with negative strides, so pass a NumPy grid and copy the result.
        basis_vectors = splinebasis.ISplineBasis(
            order=3, num_basis=num_basis, lower=0, upper=1, grid=self.grid.numpy().astype(np.float64)
        ).basis_vectors
        self.basis_vectors = torch.from_numpy(np.ascontiguousarray(basis_vectors, dtype=np.float64))

    #         def init_weights(m):
    #             if isinstance(m, nn.Linear):
    #                 torch.nn.init.kaiming_normal_(m.weight)
    #                 m.bias.data.fill_(0.01)

    #         self.coefs.apply(init_weights)

    def interp1d(self, x, y, x_new):
        """
        Linearly interpolates tabulated basis functions.

        Args:
            x (torch.Tensor): The sorted grid, shape (n_grid,).
            y (torch.Tensor): The basis functions on the grid, shape (n_grid, num_basis).
            x_new (torch.Tensor): The points to interpolate at, shape (n_points,).

        Returns:
            torch.Tensor: The interpolated values, shape (n_points, num_basis).
        """
        # 2. Find where in the original data, the values to interpolate
        #    would be inserted.
        #    Note: If x_new[n] == x[m], then m is returned by searchsorted.
        # y = torch.moveaxis(y,axis,0)
        # y = y.reshape((y.shape[0],-1))

        x_new_indices = torch.searchsorted(x, x_new)

        # 3. Clip x_new_indices so that they are within the range of
        #    self.x indices and at least 1. Removes mis-interpolation
        #    of x_new[n] = x[0]
        x_new_indices = x_new_indices.clip(1, len(x) - 1)

        # 4. Calculate the slope of regions that each x_new value falls in.
        lo = x_new_indices - 1
        hi = x_new_indices

        x_lo = x[lo]
        x_hi = x[hi]
        y_lo = y[lo]
        y_hi = y[hi]

        # Note that the following two expressions rely on the specifics of the
        # broadcasting semantics.
        slope = (y_hi - y_lo) / (x_hi - x_lo)[:, None]

        # 5. Calculate the actual value for each entry in x_new.
        y_new = slope * (x_new - x_lo)[:, None] + y_lo

        return y_new

    def forward(self, x, alpha):
        """
        Evaluates the layer.

        Args:
            x (torch.Tensor): The input features, shape (batch, in_features).
            alpha (torch.Tensor): The coverage levels in [0, 1], shape (batch,).

        Returns:
            torch.Tensor: The output, shape (batch,).
        """
        grid = self.grid.to(alpha)
        basis_vectors = self.basis_vectors.to(alpha)
        basis = self.interp1d(grid, basis_vectors, alpha)

        # print(basis.shape)
        # print(self.coefs(x).shape)
        # print(self.coefs(x))
        weighted_basis = self.coefs(x) * basis
        # print(weighted_basis.shape)
        return weighted_basis.sum(axis=-1)


class IsplineNN(nn.Module):
    """
    Network for Cal-PIT whose output is monotone in the coverage level alpha.

    An MLP of PReLU layers maps the features x to the weights of an ISplineLayer, which
    evaluates the I-splines at alpha. The weights do not depend on alpha, so the predicted
    PIT CDF is non-decreasing in alpha, runs from 0 at alpha = 0 to 1 at alpha = 1, and the
    recalibrated densities from CalPIT.transform are non-negative.

    Args:
        n_features (int): The number of features, not counting alpha.
        hidden_layers (list of int, optional): The widths of the hidden layers.
            Defaults to [512, 512, 512].
        dropout_p (float, optional): The dropout probability in the ISplineLayer.
            Defaults to 0.5.
        num_basis (int, optional): The number of I-spline basis functions. Defaults to 10.

    Attributes:
        output (str): "probability"; the network returns r itself, not its logit.

    Raises:
        ImportError: If spline-basis is not installed.
    """

    output = "probability"

    def __init__(self, n_features, hidden_layers=None, dropout_p=0.5, num_basis=10):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [512, 512, 512]
        self.all_layers = [n_features]
        self.hidden_layers = hidden_layers
        self.all_layers.extend(hidden_layers)
        self.num_basis = num_basis
        self.dropout_p = dropout_p
        self.spline_layer = ISplineLayer(
            in_features=self.hidden_layers[-1], num_basis=self.num_basis, dropout_p=self.dropout_p
        )

        self.mlp_layer_list = []
        for i in range(len(self.all_layers) - 1):
            self.mlp_layer_list.append(nn.Linear(self.all_layers[i], self.all_layers[i + 1]))
            self.mlp_layer_list.append(nn.PReLU())

        self.mlp_layers = nn.Sequential(*self.mlp_layer_list)
        self.mlp_layers.apply(_init_weights)

    def forward(self, alpha, x):
        """
        Evaluates the network.

        Args:
            alpha (torch.Tensor): The coverage levels in [0, 1], shape (batch,).
            x (torch.Tensor): The features, shape (batch, n_features).

        Returns:
            torch.Tensor: The predicted probability that the PIT is at most alpha, shape (batch,).
        """
        return self.spline_layer(self.mlp_layers(x), alpha)
