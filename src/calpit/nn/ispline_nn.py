import torch
import torch.nn as nn
from splinebasis import ISplineBasis


class ISplineLayer(nn.Module):
    def __init__(self, in_features, num_basis,dropout_p=0):
        super().__init__()
        self.in_features = in_features
        self.num_basis = num_basis
        self.coefs = nn.Sequential(nn.Linear(in_features, num_basis), nn.Softmax(dim=-1),nn.Dropout(p=dropout_p))
        self.grid = torch.linspace(0, 1, 1000)
        self.basis_vectors = ISplineBasis(
            order=3, num_basis=num_basis, lower=0, upper=1, grid=self.grid
        ).basis_vectors
        self.basis_vectors = torch.from_numpy(self.basis_vectors)

#         def init_weights(m):
#             if isinstance(m, nn.Linear):
#                 torch.nn.init.kaiming_normal_(m.weight)
#                 m.bias.data.fill_(0.01)

#         self.coefs.apply(init_weights)

    def interp1d(self, x, y, x_new):
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
    def __init__(self, input_dim, hidden_layers=[512, 512, 512],dropout_p=0.5, num_basis=10):
        super().__init__()
        self.all_layers = [input_dim + 1]
        self.hidden_layers = hidden_layers
        self.all_layers.extend(hidden_layers)
        self.num_basis = num_basis
        self.dropout_p = dropout_p
        self.spline_layer = ISplineLayer(in_features=self.hidden_layers[-1], num_basis=self.num_basis,dropout_p=self.dropout_p)

        self.mlp_layer_list = []
        for i in range(len(self.all_layers) - 1):
            self.mlp_layer_list.append(nn.Linear(self.all_layers[i], self.all_layers[i + 1]))
            self.mlp_layer_list.append(nn.PReLU())

        # self.mlp_layer_list.append(nn.Dropout(p=dropout_p))
        self.mlp_layers = nn.Sequential(*self.mlp_layer_list)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.mlp_layers.apply(init_weights)

    def forward(self, x):
        alpha = x[:, 0]

        res = self.mlp_layers(x)

        res = self.spline_layer(res, alpha)

        return res
