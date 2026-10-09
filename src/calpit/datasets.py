"""Synthetic data sets for trying out Cal-PIT.

The PyTorch data sets for training moved to calpit.nn.
"""

import numpy as np


class TuningFork:
    """
    Synthetic "tuning fork" data set with a heteroscedastic, bimodal conditional distribution.

    The first feature is Bernoulli(0.2) and selects one of two branches; the remaining features
    are Uniform(-5, 5). The spread of the target grows with the second feature in one branch and
    shrinks in the other, and the branches separate (the fork) where the second feature is
    positive. The remaining features carry no information about the target.

    Args:
        dims (int, optional): The number of features, at least 2. Defaults to 3.
        lam (float, optional): The scale of the homoscedastic noise component. Defaults to 3.
        seed (int, optional): The default random seed for generate_data. Defaults to 299792458.
    """

    def __init__(self, dims=3, lam=3, seed=299792458):
        self.dims = dims
        self.lam = lam
        self.seed = seed

    def generate_data(self, size, seed=None):
        """
        Draws a sample of features and targets.

        Args:
            size (int): The number of samples.
            seed (int, optional): The random seed. Defaults to None, which uses the seed given
                at construction.

        Returns:
            tuple: The features, shape (size, dims), and the targets, shape (size,), as a tuple
            (x_data, y_data).
        """
        if seed is None:
            seed = self.seed
        rng = np.random.default_rng(seed=seed)

        x_unif = rng.uniform(low=-5, high=5, size=size * (self.dims - 1)).reshape(size, self.dims - 1)
        x_bern = rng.binomial(n=1, p=0.2, size=size)

        eps1 = rng.normal(loc=0, scale=1, size=size)
        eps2 = rng.normal(loc=0, scale=0.1, size=size)

        x_data = np.hstack([x_bern.reshape(-1, 1), x_unif])

        double_fork = x_data[:, 1] > 0

        y_data = (1 - x_bern) * (self.lam * eps2 + 0.2 * (x_data[:, 1] + 5) * eps1) + x_bern * (
            self.lam * eps2 - 0.2 * (x_data[:, 1] - 5) * eps1
        )

        y_data += double_fork * (1 - x_bern) * 1 * x_data[:, 1] - double_fork * x_bern * 1 * x_data[:, 1]

        return x_data, y_data
