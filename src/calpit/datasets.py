import numpy as np


class TuningFork:
    def __init__(self, dims=3, lam=3, seed=299792458):
        self.dims = dims
        self.lam = lam
        self.seed = seed

    def generate_data(self, size, seed=None):
        if seed is None:
            seed = self.seed
        rng = np.random.default_rng(seed=seed)

        X_unif = rng.uniform(low=-5, high=5, size=size * (self.dims - 1)).reshape(size, self.dims - 1)
        X_bern = rng.binomial(n=1, p=0.2, size=size)

        eps1 = rng.normal(loc=0, scale=1, size=size)
        eps2 = rng.normal(loc=0, scale=0.1, size=size)

        X_data = np.hstack([X_bern.reshape(-1, 1), X_unif])

        double_fork = X_data[:, 1] > 0

        Y_data = (1 - X_bern) * (self.lam * eps2 + 0.2 * (X_data[:, 1] + 5) * eps1) + X_bern * (
            self.lam * eps2 - 0.2 * (X_data[:, 1] - 5) * eps1
        )

        Y_data += double_fork * (1 - X_bern) * 1 * X_data[:, 1] - double_fork * X_bern * 1 * X_data[:, 1]

        return X_data, Y_data
