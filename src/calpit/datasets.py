from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from calpit.nn import utils as nn_utils


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


# Kept for backwards compatibility; the implementation lives in calpit.nn.utils.
RandomDataset = nn_utils.RandomDataset


class PhotometryDataset(Dataset):
    def __init__(self, file_path=None, pit=None, scaler=None):
        self.pit = pit
        self.scaler = scaler
        try:
            import h5py  # noqa: PLC0415 - optional dependency.
        except ImportError as error:
            raise ImportError(
                "PhotometryDataset requires the optional dependency h5py. "
                "Install it with: pip install 'calpit[hdf5]'"
            ) from error
        if Path(file_path).suffix == ".hdf5":
            self.file = h5py.File(file_path, "r")

    def __len__(self):
        key = list(self.file.keys())[0]
        return len(self.file[key])

    def __getitem__(self, idx):
        x = self.file["dered_color_features"][idx]
        if self.scaler:
            x = self.scaler.transform(x.reshape(1, -1))
        x = torch.tensor(x.squeeze())
        y = torch.tensor(self.pit[idx])

        alpha = torch.rand(1)
        feature = torch.hstack([alpha, x])
        target = (y <= alpha).float()

        return feature, target
