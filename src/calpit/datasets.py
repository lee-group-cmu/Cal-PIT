from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from calpit.nn import utils as nn_utils


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


# Kept for backwards compatibility; the implementation lives in calpit.nn.utils.
RandomDataset = nn_utils.RandomDataset


class PhotometryDataset(Dataset):
    """
    Training set for Cal-PIT read lazily from an HDF5 file of photometric features.

    Like calpit.nn.utils.RandomDataset, each item is the features prepended with a coverage
    level alpha drawn from Uniform(0, 1), and the target is 1 when the PIT is at most alpha.
    The features are read row by row from the "dered_color_features" data set, so the file
    need not fit in memory.

    Args:
        file_path (str or pathlib.Path): The path to an .hdf5 file with a
            "dered_color_features" data set of shape (n_samples, n_features).
        pit (np.ndarray): The PIT values, shape (n_samples,).
        scaler (optional): An object with a scikit-learn style transform method applied to each
            row of features. Defaults to None, which leaves the features unscaled.

    Raises:
        ImportError: If h5py is not installed.
    """

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
