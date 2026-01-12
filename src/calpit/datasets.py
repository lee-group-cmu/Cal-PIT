import numpy as np
from torch.utils.data import Dataset
import h5py
import torch
from pathlib import Path

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
        
class PhotometryDataset(Dataset):
    def __init__(self, file_path=None, pit=None, scaler=None):
        self.pit = pit
        self.scaler = scaler
        if Path(file_path).suffix == '.hdf5':
            self.file = h5py.File(file_path, 'r')

    def __len__(self):
        key = list(self.file.keys())[0]
        return len(self.file[key])

    def __getitem__(self, idx):
        x = self.file['dered_color_features'][idx]
        if self.scaler:
            x = self.scaler.transform(x.reshape(1,-1))
        x = torch.tensor(x.squeeze())
        y = torch.tensor(self.pit[idx])

        alpha = torch.rand(1)
        feature = torch.hstack([alpha, x])
        target = (y <= alpha).float()

        return feature, target
        
        