"""PyTorch data sets of (alpha, x, target) rows for training Cal-PIT networks.

Every item is a tuple (alpha, x, target): the coverage level, shape (); the
features, shape (n_features,); and the target 1.0 if the PIT is at most alpha
and 0.0 otherwise, shape (). A DataLoader batches them into the
(alpha, x) inputs of the Cal-PIT network contract (see calpit.nn.models).
"""

import pathlib

import numpy as np
import numpy.typing as npt
import torch
from torch.utils import data

from calpit import _optional


class CoverageDataset(data.Dataset):
    """Training set that draws a fresh coverage level for every item.

    Each time an item is read, alpha is drawn from Uniform(0, 1) with the torch
    random number generator, so every epoch sees new (alpha, x) pairs.

    Args:
        x: The features, shape (n_objects, n_features).
        pit: The PIT values, shape (n_objects,).
        oversample: How many items each object yields per epoch; the data set
            has int(n_objects * oversample) items.

    Attributes:
        x: The features as float32, shape (n_objects, n_features).
        pit: The PIT values, shape (n_objects,).
        oversample: The oversampling factor.
    """

    def __init__(self, x: npt.ArrayLike, pit: npt.ArrayLike, oversample: float = 1) -> None:
        self.x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        self.pit = np.asarray(pit)
        self.oversample = oversample

    def __len__(self) -> int:
        """Returns the number of items per epoch."""
        return int(len(self.x) * self.oversample)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the item (alpha, x, target) of object index % n_objects."""
        index %= len(self.x)
        alpha = torch.rand(1)
        target = (self.pit[index] <= alpha).float()
        return alpha[0], self.x[index], target[0]


class CoverageGridDataset(data.Dataset):
    """Validation set of every object at every coverage level of a fixed grid.

    The rows run through the grid alpha-major: row i pairs alpha[i // n_objects]
    with object i % n_objects. Each item is a whole batch of consecutive rows,
    built on the fly, so memory does not grow with the grid. Iterate over it
    directly, or wrap it in DataLoader(dataset, batch_size=None).

    Args:
        x: The features, shape (n_objects, n_features).
        pit: The PIT values, shape (n_objects,).
        alpha: The coverage levels, shape (n_alpha,).
        batch_size: The number of rows per item.

    Attributes:
        x: The features as float32, shape (n_objects, n_features).
        pit: The PIT values, shape (n_objects,).
        alpha: The coverage levels, shape (n_alpha,). The network gets them as
            float32, but the targets compare the PIT with these values.
        batch_size: The number of rows per item.
    """

    def __init__(
        self, x: npt.ArrayLike, pit: npt.ArrayLike, alpha: npt.ArrayLike, batch_size: int = 2048
    ) -> None:
        self.x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        self.pit = np.asarray(pit)
        self.alpha = np.ravel(np.asarray(alpha))
        self.batch_size = batch_size

    def n_rows(self) -> int:
        """Returns the number of (alpha, object) rows, n_alpha * n_objects."""
        return len(self.alpha) * len(self.x)

    def __len__(self) -> int:
        """Returns the number of batches."""
        return -(-self.n_rows() // self.batch_size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns batch index as (alpha, x, target), each with up to batch_size rows."""
        if not 0 <= index < len(self):
            raise IndexError(f"batch index out of range: {index=}, {len(self)=}")
        rows = np.arange(index * self.batch_size, min((index + 1) * self.batch_size, self.n_rows()))
        alpha_index, object_index = np.divmod(rows, len(self.x))
        alpha = self.alpha[alpha_index]
        target = torch.as_tensor(self.pit[object_index] <= alpha, dtype=torch.float32)
        return torch.as_tensor(alpha, dtype=torch.float32), self.x[object_index], target


class PhotometryDataset(data.Dataset):
    """Training set read lazily from an HDF5 file of photometric features.

    Like CoverageDataset, alpha is drawn from Uniform(0, 1) for every item. The
    features are read row by row, so the file need not fit in memory. The file
    stays open for the life of the data set; call close() when done with it.

    Args:
        file_path: The path to an .hdf5 file.
        pit: The PIT values, shape (n_objects,).
        scaler: An object with a scikit-learn style transform method applied to
            each row of features, or None to leave them unscaled.
        key: The HDF5 data set of features, shape (n_objects, n_features).

    Attributes:
        file: The open h5py.File.
        pit: The PIT values, shape (n_objects,).
        scaler: The scaler, or None.
        key: The HDF5 data set of features.

    Raises:
        ImportError: If h5py is not installed.
    """

    def __init__(
        self,
        file_path: str | pathlib.Path,
        pit: npt.ArrayLike,
        scaler: object | None = None,
        key: str = "dered_color_features",
    ) -> None:
        h5py = _optional.import_optional("h5py", "hdf5", "PhotometryDataset")
        self.pit = np.asarray(pit)
        self.scaler = scaler
        self.key = key
        self.file = h5py.File(file_path, "r")

    def __len__(self) -> int:
        """Returns the number of objects."""
        return len(self.file[self.key])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the item (alpha, x, target) of object index."""
        x = self.file[self.key][index]
        if self.scaler is not None:
            x = self.scaler.transform(x.reshape(1, -1))  # type: ignore[attr-defined]
        alpha = torch.rand(1)
        target = (self.pit[index] <= alpha).float()
        return alpha[0], torch.as_tensor(np.ravel(x), dtype=torch.float32), target[0]

    def close(self) -> None:
        """Closes the HDF5 file."""
        self.file.close()
