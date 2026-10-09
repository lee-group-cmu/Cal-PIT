"""Diagnose and recalibrate conditional density estimates with Cal-PIT.

The core needs only NumPy and SciPy. PyTorch models need the torch extra,
scikit-learn models the sklearn extra and qp ensembles the qp extra.
"""

from calpit import coverage, datasets, diagnostics, estimator, metrics, representations, utils

CalPIT = estimator.CalPIT
NotFittedError = estimator.NotFittedError
GridCDE = representations.GridCDE
QuantileCDE = representations.QuantileCDE
SampleCDE = representations.SampleCDE

__all__ = [
    "CalPIT",
    "GridCDE",
    "NotFittedError",
    "QuantileCDE",
    "SampleCDE",
    "coverage",
    "datasets",
    "diagnostics",
    "estimator",
    "metrics",
    "representations",
    "utils",
]
