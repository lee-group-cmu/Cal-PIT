"""Unconstrained monotonic neural networks (UMNN).

Adapted from https://github.com/AWehenkel/UMNN (Wehenkel & Louppe 2019,
"Unconstrained Monotonic Neural Networks", NeurIPS), copyright (c) 2020
Antoine Wehenkel, under the BSD 3-Clause License included in this directory.
"""

from .MonotonicNN import MonotonicNN, IntegrandNN  # noqa
from .NeuralIntegral import NeuralIntegral  # noqa
from .ParallelNeuralIntegral import ParallelNeuralIntegral  # noqa
