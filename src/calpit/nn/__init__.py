"""PyTorch components of Cal-PIT, for CalPIT and for hand-written training loops.

Requires the torch extra: pip install 'calpit[torch]'. The Lightning pieces
live in calpit.nn.lightning, which is imported only when used.
"""

from calpit import _optional

_optional.import_optional("torch", "torch", "calpit.nn")

# The modules below need torch, which the check above makes sure of.
from calpit.nn import data, ispline_nn, models, training, utils  # noqa: E402

ConcatAlpha = models.ConcatAlpha
CoverageDataset = data.CoverageDataset
CoverageGridDataset = data.CoverageGridDataset
IsplineNN = ispline_nn.IsplineNN
MLP = models.MLP
MonotonicNN = models.MonotonicNN
OUTPUT_TYPES = models.OUTPUT_TYPES
PhotometryDataset = data.PhotometryDataset
cde_loss = utils.cde_loss
coverage_loss = training.coverage_loss
output_type = models.output_type
predict_pit_cdf = training.predict_pit_cdf
trapz_grid_torch = utils.trapz_grid_torch
