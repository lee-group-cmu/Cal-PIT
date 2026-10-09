"""Conversion between calpit CDEs and qp ensembles.

qp (https://github.com/LSSTDESC/qp) is the format LSST DESC uses to store
photometric redshift PDFs. calpit only reads and writes it at the edges: a qp
ensemble becomes a GridCDE or a QuantileCDE, and the recalibrated CDEs go back
into a qp ensemble.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from calpit import _optional, representations


def _pdf_name(ensemble: Any) -> str:
    name = np.ravel(ensemble.metadata["pdf_name"])[0]
    return name.decode() if isinstance(name, bytes) else str(name)


def from_qp(
    ensemble: Any,
    y_grid: npt.ArrayLike | None = None,
    quantile_levels: npt.ArrayLike | None = None,
) -> representations.GridCDE | representations.QuantileCDE:
    """Converts a qp ensemble into calpit CDEs.

    An interp ensemble becomes a GridCDE on its own grid and a quant ensemble a
    QuantileCDE at its own levels. Any other ensemble, or one of those two
    resampled, needs y_grid or quantile_levels.

    Args:
        ensemble: The qp.Ensemble.
        y_grid: The grid to evaluate the densities on, shape (n_grid,).
        quantile_levels: The levels to evaluate the quantiles at,
            shape (n_levels,). Ignored when y_grid is given.

    Returns:
        A GridCDE when y_grid is given or the ensemble is an interp ensemble,
        otherwise a QuantileCDE.
    """
    _optional.import_optional("qp", "qp", "calpit.qp_io")
    if y_grid is not None:
        y_grid = np.ravel(np.asarray(y_grid, dtype=float))
        return representations.GridCDE(ensemble.pdf(y_grid), y_grid)
    if quantile_levels is not None:
        levels = np.ravel(np.asarray(quantile_levels, dtype=float))
        return representations.QuantileCDE(levels, ensemble.ppf(levels[None, :]))
    name = _pdf_name(ensemble)
    if name == "interp":
        return representations.GridCDE(ensemble.objdata["yvals"], ensemble.metadata["xvals"])
    if name == "quant":
        return representations.QuantileCDE(ensemble.metadata["quants"], ensemble.objdata["locs"])
    raise ValueError(f"pass y_grid or quantile_levels to convert a qp {name!r} ensemble: {name=}")


def to_qp(
    cde: representations.GridCDE | representations.QuantileCDE | representations.SampleCDE,
) -> Any:
    """Converts calpit CDEs into a qp ensemble.

    Args:
        cde: The CDEs. Samples are stored as quantiles at the levels
            i / (m + 1), as SampleCDE.to_quantiles gives them.

    Returns:
        A qp.Ensemble: an interp ensemble for a GridCDE, otherwise a quant
        ensemble.
    """
    qp = _optional.import_optional("qp", "qp", "calpit.qp_io")
    if isinstance(cde, representations.GridCDE):
        return qp.Ensemble(qp.interp, data={"xvals": cde.y_grid, "yvals": cde.pdf})
    if isinstance(cde, representations.SampleCDE):
        cde = cde.to_quantiles()
    return qp.Ensemble(qp.quant, data={"quants": cde.levels, "locs": cde.locations})
