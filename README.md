# Cal-PIT

[![Documentation](https://readthedocs.org/projects/cal-pit/badge/?version=latest)](https://docs.readthedocs.io/en/stable/badges.html) [![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

Full Documentation
---------------------------
Full documentation for the project is available on [Read the Docs](https://cal-pit.readthedocs.io/en/latest/)

Overview
---------------------------

`calpit` diagnoses and recalibrates conditional density estimates (CDEs). It learns how the probability integral transform (PIT) of the CDEs varies with the features, which gives local P-P plots that show where the CDEs are miscalibrated, and a recalibration of the CDEs themselves.

- CDEs can be densities on a grid, quantiles, samples or [qp](https://github.com/LSSTDESC/qp) ensembles.
- Cal-PIT can be learned by any PyTorch network, trained with PyTorch Lightning, or by any scikit-learn classifier.
- `CalPIT` follows the scikit-learn API, and every piece of the training is importable for hand-written training loops.


Basic Usage
---------------------------

```python
import calpit

recalibrator = calpit.CalPIT().fit(x_calib, y_calib, calpit.GridCDE(pdf_calib, y_grid))

local = recalibrator.diagnose(x_test)  # Local P-P curves and their distance from the diagonal.
cde_new = recalibrator.transform(x_test, calpit.GridCDE(pdf_test, y_grid))  # Recalibrated CDEs.
```

See the [user guide](https://cal-pit.readthedocs.io/en/latest/usage.html) for quantile, sample and qp CDEs, scikit-learn models and [hand-written training loops](https://cal-pit.readthedocs.io/en/latest/training_loop.html).


Installation
---------------------------

```console
   pip install 'calpit[torch]'
```

The core needs only NumPy and SciPy; models and file formats come with extras:

| Extra | Installs | Needed for |
|---|---|---|
| `torch` | `torch`, `lightning`, `spline-basis` | PyTorch models, including the default `calpit.nn.IsplineNN` |
| `sklearn` | `scikit-learn` | scikit-learn models |
| `qp` | `qp-prob` | `calpit.qp_io` |
| `hdf5` | `h5py` | `calpit.nn.PhotometryDataset` |
| `plot` | `matplotlib` | `calpit.utils.plot_pit`, `calpit.diagnostics.plot_local_pp` |
| `all` | all of the above | |

`calpit` needs Python 3.11 or later.

To install the latest version of the code from Github, you can run the following command:

```console
  pip install git+https://github.com/lee-group-cmu/Cal-PIT
```

If you would like to install the package for development purposes, you can clone the repository and install the package in editable mode:

```console
   >> git clone https://github.com/lee-group-cmu/Cal-PIT.git
   >> cd Cal-PIT
   >> pip install -e '.[dev]'
```

