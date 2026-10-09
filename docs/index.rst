.. calpit documentation main file.

Welcome to `calpit`'s documentation!
========================================================================================


Overview
---------------------------

`calpit` diagnoses and recalibrates conditional density estimates (CDEs). Given
a calibration set of features ``x``, true values ``y`` and the CDEs some model
predicted for them, it learns how the probability integral transform (PIT) of
those CDEs varies with ``x``. That gives local P-P plots that show where in
feature space the CDEs are miscalibrated, and a recalibration of the CDEs
themselves.

- **Any representation.** CDEs can come as densities on a grid, as quantiles,
  as samples, or as `qp <https://github.com/LSSTDESC/qp>`_ ensembles, and are
  recalibrated in the same representation.
- **Any model.** Cal-PIT can be learned by any PyTorch network, trained with
  PyTorch Lightning, or by any scikit-learn classifier.
- **The scikit-learn API.** ``CalPIT`` has ``fit``, ``predict``, ``transform``,
  ``get_params`` and ``set_params``, and works with ``sklearn.base.clone``.
- **Your own training loop.** Every piece of the training is importable, for
  models too unusual for ``CalPIT.fit``.


Basic Usage
---------------------------

.. code-block:: python

   import calpit

   # CDEs of the calibration set, here as densities on a grid.
   cde_calib = calpit.GridCDE(pdf_calib, y_grid)

   # Learn the conditional PIT distribution with the default monotone network.
   recalibrator = calpit.CalPIT().fit(x_calib, y_calib, cde_calib)

   # Local P-P curves of the test CDEs, and how far they stray from the diagonal.
   local = recalibrator.diagnose(x_test)
   print(local.ks)

   # Recalibrated test CDEs, in the same representation.
   cde_new = recalibrator.transform(x_test, calpit.GridCDE(pdf_test, y_grid))

See :doc:`usage` for quantile, sample and qp CDEs, scikit-learn models and the
options, and :doc:`training_loop` for training a network by hand.


Installation
---------------------------

.. code-block:: console

   >> pip install 'calpit[torch]'

The core of `calpit` needs only NumPy and SciPy. The models and file formats
come with extras:

- ``torch`` installs PyTorch, Lightning and ``spline-basis``, for PyTorch models
  and the default ``calpit.nn.IsplineNN``.
- ``sklearn`` installs scikit-learn, for scikit-learn models.
- ``qp`` installs ``qp-prob``, for ``calpit.qp_io``.
- ``hdf5`` installs ``h5py``, for ``calpit.nn.PhotometryDataset``.
- ``plot`` installs ``matplotlib``, for ``calpit.utils.plot_pit`` and
  ``calpit.diagnostics.plot_local_pp``.
- ``all`` installs all of the above.

To install the latest version from GitHub, or for development:

.. code-block:: console

   >> pip install 'calpit[torch] @ git+https://github.com/lee-group-cmu/Cal-PIT'
   >> git clone https://github.com/lee-group-cmu/Cal-PIT.git
   >> cd Cal-PIT
   >> pip install -e '.[dev]'

.. note::

   `calpit` needs Python 3.11 or later. To use a GPU, install the PyTorch build
   for your system first, following the `PyTorch website
   <https://pytorch.org/get-started/locally/>`_.

References
---------------------------
The `calpit` package is based on the work described in following papers:

- :cite:t:`Dey2021RecalPhotoz` and :cite:t:`Dey2022Recal`, which introduce the
  recalibration framework for conditional density estimates.
- :cite:t:`Zhao2021Diagnostics`, which introduces diagnostics for conditional
  density estimation methods.

It also uses:

- the monotone rearrangement of Chernozhukov, Fernández-Val & Galichon (2010),
  "Quantile and Probability Curves Without Crossing", Econometrica 78, 1093,
  `arXiv:0704.3649 <https://arxiv.org/abs/0704.3649>`_,
  `doi:10.3982/ECTA7880 <https://doi.org/10.3982/ECTA7880>`_;
- the unconstrained monotonic neural network of :cite:t:`Wehenkel2019UMNN`, for
  ``calpit.nn.MonotonicNN``.


.. bibliography::
   :all:


.. toctree::
   :hidden:

   Home Page <self>
   User guide <usage>
   Writing your own training loop <training_loop>
   Examples <notebooks>
   API Reference <autoapi/index>
