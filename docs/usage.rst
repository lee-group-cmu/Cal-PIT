User guide
========================================================================================

What Cal-PIT learns
---------------------------

The PIT of a CDE with CDF :math:`F(y \mid x)` at the true value is
:math:`F(y_{\mathrm{true}} \mid x)`. If the CDEs are calibrated at :math:`x`, the
PIT is uniform there, so

.. math::

   r(\alpha; x) = P(\mathrm{PIT} \leq \alpha \mid x)

equals :math:`\alpha`. Cal-PIT learns :math:`r` by regressing the indicator
:math:`\mathrm{PIT} \leq \alpha` on :math:`(\alpha, x)` over a calibration set.
Then:

- ``CalPIT.predict(x, alpha)`` returns :math:`\hat r(\alpha; x)`, the local
  P-P curve at each :math:`x`, and ``CalPIT.diagnose(x)`` how far each curve
  strays from the diagonal (:cite:t:`Zhao2021Diagnostics`).
- ``CalPIT.transform(x, cde)`` recalibrates the CDEs by composing their CDF with
  :math:`\hat r`, :math:`F_{\mathrm{new}}(y \mid x) = \hat r(F(y \mid x); x)`, or
  their quantile function with its inverse,
  :math:`Q_{\mathrm{new}}(\tau \mid x) = Q(\hat r^{-1}(\tau; x) \mid x)`
  (:cite:t:`Dey2022Recal`).


CDE representations
---------------------------

Every representation provides the PIT of the true values, to fit Cal-PIT, and a
recalibration in the same representation. The PIT and the recalibrated CDEs are
the same quantities whichever representation the CDEs come in; only the
interpolation error differs.

.. code-block:: python

   import calpit

   # Densities on a common grid: pdf has shape (n_objects, n_grid).
   cde = calpit.GridCDE(pdf, y_grid)

   # Quantiles at common levels: locations has shape (n_objects, n_levels).
   cde = calpit.QuantileCDE(levels, locations)

   # Samples from each predictive distribution: shape (n_objects, n_samples).
   cde = calpit.SampleCDE(samples, random_state=0)

   recalibrator = calpit.CalPIT().fit(x_calib, y_calib, cde_calib)
   cde_new = recalibrator.transform(x_test, cde_test)  # Same class as cde_test.

- **Grid.** The recalibrated CDF on the grid is interpolated with a PCHIP spline
  and differentiated, so the new densities can be slightly negative or fail to
  integrate to one; ``calpit.utils.normalize`` fixes both. The PIT integrates
  the density up to the last grid point at or below the true value, as `calpit`
  always has, so it is biased low by up to the density times the grid spacing:
  use a grid fine enough for that not to matter.
- **Quantiles.** The CDF is interpolated linearly between the quantiles, and the
  recalibrated quantiles are at the same levels. Outside the outermost quantiles
  the CDF is taken as the outermost level, so include the levels 0 and 1, at the
  ends of the support.
- **Samples.** With :math:`k` of the :math:`m` samples below the true value, the
  PIT is :math:`(k + U) / (m + 1)` with :math:`U` uniform, which is exactly
  uniform for calibrated samples. The recalibrated CDEs are the sorted samples,
  each moved to the recalibrated quantile at its level :math:`i / (m + 1)`.

qp ensembles
^^^^^^^^^^^^

``calpit.qp_io`` converts `qp <https://github.com/LSSTDESC/qp>`_ ensembles at the
edges; `calpit` does not use qp internally.

.. code-block:: python

   from calpit import qp_io

   cde = qp_io.from_qp(ensemble)                  # interp -> GridCDE, quant -> QuantileCDE
   cde = qp_io.from_qp(ensemble, y_grid=y_grid)   # any ensemble, evaluated on a grid
   ensemble_new = qp_io.to_qp(recalibrator.transform(x_test, cde))


Models
---------------------------

PyTorch networks
^^^^^^^^^^^^^^^^

Without a model, ``CalPIT`` trains ``calpit.nn.IsplineNN``, sized to the number of
features. Any other network follows one contract:

.. code-block:: python

   class MyNetwork(torch.nn.Module):
       output = "logit"  # or "probability"; "logit" when absent

       def forward(self, alpha, x):
           # alpha: (batch,), x: (batch, n_features) -> (batch,)
           ...

A ``"logit"`` network is trained with ``BCEWithLogitsLoss`` and can output any real
number; a ``"probability"`` network must output values in :math:`[0, 1]`. A network
can also define ``forward_curves(alpha, x)``, with ``alpha`` of shape
``(batch, n_alpha)``, to predict all the :math:`\alpha` of an object in one pass;
``IsplineNN`` does, so its MLP runs once per object.
``calpit.nn.ConcatAlpha(network)`` adapts a network that takes one tensor, giving
it :math:`[\alpha, x]` with :math:`\alpha` in column 0. ``CalPIT`` copies the
network it is given and trains the copy, ``model_``. To size a network from the
data, pass a callable that takes the number of features:

.. code-block:: python

   import functools

   recalibrator = calpit.CalPIT(functools.partial(calpit.nn.MLP, hidden_layers=[128, 128]))

The networks in ``calpit.nn``:

- ``IsplineNN`` maps :math:`x` to the weights of I-splines in :math:`\alpha`, so
  :math:`\hat r` rises from 0 at :math:`\alpha = 0` to 1 at :math:`\alpha = 1`
  and is non-decreasing for every :math:`x`.
- ``MonotonicNN`` is monotone in :math:`\alpha` by construction. The
  unconstrained monotonic neural network models a function that is monotone in
  one input as the integral of a strictly positive function computed by a
  free-form neural network (:cite:t:`Wehenkel2019UMNN`).
- ``MLP`` is a plain multi-layer perceptron of :math:`[\alpha, x]`, with no
  monotonicity.

Training uses a PyTorch Lightning ``Trainer``. Pass any of its arguments through
``trainer_kwargs``, for example to choose devices, precision or a logger:

.. code-block:: python

   import lightning

   recalibrator = calpit.CalPIT(
       max_epochs=200,
       patience=20,
       trainer_kwargs={
           "accelerator": "gpu",
           "precision": "16-mixed",
           "logger": lightning.pytorch.loggers.CSVLogger("logs"),
       },
   )

By default the trainer picks the accelerator itself, uses one device, logs
nothing and writes no checkpoint files; the best weights are kept in memory.
On several devices the validation loss is summed over all of them, so every
process stops at the same epoch.
After ``fit``, ``train_loss_`` and ``val_bce_`` hold the loss curves.

scikit-learn classifiers
^^^^^^^^^^^^^^^^^^^^^^^^

Any classifier with ``predict_proba`` works:

.. code-block:: python

   from sklearn import ensemble

   recalibrator = calpit.CalPIT(ensemble.HistGradientBoostingClassifier(), n_alpha=50)

A scikit-learn estimator is fitted once on a fixed data set, so each calibration
object contributes ``n_alpha`` rows :math:`[\alpha, x]` with target
:math:`\mathrm{PIT} \leq \alpha`. The :math:`\alpha` are stratified: :math:`[0, 1]` is
cut into ``n_alpha`` equal strata and each object gets one uniform draw in each.
The classifier is fitted on ``n_objects * n_alpha`` rows, so choose ``n_alpha``
with memory in mind.

If the classifier has a ``monotonic_cst`` parameter left at ``None``, as
``HistGradientBoostingClassifier`` does, the fitted clone is constrained to
increase in :math:`\alpha`. Tree ensembles are piecewise constant in
:math:`\alpha`, so :math:`\hat r` is predicted at ``n_alpha + 1`` evenly spaced
:math:`\alpha` and interpolated linearly in between, which keeps the recalibrated
densities free of spikes.


Monotone rearrangement
---------------------------

``CalPIT.transform`` needs :math:`\hat r` to be non-decreasing in :math:`\alpha`,
and a network like ``MLP`` does not guarantee it. So, by default
(``rearrange=True``), ``CalPIT`` rearranges every prediction, following
Chernozhukov, Fernández-Val & Galichon (2010), "Quantile and Probability Curves
Without Crossing", `arXiv:0704.3649 <https://arxiv.org/abs/0704.3649>`_.
Sorting the values predicted at a set of alpha (monotone rearrangement) makes any
estimate of r non-decreasing in alpha. If the true r is non-decreasing, the
rearranged estimate is never further from it in Lp distance than the original
estimate was.

Predictions that are already non-decreasing, such as those of ``IsplineNN`` and
``MonotonicNN``, come back unchanged. Pass ``rearrange=False`` to see a model's
raw predictions; ``calpit.coverage.rearrange`` applies the rearrangement on its
own.


Diagnostics
---------------------------

.. code-block:: python

   local = recalibrator.diagnose(x_test)  # 101 alpha from 0 to 1 by default
   local.pit_cdf        # (n_objects, n_alpha): the local P-P curves
   local.ks             # largest distance of each curve from the diagonal
   local.cvm            # root mean squared distance from the diagonal
   local.coverage(0.9)  # actual coverage of each central 90% interval

   from calpit import diagnostics

   fig, ax = diagnostics.plot_local_pp(local, indices=[0, 1, 2])


Reproducibility and saving
---------------------------

``random_state`` (default 299792458) seeds the train/validation split, the
:math:`\alpha` draws, the initialization of a network built by ``CalPIT`` and the
PyTorch random numbers during training, without changing the global random
state. A fitted ``CalPIT`` pickles like any scikit-learn estimator:

.. code-block:: python

   import pickle

   with open("recalibrator.pkl", "wb") as f:
       pickle.dump(recalibrator, f)
