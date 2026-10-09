Writing your own training loop
========================================================================================

``CalPIT.fit`` covers most networks, but a complicated model may need its own
optimizer, schedule, data pipeline or distributed setup. Every step of the
training is importable, so a hand-written loop only has to do what is special
about the model. The trained network then goes into
``CalPIT.from_fitted``, which predicts, diagnoses and recalibrates exactly as a
fitted ``CalPIT`` does.

The pieces
---------------------------

========================================  =====================================================
``CDE.pit(y)``                            PIT values of grid, quantile, sample or qp CDEs
``calpit.coverage.train_val_split``       split by object, so all of an object's rows stay
                                          on one side
``calpit.nn.CoverageDataset``             training items ``(alpha, x, target)`` with a fresh
                                          :math:`\alpha \sim U(0, 1)` every time an item is read;
                                          ``.batches(batch_size)`` loads whole batches at once
``calpit.nn.CoverageGridDataset``         validation batches: every object at every
                                          :math:`\alpha` of a fixed grid, built on the fly
``calpit.nn.coverage_loss``               binary cross entropy for logit or probability
                                          outputs
``calpit.nn.output_type``                 what a network returns, ``"logit"`` or
                                          ``"probability"``
``calpit.nn.predict_pit_cdf``             :math:`\hat r(\alpha; x)` from any network, batched
``calpit.coverage.rearrange``             monotone rearrangement of predictions
``calpit.nn.lightning.CalPITModule``      the ``LightningModule`` that ``CalPIT.fit`` trains
``calpit.nn.lightning.``                  early stopping that keeps the best weights in
``BestWeightsEarlyStopping``              memory
``calpit.CalPIT.from_fitted``             wraps a trained network or classifier
========================================  =====================================================

The network follows the contract in :doc:`usage`: ``forward(alpha, x)`` returns
one logit (or probability) per row.

A plain PyTorch loop
---------------------------

.. code-block:: python

   import numpy as np
   import torch

   import calpit
   import calpit.nn

   # 1. PIT values of the calibration CDEs, in any representation.
   pit = calpit.GridCDE(cde_calib, y_grid).pit(y_calib)

   # 2. Hold out validation objects, and build the data sets.
   x_train, pit_train, x_val, pit_val = calpit.coverage.train_val_split(
       x_calib, pit, val_fraction=0.1, random_state=0
   )
   train_loader = calpit.nn.CoverageDataset(x_train, pit_train).batches(batch_size=2048)
   val_set = calpit.nn.CoverageGridDataset(x_val, pit_val, alpha=np.linspace(0.001, 0.999, 201))

   # 3. Train.
   model = MyNetwork()  # forward(alpha, x) -> logit
   output_type = calpit.nn.output_type(model)
   optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
   for epoch in range(100):
       model.train()
       for alpha, x, target in train_loader:
           optimizer.zero_grad()
           loss = calpit.nn.coverage_loss(model(alpha, x), target, output_type)
           loss.backward()
           optimizer.step()

       model.eval()
       with torch.no_grad():
           val_loss = sum(
               calpit.nn.coverage_loss(model(alpha, x), target, output_type, reduction="sum").item()
               for alpha, x, target in val_set
           ) / val_set.n_rows()

   # 4. Use the trained network like a fitted CalPIT.
   recalibrator = calpit.CalPIT.from_fitted(model)
   cde_new = recalibrator.transform(x_test, calpit.GridCDE(cde_test, y_grid))
   local = recalibrator.diagnose(x_test)

Move ``alpha``, ``x`` and ``target`` to the network's device inside the loop when
training on a GPU. ``CoverageGridDataset`` yields whole batches; wrap it in
``DataLoader(val_set, batch_size=None)`` for worker processes.

Your own Lightning Trainer
---------------------------

To keep Lightning's devices, loggers and callbacks but control the trainer
yourself, train ``CalPITModule`` directly. Subclass it to change the loss or the
optimizer (``configure_optimizers``).

.. code-block:: python

   import lightning
   from torch.utils import data

   from calpit.nn import lightning as calpit_lightning

   module = calpit_lightning.CalPITModule(MyNetwork(), lr=1e-3)
   trainer = lightning.Trainer(
       max_epochs=500,
       callbacks=[calpit_lightning.BestWeightsEarlyStopping(patience=20)],
       logger=lightning.pytorch.loggers.TensorBoardLogger("logs"),
   )
   trainer.fit(module, train_loader, data.DataLoader(val_set, batch_size=None))

   recalibrator = calpit.CalPIT.from_fitted(module.model)

Other libraries
---------------------------

``calpit.coverage`` builds the training rows with NumPy alone, so any library
with a scikit-learn style classifier works, for example LightGBM or XGBoost:

.. code-block:: python

   import lightgbm

   alpha = calpit.coverage.stratified_alpha(len(x_calib), n_alpha=50, random_state=0)
   features, targets = calpit.coverage.expand_coverage(x_calib, pit, alpha)
   classifier = lightgbm.LGBMClassifier(monotone_constraints=[1] + [0] * x_calib.shape[1])
   classifier.fit(features, targets)

   recalibrator = calpit.CalPIT.from_fitted(classifier, n_alpha=50)

The features are :math:`[\alpha, x]`, with :math:`\alpha` in column 0, and the
targets are booleans. ``n_alpha`` in ``from_fitted`` sets the :math:`\alpha`
knots that predictions are interpolated between.
