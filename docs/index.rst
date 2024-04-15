.. calpit documentation main file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to `calpit`'s documentation!
========================================================================================


Overview
---------------------------

`calpit` is a Python package for diagnosing and recalibrating conditional density estimates. The package is built on top of Pytorch (with other ML backends to be added soon) and provides a simple and flexible interface matching the scikit-learn API.


Basic Usage
---------------------------
The following is a basic recipe for using the `calpit` package:

.. code-block:: python

   from calpit import CalPit #import the CalPit class
   
   calpit_model = CalPit(model=model) #Any Pytorch model CalPit class
   
   trained_model = calpit_model.fit(x_calib,y_calib, cde_cali,y_grid) #Fit the model with a calibration dataset
   
   pp_result = calpit_model.predict(x_test, cov_grid) #Predict the local PIT distribution for a test dataset
   
   new_cde = calpit_model.transform(x_test, cde_test, y_grid) #Recalibrate the conditional density estimate for a test dataset





Installation
---------------------------

To install the current release of the package, you can run the following command:

.. code-block:: console

   >> pip install calpit

To install the latest version of the code from Github, you can run the following command:

.. code-block:: console

   >> pip install git+https://github.com/lee-group-cmu/Cal-PIT

If you would like to install the package for development purposes, you can clone the repository and install the package in editable mode:

.. code-block:: console

   >> git clone https://github.com/lee-group-cmu/Cal-PIT.git
   >> cd Cal-PIT
   >> pip install -e .

.. note::

   - The package is intended for use with Python 3.10 or later.
   - Pytorch is a required dependency for calpit. Please follow the instructions on the `Pytorch website <https://pytorch.org/get-started/locally/>`_ to install the appropriate version for your system.

References
---------------------------
The `calpit` package is based on the work described in following papers:

- :cite:t:`Dey2021RecalPhotoz` and :cite:t:`Dey2022Recal` which introduces the recalibration framework for conditional density estimates.
- :cite:t:`Zhao2021Diagnostics`, which introduces diagnostics for conditional density estimation methods.


.. bibliography::
   :all:


.. toctree::
   :hidden:
   Home Page <self>
   Examples <notebooks>
   API Reference <autoapi/index>
   
