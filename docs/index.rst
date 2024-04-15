.. calpit documentation main file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to `calpit`'s documentation!
========================================================================================
Overview
---------------------------

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
The package is based on the following papers:
- :cite:t:`Zhao2021Diagnostics` for diagnostics
.. bibliography::

.. toctree::
   :hidden:

   Home Page <self>
   Examples <notebooks>
   API Reference <autoapi/index>
   
