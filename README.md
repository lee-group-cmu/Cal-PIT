# Cal-PIT

[![Documentation](https://readthedocs.org/projects/cal-pit/badge/?version=latest)](https://docs.readthedocs.io/en/stable/badges.html) [![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

Full Documentation
---------------------------
Full documentation for the project is available on [Read the Docs](https://cal-pit.readthedocs.io/en/latest/)

Overview
---------------------------

`calpit` is a Python package for diagnosing and recalibrating conditional density estimates. The package is built on top of Pytorch (with other ML backends to be added soon) and provides a simple and flexible interface matching the scikit-learn API.


Basic Usage
---------------------------
The following is a basic recipe for using the `calpit` package:

```python

   from calpit import CalPit #import the CalPit class
   
   calpit_model = CalPit(model=model) #Any Pytorch model CalPit class
   
   trained_model = calpit_model.fit(x_calib,y_calib, cde_cali,y_grid) #Fit the model with a calibration dataset
   
   pp_result = calpit_model.predict(x_test, cov_grid) #Predict the local PIT distribution for a test dataset
   
   new_cde = calpit_model.transform(x_test, cde_test, y_grid) #Recalibrate the conditional density estimate for a test dataset
```





Installation
---------------------------

To install the current release of the package, you can run the following command:

```console
   pip install calpit
```

To install the latest version of the code from Github, you can run the following command:

```console
  pip install git+https://github.com/lee-group-cmu/Cal-PIT
```

If you would like to install the package for development purposes, you can clone the repository and install the package in editable mode:

```console
   >> git clone https://github.com/lee-group-cmu/Cal-PIT.git
   >> cd Cal-PIT
   >> pip install -e .
```

