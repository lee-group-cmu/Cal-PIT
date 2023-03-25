import numpy as np
from nn.utils import EarlyStopping, RandomDataset


class CalPIT:
    def __init__(self, x_calib, y_calib, cde_calib, model):
        self.x_calib = x_calib
        self.y_calib = y_calib
        self.cde_calib = cde_calib
        self.model = model

    def fit(self):
        # Take in hyperparameters, contains the training loop
        return self.model

    def predict_pit(X_test, n_gamma, gamma_grid):
        return predicted_PIT

    def predict_cde(X_test, y_grid, n_grid):
        return predicted_CDE
