from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.interpolate import PchipInterpolator
from tqdm import trange


from calpit.nn.models import MLP
from calpit.nn.utils import count_parameters, RandomDataset, EarlyStopping
from calpit.metrics import probability_integral_transform
from calpit.utils import trapz_grid


class CalPit:
    def __init__(self, model, input_dim=None, hidden_layers=None, **args):
        """
        Initializes an instance of the CalPit Class.

        Args:
            model (str or torch.nn.Module): The model to be used to learn the conditional PIT.
            Can be any pytorch model that outputs a value between 0 and 1.
            A string with the name of an inbuilt can also be provided. Currently supports: `mlp`

            input_dim (int, optional): The input dimension of the model. Defaults to None.
            hidden_layers (list, optional): A list of hidden layer sizes for the MLP models. Defaults to None.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model == "mlp":
            self.model = MLP(input_dim + 1, hidden_layers, 1).to(self.device)
        else:
            self.model = model.to(self.device)

        count_parameters(self.model)

        self.training_loss = None
        self.validation_bce = None
        self.val_loss_min = None

    def fit(
        self,
        x_calib,
        y_calib=None,
        cde_calib=None,
        y_grid=None,
        pit_calib=None,
        oversample=1,
        n_cov_val=201,
        patience=20,
        n_epochs=1000,
        lr=0.001,
        weight_decay=1e-5,
        batch_size=2048,
        frac_train=0.9,
        lr_decay=0.99,
        trace_func=print,
        seed=299792458,
        num_workers=1,
        checkpt_path="_results/checkpoint_.pt",
    ):
        """
        Train the model using the calibration data.

        Args:
            x_calib (numpy.ndarray): The input features for calibration data.
            y_calib (numpy.ndarray, optional): The target values for calibration data.
            cde_calib (numpy.ndarray, optional): The conditional density estimates for calibration.
            y_grid (numpy.ndarray, optional): The grid of target values for calibration.
            pit_calib (numpy.ndarray, optional): The probability integral transforms for the given CDEs evaluated at y_calib.
            Either pit_calib or y_calib, cde_calib and y_grid must be provided.
            oversample (int, optional): The oversampling factor for the training data. Default is 1.
            This is used to upsample the number of coverage values used for training.
            n_cov_val (int, optional): The number of coverage values to use for validation. Default is 201.
            patience (int, optional): The number of epochs to wait for improvement in validation loss before early stopping. Default is 20.
            n_epochs (int, optional): The maximum number of epochs for training. Default is 1000.
            lr (float, optional): The initial learning rate for the optimizer (AdamW). Default is 0.001.
            weight_decay (float, optional): The weight decay for the optimizer. Default is 1e-5.
            batch_size (int, optional): The batch size for training and validation. Default is 2048.
            frac_train (float, optional): The fraction of data to use for training.
            The rest is used for the validation set used to determine when to stop training. Default is 0.9.
            lr_decay (float, optional): The learning rate decay factor for the rule,
            learning_rate(epoch) = lr*lr_decay ** epoch. Default is 0.99.
            trace_func (function, optional): The function used for printing training progress. Default is print.
            seed (int, optional): The random seed for reproducibility. Default is 299792458.
            num_workers (int, optional): The number of CPU worker threads for data loading. Default is 1.
            checkpt_path (str, optional): The path to save the checkpoint of the best model. Default is "_results/checkpoint_.pt".

        Returns:
            torch.nn.Module: The trained model.
        """
        # method implementation
        if pit_calib is None:
            if y_calib is None or cde_calib is None or y_grid is None:
                raise ValueError("Either pit_calib or, y_calib, cde_calib and y_grid must be provided")
            pit_calib = probability_integral_transform(cde_calib, y_grid, y_calib)

        cov_grid = np.linspace(0.001, 0.999, n_cov_val)
        # Split into train and valid sets
        train_size = int(frac_train * len(x_calib))
        valid_size = len(x_calib) - train_size

        rnd_idx = np.random.default_rng(seed=seed).permutation(len(x_calib))
        x_train_rnd = x_calib[rnd_idx[:train_size]]
        x_val_rnd = x_calib[rnd_idx[train_size:]]
        pit_train_rand = pit_calib[rnd_idx[:train_size]]
        pit_val_rand = pit_calib[rnd_idx[train_size:]]

        # Creat randomized Data set for training
        trainset = RandomDataset(x_train_rnd, pit_train_rand, oversample=oversample)

        # Create static dataset for validation
        feature_val = torch.cat(
            [
                torch.Tensor(np.repeat(cov_grid, len(x_val_rnd)))[:, None],
                torch.Tensor(np.tile(x_val_rnd, (n_cov_val, 1))),
            ],
            dim=-1,
        )
        target_val = torch.Tensor(
            np.tile(pit_val_rand, n_cov_val) <= np.repeat(cov_grid, len(x_val_rnd))
        ).float()[:, None]

        validset = TensorDataset(feature_val, target_val)

        # Create Data loader
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Initialize the Model and optimizer, etc.
        training_loss = []
        validation_bce = []

        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # Use lr decay
        schedule_rule = lambda epoch: lr_decay**epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule_rule)
        checkpt_path = Path(checkpt_path)
        checkpt_path.parent.mkdir(parents=True, exist_ok=True)
        early_stopping = EarlyStopping(
            patience=patience, verbose=True, path=checkpt_path, trace_func=trace_func
        )

        # Training loop for all epochs
        for epoch in range(1, n_epochs + 1):
            training_loss_batch = []
            validation_mse_batch = []
            validation_bce_batch = []

            # Training loop per epoch
            self.model.train()  # prep model for training
            for batch, (feature, target) in enumerate(train_dataloader, start=1):
                feature = feature.to(self.device)
                target = target.to(self.device)

                # Zero your gradients for every batch!
                optimizer.zero_grad()
                # Make predictions for this batch
                output = self.model(feature.float())

                # Compute the loss and its gradients

                loss_fn = torch.nn.BCELoss(reduction="sum")
                loss = loss_fn(torch.clamp(torch.squeeze(output), min=0.0, max=1.0), torch.squeeze(target))

                loss.backward()
                # Adjust learning weights
                optimizer.step()

                # record training loss
                training_loss_batch.append(loss.item())

            # Validation
            self.model.eval()  # prep model for evaluation

            for feature, target in valid_dataloader:

                feature = feature.to(self.device)
                target = target.to(self.device)

                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(feature.float())

                # calculate the loss
                mse = ((output - target.float()) ** 2).sum()
                # record validation loss
                validation_mse_batch.append(mse.item())

                criterion = torch.nn.BCELoss(reduction="sum")
                bce = criterion(torch.clamp(torch.squeeze(output), min=0, max=1), torch.squeeze(target))
                validation_bce_batch.append(bce.item())

            # calculate average loss over an epoch
            train_loss_epoch = np.sum(training_loss_batch) / (train_size * oversample)
            valid_bce_epoch = np.sum(validation_bce_batch) / (valid_size * n_cov_val)
            training_loss.append(train_loss_epoch)
            validation_bce.append(valid_bce_epoch)

            epoch_len = len(str(n_epochs))
            # print training/validation statistics
            msg = (
                f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] | "
                + f"train_loss: {train_loss_epoch:.5f} |"
                + f"valid_bce: {valid_bce_epoch:.5f} | "
            )

            trace_func(msg)

            # change the lr
            scheduler.step()

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_bce_epoch, self.model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        # # load the last checkpoint with the best model
        self.model.load_state_dict(torch.load(checkpt_path))
        self.training_loss = np.array(training_loss)
        self.validation_bce = np.array(validation_bce)
        self.val_loss_min = early_stopping.val_loss_min
        return self.model

    def predict(self, x_test, cov_grid, batch_size=2048):
        """
        Predicts the conditional PIT values for the given test data and coverage grid.

        Args:
            x_test (numpy.ndarray): The input features of the test data.
            cov_grid (numpy.ndarray): The coverage grid at which the PIT values are to be evaluated.
            batch_size (int, optional): The batch size for prediction. Defaults to 2048.

        Returns:
            numpy.ndarray: The predicted conditional PIT values.
        """
        self.model.eval()
        self.model.to(self.device)

        pred_pit = []
        n_test = len(x_test)
        n_cov = len(cov_grid)
        n_batches = (n_test - 1) // batch_size + 1

        for i in trange(n_batches):
            x = x_test[i * batch_size : (i + 1) * batch_size]
            if cov_grid.ndim == 1:
                with torch.no_grad():
                    pred_pit_batch = (
                        self.model(
                            torch.Tensor(
                                np.hstack([np.repeat(cov_grid, len(x))[:, None], np.tile(x, (n_cov, 1))])
                            ).to(self.device)
                        )
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(n_cov, -1)
                        .T
                    )
            elif cov_grid.ndim == 2:
                c = cov_grid[i * batch_size : (i + 1) * batch_size]
                with torch.no_grad():
                    pred_pit_batch = (
                        self.model(
                            torch.Tensor(
                                np.hstack([np.ravel(c)[:, None], np.repeat(x, c.shape[1], axis=0)])
                            ).to(self.device)
                        )
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(len(x), -1)
                    )

            pred_pit_batch[pred_pit_batch < 0] = 0
            pred_pit_batch[pred_pit_batch > 1] = 1
            pred_pit.extend(pred_pit_batch)
        return np.array(pred_pit)

    def transform(self, x_test, cde_test, y_grid, batch_size=2048):
        """
        Transforms the input CDEs for a test data set to calibrated CDEs.

        Args:
            x_test (array-like): The input features of the test data.
            cde_test (array-like): The initial CDEs for the test data that is to be transformed.
            y_grid (array-like): The grid of values for the CDEs.
            batch_size (int, optional): The batch size for prediction. Defaults to 2048.

        Returns:
            numpy.ndarray: The transformed CDEs for the given.
        """
        cdf_test = trapz_grid(cde_test, y_grid)
        cdf_test_new = self.predict(x_test, cov_grid=cdf_test, batch_size=batch_size)
        cdf_funct = PchipInterpolator(y_grid, cdf_test_new, extrapolate=True, axis=1)
        pdf_func = cdf_funct.derivative(1)
        cde_test_new = pdf_func(y_grid)
        return cde_test_new

    def fit_transform(self, **args):
        """Fit the model and transform the data in one go"""
        raise NotImplementedError
