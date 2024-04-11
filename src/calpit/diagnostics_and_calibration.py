import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from tqdm import trange


from calpit.nn.models import MLP
from calpit.nn.utils import count_parameters, RandomDataset, EarlyStopping
from calpit.metrics import probability_integral_transform


class CalPit:
    def __init__(self, model, input_size=None, hidden_layers=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model == "mlp":
            self.model = MLP(input_size + 1, 1, hidden_layers).to(self.device)
        else:
            self.model = model.to(self.device)

        count_parameters(self.model)

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
        checkpt_path="./checkpoint_.pt",
    ):

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
        validation_mse = []
        validation_bce = []
        #         validation_weighted_mse = []
        #         cal_loss = []

        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # Use lr decay
        schedule_rule = lambda epoch: lr_decay**epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule_rule)

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
        self.training_loss = training_loss
        self.validation_bce = validation_bce
        return self.model

    def predict(self, x_test, cov_grid, batch_size=2048):
        self.model.eval()
        self.model.to(self.device)

        pred_pit = []
        n_test = len(x_test)
        n_cov = len(cov_grid)
        n_batches = (n_test - 1) // batch_size + 1

        for i in trange(n_batches):
            x = x_test[i * batch_size : (i + 1) * batch_size]
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

            pred_pit_batch[pred_pit_batch < 0] = 0
            pred_pit_batch[pred_pit_batch > 1] = 1
            pred_pit.extend(pred_pit_batch)
        return np.array(pred_pit)

    def transform(self, x_test, y_grid):
        raise NotImplementedError
        return 0

    def fit_transform(self, x_test, y_grid):
        raise NotImplementedError
        return 0
