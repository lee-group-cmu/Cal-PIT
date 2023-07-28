import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
import torch.nn.functional as F


class RandomDataset(Dataset):
    def __init__(self, X, Y, oversample=1):
        self.X = X
        self.Y = Y
        self.len_x = len(X)
        self.oversample = oversample

    def __len__(self):
        return int(len(self.X) * self.oversample)

    def __getitem__(self, idx):
        alpha = torch.rand(1)
        feature = torch.hstack((alpha, torch.Tensor(self.X[idx % self.len_x])))
        # target = (self.Y[idx % self.len_x] <= alpha).float()
        target = self.Y[idx % self.len_x]
        return feature, torch.squeeze(torch.Tensor([target]))


class CalPitLitModule(pl.LightningModule):
    def __init__(self, model, lr=1e-3, lr_decay=0.95):
        super().__init__()

        self.network = model
        self.lr = lr
        self.lr_decay = lr_decay

        self.loss = F.mse_loss
        # self.automatic_optimization = False

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, target = batch
        y_hat = self.network(x)

        alpha = x[:, 0]

        y = (target <= alpha).float()

        loss = self.loss(torch.clip(y_hat, 0, 1), torch.clip(y, 0, 1))
        # Logging to TensorBoard by default
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, target = batch
        y_hat = self.network(x)
        alpha = x[:, 0]
        # y = (target <= alpha).float()
        y = target

        loss = self.loss(torch.clip(y_hat, 0, 1), torch.clip(y, 0, 1))
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_decay = self.lr_decay

        def schedule_rule(epoch):
            return lr_decay**epoch

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule_rule, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def cde_loss(self, cde_estimates, z_grid, z_test):
        if len(z_test.shape) == 1:
            z_test = z_test.reshape(-1, 1)
        if len(z_grid.shape) == 1:
            z_grid = z_grid.reshape(-1, 1)

        n_obs, n_grid = cde_estimates.shape
        n_samples, feats_samples = z_test.shape
        n_grid_points, feats_grid = z_grid.shape

        if n_obs != n_samples:
            raise ValueError(
                "Number of samples in CDEs should be the same as in z_test."
                "Currently %s and %s." % (n_obs, n_samples)
            )
        if n_grid != n_grid_points:
            raise ValueError(
                "Number of grid points in CDEs should be the same as in z_grid."
                "Currently %s and %s." % (n_grid, n_grid_points)
            )

        if feats_samples != feats_grid:
            raise ValueError(
                "Dimensionality of test points and grid points need to coincise."
                "Currently %s and %s." % (feats_samples, feats_grid)
            )

        integrals = torch.trapz(cde_estimates**2, torch.squeeze(z_grid), axis=1)

        nn_ids = torch.argmin(torch.abs(z_grid - z_test.T), axis=0)
        likeli = cde_estimates[(tuple(torch.arange(n_samples)), tuple(nn_ids))]

        losses = integrals - 2 * likeli
        loss = torch.mean(losses)
        se_error = torch.std(losses, axis=0) / (n_obs**0.5)

        return loss, se_error
