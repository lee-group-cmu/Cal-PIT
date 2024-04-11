import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from .utils import get_pit
from .nn.nn_utils import CalPitLitModule, RandomDataset


class CalPitModel:
    def __init__(self, model, x_calib, y_calib=None, cde_calib=None, cde_grid=None, pit_values_calib=None):
        self.model = model
        self.x_calib = x_calib
        if pit_values_calib is None:
            if y_calib is None or cde_calib is None or cde_grid is None:
                raise ValueError(
                    "Either pit_values_calib or y_calib and cde_calib and cde_grid must be provided"
                )
        if pit_values_calib is not None:
            self.pit_values_calib = pit_values_calib
        else:
            self.y_calib = y_calib
            self.cde_calib = cde_calib
            self.cde_grid = cde_grid
            self.pit_values_calib = get_pit(self.cde_calib, self.cde_grid, self.y_calib)

    def fit(
        self,
        batch_size,
        frac_train=0.8,
        model_path="./",
        alphas_grid=np.linspace(0, 1, 100),
        oversample=50,
        lr=1e-3,
        lr_decay=0.95,
        patience=10,
        num_workers=1,
        accelerator="gpu",
        devices=1,
        **kwargs
    ):
        self.model = CalPitLitModule(self.model, lr=lr, lr_decay=lr_decay)
        train_size = int(frac_train * len(self.x_calib))

        rnd_idx = np.random.default_rng().permutation(len(self.x_calib))
        x_train_rnd = self.x_calib[rnd_idx[:train_size]]
        x_val_rnd = self.x_calib[rnd_idx[train_size:]]
        pit_train_rand = self.pit_values_calib[rnd_idx[:train_size]]
        pit_val_rand = self.pit_values_calib[rnd_idx[train_size:]]

        # Creat randomized Data set for training
        trainset = RandomDataset(x_train_rnd, pit_train_rand, oversample=oversample)

        # Create static dataset for validation
        feature_val = torch.cat(
            [
                torch.Tensor(np.repeat(alphas_grid, len(x_val_rnd)))[:, None],
                torch.Tensor(np.tile(x_val_rnd, (len(alphas_grid), 1))),
            ],
            dim=-1,
        )
        target_val = torch.Tensor(
            np.tile(pit_val_rand, len(alphas_grid)) <= np.repeat(alphas_grid, len(x_val_rnd))
        ).float()

        validset = TensorDataset(feature_val, target_val)

        # Create Data loader
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=0.00, patience=patience, verbose=True, mode="min", check_finite=True
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=model_path,
            save_top_k=1,  # save the best model
            mode="min",
            every_n_epochs=1,
            save_last=True,
        )

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            callbacks=[early_stop_callback, checkpoint_callback],
            check_val_every_n_epoch=1,
            enable_checkpointing=True,
        )

        trainer.fit(model=self.model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

        self.model = self.model.load_from_checkpoint(
            checkpoint_callback.best_model_path, model=self.model.network, lr=lr, lr_decay=lr_decay
        )

        return self.model

    def predict_pit(self, X_test, n_gamma, gamma_grid):
        trainer = Trainer(gpus=1, ...) 
        model = plModel.load_from_checkpoint('lightning_logs/version_0/checkpoints/model_epoch=17_val_loss=95.99.ckpt')
        # not needed
        # model.eval()

        trainer.test(model, test_dataloaders)

        # move the rest to the LightningModule:

        class PLModel: 

        def __init__(self, ...)
        self.accuracy = pl.metrics.Accuracy()        

        def test_step(self, batch, batch_idx)
            data = batch
            correct_count = 0
            nsample = len(test_loader)
            output = self(data['x'])
            label = data['label'].data.cpu().numpy()  # optimize this and use torch operations on the gpu instead of numpy
            
            # pred = nn.functional.softmax(output, dim=1)
            # pred = np.argmax(pred.data.cpu().numpy(), axis = 1) 

            # use the Metrics [1] package instead, you can directly feed it logits
            # and it correctly works across multiple gpus should you need that.
            self.accuracy(output, label)

    def test_epoch_end(self, outputs):
        # log or print
        self.log('test_accuracy', self.accuracy.compute())

    def predict_cde(self, X_test, y_grid, n_grid):
        pass
