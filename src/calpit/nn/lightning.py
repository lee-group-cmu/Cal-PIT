"""PyTorch Lightning training of Cal-PIT networks.

CalPIT.fit uses fit_lightning. Use CalPITModule and BestWeightsEarlyStopping
directly to train with your own Trainer, loggers or callbacks, then wrap the
trained network with CalPIT.from_fitted.
"""

import contextlib
import copy
import functools
import logging
import warnings
from collections.abc import Iterator, Mapping
from typing import Any

import lightning
import numpy as np
import numpy.typing as npt
import torch
from lightning.pytorch.utilities import types as lightning_types
from torch.utils import data

from calpit.nn import data as calpit_data
from calpit.nn import models, training

_LIGHTNING_LOGGERS = ("lightning.pytorch", "lightning.fabric")


def _exponential_decay(lr_decay: float, epoch: int) -> float:
    return lr_decay**epoch


class _LossSums:
    """Running sum of per-batch summed losses and of the rows they cover."""

    def __init__(self) -> None:
        self.losses: list[torch.Tensor] = []
        self.n_rows = 0

    def add(self, loss: torch.Tensor, n_rows: int) -> None:
        self.losses.append(loss.detach())
        self.n_rows += n_rows

    def mean(self, trainer: lightning.Trainer) -> float:
        """Returns the mean loss per row, over all processes when training on several."""
        # Each batch's float32 sum is added in float64, keeping the GPU free of
        # a synchronizing .item() per batch.
        total = float(np.sum(torch.stack(self.losses).double().cpu().numpy())) if self.losses else 0.0
        n_rows = float(self.n_rows)
        if trainer.world_size > 1:
            sums = torch.tensor([total, n_rows], dtype=torch.float64, device=trainer.strategy.root_device)
            total, n_rows = trainer.strategy.reduce(sums, reduce_op="sum").tolist()
        return total / n_rows if n_rows else float("nan")


class CalPITModule(lightning.LightningModule):
    """LightningModule that trains a Cal-PIT network on (alpha, x, target) batches.

    The loss is the binary cross entropy summed over each batch
    (calpit.nn.coverage_loss). The optimizer is AdamW, with the learning rate
    lr * lr_decay**epoch. Override configure_optimizers to change either.

    Args:
        model: The network, which follows the contract in calpit.nn.models.
        lr: The initial learning rate.
        weight_decay: The AdamW weight decay.
        lr_decay: The factor the learning rate shrinks by every epoch.

    Attributes:
        model: The network.
        lr: The initial learning rate.
        weight_decay: The AdamW weight decay.
        lr_decay: The learning-rate decay per epoch.
        train_loss_history: The mean training loss per row of every epoch.
        val_bce_history: The mean validation binary cross entropy per row of
            every epoch.
    """

    def __init__(
        self, model: torch.nn.Module, lr: float = 1e-3, weight_decay: float = 1e-5, lr_decay: float = 0.99
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.output_type = models.output_type(model)
        self.train_loss_history: list[float] = []
        self.val_bce_history: list[float] = []
        self._train_sums = _LossSums()
        self._val_sums = _LossSums()

    def forward(self, alpha: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the network.

        Args:
            alpha: The coverage levels, shape (batch,).
            x: The features, shape (batch, n_features).

        Returns:
            The network's output, shape (batch,).
        """
        return self.model(alpha, x)

    def _summed_loss(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        alpha, x, target = batch
        return training.coverage_loss(self.model(alpha, x), target, self.output_type, reduction="sum")

    def on_train_epoch_start(self) -> None:
        """Starts the sum of the training loss."""
        self._train_sums = _LossSums()

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Returns the summed loss of a batch."""
        del batch_idx  # Unused.
        loss = self._summed_loss(batch)
        self._train_sums.add(loss, len(batch[2]))
        return loss

    def on_train_epoch_end(self) -> None:
        """Records and logs the mean training loss per row as train_loss."""
        self.train_loss_history.append(self._train_sums.mean(self.trainer))
        self.log("train_loss", self.train_loss_history[-1])

    def on_validation_epoch_start(self) -> None:
        """Starts the sum of the validation loss."""
        self._val_sums = _LossSums()

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Adds the summed binary cross entropy of a batch."""
        del batch_idx  # Unused.
        self._val_sums.add(self._summed_loss(batch), len(batch[2]))

    def on_validation_epoch_end(self) -> None:
        """Records and logs the mean validation binary cross entropy per row as val_bce."""
        if not self.trainer.sanity_checking:
            self.val_bce_history.append(self._val_sums.mean(self.trainer))
            self.log("val_bce", self.val_bce_history[-1])

    def configure_optimizers(self) -> lightning_types.OptimizerLRSchedulerConfig:
        """Returns AdamW with the learning rate decaying every epoch."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=functools.partial(_exponential_decay, self.lr_decay)
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


class BestWeightsEarlyStopping(lightning.Callback):
    """Stops training when the validation loss stops improving and restores the best weights.

    It monitors the val_bce_history of a CalPITModule, in float64 and summed
    over all processes when training on several devices. An epoch
    counts as an improvement when its validation loss is at most the
    best so far. Training stops after `patience` epochs in a row without one.
    The best weights are kept in memory, not written to disk, and loaded back
    when training ends.

    Args:
        patience: The number of epochs without improvement to stop after.

    Attributes:
        patience: The number of epochs without improvement to stop after.
        best_score: The lowest validation loss so far, or None before the
            first validation.
        wait_count: The number of epochs since the last improvement.
        best_state: A copy of the state dict at the best epoch, or None.
    """

    def __init__(self, patience: int = 20) -> None:
        super().__init__()
        self.patience = patience
        self.best_score: float | None = None
        self.wait_count = 0
        self.best_state: dict[str, torch.Tensor] | None = None

    def on_validation_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule) -> None:
        """Compares the epoch's validation loss with the best and stops if it is time."""
        if trainer.sanity_checking:
            return
        if not isinstance(pl_module, CalPITModule):
            raise TypeError(f"BestWeightsEarlyStopping needs a CalPITModule: {type(pl_module)=}")
        score = pl_module.val_bce_history[-1]
        if self.best_score is None or score <= self.best_score:
            self.best_score = score
            self.wait_count = 0
            self.best_state = copy.deepcopy(pl_module.state_dict())
        else:
            self.wait_count += 1
        # The validation loss is summed over all processes, so they agree; the
        # reduction makes sure every process stops at the same epoch.
        should_stop = trainer.strategy.reduce_boolean_decision(self.wait_count >= self.patience, all=False)
        trainer.should_stop = trainer.should_stop or should_stop

    def on_fit_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule) -> None:
        """Loads the best weights back into the module."""
        del trainer  # Unused.
        if self.best_state is not None:
            pl_module.load_state_dict(self.best_state)

    def state_dict(self) -> dict[str, Any]:
        """Returns the callback's state for Lightning checkpoints."""
        return {"best_score": self.best_score, "wait_count": self.wait_count}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restores the callback's state from a Lightning checkpoint."""
        self.best_score = state_dict["best_score"]
        self.wait_count = state_dict["wait_count"]


def fit_lightning(
    model: torch.nn.Module,
    x_train: npt.ArrayLike,
    pit_train: npt.ArrayLike,
    x_val: npt.ArrayLike | None = None,
    pit_val: npt.ArrayLike | None = None,
    *,
    alpha_val: npt.ArrayLike | None = None,
    oversample: float = 1,
    batch_size: int = 2048,
    max_epochs: int = 1000,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    lr_decay: float = 0.99,
    patience: int = 20,
    num_workers: int = 0,
    verbose: bool = False,
    trainer_kwargs: Mapping[str, Any] | None = None,
) -> tuple[CalPITModule, lightning.Trainer]:
    """Trains a Cal-PIT network with a Lightning Trainer.

    The training set draws a fresh alpha for every item each epoch
    (CoverageDataset). With a validation set, every validation object is
    scored at every alpha_val (CoverageGridDataset), training stops early and
    the best weights are restored (BestWeightsEarlyStopping).

    Args:
        model: The network, trained in place.
        x_train: The training features, shape (n_train, n_features).
        pit_train: The training PIT values, shape (n_train,).
        x_val: The validation features, shape (n_val, n_features), or None to
            train for max_epochs without validation.
        pit_val: The validation PIT values, shape (n_val,).
        alpha_val: The validation coverage levels, shape (n_alpha,). None uses
            201 levels from 0.001 to 0.999.
        oversample: How many training items each object yields per epoch.
        batch_size: The number of rows per batch.
        max_epochs: The maximum number of epochs.
        lr: The initial learning rate.
        weight_decay: The AdamW weight decay.
        lr_decay: The factor the learning rate shrinks by every epoch.
        patience: The number of epochs without improvement to stop after.
        num_workers: The number of DataLoader worker processes.
        verbose: Whether to show Lightning's progress bar, model summary and
            messages.
        trainer_kwargs: Arguments for lightning.Trainer that override the
            defaults: accelerator "auto", one device, no logger, no checkpoint
            files. Callbacks given here are added to calpit's own.

    Returns:
        The trained module and the trainer, as a tuple (module, trainer).
    """
    if alpha_val is None:
        alpha_val = np.linspace(0.001, 0.999, 201)
    module = CalPITModule(model, lr=lr, weight_decay=weight_decay, lr_decay=lr_decay)
    train_loader = calpit_data.CoverageDataset(x_train, pit_train, oversample=oversample).batches(
        batch_size, num_workers=num_workers
    )
    callbacks: list[lightning.Callback] = []
    val_loader = None
    if x_val is not None and pit_val is not None and len(np.asarray(x_val)):
        val_loader = data.DataLoader(
            calpit_data.CoverageGridDataset(x_val, pit_val, alpha_val, batch_size=batch_size),
            batch_size=None,
            num_workers=num_workers,
        )
        callbacks.append(BestWeightsEarlyStopping(patience=patience))
    options: dict[str, Any] = {
        "max_epochs": max_epochs,
        "accelerator": "auto",
        "devices": 1,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": verbose,
        "enable_model_summary": verbose,
        "num_sanity_val_steps": 0,
    }
    options.update(trainer_kwargs or {})
    options["callbacks"] = callbacks + list(options.get("callbacks") or [])
    with _quiet(not verbose):
        trainer = lightning.Trainer(**options)
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return module, trainer


@contextlib.contextmanager
def _quiet(enabled: bool) -> Iterator[None]:
    """Silences Lightning's messages and warnings while enabled."""
    if not enabled:
        yield
        return
    levels = {name: logging.getLogger(name).level for name in _LIGHTNING_LOGGERS}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="lightning")
        warnings.filterwarnings("ignore", module="torch.utils.data")
        for name in _LIGHTNING_LOGGERS:
            logging.getLogger(name).setLevel(logging.ERROR)
        try:
            yield
        finally:
            for name, level in levels.items():
                logging.getLogger(name).setLevel(level)
