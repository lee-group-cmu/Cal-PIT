"""CalPIT: diagnose and recalibrate conditional density estimates, scikit-learn style."""

import copy
import inspect
import sys
from collections.abc import Mapping
from typing import Any, Self, TypeVar

import numpy as np
import numpy.typing as npt

from calpit import _optional, _sklearn_backend, coverage, diagnostics, representations

FloatArray = npt.NDArray[np.floating]

DEFAULT_RANDOM_STATE = 299792458

_Cde = TypeVar("_Cde", bound=representations.ConditionalDensities)


class NotFittedError(ValueError, AttributeError):
    """CalPIT was used before it was fitted, as in scikit-learn."""


def _is_torch_module(model: object) -> bool:
    # A torch module can only exist once torch is imported, so this never imports torch itself.
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(model, torch.nn.Module)


def _default_model(n_features: int) -> Any:
    _optional.import_optional("torch", "torch", "CalPIT's default model")
    nn = _optional.import_optional("calpit.nn", "torch", "CalPIT's default model")
    return nn.IsplineNN(n_features)


class CalPIT:
    """Diagnoses and recalibrates conditional density estimates (CDEs) with Cal-PIT.

    Cal-PIT learns r(alpha; x) = P(PIT <= alpha | x), the conditional CDF of
    the probability integral transform, by regressing the indicator
    PIT <= alpha on (alpha, x) over a calibration set. Its predictions give
    local P-P plots that diagnose the CDEs at any x (predict, diagnose), and
    recalibrate them by composing each CDF with r (transform).

    The model can be a PyTorch network, trained with PyTorch Lightning, or a
    scikit-learn classifier. A network follows the contract in calpit.nn.models:
    forward(alpha, x) returns the logit of r, or r itself if its `output`
    attribute is "probability". A classifier is fitted on rows [alpha, x]
    with n_alpha stratified alpha per object (calpit.coverage).

    The estimator follows the scikit-learn API: the constructor only stores
    its arguments, get_params and set_params work with sklearn.base.clone,
    fit returns the estimator, and what fit learns ends in an underscore. The
    model passed in is copied, never trained in place.

    Args:
        model: A torch.nn.Module; a callable that takes n_features and
            returns one, such as functools.partial(calpit.nn.MLP,
            hidden_layers=[64, 64]); a scikit-learn classifier with
            predict_proba; or None for calpit.nn.IsplineNN(n_features).
        rearrange: Whether to sort the predictions of r along alpha so they
            are non-decreasing (calpit.coverage.rearrange), which
            CalPIT.transform needs. Predictions that are already
            non-decreasing, such as those of IsplineNN or MonotonicNN, are
            unchanged.
        val_fraction: Torch only. The fraction of calibration objects held out
            to stop training early; 0 trains for max_epochs.
        n_alpha_val: Torch only. The number of alpha, from 0.001 to 0.999, every
            validation object is scored at.
        oversample: Torch only. How many training items each object yields per
            epoch, each with a fresh alpha.
        batch_size: Torch only. The number of (alpha, x) rows per training
            batch.
        predict_batch_size: The number of objects per prediction batch; each
            batch has predict_batch_size * n_alpha rows.
        max_epochs: Torch only. The maximum number of epochs.
        lr: Torch only. The initial AdamW learning rate.
        weight_decay: Torch only. The AdamW weight decay.
        lr_decay: Torch only. The learning rate is lr * lr_decay**epoch.
        patience: Torch only. The number of epochs without improvement in the
            validation loss to stop after.
        num_workers: Torch only. The number of DataLoader worker processes.
        trainer_kwargs: Torch only. Arguments for lightning.Trainer, such as
            accelerator, devices, precision, logger or callbacks, that override
            calpit's defaults.
        n_alpha: Scikit-learn only. The number of stratified alpha per
            calibration object; the classifier is fitted on
            n_objects * n_alpha rows. Predictions are made at n_alpha + 1
            evenly spaced alpha and interpolated linearly in between.
        random_state: The seed of the train/validation split, the alpha draws,
            the model initialization of a factory or the default model, and
            the torch random numbers in training. None leaves them unseeded.
        verbose: Whether to show training progress.

    Attributes:
        model_: The fitted network or classifier.
        backend_: "torch" or "sklearn".
        n_features_in_: The number of features seen in fit.
        train_loss_: Torch only. The training loss per epoch.
        val_bce_: Torch only. The validation binary cross entropy per epoch.
        best_val_bce_: Torch only. The lowest validation loss, whose weights
            model_ has.
        device_: Torch only. The device the network was trained on.
    """

    def __init__(
        self,
        model: Any = None,
        *,
        rearrange: bool = True,
        val_fraction: float = 0.1,
        n_alpha_val: int = 201,
        oversample: float = 1,
        batch_size: int = 2048,
        predict_batch_size: int = 2048,
        max_epochs: int = 1000,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        lr_decay: float = 0.99,
        patience: int = 20,
        num_workers: int = 0,
        trainer_kwargs: Mapping[str, Any] | None = None,
        n_alpha: int = 50,
        random_state: int | None = DEFAULT_RANDOM_STATE,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.rearrange = rearrange
        self.val_fraction = val_fraction
        self.n_alpha_val = n_alpha_val
        self.oversample = oversample
        self.batch_size = batch_size
        self.predict_batch_size = predict_batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.patience = patience
        self.num_workers = num_workers
        self.trainer_kwargs = trainer_kwargs
        self.n_alpha = n_alpha
        self.random_state = random_state
        self.verbose = verbose

    @classmethod
    def from_fitted(cls, model: Any, **params: Any) -> Self:
        """Wraps a network or classifier that was trained outside CalPIT.

        Use it after a hand-written training loop: the result predicts,
        diagnoses and transforms like a fitted CalPIT. The model is used as
        is, not copied.

        Args:
            model: A trained network that follows the contract in
                calpit.nn.models, or a classifier fitted on [alpha, x] rows
                with boolean targets PIT <= alpha.
            **params: Other constructor arguments, such as rearrange or
                predict_batch_size.

        Returns:
            The fitted CalPIT.
        """
        estimator = cls(model=model, **params)
        estimator.model_ = model
        if _is_torch_module(model):
            estimator.backend_ = "torch"
            parameter = next(model.parameters(), None)
            estimator.device_ = None if parameter is None else parameter.device
        elif _sklearn_backend.is_sklearn_classifier(model):
            estimator.backend_ = "sklearn"
            if hasattr(model, "n_features_in_"):
                estimator.n_features_in_ = model.n_features_in_ - 1
        else:
            raise TypeError(f"model must be a torch.nn.Module or a scikit-learn classifier: {type(model)=}")
        return estimator

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Returns the constructor arguments, as scikit-learn estimators do.

        Args:
            deep: Unused; there are no nested estimators to expand.

        Returns:
            The arguments by name.
        """
        del deep  # Unused.
        return {name: getattr(self, name) for name in inspect.signature(type(self)).parameters}

    def set_params(self, **params: Any) -> Self:
        """Sets constructor arguments, as scikit-learn estimators do.

        Args:
            **params: The arguments to set, by name.

        Returns:
            The estimator.
        """
        valid = inspect.signature(type(self)).parameters
        for name, value in params.items():
            if name not in valid:
                raise ValueError(f"invalid parameter for CalPIT: {name=}")
            setattr(self, name, value)
        return self

    def __repr__(self) -> str:
        """Returns the class name and the arguments that differ from the defaults."""
        defaults = inspect.signature(type(self)).parameters
        changed = [
            f"{name}={value!r}"
            for name, value in self.get_params().items()
            if value is not defaults[name].default and value != defaults[name].default
        ]
        return f"{type(self).__name__}({', '.join(changed)})"

    def __sklearn_is_fitted__(self) -> bool:
        """Returns whether fit has run, for sklearn.utils.validation.check_is_fitted."""
        return hasattr(self, "model_")

    def _check_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise NotFittedError("this CalPIT is not fitted yet; call fit or from_fitted first")

    def fit(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        cde: representations.ConditionalDensities | None = None,
        *,
        pit: npt.ArrayLike | None = None,
    ) -> Self:
        """Learns r(alpha; x) from a calibration set.

        Args:
            x: The features, shape (n_objects, n_features).
            y: The true values, shape (n_objects,).
            cde: The CDEs of the calibration objects: a calpit.GridCDE,
                QuantileCDE or SampleCDE (calpit.qp_io.from_qp converts a qp
                ensemble).
            pit: The PIT values, shape (n_objects,), in place of y and cde.

        Returns:
            The fitted estimator.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"x must have shape (n_objects, n_features): {x.shape=}")
        if pit is None:
            if y is None or cde is None:
                raise ValueError("pass either pit, or both y and cde")
            pit = cde.pit(y)
        pit = np.asarray(pit, dtype=float)
        if pit.shape != (len(x),):
            raise ValueError(f"pit must have shape (n_objects,): {pit.shape=}, {x.shape=}")
        self.n_features_in_ = x.shape[1]
        if _sklearn_backend.is_sklearn_classifier(self.model):
            self.backend_ = "sklearn"
            self.model_ = _sklearn_backend.fit_sklearn(self.model, x, pit, self.n_alpha, self.random_state)
        else:
            self._fit_torch(x, pit)
        return self

    def _fit_torch(self, x: np.ndarray, pit: np.ndarray) -> None:
        if self.model is not None and not _is_torch_module(self.model) and not callable(self.model):
            raise TypeError(
                "model must be a torch.nn.Module, a scikit-learn classifier, a callable or None: "
                f"{type(self.model)=}"
            )
        torch = _optional.import_optional("torch", "torch", "CalPIT with a PyTorch model")
        calpit_lightning = _optional.import_optional(
            "calpit.nn.lightning", "torch", "CalPIT with a PyTorch model"
        )
        with torch.random.fork_rng(enabled=self.random_state is not None):
            if self.random_state is not None:
                torch.manual_seed(self.random_state)
            if self.model is None:
                model = _default_model(x.shape[1])
            elif _is_torch_module(self.model):
                model = copy.deepcopy(self.model)
            else:
                model = self.model(x.shape[1])
            x_train, pit_train, x_val, pit_val = coverage.train_val_split(
                x, pit, self.val_fraction, self.random_state
            )
            module, trainer = calpit_lightning.fit_lightning(
                model,
                x_train,
                pit_train,
                x_val,
                pit_val,
                alpha_val=np.linspace(0.001, 0.999, self.n_alpha_val),
                oversample=self.oversample,
                batch_size=self.batch_size,
                max_epochs=self.max_epochs,
                lr=self.lr,
                weight_decay=self.weight_decay,
                lr_decay=self.lr_decay,
                patience=self.patience,
                num_workers=self.num_workers,
                verbose=self.verbose,
                trainer_kwargs=self.trainer_kwargs,
            )
        self.backend_ = "torch"
        self.model_ = module.model
        self.device_ = trainer.strategy.root_device
        self.train_loss_ = np.array(module.train_loss_history)
        self.val_bce_ = np.array(module.val_bce_history)
        self.best_val_bce_ = float(np.min(self.val_bce_)) if len(self.val_bce_) else None

    def _predict_device(self) -> Any:
        device = getattr(self, "device_", None)
        if device is not None and device.type == "cuda":
            torch = sys.modules["torch"]
            if not torch.cuda.is_available():
                return torch.device("cpu")
        return device

    def predict(self, x: npt.ArrayLike, alpha: npt.ArrayLike | None = None) -> FloatArray:
        """Predicts r(alpha; x) = P(PIT <= alpha | x), the local P-P curves.

        Args:
            x: The features, shape (n_objects, n_features).
            alpha: The coverage levels, shape (n_alpha,) for the same levels
                for every object, or (n_objects, n_alpha). None uses 101 levels
                from 0 to 1.

        Returns:
            The predicted PIT CDF in [0, 1], shape (n_objects, n_alpha),
            rearranged to be non-decreasing in alpha unless rearrange is False.
        """
        self._check_fitted()
        x = np.asarray(x, dtype=float)
        alpha = np.linspace(0.0, 1.0, 101) if alpha is None else np.asarray(alpha, dtype=float)
        if self.backend_ == "sklearn":
            pit_cdf = _sklearn_backend.predict_sklearn(
                self.model_, x, alpha, self.n_alpha, self.predict_batch_size
            )
        else:
            calpit_nn = _optional.import_optional("calpit.nn", "torch", "CalPIT with a PyTorch model")
            pit_cdf = calpit_nn.predict_pit_cdf(
                self.model_, x, alpha, self.predict_batch_size, self._predict_device()
            )
        return coverage.rearrange(pit_cdf, alpha) if self.rearrange else pit_cdf

    def diagnose(self, x: npt.ArrayLike, alpha: npt.ArrayLike | None = None) -> diagnostics.LocalCalibration:
        """Predicts the local P-P curves and how far they stray from the diagonal.

        Args:
            x: The features, shape (n_objects, n_features).
            alpha: The increasing coverage levels, shape (n_alpha,). None uses
                101 levels from 0 to 1.

        Returns:
            The local calibration of the CDEs at each x.
        """
        alpha = np.linspace(0.0, 1.0, 101) if alpha is None else np.ravel(np.asarray(alpha, dtype=float))
        return diagnostics.LocalCalibration(alpha=alpha, pit_cdf=self.predict(x, alpha))

    def transform(self, x: npt.ArrayLike, cde: _Cde) -> _Cde:
        """Recalibrates CDEs.

        Args:
            x: The features, shape (n_objects, n_features).
            cde: The CDEs to recalibrate, one per row of x.

        Returns:
            The recalibrated CDEs, in the same representation as cde.
        """
        if len(cde) != len(np.asarray(x)):
            raise ValueError(f"cde and x must have one row per object: {len(cde)=}, {len(np.asarray(x))=}")
        return cde.recalibrate(self.predict(x, cde.recalibration_alpha()))

    def fit_transform(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        cde: _Cde,
    ) -> _Cde:
        """Fits on a calibration set and recalibrates its own CDEs.

        Args:
            x: The features, shape (n_objects, n_features).
            y: The true values, shape (n_objects,).
            cde: The CDEs, one per row of x.

        Returns:
            The recalibrated CDEs, in the same representation as cde.
        """
        return self.fit(x, y, cde).transform(x, cde)
