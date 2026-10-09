"""Cal-PIT with scikit-learn classifiers.

A scikit-learn estimator is fitted once on a fixed data set, so the fresh alpha
per item and epoch of the PyTorch backend is replaced by n_alpha stratified
draws of alpha per object (calpit.coverage.stratified_alpha), expanded into
(alpha, x) rows (calpit.coverage.expand_coverage).
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from calpit import _optional, coverage


def is_sklearn_classifier(model: object) -> bool:
    """Returns whether model looks like a scikit-learn classifier with predict_proba."""
    return all(hasattr(model, name) for name in ("fit", "predict_proba", "get_params"))


def fit_sklearn(
    model: Any,
    x: np.ndarray,
    pit: np.ndarray,
    n_alpha: int,
    random_state: coverage.SeedLike,
) -> Any:
    """Fits a clone of a scikit-learn classifier to the indicators PIT <= alpha.

    If the classifier has a monotonic_cst parameter left at None, as
    HistGradientBoostingClassifier does, the clone is constrained to increase
    in alpha, so its predictions are non-decreasing in alpha without
    rearrangement.

    Args:
        model: The unfitted classifier; it is cloned, not changed.
        x: The features, shape (n_objects, n_features).
        pit: The PIT values, shape (n_objects,).
        n_alpha: The number of stratified alpha per object.
        random_state: The seed or generator of the alpha draws.

    Returns:
        The fitted clone.
    """
    base = _optional.import_optional("sklearn.base", "sklearn", "CalPIT with a scikit-learn model")
    estimator = base.clone(model)
    if "monotonic_cst" in estimator.get_params() and estimator.get_params()["monotonic_cst"] is None:
        estimator.set_params(monotonic_cst=[1] + [0] * x.shape[1])
    alpha = coverage.stratified_alpha(len(x), n_alpha, random_state)
    features, targets = coverage.expand_coverage(x, pit, alpha)
    estimator.fit(features, targets)
    return estimator


def predict_sklearn(
    estimator: Any, x: np.ndarray, alpha: npt.ArrayLike, n_knots: int, batch_size: int = 2048
) -> np.ndarray:
    """Predicts r(alpha; x) = P(PIT <= alpha | x) with a fitted classifier.

    Classifiers such as tree ensembles are piecewise constant in alpha, and a
    staircase in alpha would turn into spikes in the recalibrated densities.
    So r is predicted at n_knots + 1 evenly spaced alpha from 0 to 1 and
    interpolated linearly between them, which keeps it continuous and, for a
    classifier constrained to increase in alpha, monotone.

    Args:
        estimator: The classifier, fitted on [alpha, x] rows with boolean
            targets.
        x: The features, shape (n_objects, n_features).
        alpha: The coverage levels, shape (n_alpha,) or (n_objects, n_alpha).
        n_knots: The number of intervals between the alpha knots.
        batch_size: The number of objects per call to predict_proba.

    Returns:
        The predicted PIT CDF, shape (n_objects, n_alpha).
    """
    alpha = np.asarray(alpha, dtype=float)
    knots = np.linspace(0.0, 1.0, n_knots + 1)
    classes = list(estimator.classes_)
    batches = []
    for start in range(0, len(x), batch_size):
        x_batch = x[start : start + batch_size]
        features, _ = coverage.expand_coverage(x_batch, np.zeros(len(x_batch)), knots)
        if True in classes:
            probability = estimator.predict_proba(features)[:, classes.index(True)]
        else:  # Every training row had PIT > alpha.
            probability = np.zeros(len(features))
        batches.append(probability.reshape(len(x_batch), -1))
    at_knots = np.concatenate(batches) if batches else np.empty((0, len(knots)))
    alpha = np.broadcast_to(alpha, (len(at_knots), alpha.shape[-1]))
    upper = np.clip(np.searchsorted(knots, alpha), 1, n_knots)
    weight = np.clip((alpha - knots[upper - 1]) / (knots[upper] - knots[upper - 1]), 0.0, 1.0)
    lower_values = np.take_along_axis(at_knots, upper - 1, axis=1)
    upper_values = np.take_along_axis(at_knots, upper, axis=1)
    return lower_values + weight * (upper_values - lower_values)
