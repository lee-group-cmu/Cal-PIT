"""Building blocks of Cal-PIT that do not depend on a machine-learning library.

Cal-PIT learns the conditional CDF of the probability integral transform (PIT),
r(alpha; x) = P(PIT <= alpha | x), by regressing the indicator PIT <= alpha on
the coverage level alpha and the features x. The functions here split the
calibration data, build the (alpha, x) training rows for estimators that are
fitted once on a fixed data set, and make predictions of r non-decreasing in
alpha by monotone rearrangement.
"""

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]
SeedLike = int | np.random.Generator | None


def train_val_split(
    x: npt.ArrayLike,
    pit: npt.ArrayLike,
    val_fraction: float,
    random_state: SeedLike = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits calibration objects at random into a training and a validation set.

    The split is by object, so the many (alpha, x) rows that one object yields
    all fall on the same side of it.

    Args:
        x: The features, shape (n_objects, n_features).
        pit: The PIT values, shape (n_objects,).
        val_fraction: The fraction of objects in the validation set, in
            [0, 1).
        random_state: The seed or generator of the permutation.

    Returns:
        The training features, training PIT values, validation features and
        validation PIT values, as a tuple (x_train, pit_train, x_val, pit_val).
    """
    x = np.asarray(x)
    pit = np.asarray(pit)
    if not 0 <= val_fraction < 1:
        raise ValueError(f"val_fraction must be in [0, 1): {val_fraction=}")
    train_size = int((1 - val_fraction) * len(x))
    order = np.random.default_rng(random_state).permutation(len(x))
    train, val = order[:train_size], order[train_size:]
    return x[train], pit[train], x[val], pit[val]


def stratified_alpha(n_objects: int, n_alpha: int, random_state: SeedLike = None) -> FloatArray:
    """Draws coverage levels that cover [0, 1] evenly for every object.

    [0, 1] is cut into n_alpha equal strata and each object gets one uniform
    draw in each stratum. This covers [0, 1] as evenly as a fixed grid without
    every object sharing the same alpha values.

    Args:
        n_objects: The number of objects.
        n_alpha: The number of coverage levels per object.
        random_state: The seed or generator of the draws.

    Returns:
        The coverage levels, shape (n_objects, n_alpha), increasing along each
        row.
    """
    rng = np.random.default_rng(random_state)
    return (np.arange(n_alpha) + rng.uniform(size=(n_objects, n_alpha))) / n_alpha


def expand_coverage(
    x: npt.ArrayLike, pit: npt.ArrayLike, alpha: npt.ArrayLike
) -> tuple[FloatArray, np.ndarray]:
    """Builds the rows that a classifier of PIT <= alpha is trained on.

    Each object yields one row per coverage level: the features are alpha
    followed by x, and the target is whether the object's PIT is at most alpha.
    The rows of one object are consecutive.

    Args:
        x: The features, shape (n_objects, n_features).
        pit: The PIT values, shape (n_objects,).
        alpha: The coverage levels, shape (n_alpha,) to use the same levels for
            every object, or (n_objects, n_alpha).

    Returns:
        The features [alpha, x], shape (n_objects * n_alpha, n_features + 1),
        and the boolean targets, shape (n_objects * n_alpha,), as a tuple
        (features, targets).
    """
    x = np.asarray(x, dtype=float)
    pit = np.asarray(pit, dtype=float)
    alpha = np.broadcast_to(np.asarray(alpha, dtype=float), (len(x), np.shape(alpha)[-1]))
    n_alpha = alpha.shape[1]
    features = np.hstack([alpha.reshape(-1, 1), np.repeat(x, n_alpha, axis=0)])
    targets = np.repeat(pit, n_alpha) <= alpha.ravel()
    return features, targets


def rearrange(pit_cdf: npt.ArrayLike, alpha: npt.ArrayLike) -> FloatArray:
    """Makes predictions of the PIT CDF non-decreasing in alpha by sorting them.

    This is the monotone rearrangement of Chernozhukov, Fernandez-Val and
    Galichon (2010), https://arxiv.org/abs/0704.3649: the values predicted at a
    set of alpha are sorted and handed out in increasing order of alpha, which
    makes any estimate of r non-decreasing in alpha. If the true r is
    non-decreasing, the rearranged estimate is never further from it in Lp
    distance (here over the points evaluated, with equal weight) than the
    original estimate was (their Proposition 4). Predictions that are already
    non-decreasing come back unchanged.

    Args:
        pit_cdf: The predicted PIT CDF, shape (n_objects, n_alpha).
        alpha: The coverage levels it was predicted at, shape (n_alpha,) or
            (n_objects, n_alpha).

    Returns:
        The rearranged PIT CDF, shape (n_objects, n_alpha).
    """
    pit_cdf = np.asarray(pit_cdf)
    alpha = np.broadcast_to(np.asarray(alpha), pit_cdf.shape)
    order = np.argsort(alpha, axis=-1, kind="stable")
    rearranged = np.empty_like(pit_cdf)
    np.put_along_axis(rearranged, order, np.sort(pit_cdf, axis=-1), axis=-1)
    return rearranged
