"""Baselines a QSAR model must beat before its score means anything.

An R-squared of 0.45 on a held-out scaffold split sounds like a working model.
Whether it is depends entirely on what the trivial alternatives achieve on the
same split, and the trivial alternatives are often startlingly good:

* **Mean predictor.** Predicts the training mean for everything. Its test
  R-squared is approximately zero *by construction* when train and test have the
  same mean -- but on a scaffold split they frequently do not, and the mean
  predictor's score drifts negative. That makes it the honest zero point, and a
  model scoring 0.1 above it is barely doing anything.

* **Nearest-neighbour by similarity.** Predicts the activity of the most similar
  training compound. On congeneric series this is extremely strong, and a
  descriptor-based model that cannot beat it has learned nothing a similarity
  lookup does not already provide. This is the baseline most QSAR papers omit
  and the one that most often deflates a result.

* **Label permutation.** Trains the real model on shuffled labels. The score
  distribution over many shuffles is what the model achieves from dataset
  structure alone -- split composition, feature count, overfitting capacity. A
  real score inside that distribution is not evidence of a structure-activity
  relationship, regardless of how large it is.

The permutation baseline is the one that catches the specific failure in the
predecessor project: there, QSAR targets were partly computed from the same
molecular properties used as features, so the model was fitting an algebraic
identity. Permuted labels would have scored near zero while the real labels
scored high -- which looks like success. The tell is different: a model fitting
an identity achieves a near-perfect score that permutation *cannot* reach, with
no biological content at all. So permutation testing is necessary but not
sufficient, and :func:`detect_target_leakage` addresses the identity case
directly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class BaselineScores:
    """What the trivial alternatives achieved on this split.

    Attributes:
        mean_predictor_r2: Test R-squared of predicting the training mean.
        mean_predictor_rmse: Test RMSE of the same.
        nearest_neighbour_r2: Test R-squared of the 1-NN predictor, or ``None``
            if no similarity information was supplied.
        nearest_neighbour_rmse: Test RMSE of the 1-NN predictor.
        permutation_r2_mean: Mean test R-squared over permuted-label fits.
        permutation_r2_p95: 95th percentile of the permuted distribution. A real
            model must clear this to claim it found signal.
        n_permutations: How many shuffles were run.
        label_spread: Standard deviation of the test labels. A test set with
            almost no variance cannot produce a meaningful R-squared at all, and
            this is the number that reveals it.
    """

    mean_predictor_r2: float
    mean_predictor_rmse: float
    nearest_neighbour_r2: float | None
    nearest_neighbour_rmse: float | None
    permutation_r2_mean: float | None
    permutation_r2_p95: float | None
    n_permutations: int
    label_spread: float

    def verdict(self, model_r2: float) -> str:
        """Plain-language judgement on whether ``model_r2`` is meaningful."""
        lines: list[str] = []

        if self.label_spread < 0.3:
            lines.append(
                f"The test labels span only {self.label_spread:.2f} log units of "
                "standard deviation. R-squared is a ratio against label variance, "
                "so with this little spread it is numerically unstable and should "
                "not be the headline metric -- report RMSE instead."
            )

        margin = model_r2 - self.mean_predictor_r2
        if margin <= 0.05:
            lines.append(
                f"The model scores {model_r2:.3f} against {self.mean_predictor_r2:.3f} "
                "for predicting the training mean. It has learned essentially "
                "nothing."
            )
        else:
            lines.append(
                f"The model beats the mean predictor by {margin:.3f} R-squared."
            )

        if self.nearest_neighbour_r2 is not None:
            if model_r2 <= self.nearest_neighbour_r2:
                lines.append(
                    f"A 1-nearest-neighbour similarity lookup scores "
                    f"{self.nearest_neighbour_r2:.3f}, at or above the model. The "
                    "model adds nothing over looking up the most similar known "
                    "compound, which is far cheaper and more interpretable."
                )
            else:
                lines.append(
                    f"The model beats 1-NN similarity "
                    f"({self.nearest_neighbour_r2:.3f}) by "
                    f"{model_r2 - self.nearest_neighbour_r2:.3f}."
                )

        if self.permutation_r2_p95 is not None:
            if model_r2 <= self.permutation_r2_p95:
                lines.append(
                    f"The model's score falls inside the permuted-label "
                    f"distribution (95th percentile {self.permutation_r2_p95:.3f}). "
                    "This is not evidence of a structure-activity relationship."
                )
            else:
                lines.append(
                    f"The model clears the permuted-label 95th percentile "
                    f"({self.permutation_r2_p95:.3f})."
                )

        if model_r2 > 0.95:
            lines.append(
                f"R-squared of {model_r2:.3f} on held-out chemotypes is higher "
                "than real bioactivity data supports. Suspect target leakage: a "
                "label computed from the features, or the same compound in both "
                "partitions. Run detect_target_leakage before reporting this."
            )

        return "\n".join(lines)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination, computed against the *test* label variance.

    Note the denominator: the variance of ``y_true``, not of the training labels.
    This is the standard definition and it means R-squared can be arbitrarily
    negative when predictions are worse than the test set's own mean -- which is
    informative and should not be clipped to zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    if denominator == 0.0:
        raise ValueError(
            "test labels have zero variance; R-squared is undefined. Report RMSE."
        )
    residual = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - residual / denominator


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_predictor(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """Predict the training mean for every test compound."""
    return np.full(n_test, float(np.asarray(y_train, dtype=float).mean()))


def nearest_neighbour_predictor(
    similarity_to_train: np.ndarray, y_train: np.ndarray
) -> np.ndarray:
    """Predict each test compound's activity from its most similar training compound.

    Args:
        similarity_to_train: Matrix of shape ``(n_test, n_train)`` where higher
            means more similar. Tanimoto on ECFP fingerprints is the usual
            choice; any similarity works.
        y_train: Training labels.

    Returns:
        One prediction per test compound.
    """
    similarity = np.asarray(similarity_to_train, dtype=float)
    labels = np.asarray(y_train, dtype=float)
    if similarity.ndim != 2 or similarity.shape[1] != labels.shape[0]:
        raise ValueError(
            f"similarity matrix {similarity.shape} does not match "
            f"{labels.shape[0]} training labels"
        )
    nearest = np.argmax(similarity, axis=1)
    return labels[nearest]


def permutation_scores(
    fit_predict: Callable[[np.ndarray], np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    n_permutations: int = 20,
    seed: int = 0,
) -> list[float]:
    """Score the model repeatedly on shuffled training labels.

    Args:
        fit_predict: Takes shuffled training labels, fits the model, returns test
            predictions. The caller closes over the features so this module stays
            independent of any particular estimator.
        y_train: True training labels, shuffled in place of being used.
        y_test: True test labels, used only for scoring.
        n_permutations: Number of shuffles. Twenty gives a usable 95th
            percentile; more is better and linearly more expensive.
        seed: Reproducibility.

    Returns:
        One R-squared per permutation. The distribution, not its mean, is the
        useful output: the upper tail is the bar a real model must clear.
    """
    rng = random.Random(seed)
    labels = list(np.asarray(y_train, dtype=float))
    y_test_array = np.asarray(y_test, dtype=float)
    scores: list[float] = []

    for _ in range(n_permutations):
        shuffled = labels[:]
        rng.shuffle(shuffled)
        predictions = fit_predict(np.asarray(shuffled, dtype=float))
        try:
            scores.append(r_squared(y_test_array, predictions))
        except ValueError:
            continue
    return scores


def compute_baselines(
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    similarity_to_train: np.ndarray | None = None,
    fit_predict: Callable[[np.ndarray], np.ndarray] | None = None,
    n_permutations: int = 20,
    seed: int = 0,
) -> BaselineScores:
    """Compute every available baseline for one split.

    ``similarity_to_train`` and ``fit_predict`` are optional because not every
    caller can supply them, but a report that omits them is weaker and
    :meth:`BaselineScores.verdict` says so by staying silent on those baselines
    rather than implying they passed.
    """
    y_train_array = np.asarray(y_train, dtype=float)
    y_test_array = np.asarray(y_test, dtype=float)

    mean_predictions = mean_predictor(y_train_array, len(y_test_array))
    mean_r2 = r_squared(y_test_array, mean_predictions)
    mean_rmse = rmse(y_test_array, mean_predictions)

    nn_r2: float | None = None
    nn_rmse: float | None = None
    if similarity_to_train is not None:
        nn_predictions = nearest_neighbour_predictor(similarity_to_train, y_train_array)
        nn_r2 = r_squared(y_test_array, nn_predictions)
        nn_rmse = rmse(y_test_array, nn_predictions)

    permutation_mean: float | None = None
    permutation_p95: float | None = None
    n_done = 0
    if fit_predict is not None and n_permutations > 0:
        scores = permutation_scores(
            fit_predict,
            y_train_array,
            y_test_array,
            n_permutations=n_permutations,
            seed=seed,
        )
        n_done = len(scores)
        if scores:
            permutation_mean = float(np.mean(scores))
            permutation_p95 = float(np.percentile(scores, 95))

    return BaselineScores(
        mean_predictor_r2=mean_r2,
        mean_predictor_rmse=mean_rmse,
        nearest_neighbour_r2=nn_r2,
        nearest_neighbour_rmse=nn_rmse,
        permutation_r2_mean=permutation_mean,
        permutation_r2_p95=permutation_p95,
        n_permutations=n_done,
        label_spread=float(np.std(y_test_array)),
    )


def binned_feature_r2(
    feature: np.ndarray, labels: np.ndarray, *, n_bins: int = 10
) -> float:
    """How well one feature alone predicts the label, allowing any shape.

    Bins the feature into quantiles and predicts each bin's mean label. This
    captures non-monotonic dependence, which plain correlation cannot: a label
    built from ``-|mw - 400|`` rises then falls with molecular weight, giving a
    Pearson correlation near zero while being perfectly determined by it.

    Returns:
        R-squared of the binned-mean predictor, floored at 0.0. Below-zero values
        are not meaningful for a within-sample binned estimate.
    """
    x = np.asarray(feature, dtype=float)
    y = np.asarray(labels, dtype=float)
    if x.shape != y.shape:
        raise ValueError("feature and labels must have the same length")
    if np.std(y) == 0 or np.std(x) == 0:
        return 0.0

    effective_bins = max(2, min(n_bins, len(np.unique(x))))
    edges = np.quantile(x, np.linspace(0.0, 1.0, effective_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    assignments = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    predictions = np.empty_like(y)
    for bin_index in np.unique(assignments):
        mask = assignments == bin_index
        predictions[mask] = y[mask].mean()

    denominator = float(np.sum((y - y.mean()) ** 2))
    if denominator == 0.0:
        return 0.0
    residual = float(np.sum((y - predictions) ** 2))
    return max(0.0, 1.0 - residual / denominator)


def detect_target_leakage(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str] | None = None,
    *,
    correlation_threshold: float = 0.95,
    binned_r2_threshold: float = 0.9,
    n_bins: int = 10,
) -> list[str]:
    """Find individual features that largely determine the label.

    Two complementary screens, because one is not enough:

    * **Linear.** Absolute Pearson correlation above ``correlation_threshold``.
      No real molecular descriptor correlates with potency above about 0.5, so a
      value near 1.0 means the label was computed from the feature.
    * **Non-linear.** Binned-mean R-squared above ``binned_r2_threshold``, which
      catches dependence of any shape, including the non-monotonic kind that
      defeats correlation entirely.

    **What this cannot catch, and why it is not the primary defence.** A label
    synthesised from *several* features is invisible to both screens. The
    predecessor project built its synthetic activity as
    ``(f(molecular_weight) + g(logp)) / 2 + noise``; neither descriptor alone
    determines it, the dependence on each is non-monotonic, and the resulting
    held-out R-squared was about 0.69 -- an unremarkable, publishable number that
    invites no scrutiny whatsoever. No statistical screen reliably recovers that.

    The real guarantee is structural and lives in
    :mod:`chemdisco.qsar.dataset`: a training label must be a
    :class:`~chemdisco.provenance.Quantity` descending from a measurement, and
    :func:`chemdisco.qsar.dataset.verify_label_provenance` refuses anything else.
    Treat this function as a cheap extra screen for crude cases, never as
    evidence that a target is sound.

    Args:
        features: Array of shape ``(n_samples, n_features)``.
        labels: Array of shape ``(n_samples,)``.
        feature_names: Optional names for the report.
        correlation_threshold: Absolute Pearson correlation to flag.
        binned_r2_threshold: Binned-mean R-squared to flag.
        n_bins: Quantile bins for the non-linear screen.

    Returns:
        Human-readable warnings, empty when neither screen fires.
    """
    X = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"{X.shape[0]} feature rows against {y.shape[0]} labels"
        )
    if np.std(y) == 0:
        return ["labels have zero variance; leakage detection is not meaningful"]

    names = (
        list(feature_names)
        if feature_names is not None
        else [f"feature_{i}" for i in range(X.shape[1])]
    )
    if len(names) != X.shape[1]:
        raise ValueError(
            f"{len(names)} feature names for {X.shape[1]} feature columns"
        )

    warnings: list[str] = []
    for index in range(X.shape[1]):
        column = X[:, index]
        if np.std(column) == 0:
            continue

        correlation = float(np.corrcoef(column, y)[0, 1])
        if abs(correlation) >= correlation_threshold:
            warnings.append(
                f"feature '{names[index]}' correlates with the label at "
                f"r={correlation:.4f}. No real molecular descriptor predicts "
                "potency this well. The label was probably computed from this "
                "feature -- check how the target was constructed before trusting "
                "any model trained on it."
            )
            continue

        binned = binned_feature_r2(column, y, n_bins=n_bins)
        if binned >= binned_r2_threshold:
            warnings.append(
                f"feature '{names[index]}' alone explains R2={binned:.4f} of the "
                f"label through a non-linear relationship (its linear correlation "
                f"is only r={correlation:.3f}, which is why a correlation check "
                "would miss it). A single descriptor does not determine potency "
                "this precisely -- check how the target was constructed."
            )
    return warnings
