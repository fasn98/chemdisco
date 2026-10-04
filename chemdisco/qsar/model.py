"""QSAR regressors that return quantities, not bare floats.

Every prediction leaves this module as a :class:`~chemdisco.provenance.Quantity`
with ``Origin.PREDICTED``, the model's identity as its source, an uncertainty
estimate, and an applicability-domain verdict. That makes it impossible for a
prediction to be rendered or exported as though it were a measurement, and it
makes an out-of-domain prediction visibly unfit for ranking.

On uncertainty from a random forest: the spread of the individual trees'
predictions is a usable, cheap proxy for how much the training data constrains a
given input, and it is what the forest already computed. It is *not* a calibrated
confidence interval -- the trees are correlated, so the spread systematically
understates true predictive uncertainty. The number is reported because a bad
uncertainty estimate is still far more informative than none, and the limitation
is recorded in the quantity's notes rather than left for the reader to assume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from ..provenance import Quantity
from .applicability import ApplicabilityDomain

ModelKind = Literal["random_forest", "gradient_boosting", "ridge"]

#: Fixed seed so a reported result can be reproduced exactly. Varying it across
#: runs and reporting the spread is better practice for a final figure; a fixed
#: default makes day-to-day work deterministic.
DEFAULT_RANDOM_STATE = 20260104


def _build_estimator(kind: ModelKind, random_state: int, **overrides: Any):
    """Construct an estimator with defaults suited to small, noisy QSAR sets.

    The hyperparameters here are chosen for datasets of a few hundred to a few
    thousand compounds with experimental noise near 0.5 log units, where the
    dominant risk is overfitting a congeneric series rather than underfitting.
    Hence shallow-ish trees and a minimum leaf size above one.
    """
    if kind == "random_forest":
        params: dict[str, Any] = {
            "n_estimators": 500,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "n_jobs": -1,
            "random_state": random_state,
        }
        params.update(overrides)
        return RandomForestRegressor(**params)
    if kind == "gradient_boosting":
        params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.8,
            "random_state": random_state,
        }
        params.update(overrides)
        return GradientBoostingRegressor(**params)
    if kind == "ridge":
        # A linear model is worth fitting as a transparency baseline: when it
        # matches the ensemble, the relationship is essentially linear in the
        # descriptors and the ensemble's opacity buys nothing.
        params = {"alpha": 1.0, "random_state": random_state}
        params.update(overrides)
        return Ridge(**params)
    raise ValueError(f"unknown model kind {kind!r}")


@dataclass(slots=True)
class QSARModel:
    """A fitted QSAR regressor with a domain of applicability attached.

    Attributes:
        kind: Which estimator was fitted.
        feature_names: Descriptor names, in column order. Stored so a prediction
            cannot be made with features in a different order -- a silent and
            catastrophic error otherwise.
        label_name: What the target represents, e.g. ``"pIC50 (BACE1)"``.
        curation_summary: The curation and aggregation policy that produced the
            training data, carried so a reported metric always travels with the
            decisions behind it.
    """

    kind: ModelKind = "random_forest"
    feature_names: tuple[str, ...] = ()
    label_name: str = "pActivity"
    curation_summary: str = ""
    random_state: int = DEFAULT_RANDOM_STATE
    _estimator: Any = None
    _domain: ApplicabilityDomain | None = None
    _n_train: int = 0
    _train_label_mean: float | None = None
    _train_label_std: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_fitted(self) -> bool:
        return self._estimator is not None

    @property
    def identity(self) -> str:
        """The string recorded as the provenance source of every prediction."""
        return (
            f"{self.kind} on {len(self.feature_names)} descriptors, "
            f"n_train={self._n_train}, target={self.label_name}"
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: Sequence[str] | None = None,
        fit_domain: bool = True,
        **estimator_overrides: Any,
    ) -> "QSARModel":
        """Fit the estimator and the applicability domain together.

        The domain is fitted from the same training matrix, in the same call, so
        the two cannot fall out of step. A model whose domain was fitted on
        different data would report confident verdicts about the wrong region of
        chemical space.
        """
        features = np.asarray(X, dtype=float)
        labels = np.asarray(y, dtype=float)

        if features.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {features.shape}")
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"{features.shape[0]} feature rows against {labels.shape[0]} labels"
            )
        if features.shape[0] < 10:
            raise ValueError(
                f"refusing to fit on {features.shape[0]} compounds. A model from a "
                "handful of points will produce confident predictions with no "
                "support, which is worse than having no model. Gather more data "
                "or state that the target has too little."
            )
        if float(np.std(labels)) == 0.0:
            raise ValueError(
                "training labels have zero variance; there is nothing to learn"
            )
        if not np.all(np.isfinite(features)):
            raise ValueError(
                "feature matrix contains non-finite values. Impute explicitly and "
                "record that you did, or drop the affected compounds -- do not let "
                "a NaN reach the estimator."
            )

        if feature_names is not None:
            if len(feature_names) != features.shape[1]:
                raise ValueError(
                    f"{len(feature_names)} names for {features.shape[1]} columns"
                )
            self.feature_names = tuple(feature_names)
        elif not self.feature_names:
            self.feature_names = tuple(
                f"feature_{i}" for i in range(features.shape[1])
            )

        self._estimator = _build_estimator(
            self.kind, self.random_state, **estimator_overrides
        )
        self._estimator.fit(features, labels)
        self._n_train = int(features.shape[0])
        self._train_label_mean = float(np.mean(labels))
        self._train_label_std = float(np.std(labels))
        self._domain = ApplicabilityDomain.fit(features) if fit_domain else None

        collected = list(self.notes)
        if features.shape[0] < 50:
            collected.append(
                f"fitted on only {features.shape[0]} compounds; treat every "
                "prediction as provisional"
            )
        if features.shape[1] > features.shape[0]:
            collected.append(
                f"{features.shape[1]} descriptors for {features.shape[0]} "
                "compounds: more features than samples invites overfitting"
            )
        self.notes = tuple(collected)
        return self

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("model is not fitted")

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Point predictions as a plain array, for metric computation.

        Used by :mod:`chemdisco.qsar.evaluate`, which needs arrays. Anything
        user-facing should go through :meth:`predict`, which carries provenance.
        """
        self._require_fitted()
        features = np.asarray(X, dtype=float)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"model expects {len(self.feature_names)} descriptors, got "
                f"{features.shape[1]}. Feature order must match training exactly."
            )
        return np.asarray(self._estimator.predict(features), dtype=float)

    def _tree_spread(self, X: np.ndarray) -> np.ndarray | None:
        """Standard deviation across ensemble members, when available."""
        estimators = getattr(self._estimator, "estimators_", None)
        if estimators is None or self.kind != "random_forest":
            return None
        per_tree = np.stack([tree.predict(X) for tree in estimators])
        return np.std(per_tree, axis=0)

    def predict(
        self,
        X: np.ndarray,
        *,
        max_similarity: np.ndarray | None = None,
    ) -> list[Quantity]:
        """Predict, returning one provenance-carrying quantity per compound.

        Args:
            X: Descriptor matrix in the training column order.
            max_similarity: Optional similarity to the nearest training compound
                per candidate, which sharpens the domain verdict considerably for
                fingerprint-based work.

        Returns:
            One :class:`~chemdisco.provenance.Quantity` per row. Predictions
            outside the applicability domain are returned -- not suppressed --
            with ``in_domain=False``, so they are visible but excluded from
            ranking by :attr:`Quantity.is_trustworthy_for_ranking`.
        """
        self._require_fitted()
        features = np.asarray(X, dtype=float)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        values = self.predict_raw(features)
        spread = self._tree_spread(features)

        verdicts = (
            self._domain.assess(features, max_similarity=max_similarity)
            if self._domain is not None
            else None
        )

        quantities: list[Quantity] = []
        for index, value in enumerate(values):
            notes = list(self.notes)
            uncertainty: float | None = None
            if spread is not None:
                uncertainty = float(spread[index])
                notes.append(
                    "uncertainty is the spread across forest trees; trees are "
                    "correlated, so this understates true predictive uncertainty"
                )

            in_domain: bool | None = None
            if verdicts is not None:
                verdict = verdicts[index]
                in_domain = verdict.in_domain
                if not verdict.in_domain:
                    notes.append(verdict.describe())
                elif verdict.methods_disagree:
                    notes.append(
                        "applicability-domain methods disagreed; "
                        + verdict.describe()
                    )

            quantities.append(
                Quantity.predicted(
                    float(value),
                    None,
                    self.identity,
                    uncertainty=uncertainty,
                    in_domain=in_domain,
                    notes=notes,
                )
            )
        return quantities

    def feature_importances(self) -> dict[str, float] | None:
        """Importances by descriptor name, or ``None`` for models without them.

        Worth inspecting as a sanity check rather than an explanation: if a
        single descriptor dominates on a bioactivity task, that is more likely a
        leakage signature than a biological insight. See
        :func:`chemdisco.qsar.baselines.detect_target_leakage`.
        """
        self._require_fitted()
        importances = getattr(self._estimator, "feature_importances_", None)
        if importances is None:
            return None
        paired = dict(zip(self.feature_names, (float(v) for v in importances)))
        return dict(sorted(paired.items(), key=lambda kv: -kv[1]))

    def metadata(self) -> dict[str, Any]:
        """Everything needed to interpret this model's outputs.

        Serialised alongside any saved model and embedded in reports, so a
        metric can never be quoted without the curation and training context
        that produced it.
        """
        self._require_fitted()
        return {
            "kind": self.kind,
            "identity": self.identity,
            "label_name": self.label_name,
            "n_train": self._n_train,
            "n_features": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "train_label_mean": self._train_label_mean,
            "train_label_std": self._train_label_std,
            "random_state": self.random_state,
            "curation_summary": self.curation_summary,
            "has_applicability_domain": self._domain is not None,
            "notes": list(self.notes),
        }


def make_fit_predict(
    kind: ModelKind,
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    **overrides: Any,
):
    """Build the callable :func:`~chemdisco.qsar.baselines.permutation_scores` needs.

    Closes over the feature matrices so the permutation test varies only the
    labels -- which is the whole point of the test, and easy to get wrong by
    accidentally re-deriving features from the shuffled labels.
    """
    features_train = np.asarray(X_train, dtype=float)
    features_test = np.asarray(X_test, dtype=float)

    def fit_predict(y_shuffled: np.ndarray) -> np.ndarray:
        estimator = _build_estimator(kind, random_state, **overrides)
        estimator.fit(features_train, np.asarray(y_shuffled, dtype=float))
        return np.asarray(estimator.predict(features_test), dtype=float)

    return fit_predict
