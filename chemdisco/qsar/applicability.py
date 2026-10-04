"""Applicability domain: deciding when a model should decline to answer.

A fitted model returns a number for any input. Hand it a molecule unlike
anything in its training set and it will still produce a confident-looking
pIC50, interpolating in a region where it has no data. This matters more here
than in most machine-learning settings, because the whole point of generative
candidate design is to produce molecules that are *new* -- which is precisely
the regime where the scoring model is least reliable.

So the generative loop has a built-in conflict: the more novel a candidate, the
less its predicted activity means. Ignoring that conflict produces a ranked list
of exotic structures whose high scores are extrapolation artefacts. The fix is
not a better model; it is making the model report when it is out of its depth,
and refusing to rank on predictions that are.

Three domain definitions are implemented, because they disagree and the
disagreement is informative:

**k-nearest-neighbour distance.** Compare a candidate's mean distance to its *k*
nearest training compounds against the distribution of that same statistic
within the training set. Intuitive, works with any metric, and the standard
choice for fingerprint spaces.

**Similarity threshold.** Out of domain when the single most similar training
compound falls below a fixed similarity. Crude, but it is the criterion a
medicinal chemist actually applies, and a candidate with no training neighbour
above 0.3 Tanimoto is a genuinely different chemotype.

**Leverage.** The Williams-plot ``h = x'(X'X)^-1 x``, thresholded at ``3p/n``.
Classical in the QSAR literature and required by some regulatory guidance. It
measures distance in descriptor space under the model's own geometry, which
makes it informative for linear models and much less so for tree ensembles.

All three are reported. A candidate inside all three is safe to rank; one that
disagrees across methods is flagged rather than silently resolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

DomainMethod = Literal["knn_distance", "similarity", "leverage"]


@dataclass(frozen=True, slots=True)
class DomainVerdict:
    """Whether one candidate falls inside the model's applicability domain.

    Attributes:
        in_domain: The combined verdict. ``True`` only when every assessed
            method agrees, because a disagreement means at least one method sees
            extrapolation and the conservative reading is the correct one for a
            number that will be used to rank drug candidates.
        knn_distance: Mean distance to the k nearest training compounds.
        knn_threshold: The training-set percentile this was compared against.
        max_similarity: Similarity to the single nearest training compound.
        similarity_threshold: Minimum similarity required.
        leverage: Williams-plot leverage, when descriptors were supplied.
        leverage_threshold: ``3p/n``.
        methods_disagree: ``True`` when the methods reach different conclusions.
        reasons: Why the candidate was placed outside, if it was.
    """

    in_domain: bool
    knn_distance: float | None = None
    knn_threshold: float | None = None
    max_similarity: float | None = None
    similarity_threshold: float | None = None
    leverage: float | None = None
    leverage_threshold: float | None = None
    methods_disagree: bool = False
    reasons: tuple[str, ...] = ()

    def describe(self) -> str:
        if self.in_domain and not self.methods_disagree:
            return "inside the applicability domain"
        text = "OUTSIDE the applicability domain: " + "; ".join(self.reasons)
        if self.methods_disagree:
            text += (
                " (the domain definitions disagree, so this verdict is the "
                "conservative reading)"
            )
        return text


def knn_distances(
    query: np.ndarray, reference: np.ndarray, *, k: int = 5
) -> np.ndarray:
    """Mean Euclidean distance from each query row to its ``k`` nearest references.

    Args:
        query: Shape ``(n_query, n_features)``.
        reference: Shape ``(n_reference, n_features)``, normally the training set.
        k: Neighbour count. Clamped to the reference size, because asking for
            more neighbours than exist should not fail silently with padding.

    Returns:
        One mean distance per query row.
    """
    Q = np.asarray(query, dtype=float)
    R = np.asarray(reference, dtype=float)
    if Q.ndim == 1:
        Q = Q.reshape(1, -1)
    if R.ndim == 1:
        R = R.reshape(1, -1)
    if Q.shape[1] != R.shape[1]:
        raise ValueError(
            f"query has {Q.shape[1]} features, reference has {R.shape[1]}"
        )
    if R.shape[0] == 0:
        raise ValueError("reference set is empty; no domain can be defined")

    effective_k = max(1, min(k, R.shape[0]))
    # Pairwise distances via the expansion ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b,
    # clipped at zero because floating-point cancellation can make the expansion
    # slightly negative for identical points.
    squared = (
        np.sum(Q**2, axis=1)[:, None]
        + np.sum(R**2, axis=1)[None, :]
        - 2.0 * Q @ R.T
    )
    distances = np.sqrt(np.maximum(squared, 0.0))
    partitioned = np.partition(distances, effective_k - 1, axis=1)[:, :effective_k]
    return partitioned.mean(axis=1)


def knn_threshold_from_training(
    train_features: np.ndarray, *, k: int = 5, percentile: float = 95.0
) -> float:
    """The k-NN distance above which a candidate is unlike the training set.

    Computed as a percentile of the training set's own leave-one-out k-NN
    distances, so the threshold adapts to how tightly the training data is
    clustered instead of being an arbitrary constant. A sparse, chemically
    diverse training set earns a wide domain; a single congeneric series earns a
    narrow one, which is the correct behaviour.
    """
    X = np.asarray(train_features, dtype=float)
    if X.shape[0] < 2:
        raise ValueError("need at least two training compounds to define a domain")

    effective_k = max(1, min(k, X.shape[0] - 1))
    squared = (
        np.sum(X**2, axis=1)[:, None]
        + np.sum(X**2, axis=1)[None, :]
        - 2.0 * X @ X.T
    )
    distances = np.sqrt(np.maximum(squared, 0.0))
    # Exclude each point's zero distance to itself, which would otherwise pull
    # every training distance down and inflate the domain.
    np.fill_diagonal(distances, np.inf)
    partitioned = np.partition(distances, effective_k - 1, axis=1)[:, :effective_k]
    own_distances = partitioned.mean(axis=1)
    return float(np.percentile(own_distances, percentile))


def leverages(query: np.ndarray, train_features: np.ndarray) -> np.ndarray:
    """Williams-plot leverage for each query row.

    Uses a pseudo-inverse of ``X'X`` so correlated or collinear descriptor sets
    -- which is most of them, since molecular descriptors are heavily
    intercorrelated -- do not raise. With rank deficiency the leverage is
    computed in the subspace the training data actually spans, which is the
    meaningful quantity anyway.
    """
    Q = np.asarray(query, dtype=float)
    X = np.asarray(train_features, dtype=float)
    if Q.ndim == 1:
        Q = Q.reshape(1, -1)
    if Q.shape[1] != X.shape[1]:
        raise ValueError(f"query has {Q.shape[1]} features, training has {X.shape[1]}")

    gram_inverse = np.linalg.pinv(X.T @ X)
    return np.einsum("ij,jk,ik->i", Q, gram_inverse, Q)


def leverage_threshold(train_features: np.ndarray) -> float:
    """The conventional ``3p/n`` warning leverage."""
    X = np.asarray(train_features, dtype=float)
    n_samples, n_features = X.shape
    if n_samples == 0:
        raise ValueError("empty training set")
    # p is the number of model parameters: descriptors plus the intercept.
    return 3.0 * (n_features + 1) / n_samples


@dataclass(frozen=True, slots=True)
class ApplicabilityDomain:
    """A fitted domain, ready to judge candidates.

    Built from the training set once, then applied to any number of candidates.
    Keeping it a fitted object rather than a free function means the thresholds
    cannot drift between the model's training data and what it is judged against.
    """

    train_features: np.ndarray
    k: int = 5
    knn_percentile: float = 95.0
    similarity_floor: float = 0.3
    use_leverage: bool = True
    _knn_threshold: float = 0.0
    _leverage_threshold: float = 0.0

    @classmethod
    def fit(
        cls,
        train_features: np.ndarray,
        *,
        k: int = 5,
        knn_percentile: float = 95.0,
        similarity_floor: float = 0.3,
        use_leverage: bool = True,
    ) -> ApplicabilityDomain:
        X = np.asarray(train_features, dtype=float)
        return cls(
            train_features=X,
            k=k,
            knn_percentile=knn_percentile,
            similarity_floor=similarity_floor,
            use_leverage=use_leverage,
            _knn_threshold=knn_threshold_from_training(
                X, k=k, percentile=knn_percentile
            ),
            _leverage_threshold=leverage_threshold(X) if use_leverage else 0.0,
        )

    def assess(
        self,
        features: np.ndarray,
        *,
        max_similarity: np.ndarray | None = None,
    ) -> list[DomainVerdict]:
        """Judge each row of ``features``.

        Args:
            features: Candidate descriptors, shape ``(n, n_features)``.
            max_similarity: Optional per-candidate similarity to the nearest
                training compound, shape ``(n,)``. Supplied separately because
                fingerprint similarity is computed at the RDKit edge and this
                module stays toolkit-free.

        Returns:
            One verdict per candidate.
        """
        X = np.asarray(features, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        distances = knn_distances(X, self.train_features, k=self.k)
        lev = leverages(X, self.train_features) if self.use_leverage else None
        similarity = (
            np.asarray(max_similarity, dtype=float)
            if max_similarity is not None
            else None
        )
        if similarity is not None and similarity.shape[0] != X.shape[0]:
            raise ValueError(
                f"{similarity.shape[0]} similarities for {X.shape[0]} candidates"
            )

        verdicts: list[DomainVerdict] = []
        for index in range(X.shape[0]):
            reasons: list[str] = []
            calls: list[bool] = []

            distance_ok = bool(distances[index] <= self._knn_threshold)
            calls.append(distance_ok)
            if not distance_ok:
                reasons.append(
                    f"mean distance to the {self.k} nearest training compounds is "
                    f"{distances[index]:.3f}, above the "
                    f"{self.knn_percentile:g}th-percentile training distance of "
                    f"{self._knn_threshold:.3f}"
                )

            similarity_value: float | None = None
            if similarity is not None:
                similarity_value = float(similarity[index])
                similarity_ok = similarity_value >= self.similarity_floor
                calls.append(similarity_ok)
                if not similarity_ok:
                    reasons.append(
                        f"nearest training compound is only {similarity_value:.3f} "
                        f"similar, below the {self.similarity_floor:g} floor; this "
                        "is a chemotype the model has not seen"
                    )

            leverage_value: float | None = None
            if lev is not None:
                leverage_value = float(lev[index])
                leverage_ok = leverage_value <= self._leverage_threshold
                calls.append(leverage_ok)
                if not leverage_ok:
                    reasons.append(
                        f"leverage {leverage_value:.3f} exceeds the 3p/n warning "
                        f"threshold of {self._leverage_threshold:.3f}"
                    )

            verdicts.append(
                DomainVerdict(
                    in_domain=all(calls),
                    knn_distance=float(distances[index]),
                    knn_threshold=self._knn_threshold,
                    max_similarity=similarity_value,
                    similarity_threshold=(
                        self.similarity_floor if similarity is not None else None
                    ),
                    leverage=leverage_value,
                    leverage_threshold=(
                        self._leverage_threshold if lev is not None else None
                    ),
                    methods_disagree=len(set(calls)) > 1,
                    reasons=tuple(reasons),
                )
            )
        return verdicts

    def coverage(
        self, features: np.ndarray, *, max_similarity: np.ndarray | None = None
    ) -> float:
        """Fraction of candidates inside the domain.

        Worth reporting prominently for a generated library: if 90% of generated
        candidates are out of domain, the ranking is mostly extrapolation and the
        generator needs to stay closer to known chemistry -- or the model needs
        more diverse training data before it can referee this library at all.
        """
        verdicts = self.assess(features, max_similarity=max_similarity)
        if not verdicts:
            raise ValueError("no candidates to assess")
        return sum(1 for v in verdicts if v.in_domain) / len(verdicts)
