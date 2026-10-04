"""Building a training matrix from curated points, with provenance enforced.

This module holds the structural guarantee against the predecessor project's
central defect, and it exists because a statistical guard cannot provide one.

That project synthesised activity labels from molecular weight and LogP when no
measurement was available, then trained on molecular weight and LogP as features.
Attempting to catch this after the fact fails:

* **Linear correlation misses it.** The synthetic label depended on
  ``-|mw - 400|``, which rises and falls, so the Pearson correlation between
  molecular weight and the label is near zero.
* **Per-feature checks miss it.** The label was a sum over two features. Neither
  alone predicts it well; together they determine it.
* **"Too good to be true" misses it.** With the noise term the construction
  actually scored R-squared near 0.69 on a held-out set -- an entirely
  believable QSAR result, which is far more dangerous than an implausible one
  because nothing about it invites scrutiny.

The only reliable defence is to make a fabricated label unrepresentable. A
training target here is built exclusively from
:class:`~chemdisco.provenance.Quantity` values whose origin is ``MEASURED`` or
``DERIVED``, and :func:`labels_from_points` raises on anything else. A heuristic
cannot become a regression target through carelessness; it would take someone
deliberately mislabelling a quantity's origin, which is a visible act in review
rather than an accident.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..curate.records import CuratedPoint
from ..provenance import Origin, ProvenanceError, Quantity

#: Origins acceptable for a supervised regression target. A measurement, or a
#: deterministic transformation of one. Nothing else.
VALID_LABEL_ORIGINS: frozenset[Origin] = frozenset({Origin.MEASURED, Origin.DERIVED})


class LabelProvenanceError(ProvenanceError):
    """Raised when a proposed training label is not traceable to a measurement."""


def verify_label_provenance(labels: Sequence[Quantity]) -> None:
    """Raise unless every label descends from a real measurement.

    This is the check that makes synthetic labels impossible rather than merely
    discouraged. It is called by :func:`labels_from_points` and should be called
    by any other path that constructs a training target.

    Raises:
        LabelProvenanceError: naming the offending origins and how many labels
            carried each, so the message points at the source of the problem
            rather than at one unlucky row.
    """
    if not labels:
        raise LabelProvenanceError("no labels supplied")

    offenders: dict[Origin, int] = {}
    unknown = 0
    for label in labels:
        if not label.is_known:
            unknown += 1
            continue
        if label.origin not in VALID_LABEL_ORIGINS:
            offenders[label.origin] = offenders.get(label.origin, 0) + 1

    problems: list[str] = []
    if unknown:
        problems.append(
            f"{unknown} label(s) have no value. A compound with no measured "
            "activity has no place in a training set; it must be dropped, not "
            "imputed."
        )
    for origin, count in sorted(offenders.items(), key=lambda kv: -kv[1]):
        problems.append(
            f"{count} label(s) have origin '{origin.value}'. Training on a "
            f"{origin.value} target teaches the model to reproduce the rule that "
            "generated it, and the resulting metric measures nothing about "
            "biology."
        )

    if problems:
        raise LabelProvenanceError(
            "training labels must descend from measurements:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )


@dataclass(frozen=True, slots=True)
class TrainingSet:
    """A matrix, a target, and the record of where both came from.

    Attributes:
        X: Descriptor matrix, shape ``(n, n_features)``.
        y: Target values, shape ``(n,)``.
        feature_names: Column names in order.
        compound_ids: Row identifiers, for tracing a prediction back.
        smiles: Row structures, used for scaffold perception at the toolkit edge.
        label_name: What the target is, e.g. ``"pIC50 (BACE1)"``.
        label_uncertainties: Per-row uncertainty where the curation could state
            one, ``None`` where it could not. Carried so a weighted fit or an
            uncertainty-aware evaluation remains possible later.
        provenance_summary: How the labels were produced.
    """

    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    compound_ids: tuple[str, ...]
    smiles: tuple[str, ...]
    label_name: str
    label_uncertainties: tuple[float | None, ...]
    provenance_summary: str

    def __post_init__(self) -> None:
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {self.X.shape}")
        n = self.X.shape[0]
        for name, sequence in (
            ("y", self.y),
            ("compound_ids", self.compound_ids),
            ("smiles", self.smiles),
            ("label_uncertainties", self.label_uncertainties),
        ):
            if len(sequence) != n:
                raise ValueError(
                    f"{name} has {len(sequence)} entries for {n} rows in X"
                )
        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError(
                f"{len(self.feature_names)} feature names for "
                f"{self.X.shape[1]} columns"
            )

    @property
    def n_compounds(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def subset(self, indices: Sequence[int]) -> TrainingSet:
        """Take a partition, keeping every parallel array aligned.

        Splitting by hand is where index-alignment bugs appear, and a misaligned
        label array produces a quietly meaningless model rather than an error.
        """
        index_list = list(indices)
        return TrainingSet(
            X=self.X[index_list],
            y=self.y[index_list],
            feature_names=self.feature_names,
            compound_ids=tuple(self.compound_ids[i] for i in index_list),
            smiles=tuple(self.smiles[i] for i in index_list),
            label_name=self.label_name,
            label_uncertainties=tuple(self.label_uncertainties[i] for i in index_list),
            provenance_summary=self.provenance_summary,
        )

    def describe(self) -> str:
        known_uncertainty = sum(1 for u in self.label_uncertainties if u is not None)
        return (
            f"{self.n_compounds} compounds x {self.n_features} descriptors; "
            f"target={self.label_name}; "
            f"label range {float(self.y.min()):.2f}-{float(self.y.max()):.2f}, "
            f"sd {float(self.y.std()):.2f}; "
            f"{known_uncertainty} labels carry a replicate-based uncertainty.\n"
            f"Provenance: {self.provenance_summary}"
        )


def labels_from_points(points: Sequence[CuratedPoint]) -> list[Quantity]:
    """Extract label quantities from curated points, verifying provenance."""
    labels = [point.pactivity for point in points]
    verify_label_provenance(labels)
    return labels


def build_training_set(
    points: Sequence[CuratedPoint],
    descriptor_fn: Callable[[Sequence[str]], tuple[np.ndarray, Sequence[str]]],
    *,
    label_name: str = "pActivity",
    provenance_summary: str = "",
) -> TrainingSet:
    """Assemble a training set from curated points.

    Args:
        points: Output of :func:`chemdisco.curate.aggregate.curate`.
        descriptor_fn: Takes SMILES and returns ``(matrix, feature_names)``.
            Injected rather than imported so this module needs no chemistry
            toolkit and the descriptor implementation can be swapped or stubbed.
        label_name: Human-readable description of the target.
        provenance_summary: Curation policy description, carried into the model's
            metadata so a reported metric always travels with its curation.

    Returns:
        A :class:`TrainingSet` whose labels are verified to descend from
        measurements.

    Raises:
        LabelProvenanceError: if any label is heuristic, predicted or unknown.
        ValueError: if the descriptor function returns a misshapen matrix, or
            produces non-finite values -- which must be handled explicitly at the
            descriptor layer rather than silently imputed here.
    """
    if not points:
        raise ValueError("no curated points supplied")

    labels = labels_from_points(points)
    smiles = [point.smiles for point in points]

    matrix, feature_names = descriptor_fn(smiles)
    matrix = np.asarray(matrix, dtype=float)

    if matrix.shape[0] != len(points):
        raise ValueError(
            f"descriptor function returned {matrix.shape[0]} rows for "
            f"{len(points)} compounds"
        )
    if not np.all(np.isfinite(matrix)):
        bad_rows = int(np.sum(~np.all(np.isfinite(matrix), axis=1)))
        raise ValueError(
            f"{bad_rows} compound(s) produced non-finite descriptors. Drop them "
            "and record the loss, or compute the missing descriptors properly -- "
            "substituting a typical value here would put a fabricated molecule "
            "into the training set."
        )

    return TrainingSet(
        X=matrix,
        y=np.array([label.require() for label in labels], dtype=float),
        feature_names=tuple(feature_names),
        compound_ids=tuple(point.compound_id for point in points),
        smiles=tuple(smiles),
        label_name=label_name,
        label_uncertainties=tuple(label.uncertainty for label in labels),
        provenance_summary=provenance_summary
        or "; ".join(sorted({label.origin.value for label in labels})),
    )
