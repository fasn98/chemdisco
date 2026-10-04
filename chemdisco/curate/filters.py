"""Curation filters: deciding which reported measurements are defensible.

Every filter here removes data, and removing data is the most consequential
thing this package does to a model's reported performance. So each filter
states its scientific justification in its docstring, is individually
switchable, and reports what it dropped.

The defaults are deliberately strict. A QSAR model trained on ChEMBL without
these filters will mix nanomolar binding constants against a purified protein
with percent-inhibition readouts from a cell lysate and call the result one
structure-activity relationship. The resulting R-squared is not a measure of
anything.

What the defaults enforce, and why:

``confidence_score >= 8``
    ChEMBL's target-confidence scale runs 0-9. Only 8 (homologous single
    protein) and 9 (direct single protein) mean the measurement is about the
    protein you asked for. Scores of 4-7 may be against a protein complex or a
    family; 1-3 may be a whole cell or organism, where potency is confounded by
    permeability, efflux and metabolism.

``assay_type in {B, F}``
    Binding and functional assays measure target engagement. ADMET (A), toxicity
    (T) and physicochemical (P) assays measure something else entirely and must
    not share a regression target with them.

``data_validity_comment is null``
    ChEMBL flags values it distrusts: "Potential transcription error",
    "Outside typical range", "Non standard unit for type". These are ChEMBL
    telling you the number is probably wrong. Honouring that costs a few percent
    of the data and removes most of the catastrophic outliers.

``potential_duplicate == False``
    Duplicate measurements inflate dataset size, bias aggregation toward
    whichever value was deposited twice, and leak between train and test splits.

``relation == "="``
    See :mod:`chemdisco.units`: a censored value is a bound, not a measurement.

Activity-type mixing
    IC50 (functional inhibition, depends on substrate concentration) and Ki
    (thermodynamic binding constant) are related but not interchangeable; the
    Cheng-Prusoff relation between them needs assay parameters that ChEMBL does
    not record. The default therefore keeps one family per dataset and refuses
    to pool them, which is stricter than common practice in the literature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from ..units import pactivity
from .records import ActivityRecord, Rejection

#: ChEMBL assay types that measure engagement with the intended target.
TARGET_ENGAGEMENT_ASSAY_TYPES: frozenset[str] = frozenset({"B", "F"})

#: Activity-type families that may be pooled into a single regression target.
#: Binding constants and functional potencies are kept apart on purpose.
ACTIVITY_FAMILIES: dict[str, frozenset[str]] = {
    "binding": frozenset({"Ki", "Kd", "KD", "pKi", "pKd"}),
    "functional": frozenset({"IC50", "EC50", "XC50", "AC50", "pIC50", "pEC50"}),
    "antimicrobial": frozenset({"MIC", "MIC50", "MBC"}),
    "cellular_growth": frozenset({"GI50", "CC50", "TD50", "LC50"}),
}


def family_of(activity_type: str) -> str | None:
    """Return the activity family of ``activity_type``, or ``None`` if unknown."""
    for name, members in ACTIVITY_FAMILIES.items():
        if activity_type in members:
            return name
    return None


@dataclass(frozen=True, slots=True)
class CurationPolicy:
    """The switches governing which measurements survive.

    Every default is strict. Loosening one is a legitimate research decision,
    but it must be a decision: the policy is recorded in the curation report and
    in model metadata, so a reported R-squared always travels with the curation
    that produced it.

    Attributes:
        min_confidence_score: Lowest acceptable ChEMBL target confidence.
            ``None`` disables the check, which also accepts records where ChEMBL
            recorded no score at all.
        allowed_assay_types: Assay-type letters to keep. Empty set disables.
        allowed_families: Activity families to keep. Mixing families in one
            dataset requires naming more than one here, deliberately.
        require_exact_relation: Keep only ``relation == "="``.
        reject_validity_flagged: Drop records carrying a
            ``data_validity_comment``.
        reject_potential_duplicates: Drop records ChEMBL flagged as duplicates.
        plausible_pactivity_range: Inclusive bounds outside which a pActivity is
            treated as a unit error in the source record.
        require_smiles: Drop records with no structure -- nothing can be
            computed from them.
        max_pchembl_disagreement: If the source supplies its own ``pchembl_value``
            and ours differs by more than this many log units, drop the record:
            one of the two conversions is wrong and we cannot tell which.
            ``None`` disables the cross-check.
    """

    min_confidence_score: int | None = 8
    allowed_assay_types: frozenset[str] = TARGET_ENGAGEMENT_ASSAY_TYPES
    allowed_families: frozenset[str] = frozenset({"functional"})
    require_exact_relation: bool = True
    reject_validity_flagged: bool = True
    reject_potential_duplicates: bool = True
    plausible_pactivity_range: tuple[float, float] = (2.0, 12.0)
    require_smiles: bool = True
    max_pchembl_disagreement: float | None = 0.1

    def describe(self) -> str:
        """Human-readable policy summary, embedded in model metadata."""
        lines = [
            f"min_confidence_score={self.min_confidence_score}",
            f"allowed_assay_types={sorted(self.allowed_assay_types) or 'any'}",
            f"allowed_families={sorted(self.allowed_families) or 'any'}",
            f"require_exact_relation={self.require_exact_relation}",
            f"reject_validity_flagged={self.reject_validity_flagged}",
            f"reject_potential_duplicates={self.reject_potential_duplicates}",
            f"plausible_pactivity_range={self.plausible_pactivity_range}",
            f"max_pchembl_disagreement={self.max_pchembl_disagreement}",
        ]
        return "; ".join(lines)


#: A policy that keeps everything structurally usable. Useful for inspecting
#: how much the strict policy removes and why -- never for training a model
#: whose metrics you intend to report.
PERMISSIVE_POLICY = CurationPolicy(
    min_confidence_score=None,
    allowed_assay_types=frozenset(),
    allowed_families=frozenset(),
    require_exact_relation=True,
    reject_validity_flagged=False,
    reject_potential_duplicates=False,
    max_pchembl_disagreement=None,
)


def _check_smiles(record: ActivityRecord, policy: CurationPolicy) -> str | None:
    if policy.require_smiles and not (record.smiles or "").strip():
        return "no structure reported"
    return None


def _check_confidence(record: ActivityRecord, policy: CurationPolicy) -> str | None:
    if policy.min_confidence_score is None:
        return None
    if record.confidence_score is None:
        return (
            "no target-confidence score recorded; cannot establish the "
            "measurement is against the intended protein"
        )
    if record.confidence_score < policy.min_confidence_score:
        return (
            f"confidence_score {record.confidence_score} below "
            f"{policy.min_confidence_score}; target assignment too weak"
        )
    return None


def _check_assay_type(record: ActivityRecord, policy: CurationPolicy) -> str | None:
    if not policy.allowed_assay_types:
        return None
    assay_type = (record.assay_type or "").strip().upper()
    if not assay_type:
        return "no assay type recorded"
    if assay_type not in policy.allowed_assay_types:
        return (
            f"assay_type '{assay_type}' not in "
            f"{sorted(policy.allowed_assay_types)}; measures something other "
            "than target engagement"
        )
    return None


def _check_family(record: ActivityRecord, policy: CurationPolicy) -> str | None:
    if not policy.allowed_families:
        return None
    family = family_of((record.activity_type or "").strip())
    if family is None:
        return (
            f"activity_type '{record.activity_type}' belongs to no known "
            "family; refusing to pool it with recognised endpoints"
        )
    if family not in policy.allowed_families:
        return (
            f"activity family '{family}' not in {sorted(policy.allowed_families)}; "
            "pooling incommensurable endpoints would corrupt the target"
        )
    return None


def _check_validity_comment(
    record: ActivityRecord, policy: CurationPolicy
) -> str | None:
    if not policy.reject_validity_flagged:
        return None
    comment = (record.data_validity_comment or "").strip()
    if comment:
        return f"source flagged the value: '{comment}'"
    return None


def _check_duplicate(record: ActivityRecord, policy: CurationPolicy) -> str | None:
    if policy.reject_potential_duplicates and record.potential_duplicate:
        return "source flagged this as a potential duplicate measurement"
    return None


#: Ordered structural checks. Order matters only for which reason gets reported
#: first; each is independent.
_STRUCTURAL_CHECKS: tuple[
    tuple[str, Callable[[ActivityRecord, CurationPolicy], str | None]], ...
] = (
    ("missing_structure", _check_smiles),
    ("low_target_confidence", _check_confidence),
    ("wrong_assay_type", _check_assay_type),
    ("incommensurable_endpoint", _check_family),
    ("source_flagged_invalid", _check_validity_comment),
    ("potential_duplicate", _check_duplicate),
)


@dataclass(slots=True)
class FilterOutcome:
    """Survivors and rejections from a filtering pass."""

    kept: list[tuple[ActivityRecord, float]] = field(default_factory=list)
    """Records that survived, each paired with its converted pActivity."""

    rejected: list[Rejection] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return len(self.kept) + len(self.rejected)


def filter_records(
    records: Iterable[ActivityRecord],
    policy: CurationPolicy | None = None,
) -> FilterOutcome:
    """Apply ``policy`` to ``records``, returning survivors and reasoned rejections.

    Each survivor is paired with its pActivity, converted by
    :func:`chemdisco.units.pactivity` with full unit awareness. A record whose
    unit cannot be converted is rejected with that reason rather than silently
    assumed to be nanomolar.

    The function is pure: no network, no RDKit, no mutation of the inputs. That
    is what lets the whole curation policy be unit-tested without a chemistry
    toolkit installed.
    """
    policy = policy or CurationPolicy()
    outcome = FilterOutcome()

    for record in records:
        reason: str | None = None
        rule = ""
        for rule_name, check in _STRUCTURAL_CHECKS:
            reason = check(record, policy)
            if reason is not None:
                rule = rule_name
                break
        if reason is not None:
            outcome.rejected.append(Rejection(record, rule, reason))
            continue

        if policy.require_exact_relation and (record.relation or "=").strip() not in (
            "=",
            "==",
        ):
            outcome.rejected.append(
                Rejection(
                    record,
                    "censored_measurement",
                    f"relation '{record.relation}' is a bound, not a point value",
                )
            )
            continue

        quantity = pactivity(
            record.value,
            record.unit,
            activity_type=record.activity_type,
            source=record.citation,
            relation=record.relation,
            molecular_weight=record.molecular_weight,
        )

        if not quantity.is_known:
            outcome.rejected.append(
                Rejection(record, "unconvertible_value", quantity.source)
            )
            continue

        value = float(quantity.value)  # type: ignore[arg-type]
        low, high = policy.plausible_pactivity_range
        if not low <= value <= high:
            outcome.rejected.append(
                Rejection(
                    record,
                    "implausible_pactivity",
                    f"pActivity {value:.2f} outside plausible range "
                    f"[{low}, {high}]; probable unit error in the source",
                )
            )
            continue

        if (
            policy.max_pchembl_disagreement is not None
            and record.pchembl_value is not None
        ):
            delta = abs(value - float(record.pchembl_value))
            if delta > policy.max_pchembl_disagreement:
                outcome.rejected.append(
                    Rejection(
                        record,
                        "pchembl_disagreement",
                        f"our pActivity {value:.2f} disagrees with source "
                        f"pchembl_value {record.pchembl_value:.2f} by "
                        f"{delta:.2f} log units; one conversion is wrong",
                    )
                )
                continue

        outcome.kept.append((record, value))

    return outcome


def summarise_rejections(rejections: Sequence[Rejection]) -> dict[str, int]:
    """Count rejections by rule, highest first."""
    counts: dict[str, int] = {}
    for rejection in rejections:
        counts[rejection.rule] = counts.get(rejection.rule, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
