"""The activity record: one measurement, as reported, before any judgement.

Deliberately dumb. A record holds what the source database said, including the
fields a curator needs in order to *reject* it -- assay type, confidence score,
validity comment, duplicate flag. The predecessor project dropped these on
import and so had no way to tell a direct single-protein binding assay from a
whole-organism phenotypic readout, and no way to notice that ChEMBL itself had
flagged a value as a probable transcription error.

Keeping the rejection metadata on the record means curation decisions are
auditable after the fact: every dropped measurement can say why it was dropped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..provenance import Quantity


@dataclass(frozen=True, slots=True)
class ActivityRecord:
    """A single reported bioactivity measurement.

    Attributes:
        activity_id: Source-database identifier for this measurement, used as
            the citation in derived quantities.
        compound_id: Source identifier of the compound (e.g. ChEMBL molecule id).
        smiles: Structure exactly as the source reported it, unstandardised.
            Standardisation happens at the RDKit edge, not here.
        target_id: Source identifier of the target.
        activity_type: ``standard_type``, e.g. ``"IC50"``, ``"Ki"``, ``"Potency"``.
        value: ``standard_value`` as reported.
        unit: ``standard_units`` as reported.
        relation: ``standard_relation``; ``"="`` for a point measurement.
        assay_id: Source identifier of the assay protocol.
        assay_type: ChEMBL assay type letter -- ``B`` binding, ``F`` functional,
            ``A`` ADMET, ``T`` toxicity, ``P`` physicochemical, ``U`` unclassified.
        confidence_score: ChEMBL target-assignment confidence, 0-9. A score of 9
            is a direct single-protein target; 8 is a homologous single protein;
            below that the measurement may be against a complex, a family, or a
            whole organism.
        data_validity_comment: ChEMBL's own warning about the value, e.g.
            ``"Outside typical range"`` or ``"Potential transcription error"``.
            ``None`` when ChEMBL raised nothing.
        potential_duplicate: ChEMBL's flag that this value likely duplicates
            another record.
        pchembl_value: ChEMBL's own pre-computed pActivity. When present it is
            an independent check on our unit conversion, not a replacement for
            it -- a disagreement means one of the two is wrong and the record
            deserves attention.
        molecular_weight: Needed only to convert mass-per-volume units.
        extra: Any further source fields, carried through untouched.
    """

    activity_id: str
    compound_id: str
    smiles: str
    target_id: str
    activity_type: str
    value: float | None
    unit: str | None
    relation: str = "="
    assay_id: str = ""
    assay_type: str = ""
    confidence_score: int | None = None
    data_validity_comment: str | None = None
    potential_duplicate: bool = False
    pchembl_value: float | None = None
    molecular_weight: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """Citation string used as provenance for anything derived from this."""
        parts = [self.activity_id or "unknown-activity"]
        if self.assay_id:
            parts.append(f"assay {self.assay_id}")
        if self.target_id:
            parts.append(f"target {self.target_id}")
        return " / ".join(parts)


@dataclass(frozen=True, slots=True)
class Rejection:
    """A record that did not survive curation, with the reason.

    Rejections are returned alongside survivors rather than silently dropped.
    A curation step that cannot report its losses cannot be reviewed, and the
    loss rate is itself a signal: discarding 95% of a target's data usually
    means the filter is wrong, not that the data is.
    """

    record: ActivityRecord
    rule: str
    detail: str

    def as_row(self) -> dict[str, Any]:
        return {
            "activity_id": self.record.activity_id,
            "compound_id": self.record.compound_id,
            "target_id": self.record.target_id,
            "rule": self.rule,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class CuratedPoint:
    """One compound, one target, one defensible activity value.

    The product of curation: a structure plus a pActivity quantity that carries
    its full provenance, plus the identifiers of every measurement that went
    into it so the aggregation can be undone and inspected.
    """

    compound_id: str
    smiles: str
    target_id: str
    pactivity: Quantity
    n_measurements: int
    source_activity_ids: tuple[str, ...]
    spread_log_units: float | None = None
    activity_types: tuple[str, ...] = ()

    def as_row(self) -> dict[str, Any]:
        return {
            "compound_id": self.compound_id,
            "smiles": self.smiles,
            "target_id": self.target_id,
            "pactivity": self.pactivity.value,
            "pactivity_label": self.pactivity.label(),
            "origin": self.pactivity.origin.value,
            "n_measurements": self.n_measurements,
            "spread_log_units": self.spread_log_units,
            "activity_types": ",".join(self.activity_types),
            "source_activity_ids": ",".join(self.source_activity_ids),
        }


@dataclass(frozen=True, slots=True)
class CurationReport:
    """Everything curation produced, kept together so it can be reported.

    ``kept`` and ``rejected`` always account for the whole input. The invariant
    is checked in tests: a curation step that loses records without recording
    them is a bug.
    """

    kept: tuple[CuratedPoint, ...]
    rejected: tuple[Rejection, ...]
    n_input: int

    @property
    def n_kept(self) -> int:
        return len(self.kept)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)

    @property
    def retention(self) -> float | None:
        """Fraction of input measurements that survived, or ``None`` if empty.

        ``None`` for an empty input rather than 0.0 or 1.0: with nothing in,
        there is no retention rate, and either number would be a claim the data
        does not support.
        """
        if self.n_input == 0:
            return None
        return (self.n_input - self.n_rejected) / self.n_input

    def rejection_summary(self) -> dict[str, int]:
        """Count of rejections per rule, for the curation report."""
        counts: dict[str, int] = {}
        for rejection in self.rejected:
            counts[rejection.rule] = counts.get(rejection.rule, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def describe(self) -> str:
        """Plain-language account of what curation did, for logs and reports."""
        lines = [
            f"{self.n_input} measurements in, "
            f"{self.n_kept} curated points out, "
            f"{self.n_rejected} measurements rejected.",
        ]
        for rule, count in self.rejection_summary().items():
            lines.append(f"  {count:>6} rejected by {rule}")
        if self.n_input and self.n_kept == 0:
            lines.append(
                "  WARNING: nothing survived curation. Check the filters before "
                "concluding the target has no data."
            )
        return "\n".join(lines)


def records_to_rows(records: Sequence[ActivityRecord]) -> list[dict[str, Any]]:
    """Flatten records for inspection in a dataframe or CSV."""
    return [
        {
            "activity_id": r.activity_id,
            "compound_id": r.compound_id,
            "target_id": r.target_id,
            "activity_type": r.activity_type,
            "value": r.value,
            "unit": r.unit,
            "relation": r.relation,
            "assay_type": r.assay_type,
            "confidence_score": r.confidence_score,
            "data_validity_comment": r.data_validity_comment,
            "pchembl_value": r.pchembl_value,
            "smiles": r.smiles,
        }
        for r in records
    ]
