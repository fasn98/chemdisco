"""Aggregating replicate measurements into one value per compound and target.

Public databases contain the same compound-target pair many times over, from
different laboratories, years and assay protocols. Three things can go wrong if
this is handled carelessly:

1. **Train/test leakage.** If the same compound appears in both splits under two
   activity ids, a model can memorise it and the held-out score is meaningless.
   Collapsing replicates before splitting is what prevents this, which is why
   aggregation must happen *before* :mod:`chemdisco.split`, not after.
2. **Outlier domination.** A mean is pulled by a single mistyped value. The
   median is used here throughout.
3. **Silently averaging a contradiction.** Two labs reporting 10 nM and 10 uM
   for the same pair disagree by three log units; their median is a number no
   experiment ever produced. Such pairs are discarded, not averaged, because the
   honest statement is "the literature disagrees" rather than a fabricated
   midpoint.

The experimental noise floor for an inter-laboratory IC50 comparison is roughly
0.5 log units, and ChEMBL-wide analyses put the standard deviation of replicate
pIC50 measurements near 0.7. A spread tolerance of 1.0 log unit is therefore
near the limit of what can be called agreement, and is the default here.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from ..provenance import Origin, Quantity
from .records import ActivityRecord, CuratedPoint, CurationReport, Rejection

#: Default maximum spread, in log units, between replicate measurements of the
#: same compound-target pair that will still be treated as agreement.
DEFAULT_MAX_SPREAD_LOG_UNITS = 1.0


def default_compound_key(record: ActivityRecord) -> str:
    """Group replicates by the source's compound identifier.

    Adequate within a single database export, where one molecule has one id.
    It is **not** adequate across databases or where salts and tautomers appear
    as separate entries: ChEMBL lists the free base and the hydrochloride of one
    drug under different ids, and grouping by id leaves both in the dataset as
    independent compounds, which leaks across a split.

    For cross-source work, pass a key function built on a standardised
    structure hash -- see :func:`chemdisco.chem.standardize.inchikey_of`, which
    needs RDKit and therefore lives at the toolkit edge.
    """
    return record.compound_id


@dataclass(frozen=True, slots=True)
class AggregationPolicy:
    """How replicate measurements are combined.

    Attributes:
        max_spread_log_units: Discard a compound-target group whose measurements
            span more than this. ``None`` disables the check and aggregates
            regardless of disagreement, which is not recommended.
        min_measurements: Require at least this many measurements per group.
            Raising it above 1 trades dataset size for confidence.
        statistic: ``"median"`` or ``"mean"``. Median is the default and the
            recommended choice; mean is offered for comparison with published
            work that used it.
        report_spread_as_uncertainty: Attach half the spread to the aggregated
            quantity as an uncertainty estimate, so downstream code can see how
            well-determined each label is.
    """

    max_spread_log_units: float | None = DEFAULT_MAX_SPREAD_LOG_UNITS
    min_measurements: int = 1
    statistic: str = "median"
    report_spread_as_uncertainty: bool = True

    def __post_init__(self) -> None:
        if self.statistic not in ("median", "mean"):
            raise ValueError("statistic must be 'median' or 'mean'")
        if self.min_measurements < 1:
            raise ValueError("min_measurements must be at least 1")

    def describe(self) -> str:
        return (
            f"statistic={self.statistic}; "
            f"max_spread_log_units={self.max_spread_log_units}; "
            f"min_measurements={self.min_measurements}"
        )


def _combine(values: Sequence[float], statistic: str) -> float:
    if statistic == "median":
        return float(statistics.median(values))
    return float(statistics.fmean(values))


def aggregate_measurements(
    measurements: Iterable[tuple[ActivityRecord, float]],
    policy: AggregationPolicy | None = None,
    compound_key: Callable[[ActivityRecord], str] = default_compound_key,
) -> tuple[list[CuratedPoint], list[Rejection]]:
    """Collapse replicate measurements into one :class:`CuratedPoint` per pair.

    Args:
        measurements: Pairs of record and already-converted pActivity, as
            returned by :func:`chemdisco.curate.filters.filter_records`.
        policy: Aggregation rules; defaults to :class:`AggregationPolicy`.
        compound_key: How to decide two records describe the same compound.
            Defaults to the source identifier; pass a structure-hash function
            for cross-source datasets.

    Returns:
        A tuple of curated points and rejections. Rejections here are whole
        groups: every measurement in a discarded group is reported, so the
        accounting against the input remains exact.
    """
    policy = policy or AggregationPolicy()

    groups: dict[tuple[str, str], list[tuple[ActivityRecord, float]]] = defaultdict(list)
    for record, value in measurements:
        groups[(compound_key(record), record.target_id)].append((record, value))

    kept: list[CuratedPoint] = []
    rejected: list[Rejection] = []

    for (_, target_id), members in groups.items():
        values = [value for _, value in members]
        records = [record for record, _ in members]

        if len(values) < policy.min_measurements:
            for record in records:
                rejected.append(
                    Rejection(
                        record,
                        "too_few_replicates",
                        f"{len(values)} measurement(s), policy requires "
                        f"{policy.min_measurements}",
                    )
                )
            continue

        spread = max(values) - min(values) if len(values) > 1 else 0.0

        if (
            policy.max_spread_log_units is not None
            and spread > policy.max_spread_log_units
        ):
            detail = (
                f"{len(values)} measurements span {spread:.2f} log units "
                f"(max {policy.max_spread_log_units}); the sources disagree and "
                "an average would be a value no experiment produced"
            )
            for record in records:
                rejected.append(Rejection(record, "irreconcilable_replicates", detail))
            continue

        combined = _combine(values, policy.statistic)
        activity_ids = tuple(record.activity_id for record in records)
        activity_types = tuple(sorted({record.activity_type for record in records}))

        notes = [
            f"{policy.statistic} of {len(values)} measurement(s)",
            f"types={','.join(activity_types)}",
        ]
        if len(values) > 1:
            notes.append(f"spread={spread:.2f} log units")

        uncertainty: float | None = None
        if policy.report_spread_as_uncertainty and len(values) > 1:
            # Half-spread is a crude but honest dispersion estimate for the
            # small replicate counts typical here, where a sample standard
            # deviation from two or three points is not meaningful.
            uncertainty = spread / 2.0

        quantity = Quantity(
            value=combined,
            unit=None,
            origin=Origin.DERIVED,
            source="; ".join(activity_ids),
            uncertainty=uncertainty,
            notes=tuple(notes),
        )

        # The representative structure comes from the first record in the group.
        # Members share a compound key, so within a single source they share a
        # structure; across sources the caller's key function is responsible for
        # that guarantee.
        kept.append(
            CuratedPoint(
                compound_id=records[0].compound_id,
                smiles=records[0].smiles,
                target_id=target_id,
                pactivity=quantity,
                n_measurements=len(values),
                source_activity_ids=activity_ids,
                spread_log_units=spread if len(values) > 1 else None,
                activity_types=activity_types,
            )
        )

    kept.sort(key=lambda point: (point.target_id, point.compound_id))
    return kept, rejected


def curate(
    records: Sequence[ActivityRecord],
    *,
    policy=None,
    aggregation: AggregationPolicy | None = None,
    compound_key: Callable[[ActivityRecord], str] = default_compound_key,
) -> CurationReport:
    """Run the full curation pipeline: filter, then aggregate.

    This is the function a caller should use. It guarantees the accounting
    invariant that every input measurement is either represented in a kept point
    or named in a rejection -- the property that makes a curation report
    reviewable.

    Args:
        records: Raw measurements from a database client.
        policy: A :class:`chemdisco.curate.filters.CurationPolicy`. Imported
            lazily to keep this module free of a circular import.
        aggregation: Replicate-combination rules.
        compound_key: Compound identity function.
    """
    from .filters import filter_records  # local import avoids a cycle

    outcome = filter_records(records, policy)
    kept, aggregation_rejections = aggregate_measurements(
        outcome.kept, aggregation, compound_key
    )

    return CurationReport(
        kept=tuple(kept),
        rejected=tuple(outcome.rejected) + tuple(aggregation_rejections),
        n_input=len(records),
    )
