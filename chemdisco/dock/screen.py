"""Running a virtual screen: many ligands, one receptor, honest accounting.

Batch docking is mostly bookkeeping, and the bookkeeping is where screens go
wrong. Two failures in particular:

**Silent attrition.** Ligands that fail preparation disappear from the result
set. If failures correlate with a group -- and they do, because large flexible
peptidomimetics fail embedding more often than rigid heterocycles -- then the
actives and the decoys are thinned at different rates and the enrichment figure
is measuring that instead of binding. :class:`ScreenResult` therefore accounts
for every input ligand and reports the failure rate per group.

**Comparing across receptors or boxes.** Vina scores are only comparable within
one receptor and one search box. Mixing runs is how a screen produces a ranked
list ordered mostly by which structure each ligand happened to be docked into,
so the receptor and box are recorded on the result and checked.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .box import Box
from .engine import DockingResult, dock


@dataclass(slots=True)
class ScreenResult:
    """Everything a virtual screen produced, successes and failures alike.

    Attributes:
        results: One :class:`DockingResult` per input ligand, in input order,
            including the failures. Keeping the failures in place is what makes
            the labels stay aligned with the scores.
        labels: Optional group label per ligand -- 1 for a known active, 0 for a
            decoy, used by the enrichment analysis.
        receptor_id: The single receptor every ligand was docked into.
        box: The single search box used.
        elapsed_seconds: Wall-clock time, for planning larger runs.
        stopped_early: Set when a time budget cut the run short, so a partial
            screen is never mistaken for a complete one.
        n_requested: How many ligands were asked for, which differs from
            ``n_total`` when the run stopped early.
    """

    results: list[DockingResult] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    receptor_id: str = ""
    box: Box | None = None
    elapsed_seconds: float = 0.0
    stopped_early: str = ""
    n_requested: int = 0

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def n_succeeded(self) -> int:
        return sum(1 for result in self.results if result.ok)

    @property
    def n_failed(self) -> int:
        return self.n_total - self.n_succeeded

    @property
    def seconds_per_ligand(self) -> float:
        """Throughput, which is what a larger run has to be planned against."""
        return self.elapsed_seconds / self.n_total if self.n_total else 0.0

    def failure_reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in self.results:
            if result.ok:
                continue
            # Collapse to the leading clause so similar failures group together.
            reason = (result.error or "unknown").split(":")[0].strip()
            counts[reason] = counts.get(reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def failure_rate_by_group(self) -> dict[int, float]:
        """Failure rate per label group.

        The number that decides whether an enrichment figure is trustworthy. If
        actives fail preparation at 5% and decoys at 25%, the two groups were
        thinned differently and the surviving sets are no longer comparable.
        """
        if not self.labels:
            return {}
        totals: dict[int, int] = {}
        failures: dict[int, int] = {}
        for result, label in zip(self.results, self.labels, strict=True):
            totals[label] = totals.get(label, 0) + 1
            if not result.ok:
                failures[label] = failures.get(label, 0) + 1
        return {
            label: failures.get(label, 0) / count for label, count in totals.items()
        }

    def scored(self) -> tuple[list[int], list[float]]:
        """Labels and scores for the ligands that docked successfully.

        Returns:
            ``(labels, scores)`` aligned with each other. Raises if no labels
            were supplied, since an enrichment analysis without them is not
            possible.
        """
        if not self.labels:
            raise ValueError(
                "no labels were supplied, so actives cannot be separated from "
                "decoys"
            )
        labels: list[int] = []
        scores: list[float] = []
        for result, label in zip(self.results, self.labels, strict=True):
            if result.ok and result.best_score is not None:
                labels.append(label)
                scores.append(result.best_score)
        return labels, scores

    def describe(self) -> str:
        lines = [
            f"Screened {self.n_total} ligand(s) against "
            f"{self.receptor_id or 'a receptor'} in {self.elapsed_seconds:.0f}s"
            + (
                f" ({self.seconds_per_ligand:.1f}s each)"
                if self.seconds_per_ligand
                else ""
            ),
            f"  {self.n_succeeded} docked, {self.n_failed} failed",
        ]
        if self.stopped_early:
            lines.append(f"  STOPPED EARLY: {self.stopped_early}")
        for reason, count in self.failure_reasons().items():
            lines.append(f"    {count:>5} {reason}")

        rates = self.failure_rate_by_group()
        if len(rates) > 1:
            formatted = ", ".join(
                f"{'actives' if label == 1 else 'decoys'} {rate:.1%}"
                for label, rate in sorted(rates.items(), reverse=True)
            )
            lines.append(f"  failure rate by group: {formatted}")
            spread = max(rates.values()) - min(rates.values())
            if spread > 0.1:
                lines.append(
                    f"  WARNING: the groups failed at rates differing by "
                    f"{spread:.0%}. They were thinned unevenly, so the surviving "
                    "sets are no longer comparable and any enrichment figure "
                    "below is partly measuring that difference."
                )
        return "\n".join(lines)


def screen(
    smiles_list: Sequence[str],
    receptor_pdbqt: str,
    box: Box,
    *,
    labels: Sequence[int] | None = None,
    receptor_id: str = "",
    progress: Callable[[int, int, DockingResult], None] | None = None,
    time_budget_seconds: float | None = None,
    **dock_kwargs,
) -> ScreenResult:
    """Dock every ligand in ``smiles_list`` into one receptor.

    Args:
        smiles_list: Ligands to dock.
        receptor_pdbqt: One prepared receptor, used for all of them. Scores are
            only comparable within a single receptor and box.
        box: One search box, likewise.
        labels: Optional 1/0 group labels, required for enrichment analysis.
        receptor_id: Recorded on the result and in each score's provenance.
        progress: Called after each ligand with ``(index, total, result)``.
        time_budget_seconds: Stop and return what has been docked so far once
            this much time has elapsed. Without it a screen that outruns its
            environment's limit is killed and returns nothing -- which is how an
            hour of docking produces no data at all. Stopping deliberately keeps
            the partial result and records that it is partial.
        **dock_kwargs: Passed through to :func:`chemdisco.dock.engine.dock`.

    Returns:
        A :class:`ScreenResult` holding one entry per docked ligand, failures
        included and in input order, so the labels stay aligned.

    Note:
        When a budget cuts the run short, the ligands docked are a *prefix* of
        the input. If actives and decoys are supplied in blocks, the prefix is
        all actives and no decoys, which is useless. Interleave the groups before
        calling, or accept that a truncated run must be discarded.
    """
    if labels is not None and len(labels) != len(smiles_list):
        raise ValueError(
            f"{len(labels)} labels for {len(smiles_list)} ligands"
        )

    started = time.monotonic()
    result = ScreenResult(
        labels=[],
        receptor_id=receptor_id,
        box=box,
        n_requested=len(smiles_list),
    )
    all_labels = list(labels) if labels is not None else []

    for index, smiles in enumerate(smiles_list):
        if time_budget_seconds is not None:
            elapsed = time.monotonic() - started
            if elapsed > time_budget_seconds:
                result.stopped_early = (
                    f"time budget of {time_budget_seconds:.0f}s reached after "
                    f"{index} of {len(smiles_list)} ligands"
                )
                break
        docked = dock(
            smiles, receptor_pdbqt, box, receptor_id=receptor_id, **dock_kwargs
        )
        result.results.append(docked)
        if all_labels:
            result.labels.append(all_labels[index])
        if progress is not None:
            progress(index + 1, len(smiles_list), docked)

    result.elapsed_seconds = time.monotonic() - started
    return result


def shard_by_label(
    smiles_list: Sequence[str],
    labels: Sequence[int],
    *,
    shard: int,
    n_shards: int,
) -> tuple[list[str], list[int]]:
    """Take this shard's slice, keeping every group proportionally represented.

    Striding an interleaved list looks like it should work and does not. Round-
    robin interleaving of two groups is periodic, so taking every n-th element
    lands on a fixed phase: a six-way stride of a decoy-active-decoy-active list
    gave three shards with 13-14 actives each and three with none. All six
    completed, so the pooled result was unharmed -- but a single shard running
    out of budget would have skewed the pool badly, and a shard that is all
    decoys cannot be analysed on its own at all.

    Dealing each group separately keeps every shard a miniature of the whole.
    """
    if len(smiles_list) != len(labels):
        raise ValueError(f"{len(smiles_list)} ligands against {len(labels)} labels")
    if n_shards < 1:
        raise ValueError("n_shards must be at least 1")
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard {shard} outside 0..{n_shards - 1}")

    groups: dict[int, list[str]] = {}
    for smiles, label in zip(smiles_list, labels, strict=True):
        groups.setdefault(label, []).append(smiles)

    chosen_smiles: list[str] = []
    chosen_labels: list[int] = []
    for label in sorted(groups):
        members = groups[label][shard::n_shards]
        chosen_smiles.extend(members)
        chosen_labels.extend([label] * len(members))

    # Interleave within the shard too, so a budget cut leaves it balanced.
    return interleave_by_label(chosen_smiles, chosen_labels)


def interleave_by_label(
    smiles_list: Sequence[str], labels: Sequence[int]
) -> tuple[list[str], list[int]]:
    """Reorder so each group is spread evenly through the run.

    Necessary whenever a time budget might truncate a screen. Supplied in
    blocks, a truncated run holds every active and no decoys, which cannot
    support an enrichment estimate at all. Interleaved, a truncated run is a
    smaller but still balanced screen -- less precise, but usable.
    """
    if len(smiles_list) != len(labels):
        raise ValueError(f"{len(smiles_list)} ligands against {len(labels)} labels")

    groups: dict[int, list[str]] = {}
    for smiles, label in zip(smiles_list, labels, strict=True):
        groups.setdefault(label, []).append(smiles)

    ordered_smiles: list[str] = []
    ordered_labels: list[int] = []
    # Round-robin across the groups, largest first, so the ratio holds at every
    # prefix rather than only at the end.
    while any(groups.values()):
        for label in sorted(groups, key=lambda key: -len(groups[key])):
            if groups[label]:
                ordered_smiles.append(groups[label].pop(0))
                ordered_labels.append(label)
    return ordered_smiles, ordered_labels


def triage_candidates(
    screen_result: ScreenResult,
    reference_scores: Sequence[float],
    *,
    percentile: float = 50.0,
) -> tuple[list[int], str]:
    """Keep candidates scoring at least as well as the known actives typically do.

    The honest use of docking in this pipeline, given what was measured about it.
    Vina cannot reliably *order* candidates on this target -- the redocking study
    showed the pose ranking is not determined by the score, and enrichment is
    modest at best -- but a candidate scoring far worse than every known active
    almost certainly does not fit the pocket. So docking is used as a filter
    against a reference distribution, not as a ranking.

    Args:
        screen_result: Docked candidates.
        reference_scores: Scores of known actives, docked in the *same* receptor
            and box. Mixing runs makes the comparison meaningless.
        percentile: Reference percentile a candidate must beat. The default
            median is deliberately permissive: the aim is discarding the clearly
            implausible, not selecting winners.

    Returns:
        ``(kept_indices, explanation)``. Indices refer to positions in
        ``screen_result.results``.
    """
    import numpy as np

    if not reference_scores:
        raise ValueError(
            "no reference scores supplied; a docking score has no meaning "
            "without a distribution from known actives in the same setup"
        )

    threshold = float(np.percentile(np.asarray(reference_scores, dtype=float), percentile))
    kept = [
        index
        for index, result in enumerate(screen_result.results)
        if result.ok
        and result.best_score is not None
        and result.best_score <= threshold
    ]

    explanation = (
        f"Kept {len(kept)} of {screen_result.n_succeeded} docked candidates "
        f"scoring at or below {threshold:.2f} kcal/mol, the {percentile:.0f}th "
        f"percentile of {len(reference_scores)} known actives docked in the same "
        "receptor and box.\n"
        "This is a filter, not a ranking. Docking on this target does not order "
        "candidates reliably, so the survivors are not ranked against each other "
        "and their order carries no information."
    )
    return kept, explanation
