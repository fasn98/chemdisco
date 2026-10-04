"""Property-matched decoys: the control a virtual screen needs.

A docking score that separates known actives from random library compounds has
proved almost nothing. Vina's score grows close to linearly with molecular size,
and actives for most targets are larger and more lipophilic than a random
compound, so a screen can report excellent enrichment while measuring nothing
but molecular weight.

The standard correction, from the DUD-E benchmark set, is to choose decoys that
match the actives on the physicochemical properties a scoring function is biased
by -- molecular weight, lipophilicity, hydrogen-bond donors and acceptors,
rotatable bonds, net charge -- while being topologically *dissimilar*, so they
are unlikely to share the actives' binding mode. Enrichment measured against
such a set cannot be explained by size, and what remains is attributable to fit.

Two limitations, stated because they bound what any result here can claim:

**Decoys are presumed inactive, not known inactive.** They are compounds nobody
has tested against this target. A small fraction genuinely bind, which depresses
measured enrichment slightly. This is the standard compromise -- real measured
inactives are scarce, because negative results are rarely published.

**Property matching can be gamed by the matching itself.** If the decoy pool is
too small, the matcher is forced to accept poor matches and the property bias
creeps back. :class:`DecoySelection` reports the achieved match quality so that
failure is visible rather than assumed away.

This module is pure. The properties come in as numbers and the similarity as a
matrix, so the selection logic -- where a subtle bias would hide -- is testable
without a chemistry toolkit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

#: Properties matched between actives and decoys, with the tolerance for each.
#: Tolerances follow DUD-E's: wide enough that a pool of ordinary compounds can
#: supply matches, tight enough that a scoring function cannot tell the groups
#: apart on these numbers alone.
DEFAULT_TOLERANCES: dict[str, float] = {
    "molecular_weight": 25.0,
    "logp": 1.0,
    "hbd": 1.0,
    "hba": 2.0,
    "rotatable_bonds": 2.0,
    "formal_charge": 0.0,
}

#: Maximum Tanimoto similarity between a decoy and any active. Above this the
#: decoy may share the actives' binding mode, which would make it a probable
#: active rather than a control. DUD-E uses 0.25 on ECFP4.
MAX_DECOY_SIMILARITY = 0.25

#: Decoys per active. DUD-E uses 50; this defaults lower because every decoy
#: costs a docking run, and the enrichment estimate's precision improves only as
#: the square root of the count.
DEFAULT_DECOYS_PER_ACTIVE = 10


@dataclass(frozen=True, slots=True)
class DecoySelection:
    """Chosen decoys, and how well they actually match.

    The match quality is reported rather than assumed. A matcher forced to accept
    poor matches by a thin candidate pool reintroduces exactly the bias the
    decoys exist to remove, and that must be visible.

    Attributes:
        decoy_indices: Positions in the candidate pool that were selected.
        active_indices: Positions in the actives list each decoy was matched to.
        property_deltas: Per-property mean absolute difference between each
            decoy and its matched active.
        max_similarity_used: Highest active-decoy similarity among those chosen.
        n_requested: How many decoys were asked for.
        shortfall_reasons: Why fewer were found, when fewer were.
    """

    decoy_indices: tuple[int, ...]
    active_indices: tuple[int, ...]
    property_deltas: dict[str, float] = field(default_factory=dict)
    max_similarity_used: float = 0.0
    n_requested: int = 0
    shortfall_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def n_selected(self) -> int:
        return len(self.decoy_indices)

    @property
    def is_adequate(self) -> bool:
        """Whether this selection can support an enrichment claim.

        Requires most of the requested decoys and no property drifting beyond
        its tolerance on average. A selection failing this does not invalidate a
        screen, but any enrichment measured against it may be measuring the
        property gap instead of binding.
        """
        if self.n_requested and self.n_selected < 0.5 * self.n_requested:
            return False
        return all(
            delta <= DEFAULT_TOLERANCES.get(name, float("inf"))
            for name, delta in self.property_deltas.items()
        )

    def describe(self) -> str:
        lines = [
            f"{self.n_selected} decoys selected"
            + (f" of {self.n_requested} requested" if self.n_requested else "")
        ]
        if self.property_deltas:
            lines.append("  mean |active - decoy| per property:")
            for name, delta in sorted(self.property_deltas.items()):
                tolerance = DEFAULT_TOLERANCES.get(name)
                flag = ""
                if tolerance is not None and delta > tolerance:
                    flag = f"  EXCEEDS tolerance {tolerance}"
                lines.append(f"    {name}: {delta:.2f}{flag}")
        lines.append(
            f"  highest active-decoy similarity used: {self.max_similarity_used:.3f}"
        )
        if self.shortfall_reasons:
            lines.append("  candidates rejected:")
            for reason, count in sorted(
                self.shortfall_reasons.items(), key=lambda kv: -kv[1]
            ):
                lines.append(f"    {count:>6} by {reason}")
        if not self.is_adequate:
            lines.append(
                "  WARNING: this decoy set does not match the actives closely "
                "enough to control for the scoring function's size bias. "
                "Enrichment measured against it may be measuring the property "
                "gap rather than binding. Widen the candidate pool."
            )
        return "\n".join(lines)


def select_decoys(
    active_properties: Sequence[dict[str, float]],
    candidate_properties: Sequence[dict[str, float]],
    similarity_to_actives: np.ndarray,
    *,
    decoys_per_active: int = DEFAULT_DECOYS_PER_ACTIVE,
    tolerances: dict[str, float] | None = None,
    max_similarity: float = MAX_DECOY_SIMILARITY,
) -> DecoySelection:
    """Choose decoys matching the actives on properties but not on topology.

    Args:
        active_properties: One property dictionary per active.
        candidate_properties: One per compound in the decoy pool.
        similarity_to_actives: Shape ``(n_candidates, n_actives)``, higher is
            more similar. Tanimoto on ECFP4 is the usual choice.
        decoys_per_active: How many to select for each active.
        tolerances: Per-property maximum difference; defaults to
            :data:`DEFAULT_TOLERANCES`.
        max_similarity: Reject a candidate resembling any active more than this.

    Returns:
        A :class:`DecoySelection` reporting what was chosen and how well it
        matches. Each candidate is used at most once, so a pool smaller than the
        requested count yields a shortfall rather than duplicates -- duplicated
        decoys would make an enrichment estimate look more precise than the data
        supports.
    """
    tolerances = tolerances or DEFAULT_TOLERANCES
    similarity = np.asarray(similarity_to_actives, dtype=float)

    if similarity.ndim != 2:
        raise ValueError(f"similarity must be 2-D, got shape {similarity.shape}")
    if similarity.shape[0] != len(candidate_properties):
        raise ValueError(
            f"{similarity.shape[0]} similarity rows for "
            f"{len(candidate_properties)} candidates"
        )
    if similarity.shape[1] != len(active_properties):
        raise ValueError(
            f"{similarity.shape[1]} similarity columns for "
            f"{len(active_properties)} actives"
        )
    if not active_properties:
        raise ValueError("no actives supplied; there is nothing to match against")

    # A candidate too similar to ANY active is excluded outright: it might share
    # the binding mode and would then be a probable active, not a control.
    candidate_max_similarity = (
        similarity.max(axis=1) if similarity.shape[1] else np.zeros(len(candidate_properties))
    )

    taken: set[int] = set()
    chosen: list[int] = []
    matched_to: list[int] = []
    rejections: dict[str, int] = {}
    deltas: dict[str, list[float]] = {name: [] for name in tolerances}
    highest_similarity = 0.0

    def reject(reason: str) -> None:
        rejections[reason] = rejections.get(reason, 0) + 1

    for active_index, active in enumerate(active_properties):
        found = 0
        # Order candidates by how closely they match this active, so the best
        # available matches are used first rather than whichever came first in
        # the pool.
        scored: list[tuple[float, int]] = []
        for candidate_index, candidate in enumerate(candidate_properties):
            if candidate_index in taken:
                continue
            if candidate_max_similarity[candidate_index] > max_similarity:
                continue
            distance = 0.0
            fits = True
            for name, tolerance in tolerances.items():
                if name not in active or name not in candidate:
                    continue
                difference = abs(float(active[name]) - float(candidate[name]))
                if difference > tolerance:
                    fits = False
                    break
                # Normalise so no single property dominates the ordering.
                distance += difference / (tolerance if tolerance > 0 else 1.0)
            if fits:
                scored.append((distance, candidate_index))

        for _, candidate_index in sorted(scored):
            if found >= decoys_per_active:
                break
            taken.add(candidate_index)
            chosen.append(candidate_index)
            matched_to.append(active_index)
            highest_similarity = max(
                highest_similarity, float(candidate_max_similarity[candidate_index])
            )
            candidate = candidate_properties[candidate_index]
            for name in tolerances:
                if name in active and name in candidate:
                    deltas[name].append(
                        abs(float(active[name]) - float(candidate[name]))
                    )
            found += 1

        if found < decoys_per_active:
            reject(f"too few matches for active {active_index}")

    # Account for the pool-wide exclusions, which explain a shortfall.
    too_similar = int(np.sum(candidate_max_similarity > max_similarity))
    if too_similar:
        rejections[f"similarity above {max_similarity}"] = too_similar

    return DecoySelection(
        decoy_indices=tuple(chosen),
        active_indices=tuple(matched_to),
        property_deltas={
            name: float(np.mean(values)) for name, values in deltas.items() if values
        },
        max_similarity_used=highest_similarity,
        n_requested=len(active_properties) * decoys_per_active,
        shortfall_reasons=rejections,
    )


def property_gap(
    active_properties: Sequence[dict[str, float]],
    decoy_properties: Sequence[dict[str, float]],
) -> dict[str, float]:
    """Mean difference per property between the two groups.

    The diagnostic that says whether a measured enrichment could be explained by
    properties alone. A molecular-weight gap of 80 Da between actives and decoys
    is enough for Vina's size bias to produce impressive-looking enrichment with
    no binding information in it at all.
    """
    if not active_properties or not decoy_properties:
        raise ValueError("both groups must be non-empty")

    names = set(active_properties[0]) & set(decoy_properties[0])
    gaps: dict[str, float] = {}
    for name in names:
        active_mean = float(np.mean([float(p[name]) for p in active_properties]))
        decoy_mean = float(np.mean([float(p[name]) for p in decoy_properties]))
        gaps[name] = active_mean - decoy_mean
    return gaps


def describe_property_gap(gaps: dict[str, float]) -> str:
    """Render a property gap, flagging any large enough to explain enrichment."""
    lines = ["Property gap (active mean - decoy mean):"]
    concerning: list[str] = []
    for name, gap in sorted(gaps.items()):
        tolerance = DEFAULT_TOLERANCES.get(name)
        flag = ""
        if tolerance is not None and abs(gap) > tolerance:
            flag = "  <- large enough to bias a docking score on its own"
            concerning.append(name)
        lines.append(f"  {name}: {gap:+.2f}{flag}")
    if concerning:
        lines.append(
            "  WARNING: the groups differ on "
            + ", ".join(concerning)
            + ". Any enrichment measured here may be the scoring function's "
            "property bias rather than recognition of the binding site."
        )
    else:
        lines.append(
            "  The groups are matched on every property, so enrichment cannot be "
            "attributed to size or lipophilicity."
        )
    return "\n".join(lines)
