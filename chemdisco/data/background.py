"""A feature-profile background drawn from structurally unrelated targets.

Why this module exists, stated plainly, because it is the record of a method
error rather than a design decision.

The conserved-feature check in :mod:`chemdisco.generate.pharmacophore` asks which
functional groups the known actives share *and* are rarer elsewhere. The "rarer
elsewhere" half needs a comparison set, and the first attempt took it from the
weakly active end of the target's own curated data. That looked like the
conservative choice -- real measured compounds, same assay, no presumed
inactives -- and it did not work. Amidine came out at 1.9x enrichment, aromatic
halogen at 1.8x, primary amine at 1.9x: three features indistinguishable, so a
polyfluorinated biaryl with no basic nitrogen at all passed a check built
specifically to catch it. Filtering the background to compounds below 0.4
Tanimoto from every active moved amidine prevalence from 50% to 48%.

The reason is structural. **A target's ChEMBL record contains no "does not bind"
population.** Every compound in it was synthesised and tested against that
target, so nearly all of them carry its warhead whatever their potency, and
ECFP4 similarity cannot separate chemotypes that share one small motif. No
filter on that dataset can produce the contrast the check needs.

So the background has to come from elsewhere: compounds measured against targets
that have nothing to do with the one under study. Those are ordinary drug-like
molecules with the same medicinal-chemistry provenance -- real synthesised
compounds from real programmes, not generated structures and not a property
distribution sampled from a catalogue -- and no reason to carry the target's
anchoring group.

Two honest caveats come with that choice:

1. These compounds are **presumed** non-binders, not measured ones. Nobody tested
   them against the target. A few may well bind it. That contamination dilutes
   enrichment rather than inventing it, so the direction of the error is safe,
   but the word "presumed" belongs in the output and :meth:`BackgroundSet.describe`
   puts it there. This is the opposite trade-off from the docking-enrichment
   decoys, which deliberately use *measured* weak binders because there a harder
   test is an honest one; here shared chemistry erases the signal instead.
2. A compound active against an unrelated target can still have been tested
   against this one. Any molecule that appears in the target's own dataset is
   excluded by identifier, and the count of exclusions is reported -- if that
   number is large, the "unrelated" claim deserves a second look.

The target list below spans kinases, aminergic and peptide GPCRs, nuclear
receptors and two unrelated enzymes. It is deliberately not tuned to make any
particular feature look discriminating: aminergic GPCR ligands are full of basic
amines, which makes ``basic_nitrogen_any`` *harder* to call discriminating for an
aspartyl protease, not easier. A background chosen to flatter the check would be
worse than no background.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .chembl import ChEMBLClient, ChEMBLError


@dataclass(frozen=True, slots=True)
class BackgroundTarget:
    """One protein whose ligands serve as ordinary drug-like comparison matter."""

    name: str
    accession: str
    family: str


#: Targets from families unrelated to any protease, chosen for breadth rather
#: than for the answer they produce.
#:
#: Each entry is resolved through UniProt accession rather than name, for the
#: reason :meth:`ChEMBLClient.targets_for_uniprot` documents: a name search pools
#: orthologues and cell-based assays into one set. A target that fails to resolve
#: is skipped and recorded, not silently dropped -- see
#: :attr:`BackgroundSet.failures`.
UNRELATED_TARGETS: tuple[BackgroundTarget, ...] = (
    BackgroundTarget("EGFR", "P00533", "protein kinase"),
    BackgroundTarget("CDK2", "P24941", "protein kinase"),
    BackgroundTarget("MAPK14", "Q16539", "protein kinase"),
    BackgroundTarget("ADRB2", "P07550", "aminergic GPCR"),
    BackgroundTarget("HTR1A", "P08908", "aminergic GPCR"),
    BackgroundTarget("OPRM1", "P35372", "peptide GPCR"),
    BackgroundTarget("ESR1", "P03372", "nuclear receptor"),
    BackgroundTarget("AR", "P10275", "nuclear receptor"),
    BackgroundTarget("CA2", "P00918", "carbonic anhydrase"),
    BackgroundTarget("PTGS2", "P35354", "oxidoreductase"),
)


@dataclass(frozen=True, slots=True)
class BackgroundSet:
    """Structures gathered as a comparison population, with their accounting.

    Attributes:
        smiles: The background structures, interleaved across targets so that
            truncating the list keeps the families balanced.
        per_target: How many structures each target contributed.
        families: Which protein families are represented, and by how many
            structures.
        n_excluded_overlap: Compounds dropped because they also appear in the
            target's own dataset. A large number undermines the claim that these
            targets are unrelated.
        n_duplicates: Compounds dropped because another target already supplied
            them. Promiscuous compounds show up repeatedly; counting each once
            keeps a single molecule from skewing prevalence.
        failures: Targets that could not be retrieved, and why.
    """

    smiles: tuple[str, ...] = ()
    per_target: Mapping[str, int] = field(default_factory=dict)
    families: Mapping[str, int] = field(default_factory=dict)
    n_excluded_overlap: int = 0
    n_duplicates: int = 0
    failures: Mapping[str, str] = field(default_factory=dict)

    #: Below this many structures, a prevalence figure is noise: one compound in
    #: 50 moves a percentage by two points, and the enrichment ratio by more.
    MIN_COMPOUNDS: ClassVar[int] = 200

    #: Below this many families, the background describes one kind of chemistry
    #: rather than drug-like chemistry at large. Three distinct families is the
    #: minimum at which "rarer elsewhere" means something general.
    MIN_FAMILIES: ClassVar[int] = 3

    @property
    def is_usable(self) -> bool:
        """Whether this set is large and broad enough to support a prevalence."""
        return (
            len(self.smiles) >= self.MIN_COMPOUNDS
            and len(self.families) >= self.MIN_FAMILIES
        )

    def describe(self) -> str:
        if not self.smiles:
            lines = ["Background set: empty."]
            if self.failures:
                lines.append(
                    "  every target failed to retrieve: "
                    + "; ".join(
                        f"{name} ({reason})" for name, reason in sorted(
                            self.failures.items()
                        )
                    )
                )
            return "\n".join(lines)

        lines = [
            f"Background set: {len(self.smiles)} compounds from "
            f"{len(self.per_target)} unrelated targets across "
            f"{len(self.families)} protein families."
        ]
        for family, count in sorted(self.families.items(), key=lambda kv: -kv[1]):
            contributors = ", ".join(
                name
                for name, target in _TARGETS_BY_NAME.items()
                if target.family == family and name in self.per_target
            )
            lines.append(f"  {family}: {count} compounds ({contributors})")
        lines.append(
            "  These are PRESUMED non-binders, not measured ones: nobody tested "
            "them against this target. Any that do bind it dilute the enrichment "
            "rather than inflate it, so the error runs in the safe direction, but "
            "the figures below are a presumption and not a measurement."
        )
        if self.n_excluded_overlap:
            lines.append(
                f"  {self.n_excluded_overlap} compound(s) excluded for also "
                "appearing in this target's own dataset. Those are measured "
                "against it, so leaving them in would put actives in the "
                "background and suppress exactly the signal being looked for."
            )
        if self.n_duplicates:
            lines.append(
                f"  {self.n_duplicates} duplicate(s) collapsed: promiscuous "
                "compounds appear against several targets and are counted once."
            )
        if self.failures:
            lines.append(
                "  not retrieved: "
                + "; ".join(
                    f"{name} ({reason})"
                    for name, reason in sorted(self.failures.items())
                )
            )
        if not self.is_usable:
            lines.append(
                f"  TOO SMALL OR TOO NARROW to use: {len(self.smiles)} compounds "
                f"from {len(self.families)} famil{'y' if len(self.families) == 1 else 'ies'} "
                f"(need {self.MIN_COMPOUNDS} and {self.MIN_FAMILIES}). A "
                "prevalence from fewer is noise, and from one family describes "
                "that family rather than drug-like chemistry."
            )
        return "\n".join(lines)


_TARGETS_BY_NAME: dict[str, BackgroundTarget] = {
    target.name: target for target in UNRELATED_TARGETS
}


def assemble_background(
    contributions: Mapping[str, Sequence[tuple[str, str]]],
    *,
    exclude_molecule_ids: Iterable[str] = (),
    max_total: int = 1200,
    families: Mapping[str, str] | None = None,
    failures: Mapping[str, str] | None = None,
) -> BackgroundSet:
    """Merge per-target compound lists into one balanced background.

    Pure: no network, no chemistry toolkit. The balancing is the part worth
    testing, and it is the part that broke elsewhere in this codebase -- docking
    shards were once filled by striding a sorted list, which left half of them
    with no actives at all. The same mistake here would be quieter: truncating a
    target-ordered list at ``max_total`` could leave a background made entirely of
    kinase inhibitors, and "rarer than in kinase inhibitors" is not the question
    the feature check asks.

    So compounds are taken one target at a time in rotation. A target that runs
    out stops contributing and the rotation continues, which means the balance is
    as even as the available data allows rather than even on paper.

    Args:
        contributions: ``{target_name: [(molecule_id, smiles), ...]}``.
        exclude_molecule_ids: Compounds measured against the target under study.
        max_total: Ceiling on the merged set.
        families: ``{target_name: family}``; defaults to :data:`UNRELATED_TARGETS`.
        failures: Targets that could not be retrieved, carried through for the
            report.
    """
    excluded = {str(identifier) for identifier in exclude_molecule_ids if identifier}
    family_of = dict(families or {
        name: target.family for name, target in _TARGETS_BY_NAME.items()
    })

    queues = {
        name: list(items)
        for name, items in contributions.items()
        if items
    }
    cursors = dict.fromkeys(queues, 0)

    chosen: list[str] = []
    per_target: dict[str, int] = {}
    seen: set[str] = set()
    n_excluded = 0
    n_duplicates = 0

    # Round-robin in a stable order: the result must not depend on dict ordering,
    # because every docking shard rebuilds this and they have to agree.
    order = sorted(queues)
    while len(chosen) < max_total:
        progressed = False
        for name in order:
            if len(chosen) >= max_total:
                break
            queue = queues[name]
            index = cursors[name]
            # Advance past anything unusable before taking one, so an excluded
            # compound does not cost this target its turn.
            while index < len(queue):
                molecule_id, smiles = queue[index]
                index += 1
                if not smiles or not smiles.strip():
                    continue
                key = str(molecule_id) if molecule_id else smiles.strip()
                if key in excluded:
                    n_excluded += 1
                    continue
                if key in seen:
                    n_duplicates += 1
                    continue
                seen.add(key)
                chosen.append(smiles.strip())
                per_target[name] = per_target.get(name, 0) + 1
                progressed = True
                break
            cursors[name] = index
        if not progressed:
            break

    family_counts: dict[str, int] = {}
    for name, count in per_target.items():
        family = family_of.get(name, "unclassified")
        family_counts[family] = family_counts.get(family, 0) + count

    return BackgroundSet(
        smiles=tuple(chosen),
        per_target=per_target,
        families=family_counts,
        n_excluded_overlap=n_excluded,
        n_duplicates=n_duplicates,
        failures=dict(failures or {}),
    )


def fetch_unrelated_background(
    client: ChEMBLClient,
    *,
    exclude_molecule_ids: Iterable[str] = (),
    targets: Sequence[BackgroundTarget] = UNRELATED_TARGETS,
    per_target: int = 200,
    max_total: int = 1200,
    activity_types: Sequence[str] = ("IC50", "Ki"),
) -> BackgroundSet:
    """Retrieve ligands of unrelated targets to serve as a background.

    Only structures are used. Potency is irrelevant here: the question is what
    ordinary medicinal-chemistry compounds look like, and a weak kinase inhibitor
    answers it as well as a potent one.

    A target that cannot be resolved or retrieved is recorded in
    :attr:`BackgroundSet.failures` and the rest proceed. One unavailable protein
    must not cost the whole check, and a silently shorter background would be
    worse than a reported gap.
    """
    contributions: dict[str, list[tuple[str, str]]] = {}
    failures: dict[str, str] = {}
    families = {target.name: target.family for target in targets}

    for target in targets:
        try:
            target_id = client.single_protein_target(target.accession)
            activities = client.activities_for_target(
                target_id,
                activity_types=tuple(activity_types),
                # Several activities per molecule is normal, so ask for more rows
                # than the number of distinct compounds wanted.
                max_records=per_target * 4,
            )
        except ChEMBLError as error:
            failures[target.name] = str(error).split(".")[0][:120]
            continue

        unique: dict[str, str] = {}
        for record in activities:
            molecule_id = record.get("molecule_chembl_id")
            smiles = record.get("canonical_smiles")
            if not molecule_id or not smiles:
                continue
            unique.setdefault(str(molecule_id), str(smiles))
            if len(unique) >= per_target:
                break
        if unique:
            contributions[target.name] = list(unique.items())
        else:
            failures[target.name] = "no activities with structures returned"

    return assemble_background(
        contributions,
        exclude_molecule_ids=exclude_molecule_ids,
        max_total=max_total,
        families=families,
        failures=failures,
    )


def summarise(background: BackgroundSet) -> dict[str, Any]:
    """A JSON-serialisable record of where the background came from."""
    return {
        "n_compounds": len(background.smiles),
        "per_target": dict(background.per_target),
        "families": dict(background.families),
        "n_excluded_overlap": background.n_excluded_overlap,
        "n_duplicates": background.n_duplicates,
        "failures": dict(background.failures),
        "is_usable": background.is_usable,
        "source": "unrelated ChEMBL targets (presumed non-binders)",
    }
