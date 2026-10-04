"""Bemis-Murcko scaffold perception.

The bridge between RDKit and the toolkit-free splitting logic in
:mod:`chemdisco.split.scaffold`. This module turns structures into scaffold
strings; that module decides how to partition them. Keeping the two apart is what
lets the splitting logic -- where leakage bugs live -- be tested without a
chemistry install.

Two levels of abstraction, and the choice between them materially changes a
split:

**Murcko scaffold.** Ring systems plus the linkers joining them, with side chains
removed but atom types and bond orders kept. A benzimidazole and a benzoxazole
core are different scaffolds.

**Generic (graph) framework.** The same skeleton with every atom made carbon and
every bond single. Benzimidazole and benzoxazole collapse into one framework.

The generic framework produces far fewer, larger groups, so a split built on it is
considerably harder: the test set then contains no compound sharing even the
*shape* of a training compound. That is the right strictness when the goal is
scaffold hopping; the standard Murcko level is the right default and the one
comparable to published benchmarks.

A molecule with no rings has no Murcko scaffold, and RDKit returns an empty
string. That is handled explicitly by the splitter rather than papered over here.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from .standardize import RDKIT_AVAILABLE, require_rdkit

if RDKIT_AVAILABLE:  # pragma: no branch
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold


@lru_cache(maxsize=100_000)
def murcko_scaffold(smiles: str, *, generic: bool = False) -> str:
    """The Bemis-Murcko scaffold of ``smiles`` as canonical SMILES.

    Args:
        smiles: Structure to analyse.
        generic: Return the generic framework, with all atoms carbon and all
            bonds single, instead of the standard Murcko scaffold.

    Returns:
        Canonical SMILES of the scaffold. An empty string for an acyclic molecule
        -- which has no scaffold by definition -- and for a structure that fails
        to parse. Those two cases are distinguished by
        :func:`scaffolds_for_dataset`, which reports parse failures separately so
        they are not quietly pooled with the acyclic compounds.
    """
    require_rdkit()
    text = (smiles or "").strip()
    if not text:
        return ""
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return ""
        if generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        # Scaffold perception can fail on exotic valences and unusual ring
        # systems. An empty string routes the molecule to the acyclic handling,
        # which is conservative: it becomes its own group rather than joining one
        # it may not belong to.
        return ""


def scaffolds_for_dataset(
    smiles_list: Sequence[str], *, generic: bool = False
) -> tuple[list[str], list[int], list[int]]:
    """Scaffolds for a dataset, with acyclic and unparseable entries identified.

    Returns:
        ``(scaffolds, acyclic_indices, unparseable_indices)``. Both index lists
        matter for a curation report: a dataset that is 30% acyclic will split
        very differently from one that is 2% acyclic, and unparseable structures
        should be removed before modelling rather than carried as empty scaffolds.
    """
    require_rdkit()
    scaffolds: list[str] = []
    acyclic: list[int] = []
    unparseable: list[int] = []

    for index, smiles in enumerate(smiles_list):
        text = str(smiles).strip()
        mol = Chem.MolFromSmiles(text) if text else None
        if mol is None:
            unparseable.append(index)
            scaffolds.append("")
            continue
        scaffold = murcko_scaffold(text, generic=generic)
        if not scaffold:
            acyclic.append(index)
        scaffolds.append(scaffold)

    return scaffolds, acyclic, unparseable


def scaffold_summary(smiles_list: Sequence[str], *, generic: bool = False) -> str:
    """A human-readable account of the dataset's scaffold diversity.

    Worth printing before training. A dataset of 2000 compounds over 8 scaffolds
    is one optimisation campaign, and no split of it can measure generalisation to
    new chemotypes; the same 2000 over 400 scaffolds is a genuinely diverse set.
    The ratio decides how much a scaffold-split score is worth.
    """
    scaffolds, acyclic, unparseable = scaffolds_for_dataset(
        smiles_list, generic=generic
    )
    distinct = {s for s in scaffolds if s}
    counts: dict[str, int] = {}
    for scaffold in scaffolds:
        if scaffold:
            counts[scaffold] = counts.get(scaffold, 0) + 1

    n = len(smiles_list)
    lines = [
        f"{n} molecules over {len(distinct)} distinct "
        f"{'generic frameworks' if generic else 'Murcko scaffolds'}",
        f"  {len(acyclic)} acyclic (no scaffold), {len(unparseable)} unparseable",
    ]
    if counts:
        ordered = sorted(counts.items(), key=lambda kv: -kv[1])
        largest_share = ordered[0][1] / n
        lines.append(
            f"  largest scaffold group holds {ordered[0][1]} molecules "
            f"({largest_share:.1%} of the dataset)"
        )
        singletons = sum(1 for count in counts.values() if count == 1)
        lines.append(f"  {singletons} scaffolds appear exactly once")
        if len(distinct) < 10:
            lines.append(
                "  WARNING: fewer than ten scaffolds. This is one or two chemical "
                "series, so a scaffold split cannot measure generalisation to new "
                "chemotypes -- it can only measure it to these few."
            )
        if largest_share > 0.5:
            lines.append(
                "  WARNING: over half the dataset shares one scaffold. Any split "
                "will be dominated by where that single group lands, and the "
                "achieved test fraction will be far from the requested one."
            )
    return "\n".join(lines)
