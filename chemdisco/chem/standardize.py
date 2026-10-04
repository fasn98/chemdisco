"""Structure standardisation: making two records of the same molecule identical.

Without this step a dataset silently contains duplicates. ChEMBL lists a free
base and its hydrochloride salt under different molecule ids with different
canonical SMILES; a SMILES-string comparison treats them as two compounds, so
both survive deduplication and can land on opposite sides of a train/test split.
That is leakage of the most direct kind -- the same molecule in both partitions --
and it is invisible unless structures are standardised first.

The pipeline applied here, in order, follows the RDKit ``MolStandardize``
conventions:

1. **Cleanup.** Sanitise, remove explicit hydrogens, disconnect metal-ligand
   bonds, normalise functional-group representations (nitro groups and N-oxides
   are drawn several equivalent ways), reionise acids consistently.
2. **Largest-fragment selection.** Keep the organic parent, discarding counter-
   ions and solvates. This is what collapses salt forms onto one structure.
3. **Uncharging.** Neutralise where chemically reasonable, so a carboxylate and
   its conjugate acid are one compound.
4. **Optional tautomer canonicalisation.** Off by default: it is slow and the
   RDKit implementation can pick a chemically unusual canonical form. It is
   nevertheless the only way to merge records drawn as different tautomers, so it
   is available as a switch.

Stereochemistry is deliberately **preserved**. Enantiomers routinely differ in
potency by orders of magnitude, so merging them would pool genuinely different
compounds -- the opposite of the problem this module solves.

This module requires RDKit, so it is one of the toolkit-edge modules. Everything
it feeds -- curation, splitting, evaluation -- stays toolkit-free and testable
without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

from ..provenance import Quantity

try:  # pragma: no cover - import guard exercised only without RDKit
    from rdkit import Chem, RDLogger
    from rdkit.Chem import inchi
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDKIT_AVAILABLE = True
except ImportError:  # pragma: no cover
    RDKIT_AVAILABLE = False
    Chem = None  # type: ignore[assignment]


def require_rdkit() -> None:
    """Raise a useful error when RDKit is missing.

    The message names the install command, because the usual failure mode is a
    fresh environment rather than a genuine incompatibility.
    """
    if not RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for structure handling. Install it with "
            "'pip install rdkit' (conda-forge also works: "
            "'conda install -c conda-forge rdkit'). The curation, splitting and "
            "evaluation layers of this package do not need it."
        )


def silence_rdkit_logs() -> None:
    """Suppress RDKit's C++ stderr chatter.

    RDKit logs every parse failure directly to stderr, which buries real output
    when sweeping thousands of database records. Parse failures are reported
    through return values here instead, so nothing is lost by silencing them.
    """
    require_rdkit()
    RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True, slots=True)
class StandardizedStructure:
    """One molecule after standardisation, with what happened to it recorded.

    Attributes:
        input_smiles: Exactly what was supplied.
        smiles: Canonical SMILES after standardisation, or ``None`` on failure.
        inchikey: InChIKey of the standardised structure, the identity hash used
            for deduplication and novelty checking.
        parent_changed: Whether a salt, solvate or counterion was stripped.
        charge_changed: Whether neutralisation altered the structure.
        tautomer_changed: Whether tautomer canonicalisation altered it.
        error: Why standardisation failed, when it did.
        notes: What was done, for the audit trail.
    """

    input_smiles: str
    smiles: str | None
    inchikey: str | None
    parent_changed: bool = False
    charge_changed: bool = False
    tautomer_changed: bool = False
    error: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.smiles is not None and self.error is None


def validate_smiles(smiles: str) -> tuple[bool, str | None]:
    """Check whether ``smiles`` parses, returning the reason if not.

    Returns:
        ``(True, None)`` when valid, ``(False, reason)`` otherwise. RDKit's own
        message is forwarded where available, since "unclosed ring" or
        "explicit valence greater than permitted" tells a curator which records
        in the source are malformed.
    """
    require_rdkit()
    text = (smiles or "").strip()
    if not text:
        return False, "empty SMILES"
    try:
        mol = Chem.MolFromSmiles(text, sanitize=True)
    except Exception as error:  # RDKit raises bare RuntimeError subclasses
        return False, f"RDKit rejected the structure: {error}"
    if mol is None:
        # Re-parse without sanitisation to distinguish a syntax error from a
        # chemistry error; the two call for different fixes in the source data.
        unsanitised = Chem.MolFromSmiles(text, sanitize=False)
        if unsanitised is None:
            return False, "SMILES syntax is invalid"
        try:
            Chem.SanitizeMol(unsanitised)
        except Exception as error:
            return False, f"structure parses but fails sanitisation: {error}"
        return False, "structure failed to parse for an unreported reason"
    if mol.GetNumHeavyAtoms() == 0:
        return False, "structure contains no heavy atoms"
    return True, None


def standardize(
    smiles: str, *, canonicalize_tautomer: bool = False
) -> StandardizedStructure:
    """Standardise one structure.

    Args:
        smiles: Input SMILES, as the source database wrote it.
        canonicalize_tautomer: Apply tautomer canonicalisation. Off by default --
            it is roughly an order of magnitude slower than the rest of the
            pipeline and occasionally selects an unusual canonical form. Turn it
            on when merging records across databases, where tautomer differences
            are a real source of duplicate compounds.

    Returns:
        A :class:`StandardizedStructure`. Failures are returned, never raised:
        a database sweep of thousands of records will contain malformed entries,
        and that is ordinary rather than exceptional.
    """
    require_rdkit()
    text = (smiles or "").strip()

    valid, reason = validate_smiles(text)
    if not valid:
        return StandardizedStructure(
            input_smiles=text, smiles=None, inchikey=None, error=reason
        )

    notes: list[str] = []
    try:
        mol = Chem.MolFromSmiles(text)
        original_smiles = Chem.MolToSmiles(mol)

        cleaned = rdMolStandardize.Cleanup(mol)

        parent = rdMolStandardize.FragmentParent(cleaned)
        parent_changed = Chem.MolToSmiles(parent) != Chem.MolToSmiles(cleaned)
        if parent_changed:
            notes.append("stripped counterions or solvate to the organic parent")

        uncharger = rdMolStandardize.Uncharger()
        neutral = uncharger.uncharge(parent)
        charge_changed = Chem.MolToSmiles(neutral) != Chem.MolToSmiles(parent)
        if charge_changed:
            notes.append("neutralised ionisable groups")

        final = neutral
        tautomer_changed = False
        if canonicalize_tautomer:
            enumerator = rdMolStandardize.TautomerEnumerator()
            canonical = enumerator.Canonicalize(neutral)
            tautomer_changed = Chem.MolToSmiles(canonical) != Chem.MolToSmiles(neutral)
            final = canonical
            if tautomer_changed:
                notes.append("canonicalised tautomer")

        if final is None or final.GetNumHeavyAtoms() == 0:
            return StandardizedStructure(
                input_smiles=text,
                smiles=None,
                inchikey=None,
                error="standardisation removed every heavy atom",
            )

        canonical_smiles = Chem.MolToSmiles(final)
        key = inchi.MolToInchiKey(final) or None
        if key is None:
            notes.append("InChIKey generation failed; identity matching will be "
                         "by canonical SMILES, which is less reliable")

        if canonical_smiles == original_smiles and not notes:
            notes.append("already standard")

        return StandardizedStructure(
            input_smiles=text,
            smiles=canonical_smiles,
            inchikey=key,
            parent_changed=parent_changed,
            charge_changed=charge_changed,
            tautomer_changed=tautomer_changed,
            notes=tuple(notes),
        )

    except Exception as error:
        return StandardizedStructure(
            input_smiles=text,
            smiles=None,
            inchikey=None,
            error=f"standardisation raised {type(error).__name__}: {error}",
        )


@lru_cache(maxsize=100_000)
def _cached_standardize(smiles: str, canonicalize_tautomer: bool) -> StandardizedStructure:
    return standardize(smiles, canonicalize_tautomer=canonicalize_tautomer)


def standardize_many(
    smiles_list: Iterable[str], *, canonicalize_tautomer: bool = False
) -> list[StandardizedStructure]:
    """Standardise many structures, caching repeats.

    Database sweeps contain the same structure many times over, so caching is a
    substantial saving. Results are deterministic, which makes caching safe.
    """
    return [
        _cached_standardize(str(smiles).strip(), canonicalize_tautomer)
        for smiles in smiles_list
    ]


def inchikey_of(smiles: str, *, canonicalize_tautomer: bool = False) -> str | None:
    """The identity hash for a structure, or ``None`` if it cannot be computed.

    This is the function to pass as ``compound_key`` to
    :func:`chemdisco.curate.aggregate.aggregate_measurements` for cross-database
    work, so salt forms of one compound aggregate together instead of surviving
    as separate rows.
    """
    result = _cached_standardize((smiles or "").strip(), canonicalize_tautomer)
    return result.inchikey


def skeleton_inchikey(smiles: str) -> str | None:
    """The InChIKey's first block: identity ignoring stereochemistry and charge.

    Useful for a deliberately looser match -- finding whether *some* stereoisomer
    of a candidate is already known, for instance. Not for deduplicating a
    training set, where enantiomers must stay distinct.
    """
    key = inchikey_of(smiles)
    return key.split("-")[0] if key else None


def deduplicate_by_structure(
    smiles_list: Sequence[str], *, canonicalize_tautomer: bool = False
) -> tuple[list[int], dict[str, list[int]]]:
    """Find structurally identical entries.

    Returns:
        ``(representatives, duplicate_groups)`` where ``representatives`` holds
        the first index of each distinct structure and ``duplicate_groups`` maps
        each InChIKey to every index carrying it. The groups are returned rather
        than discarded so a caller can report how many duplicates were present --
        a number worth knowing, since a high duplicate rate usually means a
        database query was too broad.
    """
    groups: dict[str, list[int]] = {}
    unparseable: list[int] = []

    for index, smiles in enumerate(smiles_list):
        result = _cached_standardize(str(smiles).strip(), canonicalize_tautomer)
        if not result.ok or result.inchikey is None:
            unparseable.append(index)
            continue
        groups.setdefault(result.inchikey, []).append(index)

    representatives = sorted(indices[0] for indices in groups.values())
    if unparseable:
        # Unparseable structures are kept as their own representatives rather
        # than dropped here; dropping belongs to curation, which reports it.
        representatives = sorted(representatives + unparseable)
    return representatives, groups


def molecular_weight(smiles: str) -> Quantity:
    """Exact molecular weight as a derived quantity, or an explicit unknown.

    Returned as a :class:`~chemdisco.provenance.Quantity` rather than a float so
    a failed calculation cannot enter a pipeline as a plausible number -- the
    predecessor's ``props.get('molecular_weight', 300)`` pattern.
    """
    require_rdkit()
    from rdkit.Chem import Descriptors

    result = _cached_standardize((smiles or "").strip(), False)
    if not result.ok:
        return Quantity.unknown("Da", f"could not parse structure: {result.error}")
    mol = Chem.MolFromSmiles(result.smiles)
    if mol is None:
        return Quantity.unknown("Da", "standardised SMILES failed to re-parse")
    return Quantity.derived(
        float(Descriptors.MolWt(mol)),
        "Da",
        f"RDKit MolWt of standardised structure {result.smiles}",
    )
