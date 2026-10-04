"""Molecular descriptors and fingerprints.

Two representations, for different jobs:

**ECFP (Morgan) fingerprints.** The workhorse for similarity and for QSAR on
congeneric series. A circular fingerprint hashing each atom's environment out to
a given radius into a bit vector. Radius 2 (diameter 4, hence "ECFP4") at 2048
bits is the standard choice and the one most comparable to published work.
Fingerprints carry no physical interpretation, so a fingerprint model cannot be
read as a structure-activity story -- but they consistently outperform descriptor
sets for potency prediction within a series.

**Physicochemical descriptors.** Interpretable, few, and comparable across
chemotypes. Useful when the question is why a model behaves as it does, and
necessary for the Lipinski-style filters and the applicability-domain leverage
calculation, which needs a low-dimensional continuous space to be meaningful.

A note on failure: every function here returns an explicit absence when a
descriptor cannot be computed. The predecessor project substituted typical values
-- ``props.get('logp', 2.0)`` -- so a molecule whose descriptors failed entered
the model dressed as an average drug. :func:`descriptor_matrix` therefore returns
a validity mask alongside the matrix and leaves dropping to the caller, which
records the loss.

Uses the modern ``rdFingerprintGenerator`` API. The older
``GetMorganFingerprintAsBitVect`` is deprecated in current RDKit and emits
warnings that bury real output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..provenance import Quantity
from .standardize import RDKIT_AVAILABLE, require_rdkit

if RDKIT_AVAILABLE:  # pragma: no branch
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors

#: Default Morgan radius. 2 gives ECFP4 in the usual naming.
DEFAULT_RADIUS = 2

#: Default fingerprint length. 2048 is the common choice; 1024 collides more,
#: 4096 is sparser and slower with little gain at these dataset sizes.
DEFAULT_N_BITS = 2048

#: Interpretable descriptors, each chosen for a specific reason rather than by
#: taking everything RDKit offers. A model on 200 auto-generated descriptors with
#: 300 compounds overfits, and the importances become unreadable.
PHYSCHEM_DESCRIPTORS: tuple[str, ...] = (
    "molecular_weight",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "aromatic_rings",
    "rings",
    "heavy_atoms",
    "fraction_csp3",
    "formal_charge",
    "heteroatoms",
    "stereocentres",
    "molar_refractivity",
)


@dataclass(frozen=True, slots=True)
class DescriptorResult:
    """Descriptors for one molecule, or a recorded failure."""

    smiles: str
    values: dict[str, float] | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.values is not None

    def as_quantities(self) -> dict[str, Quantity]:
        """Descriptors as provenance-carrying quantities for display or export.

        Each is ``DERIVED``: computed deterministically from the structure by a
        published algorithm. Note that LogP and molar refractivity are Crippen
        *estimates* from group contributions rather than measurements, which the
        source string records.
        """
        if self.values is None:
            return {
                name: Quantity.unknown(None, self.error or "descriptor calculation failed")
                for name in PHYSCHEM_DESCRIPTORS
            }
        units = {"molecular_weight": "Da", "tpsa": "A^2"}
        sources = {
            "logp": "RDKit Crippen cLogP (group-contribution estimate, not measured)",
            "molar_refractivity": "RDKit Crippen MR (group-contribution estimate)",
        }
        return {
            name: Quantity.derived(
                value,
                units.get(name),
                sources.get(name, f"RDKit {name} of {self.smiles}"),
            )
            for name, value in self.values.items()
        }


def _safe(func, mol, name: str) -> float:
    """Call an RDKit descriptor, raising with the descriptor's name on failure.

    Raising rather than returning a default is the point: the caller records the
    molecule as failed instead of fabricating a value for it.
    """
    value = func(mol)
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} computed as non-finite")
    return numeric


def compute_descriptors(smiles: str) -> DescriptorResult:
    """Compute the interpretable descriptor set for one structure."""
    require_rdkit()
    text = (smiles or "").strip()
    mol = Chem.MolFromSmiles(text) if text else None
    if mol is None:
        return DescriptorResult(smiles=text, values=None, error="structure failed to parse")

    try:
        values = {
            "molecular_weight": _safe(Descriptors.MolWt, mol, "MolWt"),
            "logp": _safe(Crippen.MolLogP, mol, "MolLogP"),
            "tpsa": _safe(rdMolDescriptors.CalcTPSA, mol, "TPSA"),
            "hbd": _safe(rdMolDescriptors.CalcNumHBD, mol, "NumHBD"),
            "hba": _safe(rdMolDescriptors.CalcNumHBA, mol, "NumHBA"),
            "rotatable_bonds": _safe(
                rdMolDescriptors.CalcNumRotatableBonds, mol, "NumRotatableBonds"
            ),
            "aromatic_rings": _safe(
                rdMolDescriptors.CalcNumAromaticRings, mol, "NumAromaticRings"
            ),
            "rings": _safe(rdMolDescriptors.CalcNumRings, mol, "NumRings"),
            "heavy_atoms": float(mol.GetNumHeavyAtoms()),
            "fraction_csp3": _safe(
                rdMolDescriptors.CalcFractionCSP3, mol, "FractionCSP3"
            ),
            "formal_charge": float(Chem.GetFormalCharge(mol)),
            "heteroatoms": _safe(
                rdMolDescriptors.CalcNumHeteroatoms, mol, "NumHeteroatoms"
            ),
            "stereocentres": float(
                len(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False))
            ),
            "molar_refractivity": _safe(Crippen.MolMR, mol, "MolMR"),
        }
    except Exception as error:
        return DescriptorResult(
            smiles=text,
            values=None,
            error=f"{type(error).__name__}: {error}",
        )

    return DescriptorResult(smiles=text, values=values)


def descriptor_matrix(
    smiles_list: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, list[str]]:
    """Build a descriptor matrix, reporting which molecules failed.

    Returns:
        ``(matrix, feature_names, valid_mask, errors)``. ``matrix`` has one row
        per input, with failed rows left as zeros -- they must be removed using
        ``valid_mask`` before training, which is why the mask is returned rather
        than the rows being silently dropped. Dropping here would misalign the
        matrix against the caller's labels and identifiers.
    """
    require_rdkit()
    results = [compute_descriptors(smiles) for smiles in smiles_list]
    matrix = np.zeros((len(results), len(PHYSCHEM_DESCRIPTORS)), dtype=float)
    valid = np.zeros(len(results), dtype=bool)
    errors: list[str] = []

    for index, result in enumerate(results):
        if result.ok and result.values is not None:
            matrix[index] = [result.values[name] for name in PHYSCHEM_DESCRIPTORS]
            valid[index] = True
        else:
            errors.append(f"row {index} ({result.smiles[:60]}): {result.error}")

    return matrix, PHYSCHEM_DESCRIPTORS, valid, errors


def _generator(radius: int, n_bits: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def fingerprint(
    smiles: str, *, radius: int = DEFAULT_RADIUS, n_bits: int = DEFAULT_N_BITS
):
    """One Morgan fingerprint as an RDKit bit vector, or ``None`` on failure.

    The bit-vector form is kept rather than converted to numpy because RDKit's
    native Tanimoto implementation operates on it and is far faster than a numpy
    equivalent for the pairwise comparisons similarity searching needs.
    """
    require_rdkit()
    mol = Chem.MolFromSmiles((smiles or "").strip())
    if mol is None:
        return None
    return _generator(radius, n_bits).GetFingerprint(mol)


def fingerprint_matrix(
    smiles_list: Sequence[str],
    *,
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_N_BITS,
) -> tuple[np.ndarray, np.ndarray]:
    """Fingerprints as a dense binary matrix, with a validity mask.

    Dense rather than sparse: at 2048 bits and the dataset sizes here the memory
    cost is trivial, and scikit-learn estimators handle dense input faster.

    Returns:
        ``(matrix, valid_mask)`` of shapes ``(n, n_bits)`` and ``(n,)``.
    """
    require_rdkit()
    generator = _generator(radius, n_bits)
    matrix = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    valid = np.zeros(len(smiles_list), dtype=bool)

    for index, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            continue
        bit_vector = generator.GetFingerprint(mol)
        matrix[index] = np.frombuffer(
            bytes(bit_vector.ToBitString(), "ascii"), dtype=np.uint8
        ) - ord("0")
        valid[index] = True

    return matrix, valid


def combined_features(
    smiles_list: Sequence[str],
    *,
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_N_BITS,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, list[str]]:
    """Fingerprint bits concatenated with physicochemical descriptors.

    The combination is a reasonable default for QSAR: fingerprints supply the
    local structural detail that drives potency within a series, descriptors
    supply the global properties that transfer across series.

    One caveat worth stating: the two blocks are on wildly different scales --
    binary bits against molecular weights in the hundreds. Tree ensembles are
    scale-invariant so this is harmless for them, but a linear or distance-based
    model on this matrix will be dominated entirely by the descriptor columns and
    must have them standardised first.

    Returns:
        ``(matrix, feature_names, valid_mask, errors)``. A row is valid only when
        both representations succeeded.
    """
    fingerprints, fingerprint_valid = fingerprint_matrix(
        smiles_list, radius=radius, n_bits=n_bits
    )
    descriptors, descriptor_names, descriptor_valid, errors = descriptor_matrix(
        smiles_list
    )
    matrix = np.hstack([fingerprints.astype(float), descriptors])
    names = tuple(f"ecfp_{i}" for i in range(fingerprints.shape[1])) + tuple(
        descriptor_names
    )
    return matrix, names, fingerprint_valid & descriptor_valid, errors
