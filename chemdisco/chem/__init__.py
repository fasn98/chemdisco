"""Structure handling. The RDKit edge of the package.

Everything in here needs a chemistry toolkit. Everything that consumes it --
curation, splitting, QSAR evaluation -- does not, which is what keeps the
scientific logic testable without one.
"""

from .standardize import (
    RDKIT_AVAILABLE,
    StandardizedStructure,
    deduplicate_by_structure,
    inchikey_of,
    molecular_weight,
    require_rdkit,
    silence_rdkit_logs,
    skeleton_inchikey,
    standardize,
    standardize_many,
    validate_smiles,
)

__all__ = [
    "RDKIT_AVAILABLE",
    "StandardizedStructure",
    "deduplicate_by_structure",
    "inchikey_of",
    "molecular_weight",
    "require_rdkit",
    "silence_rdkit_logs",
    "skeleton_inchikey",
    "standardize",
    "standardize_many",
    "validate_smiles",
]
