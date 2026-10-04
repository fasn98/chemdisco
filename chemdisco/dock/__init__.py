"""Molecular docking.

The predecessor project's "docking" was a sum of property bonuses plus
``random.uniform(-0.5, 0.5)``, reported in kcal/mol. It was removed rather than
reimplemented, and this is the replacement: real geometry, a real scoring
function, and a box whose position is justified and recorded.

Layered so the quiet failures are testable. PDB parsing, ligand identification
and box geometry are pure functions -- no engine, no toolkit, no network -- because
those are where a docking run goes wrong without saying so. A failed ligand
preparation raises; a box centred on a crystallisation additive produces a full
set of plausible poses for a site that binds nothing.
"""

from .engine import (
    VINA_ERROR_KCAL,
    DockingError,
    DockingResult,
    Pose,
    RedockValidation,
    dock,
    meeko_available,
    nearest_neighbour_rmsd,
    obabel_available,
    prepare_ligand_pdbqt,
    prepare_receptor_pdbqt,
    redock_validation,
    scores_are_distinguishable,
    toolchain_report,
    vina_available,
)
from .box import (
    Box,
    atoms_within,
    blind_box,
    box_from_ligand,
    box_from_points,
    box_from_residues,
    centroid,
    extent,
    pose_is_correct,
    rmsd,
)
from .pdb import (
    MIN_LIGAND_HEAVY_ATOMS,
    NON_LIGAND_RESIDUES,
    Atom,
    Residue,
    Structure,
    parse_pdb,
    strip_to_receptor,
    write_pdb,
)

__all__ = [
    "MIN_LIGAND_HEAVY_ATOMS",
    "VINA_ERROR_KCAL",
    "DockingError",
    "DockingResult",
    "Pose",
    "RedockValidation",
    "dock",
    "meeko_available",
    "nearest_neighbour_rmsd",
    "obabel_available",
    "prepare_ligand_pdbqt",
    "prepare_receptor_pdbqt",
    "redock_validation",
    "scores_are_distinguishable",
    "toolchain_report",
    "vina_available",
    "NON_LIGAND_RESIDUES",
    "Atom",
    "Box",
    "Residue",
    "Structure",
    "atoms_within",
    "blind_box",
    "box_from_ligand",
    "box_from_points",
    "box_from_residues",
    "centroid",
    "extent",
    "parse_pdb",
    "pose_is_correct",
    "rmsd",
    "strip_to_receptor",
    "write_pdb",
]
