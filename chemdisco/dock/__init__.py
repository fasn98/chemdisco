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

from .decoys import (
    DEFAULT_TOLERANCES,
    MAX_DECOY_SIMILARITY,
    DecoySelection,
    balance_selection,
    describe_property_gap,
    property_gap,
    select_decoys,
)
from .enrichment import (
    EnrichmentResult,
    analyse_enrichment,
    auc_roc,
    bedroc,
    enrichment_factor,
    max_enrichment_factor,
)
from .screen import (
    ScreenResult,
    interleave_by_label,
    screen,
    triage_candidates,
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
from .pdb import (
    MAX_PEPTIDE_LIGAND_RESIDUES,
    MIN_LIGAND_HEAVY_ATOMS,
    NON_LIGAND_RESIDUES,
    Atom,
    Chain,
    Residue,
    Structure,
    parse_pdb,
    strip_to_receptor,
    write_pdb,
)

__all__ = [
    "DEFAULT_TOLERANCES",
    "MAX_DECOY_SIMILARITY",
    "MAX_PEPTIDE_LIGAND_RESIDUES",
    "DecoySelection",
    "EnrichmentResult",
    "ScreenResult",
    "analyse_enrichment",
    "auc_roc",
    "balance_selection",
    "bedroc",
    "describe_property_gap",
    "enrichment_factor",
    "interleave_by_label",
    "max_enrichment_factor",
    "property_gap",
    "screen",
    "select_decoys",
    "triage_candidates",
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
    "Chain",
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
