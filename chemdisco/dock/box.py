"""Defining the search box: where in the protein to dock.

The quietest way to get a wrong docking result. Every other parameter announces
itself when it is wrong -- a failed ligand preparation raises, a missing receptor
raises -- but a box in the wrong place produces a complete set of poses and
scores for a site that is not the binding site. The numbers look exactly like
real numbers.

Three ways to define a box, in descending order of reliability:

**From a co-crystallised ligand.** The structure already shows where a molecule
binds. Centre the box on that ligand and size it to its extent plus padding.
This is the only method that carries experimental evidence, and it is what
should be used whenever a holo structure exists.

**From named residues.** When the catalytic or binding residues are known from
the literature -- the two aspartates of an aspartyl protease, say -- their
centroid locates the site. Weaker than a bound ligand, because a residue list
can be wrong or incomplete, but still evidence-based.

**Blind docking over the whole protein.** A box enclosing everything. This is
not a weaker version of site-directed docking; it is a different and much harder
problem, and it is generally unreliable: the search space is enormous, the
scoring function was never calibrated for it, and the top pose frequently lands
in a surface groove that binds nothing. Available here, loudly labelled.

On box size: Vina's search cost grows with the volume, and so does the chance of
the top pose being a spurious surface fit. A box about 8 Å larger than the
ligand in each dimension gives room to reposition without inviting the search to
wander. Boxes beyond roughly 30 Å per side are where blind docking's problems
begin, so that threshold is flagged.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .pdb import Atom, Residue, Structure

#: Padding added to the ligand's extent in each dimension, in angstroms. Eight
#: is the common choice: enough for a somewhat larger analogue to find its pose,
#: not so much that the search drifts to the surface.
DEFAULT_PADDING = 8.0

#: Box side beyond which a search is effectively blind docking, flagged because
#: the scoring function's reliability falls off well before this.
LARGE_BOX_WARNING = 30.0

#: Smallest useful box side. Below this even a small ligand cannot rotate.
MIN_BOX_SIZE = 12.0


@dataclass(frozen=True, slots=True)
class Box:
    """A docking search box: where to look, and how the location was decided.

    ``derived_from`` is not decoration. A box from a co-crystallised ligand and
    a box from a blind sweep carry completely different evidential weight, and a
    score reported without saying which is uninterpretable.
    """

    center: tuple[float, float, float]
    size: tuple[float, float, float]
    derived_from: str
    evidence: str
    warnings: tuple[str, ...] = ()

    @property
    def volume(self) -> float:
        return self.size[0] * self.size[1] * self.size[2]

    @property
    def is_blind(self) -> bool:
        """Whether this box is large enough that the result is a blind search."""
        return max(self.size) > LARGE_BOX_WARNING

    def contains(self, point: Sequence[float]) -> bool:
        return all(
            abs(point[axis] - self.center[axis]) <= self.size[axis] / 2.0
            for axis in range(3)
        )

    def describe(self) -> str:
        lines = [
            f"Box centred at ({self.center[0]:.2f}, {self.center[1]:.2f}, "
            f"{self.center[2]:.2f}), "
            f"size {self.size[0]:.1f} x {self.size[1]:.1f} x {self.size[2]:.1f} A "
            f"({self.volume:.0f} A^3)",
            f"  derived from: {self.derived_from}",
            f"  evidence: {self.evidence}",
        ]
        lines.extend(f"  WARNING: {warning}" for warning in self.warnings)
        return "\n".join(lines)


def centroid(points: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    """Arithmetic mean position of ``points``.

    Raises:
        ValueError: on an empty sequence. There is no centroid of nothing, and
            returning the origin would place a box at (0, 0, 0) -- outside every
            real structure, and silently.
    """
    if not points:
        raise ValueError("cannot take the centroid of an empty point set")
    count = len(points)
    return (
        sum(point[0] for point in points) / count,
        sum(point[1] for point in points) / count,
        sum(point[2] for point in points) / count,
    )


def extent(
    points: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Axis-aligned bounding box of ``points`` as ``(minimum, maximum)``."""
    if not points:
        raise ValueError("cannot take the extent of an empty point set")
    return (
        (
            min(point[0] for point in points),
            min(point[1] for point in points),
            min(point[2] for point in points),
        ),
        (
            max(point[0] for point in points),
            max(point[1] for point in points),
            max(point[2] for point in points),
        ),
    )


def box_from_points(
    points: Sequence[Sequence[float]],
    *,
    padding: float = DEFAULT_PADDING,
    derived_from: str = "point set",
    evidence: str = "",
    minimum_size: float = MIN_BOX_SIZE,
) -> Box:
    """Build a box enclosing ``points`` with ``padding`` added in each dimension.

    The box is centred on the midpoint of the bounding box rather than on the
    centroid of the atoms. For an elongated ligand the two differ by several
    angstroms, and the bounding-box midpoint is what actually keeps the molecule
    centred in the search space.
    """
    low, high = extent(points)
    center = tuple((low[axis] + high[axis]) / 2.0 for axis in range(3))
    size = tuple(
        max(minimum_size, (high[axis] - low[axis]) + 2 * padding) for axis in range(3)
    )

    warnings: list[str] = []
    if max(size) > LARGE_BOX_WARNING:
        warnings.append(
            f"largest side is {max(size):.1f} A. Beyond about "
            f"{LARGE_BOX_WARNING:.0f} A the search becomes effectively blind: the "
            "scoring function was not calibrated for that volume and the top pose "
            "is often a surface groove that binds nothing."
        )
    if min(high[axis] - low[axis] for axis in range(3)) < 2.0:
        warnings.append(
            "the point set is nearly planar in one dimension; check that the "
            "right atoms were selected"
        )

    return Box(
        center=(center[0], center[1], center[2]),
        size=(size[0], size[1], size[2]),
        derived_from=derived_from,
        evidence=evidence,
        warnings=tuple(warnings),
    )


def box_from_ligand(ligand: Residue, *, padding: float = DEFAULT_PADDING) -> Box:
    """The preferred method: a box around a co-crystallised ligand.

    This is the only box definition backed by experimental evidence -- the
    structure shows a molecule bound at this position.
    """
    heavy = ligand.heavy_atoms
    if not heavy:
        raise ValueError(f"ligand {ligand.key} has no heavy atoms")

    box = box_from_points(
        [atom.coordinates for atom in heavy],
        padding=padding,
        derived_from=f"co-crystallised ligand {ligand.key}",
        evidence=(
            f"{ligand.n_heavy_atoms} heavy atoms of {ligand.name} observed bound "
            "in the crystal structure"
        ),
    )

    warnings = list(box.warnings)
    if ligand.n_heavy_atoms < 12:
        warnings.append(
            f"{ligand.name} has only {ligand.n_heavy_atoms} heavy atoms. A box "
            "around a fragment may not cover the pocket a larger candidate needs."
        )
    return Box(
        center=box.center,
        size=box.size,
        derived_from=box.derived_from,
        evidence=box.evidence,
        warnings=tuple(warnings),
    )


def box_from_residues(
    structure: Structure,
    residue_numbers: Sequence[int],
    *,
    chain: str | None = None,
    padding: float = DEFAULT_PADDING,
    evidence: str = "",
) -> Box:
    """A box around named residues, for apo structures with a known site.

    Args:
        structure: Parsed receptor.
        residue_numbers: Sequence positions of the site-defining residues.
        chain: Restrict to one chain. Strongly recommended for multimers, where
            the same residue number appears in each copy and pooling them centres
            the box between two protein molecules -- in solvent.
        padding: Added in each dimension.
        evidence: Why these residues define the site. Should cite something.

    Raises:
        ValueError: if no atoms match, rather than returning an empty-set box.
    """
    wanted = set(residue_numbers)
    atoms = [
        atom
        for atom in structure.protein_atoms
        if atom.residue_seq in wanted and (chain is None or atom.chain == chain)
    ]
    if not atoms:
        raise ValueError(
            f"no atoms found for residues {sorted(wanted)}"
            + (f" in chain {chain}" if chain else "")
            + ". Check the numbering: crystal structures often use the construct's "
            "numbering rather than the UniProt sequence's."
        )

    found_chains = {atom.chain for atom in atoms}
    box = box_from_points(
        [atom.coordinates for atom in atoms],
        padding=padding,
        derived_from=f"residues {sorted(wanted)}"
        + (f" of chain {chain}" if chain else ""),
        evidence=evidence or "residue selection supplied by the caller",
    )

    warnings = list(box.warnings)
    if chain is None and len(found_chains) > 1:
        warnings.append(
            f"the residues matched {len(found_chains)} chains "
            f"({', '.join(sorted(found_chains))}). In a multimer this centres the "
            "box between protein copies, which is in solvent, not in a pocket. "
            "Pass chain= to pick one."
        )
    if not evidence:
        warnings.append(
            "no evidence recorded for this residue selection; a box is only as "
            "good as the reason for its position"
        )
    return Box(
        center=box.center,
        size=box.size,
        derived_from=box.derived_from,
        evidence=box.evidence,
        warnings=tuple(warnings),
    )


def blind_box(structure: Structure, *, padding: float = 4.0) -> Box:
    """A box over the whole protein. A last resort, labelled as one.

    Blind docking is not site-directed docking with a bigger box. The search
    space is orders of magnitude larger, Vina's scoring function was parameterised
    on known binding sites, and published assessments put blind docking's success
    at identifying the true site well below what site-directed docking achieves.
    Use it to generate a hypothesis, never to support a conclusion.
    """
    atoms = structure.protein_atoms
    if not atoms:
        raise ValueError("structure has no protein atoms")

    box = box_from_points(
        [atom.coordinates for atom in atoms],
        padding=padding,
        derived_from="whole protein (blind docking)",
        evidence="no binding site was identified, so the entire structure is searched",
    )
    return Box(
        center=box.center,
        size=box.size,
        derived_from=box.derived_from,
        evidence=box.evidence,
        warnings=tuple(box.warnings)
        + (
            "BLIND DOCKING. The scoring function was calibrated on known binding "
            "sites, not on whole-protein searches, and the top pose often lands in "
            "a surface groove that binds nothing. Treat any score from this box as "
            "a hypothesis about where to look, not as evidence of binding.",
        ),
    )


def rmsd(
    reference: Sequence[Sequence[float]], predicted: Sequence[Sequence[float]]
) -> float:
    """Root-mean-square deviation between two sets of matched coordinates.

    The standard measure of whether a predicted pose reproduces an experimental
    one. Below 2.0 Å is the conventional threshold for "correct", which comes
    from the resolution at which a crystal structure determines atom positions --
    not from any property of the docking program.

    Both sets must be in the same atom order. This computes no superposition and
    no atom matching: for redocking, where both poses sit in the same crystal
    frame and come from the same molecule, superposing them would hide exactly
    the displacement being measured.
    """
    if len(reference) != len(predicted):
        raise ValueError(
            f"{len(reference)} reference atoms against {len(predicted)} predicted; "
            "RMSD needs matched sets in the same order"
        )
    if not reference:
        raise ValueError("cannot compute RMSD over an empty atom set")

    total = sum(
        sum((reference[i][axis] - predicted[i][axis]) ** 2 for axis in range(3))
        for i in range(len(reference))
    )
    return math.sqrt(total / len(reference))


def pose_is_correct(rmsd_value: float) -> bool:
    """Whether a redocked pose reproduces the crystal pose by the usual standard."""
    return rmsd_value < 2.0


def atoms_within(
    atoms: Sequence[Atom], center: Sequence[float], radius: float
) -> list[Atom]:
    """Atoms within ``radius`` of ``center``, for inspecting a pocket's lining."""
    radius_squared = radius * radius
    return [
        atom
        for atom in atoms
        if (atom.x - center[0]) ** 2
        + (atom.y - center[1]) ** 2
        + (atom.z - center[2]) ** 2
        <= radius_squared
    ]
