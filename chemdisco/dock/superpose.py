"""Superposing two structures of the same protein, and proving it worked.

Cross-docking needs a predicted pose and a crystallographic pose compared in one
frame of reference. They start in two: a pose docked into receptor Y is in Y's
crystal frame, and the answer it should be compared against is ligand X observed
in X's crystal frame. Two crystals of the same protein sit wherever their
unit cells put them, so the two frames differ by an arbitrary rotation and
translation.

**This is the step that fails silently.** If the frames are not reconciled, the
RMSD is dominated by the distance between two unit-cell origins -- tens of
angstroms, constant across every pose -- and it still comes out as a number with
the right units. A cross-docking matrix computed that way would show uniform
failure and look like a finding about docking.

So :func:`superpose` returns the residual RMSD of the atoms it matched, and
:func:`alignment_is_trustworthy` turns that into a verdict. Nothing downstream
should use a transform whose own residual was not checked: two structures of one
protein should superpose to well under 2 A on backbone alpha carbons, and a
residual near the size of the thing being measured means the alignment, not the
docking, produced the number.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: Backbone alpha carbons superpose to a few tenths of an angstrom between two
#: structures of one protein. A residual above this means the correspondence is
#: wrong -- mismatched residues, or two different proteins -- not that the
#: crystals disagree.
MAX_TRUSTWORTHY_RESIDUAL_A = 2.0

#: Fewer matched residues than this and the transform is being fitted to too
#: little of the structure to be relied on.
MIN_MATCHED_RESIDUES = 50


@dataclass(frozen=True, slots=True)
class Superposition:
    """A rigid transform from one crystal frame to another, with its residual.

    Attributes:
        rotation: 3x3 rotation, row-major.
        mobile_centroid: Centroid of the matched mobile atoms, subtracted before
            rotating.
        target_centroid: Centroid of the matched target atoms, added after.
        residual_rmsd: RMSD of the matched atoms after superposition. The
            evidence that the correspondence was real.
        n_matched: How many atom pairs the fit used.
    """

    rotation: tuple[tuple[float, float, float], ...]
    mobile_centroid: tuple[float, float, float]
    target_centroid: tuple[float, float, float]
    residual_rmsd: float
    n_matched: int

    def apply(
        self, points: Sequence[Sequence[float]]
    ) -> list[tuple[float, float, float]]:
        """Carry points from the mobile frame into the target frame."""
        out: list[tuple[float, float, float]] = []
        for point in points:
            shifted = (
                point[0] - self.mobile_centroid[0],
                point[1] - self.mobile_centroid[1],
                point[2] - self.mobile_centroid[2],
            )
            rotated = tuple(
                sum(self.rotation[row][column] * shifted[column] for column in range(3))
                for row in range(3)
            )
            out.append(
                (
                    rotated[0] + self.target_centroid[0],
                    rotated[1] + self.target_centroid[1],
                    rotated[2] + self.target_centroid[2],
                )
            )
        return out

    def describe(self) -> str:
        verdict = "TRUSTWORTHY" if self.is_trustworthy else "NOT TRUSTWORTHY"
        return (
            f"superposition on {self.n_matched} atom pairs, residual "
            f"{self.residual_rmsd:.3f} A -- {verdict}"
        )

    @property
    def is_trustworthy(self) -> bool:
        return alignment_is_trustworthy(self)


def alignment_is_trustworthy(superposition: Superposition) -> bool:
    """Whether a transform may be used to compare poses.

    Two conditions, both necessary. The residual must be small -- a transform
    whose own error approaches the 2 A docking criterion cannot be used to decide
    whether a pose is within 2 A. And it must have been fitted to enough of the
    structure that it describes the protein rather than a handful of residues.
    """
    return (
        superposition.residual_rmsd < MAX_TRUSTWORTHY_RESIDUAL_A
        and superposition.n_matched >= MIN_MATCHED_RESIDUES
    )


def superpose(
    mobile: Sequence[Sequence[float]], target: Sequence[Sequence[float]]
) -> Superposition:
    """Kabsch superposition of ``mobile`` onto ``target``.

    The two sequences must be in correspondence: ``mobile[i]`` and ``target[i]``
    are the same atom in two structures. Establishing that correspondence is the
    caller's job and is where the real risk lies -- Kabsch will happily fit any
    two equal-length point sets and report a large residual, which is exactly why
    the residual is returned rather than discarded.

    Raises:
        ValueError: on mismatched lengths or fewer than three points, where a
            rotation is not determined.
    """
    import numpy as np

    if len(mobile) != len(target):
        raise ValueError(
            f"superposition needs matched point sets, got {len(mobile)} and "
            f"{len(target)}. These are supposed to be the same atoms in two "
            "structures; a length mismatch means the correspondence is broken."
        )
    if len(mobile) < 3:
        raise ValueError(
            f"a rotation is not determined by {len(mobile)} point(s); need at least 3"
        )

    first = np.asarray(mobile, dtype=float)
    second = np.asarray(target, dtype=float)
    first_centroid = first.mean(axis=0)
    second_centroid = second.mean(axis=0)
    centred_first = first - first_centroid
    centred_second = second - second_centroid

    covariance = centred_first.T @ centred_second
    u, _, vt = np.linalg.svd(covariance)
    # Guard against a reflection: an improper rotation would superpose a
    # structure onto its mirror image, which fits beautifully and is wrong.
    direction = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, direction])
    rotation = vt.T @ correction @ u.T

    rotated = (rotation @ centred_first.T).T
    residual = float(np.sqrt(((rotated - centred_second) ** 2).sum(axis=1).mean()))

    return Superposition(
        rotation=tuple(tuple(float(value) for value in row) for row in rotation),
        mobile_centroid=tuple(float(value) for value in first_centroid),
        target_centroid=tuple(float(value) for value in second_centroid),
        residual_rmsd=residual,
        n_matched=len(mobile),
    )


#: A detected numbering offset must reproduce this fraction of residue
#: identities before the correspondence is believed. Two deposits of one protein
#: agree essentially perfectly once the offset is right; the measured figure on
#: BACE1 is 1.000 across 386 residues, against 0.11 for the next-best offset.
MIN_IDENTITY_AGREEMENT = 0.9


@dataclass(frozen=True, slots=True)
class ResidueCorrespondence:
    """A verified residue-number mapping between two structures of one protein.

    Attributes:
        offset: Add this to a mobile residue number to get the target's.
        identity_agreement: Fraction of mapped residues whose names agree. The
            evidence that the mapping is real rather than arithmetic.
        n_mapped: Residues the offset maps onto the other structure.
        n_agreeing: Of those, how many have the same residue name. Only these
            are used for the fit.
    """

    offset: int
    identity_agreement: float
    n_mapped: int
    n_agreeing: int

    @property
    def is_trustworthy(self) -> bool:
        return (
            self.identity_agreement >= MIN_IDENTITY_AGREEMENT
            and self.n_agreeing >= MIN_MATCHED_RESIDUES
        )

    def describe(self) -> str:
        verdict = "TRUSTWORTHY" if self.is_trustworthy else "NOT TRUSTWORTHY"
        return (
            f"numbering offset {self.offset:+d}, {self.n_agreeing}/{self.n_mapped} "
            f"residue identities agree ({self.identity_agreement:.3f}) -- {verdict}"
        )


def _alpha_carbons(
    atoms: Sequence[object], chain: str | None
) -> dict[int, tuple[str, tuple[float, float, float]]]:
    """``{residue_number: (residue_name, coordinates)}`` for alpha carbons."""
    found: dict[int, tuple[str, tuple[float, float, float]]] = {}
    for atom in atoms:
        if getattr(atom, "name", "").strip() != "CA":
            continue
        if chain is not None and getattr(atom, "chain", None) != chain:
            continue
        sequence = getattr(atom, "residue_seq", None)
        if sequence is None or sequence in found:
            continue
        found[sequence] = (
            getattr(atom, "residue_name", ""),
            atom.coordinates,
        )
    return found


def find_residue_correspondence(
    mobile_atoms: Sequence[object],
    target_atoms: Sequence[object],
    *,
    mobile_chain: str | None = None,
    target_chain: str | None = None,
    max_offset: int = 300,
) -> ResidueCorrespondence:
    """Find the numbering offset between two structures of the same protein.

    **Residue numbers are not comparable across PDB entries**, and assuming they
    are is a silent failure with a plausible-looking result. BACE1 is deposited
    under at least two conventions: 4FRS numbers its chain 58-446, on the
    pro-enzyme, while 7MYI numbers the same protein -5-385, on the mature form.
    Matching by raw number pairs residue 58 with residue 58 -- chemically
    unrelated positions -- and the identities then agree at 7%, which is chance.
    Kabsch fits that happily and returns a 24 A residual: a number with the right
    units, produced by a correspondence that was never real.

    So the offset is measured rather than assumed: every candidate offset is
    scored by how many residue *identities* it reconciles, and the winner has to
    clear :data:`MIN_IDENTITY_AGREEMENT`. On BACE1 the correct offset reconciles
    386 of 386 identities while the runner-up manages 0.11, so the decision is
    not close -- which is what makes it safe to automate.
    """
    mobile = _alpha_carbons(mobile_atoms, mobile_chain)
    target = _alpha_carbons(target_atoms, target_chain)

    best = ResidueCorrespondence(
        offset=0, identity_agreement=0.0, n_mapped=0, n_agreeing=0
    )
    for offset in range(-max_offset, max_offset + 1):
        mapped = [key for key in mobile if key + offset in target]
        if len(mapped) < MIN_MATCHED_RESIDUES:
            continue
        agreeing = sum(
            1 for key in mapped if mobile[key][0] == target[key + offset][0]
        )
        agreement = agreeing / len(mapped)
        if agreement > best.identity_agreement:
            best = ResidueCorrespondence(
                offset=offset,
                identity_agreement=agreement,
                n_mapped=len(mapped),
                n_agreeing=agreeing,
            )
    return best


def match_alpha_carbons(
    mobile_atoms: Sequence[object],
    target_atoms: Sequence[object],
    *,
    mobile_chain: str | None = None,
    target_chain: str | None = None,
    correspondence: ResidueCorrespondence | None = None,
) -> tuple[
    list[tuple[float, float, float]],
    list[tuple[float, float, float]],
    ResidueCorrespondence,
]:
    """Pair alpha carbons between two structures, via a verified correspondence.

    Only residues whose *identities* agree under the detected offset contribute,
    so a point mutation or an engineered residue drops out instead of dragging
    the fit.

    Returns ``(mobile_points, target_points, correspondence)``. Check the
    correspondence before using the points: if it is not trustworthy, the points
    are not either.
    """
    if correspondence is None:
        correspondence = find_residue_correspondence(
            mobile_atoms,
            target_atoms,
            mobile_chain=mobile_chain,
            target_chain=target_chain,
        )
    mobile = _alpha_carbons(mobile_atoms, mobile_chain)
    target = _alpha_carbons(target_atoms, target_chain)

    mobile_points: list[tuple[float, float, float]] = []
    target_points: list[tuple[float, float, float]] = []
    for key in sorted(mobile):
        mapped = key + correspondence.offset
        if mapped not in target:
            continue
        if mobile[key][0] != target[mapped][0]:
            continue
        mobile_points.append(mobile[key][1])
        target_points.append(target[mapped][1])
    return mobile_points, target_points, correspondence
