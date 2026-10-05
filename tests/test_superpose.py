"""Superposition, and the residue correspondence it depends on.

Cross-docking compares a pose docked into one crystal against a ligand observed
in another, so the two frames have to be reconciled first. That reconciliation
is the step that fails silently: get it wrong and the RMSD measures the distance
between two unit-cell origins while still looking like a pose error.

These tests cover the geometry (exact, synthetic) and the correspondence (against
the real structures, because the failure was real).
"""

from __future__ import annotations

import math
import pathlib
import unittest

from chemdisco.dock.superpose import (
    MIN_IDENTITY_AGREEMENT,
    alignment_is_trustworthy,
    find_residue_correspondence,
    match_alpha_carbons,
    superpose,
)

CACHE = pathlib.Path(__file__).resolve().parent.parent / ".cache" / "pdb"

CUBE = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 1.0),
]


def rotate_z(points, radians):
    cos, sin = math.cos(radians), math.sin(radians)
    return [(x * cos - y * sin, x * sin + y * cos, z) for x, y, z in points]


class TestGeometry(unittest.TestCase):
    def test_a_structure_superposes_onto_itself_exactly(self) -> None:
        self.assertAlmostEqual(superpose(CUBE, CUBE).residual_rmsd, 0.0, places=12)

    def test_a_pure_translation_is_absorbed_exactly(self) -> None:
        moved = [(x + 17.0, y - 4.0, z + 101.0) for x, y, z in CUBE]
        result = superpose(moved, CUBE)
        self.assertAlmostEqual(result.residual_rmsd, 0.0, places=10)

    def test_a_pure_rotation_is_absorbed_exactly(self) -> None:
        self.assertAlmostEqual(
            superpose(rotate_z(CUBE, 0.7), CUBE).residual_rmsd, 0.0, places=10
        )

    def test_applying_the_transform_recovers_the_target(self) -> None:
        moved = [(x + 5.0, y, z - 3.0) for x, y, z in rotate_z(CUBE, 1.1)]
        result = superpose(moved, CUBE)
        for got, want in zip(result.apply(moved), CUBE, strict=True):
            for a, b in zip(got, want, strict=True):
                self.assertAlmostEqual(a, b, places=9)

    def test_a_reflection_is_not_fitted_as_a_rotation(self) -> None:
        # An improper rotation superposes a structure onto its mirror image,
        # which fits beautifully and is geometrically wrong. The determinant
        # correction must prevent it, so a mirrored set must NOT reach zero.
        mirrored = [(-x, y, z) for x, y, z in CUBE]
        self.assertGreater(superpose(mirrored, CUBE).residual_rmsd, 0.1)

    def test_mismatched_lengths_are_refused(self) -> None:
        with self.assertRaises(ValueError) as context:
            superpose(CUBE, CUBE[:-1])
        self.assertIn("correspondence", str(context.exception))

    def test_too_few_points_to_determine_a_rotation_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            superpose(CUBE[:2], CUBE[:2])


class TestTrustworthiness(unittest.TestCase):
    """A transform whose own error approaches 2 A cannot judge a 2 A criterion."""

    def _fake(self, residual: float, matched: int):
        from chemdisco.dock.superpose import Superposition

        return Superposition(
            rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            mobile_centroid=(0.0, 0.0, 0.0),
            target_centroid=(0.0, 0.0, 0.0),
            residual_rmsd=residual,
            n_matched=matched,
        )

    def test_a_large_residual_is_refused(self) -> None:
        self.assertFalse(alignment_is_trustworthy(self._fake(23.9, 326)))

    def test_too_few_matched_atoms_is_refused(self) -> None:
        self.assertFalse(alignment_is_trustworthy(self._fake(0.1, 8)))

    def test_a_good_alignment_is_accepted(self) -> None:
        self.assertTrue(alignment_is_trustworthy(self._fake(0.42, 386)))


@unittest.skipUnless(
    (CACHE / "4FRS.pdb").exists() and (CACHE / "7MYI.pdb").exists(),
    "needs cached 4FRS and 7MYI; run scripts/select_crossdock_set.py first",
)
class TestResidueCorrespondenceOnRealStructures(unittest.TestCase):
    """The bug this module exists because of.

    BACE1 is deposited under two numbering conventions: 4FRS numbers its chain
    58-446, on the pro-enzyme, and 7MYI numbers the same protein -5-385, on the
    mature form. Matching by raw residue number pairs chemically unrelated
    positions, and Kabsch fits it and returns a 24 A residual -- a number with
    the right units from a correspondence that was never real.
    """

    def _load(self, pdb_id: str):
        from chemdisco.dock import parse_pdb, strip_to_receptor

        structure = parse_pdb((CACHE / f"{pdb_id}.pdb").read_text())
        return strip_to_receptor(structure), structure.best_ligand()

    def test_the_numbering_offset_is_detected_and_unambiguous(self) -> None:
        mobile, mobile_ligand = self._load("4FRS")
        target, target_ligand = self._load("7MYI")
        correspondence = find_residue_correspondence(
            mobile, target,
            mobile_chain=mobile_ligand.chain, target_chain=target_ligand.chain,
        )
        self.assertEqual(abs(correspondence.offset), 61)
        self.assertAlmostEqual(correspondence.identity_agreement, 1.0, places=6)
        self.assertTrue(correspondence.is_trustworthy)

    def test_matching_by_raw_residue_number_would_be_refused(self) -> None:
        # The regression test. Forcing offset 0 -- which is what matching by
        # residue number does -- must fail the identity check rather than produce
        # points for Kabsch to fit.
        from chemdisco.dock.superpose import ResidueCorrespondence

        mobile, mobile_ligand = self._load("4FRS")
        target, target_ligand = self._load("7MYI")
        naive = ResidueCorrespondence(
            offset=0, identity_agreement=0.07, n_mapped=325, n_agreeing=23
        )
        self.assertFalse(naive.is_trustworthy)
        points_a, points_b, _ = match_alpha_carbons(
            mobile, target,
            mobile_chain=mobile_ligand.chain, target_chain=target_ligand.chain,
            correspondence=naive,
        )
        # Only identity-agreeing residues survive, so a wrong offset yields far
        # too few points to fit -- it cannot quietly produce a 24 A superposition.
        self.assertLess(len(points_a), 50)
        self.assertEqual(len(points_a), len(points_b))

    def test_the_corrected_alignment_superposes_to_well_under_the_criterion(self) -> None:
        mobile, mobile_ligand = self._load("4FRS")
        target, target_ligand = self._load("7MYI")
        points_a, points_b, correspondence = match_alpha_carbons(
            mobile, target,
            mobile_chain=mobile_ligand.chain, target_chain=target_ligand.chain,
        )
        self.assertTrue(correspondence.is_trustworthy, correspondence.describe())
        result = superpose(points_a, points_b)
        self.assertLess(result.residual_rmsd, 1.0, result.describe())
        self.assertTrue(result.is_trustworthy)

    def test_only_identity_agreeing_residues_are_used(self) -> None:
        mobile, mobile_ligand = self._load("4FRS")
        target, target_ligand = self._load("7MYI")
        points_a, _, correspondence = match_alpha_carbons(
            mobile, target,
            mobile_chain=mobile_ligand.chain, target_chain=target_ligand.chain,
        )
        self.assertEqual(len(points_a), correspondence.n_agreeing)
        self.assertGreaterEqual(correspondence.identity_agreement, MIN_IDENTITY_AGREEMENT)

    def test_the_two_binding_sites_coincide_after_alignment(self) -> None:
        # The independent check: aligning on the backbone must also bring the two
        # ligands into the same pocket. If it does not, the alignment reconciled
        # the fold but not the site, and no pose RMSD from it would mean anything.
        mobile, mobile_ligand = self._load("4FRS")
        target, target_ligand = self._load("7MYI")
        points_a, points_b, _ = match_alpha_carbons(
            mobile, target,
            mobile_chain=mobile_ligand.chain, target_chain=target_ligand.chain,
        )
        result = superpose(points_a, points_b)
        moved = result.apply([atom.coordinates for atom in mobile_ligand.heavy_atoms])

        def centroid(points):
            count = len(points)
            return tuple(sum(p[i] for p in points) / count for i in range(3))

        here = centroid(moved)
        there = centroid([atom.coordinates for atom in target_ligand.heavy_atoms])
        gap = math.dist(here, there)
        self.assertLess(gap, 8.0, f"ligand centroids {gap:.2f} A apart after alignment")
