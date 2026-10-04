"""Tests for the docking engine.

Split in two. The reporting logic -- how a score is presented, what caveats
travel with it, when two scores may be ordered -- needs no engine and runs
everywhere. The real docking runs only where Vina is installed, and the test
that matters there is redocking: take a ligand out of a crystal structure, dock
it back, and measure how far it lands from where it actually sits.
"""

from __future__ import annotations

import os
import unittest

from chemdisco.dock import (
    VINA_ERROR_KCAL,
    Box,
    DockingResult,
    Pose,
    RedockValidation,
    nearest_neighbour_rmsd,
    obabel_available,
    scores_are_distinguishable,
    toolchain_report,
    vina_available,
)

requires_vina = unittest.skipUnless(
    vina_available() and obabel_available(),
    "the docking toolchain (vina + openbabel) is not installed here",
)

# Network-dependent checks are gated rather than run by default. A unit test
# that fetches from RCSB fails when a public service is slow, which says nothing
# about the code. The equivalent end-to-end check lives in
# scripts/validate_docking.py, where a network dependency is appropriate.
requires_network = unittest.skipUnless(
    os.environ.get("CHEMDISCO_NETWORK_TESTS") == "1",
    "network tests are off; set CHEMDISCO_NETWORK_TESTS=1 to enable",
)

SAMPLE_BOX = Box(
    center=(10.0, 10.0, 10.0),
    size=(20.0, 20.0, 20.0),
    derived_from="co-crystallised ligand LIG_A_500",
    evidence="9 heavy atoms observed bound in the crystal structure",
)

POSE_PDBQT = """\
ATOM      1  C   UNL     1       1.000   2.000   3.000  1.00  0.00     0.000 C
ATOM      2  N   UNL     1       2.000   2.000   3.000  1.00  0.00     0.000 N
ATOM      3  H   UNL     1       9.000   9.000   9.000  1.00  0.00     0.000 H
"""


class TestScoreReporting(unittest.TestCase):
    """What a reader is told about a docking score."""

    def _result(self, scores=(-9.2, -8.9, -8.1), heavy=30) -> DockingResult:
        return DockingResult(
            smiles="CC(=O)Oc1ccccc1C(=O)O",
            poses=[Pose(rank=i + 1, score=s) for i, s in enumerate(scores)],
            box=SAMPLE_BOX,
            receptor_id="1FKN",
            n_heavy_atoms=heavy,
        )

    def test_score_is_predicted_not_measured(self) -> None:
        quantity = self._result().score_quantity()
        self.assertEqual(quantity.origin.value, "predicted")
        self.assertEqual(quantity.unit, "kcal/mol")

    def test_the_method_error_travels_with_the_number(self) -> None:
        # A reader must not see -9.2 without also seeing the +/- 2.5 that makes
        # it uncomparable to -8.0.
        quantity = self._result().score_quantity()
        self.assertEqual(quantity.uncertainty, VINA_ERROR_KCAL)
        self.assertIn("2.50", quantity.label(digits=2))

    def test_the_affinity_caveat_is_attached(self) -> None:
        notes = " ".join(self._result().score_quantity().notes)
        self.assertIn("not a predicted binding affinity", notes)
        self.assertIn("Kd", notes)

    def test_the_size_bias_caveat_is_attached(self) -> None:
        notes = " ".join(self._result().score_quantity().notes)
        self.assertIn("rewards molecular size", notes)

    def test_a_docking_score_cannot_rank_on_its_own(self) -> None:
        # in_domain is None: Vina has no applicability domain in the QSAR sense,
        # and an unassessed domain keeps a prediction out of a ranked list.
        quantity = self._result().score_quantity()
        self.assertIsNone(quantity.in_domain)
        self.assertFalse(quantity.is_trustworthy_for_ranking)

    def test_failed_docking_yields_an_unknown_not_a_zero(self) -> None:
        result = DockingResult(smiles="CCO", error="ligand preparation failed")
        quantity = result.score_quantity()
        self.assertFalse(quantity.is_known)
        self.assertEqual(quantity.label(), "not computed")

    def test_ligand_efficiency_corrects_for_size(self) -> None:
        # A fragment at -6 over 15 atoms is a better start than 50 atoms at -10.
        fragment = DockingResult(
            smiles="c1ccccc1", poses=[Pose(1, -6.0)], n_heavy_atoms=15
        )
        large = DockingResult(smiles="C" * 50, poses=[Pose(1, -10.0)], n_heavy_atoms=50)
        self.assertLess(
            fragment.ligand_efficiency().require(),
            large.ligand_efficiency().require(),
            "the fragment should have the better (more negative) efficiency",
        )

    def test_ligand_efficiency_without_a_size_is_unknown(self) -> None:
        result = DockingResult(smiles="CCO", poses=[Pose(1, -7.0)], n_heavy_atoms=0)
        self.assertFalse(result.ligand_efficiency().is_known)

    def test_narrow_pose_spread_is_flagged(self) -> None:
        # Poses within a few tenths of a kcal/mol mean the search did not
        # strongly prefer one arrangement.
        result = self._result(scores=(-9.0, -8.9, -8.85))
        self.assertIn(
            "did not strongly prefer",
            " ".join(result.score_quantity().notes),
        )

    def test_score_spread_is_reported(self) -> None:
        self.assertAlmostEqual(self._result().score_spread or 0.0, 1.1, places=6)
        self.assertIsNone(
            DockingResult(smiles="C", poses=[Pose(1, -5.0)]).score_spread
        )

    def test_box_provenance_reaches_the_score(self) -> None:
        notes = " ".join(self._result().score_quantity().notes)
        self.assertIn("co-crystallised", notes)

    def test_blind_box_warning_propagates_into_the_notes(self) -> None:
        blind = Box(
            center=(0, 0, 0),
            size=(40, 40, 40),
            derived_from="whole protein (blind docking)",
            evidence="no site identified",
            warnings=("BLIND DOCKING. Treat any score as a hypothesis.",),
        )
        result = DockingResult(
            smiles="CCO",
            poses=[Pose(1, -7.0)],
            box=blind,
            warnings=list(blind.warnings),
        )
        self.assertIn("BLIND DOCKING", " ".join(result.score_quantity().notes))


class TestScoreComparison(unittest.TestCase):
    def test_scores_within_the_method_error_are_not_distinguishable(self) -> None:
        # The check that stops a ranked shortlist implying a precision the
        # calculation does not have.
        self.assertFalse(scores_are_distinguishable(-9.5, -8.0))
        self.assertFalse(scores_are_distinguishable(-9.5, -9.4))

    def test_a_large_gap_is_distinguishable(self) -> None:
        self.assertTrue(scores_are_distinguishable(-11.0, -5.0))

    def test_comparison_is_symmetric(self) -> None:
        self.assertEqual(
            scores_are_distinguishable(-11.0, -5.0),
            scores_are_distinguishable(-5.0, -11.0),
        )


class TestPoseParsing(unittest.TestCase):
    def test_coordinates_are_read_and_hydrogens_dropped(self) -> None:
        pose = Pose(rank=1, score=-7.0, pdbqt=POSE_PDBQT)
        points = pose.coordinates()
        self.assertEqual(len(points), 2)
        self.assertAlmostEqual(points[0][0], 1.0)

    def test_an_empty_pose_yields_no_coordinates(self) -> None:
        self.assertEqual(Pose(rank=1, score=-7.0).coordinates(), [])


class TestRedockReporting(unittest.TestCase):
    def test_a_passing_redock_reads_clearly(self) -> None:
        validation = RedockValidation("1OL", rmsd=0.8, score=-9.1, passed=True)
        text = validation.describe()
        self.assertIn("PASS", text)
        self.assertIn("0.80", text)

    def test_a_failing_redock_says_what_it_invalidates(self) -> None:
        # The important message: a failed redock means every other score from
        # this setup is suspect, not that this one ligand was awkward.
        validation = RedockValidation("1OL", rmsd=6.4, score=-9.9, passed=False)
        text = validation.describe()
        self.assertIn("FAIL", text)
        self.assertIn("every other score from this setup is suspect", text)

    def test_an_impossible_redock_explains_itself(self) -> None:
        validation = RedockValidation(
            "1OL", rmsd=None, score=None, passed=False, detail="no poses produced"
        )
        self.assertIn("could not be completed", validation.describe())


class TestNearestNeighbourRmsd(unittest.TestCase):
    def test_identical_sets_give_zero(self) -> None:
        points = [(0, 0, 0), (1, 1, 1)]
        self.assertAlmostEqual(nearest_neighbour_rmsd(points, points), 0.0)

    def test_order_does_not_matter(self) -> None:
        # The reason this function exists: a redocked pose's atom order does not
        # match the crystal file's.
        a = [(0, 0, 0), (5, 0, 0)]
        b = [(5, 0, 0), (0, 0, 0)]
        self.assertAlmostEqual(nearest_neighbour_rmsd(a, b), 0.0)

    def test_displacement_is_measured(self) -> None:
        self.assertAlmostEqual(
            nearest_neighbour_rmsd([(0, 0, 0)], [(3, 0, 0)]), 3.0
        )

    def test_empty_sets_raise(self) -> None:
        with self.assertRaises(ValueError):
            nearest_neighbour_rmsd([], [(0, 0, 0)])


class TestToolchainReport(unittest.TestCase):
    def test_absences_name_their_consequence(self) -> None:
        # A missing dependency should say what it costs, not just that it is
        # missing.
        report = toolchain_report()
        self.assertIn("vina", report)
        self.assertIn("meeko", report)
        self.assertIn("obabel", report)


@requires_vina
class TestRealDocking(unittest.TestCase):
    """Runs only where the toolchain is installed; in CI that is always."""

    def test_ligand_preparation_produces_pdbqt(self) -> None:
        from chemdisco.dock import prepare_ligand_pdbqt

        pdbqt, method = prepare_ligand_pdbqt("CC(=O)Oc1ccccc1C(=O)O")
        self.assertIsNotNone(pdbqt, f"preparation failed: {method}")
        assert pdbqt is not None
        self.assertIn("ATOM", pdbqt)
        self.assertIn("ROOT", pdbqt)

    def test_unparseable_smiles_fails_cleanly(self) -> None:
        from chemdisco.dock import prepare_ligand_pdbqt

        pdbqt, method = prepare_ligand_pdbqt("not a molecule")
        self.assertIsNone(pdbqt)
        self.assertIn("parse", method)

    @requires_network
    def test_docking_into_a_prepared_receptor(self) -> None:
        from chemdisco.dock import dock, parse_pdb, prepare_receptor_pdbqt, strip_to_receptor, write_pdb
        from chemdisco.dock.box import box_from_ligand

        import urllib.request

        with urllib.request.urlopen(
            "https://files.rcsb.org/download/1FKN.pdb", timeout=120
        ) as response:
            pdb_text = response.read().decode()

        structure = parse_pdb(pdb_text)
        ligand = structure.best_ligand()
        self.assertIsNotNone(ligand, "1FKN should carry a co-crystallised inhibitor")
        assert ligand is not None

        box = box_from_ligand(ligand)
        receptor_pdb = write_pdb(strip_to_receptor(structure), title="1FKN receptor")
        receptor_pdbqt, method = prepare_receptor_pdbqt(receptor_pdb)
        self.assertIsNotNone(receptor_pdbqt, f"receptor prep failed: {method}")
        assert receptor_pdbqt is not None

        result = dock(
            "CC(=O)Oc1ccccc1C(=O)O",
            receptor_pdbqt,
            box,
            receptor_id="1FKN",
            exhaustiveness=4,
            n_poses=3,
        )
        self.assertTrue(result.ok, f"docking failed: {result.error}")
        assert result.best_score is not None
        # A real score, not a sanity-free number: anything outside this range
        # would mean the setup is broken rather than the ligand weak.
        self.assertLess(result.best_score, 0.0)
        self.assertGreater(result.best_score, -20.0)


if __name__ == "__main__":
    unittest.main()


class TestPoseRankingDetermination(unittest.TestCase):
    """Whether the score actually separates the top pose from the rest.

    Encodes a measured result rather than an assumption. Redocking the 4FRS
    inhibitor produced the pose set below -- nine poses spanning 1.23 kcal/mol,
    entirely inside Vina's own 2.5 kcal/mol error, with the crystallographically
    correct pose ranked third at 1.82 A while a 4.19 A pose ranked first.
    Quadrupling exhaustiveness from 16 to 64 found more correct poses but still
    ranked a 4.22 A pose first, which settles it: the limitation is the scoring
    function, not the search.
    """

    #: The real 4FRS scores, at exhaustiveness 16.
    FOUR_FRS_SCORES = (-8.32, -8.21, -8.01, -7.84, -7.28, -7.25, -7.25, -7.24, -7.09)

    def _result(self, scores) -> DockingResult:
        return DockingResult(
            smiles="[H]/N=C1\\N[C@](C)(c2sc(-c3cncc(C#CC)c3)cc2Cl)CC(=O)N1C",
            poses=[Pose(rank=i + 1, score=s) for i, s in enumerate(scores)],
            receptor_id="4FRS",
            n_heavy_atoms=25,
        )

    def test_the_real_4frs_pose_set_is_not_separated(self) -> None:
        result = self._result(self.FOUR_FRS_SCORES)
        self.assertAlmostEqual(result.score_spread or 0.0, 1.23, places=2)
        self.assertFalse(result.pose_ranking_is_determined)

    def test_that_limitation_reaches_the_reported_quantity(self) -> None:
        notes = " ".join(self._result(self.FOUR_FRS_SCORES).score_quantity().notes)
        self.assertIn("not determined by the score", notes)

    def test_and_the_human_readable_report(self) -> None:
        self.assertIn(
            "not determined by the score",
            self._result(self.FOUR_FRS_SCORES).describe(),
        )

    def test_a_genuinely_separated_pose_set_is_recognised(self) -> None:
        # When one pose really does stand out by more than the method's error,
        # the ranking means something and must not be dismissed.
        result = self._result((-12.0, -7.0, -6.5, -6.0))
        self.assertTrue(result.pose_ranking_is_determined)
        self.assertNotIn("not determined", result.describe())

    def test_a_single_pose_is_never_determined(self) -> None:
        # One pose carries no evidence that it beat anything.
        self.assertFalse(self._result((-9.0,)).pose_ranking_is_determined)
