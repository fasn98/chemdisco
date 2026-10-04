"""Tests for the docking logic that needs no docking engine.

PDB parsing, ligand identification and box geometry are where a docking run goes
wrong quietly. A failed ligand preparation raises; a box centred on a glycerol
molecule produces a full set of poses and scores for a site that binds nothing,
and those numbers look exactly like real numbers. So this is the layer that gets
tested hardest, and it is written to be testable without Vina, RDKit or a
network.

The PDB fragments below are real column layouts, not simplified ones. A parser
tested only against tidy input passes until it meets a four-character atom name
or an alternate conformation.
"""

from __future__ import annotations

import unittest

from chemdisco.dock.box import (
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
from chemdisco.dock.pdb import (
    MIN_LIGAND_HEAVY_ATOMS,
    NON_LIGAND_RESIDUES,
    parse_pdb,
    strip_to_receptor,
    write_pdb,
)

# A minimal but column-accurate structure: two protein residues, a water, a
# glycerol (the classic false ligand) and a drug-sized heteroatom group.
SAMPLE_PDB = """\
HEADER    HYDROLASE                               01-JAN-00   1ABC
TITLE     A TEST STRUCTURE FOR PARSER VALIDATION
EXPDTA    X-RAY DIFFRACTION
REMARK   2 RESOLUTION.    1.80 ANGSTROMS.
ATOM      1  N   ASP A  32      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  ASP A  32      11.000  10.000  10.000  1.00 20.00           C
ATOM      3  C   ASP A  32      12.000  10.000  10.000  1.00 20.00           C
ATOM      4  O   ASP A  32      13.000  10.000  10.000  1.00 20.00           O
ATOM      5  N   ASP A 228      10.000  14.000  10.000  1.00 22.00           N
ATOM      6  CA  ASP A 228      11.000  14.000  10.000  1.00 22.00           C
HETATM    7  O   HOH A 401      30.000  30.000  30.000  1.00 35.00           O
HETATM    8  C1  GOL A 402      40.000  40.000  40.000  1.00 40.00           C
HETATM    9  C2  GOL A 402      41.000  40.000  40.000  1.00 40.00           C
HETATM   10  C3  GOL A 402      42.000  40.000  40.000  1.00 40.00           C
HETATM   11  O1  GOL A 402      40.000  41.000  40.000  1.00 40.00           O
HETATM   12  O2  GOL A 402      41.000  41.000  40.000  1.00 40.00           O
HETATM   13  O3  GOL A 402      42.000  41.000  40.000  1.00 40.00           O
HETATM   14  C1  LIG A 500      20.000  20.000  20.000  1.00 25.00           C
HETATM   15  C2  LIG A 500      21.000  20.000  20.000  1.00 25.00           C
HETATM   16  C3  LIG A 500      22.000  20.000  20.000  1.00 25.00           C
HETATM   17  C4  LIG A 500      23.000  20.000  20.000  1.00 25.00           C
HETATM   18  N1  LIG A 500      20.000  21.000  20.000  1.00 25.00           N
HETATM   19  N2  LIG A 500      21.000  21.000  20.000  1.00 25.00           N
HETATM   20  O1  LIG A 500      22.000  21.000  20.000  1.00 25.00           O
HETATM   21  O2  LIG A 500      23.000  21.000  20.000  1.00 25.00           O
HETATM   22  C5  LIG A 500      20.000  20.000  22.000  1.00 25.00           C
HETATM   23  ZN  ZN  A 600      50.000  50.000  50.000  1.00 15.00          ZN
END
"""


def hetatm_line(
    serial: int,
    name: str,
    residue: str,
    chain: str,
    sequence: int,
    x: float,
    y: float,
    z: float,
    element: str,
) -> str:
    """Build a column-correct HETATM record.

    Hand-written format strings in test fixtures drift out of alignment, and a
    misaligned line is skipped as malformed -- so a test can pass because its
    data never reached the parser. That is how the first version of
    test_largest_candidate_wins passed while testing nothing.
    """
    formatted_name = f"{name:<4}" if len(name) >= 4 else f" {name:<3}"
    return (
        f"HETATM{serial:>5} {formatted_name} "
        f"{residue:>3} {chain:>1}{sequence:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
        f"{1.00:>6.2f}{20.00:>6.2f}"
        f"{'':>10}{element:>2}"
    )


class TestParsing(unittest.TestCase):
    def test_header_fields_are_read(self) -> None:
        structure = parse_pdb(SAMPLE_PDB)
        self.assertEqual(structure.pdb_id, "1ABC")
        self.assertIn("PARSER VALIDATION", structure.title)
        self.assertIn("X-RAY", structure.method)
        self.assertAlmostEqual(structure.resolution or 0.0, 1.80)

    def test_protein_and_hetatms_are_separated(self) -> None:
        structure = parse_pdb(SAMPLE_PDB)
        self.assertEqual(len(structure.protein_atoms), 6)
        self.assertEqual(len(structure.hetatms), 17)

    def test_coordinates_are_read_from_fixed_columns(self) -> None:
        structure = parse_pdb(SAMPLE_PDB)
        first = structure.protein_atoms[0]
        self.assertAlmostEqual(first.x, 10.0)
        self.assertEqual(first.name, "N")
        self.assertEqual(first.residue_name, "ASP")
        self.assertEqual(first.residue_seq, 32)

    def test_four_character_atom_names_are_handled(self) -> None:
        # A four-character name starts one column earlier than a shorter one.
        # Splitting on whitespace gets this wrong.
        line = (
            "ATOM      1 HG11 VAL A  10      "
            "1.000   2.000   3.000  1.00 20.00           H\n"
        )
        structure = parse_pdb(line, keep_hydrogens=True)
        self.assertEqual(structure.atoms[0].name, "HG11")
        self.assertEqual(structure.atoms[0].residue_name, "VAL")

    def test_hydrogens_are_dropped_by_default(self) -> None:
        line = (
            "ATOM      1  H   VAL A  10      "
            "1.000   2.000   3.000  1.00 20.00           H\n"
        )
        self.assertEqual(len(parse_pdb(line).atoms), 0)
        self.assertEqual(len(parse_pdb(line, keep_hydrogens=True).atoms), 1)

    def test_only_the_first_alternate_conformation_is_kept(self) -> None:
        # Keeping both would place one atom at two positions and inflate any
        # geometry computed from it.
        text = (
            "ATOM      1  CA AMET A   1      "
            "1.000   1.000   1.000  0.60 20.00           C\n"
            "ATOM      2  CA BMET A   1      "
            "9.000   9.000   9.000  0.40 20.00           C\n"
        )
        structure = parse_pdb(text)
        self.assertEqual(len(structure.atoms), 1)
        self.assertAlmostEqual(structure.atoms[0].x, 1.0)
        self.assertTrue(any("alternate" in w for w in structure.warnings))

    def test_only_the_first_model_is_read(self) -> None:
        # An NMR ensemble holds twenty copies of the same molecule; merging them
        # would put every atom at twenty slightly different places.
        text = (
            "MODEL        1\n"
            "ATOM      1  CA  MET A   1      "
            "1.000   1.000   1.000  1.00 20.00           C\n"
            "ENDMDL\n"
            "MODEL        2\n"
            "ATOM      1  CA  MET A   1      "
            "5.000   5.000   5.000  1.00 20.00           C\n"
            "ENDMDL\n"
        )
        structure = parse_pdb(text)
        self.assertEqual(len(structure.atoms), 1)
        self.assertAlmostEqual(structure.atoms[0].x, 1.0)

    def test_malformed_lines_are_counted_not_fatal(self) -> None:
        text = SAMPLE_PDB + "ATOM     99  CA  XXX A   1      bad     bad     bad\n"
        structure = parse_pdb(text)
        self.assertTrue(any("could not be parsed" in w for w in structure.warnings))
        self.assertEqual(len(structure.protein_atoms), 6)

    def test_low_resolution_is_flagged(self) -> None:
        text = SAMPLE_PDB.replace("RESOLUTION.    1.80", "RESOLUTION.    3.40")
        structure = parse_pdb(text)
        self.assertTrue(any("resolution 3.40" in w for w in structure.warnings))

    def test_a_structure_with_no_protein_is_flagged(self) -> None:
        structure = parse_pdb("HETATM    1  O   HOH A 401      1.0   1.0   1.0\n")
        self.assertTrue(any("not a receptor" in w for w in structure.warnings))


class TestLigandIdentification(unittest.TestCase):
    """The quiet failure: docking into a buffer molecule."""

    def test_water_is_not_a_ligand(self) -> None:
        structure = parse_pdb(SAMPLE_PDB)
        names = {residue.name for residue in structure.candidate_ligands()}
        self.assertNotIn("HOH", names)

    def test_glycerol_is_not_a_ligand(self) -> None:
        # The single most common false ligand: a cryoprotectant that sits in
        # pockets and has enough atoms to look plausible.
        structure = parse_pdb(SAMPLE_PDB)
        names = {residue.name for residue in structure.candidate_ligands()}
        self.assertNotIn("GOL", names)

    def test_metal_ions_are_not_ligands(self) -> None:
        structure = parse_pdb(SAMPLE_PDB)
        names = {residue.name for residue in structure.candidate_ligands()}
        self.assertNotIn("ZN", names)

    def test_the_real_ligand_is_found(self) -> None:
        structure = parse_pdb(SAMPLE_PDB)
        ligand = structure.best_ligand()
        self.assertIsNotNone(ligand)
        assert ligand is not None
        self.assertEqual(ligand.name, "LIG")
        self.assertEqual(ligand.n_heavy_atoms, 9)

    def test_small_organic_fragments_are_excluded_by_size(self) -> None:
        # Below the heavy-atom floor an unlisted additive is as likely as a
        # binder, and a box around it would be too small to be useful anyway.
        text = (
            "ATOM      1  CA  ALA A   1      "
            "1.000   1.000   1.000  1.00 20.00           C\n"
            "HETATM    2  C1  XYZ A 500      "
            "2.000   2.000   2.000  1.00 20.00           C\n"
            "HETATM    3  C2  XYZ A 500      "
            "3.000   2.000   2.000  1.00 20.00           C\n"
        )
        structure = parse_pdb(text)
        self.assertEqual(structure.candidate_ligands(), [])

    def test_inorganic_groups_are_excluded_even_if_unlisted(self) -> None:
        # The name list cannot be exhaustive, so a carbon requirement catches
        # inorganic clusters that slip through it.
        lines = [
            "ATOM      1  CA  ALA A   1       "
            "1.000   1.000   1.000  1.00 20.00           C"
        ]
        lines += [
            hetatm_line(index + 2, f"S{index}", "UNL", "A", 500,
                        float(index), 2.0, 2.0, "S")
            for index in range(10)
        ]
        structure = parse_pdb("\n".join(lines) + "\n")
        # The atoms must actually have parsed, or this passes for the wrong reason.
        self.assertEqual(len(structure.hetatms), 10)
        self.assertEqual(structure.candidate_ligands(), [])

    def test_apo_structure_reports_no_ligand_rather_than_guessing(self) -> None:
        text = "\n".join(
            line
            for line in SAMPLE_PDB.splitlines()
            if "LIG" not in line
        )
        structure = parse_pdb(text + "\n")
        self.assertIsNone(structure.best_ligand())
        self.assertIn("apo structure", structure.describe())

    def test_largest_candidate_wins(self) -> None:
        text = SAMPLE_PDB.replace("END\n", "")
        text += "\n".join(
            hetatm_line(index + 30, f"C{index}", "BIG", "A", 700,
                        60.0 + index, 60.0, 60.0, "C")
            for index in range(12)
        ) + "\nEND\n"
        structure = parse_pdb(text)
        best = structure.best_ligand()
        assert best is not None
        self.assertEqual(best.n_heavy_atoms, 12)
        self.assertEqual(best.name, "BIG")

    def test_exclusion_list_covers_the_usual_suspects(self) -> None:
        for residue in ("HOH", "GOL", "EDO", "SO4", "DMS", "PEG", "MPD", "NAG"):
            with self.subTest(residue=residue):
                self.assertIn(residue, NON_LIGAND_RESIDUES)

    def test_heavy_atom_floor_is_documented_and_used(self) -> None:
        self.assertGreaterEqual(MIN_LIGAND_HEAVY_ATOMS, 6)


class TestReceptorPreparation(unittest.TestCase):
    def test_stripping_removes_the_ligand_and_solvent(self) -> None:
        # Leaving the co-crystallised ligand in place scores a new molecule
        # against a pocket that has no room for it.
        structure = parse_pdb(SAMPLE_PDB)
        receptor = strip_to_receptor(structure)
        self.assertEqual(len(receptor), 6)
        self.assertTrue(all(not atom.is_hetatm for atom in receptor))

    def test_chain_selection(self) -> None:
        structure = parse_pdb(SAMPLE_PDB)
        self.assertEqual(len(strip_to_receptor(structure, keep_chains={"A"})), 6)
        self.assertEqual(len(strip_to_receptor(structure, keep_chains={"B"})), 0)

    def test_round_trip_through_the_writer(self) -> None:
        structure = parse_pdb(SAMPLE_PDB)
        text = write_pdb(strip_to_receptor(structure), title="receptor")
        reparsed = parse_pdb(text)
        self.assertEqual(len(reparsed.atoms), 6)
        for original, copy in zip(
            structure.protein_atoms, reparsed.atoms, strict=True
        ):
            self.assertAlmostEqual(original.x, copy.x, places=3)
            self.assertEqual(original.element, copy.element)
            self.assertEqual(original.residue_name, copy.residue_name)


class TestGeometry(unittest.TestCase):
    def test_centroid(self) -> None:
        self.assertEqual(centroid([(0, 0, 0), (2, 2, 2)]), (1.0, 1.0, 1.0))

    def test_centroid_of_nothing_raises(self) -> None:
        # Returning the origin would place a box at (0,0,0) -- outside every real
        # structure, and silently.
        with self.assertRaises(ValueError):
            centroid([])

    def test_extent(self) -> None:
        low, high = extent([(0, 1, 2), (4, 5, 6), (-1, 0, 3)])
        self.assertEqual(low, (-1, 0, 2))
        self.assertEqual(high, (4, 5, 6))

    def test_rmsd_of_identical_poses_is_zero(self) -> None:
        points = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
        self.assertAlmostEqual(rmsd(points, points), 0.0)

    def test_rmsd_is_the_usual_formula(self) -> None:
        # Two atoms displaced by 3 and 4 along one axis: sqrt((9+16)/2).
        self.assertAlmostEqual(
            rmsd([(0, 0, 0), (0, 0, 0)], [(3, 0, 0), (4, 0, 0)]),
            (25 / 2) ** 0.5,
        )

    def test_rmsd_requires_matched_sets(self) -> None:
        with self.assertRaises(ValueError):
            rmsd([(0, 0, 0)], [(0, 0, 0), (1, 1, 1)])

    def test_two_angstrom_threshold(self) -> None:
        self.assertTrue(pose_is_correct(1.9))
        self.assertFalse(pose_is_correct(2.1))

    def test_atoms_within_radius(self) -> None:
        structure = parse_pdb(SAMPLE_PDB)
        near = atoms_within(structure.protein_atoms, (10.0, 10.0, 10.0), 2.5)
        self.assertEqual(len(near), 3)


class TestBoxes(unittest.TestCase):
    def test_box_is_centred_on_the_bounding_box_not_the_centroid(self) -> None:
        # For an elongated ligand the two differ by several angstroms, and the
        # bounding-box midpoint is what keeps the molecule inside the search space.
        points = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (100, 0, 0)]
        box = box_from_points(points, padding=0.0)
        self.assertAlmostEqual(box.center[0], 50.0)

    def test_padding_is_added_on_both_sides(self) -> None:
        box = box_from_points([(0, 0, 0), (10, 0, 0)], padding=5.0, minimum_size=0.0)
        self.assertAlmostEqual(box.size[0], 20.0)

    def test_minimum_size_is_enforced(self) -> None:
        box = box_from_points([(0, 0, 0)], padding=0.0)
        self.assertGreaterEqual(min(box.size), 12.0)

    def test_oversized_box_is_flagged_as_effectively_blind(self) -> None:
        box = box_from_points([(0, 0, 0), (60, 60, 60)], padding=8.0)
        self.assertTrue(box.is_blind)
        self.assertTrue(any("effectively blind" in w for w in box.warnings))

    def test_box_from_ligand_uses_the_ligand_position(self) -> None:
        structure = parse_pdb(SAMPLE_PDB)
        ligand = structure.best_ligand()
        assert ligand is not None
        box = box_from_ligand(ligand, padding=8.0)
        # The ligand spans x 20-23, y 20-21, z 20-22.
        self.assertAlmostEqual(box.center[0], 21.5)
        self.assertAlmostEqual(box.center[1], 20.5)
        self.assertAlmostEqual(box.center[2], 21.0)
        self.assertIn("co-crystallised", box.derived_from)

    def test_box_from_ligand_is_nowhere_near_the_glycerol(self) -> None:
        # The whole point of the exclusion list, expressed as geometry.
        structure = parse_pdb(SAMPLE_PDB)
        ligand = structure.best_ligand()
        assert ligand is not None
        box = box_from_ligand(ligand)
        self.assertFalse(box.contains((40.0, 40.0, 40.0)))

    def test_box_from_a_fragment_warns_about_coverage(self) -> None:
        structure = parse_pdb(SAMPLE_PDB)
        ligand = structure.best_ligand()
        assert ligand is not None
        box = box_from_ligand(ligand)
        self.assertTrue(any("heavy atoms" in w for w in box.warnings))

    def test_box_from_residues(self) -> None:
        structure = parse_pdb(SAMPLE_PDB)
        box = box_from_residues(
            structure, [32, 228], chain="A", evidence="catalytic aspartate dyad"
        )
        self.assertAlmostEqual(box.center[1], 12.0)
        self.assertIn("catalytic", box.evidence)

    def test_box_from_residues_without_evidence_warns(self) -> None:
        structure = parse_pdb(SAMPLE_PDB)
        box = box_from_residues(structure, [32], chain="A")
        self.assertTrue(any("no evidence" in w for w in box.warnings))

    def test_box_from_missing_residues_raises_with_a_usable_message(self) -> None:
        structure = parse_pdb(SAMPLE_PDB)
        with self.assertRaises(ValueError) as context:
            box_from_residues(structure, [9999], chain="A")
        self.assertIn("numbering", str(context.exception))

    def test_multi_chain_residue_selection_is_flagged(self) -> None:
        # In a multimer this centres the box between protein copies -- in solvent.
        text = SAMPLE_PDB + (
            "ATOM     50  CA  ASP B  32      "
            "90.000  90.000  90.000  1.00 20.00           C\n"
        )
        structure = parse_pdb(text)
        box = box_from_residues(structure, [32], evidence="dyad")
        self.assertTrue(any("multimer" in w for w in box.warnings))

    def test_blind_box_shouts_about_itself(self) -> None:
        structure = parse_pdb(SAMPLE_PDB)
        box = blind_box(structure)
        self.assertIn("BLIND DOCKING", " ".join(box.warnings))
        self.assertIn("hypothesis", " ".join(box.warnings))

    def test_box_describes_its_provenance(self) -> None:
        structure = parse_pdb(SAMPLE_PDB)
        ligand = structure.best_ligand()
        assert ligand is not None
        text = box_from_ligand(ligand).describe()
        self.assertIn("derived from", text)
        self.assertIn("evidence", text)

    def test_contains(self) -> None:
        box = Box(center=(0, 0, 0), size=(10, 10, 10), derived_from="t", evidence="t")
        self.assertTrue(box.contains((4.9, 0, 0)))
        self.assertFalse(box.contains((5.1, 0, 0)))


if __name__ == "__main__":
    unittest.main()
