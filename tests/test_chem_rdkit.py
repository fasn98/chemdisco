"""Tests for the RDKit edge: standardisation, descriptors, scaffolds, alerts.

Skipped entirely when RDKit is absent, which is how the rest of the suite stays
runnable in a bare environment. In CI RDKit is installed, so these run there and
are the gate on anything touching chemistry.

The structures used are real and deliberately chosen. Imatinib and its mesylate
salt exercise salt stripping on a case that genuinely appears in ChEMBL under two
molecule ids. Quercetin is a catechol and a known PAINS match. Taxol is the
synthetic-accessibility upper bound.
"""

from __future__ import annotations

import unittest

from chemdisco.chem.standardize import RDKIT_AVAILABLE

requires_rdkit = unittest.skipUnless(
    RDKIT_AVAILABLE, "RDKit is not installed in this environment"
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
IMATINIB = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
IMATINIB_MESYLATE = IMATINIB + ".CS(=O)(=O)O"
SODIUM_BENZOATE = "[Na+].[O-]C(=O)c1ccccc1"
QUERCETIN = "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
ETHANOL = "CCO"
BENZENE = "c1ccccc1"


@requires_rdkit
class TestValidation(unittest.TestCase):
    def test_valid_smiles_accepted(self) -> None:
        from chemdisco.chem.standardize import validate_smiles

        ok, reason = validate_smiles(ASPIRIN)
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_unclosed_ring_is_rejected_with_a_reason(self) -> None:
        # The predecessor crashed on exactly this class of malformed SMILES.
        from chemdisco.chem.standardize import validate_smiles

        ok, reason = validate_smiles("c1ccccc")
        self.assertFalse(ok)
        self.assertIsNotNone(reason)

    def test_bad_valence_is_rejected(self) -> None:
        from chemdisco.chem.standardize import validate_smiles

        ok, reason = validate_smiles("C(C)(C)(C)(C)C")
        self.assertFalse(ok)
        self.assertIn("sanitis", (reason or "").lower())

    def test_empty_is_rejected(self) -> None:
        from chemdisco.chem.standardize import validate_smiles

        self.assertFalse(validate_smiles("")[0])


@requires_rdkit
class TestStandardization(unittest.TestCase):
    def test_salt_is_stripped_to_the_parent(self) -> None:
        # This is the duplicate that leaks across a train/test split if missed.
        from chemdisco.chem.standardize import standardize

        free_base = standardize(IMATINIB)
        mesylate = standardize(IMATINIB_MESYLATE)
        self.assertTrue(free_base.ok)
        self.assertTrue(mesylate.ok)
        self.assertEqual(free_base.smiles, mesylate.smiles)
        self.assertTrue(mesylate.parent_changed)

    def test_salt_and_free_base_share_an_inchikey(self) -> None:
        from chemdisco.chem.standardize import inchikey_of

        self.assertEqual(inchikey_of(IMATINIB), inchikey_of(IMATINIB_MESYLATE))

    def test_charged_form_is_neutralised(self) -> None:
        from chemdisco.chem.standardize import inchikey_of

        self.assertEqual(
            inchikey_of(SODIUM_BENZOATE), inchikey_of("OC(=O)c1ccccc1")
        )

    def test_stereochemistry_is_preserved(self) -> None:
        # Enantiomers differ in potency by orders of magnitude; merging them
        # would pool genuinely different compounds.
        from chemdisco.chem.standardize import inchikey_of

        r_form = inchikey_of("C[C@H](N)C(=O)O")
        s_form = inchikey_of("C[C@@H](N)C(=O)O")
        self.assertIsNotNone(r_form)
        self.assertNotEqual(r_form, s_form)

    def test_skeleton_key_ignores_stereochemistry(self) -> None:
        from chemdisco.chem.standardize import skeleton_inchikey

        self.assertEqual(
            skeleton_inchikey("C[C@H](N)C(=O)O"),
            skeleton_inchikey("C[C@@H](N)C(=O)O"),
        )

    def test_failure_is_returned_not_raised(self) -> None:
        from chemdisco.chem.standardize import standardize

        result = standardize("not a molecule at all")
        self.assertFalse(result.ok)
        self.assertIsNotNone(result.error)

    def test_deduplication_groups_salt_forms(self) -> None:
        from chemdisco.chem.standardize import deduplicate_by_structure

        representatives, groups = deduplicate_by_structure(
            [IMATINIB, IMATINIB_MESYLATE, ASPIRIN]
        )
        self.assertEqual(len(representatives), 2)
        self.assertTrue(any(len(indices) == 2 for indices in groups.values()))

    def test_molecular_weight_is_a_derived_quantity(self) -> None:
        from chemdisco.chem.standardize import molecular_weight

        quantity = molecular_weight(ASPIRIN)
        self.assertTrue(quantity.is_known)
        self.assertAlmostEqual(quantity.require(), 180.16, places=1)
        self.assertEqual(quantity.origin.value, "derived")

    def test_molecular_weight_of_garbage_is_unknown_not_a_default(self) -> None:
        from chemdisco.chem.standardize import molecular_weight

        quantity = molecular_weight("$$$")
        self.assertFalse(quantity.is_known)
        self.assertEqual(quantity.label(), "not computed")


@requires_rdkit
class TestDescriptors(unittest.TestCase):
    def test_known_values_are_reproduced(self) -> None:
        from chemdisco.chem.descriptors import compute_descriptors

        result = compute_descriptors(ASPIRIN)
        self.assertTrue(result.ok)
        assert result.values is not None
        self.assertAlmostEqual(result.values["molecular_weight"], 180.16, places=1)
        self.assertAlmostEqual(result.values["tpsa"], 63.6, places=1)
        self.assertEqual(result.values["aromatic_rings"], 1.0)

    def test_failure_reports_rather_than_imputing(self) -> None:
        from chemdisco.chem.descriptors import compute_descriptors

        result = compute_descriptors("c1ccccc")
        self.assertFalse(result.ok)
        quantities = result.as_quantities()
        # Every descriptor must be an explicit unknown, not a typical value.
        for name, quantity in quantities.items():
            with self.subTest(descriptor=name):
                self.assertFalse(quantity.is_known)

    def test_matrix_reports_which_rows_failed(self) -> None:
        from chemdisco.chem.descriptors import descriptor_matrix

        matrix, names, valid, errors = descriptor_matrix(
            [ASPIRIN, "c1ccccc", CAFFEINE]
        )
        self.assertEqual(matrix.shape[0], 3)
        self.assertEqual(list(valid), [True, False, True])
        self.assertEqual(len(errors), 1)

    def test_fingerprints_have_the_requested_width(self) -> None:
        from chemdisco.chem.descriptors import fingerprint_matrix

        matrix, valid = fingerprint_matrix([ASPIRIN, CAFFEINE], n_bits=1024)
        self.assertEqual(matrix.shape, (2, 1024))
        self.assertTrue(all(valid))
        self.assertTrue(set(matrix.flatten().tolist()) <= {0, 1})

    def test_identical_structures_give_identical_fingerprints(self) -> None:
        from chemdisco.chem.descriptors import fingerprint_matrix

        matrix, _ = fingerprint_matrix([ASPIRIN, ASPIRIN])
        self.assertTrue((matrix[0] == matrix[1]).all())

    def test_combined_features_concatenate_both_blocks(self) -> None:
        from chemdisco.chem.descriptors import PHYSCHEM_DESCRIPTORS, combined_features

        matrix, names, valid, _ = combined_features([ASPIRIN, CAFFEINE], n_bits=256)
        self.assertEqual(matrix.shape, (2, 256 + len(PHYSCHEM_DESCRIPTORS)))
        self.assertEqual(len(names), matrix.shape[1])
        self.assertTrue(all(valid))


@requires_rdkit
class TestScaffolds(unittest.TestCase):
    def test_acyclic_molecule_has_no_scaffold(self) -> None:
        from chemdisco.chem.scaffold import murcko_scaffold

        self.assertEqual(murcko_scaffold(ETHANOL), "")

    def test_aromatic_ring_yields_a_scaffold(self) -> None:
        from chemdisco.chem.scaffold import murcko_scaffold

        self.assertTrue(murcko_scaffold(ASPIRIN))

    def test_side_chains_are_removed(self) -> None:
        from chemdisco.chem.scaffold import murcko_scaffold

        # Toluene and ethylbenzene differ only in a side chain, so they share a
        # scaffold and must never be split apart.
        self.assertEqual(murcko_scaffold("Cc1ccccc1"), murcko_scaffold("CCc1ccccc1"))

    def test_generic_framework_merges_heteroatom_variants(self) -> None:
        from chemdisco.chem.scaffold import murcko_scaffold

        pyridine = murcko_scaffold("c1ccncc1", generic=True)
        benzene = murcko_scaffold(BENZENE, generic=True)
        self.assertEqual(pyridine, benzene)
        # At the standard Murcko level they remain distinct.
        self.assertNotEqual(
            murcko_scaffold("c1ccncc1"), murcko_scaffold(BENZENE)
        )

    def test_dataset_scan_separates_acyclic_from_unparseable(self) -> None:
        from chemdisco.chem.scaffold import scaffolds_for_dataset

        scaffolds, acyclic, unparseable = scaffolds_for_dataset(
            [ASPIRIN, ETHANOL, "c1ccccc"]
        )
        self.assertEqual(acyclic, [1])
        self.assertEqual(unparseable, [2])
        self.assertEqual(len(scaffolds), 3)

    def test_summary_warns_about_a_single_series(self) -> None:
        from chemdisco.chem.scaffold import scaffold_summary

        summary = scaffold_summary([ASPIRIN] * 20)
        self.assertIn("WARNING", summary)
        self.assertIn("one or two chemical series", summary)

    def test_scaffold_split_runs_on_real_structures(self) -> None:
        # The integration point: RDKit perception feeding the toolkit-free splitter.
        from chemdisco.chem.scaffold import scaffolds_for_dataset
        from chemdisco.split import scaffold_split, verify_disjoint

        structures = (
            [ASPIRIN, "CC(=O)Oc1ccccc1C(=O)OC", "CC(=O)Oc1ccccc1C(N)=O"]
            + [CAFFEINE, "Cn1cnc2c1c(=O)[nH]c(=O)n2C"]
            + [QUERCETIN, "O=c1cc(-c2ccccc2)oc2ccccc12"]
            + [IMATINIB]
        )
        scaffolds, _, _ = scaffolds_for_dataset(structures)
        split = scaffold_split(scaffolds, test_fraction=0.3)
        self.assertEqual(verify_disjoint(split, scaffolds), [])
        self.assertTrue(split.test)


@requires_rdkit
class TestAlertsAndAccessibility(unittest.TestCase):
    def test_quercetin_matches_pains(self) -> None:
        # A catechol: a textbook PAINS match through redox cycling.
        from chemdisco.chem.alerts import screen_alerts

        report = screen_alerts(QUERCETIN)
        self.assertIsNone(report.error)
        self.assertTrue(report.pains, "quercetin should match a PAINS pattern")
        # The report must say which pattern and must not present it as a verdict.
        self.assertIn("false positives", report.describe())

    def test_caffeine_is_clean(self) -> None:
        from chemdisco.chem.alerts import screen_alerts

        report = screen_alerts(CAFFEINE)
        self.assertIsNone(report.error)
        self.assertEqual(report.n_alerts, 0)
        self.assertTrue(report.clean)

    def test_alert_screening_failure_is_reported(self) -> None:
        from chemdisco.chem.alerts import screen_alerts

        report = screen_alerts("c1ccccc")
        self.assertIsNotNone(report.error)
        self.assertFalse(report.clean)

    def test_sascore_orders_easy_below_hard(self) -> None:
        from chemdisco.chem.alerts import synthetic_accessibility

        easy = synthetic_accessibility(ASPIRIN)
        hard = synthetic_accessibility(
            # Paclitaxel: among the most synthetically demanding drugs in use.
            "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(c1ccccc1)NC(=O)c1ccccc1)O)O)OC(=O)c1ccccc1)(CO4)OC(C)=O)O)C)OC(C)=O"
        )
        if not (easy.is_known and hard.is_known):
            self.skipTest("RDKit SA_Score contrib module unavailable")
        self.assertLess(easy.require(), hard.require())
        self.assertEqual(easy.origin.value, "heuristic")

    def test_sascore_is_labelled_a_heuristic_not_a_measurement(self) -> None:
        from chemdisco.chem.alerts import synthetic_accessibility

        quantity = synthetic_accessibility(ASPIRIN)
        if not quantity.is_known:
            self.skipTest("RDKit SA_Score contrib module unavailable")
        self.assertIn("heuristic", quantity.label())

    def test_lipinski_counts_violations_and_states_its_limits(self) -> None:
        from chemdisco.chem.alerts import lipinski_violations

        clean = lipinski_violations(ASPIRIN)
        self.assertEqual(clean.require(), 0.0)
        self.assertTrue(
            any("violate it" in note for note in clean.notes),
            "the caveat about approved drugs violating the rule must travel with it",
        )


@requires_rdkit
class TestSimilarity(unittest.TestCase):
    def test_identical_structures_are_perfectly_similar(self) -> None:
        from chemdisco.chem.similarity import tanimoto

        self.assertAlmostEqual(tanimoto(ASPIRIN, ASPIRIN) or 0.0, 1.0)

    def test_unrelated_structures_are_dissimilar(self) -> None:
        from chemdisco.chem.similarity import tanimoto

        value = tanimoto(ASPIRIN, CAFFEINE)
        self.assertIsNotNone(value)
        self.assertLess(value or 1.0, 0.3)

    def test_parse_failure_is_distinguishable_from_zero_similarity(self) -> None:
        from chemdisco.chem.similarity import similarity_matrix

        matrix = similarity_matrix(["c1ccccc"], [ASPIRIN])
        self.assertEqual(matrix[0, 0], -1.0)

    def test_exact_match_is_caught_through_a_salt_form(self) -> None:
        from chemdisco.chem.similarity import assess_novelty

        verdicts = assess_novelty([IMATINIB_MESYLATE], [IMATINIB])
        self.assertTrue(verdicts[0].is_exact_match)
        self.assertFalse(verdicts[0].is_novel)
        self.assertIn("rediscovery", verdicts[0].describe())

    def test_genuinely_different_structure_is_novel(self) -> None:
        from chemdisco.chem.similarity import assess_novelty

        verdicts = assess_novelty([CAFFEINE], [ASPIRIN])
        self.assertTrue(verdicts[0].is_novel)

    def test_novel_structure_is_flagged_as_extrapolation_territory(self) -> None:
        from chemdisco.chem.similarity import assess_novelty

        verdict = assess_novelty([CAFFEINE], [ASPIRIN])[0]
        self.assertIn("extrapolation", verdict.describe())

    def test_diverse_subset_collapses_near_duplicates(self) -> None:
        from chemdisco.chem.similarity import diverse_subset

        selected = diverse_subset([ASPIRIN, ASPIRIN, ASPIRIN, CAFFEINE])
        self.assertEqual(len(selected), 2)


@requires_rdkit
class TestGeneration(unittest.TestCase):
    """BRICS recombination on a small but real fragment set."""

    ACTIVES = [
        "Cc1ccc(NC(=O)c2ccccc2)cc1",
        "COc1ccc(NC(=O)c2ccc(Cl)cc2)cc1",
        "Clc1ccc(NC(=O)c2ccccn2)cc1",
        "Cc1ccc(S(=O)(=O)Nc2ccccc2)cc1",
        "O=C(Nc1ccccc1)c1ccc(N2CCOCC2)cc1",
    ]

    def test_decomposition_produces_fragments(self) -> None:
        from chemdisco.generate import decompose_to_fragments

        fragments, errors = decompose_to_fragments(self.ACTIVES)
        self.assertEqual(errors, [])
        self.assertGreater(len(fragments), 3)

    def test_generation_produces_structures_and_reports_attrition(self) -> None:
        from chemdisco.generate import GenerationPolicy, generate_candidates

        report = generate_candidates(
            self.ACTIVES,
            policy=GenerationPolicy(max_generated=300),
            seed=1,
        )
        self.assertGreater(report.n_fragments, 0)
        self.assertGreater(report.n_generated, 0)
        # The attrition account is the informative part, whatever survives.
        text = report.describe()
        self.assertIn("Generated:", text)
        self.assertIn("Retained:", text)

    def test_generated_structures_are_valid_and_distinct(self) -> None:
        from chemdisco.chem.standardize import validate_smiles
        from chemdisco.generate import GenerationPolicy, generate_candidates

        report = generate_candidates(
            self.ACTIVES, policy=GenerationPolicy(max_generated=300), seed=2
        )
        keys = set()
        for candidate in report.candidates:
            with self.subTest(smiles=candidate.smiles):
                self.assertTrue(validate_smiles(candidate.smiles)[0])
                self.assertNotIn(candidate.inchikey, keys)
                keys.add(candidate.inchikey)

    def test_candidates_are_not_rankable_without_a_prediction(self) -> None:
        from chemdisco.generate import GenerationPolicy, generate_candidates

        report = generate_candidates(
            self.ACTIVES, policy=GenerationPolicy(max_generated=200), seed=3
        )
        for candidate in report.candidates:
            self.assertIsNone(candidate.predicted_activity)
            self.assertFalse(
                candidate.is_rankable,
                "an unscored candidate must never be rankable",
            )
        self.assertEqual(report.ranked(), [])

    def test_policy_audit_passes_a_well_matched_policy(self) -> None:
        from chemdisco.generate import audit_policy

        audit = audit_policy(self.ACTIVES)
        self.assertEqual(audit.n_actives, len(self.ACTIVES))
        self.assertFalse(
            audit.policy_is_suspect,
            f"these small amides should survive the default policy: "
            f"{audit.describe()}",
        )

    def test_audit_threshold_catches_the_real_bace1_figure(self) -> None:
        # Not a hypothetical. The first live BACE1 audit returned exactly this:
        # 21 of 40 known actives surviving, 18 lost to Brenk. A half threshold
        # called that acceptable, which is what prompted raising it to 0.8.
        from chemdisco.generate import PolicyAudit

        audit = PolicyAudit(
            n_actives=40,
            n_passing=21,
            rejections={"Brenk alert": 18, "too large (over 50 heavy atoms)": 1},
        )
        self.assertTrue(audit.policy_is_suspect)
        self.assertEqual(audit.dominant_rule, "Brenk alert")
        text = audit.describe()
        self.assertIn("48%", text)
        # The Brenk-specific explanation must appear, since that is the
        # actionable part.
        self.assertIn("lead-likeness", text)

    def test_policy_audit_flags_a_mis_calibrated_policy(self) -> None:
        # The BACE1 case, in miniature. A size ceiling below the actives
        # themselves rejects the chemistry the generator is supposed to explore,
        # and the audit must say so rather than letting the attrition be read as
        # a fact about the candidates.
        from chemdisco.generate import GenerationPolicy, audit_policy

        audit = audit_policy(self.ACTIVES, GenerationPolicy(max_heavy_atoms=5))
        self.assertTrue(audit.policy_is_suspect)
        self.assertEqual(audit.n_passing, 0)
        self.assertIn("mis-calibrated", audit.describe())
        self.assertTrue(audit.failing_examples)

    def test_generation_report_carries_the_audit(self) -> None:
        from chemdisco.generate import GenerationPolicy, generate_candidates

        report = generate_candidates(
            self.ACTIVES, policy=GenerationPolicy(max_generated=100), seed=5
        )
        self.assertIsNotNone(report.policy_audit)
        self.assertIn("Policy audit", report.describe())

    def test_empty_output_under_a_bad_policy_blames_the_policy(self) -> None:
        from chemdisco.generate import GenerationPolicy, generate_candidates

        report = generate_candidates(
            self.ACTIVES,
            policy=GenerationPolicy(max_generated=100, max_heavy_atoms=5),
            seed=6,
        )
        self.assertEqual(report.candidates, [])
        text = report.describe()
        self.assertIn("wrong for this target class", text)
        self.assertNotIn("Both are findings", text)

    def test_too_few_fragments_is_reported_as_a_finding(self) -> None:
        from chemdisco.generate import generate_candidates

        report = generate_candidates([ETHANOL])
        self.assertEqual(report.candidates, [])
        self.assertTrue(
            any("not a usable approach" in note for note in report.notes),
            "a target with too little chemistry to recombine must say so plainly",
        )


if __name__ == "__main__":
    unittest.main()
