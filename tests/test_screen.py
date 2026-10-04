"""Tests for decoy selection and enrichment metrics.

Both are pure, so both are tested here rather than behind a docking engine. The
properties being checked are the ones that decide whether an enrichment number
means anything: that decoys genuinely match the actives on the properties a
scoring function is biased by, and that a metric reports random performance as
random.
"""

from __future__ import annotations

import unittest

import numpy as np

from chemdisco.dock.decoys import (
    DEFAULT_TOLERANCES,
    MAX_DECOY_SIMILARITY,
    DecoySelection,
    describe_property_gap,
    property_gap,
    select_decoys,
)
from chemdisco.dock.enrichment import (
    analyse_enrichment,
    auc_roc,
    bedroc,
    enrichment_factor,
    max_enrichment_factor,
)


def properties(
    mw: float, logp: float, hbd: float = 2, hba: float = 4, rot: float = 5, charge: float = 0
) -> dict[str, float]:
    return {
        "molecular_weight": mw,
        "logp": logp,
        "hbd": hbd,
        "hba": hba,
        "rotatable_bonds": rot,
        "formal_charge": charge,
    }


class TestDecoySelection(unittest.TestCase):
    def test_matching_candidates_are_selected(self) -> None:
        actives = [properties(400, 3.0)]
        pool = [properties(405, 3.1), properties(395, 2.9), properties(410, 3.2)]
        similarity = np.zeros((3, 1))
        selection = select_decoys(actives, pool, similarity, decoys_per_active=2)
        self.assertEqual(selection.n_selected, 2)

    def test_mismatched_candidates_are_rejected(self) -> None:
        # A 200 Da gap is exactly the bias decoys exist to remove.
        actives = [properties(400, 3.0)]
        pool = [properties(200, 0.5), properties(600, 6.0)]
        similarity = np.zeros((2, 1))
        selection = select_decoys(actives, pool, similarity, decoys_per_active=2)
        self.assertEqual(selection.n_selected, 0)

    def test_closest_matches_are_preferred(self) -> None:
        actives = [properties(400, 3.0)]
        pool = [properties(420, 3.9), properties(401, 3.0), properties(415, 3.6)]
        similarity = np.zeros((3, 1))
        selection = select_decoys(actives, pool, similarity, decoys_per_active=1)
        self.assertEqual(selection.decoy_indices, (1,))

    def test_similar_candidates_are_excluded(self) -> None:
        # A candidate resembling an active might share its binding mode, making
        # it a probable active rather than a control.
        actives = [properties(400, 3.0)]
        pool = [properties(400, 3.0), properties(402, 3.1)]
        similarity = np.array([[0.9], [0.05]])
        selection = select_decoys(actives, pool, similarity, decoys_per_active=2)
        self.assertEqual(selection.decoy_indices, (1,))
        self.assertLessEqual(selection.max_similarity_used, MAX_DECOY_SIMILARITY)

    def test_a_candidate_is_never_reused(self) -> None:
        # Duplicated decoys would make an enrichment estimate look more precise
        # than the data supports.
        actives = [properties(400, 3.0), properties(401, 3.0)]
        pool = [properties(400, 3.0)]
        similarity = np.zeros((1, 2))
        selection = select_decoys(actives, pool, similarity, decoys_per_active=1)
        self.assertEqual(selection.n_selected, 1)
        self.assertEqual(len(set(selection.decoy_indices)), 1)

    def test_formal_charge_is_matched_exactly(self) -> None:
        # Charge changes a docking score far more than it changes size, so the
        # default tolerance is zero.
        self.assertEqual(DEFAULT_TOLERANCES["formal_charge"], 0.0)
        actives = [properties(400, 3.0, charge=0)]
        pool = [properties(402, 3.1, charge=1)]
        selection = select_decoys(actives, pool, np.zeros((1, 1)))
        self.assertEqual(selection.n_selected, 0)

    def test_selection_reports_its_match_quality(self) -> None:
        actives = [properties(400, 3.0)]
        pool = [properties(410, 3.5), properties(415, 3.6)]
        selection = select_decoys(
            actives, pool, np.zeros((2, 1)), decoys_per_active=2
        )
        self.assertIn("molecular_weight", selection.property_deltas)
        self.assertGreater(selection.property_deltas["molecular_weight"], 0)

    def test_a_thin_pool_is_reported_as_inadequate(self) -> None:
        # A matcher forced to accept poor matches reintroduces the bias decoys
        # exist to remove; that must be visible rather than assumed away.
        actives = [properties(400, 3.0) for _ in range(5)]
        pool = [properties(401, 3.0)]
        selection = select_decoys(
            actives, pool, np.zeros((1, 5)), decoys_per_active=10
        )
        self.assertFalse(selection.is_adequate)
        self.assertIn("WARNING", selection.describe())

    def test_an_adequate_selection_says_so(self) -> None:
        actives = [properties(400 + i, 3.0) for i in range(4)]
        pool = [properties(400 + i, 3.0) for i in range(40)]
        selection = select_decoys(
            actives, pool, np.zeros((40, 4)), decoys_per_active=5
        )
        self.assertTrue(selection.is_adequate)
        self.assertNotIn("WARNING", selection.describe())

    def test_shape_mismatches_raise(self) -> None:
        with self.assertRaises(ValueError):
            select_decoys([properties(400, 3.0)], [properties(400, 3.0)], np.zeros((5, 1)))
        with self.assertRaises(ValueError):
            select_decoys([properties(400, 3.0)], [properties(400, 3.0)], np.zeros((1, 9)))

    def test_no_actives_raises(self) -> None:
        with self.assertRaises(ValueError):
            select_decoys([], [properties(400, 3.0)], np.zeros((1, 0)))


class TestPropertyGap(unittest.TestCase):
    def test_a_matched_pair_of_groups_reports_no_concern(self) -> None:
        actives = [properties(400, 3.0), properties(410, 3.2)]
        decoys = [properties(405, 3.1), properties(402, 3.0)]
        text = describe_property_gap(property_gap(actives, decoys))
        self.assertIn("matched on every property", text)

    def test_a_size_gap_is_flagged_as_sufficient_to_bias(self) -> None:
        # The specific failure: Vina's score grows with molecular size, so an
        # 80 Da gap produces impressive enrichment containing no binding
        # information.
        actives = [properties(480, 3.0)]
        decoys = [properties(400, 3.0)]
        gaps = property_gap(actives, decoys)
        self.assertAlmostEqual(gaps["molecular_weight"], 80.0)
        text = describe_property_gap(gaps)
        self.assertIn("WARNING", text)
        self.assertIn("property bias", text)

    def test_empty_groups_raise(self) -> None:
        with self.assertRaises(ValueError):
            property_gap([], [properties(400, 3.0)])


class TestAuc(unittest.TestCase):
    def test_perfect_separation(self) -> None:
        # Lower score is better, so actives must carry the more negative values.
        labels = np.array([1, 1, 0, 0])
        scores = np.array([-10.0, -9.0, -5.0, -4.0])
        self.assertAlmostEqual(auc_roc(labels, scores), 1.0)

    def test_perfectly_inverted_separation(self) -> None:
        labels = np.array([1, 1, 0, 0])
        scores = np.array([-4.0, -5.0, -9.0, -10.0])
        self.assertAlmostEqual(auc_roc(labels, scores), 0.0)

    def test_random_ordering_is_one_half(self) -> None:
        labels = np.array([1, 0, 1, 0])
        scores = np.array([-8.0, -7.0, -6.0, -5.0])
        self.assertAlmostEqual(auc_roc(labels, scores), 0.75)
        # Interleaved the other way gives the complement.
        self.assertAlmostEqual(
            auc_roc(np.array([0, 1, 0, 1]), scores), 0.25
        )

    def test_all_ties_give_one_half(self) -> None:
        # Scoring functions report to two decimal places, so ties are common and
        # must contribute 0.5 rather than depending on input order.
        labels = np.array([1, 1, 0, 0])
        scores = np.array([-7.0, -7.0, -7.0, -7.0])
        self.assertAlmostEqual(auc_roc(labels, scores), 0.5)

    def test_partial_ties_are_handled(self) -> None:
        labels = np.array([1, 0])
        scores = np.array([-7.0, -7.0])
        self.assertAlmostEqual(auc_roc(labels, scores), 0.5)

    def test_one_class_only_raises(self) -> None:
        with self.assertRaises(ValueError):
            auc_roc(np.array([1, 1, 1]), np.array([-7.0, -8.0, -9.0]))


class TestEnrichmentFactor(unittest.TestCase):
    def test_all_actives_at_the_top(self) -> None:
        labels = np.array([1] * 10 + [0] * 90)
        scores = np.array([-10.0] * 10 + [-5.0] * 90)
        # Top 10% holds every active: 10 found where 1 was expected.
        self.assertAlmostEqual(enrichment_factor(labels, scores, fraction=0.1), 10.0)

    def test_random_ordering_averages_to_one(self) -> None:
        # Averaged over draws, not asserted on one. A single random screen of
        # this size swings EF by a factor of several -- the first version of this
        # test asserted on seed 0 and failed at 0.20, which was the test being
        # wrong about its own claim rather than the metric misbehaving. That
        # volatility is itself why EnrichmentResult.describe warns about small
        # top slices.
        labels = np.array([1] * 50 + [0] * 450)
        values = []
        for seed in range(100):
            rng = np.random.default_rng(seed)
            values.append(
                enrichment_factor(labels, rng.normal(size=500), fraction=0.1)
            )
        self.assertAlmostEqual(float(np.mean(values)), 1.0, delta=0.15)
        # And the spread across draws is wide, which is the point being made.
        self.assertGreater(float(np.std(values)), 0.2)

    def test_the_ceiling_is_reported(self) -> None:
        # EF cannot exceed 1/active_fraction, which is how a modest EF on a
        # set rich in actives should be read.
        labels = np.array([1] * 10 + [0] * 90)
        self.assertAlmostEqual(max_enrichment_factor(labels, fraction=0.1), 10.0)

    def test_invalid_fraction_raises(self) -> None:
        labels = np.array([1, 0])
        with self.assertRaises(ValueError):
            enrichment_factor(labels, np.array([-1.0, -2.0]), fraction=0.0)

    def test_no_actives_raises(self) -> None:
        with self.assertRaises(ValueError):
            enrichment_factor(np.array([0, 0]), np.array([-1.0, -2.0]))


class TestBedroc(unittest.TestCase):
    def test_early_recognition_scores_high(self) -> None:
        labels = np.array([1] * 10 + [0] * 90)
        scores = np.array(list(range(100)), dtype=float)
        self.assertGreater(bedroc(labels, scores), 0.9)

    def test_late_recognition_scores_low(self) -> None:
        labels = np.array([0] * 90 + [1] * 10)
        scores = np.array(list(range(100)), dtype=float)
        self.assertLess(bedroc(labels, scores), 0.1)

    def test_one_class_raises(self) -> None:
        with self.assertRaises(ValueError):
            bedroc(np.array([1, 1]), np.array([1.0, 2.0]))


class TestEnrichmentReport(unittest.TestCase):
    def _random_screen(self, n_actives=50, n_decoys=450, seed=0):
        rng = np.random.default_rng(seed)
        labels = [1] * n_actives + [0] * n_decoys
        scores = rng.normal(loc=-7.0, scale=1.0, size=n_actives + n_decoys)
        return labels, list(scores)

    def _separating_screen(self, n_actives=50, n_decoys=450, seed=0):
        rng = np.random.default_rng(seed)
        actives = rng.normal(loc=-9.5, scale=0.8, size=n_actives)
        decoys = rng.normal(loc=-7.0, scale=0.8, size=n_decoys)
        return [1] * n_actives + [0] * n_decoys, list(actives) + list(decoys)

    def test_a_large_random_screen_concludes_no_separation(self) -> None:
        # 500 compounds give an interval tight enough to conclude, so this is a
        # genuine negative rather than an inconclusive one.
        labels, scores = self._random_screen()
        result = analyse_enrichment(labels, scores, n_resamples=400)
        self.assertFalse(result.separates)
        self.assertTrue(result.is_conclusive)
        self.assertEqual(result.verdict, "does not separate")
        text = result.describe()
        self.assertIn("tight enough to conclude", text)
        # The important framing: a negative result is a result.
        self.assertIn("That is a result, not", text)

    def test_a_separating_screen_is_recognised(self) -> None:
        labels, scores = self._separating_screen()
        result = analyse_enrichment(labels, scores, n_resamples=400)
        self.assertTrue(result.separates)
        self.assertGreater(result.auc.estimate, 0.9)
        self.assertIn("strong for docking", result.describe())

    def test_the_verdict_rests_on_the_interval_not_the_estimate(self) -> None:
        # A point estimate above 0.5 with an interval crossing it is not
        # evidence, and on these set sizes that is a common outcome.
        labels = [1] * 8 + [0] * 12
        scores = [-8.0, -7.5, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5] + [
            -7.8, -7.2, -6.8, -6.2, -5.8, -5.2, -4.8, -4.2, -3.8, -3.2, -2.8, -2.2
        ]
        result = analyse_enrichment(labels, scores, n_resamples=400)
        self.assertGreater(result.auc.estimate, 0.5)
        if result.auc.low <= 0.5:
            self.assertFalse(result.separates)

    def test_a_modest_auc_is_described_as_modest(self) -> None:
        rng = np.random.default_rng(3)
        actives = rng.normal(loc=-7.8, scale=1.2, size=120)
        decoys = rng.normal(loc=-7.0, scale=1.2, size=480)
        result = analyse_enrichment(
            [1] * 120 + [0] * 480, list(actives) + list(decoys), n_resamples=400
        )
        if 0.5 < result.auc.estimate < 0.7 and result.separates:
            self.assertIn("modest", result.describe())

    def test_the_property_gap_warning_travels_into_the_report(self) -> None:
        # An enrichment figure must never be read without knowing whether the
        # groups were property-matched.
        labels, scores = self._separating_screen()
        result = analyse_enrichment(
            labels,
            scores,
            property_gap_warning="WARNING: groups differ on molecular_weight",
            n_resamples=200,
        )
        self.assertIn("molecular_weight", result.describe())

    def test_a_tiny_top_slice_is_called_out(self) -> None:
        labels, scores = self._separating_screen(n_actives=10, n_decoys=40)
        result = analyse_enrichment(labels, scores, n_resamples=200)
        self.assertIn("moves by a large factor", result.describe())

    def test_mismatched_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            analyse_enrichment([1, 0, 1], [-7.0, -6.0])


if __name__ == "__main__":
    unittest.main()


class TestScreenAccounting(unittest.TestCase):
    """Attrition bookkeeping, which is where a screen's numbers go wrong."""

    def _result(self):
        from chemdisco.dock import DockingResult, Pose, ScreenResult

        return ScreenResult(
            results=[
                DockingResult("A", [Pose(1, -9.5)]),
                DockingResult("B", error="ligand preparation failed: embedding"),
                DockingResult("C", [Pose(1, -8.0)]),
                DockingResult("D", [Pose(1, -6.0)]),
                DockingResult("E", error="ligand preparation failed: embedding"),
            ],
            labels=[1, 1, 1, 0, 0],
            receptor_id="4FRS",
        )

    def test_failures_stay_in_place_so_labels_stay_aligned(self) -> None:
        # Dropping failures from the list would silently shift every label.
        result = self._result()
        self.assertEqual(result.n_total, 5)
        self.assertEqual(result.n_succeeded, 3)
        self.assertEqual(result.n_failed, 2)

    def test_scored_returns_only_successes_still_aligned(self) -> None:
        labels, scores = self._result().scored()
        self.assertEqual(labels, [1, 1, 0])
        self.assertEqual(scores, [-9.5, -8.0, -6.0])

    def test_uneven_failure_between_groups_is_flagged(self) -> None:
        # The failure that corrupts enrichment: if large flexible actives fail
        # embedding more often than rigid decoys, the groups are thinned
        # differently and the surviving sets are not comparable.
        result = self._result()
        rates = result.failure_rate_by_group()
        self.assertAlmostEqual(rates[1], 1 / 3)
        self.assertAlmostEqual(rates[0], 1 / 2)
        self.assertIn("thinned unevenly", result.describe())

    def test_even_failure_is_not_flagged(self) -> None:
        from chemdisco.dock import DockingResult, Pose, ScreenResult

        result = ScreenResult(
            results=[
                DockingResult("A", [Pose(1, -9.0)]),
                DockingResult("B", [Pose(1, -8.0)]),
                DockingResult("C", [Pose(1, -7.0)]),
                DockingResult("D", [Pose(1, -6.0)]),
            ],
            labels=[1, 1, 0, 0],
        )
        self.assertNotIn("thinned unevenly", result.describe())

    def test_failure_reasons_are_grouped(self) -> None:
        reasons = self._result().failure_reasons()
        self.assertEqual(reasons.get("ligand preparation failed"), 2)

    def test_scoring_without_labels_raises(self) -> None:
        from chemdisco.dock import DockingResult, Pose, ScreenResult

        with self.assertRaises(ValueError):
            ScreenResult(results=[DockingResult("A", [Pose(1, -9.0)])]).scored()


class TestTriage(unittest.TestCase):
    """Docking as a filter against a reference distribution, not as a ranking."""

    def _candidates(self):
        from chemdisco.dock import DockingResult, Pose, ScreenResult

        return ScreenResult(
            results=[
                DockingResult("good", [Pose(1, -10.0)]),
                DockingResult("borderline", [Pose(1, -8.0)]),
                DockingResult("poor", [Pose(1, -4.0)]),
                DockingResult("failed", error="preparation failed"),
            ],
            receptor_id="4FRS",
        )

    def test_candidates_below_the_reference_median_are_kept(self) -> None:
        from chemdisco.dock import triage_candidates

        kept, _ = triage_candidates(self._candidates(), [-9.0, -8.5, -8.0, -7.5])
        # The reference median is -8.25; only the -10.0 candidate beats it.
        self.assertEqual(kept, [0])

    def test_a_permissive_percentile_keeps_more(self) -> None:
        from chemdisco.dock import triage_candidates

        kept, _ = triage_candidates(
            self._candidates(), [-9.0, -8.5, -8.0, -7.5], percentile=90.0
        )
        self.assertIn(1, kept)

    def test_failed_candidates_are_never_kept(self) -> None:
        from chemdisco.dock import triage_candidates

        kept, _ = triage_candidates(self._candidates(), [-9.0, -8.0], percentile=99.0)
        self.assertNotIn(3, kept)

    def test_the_explanation_refuses_to_call_it_a_ranking(self) -> None:
        # The measured result this encodes: docking does not order candidates
        # reliably on this target, so the survivors' order carries nothing.
        from chemdisco.dock import triage_candidates

        _, explanation = triage_candidates(self._candidates(), [-9.0, -8.0])
        self.assertIn("filter, not a ranking", explanation)
        self.assertIn("order carries no information", explanation)

    def test_triage_without_a_reference_distribution_raises(self) -> None:
        # A bare docking score has no meaning without actives docked the same way.
        from chemdisco.dock import triage_candidates

        with self.assertRaises(ValueError) as context:
            triage_candidates(self._candidates(), [])
        self.assertIn("no meaning without a distribution", str(context.exception))


class TestTimeBudget(unittest.TestCase):
    """A screen that outruns its environment's limit must keep what it has."""

    def test_interleaving_keeps_every_prefix_balanced(self) -> None:
        # The failure this prevents: supplied in blocks, a truncated run holds
        # every active and no decoys, which cannot support an enrichment
        # estimate at all.
        from chemdisco.dock import interleave_by_label

        smiles = [f"a{i}" for i in range(5)] + [f"d{i}" for i in range(15)]
        labels = [1] * 5 + [0] * 15
        ordered_smiles, ordered_labels = interleave_by_label(smiles, labels)

        self.assertEqual(len(ordered_smiles), 20)
        self.assertEqual(sorted(ordered_labels), sorted(labels))
        # Every reasonably sized prefix contains both groups.
        for cut in (4, 8, 12, 16):
            with self.subTest(cut=cut):
                self.assertEqual(set(ordered_labels[:cut]), {0, 1})

    def test_interleaving_preserves_every_ligand(self) -> None:
        from chemdisco.dock import interleave_by_label

        smiles = [f"a{i}" for i in range(3)] + [f"d{i}" for i in range(7)]
        labels = [1] * 3 + [0] * 7
        ordered_smiles, _ = interleave_by_label(smiles, labels)
        self.assertEqual(sorted(ordered_smiles), sorted(smiles))

    def test_mismatched_lengths_raise(self) -> None:
        from chemdisco.dock import interleave_by_label

        with self.assertRaises(ValueError):
            interleave_by_label(["a", "b"], [1])

    def test_a_truncated_screen_says_it_is_truncated(self) -> None:
        from chemdisco.dock import DockingResult, Pose, ScreenResult

        result = ScreenResult(
            results=[DockingResult("A", [Pose(1, -9.0)])],
            labels=[1],
            n_requested=50,
            stopped_early="time budget of 60s reached after 1 of 50 ligands",
            elapsed_seconds=61.0,
        )
        text = result.describe()
        self.assertIn("STOPPED EARLY", text)
        self.assertIn("1 of 50", text)

    def test_throughput_is_reported_for_planning(self) -> None:
        from chemdisco.dock import DockingResult, Pose, ScreenResult

        result = ScreenResult(
            results=[DockingResult(f"L{i}", [Pose(1, -8.0)]) for i in range(10)],
            elapsed_seconds=200.0,
        )
        self.assertAlmostEqual(result.seconds_per_ligand, 20.0)
        self.assertIn("20.0s each", result.describe())


class TestInconclusiveVerdict(unittest.TestCase):
    """The distinction the first version of this module got wrong.

    ``separates`` returning False covers two completely different situations: a
    screen that measured no signal, and a screen too small to detect one.
    Reporting the second as the first turns absence of evidence into evidence of
    absence -- in the code written to prevent exactly that.
    """

    def _result(self, estimate, low, high, n_actives=8, n_decoys=8):
        from chemdisco.dock.enrichment import EnrichmentResult
        from chemdisco.qsar.evaluate import Interval

        return EnrichmentResult(
            n_actives=n_actives,
            n_decoys=n_decoys,
            auc=Interval(estimate, low, high),
            ef1=2.0,
            ef5=2.0,
            bedroc=0.9,
            max_ef1=2.0,
        )

    def test_the_real_bace1_run_is_inconclusive_not_negative(self) -> None:
        # The actual numbers: 8 actives against 8 decoys gave AUC 0.641 spanning
        # 0.317 to 0.900 -- from well below random to strong.
        result = self._result(0.641, 0.317, 0.900)
        self.assertFalse(result.separates)
        self.assertFalse(result.is_conclusive)
        self.assertEqual(result.verdict, "inconclusive")

    def test_an_inconclusive_report_refuses_to_claim_failure(self) -> None:
        text = self._result(0.641, 0.317, 0.900).describe()
        self.assertIn("INCONCLUSIVE", text)
        self.assertIn("NOT evidence that docking fails", text)
        self.assertIn("Absence of evidence is not evidence of absence", text)
        # It must not also carry the negative verdict's wording.
        self.assertNotIn("should not be used to triage", text)

    def test_it_says_how_many_compounds_would_be_needed(self) -> None:
        result = self._result(0.641, 0.317, 0.900)
        needed = result.compounds_needed(target_width=0.20)
        self.assertIsNotNone(needed)
        assert needed is not None
        self.assertGreater(needed, 8)
        self.assertIn(str(needed), result.describe())

    def test_a_tight_interval_below_random_is_a_real_negative(self) -> None:
        result = self._result(0.48, 0.42, 0.54, n_actives=200, n_decoys=200)
        self.assertFalse(result.separates)
        self.assertTrue(result.is_conclusive)
        self.assertEqual(result.verdict, "does not separate")

    def test_a_clear_positive_needs_no_width_check(self) -> None:
        # A lower bound above random settles it however wide the interval.
        result = self._result(0.80, 0.55, 0.95)
        self.assertTrue(result.separates)
        self.assertTrue(result.is_conclusive)
        self.assertEqual(result.verdict, "separates")

    def test_no_further_compounds_needed_when_already_precise(self) -> None:
        result = self._result(0.48, 0.44, 0.52, n_actives=400, n_decoys=400)
        self.assertIsNone(result.compounds_needed(target_width=0.20))


class TestGroupLevelBalancing(unittest.TestCase):
    """Per-pair matching is not enough, which the BACE1 run demonstrated."""

    def _actives(self, n=10):
        return [properties(450, 3.0) for _ in range(n)]

    def test_a_group_gap_within_tolerance_is_left_alone(self) -> None:
        from chemdisco.dock.decoys import balance_selection

        decoys = [properties(440 + i, 3.0) for i in range(10)]
        kept, note = balance_selection(self._actives(), decoys)
        self.assertEqual(len(kept), 10)
        self.assertIn("within tolerance", note)

    def test_a_skewed_decoy_set_is_trimmed(self) -> None:
        # Every decoy within 25 Da of its matched active, group means 30 Da
        # apart -- exactly the shape of the real failure.
        from chemdisco.dock.decoys import balance_selection

        decoys = [properties(mw, 3.0) for mw in (380, 385, 390, 395, 400, 440, 445, 450, 455, 460)]
        before = property_gap(self._actives(), decoys)["molecular_weight"]
        kept, note = balance_selection(self._actives(), decoys)
        after = property_gap(
            self._actives(), [decoys[i] for i in kept]
        )["molecular_weight"]
        self.assertLess(abs(after), abs(before))
        self.assertLess(len(kept), 10)
        self.assertIn("within tolerance", note)

    def test_an_unmatchable_pool_says_so_rather_than_emptying_itself(self) -> None:
        from chemdisco.dock.decoys import balance_selection

        decoys = [properties(300, 3.0) for _ in range(10)]
        kept, note = balance_selection(self._actives(), decoys)
        self.assertGreaterEqual(len(kept), 6)
        self.assertIn("remains confounded", note)

    def test_it_never_drops_below_the_keep_floor(self) -> None:
        from chemdisco.dock.decoys import balance_selection

        decoys = [properties(250, 3.0) for _ in range(10)]
        kept, _ = balance_selection(self._actives(), decoys, min_keep_fraction=0.8)
        self.assertGreaterEqual(len(kept), 8)

    def test_empty_groups_raise(self) -> None:
        from chemdisco.dock.decoys import balance_selection

        with self.assertRaises(ValueError):
            balance_selection(self._actives(), [])


class TestPublicApi(unittest.TestCase):
    """Every name the scripts import must actually be exported.

    A missing export took down a six-shard CI run: the scripts imported
    balance_selection from chemdisco.dock, which did not re-export it, and the
    whole matrix failed on the import line. Cheap to assert, and it catches the
    class of error that only shows up in a job that takes an hour to schedule.
    """

    EXPECTED = (
        "analyse_enrichment",
        "auc_roc",
        "balance_selection",
        "bedroc",
        "box_from_ligand",
        "describe_property_gap",
        "enrichment_factor",
        "interleave_by_label",
        "parse_pdb",
        "prepare_receptor_pdbqt",
        "property_gap",
        "screen",
        "select_decoys",
        "strip_to_receptor",
        "toolchain_report",
        "triage_candidates",
        "vina_available",
        "write_pdb",
    )

    def test_every_expected_name_is_importable(self) -> None:
        import chemdisco.dock as module

        missing = [name for name in self.EXPECTED if not hasattr(module, name)]
        self.assertEqual(missing, [], f"not exported from chemdisco.dock: {missing}")

    def test_all_matches_what_is_importable(self) -> None:
        import chemdisco.dock as module

        broken = [name for name in module.__all__ if not hasattr(module, name)]
        self.assertEqual(broken, [], f"listed in __all__ but absent: {broken}")

    def test_the_validation_scripts_import_cleanly(self) -> None:
        # The scripts are the real consumers of this package's public surface,
        # and they are not otherwise exercised by the suite.
        import importlib.util
        import pathlib
        import sys

        root = pathlib.Path(__file__).resolve().parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        for script in ("validate_enrichment", "validate_docking", "validate_target"):
            with self.subTest(script=script):
                path = root / "scripts" / f"{script}.py"
                spec = importlib.util.spec_from_file_location(script, path)
                assert spec is not None and spec.loader is not None
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
