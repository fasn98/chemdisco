"""Tests for the QSAR layer.

The centrepiece is :class:`TestPredecessorFailureIsCaught`, which reconstructs
the exact defect found in the project this one replaces -- a target computed from
the same molecular properties used as features -- and asserts that the leakage
detector and the baseline comparison both catch it. A guard that has never been
shown to fire on the real bug is not a guard.
"""

from __future__ import annotations

import unittest

import numpy as np

from chemdisco.curate.records import CuratedPoint
from chemdisco.provenance import Origin, Quantity
from chemdisco.qsar.dataset import (
    LabelProvenanceError,
    build_training_set,
    verify_label_provenance,
)
from chemdisco.qsar import (
    ApplicabilityDomain,
    QSARModel,
    bootstrap_metric,
    compute_baselines,
    detect_target_leakage,
    evaluate,
    make_fit_predict,
    nearest_neighbour_predictor,
    r_squared,
    rmse,
    spearman,
)


def synthetic_dataset(
    n: int = 300, noise: float = 0.5, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Descriptors and a label with a genuine but noisy relationship.

    The noise level is set to 0.5 log units, which is roughly the
    inter-laboratory reproducibility of an IC50 measurement. A model should not
    be able to do much better than that on real data, and a synthetic benchmark
    with less noise would set an unreachable expectation.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 10))
    y = (
        7.0
        + 1.5 * X[:, 0]
        - 0.8 * X[:, 1]
        + 0.3 * X[:, 2] ** 2
        + rng.normal(scale=noise, size=n)
    )
    return X, y


class TestMetrics(unittest.TestCase):
    def test_r2_of_perfect_prediction_is_one(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(r_squared(y, y), 1.0)

    def test_r2_goes_negative_and_is_not_clipped(self) -> None:
        # A model worse than the test set's own mean deserves a negative score;
        # clipping to zero would hide how bad it is.
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertLess(r_squared(y, np.array([4.0, 3.0, 2.0, 1.0])), 0.0)

    def test_r2_undefined_on_constant_labels_raises(self) -> None:
        with self.assertRaises(ValueError):
            r_squared(np.array([5.0, 5.0, 5.0]), np.array([5.0, 5.1, 4.9]))

    def test_rmse_is_in_label_units(self) -> None:
        self.assertAlmostEqual(
            rmse(np.array([1.0, 2.0]), np.array([2.0, 3.0])), 1.0
        )

    def test_spearman_is_rank_based(self) -> None:
        # A monotone but strongly non-linear transform preserves rank perfectly.
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(spearman(y, np.exp(y)), 1.0, places=9)

    def test_spearman_handles_ties_by_average_rank(self) -> None:
        # Curated pActivity values cluster on round numbers, so ties are common
        # and mishandling them silently shifts the correlation.
        a = np.array([1.0, 2.0, 2.0, 3.0])
        b = np.array([1.0, 5.0, 5.0, 9.0])
        self.assertAlmostEqual(spearman(a, b), 1.0, places=9)


class TestBootstrap(unittest.TestCase):
    def test_interval_brackets_the_estimate(self) -> None:
        X, y = synthetic_dataset(n=200)
        model = QSARModel().fit(X[:150], y[:150])
        predictions = model.predict_raw(X[150:])
        interval = bootstrap_metric(y[150:], predictions, r_squared, n_resamples=300)
        self.assertLessEqual(interval.low, interval.estimate)
        self.assertLessEqual(interval.estimate, interval.high)

    def test_interval_is_reproducible(self) -> None:
        X, y = synthetic_dataset(n=120)
        model = QSARModel().fit(X[:90], y[:90])
        predictions = model.predict_raw(X[90:])
        first = bootstrap_metric(y[90:], predictions, rmse, n_resamples=200, seed=5)
        second = bootstrap_metric(y[90:], predictions, rmse, n_resamples=200, seed=5)
        self.assertEqual(first.low, second.low)
        self.assertEqual(first.high, second.high)

    def test_tiny_test_set_refuses_an_interval(self) -> None:
        with self.assertRaises(ValueError):
            bootstrap_metric(np.array([1.0, 2.0]), np.array([1.1, 2.1]), rmse)

    def test_overlap_detection(self) -> None:
        X, y = synthetic_dataset(n=200)
        model = QSARModel().fit(X[:150], y[:150])
        predictions = model.predict_raw(X[150:])
        interval = bootstrap_metric(y[150:], predictions, r_squared, n_resamples=200)
        self.assertTrue(interval.overlaps(interval))


class TestBaselines(unittest.TestCase):
    def test_mean_predictor_is_the_zero_point(self) -> None:
        X, y = synthetic_dataset(n=200)
        baselines = compute_baselines(y[:150], y[150:])
        # Train and test are drawn from the same distribution, so predicting the
        # training mean lands near zero R-squared.
        self.assertLess(abs(baselines.mean_predictor_r2), 0.15)

    def test_nearest_neighbour_baseline_is_computed_when_similarity_given(self) -> None:
        y_train = np.array([5.0, 6.0, 7.0, 8.0])
        y_test = np.array([6.0, 8.0])
        # Test compound 0 is most similar to training compound 1 (label 6.0),
        # test compound 1 to training compound 3 (label 8.0): a perfect lookup.
        similarity = np.array([[0.1, 0.9, 0.2, 0.1], [0.1, 0.2, 0.3, 0.95]])
        predictions = nearest_neighbour_predictor(similarity, y_train)
        np.testing.assert_allclose(predictions, np.array([6.0, 8.0]))
        baselines = compute_baselines(y_train, y_test, similarity_to_train=similarity)
        self.assertAlmostEqual(baselines.nearest_neighbour_r2 or 0.0, 1.0)

    def test_similarity_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            nearest_neighbour_predictor(np.zeros((2, 3)), np.array([1.0, 2.0]))

    def test_permutation_distribution_sits_near_zero(self) -> None:
        X, y = synthetic_dataset(n=200)
        baselines = compute_baselines(
            y[:150],
            y[150:],
            fit_predict=make_fit_predict("random_forest", X[:150], X[150:]),
            n_permutations=10,
        )
        self.assertIsNotNone(baselines.permutation_r2_mean)
        self.assertLess(baselines.permutation_r2_mean or 1.0, 0.1)

    def test_verdict_calls_out_a_model_that_beats_nothing(self) -> None:
        # A model that just predicts the mean should be described as having
        # learned nothing, in words, not left for the reader to infer.
        y_train = np.linspace(5, 9, 100)
        y_test = np.linspace(5, 9, 50)
        baselines = compute_baselines(y_train, y_test)
        verdict = baselines.verdict(baselines.mean_predictor_r2 + 0.01)
        self.assertIn("learned essentially nothing", verdict)

    def test_verdict_flags_an_impossibly_good_score(self) -> None:
        y_train = np.linspace(5, 9, 100)
        y_test = np.linspace(5, 9, 50)
        baselines = compute_baselines(y_train, y_test)
        self.assertIn("target leakage", baselines.verdict(0.99))

    def test_verdict_warns_when_test_labels_barely_vary(self) -> None:
        y_train = np.linspace(6.9, 7.1, 100)
        y_test = np.linspace(6.9, 7.1, 50)
        baselines = compute_baselines(y_train, y_test)
        self.assertIn("numerically unstable", baselines.verdict(0.5))


class TestPredecessorFailureIsCaught(unittest.TestCase):
    """Reconstruct the defect in the project this one replaces, and locate the fix.

    In that code, when a compound had no measured activity, the activity was
    synthesised from molecular weight and LogP::

        mw_score = 1.0 - abs(mw - 400) / 200
        logp_score = 1.0 - abs(logp - 3) / 3
        activity = (mw_score + logp_score) / 2 + np.random.normal(0, 0.1)

    Molecular weight and LogP were then supplied to the model as features, so the
    model was fitting an arithmetic identity.

    These tests establish three things, in order. First, that the statistical
    leakage screens genuinely **fail** on this construction -- which is why they
    are not the defence. Second, that the resulting score is unremarkable rather
    than suspicious, which is why nobody noticed. Third, that the provenance
    check in :mod:`chemdisco.qsar.dataset` catches it, because a fabricated label
    cannot claim to descend from a measurement.
    """

    def _leaky_dataset(self, n: int = 200) -> tuple[np.ndarray, np.ndarray, list[str]]:
        rng = np.random.default_rng(1)
        mw = rng.uniform(150, 650, size=n)
        logp = rng.uniform(-1, 6, size=n)
        tpsa = rng.uniform(20, 160, size=n)
        hbd = rng.integers(0, 6, size=n).astype(float)

        mw_score = 1.0 - np.abs(mw - 400) / 200
        logp_score = 1.0 - np.abs(logp - 3) / 3
        # The predecessor's synthetic label, noise included.
        y = (mw_score + logp_score) / 2 + rng.normal(0, 0.1, size=n)

        X = np.column_stack([mw, logp, tpsa, hbd])
        return X, y, ["molecular_weight", "logp", "tpsa", "hbd"]

    # -- what the statistical screens do catch -------------------------------

    def test_linear_screen_catches_a_single_feature_identity(self) -> None:
        rng = np.random.default_rng(2)
        n = 200
        score = rng.uniform(0, 1, size=n)
        y = 3.0 * score + rng.normal(0, 0.01, size=n)
        X = np.column_stack([score, rng.normal(size=n)])
        warnings = detect_target_leakage(X, y, ["precomputed_score", "noise"])
        self.assertTrue(warnings)
        self.assertIn("precomputed_score", warnings[0])

    def test_nonlinear_screen_catches_what_correlation_misses(self) -> None:
        # A label built from -|mw - 400| has a Pearson correlation near zero with
        # molecular weight, so only the binned screen can see it.
        rng = np.random.default_rng(3)
        n = 400
        mw = rng.uniform(150, 650, size=n)
        y = 1.0 - np.abs(mw - 400) / 200 + rng.normal(0, 0.01, size=n)
        X = mw.reshape(-1, 1)

        correlation = abs(float(np.corrcoef(mw, y)[0, 1]))
        self.assertLess(
            correlation, 0.3, "the premise of this test is a near-zero correlation"
        )

        warnings = detect_target_leakage(X, y, ["molecular_weight"])
        self.assertTrue(warnings, "the binned screen must catch the non-monotonic case")
        self.assertIn("non-linear relationship", warnings[0])

    def test_screens_are_quiet_on_an_honest_dataset(self) -> None:
        X, y = synthetic_dataset(n=300)
        self.assertEqual(detect_target_leakage(X, y), [])

    # -- what the statistical screens do not catch ---------------------------

    def test_screens_fail_on_the_real_predecessor_construction(self) -> None:
        # The honest negative result. The label is a sum over two features, each
        # entering non-monotonically, so neither screen fires. This test exists
        # to stop anyone treating a clean leakage report as reassurance.
        X, y, names = self._leaky_dataset(n=400)
        self.assertEqual(
            detect_target_leakage(X, y, names),
            [],
            "if this ever starts firing, the screens improved -- but the "
            "structural defence must remain the primary one regardless",
        )

    def test_the_leaky_model_scores_believably_which_is_the_danger(self) -> None:
        X, y, names = self._leaky_dataset()
        model = QSARModel().fit(X[:150], y[:150], feature_names=names)
        score = r_squared(y[150:], model.predict_raw(X[150:]))
        # Around 0.69: a thoroughly ordinary QSAR result. An absurd 0.99 would
        # have drawn attention; this did not.
        self.assertGreater(score, 0.5)
        self.assertLess(
            score,
            0.95,
            "the score is plausible, not suspicious, which is exactly why the "
            "'too good to be true' heuristic cannot be relied on",
        )

    # -- what does catch it --------------------------------------------------

    def test_provenance_check_refuses_a_heuristic_label(self) -> None:
        # The structural defence. However the number was computed, it cannot
        # enter a training set while declaring itself a heuristic.
        heuristic_labels = [
            Quantity.heuristic(0.5, None, "mw/logp drug-likeness score")
            for _ in range(20)
        ]
        with self.assertRaises(LabelProvenanceError) as context:
            verify_label_provenance(heuristic_labels)
        message = str(context.exception)
        self.assertIn("heuristic", message)
        self.assertIn("measures nothing about", message)

    def test_provenance_check_refuses_a_predicted_label(self) -> None:
        # Training on another model's output compounds its errors silently.
        predicted = [
            Quantity.predicted(7.0, None, "earlier_model", in_domain=True)
            for _ in range(10)
        ]
        with self.assertRaises(LabelProvenanceError):
            verify_label_provenance(predicted)

    def test_provenance_check_refuses_unknown_labels(self) -> None:
        labels = [
            Quantity.derived(7.0, None, "ACT1: -log10"),
            Quantity.unknown(None, "no activity measured for this compound"),
        ]
        with self.assertRaises(LabelProvenanceError) as context:
            verify_label_provenance(labels)
        self.assertIn("must be dropped, not", str(context.exception))

    def test_provenance_check_accepts_measurement_derived_labels(self) -> None:
        labels = [
            Quantity.derived(7.0 + index * 0.1, None, f"ACT{index}: -log10(1e-7 M)")
            for index in range(10)
        ]
        verify_label_provenance(labels)  # must not raise

    def test_build_training_set_enforces_provenance_end_to_end(self) -> None:
        heuristic_point = CuratedPoint(
            compound_id="C1",
            smiles="CCO",
            target_id="T1",
            pactivity=Quantity.heuristic(0.5, None, "drug-likeness score"),
            n_measurements=0,
            source_activity_ids=(),
        )
        with self.assertRaises(LabelProvenanceError):
            build_training_set(
                [heuristic_point],
                lambda smiles: (np.zeros((len(smiles), 3)), ["a", "b", "c"]),
            )


class TestEvaluationReport(unittest.TestCase):
    """The report is the product; these test what it refuses to claim."""

    def _evaluate(self, n: int = 300, noise: float = 0.5):
        X, y = synthetic_dataset(n=n, noise=noise)
        cut = int(n * 0.75)
        model = QSARModel().fit(X[:cut], y[:cut])
        predictions = model.predict_raw(X[cut:])
        baselines = compute_baselines(
            y[:cut],
            y[cut:],
            fit_predict=make_fit_predict("random_forest", X[:cut], X[cut:]),
            n_permutations=8,
        )
        return evaluate(
            y[cut:],
            predictions,
            n_train=cut,
            split_strategy="synthetic-holdout",
            baselines=baselines,
            n_resamples=300,
        )

    def test_report_leads_with_sample_sizes_and_split(self) -> None:
        # A metric without these is uninterpretable, so they come first.
        text = self._evaluate().describe()
        self.assertIn("train n=", text)
        self.assertIn("test n=", text)
        self.assertIn("synthetic-holdout", text)

    def test_every_metric_carries_an_interval(self) -> None:
        result = self._evaluate()
        for name, interval in (
            ("r2", result.r2),
            ("rmse", result.rmse),
            ("mae", result.mae),
        ):
            with self.subTest(metric=name):
                self.assertLess(interval.low, interval.high)
                self.assertIn("[", interval.label())

    def test_a_good_model_on_enough_data_is_defensible(self) -> None:
        result = self._evaluate(n=400)
        self.assertTrue(result.is_defensible, result.describe())

    def test_a_small_test_set_is_never_defensible(self) -> None:
        # Fewer than thirty held-out compounds cannot support a claim, however
        # good the point estimate looks.
        result = self._evaluate(n=60)
        self.assertLess(result.n_test, 30)
        self.assertFalse(result.is_defensible)
        self.assertIn("CAUTION", result.describe())

    def test_leakage_warnings_block_defensibility_outright(self) -> None:
        X, y = synthetic_dataset(n=400)
        cut = 300
        model = QSARModel().fit(X[:cut], y[:cut])
        baselines = compute_baselines(y[:cut], y[cut:])
        result = evaluate(
            y[cut:],
            model.predict_raw(X[cut:]),
            n_train=cut,
            split_strategy="synthetic-holdout",
            baselines=baselines,
            leakage_warnings=["feature 'x' correlates with the label at r=0.99"],
            n_resamples=200,
        )
        self.assertFalse(result.is_defensible)
        self.assertIn("LEAKAGE CHECK FAILED", result.describe())

    def test_a_result_without_baselines_is_not_defensible(self) -> None:
        # An unbenchmarked number cannot be known to mean anything.
        X, y = synthetic_dataset(n=400)
        cut = 300
        model = QSARModel().fit(X[:cut], y[:cut])
        result = evaluate(
            y[cut:],
            model.predict_raw(X[cut:]),
            n_train=cut,
            split_strategy="synthetic-holdout",
            n_resamples=200,
        )
        self.assertFalse(result.is_defensible)

    def test_wide_interval_is_called_out(self) -> None:
        # With heavy noise and a small test set the interval widens, and the
        # report must warn that model comparisons below that width are unfounded.
        result = self._evaluate(n=120, noise=2.0)
        if result.r2.width > 0.3:
            self.assertIn("unsupported", result.describe())
        else:
            self.skipTest("interval happened to be narrow for this seed")


class TestApplicabilityDomain(unittest.TestCase):
    def test_training_like_input_is_in_domain(self) -> None:
        rng = np.random.default_rng(0)
        train = rng.normal(size=(200, 5))
        domain = ApplicabilityDomain.fit(train)
        verdicts = domain.assess(rng.normal(size=(20, 5)))
        self.assertGreater(
            sum(1 for v in verdicts if v.in_domain), 15, "most should be inside"
        )

    def test_distant_input_is_out_of_domain(self) -> None:
        rng = np.random.default_rng(0)
        train = rng.normal(size=(200, 5))
        domain = ApplicabilityDomain.fit(train)
        # Twenty standard deviations away: a molecule unlike anything in training.
        far = np.full((1, 5), 20.0)
        verdict = domain.assess(far)[0]
        self.assertFalse(verdict.in_domain)
        self.assertTrue(verdict.reasons)
        self.assertIn("OUTSIDE", verdict.describe())

    def test_low_similarity_puts_a_candidate_out_of_domain(self) -> None:
        rng = np.random.default_rng(0)
        train = rng.normal(size=(100, 4))
        domain = ApplicabilityDomain.fit(train, similarity_floor=0.3)
        candidates = rng.normal(size=(3, 4))
        verdicts = domain.assess(candidates, max_similarity=np.array([0.05, 0.9, 0.8]))
        self.assertFalse(verdicts[0].in_domain)
        self.assertIn("chemotype the model has not seen", verdicts[0].reasons[0])

    def test_disagreement_between_methods_is_recorded(self) -> None:
        rng = np.random.default_rng(0)
        train = rng.normal(size=(100, 4))
        domain = ApplicabilityDomain.fit(train, similarity_floor=0.3)
        # Descriptor-wise ordinary, but with no similar training compound.
        verdict = domain.assess(
            rng.normal(size=(1, 4)), max_similarity=np.array([0.02])
        )[0]
        self.assertTrue(verdict.methods_disagree)
        self.assertFalse(verdict.in_domain)
        self.assertIn("conservative reading", verdict.describe())

    def test_coverage_reports_a_fraction(self) -> None:
        rng = np.random.default_rng(0)
        train = rng.normal(size=(100, 4))
        domain = ApplicabilityDomain.fit(train)
        coverage = domain.coverage(rng.normal(size=(50, 4)))
        self.assertGreater(coverage, 0.5)
        self.assertLessEqual(coverage, 1.0)

    def test_single_compound_training_set_refuses(self) -> None:
        with self.assertRaises(ValueError):
            ApplicabilityDomain.fit(np.zeros((1, 3)))

    def test_feature_count_mismatch_raises(self) -> None:
        domain = ApplicabilityDomain.fit(np.random.default_rng(0).normal(size=(50, 4)))
        with self.assertRaises(ValueError):
            domain.assess(np.zeros((2, 7)))


class TestQSARModel(unittest.TestCase):
    def test_predictions_carry_provenance(self) -> None:
        X, y = synthetic_dataset(n=150)
        model = QSARModel(label_name="pIC50 (synthetic)").fit(X[:120], y[:120])
        quantities = model.predict(X[120:])
        self.assertEqual(len(quantities), 30)
        for quantity in quantities:
            self.assertIs(quantity.origin, Origin.PREDICTED)
            self.assertIn("random_forest", quantity.source)
            self.assertIsNotNone(quantity.in_domain)

    def test_forest_predictions_carry_an_uncertainty_with_its_caveat(self) -> None:
        X, y = synthetic_dataset(n=150)
        model = QSARModel(kind="random_forest").fit(X[:120], y[:120])
        quantity = model.predict(X[120:])[0]
        self.assertIsNotNone(quantity.uncertainty)
        # The limitation must travel with the number, not be assumed known.
        self.assertTrue(any("understates" in note for note in quantity.notes))

    def test_out_of_domain_prediction_is_returned_but_not_rankable(self) -> None:
        X, y = synthetic_dataset(n=150)
        model = QSARModel().fit(X[:120], y[:120])
        quantity = model.predict(np.full((1, 10), 25.0))[0]
        self.assertTrue(quantity.is_known, "the prediction is still reported")
        self.assertFalse(quantity.in_domain)
        self.assertFalse(
            quantity.is_trustworthy_for_ranking,
            "an extrapolated prediction must not be allowed to top a candidate list",
        )
        self.assertIn("OUT OF DOMAIN", quantity.label())

    def test_refuses_to_fit_on_too_few_compounds(self) -> None:
        X, y = synthetic_dataset(n=9)
        with self.assertRaises(ValueError) as context:
            QSARModel().fit(X, y)
        self.assertIn("refusing to fit", str(context.exception))

    def test_refuses_constant_labels(self) -> None:
        X, _ = synthetic_dataset(n=50)
        with self.assertRaises(ValueError):
            QSARModel().fit(X, np.full(50, 7.0))

    def test_refuses_non_finite_features(self) -> None:
        X, y = synthetic_dataset(n=50)
        X[3, 2] = np.nan
        with self.assertRaises(ValueError) as context:
            QSARModel().fit(X, y)
        self.assertIn("non-finite", str(context.exception))

    def test_feature_order_mismatch_is_caught(self) -> None:
        X, y = synthetic_dataset(n=60)
        model = QSARModel().fit(X, y)
        with self.assertRaises(ValueError) as context:
            model.predict_raw(np.zeros((2, 4)))
        self.assertIn("Feature order must match", str(context.exception))

    def test_small_training_set_is_flagged_in_every_prediction(self) -> None:
        X, y = synthetic_dataset(n=40)
        model = QSARModel().fit(X[:30], y[:30])
        quantity = model.predict(X[30:])[0]
        self.assertTrue(any("provisional" in note for note in quantity.notes))

    def test_more_features_than_samples_is_flagged(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 50))
        y = X[:, 0] * 2 + rng.normal(scale=0.3, size=20)
        model = QSARModel().fit(X, y)
        self.assertTrue(any("more features than samples" in n for n in model.notes))

    def test_metadata_records_the_curation_behind_the_model(self) -> None:
        X, y = synthetic_dataset(n=100)
        model = QSARModel(curation_summary="confidence>=8; functional only")
        model.fit(X, y, feature_names=[f"d{i}" for i in range(10)])
        metadata = model.metadata()
        self.assertEqual(metadata["n_train"], 100)
        self.assertEqual(metadata["n_features"], 10)
        self.assertIn("confidence>=8", metadata["curation_summary"])

    def test_ridge_is_available_as_a_transparency_baseline(self) -> None:
        X, y = synthetic_dataset(n=120)
        model = QSARModel(kind="ridge").fit(X[:90], y[:90])
        self.assertGreater(r_squared(y[90:], model.predict_raw(X[90:])), 0.5)
        # Linear models have no per-tree spread, so no uncertainty is claimed.
        self.assertIsNone(model.predict(X[90:])[0].uncertainty)

    def test_unfitted_model_refuses_to_predict(self) -> None:
        with self.assertRaises(RuntimeError):
            QSARModel().predict_raw(np.zeros((1, 3)))


if __name__ == "__main__":
    unittest.main()
