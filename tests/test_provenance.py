"""Tests for the provenance primitive.

These tests encode the guarantees the rest of the package relies on. If any of
them fails, a number somewhere can lose track of where it came from, which is
the specific failure this project exists to prevent.

Written with ``unittest`` rather than pytest fixtures so the suite runs under
the standard library alone -- useful in constrained environments and in CI
before dependencies are installed.
"""

from __future__ import annotations

import math
import unittest

from chemdisco.provenance import Origin, ProvenanceError, Quantity, weakest


class TestOriginOrdering(unittest.TestCase):
    def test_trust_ranking_is_strict(self) -> None:
        self.assertLess(Origin.MEASURED.rank, Origin.DERIVED.rank)
        self.assertLess(Origin.DERIVED.rank, Origin.PREDICTED.rank)
        self.assertLess(Origin.PREDICTED.rank, Origin.HEURISTIC.rank)

    def test_weakest_picks_least_trustworthy(self) -> None:
        self.assertIs(
            weakest([Origin.MEASURED, Origin.PREDICTED, Origin.DERIVED]),
            Origin.PREDICTED,
        )
        self.assertIs(weakest([Origin.MEASURED]), Origin.MEASURED)

    def test_weakest_refuses_empty_rather_than_defaulting(self) -> None:
        # Returning a default origin here would be exactly the invented-value
        # behaviour the module forbids.
        with self.assertRaises(ValueError):
            weakest([])


class TestQuantityConstruction(unittest.TestCase):
    def test_measured_requires_a_source(self) -> None:
        with self.assertRaises(ProvenanceError):
            Quantity(value=1.0, unit="nM", origin=Origin.MEASURED, source="")
        with self.assertRaises(ProvenanceError):
            Quantity(value=1.0, unit="nM", origin=Origin.MEASURED, source="   ")

    def test_measured_with_source_is_accepted(self) -> None:
        q = Quantity.measured(5.0, "nM", "CHEMBL_ACT_1")
        self.assertEqual(q.value, 5.0)
        self.assertIs(q.origin, Origin.MEASURED)

    def test_origin_is_mandatory_and_typed(self) -> None:
        with self.assertRaises(TypeError):
            Quantity(value=1.0, unit=None)  # type: ignore[call-arg]
        with self.assertRaises(ProvenanceError):
            Quantity(value=1.0, unit=None, origin="measured")  # type: ignore[arg-type]

    def test_nan_and_inf_are_refused(self) -> None:
        # NaN propagates silently through arithmetic and comparisons, so it is
        # a worse representation of "unknown" than None.
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ProvenanceError):
                    Quantity.heuristic(bad, None, "rule")

    def test_booleans_are_not_numbers(self) -> None:
        with self.assertRaises(ProvenanceError):
            Quantity.heuristic(True, None, "rule")  # type: ignore[arg-type]

    def test_negative_uncertainty_refused(self) -> None:
        with self.assertRaises(ProvenanceError):
            Quantity.measured(5.0, "nM", "src", uncertainty=-0.1)

    def test_uncertainty_without_value_refused(self) -> None:
        with self.assertRaises(ProvenanceError):
            Quantity(
                value=None,
                unit=None,
                origin=Origin.DERIVED,
                source="x",
                uncertainty=0.5,
            )

    def test_in_domain_only_for_predictions(self) -> None:
        with self.assertRaises(ProvenanceError):
            Quantity(
                value=1.0,
                unit=None,
                origin=Origin.HEURISTIC,
                source="rule",
                in_domain=True,
            )
        # Permitted on a prediction.
        q = Quantity.predicted(1.0, None, "rf_v1", in_domain=False)
        self.assertFalse(q.in_domain)

    def test_quantity_is_immutable(self) -> None:
        q = Quantity.heuristic(1.0, None, "rule")
        with self.assertRaises(Exception):
            q.value = 2.0  # type: ignore[misc]


class TestUnknown(unittest.TestCase):
    def test_unknown_carries_a_reason(self) -> None:
        q = Quantity.unknown("kcal/mol", "no 3D structure available")
        self.assertIsNone(q.value)
        self.assertFalse(q.is_known)
        self.assertIn("no 3D structure", q.source)

    def test_unknown_renders_as_text_not_zero(self) -> None:
        # The predecessor returned 0.0 on failure, which sorts and plots as a
        # real result.
        self.assertEqual(Quantity.unknown(None, "failed").label(), "not computed")

    def test_require_raises_instead_of_substituting(self) -> None:
        with self.assertRaises(ProvenanceError):
            Quantity.unknown(None, "failed").require()

    def test_or_else_is_explicit_about_the_default(self) -> None:
        self.assertEqual(Quantity.unknown(None, "failed").or_else(-1.0), -1.0)
        self.assertEqual(Quantity.heuristic(3.0, None, "rule").or_else(-1.0), 3.0)


class TestRankingTrust(unittest.TestCase):
    def test_measured_may_rank(self) -> None:
        self.assertTrue(Quantity.measured(8.0, None, "src").is_trustworthy_for_ranking)

    def test_derived_may_rank(self) -> None:
        self.assertTrue(Quantity.derived(8.0, None, "conv").is_trustworthy_for_ranking)

    def test_prediction_in_domain_may_rank(self) -> None:
        q = Quantity.predicted(8.0, None, "rf_v1", in_domain=True)
        self.assertTrue(q.is_trustworthy_for_ranking)

    def test_prediction_out_of_domain_may_not_rank(self) -> None:
        q = Quantity.predicted(8.0, None, "rf_v1", in_domain=False)
        self.assertFalse(q.is_trustworthy_for_ranking)

    def test_prediction_with_unassessed_domain_may_not_rank(self) -> None:
        # An unchecked applicability domain is an unknown one. Treating it as
        # acceptable is how an extrapolated prediction reaches the top of a
        # candidate list.
        q = Quantity.predicted(8.0, None, "rf_v1")
        self.assertIsNone(q.in_domain)
        self.assertFalse(q.is_trustworthy_for_ranking)

    def test_heuristic_may_not_rank(self) -> None:
        self.assertFalse(Quantity.heuristic(8.0, None, "lipinski").is_trustworthy_for_ranking)

    def test_unknown_may_not_rank(self) -> None:
        self.assertFalse(Quantity.unknown(None, "failed").is_trustworthy_for_ranking)


class TestProvenanceLaundering(unittest.TestCase):
    """Provenance must degrade through a pipeline, never improve."""

    def test_mapping_a_prediction_cannot_make_it_measured(self) -> None:
        predicted = Quantity.predicted(9.0, None, "rf_v1", in_domain=True)
        converted = predicted.map_value(
            lambda v: v * 2, unit=None, source="doubling", origin=Origin.MEASURED
        )
        self.assertIs(converted.origin, Origin.PREDICTED)
        self.assertEqual(converted.value, 18.0)

    def test_mapping_a_measurement_through_a_model_degrades_it(self) -> None:
        measured = Quantity.measured(100.0, "nM", "CHEMBL_ACT_1")
        modelled = measured.map_value(
            lambda v: v / 10, unit="nM", source="correction model", origin=Origin.PREDICTED
        )
        self.assertIs(modelled.origin, Origin.PREDICTED)

    def test_mapping_preserves_the_citation_chain(self) -> None:
        measured = Quantity.measured(100.0, "nM", "CHEMBL_ACT_1")
        derived = measured.map_value(
            lambda v: -math.log10(v * 1e-9),
            unit=None,
            source="pIC50 conversion",
            origin=Origin.DERIVED,
        )
        self.assertIn("CHEMBL_ACT_1", derived.source)
        self.assertIn("pIC50 conversion", derived.source)
        self.assertAlmostEqual(derived.value or 0.0, 7.0, places=9)

    def test_unknown_maps_to_unknown_without_calling_the_function(self) -> None:
        calls: list[float] = []

        def record(v: float) -> float:
            calls.append(v)
            return v

        result = Quantity.unknown(None, "no structure").map_value(
            record, unit=None, source="transform"
        )
        self.assertEqual(calls, [])
        self.assertIsNone(result.value)
        self.assertIn("propagated unknown", result.notes)

    def test_unknown_mapped_from_measured_does_not_claim_measurement(self) -> None:
        # A MEASURED quantity with no value would violate the source invariant
        # on reconstruction; the mapping must degrade it to DERIVED.
        measured_unknown = Quantity(
            value=None, unit="nM", origin=Origin.MEASURED, source="CHEMBL_ACT_1"
        )
        mapped = measured_unknown.map_value(lambda v: v, unit=None, source="conv")
        self.assertIs(mapped.origin, Origin.DERIVED)


class TestRendering(unittest.TestCase):
    def test_label_always_discloses_origin(self) -> None:
        for quantity in (
            Quantity.measured(1.0, "nM", "src"),
            Quantity.derived(1.0, None, "conv"),
            Quantity.predicted(1.0, None, "rf", in_domain=True),
            Quantity.heuristic(1.0, None, "rule"),
        ):
            with self.subTest(origin=quantity.origin):
                self.assertIn(quantity.origin.value, quantity.label())

    def test_label_shouts_about_extrapolation(self) -> None:
        q = Quantity.predicted(9.5, None, "rf_v1", in_domain=False)
        self.assertIn("OUT OF DOMAIN", q.label())

    def test_label_includes_uncertainty_and_unit(self) -> None:
        q = Quantity.measured(5.25, "kcal/mol", "src", uncertainty=0.5)
        label = q.label(digits=2)
        self.assertIn("5.25", label)
        self.assertIn("0.50", label)
        self.assertIn("kcal/mol", label)


class TestSerialisation(unittest.TestCase):
    def test_roundtrip_preserves_everything(self) -> None:
        original = Quantity.predicted(
            7.5,
            None,
            "rf_v1",
            uncertainty=0.4,
            in_domain=False,
            notes=["extrapolated", "ECFP4"],
        )
        restored = Quantity.from_dict(original.to_dict())
        self.assertEqual(original, restored)

    def test_export_carries_provenance_not_a_bare_float(self) -> None:
        payload = Quantity.heuristic(2.5, None, "SAscore").to_dict()
        self.assertEqual(payload["origin"], "heuristic")
        self.assertEqual(payload["source"], "SAscore")


if __name__ == "__main__":
    unittest.main()
