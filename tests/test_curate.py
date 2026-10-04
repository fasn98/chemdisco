"""Tests for curation: filtering and replicate aggregation.

The accounting invariant is the important one: every input measurement must end
up either inside a curated point or named in a rejection. A curation step that
can lose records silently cannot be audited, and an unauditable dataset makes
every metric computed from it unreviewable.
"""

from __future__ import annotations

import unittest

from chemdisco.curate import (
    PERMISSIVE_POLICY,
    ActivityRecord,
    AggregationPolicy,
    CurationPolicy,
    aggregate_measurements,
    curate,
    family_of,
    filter_records,
)


def record(
    activity_id: str = "ACT1",
    compound_id: str = "CHEMBL1",
    *,
    smiles: str = "CCO",
    target_id: str = "CHEMBL_T1",
    activity_type: str = "IC50",
    value: float | None = 100.0,
    unit: str | None = "nM",
    relation: str = "=",
    assay_type: str = "B",
    confidence_score: int | None = 9,
    data_validity_comment: str | None = None,
    potential_duplicate: bool = False,
    pchembl_value: float | None = None,
    molecular_weight: float | None = None,
) -> ActivityRecord:
    """A record that passes the default policy, so each test varies one field."""
    return ActivityRecord(
        activity_id=activity_id,
        compound_id=compound_id,
        smiles=smiles,
        target_id=target_id,
        activity_type=activity_type,
        value=value,
        unit=unit,
        relation=relation,
        assay_type=assay_type,
        confidence_score=confidence_score,
        data_validity_comment=data_validity_comment,
        potential_duplicate=potential_duplicate,
        pchembl_value=pchembl_value,
        molecular_weight=molecular_weight,
    )


class TestFamilies(unittest.TestCase):
    def test_binding_and_functional_are_separate(self) -> None:
        self.assertEqual(family_of("Ki"), "binding")
        self.assertEqual(family_of("IC50"), "functional")
        self.assertNotEqual(family_of("Ki"), family_of("IC50"))

    def test_unknown_type_has_no_family(self) -> None:
        self.assertIsNone(family_of("Inhibition"))


class TestFilters(unittest.TestCase):
    def test_a_clean_record_survives(self) -> None:
        outcome = filter_records([record()])
        self.assertEqual(len(outcome.kept), 1)
        self.assertAlmostEqual(outcome.kept[0][1], 7.0, places=9)

    def test_low_confidence_is_rejected(self) -> None:
        # Confidence 5 may be a measurement against a protein family or complex
        # rather than the requested protein.
        outcome = filter_records([record(confidence_score=5)])
        self.assertEqual(len(outcome.kept), 0)
        self.assertEqual(outcome.rejected[0].rule, "low_target_confidence")

    def test_missing_confidence_is_rejected_by_default(self) -> None:
        outcome = filter_records([record(confidence_score=None)])
        self.assertEqual(outcome.rejected[0].rule, "low_target_confidence")

    def test_admet_assay_is_rejected(self) -> None:
        # An ADMET readout must not share a regression target with binding data.
        outcome = filter_records([record(assay_type="A")])
        self.assertEqual(outcome.rejected[0].rule, "wrong_assay_type")

    def test_binding_constant_rejected_under_functional_only_policy(self) -> None:
        # The default policy keeps the functional family; a Ki is a different
        # physical quantity and pooling the two needs an explicit decision.
        outcome = filter_records([record(activity_type="Ki")])
        self.assertEqual(outcome.rejected[0].rule, "incommensurable_endpoint")

    def test_pooling_families_requires_asking_for_it(self) -> None:
        policy = CurationPolicy(allowed_families=frozenset({"functional", "binding"}))
        outcome = filter_records([record(activity_type="Ki")], policy)
        self.assertEqual(len(outcome.kept), 1)

    def test_unknown_activity_type_is_rejected(self) -> None:
        outcome = filter_records([record(activity_type="Inhibition", unit="%")])
        self.assertEqual(outcome.rejected[0].rule, "incommensurable_endpoint")

    def test_source_validity_flag_is_honoured(self) -> None:
        outcome = filter_records(
            [record(data_validity_comment="Potential transcription error")]
        )
        self.assertEqual(outcome.rejected[0].rule, "source_flagged_invalid")
        self.assertIn("transcription error", outcome.rejected[0].detail)

    def test_potential_duplicate_is_dropped(self) -> None:
        outcome = filter_records([record(potential_duplicate=True)])
        self.assertEqual(outcome.rejected[0].rule, "potential_duplicate")

    def test_censored_relation_is_dropped(self) -> None:
        outcome = filter_records([record(relation=">")])
        self.assertEqual(outcome.rejected[0].rule, "censored_measurement")

    def test_missing_structure_is_dropped(self) -> None:
        outcome = filter_records([record(smiles="")])
        self.assertEqual(outcome.rejected[0].rule, "missing_structure")

    def test_unconvertible_unit_is_dropped_with_the_reason(self) -> None:
        outcome = filter_records([record(unit="%", activity_type="IC50")])
        self.assertEqual(outcome.rejected[0].rule, "unconvertible_value")
        self.assertIn("not a concentration", outcome.rejected[0].detail)

    def test_implausible_pactivity_is_dropped(self) -> None:
        outcome = filter_records([record(value=1.0, unit="fM")])
        self.assertEqual(outcome.rejected[0].rule, "implausible_pactivity")

    def test_pchembl_cross_check_catches_a_bad_conversion(self) -> None:
        # Our conversion of 100 nM gives 7.0. A source claiming 4.0 means one of
        # the two is wrong, and we cannot tell which, so the record goes.
        outcome = filter_records([record(pchembl_value=4.0)])
        self.assertEqual(outcome.rejected[0].rule, "pchembl_disagreement")

    def test_pchembl_cross_check_passes_when_they_agree(self) -> None:
        outcome = filter_records([record(pchembl_value=7.0)])
        self.assertEqual(len(outcome.kept), 1)

    def test_permissive_policy_keeps_structurally_usable_records(self) -> None:
        outcome = filter_records(
            [record(confidence_score=3, assay_type="A", activity_type="Ki")],
            PERMISSIVE_POLICY,
        )
        self.assertEqual(len(outcome.kept), 1)

    def test_every_record_is_accounted_for(self) -> None:
        records = [
            record("A1"),
            record("A2", confidence_score=4),
            record("A3", unit="%"),
            record("A4", relation=">"),
            record("A5", potential_duplicate=True),
        ]
        outcome = filter_records(records)
        self.assertEqual(outcome.n_total, len(records))

    def test_every_rejection_states_a_rule_and_a_reason(self) -> None:
        records = [
            record("A2", confidence_score=4),
            record("A3", unit="%"),
            record("A4", relation=">"),
            record("A5", smiles=""),
            record("A6", data_validity_comment="Outside typical range"),
        ]
        outcome = filter_records(records)
        for rejection in outcome.rejected:
            with self.subTest(activity=rejection.record.activity_id):
                self.assertTrue(rejection.rule)
                self.assertGreater(len(rejection.detail), 10)


class TestAggregation(unittest.TestCase):
    def test_single_measurement_passes_through(self) -> None:
        kept, rejected = aggregate_measurements([(record(), 7.0)])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].n_measurements, 1)
        self.assertEqual(rejected, [])
        self.assertIsNone(kept[0].spread_log_units)

    def test_replicates_are_combined_by_median(self) -> None:
        measurements = [
            (record("A1"), 7.0),
            (record("A2"), 7.2),
            (record("A3"), 7.4),
        ]
        kept, _ = aggregate_measurements(measurements)
        self.assertEqual(len(kept), 1)
        self.assertAlmostEqual(kept[0].pactivity.require(), 7.2)
        self.assertEqual(kept[0].n_measurements, 3)

    def test_median_resists_a_single_discrepant_value(self) -> None:
        # Three replicates near 7.1 and one at 7.9. The mean is 7.30, pulled up
        # by the outlier; the median is 7.15, which is what the bulk of the
        # measurements actually say. The spread is 0.9 log units, inside the
        # tolerance, so the group is kept rather than discarded -- this is the
        # case where the choice of statistic decides the label.
        measurements = [
            (record("A1"), 7.0),
            (record("A2"), 7.1),
            (record("A3"), 7.2),
            (record("A4"), 7.9),
        ]
        kept, _ = aggregate_measurements(measurements)
        self.assertAlmostEqual(kept[0].pactivity.require(), 7.15, places=6)

        mean_policy = AggregationPolicy(statistic="mean")
        kept_mean, _ = aggregate_measurements(measurements, mean_policy)
        self.assertAlmostEqual(kept_mean[0].pactivity.require(), 7.30, places=6)

    def test_irreconcilable_replicates_are_discarded_not_averaged(self) -> None:
        # 10 nM (pIC50 8) against 10 uM (pIC50 5): three log units apart. Their
        # mean, 6.5, is a number no experiment produced.
        measurements = [(record("A1"), 8.0), (record("A2"), 5.0)]
        kept, rejected = aggregate_measurements(measurements)
        self.assertEqual(kept, [])
        self.assertEqual(len(rejected), 2)
        self.assertEqual(rejected[0].rule, "irreconcilable_replicates")

    def test_spread_within_tolerance_is_kept_and_recorded(self) -> None:
        measurements = [(record("A1"), 7.0), (record("A2"), 7.8)]
        kept, rejected = aggregate_measurements(measurements)
        self.assertEqual(len(kept), 1)
        self.assertAlmostEqual(kept[0].spread_log_units or 0.0, 0.8, places=9)
        self.assertEqual(rejected, [])

    def test_spread_becomes_an_uncertainty(self) -> None:
        measurements = [(record("A1"), 7.0), (record("A2"), 7.8)]
        kept, _ = aggregate_measurements(measurements)
        self.assertAlmostEqual(kept[0].pactivity.uncertainty or 0.0, 0.4, places=9)

    def test_different_targets_are_not_merged(self) -> None:
        measurements = [
            (record("A1", target_id="T1"), 7.0),
            (record("A2", target_id="T2"), 5.0),
        ]
        kept, rejected = aggregate_measurements(measurements)
        self.assertEqual(len(kept), 2)
        self.assertEqual(rejected, [])

    def test_different_compounds_are_not_merged(self) -> None:
        measurements = [
            (record("A1", compound_id="C1"), 7.0),
            (record("A2", compound_id="C2"), 5.0),
        ]
        kept, _ = aggregate_measurements(measurements)
        self.assertEqual(len(kept), 2)

    def test_custom_compound_key_merges_salt_forms(self) -> None:
        # ChEMBL lists a free base and its hydrochloride under separate ids. A
        # structure-hash key function must be able to merge them, otherwise both
        # survive as independent compounds and leak across a split.
        measurements = [
            (record("A1", compound_id="C1_freebase"), 7.0),
            (record("A2", compound_id="C2_hcl"), 7.1),
        ]
        kept, _ = aggregate_measurements(
            measurements, compound_key=lambda r: "same_parent_structure"
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].n_measurements, 2)

    def test_min_measurements_policy_is_enforced(self) -> None:
        policy = AggregationPolicy(min_measurements=2)
        kept, rejected = aggregate_measurements([(record(), 7.0)], policy)
        self.assertEqual(kept, [])
        self.assertEqual(rejected[0].rule, "too_few_replicates")

    def test_mean_statistic_is_available_for_comparison(self) -> None:
        policy = AggregationPolicy(statistic="mean")
        measurements = [(record("A1"), 7.0), (record("A2"), 7.5)]
        kept, _ = aggregate_measurements(measurements, policy)
        self.assertAlmostEqual(kept[0].pactivity.require(), 7.25)

    def test_invalid_statistic_refused_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            AggregationPolicy(statistic="mode")

    def test_aggregated_point_cites_every_source_measurement(self) -> None:
        measurements = [(record("A1"), 7.0), (record("A2"), 7.1), (record("A3"), 7.2)]
        kept, _ = aggregate_measurements(measurements)
        self.assertEqual(kept[0].source_activity_ids, ("A1", "A2", "A3"))
        for activity_id in ("A1", "A2", "A3"):
            self.assertIn(activity_id, kept[0].pactivity.source)


class TestFullPipeline(unittest.TestCase):
    def test_accounting_invariant_holds(self) -> None:
        records = [
            record("A1", "C1", value=100.0),
            record("A2", "C1", value=120.0),
            record("A3", "C2", confidence_score=3),
            record("A4", "C3", unit="%"),
            record("A5", "C4", relation=">"),
            record("A6", "C5", value=1.0, unit="M"),
            record("A7", "C6", data_validity_comment="Outside typical range"),
        ]
        report = curate(records)
        self.assertEqual(report.n_input, len(records))

        accounted = sum(point.n_measurements for point in report.kept)
        accounted += report.n_rejected
        self.assertEqual(
            accounted,
            len(records),
            "every input measurement must be in a kept point or a rejection",
        )

    def test_report_describes_itself(self) -> None:
        records = [record("A1"), record("A2", confidence_score=3)]
        report = curate(records)
        text = report.describe()
        self.assertIn("2 measurements in", text)
        self.assertIn("low_target_confidence", text)

    def test_empty_input_gives_no_retention_rate(self) -> None:
        report = curate([])
        self.assertIsNone(report.retention)

    def test_total_wipeout_is_called_out(self) -> None:
        # Losing everything usually means the filter is wrong, not that the
        # target has no data. Silence here would send someone hunting for a
        # nonexistent data problem.
        report = curate([record(confidence_score=1), record(confidence_score=2)])
        self.assertIn("WARNING", report.describe())

    def test_retention_is_computed_over_measurements(self) -> None:
        report = curate([record("A1"), record("A2", confidence_score=1)])
        self.assertAlmostEqual(report.retention or 0.0, 0.5)

    def test_curated_rows_are_exportable(self) -> None:
        report = curate([record("A1", "C1")])
        row = report.kept[0].as_row()
        self.assertEqual(row["compound_id"], "C1")
        self.assertAlmostEqual(row["pactivity"], 7.0, places=9)
        # The export carries provenance, not just the number.
        self.assertEqual(row["origin"], "derived")
        self.assertIn("derived", row["pactivity_label"])


if __name__ == "__main__":
    unittest.main()
