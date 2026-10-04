"""Tests for unit handling and pActivity conversion.

The predecessor project contained this line:

    return -np.log10(value * 1e-9)  # Assuming nM units

Most of the cases below are regression tests against that assumption. Each one
is a real ChEMBL unit string that the old code would have silently mangled into
a plausible-looking pIC50.
"""

from __future__ import annotations

import unittest

from chemdisco.provenance import Origin
from chemdisco.units import (
    mass_per_litre_to_molar,
    normalise_unit,
    pactivity,
    pactivity_is_plausible,
    to_molar,
)


class TestNormaliseUnit(unittest.TestCase):
    def test_case_is_collapsed(self) -> None:
        self.assertEqual(normalise_unit("nM"), "nm")
        self.assertEqual(normalise_unit("UM"), "um")

    def test_both_micro_signs_are_handled(self) -> None:
        # U+00B5 MICRO SIGN and U+03BC GREEK SMALL LETTER MU are different code
        # points and both appear in ChEMBL exports. Treating them as distinct
        # units sends micromolar values down the "unrecognised" path.
        self.assertEqual(normalise_unit("µM"), "um")
        self.assertEqual(normalise_unit("μM"), "um")
        self.assertEqual(normalise_unit("uM"), "um")

    def test_slash_and_dot_notations_unify(self) -> None:
        self.assertEqual(normalise_unit("ug/mL"), normalise_unit("ug.mL-1"))

    def test_none_is_empty(self) -> None:
        self.assertEqual(normalise_unit(None), "")


class TestToMolar(unittest.TestCase):
    def test_known_scales(self) -> None:
        self.assertAlmostEqual(to_molar(1.0, "nM") or 0.0, 1e-9)
        self.assertAlmostEqual(to_molar(1.0, "uM") or 0.0, 1e-6)
        self.assertAlmostEqual(to_molar(1.0, "mM") or 0.0, 1e-3)
        self.assertAlmostEqual(to_molar(1.0, "M") or 0.0, 1.0)
        self.assertAlmostEqual(to_molar(1.0, "pM") or 0.0, 1e-12)

    def test_unknown_unit_returns_none_rather_than_guessing(self) -> None:
        self.assertIsNone(to_molar(1.0, "%"))
        self.assertIsNone(to_molar(1.0, "mg.kg-1"))
        self.assertIsNone(to_molar(1.0, None))

    def test_mass_per_volume_needs_a_weight(self) -> None:
        # 1 ug/mL of a 500 Da compound is 2 uM.
        molar = mass_per_litre_to_molar(1.0, "ug.mL-1", 500.0)
        self.assertAlmostEqual(molar or 0.0, 2e-6, places=12)

    def test_mass_conversion_refuses_a_nonpositive_weight(self) -> None:
        with self.assertRaises(ValueError):
            mass_per_litre_to_molar(1.0, "ug.mL-1", 0.0)


class TestPActivity(unittest.TestCase):
    def test_nanomolar_ic50(self) -> None:
        q = pactivity(1.0, "nM", activity_type="IC50", source="ACT1")
        self.assertAlmostEqual(q.require(), 9.0, places=9)
        self.assertIs(q.origin, Origin.DERIVED)

    def test_hundred_nanomolar_ic50(self) -> None:
        q = pactivity(100.0, "nM", activity_type="IC50", source="ACT1")
        self.assertAlmostEqual(q.require(), 7.0, places=9)

    def test_micromolar_is_not_treated_as_nanomolar(self) -> None:
        # The headline regression test. 1 uM is pIC50 6.0, not 9.0.
        q = pactivity(1.0, "uM", activity_type="IC50", source="ACT1")
        self.assertAlmostEqual(q.require(), 6.0, places=9)

    def test_millimolar(self) -> None:
        q = pactivity(1.0, "mM", activity_type="IC50", source="ACT1")
        self.assertAlmostEqual(q.require(), 3.0, places=9)

    def test_percent_inhibition_is_refused(self) -> None:
        # A 50% inhibition readout has no pActivity at all. The old code would
        # have produced -log10(50e-9) = 7.3, a very respectable fiction.
        q = pactivity(50.0, "%", activity_type="Inhibition", source="ACT1")
        self.assertFalse(q.is_known)
        self.assertIn("not a concentration", q.source)

    def test_dose_units_are_refused(self) -> None:
        q = pactivity(10.0, "mg.kg-1", activity_type="ED50", source="ACT1")
        self.assertFalse(q.is_known)

    def test_unrecognised_unit_is_refused_not_guessed(self) -> None:
        q = pactivity(1.0, "wibbles", activity_type="IC50", source="ACT1")
        self.assertFalse(q.is_known)
        self.assertIn("refusing to guess", q.source)

    def test_missing_unit_is_refused(self) -> None:
        q = pactivity(1.0, None, activity_type="IC50", source="ACT1")
        self.assertFalse(q.is_known)

    def test_already_log_scale_is_not_logged_twice(self) -> None:
        # pIC50 7.5 must stay 7.5. Converting it as if it were a concentration
        # would give -log10(7.5e-9) = 8.1, which is wrong but believable.
        q = pactivity(7.5, None, activity_type="pIC50", source="ACT1")
        self.assertAlmostEqual(q.require(), 7.5)

    def test_censored_greater_than_is_refused(self) -> None:
        q = pactivity(10000.0, "nM", activity_type="IC50", source="ACT1", relation=">")
        self.assertFalse(q.is_known)
        self.assertIn("censored", q.source)
        self.assertIn("upper bound", q.source)

    def test_censored_less_than_is_refused_with_the_other_bound(self) -> None:
        q = pactivity(1.0, "nM", activity_type="IC50", source="ACT1", relation="<")
        self.assertFalse(q.is_known)
        self.assertIn("lower bound", q.source)

    def test_none_value_is_refused(self) -> None:
        self.assertFalse(pactivity(None, "nM", activity_type="IC50", source="A").is_known)

    def test_zero_concentration_is_refused(self) -> None:
        # log of zero is undefined; the old code would raise or emit -inf.
        q = pactivity(0.0, "nM", activity_type="IC50", source="ACT1")
        self.assertFalse(q.is_known)
        self.assertIn("non-positive", q.source)

    def test_negative_concentration_is_refused(self) -> None:
        q = pactivity(-5.0, "nM", activity_type="IC50", source="ACT1")
        self.assertFalse(q.is_known)

    def test_mass_per_volume_with_weight_converts(self) -> None:
        q = pactivity(
            1.0,
            "ug.mL-1",
            activity_type="MIC",
            source="ACT1",
            molecular_weight=500.0,
        )
        self.assertAlmostEqual(q.require(), 5.69897, places=4)

    def test_mass_per_volume_without_weight_is_refused(self) -> None:
        q = pactivity(1.0, "ug.mL-1", activity_type="MIC", source="ACT1")
        self.assertFalse(q.is_known)
        self.assertIn("molecular weight", q.source)

    def test_implausible_value_is_flagged_but_returned(self) -> None:
        # 1 fM is pIC50 15, beyond anything an assay can measure. The value is
        # returned so curation can decide, but it carries a loud note.
        q = pactivity(1.0, "fM", activity_type="IC50", source="ACT1")
        self.assertTrue(q.is_known)
        self.assertTrue(any("implausible" in note for note in q.notes))
        self.assertFalse(pactivity_is_plausible(q))

    def test_provenance_records_the_arithmetic(self) -> None:
        q = pactivity(250.0, "nM", activity_type="Ki", source="ACT42")
        self.assertIn("ACT42", q.source)
        self.assertIn("log10", q.source)
        self.assertTrue(any("from 250.0 nM" in note for note in q.notes))

    def test_every_refusal_explains_itself(self) -> None:
        refusals = [
            pactivity(50.0, "%", activity_type="Inhibition", source="A"),
            pactivity(1.0, "wibbles", activity_type="IC50", source="B"),
            pactivity(0.0, "nM", activity_type="IC50", source="C"),
            pactivity(1.0, "nM", activity_type="IC50", source="D", relation=">"),
            pactivity(None, "nM", activity_type="IC50", source="E"),
        ]
        for quantity in refusals:
            with self.subTest(source=quantity.source):
                self.assertFalse(quantity.is_known)
                # A refusal with no reason cannot be reviewed or debugged.
                self.assertGreater(len(quantity.source.strip()), 10)


if __name__ == "__main__":
    unittest.main()
