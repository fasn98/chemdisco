"""The triage that actually produces the shortlist, in ``scripts/discover.py``.

There are two triage paths in this repository and only one of them is obvious.
``chemdisco.dock.triage_candidates`` is the library function, thresholding raw
score at a reference percentile. The triage the pipeline's own output comes from
is inline in ``combine()``, thresholds *ligand efficiency* at the reference
median, and never calls the library function at all.

That mattered once: an attempt to make the triage withhold changed only the
library function, left the pipeline unchanged, and so left the README describing
behaviour the code did not execute. These tests cover the inline path directly,
through the JSON it writes, because that file is what a later run is read from.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def shard(candidates: list[dict], reference_efficiencies: list[float]) -> dict:
    """A shard payload with everything ``combine()`` requires and nothing else."""
    return {
        "shard": 0,
        "reference_scores": [-9.0 - 0.01 * i for i in range(len(reference_efficiencies))],
        "reference_efficiencies": reference_efficiencies,
        "candidates": candidates,
        "receptor_id": "4FRS",
        "box_signature": "centre|size",
        "ligand_signature": "testsig0000000000",
        "curation_signature": "testcur0000000000",
        "exhaustiveness": 8,
        "cpu": 6,
        "n_requested": len(candidates),
        "elapsed_seconds": 1.0,
        "feature_profile": {
            "conserved": ["amidine"],
            "most_discriminating": ["amidine"],
            "ubiquitous": [],
            "n_actives": len(reference_efficiencies),
            "n_background": 1200,
            "has_background": True,
            "can_discriminate": True,
            "background_summary": {"source": "unrelated", "families": {}},
            "prevalence": {"amidine": 0.9},
            "background_prevalence": {"amidine": 0.04},
        },
        "n_fragments": 81,
        "anchor_motifs": ["amidine"],
        "anchor_constraint_applied": True,
        "n_anchor_fragments": 26,
        "n_plain_fragments": 55,
        "n_anchor_escapes": 0,
        "n_anchor_lost": 0,
        "policy_audit_pass_rate": 1.0,
    }


def candidate(index: int, efficiency: float, heavy_atoms: int) -> dict:
    return {
        "smiles": f"C{index}",
        "score": efficiency * heavy_atoms,
        "heavy_atoms": heavy_atoms,
        "ligand_efficiency": efficiency,
        "sascore": 4.0,
        "novelty": 0.5,
        "retains_conserved_feature": True,
        "retains_strong_feature": True,
        "conserved_features": ["amidine"],
        "duplicated_motifs": {},
    }


def run_combine(candidates: list[dict], reference_efficiencies: list[float]) -> dict:
    from scripts.discover import combine

    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        (root / "discover_shard_0.json").write_text(
            json.dumps(shard(candidates, reference_efficiencies))
        )
        output = root / "shortlist.json"
        code = combine(root, str(output))
        assert code == 0, f"combine() refused with {code}"
        return json.loads(output.read_text())


#: 30 reference actives whose median ligand efficiency is exactly -0.260, the
#: value measured on BACE1. Built as two equal halves so the median is exact
#: rather than the average of two neighbours, which would land on -0.2605 and
#: make the margin arithmetic below hard to read.
REFERENCE = [-0.270] * 15 + [-0.250] * 15


class TestTriageWithheldWhenItCannotDiscriminate(unittest.TestCase):
    """The measured BACE1 case: everything inside the method error."""

    def _tight_pool(self) -> list[dict]:
        # 30 heavy atoms, efficiencies from -0.300 to -0.260. The best margin is
        # 0.040 kcal/mol/atom * 30 atoms = 1.2 kcal/mol, inside Vina's 2.5. So no
        # candidate is distinguishably better than the threshold.
        #
        # Deliberately NOT in efficiency order: the best entry sits in the middle.
        # A pool that arrives pre-sorted cannot show whether the code sorted it.
        efficiencies = [-0.300 + 0.002 * i for i in range(21)]
        shuffled = efficiencies[10:] + efficiencies[:10]
        return [
            candidate(index, efficiency, 30)
            for index, efficiency in enumerate(shuffled)
        ]

    def test_the_verdict_is_serialised_explicitly(self) -> None:
        payload = run_combine(self._tight_pool(), REFERENCE)
        self.assertTrue(payload["triage_withheld"])
        self.assertEqual(payload["triage_n_distinguishable"], 0)

    def test_every_candidate_is_reported_not_a_passing_subset(self) -> None:
        pool = self._tight_pool()
        payload = run_combine(pool, REFERENCE)
        # The old behaviour named the subset below the median. Withholding reports
        # all of them, because a check that cannot separate does not get to pick.
        self.assertEqual(payload["n_survivors"], len(pool))
        self.assertEqual(len(payload["candidates_unranked"]), len(pool))

    def test_the_list_is_not_called_a_shortlist(self) -> None:
        payload = run_combine(self._tight_pool(), REFERENCE)
        self.assertNotIn("shortlist", payload)
        self.assertIn("candidates_unranked", payload)

    def test_the_candidates_are_not_ranked_by_efficiency(self) -> None:
        # Sorting by efficiency would imply an ordering the method cannot support.
        pool = self._tight_pool()
        payload = run_combine(pool, REFERENCE)
        reported = [c["ligand_efficiency"] for c in payload["candidates_unranked"]]
        self.assertNotEqual(
            reported,
            sorted(reported),
            "a withheld triage must not emit candidates sorted by efficiency",
        )

    def test_the_threshold_is_still_reported(self) -> None:
        payload = run_combine(self._tight_pool(), REFERENCE)
        self.assertAlmostEqual(payload["reference_median"], -0.260, places=3)
        self.assertEqual(payload["vina_error_kcal"], 2.5)


class TestTriageStillFiltersWhenItDoesDiscriminate(unittest.TestCase):
    """Withholding must not become unconditional."""

    def _separated_pool(self) -> list[dict]:
        # One candidate at -0.400 over 40 heavy atoms: a margin of
        # 0.140 * 40 = 5.6 kcal/mol, well outside the error. The rest sit above
        # the threshold and are discarded as before.
        return [candidate(0, -0.400, 40)] + [
            candidate(i, -0.200 + 0.002 * i, 30) for i in range(1, 10)
        ]

    def test_a_separated_pool_keeps_filtering(self) -> None:
        payload = run_combine(self._separated_pool(), REFERENCE)
        self.assertFalse(payload["triage_withheld"])
        self.assertEqual(payload["triage_n_distinguishable"], 1)

    def test_it_is_a_shortlist_again_and_a_strict_subset(self) -> None:
        pool = self._separated_pool()
        payload = run_combine(pool, REFERENCE)
        self.assertIn("shortlist", payload)
        self.assertNotIn("candidates_unranked", payload)
        self.assertLess(payload["n_survivors"], len(pool))
        self.assertEqual(payload["n_survivors"], 1)


if __name__ == "__main__":
    unittest.main()
