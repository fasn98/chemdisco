"""Tests for scaffold splitting.

Leakage is the failure these tests exist to catch, and it is a quiet one: a
leaking split produces a higher score, not an error. So the properties are
checked explicitly on every split -- disjointness, coverage, non-empty
partitions -- rather than assumed from the implementation looking correct.
"""

from __future__ import annotations

import unittest

from chemdisco.split import (
    SplitError,
    coverage,
    group_by_scaffold,
    iter_scaffold_splits,
    random_split,
    scaffold_split,
    verify_disjoint,
)


def series_dataset() -> list[str]:
    """A dataset shaped like a real optimisation campaign.

    Four chemical series of unequal size, which is the structure that makes a
    random split dishonest: the 40-member series would place analogues on both
    sides of the line.
    """
    return (
        ["c1ccc2[nH]ccc2c1"] * 40  # indole series, the big one
        + ["c1ccncc1"] * 25  # pyridine series
        + ["c1ccc2ncccc2c1"] * 20  # quinoline series
        + ["C1CCNCC1"] * 15  # piperidine series
    )


class TestGrouping(unittest.TestCase):
    def test_groups_collect_indices(self) -> None:
        groups = group_by_scaffold(["A", "B", "A", "C", "B"])
        self.assertEqual(groups["A"], [0, 2])
        self.assertEqual(groups["B"], [1, 4])
        self.assertEqual(groups["C"], [3])

    def test_acyclic_molecules_keep_the_empty_key(self) -> None:
        groups = group_by_scaffold(["A", "", ""])
        self.assertEqual(groups[""], [1, 2])


class TestScaffoldSplit(unittest.TestCase):
    def test_no_scaffold_crosses_the_line(self) -> None:
        scaffolds = series_dataset()
        split = scaffold_split(scaffolds, test_fraction=0.25)
        self.assertEqual(verify_disjoint(split, scaffolds), [])

    def test_every_molecule_is_used_exactly_once(self) -> None:
        scaffolds = series_dataset()
        split = scaffold_split(scaffolds, test_fraction=0.25)
        self.assertTrue(coverage(split, len(scaffolds)))

    def test_balanced_assignment_tracks_the_requested_fraction(self) -> None:
        scaffolds = series_dataset()
        split = scaffold_split(scaffolds, test_fraction=0.25, balanced=True)
        achieved = len(split.test) / len(scaffolds)
        # Groups are indivisible, so exactness is impossible; being within ten
        # points of the request is what "approximately" has to mean here.
        self.assertLess(abs(achieved - 0.25), 0.10, f"achieved {achieved:.3f}")

    def test_validation_partition_is_also_disjoint(self) -> None:
        scaffolds = series_dataset()
        split = scaffold_split(
            scaffolds, test_fraction=0.2, validation_fraction=0.2
        )
        self.assertEqual(verify_disjoint(split, scaffolds), [])
        self.assertTrue(split.validation)
        self.assertTrue(coverage(split, len(scaffolds)))

    def test_deterministic_mode_is_reproducible(self) -> None:
        scaffolds = series_dataset()
        first = scaffold_split(scaffolds, test_fraction=0.25)
        second = scaffold_split(scaffolds, test_fraction=0.25)
        self.assertEqual(first.train, second.train)
        self.assertEqual(first.test, second.test)

    def test_seeded_mode_is_reproducible_for_a_given_seed(self) -> None:
        scaffolds = series_dataset()
        first = scaffold_split(scaffolds, seed=7)
        second = scaffold_split(scaffolds, seed=7)
        self.assertEqual(first.test, second.test)

    def test_different_seeds_give_different_splits(self) -> None:
        scaffolds = series_dataset()
        tests = {scaffold_split(scaffolds, seed=s).test for s in range(6)}
        self.assertGreater(
            len(tests), 1, "seeding must actually vary which chemotypes are held out"
        )

    def test_seeded_split_does_not_depend_on_the_hash_seed(self) -> None:
        # Python randomises str.__hash__ per process unless PYTHONHASHSEED is
        # fixed. A split built on the builtin hash would differ between runs
        # while claiming to be seeded, so the implementation hashes explicitly.
        import os
        import subprocess
        import sys

        code = (
            "from chemdisco.split import scaffold_split;"
            "s=['A']*10+['B']*10+['C']*10+['D']*10;"
            "print(scaffold_split(s, seed=3).test)"
        )
        outputs = set()
        for hash_seed in ("0", "1", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed)
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                env=env,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            outputs.add(result.stdout.strip())
        self.assertEqual(len(outputs), 1, f"split varied with PYTHONHASHSEED: {outputs}")

    def test_acyclic_molecules_become_singletons_by_default(self) -> None:
        # Pooling every acyclic compound into one group would strand all of them
        # on one side of the split.
        scaffolds = ["A"] * 10 + [""] * 10
        split = scaffold_split(scaffolds, test_fraction=0.3)
        acyclic_in_train = sum(1 for i in split.train if scaffolds[i] == "")
        acyclic_in_test = sum(1 for i in split.test if scaffolds[i] == "")
        self.assertGreater(acyclic_in_train, 0)
        self.assertGreater(acyclic_in_test, 0)

    def test_pooling_acyclic_molecules_can_be_requested(self) -> None:
        scaffolds = ["A"] * 10 + ["B"] * 10 + [""] * 10
        split = scaffold_split(
            scaffolds, test_fraction=0.33, acyclic_as_singletons=False
        )
        acyclic_sides = {
            "train" if i in set(split.train) else "test"
            for i, s in enumerate(scaffolds)
            if s == ""
        }
        self.assertEqual(len(acyclic_sides), 1, "pooled acyclics must stay together")

    def test_single_scaffold_dataset_refuses_to_split(self) -> None:
        # One series cannot measure generalisation to a new chemotype. Returning
        # a split here would produce a meaningless but publishable number.
        with self.assertRaises(SplitError) as context:
            scaffold_split(["A"] * 50)
        self.assertIn("single scaffold group", str(context.exception))

    def test_empty_dataset_refuses(self) -> None:
        with self.assertRaises(SplitError):
            scaffold_split([])

    def test_invalid_fractions_refuse(self) -> None:
        scaffolds = series_dataset()
        for bad in (0.0, 1.0, -0.1, 1.5):
            with self.subTest(fraction=bad):
                with self.assertRaises(SplitError):
                    scaffold_split(scaffolds, test_fraction=bad)

    def test_fractions_summing_past_one_refuse(self) -> None:
        with self.assertRaises(SplitError):
            scaffold_split(series_dataset(), test_fraction=0.6, validation_fraction=0.5)

    def test_split_reports_what_it_actually_did(self) -> None:
        scaffolds = series_dataset()
        split = scaffold_split(scaffolds, test_fraction=0.25)
        text = split.describe()
        self.assertIn("scaffold", text)
        self.assertIn("4 scaffold groups", text)
        self.assertIn("achieved fractions", text)

    def test_repeated_splits_vary_so_variance_can_be_reported(self) -> None:
        scaffolds = series_dataset()
        splits = list(iter_scaffold_splits(scaffolds, n_repeats=5, test_fraction=0.25))
        self.assertEqual(len(splits), 5)
        for split in splits:
            self.assertEqual(verify_disjoint(split, scaffolds), [])
            self.assertTrue(coverage(split, len(scaffolds)))


class TestSplitRobustness(unittest.TestCase):
    """Property tests over many group-size configurations.

    The two balancing bugs found while writing this module were both invisible
    on a single tidy dataset and appeared only at particular ratios of group
    sizes to quotas. Sweeping configurations is what catches that class of bug;
    one example dataset does not.
    """

    def _configurations(self) -> list[list[str]]:
        import random as _random

        rng = _random.Random(20260104)
        configurations: list[list[str]] = []
        for _ in range(60):
            n_groups = rng.randint(2, 15)
            sizes = [rng.randint(1, 60) for _ in range(n_groups)]
            scaffolds: list[str] = []
            for group_index, size in enumerate(sizes):
                scaffolds.extend([f"S{group_index}"] * size)
            configurations.append(scaffolds)
        return configurations

    def test_no_configuration_leaks_or_loses_molecules(self) -> None:
        for scaffolds in self._configurations():
            for fraction in (0.1, 0.2, 0.3, 0.5):
                for seed in (None, 0, 1):
                    with self.subTest(
                        n=len(scaffolds), fraction=fraction, seed=seed
                    ):
                        try:
                            split = scaffold_split(
                                scaffolds, test_fraction=fraction, seed=seed
                            )
                        except SplitError:
                            # A refusal is an acceptable outcome; a wrong split
                            # is not. Only the splits that are produced must
                            # satisfy the invariants.
                            continue
                        self.assertEqual(verify_disjoint(split, scaffolds), [])
                        self.assertTrue(coverage(split, len(scaffolds)))
                        self.assertTrue(split.test, "test set must never be empty")
                        self.assertTrue(split.train, "train set must never be empty")

    def test_validation_partition_survives_the_sweep(self) -> None:
        for scaffolds in self._configurations():
            with self.subTest(n=len(scaffolds)):
                try:
                    split = scaffold_split(
                        scaffolds, test_fraction=0.2, validation_fraction=0.2, seed=3
                    )
                except SplitError:
                    continue
                self.assertEqual(verify_disjoint(split, scaffolds), [])
                self.assertTrue(coverage(split, len(scaffolds)))
                self.assertTrue(split.test)
                self.assertTrue(split.validation)
                self.assertTrue(split.train)

    def test_repair_pass_is_disclosed_when_it_fires(self) -> None:
        # A split that needed repairing is still honest, but the report must say
        # so: the achieved fraction will be further from the request than usual.
        fired = False
        for scaffolds in self._configurations():
            for seed in (None, 0, 1, 2, 3):
                try:
                    split = scaffold_split(scaffolds, test_fraction=0.1, seed=seed)
                except SplitError:
                    continue
                if any("had left empty" in note for note in split.notes):
                    fired = True
                    self.assertTrue(split.test)
                    self.assertEqual(verify_disjoint(split, scaffolds), [])
        self.assertTrue(
            fired,
            "the sweep should exercise the repair pass; if it never fires the "
            "test is not covering the case it claims to",
        )


class TestRandomSplit(unittest.TestCase):
    def test_random_split_covers_the_dataset(self) -> None:
        split = random_split(100, test_fraction=0.2, seed=1)
        self.assertTrue(coverage(split, 100))
        self.assertEqual(len(split.test), 20)

    def test_random_split_warns_in_its_own_notes(self) -> None:
        # The warning travels with the object, so a report generated from it
        # cannot quietly omit the caveat.
        split = random_split(100)
        joined = " ".join(split.notes)
        self.assertIn("OPTIMISTIC BASELINE ONLY", joined)

    def test_random_split_leaks_scaffolds_as_expected(self) -> None:
        # Demonstrating the problem rather than asserting a convention: on a
        # dataset of four series, a random split necessarily puts members of the
        # same series on both sides.
        scaffolds = series_dataset()
        split = random_split(len(scaffolds), test_fraction=0.25, seed=0)
        leaked = verify_disjoint(split, scaffolds)
        self.assertEqual(
            len(leaked), 4, "all four series should straddle a random split"
        )

    def test_empty_dataset_refuses(self) -> None:
        with self.assertRaises(SplitError):
            random_split(0)


if __name__ == "__main__":
    unittest.main()
