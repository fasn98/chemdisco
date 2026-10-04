"""Scaffold-based dataset splitting.

The single most consequential methodological choice in a QSAR report.

Medicinal chemistry data is not independent and identically distributed. A
ChEMBL target set is a few dozen chemical series, each a scaffold decorated
tens or hundreds of ways, because that is how optimisation campaigns are run. A
random split therefore puts near-identical analogues on both sides of the
train/test line, and the model is scored on its ability to interpolate within a
series it has already seen. Published comparisons routinely find random-split
R-squared of 0.8 collapsing to 0.3-0.4 under a scaffold split on the same data
and the same model. The 0.8 is not a measurement of anything a chemist would
use the model for.

A scaffold split assigns whole Bemis-Murcko scaffold groups to one side or the
other, so the test set contains chemotypes the model has never seen. That is the
question worth asking: does this model generalise to a new series?

This module is deliberately free of RDKit. It takes scaffolds as strings, so the
splitting logic -- the part where leakage bugs hide -- is unit-testable without a
chemistry toolkit, and the scaffold perception lives behind one function at the
toolkit edge (:func:`chemdisco.chem.scaffold.murcko_scaffold`).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence


class SplitError(ValueError):
    """Raised when a requested split cannot be produced honestly."""


@dataclass(frozen=True, slots=True)
class Split:
    """Index sets for one partition of a dataset.

    Indices refer to positions in the sequence that was split, so the caller can
    apply them to features, labels and identifiers alike.
    """

    train: tuple[int, ...]
    test: tuple[int, ...]
    validation: tuple[int, ...] = ()
    strategy: str = ""
    notes: tuple[str, ...] = ()

    @property
    def n_total(self) -> int:
        return len(self.train) + len(self.test) + len(self.validation)

    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }

    def describe(self) -> str:
        sizes = self.sizes()
        parts = [f"{name}={count}" for name, count in sizes.items() if count]
        text = f"{self.strategy or 'split'}: " + ", ".join(parts)
        if self.notes:
            text += "\n  " + "\n  ".join(self.notes)
        return text


def _stable_hash(text: str) -> int:
    """Deterministic hash independent of the interpreter's salt.

    Python's built-in ``hash`` for strings is randomised per process unless
    ``PYTHONHASHSEED`` is set, which would make a "reproducible" split silently
    differ between runs. Hashing explicitly removes that trap.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def group_by_scaffold(scaffolds: Sequence[str]) -> dict[str, list[int]]:
    """Map each scaffold to the indices of the molecules carrying it.

    An empty scaffold string -- which RDKit returns for an acyclic molecule, as
    a Murcko scaffold needs at least one ring -- is kept as its own group under
    the key ``""``. Acyclic compounds genuinely share no scaffold, so merging
    them into one group is a modelling choice with consequences; it is made
    explicit via ``acyclic_as_singletons`` in :func:`scaffold_split` rather than
    hidden here.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for index, scaffold in enumerate(scaffolds):
        groups[scaffold].append(index)
    return dict(groups)


def scaffold_split(
    scaffolds: Sequence[str],
    *,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.0,
    seed: int | None = None,
    acyclic_as_singletons: bool = True,
    balanced: bool = True,
) -> Split:
    """Partition indices so that no scaffold appears on both sides.

    Args:
        scaffolds: One scaffold string per molecule, aligned with the dataset.
        test_fraction: Target share of molecules in the test set. Achieved
            approximately: scaffold groups are indivisible, so a dataset
            dominated by one large series cannot be split to an arbitrary ratio.
        validation_fraction: Optional third partition, also scaffold-disjoint
            from the others.
        seed: Shuffles the order in which groups are assigned. ``None`` gives the
            deterministic largest-groups-to-train ordering used by MoleculeNet,
            which is reproducible but yields exactly one split; a seed lets you
            repeat the evaluation over several scaffold splits and report the
            variance, which is better practice.
        acyclic_as_singletons: Treat each acyclic molecule as its own scaffold
            group rather than pooling all of them. Pooling would place every
            acyclic compound on one side and can strand a large block of data.
        balanced: Assign each group to whichever partition is furthest below its
            quota. This tracks the requested fractions much more closely than
            filling the test set first, which commonly overshoots badly when a
            single group is large.

    Returns:
        A :class:`Split` whose notes record the achieved fractions and the
        scaffold counts, so a report can state what the split actually was
        rather than what was requested.

    Raises:
        SplitError: if the fractions are invalid, the dataset is empty, or the
            scaffold structure makes a non-empty test set impossible. The last
            case raises rather than returning an empty test set, because silently
            evaluating on nothing is how a meaningless metric gets published.
    """
    n = len(scaffolds)
    if n == 0:
        raise SplitError("cannot split an empty dataset")
    if not 0.0 < test_fraction < 1.0:
        raise SplitError(f"test_fraction must be in (0, 1), got {test_fraction}")
    if not 0.0 <= validation_fraction < 1.0:
        raise SplitError(
            f"validation_fraction must be in [0, 1), got {validation_fraction}"
        )
    if test_fraction + validation_fraction >= 1.0:
        raise SplitError(
            "test_fraction + validation_fraction must leave data for training"
        )

    raw_groups = group_by_scaffold(scaffolds)

    groups: list[tuple[str, list[int]]] = []
    for scaffold, indices in raw_groups.items():
        if scaffold == "" and acyclic_as_singletons:
            for index in indices:
                groups.append((f"<acyclic:{index}>", [index]))
        else:
            groups.append((scaffold, indices))

    if len(groups) < 2:
        raise SplitError(
            f"all {n} molecules share a single scaffold group; a "
            "scaffold-disjoint split is impossible. Either the dataset is one "
            "chemical series -- in which case no split can measure "
            "generalisation to new chemotypes -- or scaffold perception failed."
        )

    if seed is None:
        # MoleculeNet convention: largest groups to train, so the test set is
        # made of the rarer chemotypes. Ties broken by scaffold string for
        # determinism.
        ordered = sorted(groups, key=lambda item: (-len(item[1]), item[0]))
    else:
        # Seeded order. Hashing scaffold plus seed gives a reproducible shuffle
        # without depending on the order the groups happened to be built in.
        ordered = sorted(groups, key=lambda item: _stable_hash(f"{seed}:{item[0]}"))

    quotas = {
        "test": test_fraction * n,
        "validation": validation_fraction * n,
        "train": (1.0 - test_fraction - validation_fraction) * n,
    }
    assigned: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    # Groups are tracked per partition so the repair pass below can move a whole
    # group without splitting it, which would reintroduce leakage.
    held: dict[str, list[tuple[str, list[int]]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    active = [name for name, quota in quotas.items() if quota > 0]
    repairs: list[str] = []

    for scaffold_key, indices in ordered:
        if balanced:
            # Greedy on the largest absolute shortfall against quota.
            #
            # Two alternatives were tried and are wrong. Picking the partition
            # proportionally furthest below quota ties at the start, when every
            # partition is empty, so the first and largest group lands wherever
            # the iteration order happens to look first; on a 40/25/20/15
            # dataset at test_fraction=0.25 that yields an achieved test
            # fraction of 0.40 instead of 0.20. Minimising the resulting ratio
            # instead sends every group to train and empties the test set.
            # Absolute deficit is correct at the start because the training
            # quota is the largest.
            #
            # Ties are broken toward the smaller quota, so a held-out partition
            # is never starved by the larger one winning every coin flip.
            target = max(
                active,
                key=lambda name: (
                    quotas[name] - len(assigned[name]),
                    -quotas[name],
                    name,
                ),
            )
        else:
            target = "train"
            for name in ("test", "validation"):
                if quotas[name] and len(assigned[name]) < quotas[name]:
                    target = name
                    break
        assigned[target].extend(indices)
        held[target].append((scaffold_key, indices))

    # Repair pass. Scaffold groups are indivisible, so greedy assignment can
    # still leave a requested partition empty when a few groups dominate the
    # dataset. Rather than raise -- which would reject a usable dataset -- move
    # the single group whose size best matches the empty partition's quota out
    # of the fullest partition. An empty test set must never be returned: a
    # metric computed on nothing looks like a metric.
    for name in active:
        if assigned[name] or quotas[name] < 1:
            continue
        donor = max(
            (other for other in active if other != name),
            key=lambda other: len(held[other]),
            default=None,
        )
        if donor is None or len(held[donor]) < 2:
            raise SplitError(
                f"cannot fill the '{name}' partition: too few scaffold groups "
                "to divide this dataset at the requested fractions"
            )
        moved = min(held[donor], key=lambda item: abs(len(item[1]) - quotas[name]))
        held[donor].remove(moved)
        held[name].append(moved)
        for index in moved[1]:
            assigned[donor].remove(index)
        assigned[name].extend(moved[1])
        repairs.append(
            f"moved a {len(moved[1])}-molecule scaffold group from '{donor}' to "
            f"'{name}', which greedy assignment had left empty"
        )

    if not assigned["test"]:
        raise SplitError(
            "the requested test fraction produced an empty test set; the "
            "scaffold groups are too unevenly sized for this ratio"
        )
    if not assigned["train"]:
        raise SplitError("the requested fractions produced an empty training set")

    notes = [
        f"{len(groups)} scaffold groups over {n} molecules",
        f"largest group holds {max(len(i) for _, i in groups)} molecules",
        "achieved fractions: "
        + ", ".join(
            f"{name}={len(indices) / n:.3f}"
            for name, indices in assigned.items()
            if indices
        ),
        "no scaffold appears in more than one partition",
    ]
    if seed is None:
        notes.append(
            "deterministic largest-to-train ordering; vary `seed` across runs to "
            "report split-to-split variance"
        )
    else:
        notes.append(f"seeded scaffold shuffle, seed={seed}")
    notes.extend(repairs)

    return Split(
        train=tuple(sorted(assigned["train"])),
        test=tuple(sorted(assigned["test"])),
        validation=tuple(sorted(assigned["validation"])),
        strategy="scaffold",
        notes=tuple(notes),
    )


def random_split(
    n: int,
    *,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.0,
    seed: int = 0,
) -> Split:
    """A random split, provided **only** as the pessimistic comparison.

    Reporting a random-split score next to a scaffold-split score is the
    clearest way to show how much of a model's apparent accuracy comes from
    analogue leakage. The gap between the two is diagnostic: a small gap means
    the model generalises across chemotypes; a large gap means the random number
    was measuring memorisation.

    Never report this figure alone. :func:`chemdisco.qsar.evaluate.compare_splits`
    exists to make the paired comparison the default way of using it.
    """
    import random as _random  # local: see tests/test_no_fabrication.py

    if n == 0:
        raise SplitError("cannot split an empty dataset")
    rng = _random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    n_test = max(1, int(round(test_fraction * n)))
    n_validation = int(round(validation_fraction * n))
    if n_test + n_validation >= n:
        raise SplitError("fractions leave no training data")

    test = indices[:n_test]
    validation = indices[n_test : n_test + n_validation]
    train = indices[n_test + n_validation :]

    return Split(
        train=tuple(sorted(train)),
        test=tuple(sorted(test)),
        validation=tuple(sorted(validation)),
        strategy="random",
        notes=(
            f"seed={seed}",
            "OPTIMISTIC BASELINE ONLY: analogues of test compounds are present "
            "in training, so this score overstates generalisation to new "
            "chemotypes. Report it only beside a scaffold-split score.",
        ),
    )


def verify_disjoint(split: Split, scaffolds: Sequence[str]) -> list[str]:
    """Return the scaffolds that leak across partitions; empty means clean.

    Called in tests and before training. An assertion rather than an assumption:
    a leakage bug produces a believable, publishable, wrong number, so it must be
    checked rather than trusted.
    """
    partitions = {
        "train": split.train,
        "validation": split.validation,
        "test": split.test,
    }
    seen: dict[str, set[str]] = defaultdict(set)
    for name, indices in partitions.items():
        for index in indices:
            seen[scaffolds[index]].add(name)
    return sorted(
        scaffold for scaffold, names in seen.items() if len(names) > 1 and scaffold != ""
    )


def coverage(split: Split, n: int) -> bool:
    """Whether the split accounts for every index exactly once."""
    combined = list(split.train) + list(split.validation) + list(split.test)
    return sorted(combined) == list(range(n))


def iter_scaffold_splits(
    scaffolds: Sequence[str],
    *,
    n_repeats: int = 5,
    test_fraction: float = 0.2,
    base_seed: int = 0,
) -> Iterable[Split]:
    """Yield several seeded scaffold splits for variance reporting.

    One split gives one number with no error bar. A single scaffold split can
    land on an easy or a hard set of held-out chemotypes, and the spread across
    repeats is often wider than the difference between two models -- so a model
    comparison based on one split frequently cannot support its conclusion.
    """
    for repeat in range(n_repeats):
        yield scaffold_split(
            scaffolds, test_fraction=test_fraction, seed=base_seed + repeat
        )
