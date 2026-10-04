"""Evaluation: metrics with error bars, and the comparisons that give them meaning.

A single R-squared from a single split is the weakest defensible statement a
QSAR report can make, and it is the one most often made. Two things are missing:

**An error bar.** On a 200-compound test set, the 95% bootstrap interval on
R-squared routinely spans 0.2. Two models differing by 0.05 are
indistinguishable, and a conclusion drawn from that difference is not supported
by the data. :func:`evaluate` therefore refuses to report a point estimate alone.

**A reference point.** R-squared is scaled by the test set's label variance, so
the same model scores differently on an easy and a hard split of the same data.
:func:`compare_splits` runs scaffold and random splits side by side, because the
gap between them quantifies how much of the apparent accuracy is analogue
leakage -- the number that decides whether a model is useful for proposing new
chemotypes or merely for interpolating within known ones.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .baselines import BaselineScores, r_squared, rmse


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Rank correlation, implemented directly to avoid a SciPy dependency.

    Rank correlation is arguably the more relevant metric for candidate
    selection: what matters operationally is whether the model orders compounds
    correctly for which to synthesise first, not whether it nails the absolute
    pIC50.

    Ties are handled by average ranking, which is what makes this agree with the
    standard implementation on data containing repeated values -- common here,
    since curated pActivity values cluster on round numbers.
    """
    a = _average_ranks(np.asarray(y_true, dtype=float))
    b = _average_ranks(np.asarray(y_pred, dtype=float))
    if np.std(a) == 0 or np.std(b) == 0:
        raise ValueError("rank correlation undefined when one variable is constant")
    return float(np.corrcoef(a, b)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # Average the ranks within each group of tied values.
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for group_index, count in enumerate(counts):
        if count > 1:
            mask = inverse == group_index
            ranks[mask] = ranks[mask].mean()
    return ranks


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        raise ValueError("correlation undefined when one variable is constant")
    return float(np.corrcoef(a, b)[0, 1])


@dataclass(frozen=True, slots=True)
class Interval:
    """A point estimate with a bootstrap confidence interval."""

    estimate: float
    low: float
    high: float
    confidence: float = 0.95

    @property
    def width(self) -> float:
        return self.high - self.low

    def label(self, digits: int = 3) -> str:
        return (
            f"{self.estimate:.{digits}f} "
            f"[{self.low:.{digits}f}, {self.high:.{digits}f}]"
        )

    def overlaps(self, other: Interval) -> bool:
        """Whether two intervals overlap.

        Overlapping intervals mean the difference between the two numbers is not
        resolved by the data. This is the check that stops a report concluding
        one model beats another on a difference the test set cannot support.
        """
        return not (self.high < other.low or other.high < self.low)


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Bootstrap a confidence interval for ``metric`` by resampling test compounds.

    Resampling the *observations* is the right unit here: it answers "how much
    would this score move if I had tested a different sample of compounds from
    the same population", which is the uncertainty a reader needs.

    Resamples on which the metric is undefined -- a draw with zero label
    variance, say -- are skipped rather than counted as zero. Scoring them as
    zero would bias the interval downward, and silently so.
    """
    truth = np.asarray(y_true, dtype=float)
    predictions = np.asarray(y_pred, dtype=float)
    if truth.shape != predictions.shape:
        raise ValueError(
            f"{truth.shape[0]} true values against {predictions.shape[0]} predictions"
        )
    if truth.shape[0] < 3:
        raise ValueError(
            "a bootstrap interval from fewer than three observations is not "
            "meaningful; report the raw values instead"
        )

    estimate = metric(truth, predictions)
    rng = random.Random(seed)
    n = truth.shape[0]
    scores: list[float] = []
    for _ in range(n_resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        try:
            scores.append(metric(truth[indices], predictions[indices]))
        except (ValueError, ZeroDivisionError):
            continue

    if len(scores) < n_resamples // 2:
        raise ValueError(
            f"only {len(scores)} of {n_resamples} bootstrap resamples produced a "
            "defined metric; the test set is too small or too degenerate for an "
            "interval"
        )

    tail = (1.0 - confidence) / 2.0 * 100.0
    return Interval(
        estimate=estimate,
        low=float(np.percentile(scores, tail)),
        high=float(np.percentile(scores, 100.0 - tail)),
        confidence=confidence,
    )


@dataclass(frozen=True, slots=True)
class Evaluation:
    """The full result of evaluating one model on one split."""

    n_train: int
    n_test: int
    r2: Interval
    rmse: Interval
    mae: Interval
    spearman: Interval | None
    split_strategy: str
    baselines: BaselineScores | None = None
    leakage_warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        """A report a reviewer can act on.

        Deliberately leads with sample sizes and the split strategy, because a
        metric without those is uninterpretable, and ends with the baseline
        verdict, because that is what decides whether the metric means anything.
        """
        lines = [
            f"Split: {self.split_strategy} | train n={self.n_train}, test n={self.n_test}",
            f"  R2       {self.r2.label()}",
            f"  RMSE     {self.rmse.label()} log units",
            f"  MAE      {self.mae.label()} log units",
        ]
        if self.spearman is not None:
            lines.append(f"  Spearman {self.spearman.label()}")

        if self.n_test < 30:
            lines.append(
                f"  CAUTION: {self.n_test} test compounds. The interval above is "
                "wide for a reason -- treat the point estimate as indicative only."
            )
        if self.r2.width > 0.3:
            lines.append(
                f"  CAUTION: the R2 interval spans {self.r2.width:.2f}. Any model "
                "comparison on a difference smaller than this is unsupported."
            )

        if self.leakage_warnings:
            lines.append("  LEAKAGE CHECK FAILED:")
            lines.extend(f"    {warning}" for warning in self.leakage_warnings)

        if self.baselines is not None:
            lines.append("  Baselines:")
            lines.extend(
                f"    {line}" for line in self.baselines.verdict(self.r2.estimate).split("\n")
            )

        lines.extend(f"  {note}" for note in self.notes)
        return "\n".join(lines)

    @property
    def is_defensible(self) -> bool:
        """Whether this result can be reported as evidence of a working model.

        Conservative on purpose. A result failing this check is not necessarily
        worthless, but it should not be the headline claim in a report or the
        basis for ranking synthesis candidates.
        """
        # Written as a sequence of guard clauses on purpose. Each one is a
        # separate, nameable reason a result is not reportable, and keeping them
        # parallel is what makes the bar auditable at a glance. Ruff's SIM103
        # suggests collapsing the final clause into `return not (...)`, which
        # would make the last check read differently from the five above it for
        # no gain; hence the targeted suppression rather than a rewrite.
        if self.leakage_warnings:
            return False
        if self.n_test < 30:
            return False
        if self.baselines is None:
            return False
        if self.r2.estimate - self.baselines.mean_predictor_r2 <= 0.05:
            return False
        if (
            self.baselines.permutation_r2_p95 is not None
            and self.r2.estimate <= self.baselines.permutation_r2_p95
        ):
            return False
        if (  # noqa: SIM103
            self.baselines.nearest_neighbour_r2 is not None
            and self.r2.estimate <= self.baselines.nearest_neighbour_r2
        ):
            return False
        return True


def evaluate(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_train: int,
    split_strategy: str,
    baselines: BaselineScores | None = None,
    leakage_warnings: Sequence[str] = (),
    n_resamples: int = 1000,
    seed: int = 0,
) -> Evaluation:
    """Score predictions, with bootstrap intervals on every metric."""
    truth = np.asarray(y_test, dtype=float)
    predictions = np.asarray(y_pred, dtype=float)

    notes: list[str] = []
    spearman_interval: Interval | None
    try:
        spearman_interval = bootstrap_metric(
            truth, predictions, spearman, n_resamples=n_resamples, seed=seed
        )
    except ValueError as error:
        spearman_interval = None
        notes.append(f"Spearman not computed: {error}")

    return Evaluation(
        n_train=n_train,
        n_test=len(truth),
        r2=bootstrap_metric(
            truth, predictions, r_squared, n_resamples=n_resamples, seed=seed
        ),
        rmse=bootstrap_metric(
            truth, predictions, rmse, n_resamples=n_resamples, seed=seed
        ),
        mae=bootstrap_metric(truth, predictions, mae, n_resamples=n_resamples, seed=seed),
        spearman=spearman_interval,
        split_strategy=split_strategy,
        baselines=baselines,
        leakage_warnings=tuple(leakage_warnings),
        notes=tuple(notes),
    )


@dataclass(frozen=True, slots=True)
class SplitComparison:
    """Scaffold-split against random-split performance for the same model.

    The gap is the quantity of interest, and it answers a question a single
    number cannot: is this model useful for proposing new chemotypes, or only for
    interpolating within series it already knows?
    """

    scaffold: Evaluation
    random: Evaluation

    @property
    def leakage_gap(self) -> float:
        """How much R-squared the random split gains from analogue leakage."""
        return self.random.r2.estimate - self.scaffold.r2.estimate

    def describe(self) -> str:
        lines = [
            "Scaffold split (the honest estimate):",
            self.scaffold.describe(),
            "",
            "Random split (optimistic baseline, reported for comparison only):",
            self.random.describe(),
            "",
            f"Leakage gap: {self.leakage_gap:+.3f} R2.",
        ]
        if self.scaffold.r2.overlaps(self.random.r2):
            lines.append(
                "  The two intervals overlap, so this dataset does not resolve a "
                "difference between the splits -- which is itself usually a sign "
                "the test sets are too small."
            )
        elif self.leakage_gap > 0.2:
            lines.append(
                "  A gap this large means most of the random-split score came "
                "from analogues shared with training. Any performance figure "
                "quoted from a random split would substantially overstate what "
                "this model does on new chemotypes."
            )
        else:
            lines.append(
                "  A modest gap: the model's performance does not depend heavily "
                "on seeing analogues of the test compounds during training."
            )
        lines.append(
            "  Report the scaffold-split figure. The random-split figure exists "
            "to show what would have been claimed without it."
        )
        return "\n".join(lines)
