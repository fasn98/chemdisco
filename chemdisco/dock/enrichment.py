"""Enrichment: whether a docking score separates actives from decoys at all.

The question that decides whether docking is worth running on a target. A
scoring function that cannot rank known actives above property-matched decoys
adds nothing to a candidate triage, and running it anyway produces a ranked list
whose order is noise.

The metrics, and what each is for:

**AUC-ROC.** The probability that a randomly chosen active outscores a randomly
chosen decoy. 0.5 is random; 1.0 is perfect separation. Published docking AUCs
across targets mostly fall between 0.6 and 0.75, and the distribution is wide --
some targets reach 0.9, others sit at chance. AUC weights the whole ranking
equally, which is its weakness here: a virtual screen only ever buys compounds
from the top of the list.

**Enrichment factor at 1%.** How many times more actives appear in the top 1% of
the ranking than chance would give. This measures what a screen actually uses.
Its weakness is the mirror of AUC's: on a few hundred compounds the top 1% is
one or two molecules, so EF1% is extremely noisy and a single lucky active
swings it by a factor of several. Both are reported, with intervals, because
either alone misleads.

**BEDROC.** Weights early recognition continuously rather than at an arbitrary
cutoff, which avoids EF's cliff-edge sensitivity. Its alpha parameter sets how
steeply; the conventional 20.0 concentrates roughly 80% of the weight in the top
8%.

Every metric is reported with a bootstrap confidence interval. On the set sizes
these screens produce, the interval on AUC routinely spans 0.1, which is wider
than the difference between a useful screen and a useless one -- so a point
estimate alone cannot support the conclusion it appears to.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ..qsar.evaluate import Interval, bootstrap_metric

#: Alpha for BEDROC. 20.0 is conventional and puts about 80% of the weight in
#: the top 8% of the ranking.
DEFAULT_BEDROC_ALPHA = 20.0


def auc_roc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve, computed from rank statistics.

    ``scores`` are docking scores, where **lower is better** -- Vina reports
    binding free energies, so -10 beats -6. They are negated internally so the
    usual "higher is better" convention applies to the computation.

    Implemented through the Mann-Whitney U identity rather than by integrating a
    curve, which handles ties exactly: tied scores contribute 0.5 each, and ties
    are common when a scoring function reports to two decimal places.
    """
    labels = np.asarray(labels, dtype=int)
    values = -np.asarray(scores, dtype=float)

    n_active = int(np.sum(labels == 1))
    n_decoy = int(np.sum(labels == 0))
    if n_active == 0 or n_decoy == 0:
        raise ValueError(
            "AUC needs both actives and decoys; "
            f"got {n_active} active(s) and {n_decoy} decoy(s)"
        )

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # Average ranks within ties, which is what makes tied scores contribute 0.5.
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for index, count in enumerate(counts):
        if count > 1:
            mask = inverse == index
            ranks[mask] = ranks[mask].mean()

    rank_sum = float(np.sum(ranks[labels == 1]))
    u_statistic = rank_sum - n_active * (n_active + 1) / 2.0
    return u_statistic / (n_active * n_decoy)


def enrichment_factor(
    labels: np.ndarray, scores: np.ndarray, *, fraction: float = 0.01
) -> float:
    """How many times more actives appear in the top ``fraction`` than by chance.

    An EF of 1.0 is random. The theoretical maximum is ``1 / active_fraction``,
    reached when every compound in the top slice is an active -- so on a set that
    is 10% actives, EF1% cannot exceed 10 however good the screen.
    """
    labels = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=float)
    total = len(labels)
    n_active = int(np.sum(labels == 1))
    if n_active == 0:
        raise ValueError("no actives in the set")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    n_top = max(1, int(round(fraction * total)))
    order = np.argsort(values, kind="mergesort")  # lower score first: best first
    found = int(np.sum(labels[order[:n_top]] == 1))
    expected = n_active * n_top / total
    return found / expected if expected > 0 else 0.0


def max_enrichment_factor(labels: np.ndarray, *, fraction: float = 0.01) -> float:
    """The best EF this set size allows, for reading an EF against its ceiling."""
    labels = np.asarray(labels, dtype=int)
    total = len(labels)
    n_active = int(np.sum(labels == 1))
    n_top = max(1, int(round(fraction * total)))
    expected = n_active * n_top / total
    return min(n_top, n_active) / expected if expected > 0 else 0.0


def bedroc(
    labels: np.ndarray, scores: np.ndarray, *, alpha: float = DEFAULT_BEDROC_ALPHA
) -> float:
    """Boltzmann-enhanced discrimination of ROC: early recognition, continuously.

    Avoids the arbitrary cutoff of an enrichment factor by weighting each active
    by an exponential in its rank. Returns a value in [0, 1] where the random
    expectation is close to the active fraction rather than to 0.5.
    """
    labels = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=float)
    total = len(labels)
    n_active = int(np.sum(labels == 1))
    if n_active == 0 or n_active == total:
        raise ValueError("BEDROC needs both actives and decoys")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    order = np.argsort(values, kind="mergesort")
    ranked = labels[order]
    ratio = n_active / total

    # Sum of the exponential weights over the actives' ranks.
    active_ranks = np.where(ranked == 1)[0] + 1
    weighted = float(np.sum(np.exp(-alpha * active_ranks / total)))

    random_expectation = (
        ratio * (1 - math.exp(-alpha)) / (math.exp(alpha / total) - 1)
    )
    if random_expectation == 0:
        return 0.0

    scaled = weighted / random_expectation
    factor = (
        ratio
        * math.sinh(alpha / 2.0)
        / (math.cosh(alpha / 2.0) - math.cosh(alpha / 2.0 - alpha * ratio))
    )
    offset = 1.0 / (1.0 - math.exp(alpha * (1.0 - ratio)))
    return scaled * factor + offset


@dataclass(frozen=True, slots=True)
class EnrichmentResult:
    """Whether, and how well, docking separated the two groups.

    Attributes:
        n_actives: Actives scored.
        n_decoys: Decoys scored.
        auc: AUC-ROC with a bootstrap interval.
        ef1: Enrichment factor in the top 1%.
        ef5: Enrichment factor in the top 5%.
        bedroc: BEDROC at the default alpha.
        max_ef1: The ceiling EF1 this set size allows.
        property_gap_warning: Carried from the decoy selection, because an
            enrichment number cannot be read without knowing whether the groups
            were property-matched.
    """

    n_actives: int
    n_decoys: int
    auc: Interval
    ef1: float
    ef5: float
    bedroc: float
    max_ef1: float
    property_gap_warning: str = ""

    #: AUC interval width above which a screen cannot settle the question either
    #: way. An interval spanning 0.3 reaches from "worse than random" to "good",
    #: and no verdict drawn from it is supported.
    INCONCLUSIVE_INTERVAL_WIDTH: ClassVar[float] = 0.30

    @property
    def separates(self) -> bool:
        """Whether the screen demonstrably beats random selection.

        The test is on the interval's lower bound, not the point estimate. An
        AUC of 0.62 whose interval reaches below 0.5 is not evidence of anything.
        """
        return self.auc.low > 0.5

    @property
    def is_conclusive(self) -> bool:
        """Whether this screen can settle the question at all.

        The distinction that matters, and the one the first version of this
        class got wrong. ``separates`` returning False covers two completely
        different situations: a screen that measured no signal, and a screen too
        small to detect one. Reporting the second as the first turns absence of
        evidence into evidence of absence.

        A BACE1 run docking 8 actives against 8 decoys produced AUC 0.641
        [0.317, 0.900] -- an interval reaching from well below random to strong.
        The honest statement is that the run answered nothing.
        """
        if self.separates:
            return True
        return self.auc.width <= self.INCONCLUSIVE_INTERVAL_WIDTH

    @property
    def verdict(self) -> str:
        """``"separates"``, ``"does not separate"``, or ``"inconclusive"``."""
        if self.separates:
            return "separates"
        if self.is_conclusive:
            return "does not separate"
        return "inconclusive"

    def compounds_needed(self, *, target_width: float = 0.20) -> int | None:
        """Roughly how many compounds per group would narrow the interval enough.

        A bootstrap interval on AUC narrows as the square root of the group
        size, so a run with an interval twice as wide as wanted needs about four
        times the compounds. Approximate by construction -- it extrapolates from
        one observed width -- but it is the difference between "run more" and a
        guess about how many more.
        """
        if self.auc.width <= target_width:
            return None
        smaller_group = min(self.n_actives, self.n_decoys)
        if smaller_group == 0:
            return None
        factor = (self.auc.width / target_width) ** 2
        return int(math.ceil(smaller_group * factor))

    def describe(self) -> str:
        lines = [
            f"{self.n_actives} actives against {self.n_decoys} property-matched decoys",
            f"  AUC-ROC  {self.auc.label()}   (0.5 = random)",
            f"  EF 1%    {self.ef1:.2f}   (1.0 = random, ceiling {self.max_ef1:.1f})",
            f"  EF 5%    {self.ef5:.2f}",
            f"  BEDROC   {self.bedroc:.3f}",
        ]

        if self.property_gap_warning:
            lines.append("  " + self.property_gap_warning)

        if self.verdict == "inconclusive":
            needed = self.compounds_needed()
            lines.append(
                f"\n  VERDICT: INCONCLUSIVE. The AUC interval spans "
                f"{self.auc.width:.2f} ({self.auc.low:.3f} to {self.auc.high:.3f}), "
                "reaching from worse than random to good. This screen is too "
                "small to settle the question either way.\n"
                "  This is NOT evidence that docking fails here. Absence of "
                "evidence is not evidence of absence, and reporting it as such "
                "would be the error this pipeline exists to avoid."
            )
            if needed:
                lines.append(
                    f"  To narrow the interval to 0.20 would take roughly "
                    f"{needed} compounds per group, against the "
                    f"{min(self.n_actives, self.n_decoys)} used here."
                )
        elif not self.separates:
            lines.append(
                f"\n  VERDICT: the AUC interval is {self.auc.label()}, tight "
                "enough to conclude and centred at or below random. This screen "
                "does not rank actives above property-matched decoys, so docking "
                "scores should not be used to triage candidates on this target. "
                "That is a result, not a malfunction -- docking enrichment "
                "genuinely fails on many targets, and knowing it is worth more "
                "than a ranked list whose order is noise."
            )
        elif self.auc.estimate < 0.7:
            lines.append(
                f"\n  VERDICT: AUC {self.auc.estimate:.3f} beats random but is "
                "modest, which is typical for docking. Useful for discarding the "
                "clearly implausible; not reliable enough to order a shortlist."
            )
        else:
            lines.append(
                f"\n  VERDICT: AUC {self.auc.estimate:.3f} is strong for docking. "
                "The score carries real information about this site, though "
                "ranking within the top of the list remains unreliable -- see the "
                "pose-ranking result in the docking module."
            )

        n_top = max(1, int(round(0.01 * (self.n_actives + self.n_decoys))))
        if n_top < 5:
            lines.append(
                f"  Note: the top 1% is {n_top} compound(s), so EF1% here moves by "
                "a large factor on a single result. Read the AUC interval instead."
            )
        return "\n".join(lines)


def analyse_enrichment(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    property_gap_warning: str = "",
    n_resamples: int = 2000,
    seed: int = 0,
) -> EnrichmentResult:
    """Score a virtual screen against its decoy control.

    Args:
        labels: 1 for an active, 0 for a decoy.
        scores: Docking scores, lower being better.
        property_gap_warning: Any warning from the decoy selection, carried into
            the report so an enrichment figure is never read without it.
        n_resamples: Bootstrap resamples for the AUC interval.
        seed: Reproducibility.
    """
    label_array = np.asarray(labels, dtype=int)
    score_array = np.asarray(scores, dtype=float)
    if label_array.shape != score_array.shape:
        raise ValueError(
            f"{len(label_array)} labels against {len(score_array)} scores"
        )

    return EnrichmentResult(
        n_actives=int(np.sum(label_array == 1)),
        n_decoys=int(np.sum(label_array == 0)),
        auc=bootstrap_metric(
            label_array, score_array, auc_roc, n_resamples=n_resamples, seed=seed
        ),
        ef1=enrichment_factor(label_array, score_array, fraction=0.01),
        ef5=enrichment_factor(label_array, score_array, fraction=0.05),
        bedroc=bedroc(label_array, score_array),
        max_ef1=max_enrichment_factor(label_array, fraction=0.01),
        property_gap_warning=property_gap_warning,
    )
