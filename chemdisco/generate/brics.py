"""Candidate generation by BRICS fragment recombination.

What this is, stated plainly, because the honest description is less impressive
than the phrase "AI drug discovery" and considerably more useful:

BRICS -- Breaking of Retrosynthetically Interesting Chemical Substructures, Degen
and colleagues, 2008 -- cuts molecules at sixteen bond types that correspond to
known, reliable synthetic reactions. Each cut leaves a labelled attachment point,
and the labels encode which fragment ends may legitimately be joined. Recombining
fragments from a set of known active compounds therefore produces structures that
are plausibly *makeable*, which is the main failure of unconstrained generative
models: they propose molecules nobody can synthesise.

**What this method can do.** Explore combinations of the structural features
present in compounds already known to hit a target. If one active contributes a
hinge-binding heterocycle and another a solubilising side chain, BRICS can propose
the combination. That is a real and standard medicinal-chemistry move, and it
regularly produces viable starting points.

**What this method cannot do.** Invent a fragment that was not in the input. The
output lives inside the chemical space spanned by the input actives; it will not
discover a novel chemotype, and any claim that it might is false. For a target
with few known actives -- the situation for most hard diseases -- the accessible
space is correspondingly small.

**The structural conflict with scoring.** The more novel a generated candidate,
the further it sits from the QSAR model's training data, and the less its
predicted activity means. This is not a flaw to be engineered away; it is
intrinsic. The pipeline here handles it by reporting the applicability-domain
verdict for every candidate and refusing to rank on out-of-domain predictions, so
that a list of exotic structures with impressive predicted potency cannot be
mistaken for a result.

Generation is therefore the easy part. The filters, the novelty check and the
domain accounting are what make the output meaningful, and each stage's attrition
is reported: a run that proposes 10,000 structures and retains 12 has told you
something important about the fragment set.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

from ..chem.alerts import AlertReport, screen_alerts, synthetic_accessibility
from ..chem.similarity import NoveltyVerdict, assess_novelty, diverse_subset
from ..chem.standardize import RDKIT_AVAILABLE, inchikey_of, require_rdkit, standardize
from ..provenance import Quantity

if RDKIT_AVAILABLE:  # pragma: no branch
    from rdkit import Chem
    from rdkit.Chem import BRICS

#: Hard ceiling on structures drawn from the BRICS builder. The builder is an
#: effectively unbounded generator -- a few dozen fragments span combinatorially
#: many products -- so an explicit cap is required, not optional.
DEFAULT_MAX_GENERATED = 5_000


@dataclass(frozen=True, slots=True)
class GenerationPolicy:
    """Filters applied to generated structures, and the reason for each.

    Attributes:
        max_generated: Ceiling on structures drawn from the builder.
        min_heavy_atoms: Reject fragments-sized products. Below about 12 heavy
            atoms a molecule is a fragment, not a lead candidate.
        max_heavy_atoms: Reject products that grew implausibly large. BRICS
            recombination readily produces 80-atom monsters by chaining fragments.
        max_sascore: Reject hard-to-synthesise products. 6.0 is a common cut;
            above it a chemist's opinion is needed before committing.
        reject_pains: Drop PAINS matches. Worth remembering these filters produce
            false positives, so this is set-level triage rather than a verdict on
            any individual structure.
        reject_brenk: Drop Brenk alert matches.
        require_novelty: Drop products already present in the reference set.
        near_duplicate_threshold: Similarity at or above which a product counts as
            a minor variation on known chemistry rather than a new structure.
        diversity_threshold: Collapse the surviving set so no two members exceed
            this similarity. Fifty variations on one molecule are not fifty
            candidates.
    """

    max_generated: int = DEFAULT_MAX_GENERATED
    min_heavy_atoms: int = 12
    max_heavy_atoms: int = 50
    max_sascore: float | None = 6.0
    reject_pains: bool = True
    reject_brenk: bool = True
    require_novelty: bool = True
    near_duplicate_threshold: float = 0.85
    diversity_threshold: float = 0.85

    def describe(self) -> str:
        return (
            f"max_generated={self.max_generated}; "
            f"heavy_atoms={self.min_heavy_atoms}-{self.max_heavy_atoms}; "
            f"max_sascore={self.max_sascore}; "
            f"reject_pains={self.reject_pains}; reject_brenk={self.reject_brenk}; "
            f"require_novelty={self.require_novelty}; "
            f"diversity_threshold={self.diversity_threshold}"
        )


@dataclass(frozen=True, slots=True)
class Candidate:
    """One generated structure with everything known about it.

    Attributes:
        smiles: Standardised canonical SMILES.
        inchikey: Identity hash.
        sascore: Synthetic accessibility, a heuristic quantity.
        alerts: Structural-alert report.
        novelty: Novelty verdict against the reference set.
        predicted_activity: QSAR prediction, attached later by the scoring step.
            ``None`` until a model has seen it.
        parent_fragments: Which input fragments it was assembled from, so a
            chemist can see where it came from.
    """

    smiles: str
    inchikey: str | None
    sascore: Quantity
    alerts: AlertReport
    novelty: NoveltyVerdict | None = None
    predicted_activity: Quantity | None = None
    parent_fragments: tuple[str, ...] = ()

    @property
    def is_rankable(self) -> bool:
        """Whether this candidate may be ordered by its predicted activity.

        Requires a prediction that is both known and inside the model's
        applicability domain. Everything else is reported but not ranked.
        """
        return (
            self.predicted_activity is not None
            and self.predicted_activity.is_trustworthy_for_ranking
        )

    def summary(self) -> str:
        lines = [f"{self.smiles}"]
        if self.predicted_activity is not None:
            lines.append(f"  predicted activity: {self.predicted_activity.label()}")
            if not self.is_rankable:
                lines.append(
                    "  NOT RANKED: the prediction is outside the model's "
                    "applicability domain, so its magnitude carries no weight"
                )
        lines.append(f"  synthetic accessibility: {self.sascore.label(digits=2)}")
        lines.append(f"  alerts: {self.alerts.describe()}")
        if self.novelty is not None:
            lines.append(f"  novelty: {self.novelty.describe()}")
        if self.parent_fragments:
            lines.append(
                f"  assembled from {len(self.parent_fragments)} input fragments"
            )
        return "\n".join(lines)


@dataclass(slots=True)
class GenerationReport:
    """What generation produced, and what each filter removed.

    The attrition counts are the most informative part. A run that generates
    5,000 structures and keeps 8 has not failed -- it has reported that the
    fragment set is small or the filters are strict, which is actionable.
    Presenting only the 8 survivors would hide that.
    """

    candidates: list[Candidate] = field(default_factory=list)
    n_input_actives: int = 0
    n_fragments: int = 0
    n_generated: int = 0
    attrition: dict[str, int] = field(default_factory=dict)
    policy_audit: "PolicyAudit | None" = None
    notes: list[str] = field(default_factory=list)

    def record(self, stage: str) -> None:
        self.attrition[stage] = self.attrition.get(stage, 0) + 1

    def describe(self) -> str:
        lines = [
            f"Input: {self.n_input_actives} known actives, decomposed into "
            f"{self.n_fragments} distinct BRICS fragments.",
            f"Generated: {self.n_generated} structures from fragment recombination.",
        ]
        if self.attrition:
            lines.append("Removed:")
            for stage, count in sorted(
                self.attrition.items(), key=lambda kv: -kv[1]
            ):
                lines.append(f"  {count:>6} by {stage}")
        lines.append(f"Retained: {len(self.candidates)} candidates.")

        if self.policy_audit is not None:
            lines.append("")
            lines.append(self.policy_audit.describe())
            lines.append("")

        if self.n_generated and not self.candidates:
            if self.policy_audit is not None and self.policy_audit.policy_is_suspect:
                lines.append(
                    "WARNING: every generated structure was filtered out, and the "
                    "audit above shows the same filters would reject most of the "
                    "known actives. The policy is wrong for this target class, not "
                    "the candidates. Fix it to fit the chemistry rather than "
                    "loosening it until something survives."
                )
            else:
                lines.append(
                    "WARNING: every generated structure was filtered out, and the "
                    "filters do accept the known actives -- so this is not a "
                    "mis-calibrated policy. Either the fragment set is too small "
                    "to recombine usefully, or recombination genuinely produces "
                    "nothing viable here. Both are findings, not failures."
                )
        elif self.n_generated and len(self.candidates) / self.n_generated < 0.01:
            lines.append(
                f"NOTE: retention is "
                f"{len(self.candidates) / self.n_generated:.2%}. That is normal for "
                "BRICS recombination but means the surviving set is a narrow slice "
                "of what was explored."
            )

        rankable = sum(1 for c in self.candidates if c.is_rankable)
        if self.candidates and any(
            c.predicted_activity is not None for c in self.candidates
        ):
            lines.append(
                f"Of the retained candidates, {rankable} have predictions inside "
                f"the model's applicability domain and {len(self.candidates) - rankable} "
                "do not. Only the former may be ordered by predicted activity."
            )
            if rankable == 0:
                lines.append(
                    "  Every prediction is extrapolation. This is the expected "
                    "tension between novelty and model reliability: the candidates "
                    "are new enough that the model has no basis for scoring them. "
                    "Ranking them would be inventing an ordering."
                )

        lines.extend(self.notes)
        return "\n".join(lines)

    def ranked(self) -> list[Candidate]:
        """Rankable candidates, most potent predicted first.

        Candidates without a trustworthy prediction are excluded entirely rather
        than sorted to the bottom, because a mixed list invites reading position
        as evidence.
        """
        rankable = [c for c in self.candidates if c.is_rankable]
        return sorted(
            rankable,
            key=lambda c: c.predicted_activity.require(),  # type: ignore[union-attr]
            reverse=True,
        )


def decompose_to_fragments(smiles_list: Sequence[str]) -> tuple[set[str], list[str]]:
    """Decompose known actives into BRICS fragments.

    Returns:
        ``(fragments, errors)``. Fragments carry their BRICS attachment-point
        labels, which is what constrains recombination to synthetically sensible
        joins; stripping the labels would permit arbitrary bonds.
    """
    require_rdkit()
    fragments: set[str] = set()
    errors: list[str] = []

    for smiles in smiles_list:
        result = standardize(smiles)
        if not result.ok or result.smiles is None:
            errors.append(f"{smiles[:60]}: {result.error}")
            continue
        mol = Chem.MolFromSmiles(result.smiles)
        if mol is None:
            errors.append(f"{result.smiles[:60]}: standardised SMILES failed to parse")
            continue
        try:
            fragments.update(BRICS.BRICSDecompose(mol))
        except Exception as error:
            errors.append(f"{result.smiles[:60]}: BRICS decomposition failed: {error}")

    return fragments, errors


def _build_products(
    fragments: Sequence[str], *, max_generated: int, seed: int
) -> Iterator[str]:
    """Draw products from the BRICS builder, capped and shuffled.

    The builder enumerates in a fixed order that systematically favours
    combinations of whichever fragments come first, so an uncapped prefix of its
    output is a biased sample of the space. Shuffling the fragment list before
    building spreads the sample across the fragment set.

    The randomness here selects which structures to *propose*. It never touches a
    score or a measurement, which is why this module appears on the randomness
    allowlist in ``tests/test_no_fabrication.py``.
    """
    require_rdkit()
    rng = random.Random(seed)
    shuffled = list(fragments)
    rng.shuffle(shuffled)

    mols = []
    for fragment in shuffled:
        mol = Chem.MolFromSmiles(fragment)
        if mol is not None:
            mols.append(mol)

    if len(mols) < 2:
        return

    count = 0
    try:
        for product in BRICS.BRICSBuild(mols, scrambleReagents=True, maxDepth=3):
            if count >= max_generated:
                return
            try:
                product.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(product)
                yield Chem.MolToSmiles(product)
            except Exception:
                # Recombination can produce chemically invalid structures; they
                # are skipped silently here and counted by the caller.
                continue
            count += 1
    except Exception:
        return


@dataclass(frozen=True, slots=True)
class PolicyAudit:
    """How the filter policy treats the *known actives* it was pointed at.

    The question this answers: would this policy have rejected the compounds
    that are already known to work?

    It exists because the first live BACE1 run discarded all 1500 generated
    structures -- 1000 of them on Brenk alerts -- and the obvious response, to
    loosen the filters until something survives, is the wrong one. It tunes the
    policy until it produces output rather than until it is correct.

    The right question is measurable. BACE1 inhibitors are large peptidomimetics
    carrying amidine and guanidine groups, and the Brenk set flags exactly those.
    Brenk was assembled to triage HTS decks for lead-likeness; applied to a target
    class whose genuine actives contain those motifs, it rejects the right
    chemistry. If most known actives fail the policy, the policy is wrong for this
    target -- and that is a fact about the filter, not an opinion about the
    candidates.

    Attributes:
        n_actives: How many input structures were audited.
        n_passing: How many would survive the policy.
        rejections: Reason counts, in the same vocabulary as generation attrition.
        failing_examples: A few rejected actives with their reasons.
    """

    n_actives: int
    n_passing: int
    rejections: dict[str, int]
    failing_examples: tuple[tuple[str, str], ...] = ()

    @property
    def pass_rate(self) -> float | None:
        if self.n_actives == 0:
            return None
        return self.n_passing / self.n_actives

    #: Pass rate below which the policy is treated as mis-calibrated.
    #:
    #: Set at 0.8, not 0.5. The first live BACE1 audit returned 52% -- 18 of 40
    #: known actives rejected, almost all on Brenk alerts -- and a half threshold
    #: called that acceptable. It is not. A structural-alert set exists to remove
    #: compounds unlikely to become drugs; one that rejects two in five compounds
    #: already proven to hit the target is measuring the wrong thing for this
    #: chemistry. Twenty percent attrition on known actives is about the most a
    #: filter can take before it is shaping the output more than the data is.
    SUSPECT_PASS_RATE: ClassVar[float] = 0.8

    @property
    def policy_is_suspect(self) -> bool:
        """Whether the policy rejects enough known actives to be unusable here."""
        rate = self.pass_rate
        return rate is not None and rate < self.SUSPECT_PASS_RATE

    @property
    def dominant_rule(self) -> str | None:
        """The single filter responsible for the most rejected actives."""
        if not self.rejections:
            return None
        return max(self.rejections.items(), key=lambda kv: kv[1])[0]

    def describe(self) -> str:
        if self.n_actives == 0:
            return "no actives audited"
        rate = self.pass_rate or 0.0
        lines = [
            f"Policy audit: {self.n_passing} of {self.n_actives} known actives "
            f"({rate:.0%}) would survive these filters."
        ]
        for rule, count in sorted(self.rejections.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {count:>4} known active(s) rejected by {rule}")
        if self.policy_is_suspect:
            rule = self.dominant_rule
            lines.append(
                f"  WARNING: {1 - rate:.0%} of the compounds already known to hit "
                "this target would be rejected by these filters"
                + (f", mostly by '{rule}'" if rule else "")
                + ". The policy is mis-calibrated for this chemistry, so the "
                "generation attrition above says more about the filters than "
                "about the candidates. Fix the policy to fit the target class -- "
                "do not loosen it until something survives."
            )
            if rule == "Brenk alert":
                lines.append(
                    "    Brenk was assembled to triage HTS decks for "
                    "lead-likeness. Target classes whose genuine actives carry "
                    "flagged motifs -- amidines and guanidines in aspartyl "
                    "protease inhibitors, for one -- need it switched off rather "
                    "than trusted."
                )
            for smiles, reason in self.failing_examples:
                lines.append(f"    {smiles[:70]} -> {reason}")
        return "\n".join(lines)


def audit_policy(
    known_actives: Sequence[str], policy: GenerationPolicy | None = None
) -> PolicyAudit:
    """Run the generation filters over the known actives themselves.

    Novelty is excluded from the audit: a known active is by definition already
    known, so testing it against the novelty filter would reject every one of them
    and tell you nothing. Everything else -- size, alerts, synthetic accessibility
    -- applies unchanged.
    """
    require_rdkit()
    policy = policy or GenerationPolicy()

    passing = 0
    rejections: dict[str, int] = {}
    failures: list[tuple[str, str]] = []

    def reject(smiles: str, rule: str) -> None:
        rejections[rule] = rejections.get(rule, 0) + 1
        if len(failures) < 5:
            failures.append((smiles, rule))

    for smiles in known_actives:
        result = standardize(smiles)
        if not result.ok or result.smiles is None:
            reject(smiles, "failed standardisation")
            continue

        mol = Chem.MolFromSmiles(result.smiles)
        if mol is None:
            reject(smiles, "failed standardisation")
            continue

        heavy = mol.GetNumHeavyAtoms()
        if heavy < policy.min_heavy_atoms:
            reject(result.smiles, f"too small (under {policy.min_heavy_atoms} heavy atoms)")
            continue
        if heavy > policy.max_heavy_atoms:
            reject(result.smiles, f"too large (over {policy.max_heavy_atoms} heavy atoms)")
            continue

        alerts = screen_alerts(result.smiles)
        if policy.reject_pains and alerts.pains:
            reject(result.smiles, "PAINS alert")
            continue
        if policy.reject_brenk and alerts.brenk:
            reject(result.smiles, "Brenk alert")
            continue

        sascore = synthetic_accessibility(result.smiles)
        if (
            policy.max_sascore is not None
            and sascore.is_known
            and sascore.require() > policy.max_sascore
        ):
            reject(result.smiles, f"synthetic accessibility above {policy.max_sascore}")
            continue

        passing += 1

    return PolicyAudit(
        n_actives=len(known_actives),
        n_passing=passing,
        rejections=rejections,
        failing_examples=tuple(failures),
    )


def generate_candidates(
    known_actives: Sequence[str],
    *,
    policy: GenerationPolicy | None = None,
    reference_smiles: Sequence[str] | None = None,
    seed: int = 0,
) -> GenerationReport:
    """Generate and filter candidate structures from known actives.

    Args:
        known_actives: SMILES of compounds known to hit the target. These supply
            the fragment vocabulary, so the output is bounded by their chemistry.
        policy: Filter settings; defaults to :class:`GenerationPolicy`.
        reference_smiles: Structures to judge novelty against. Defaults to
            ``known_actives``, which catches rediscovery of the inputs but not of
            other known compounds -- pass a wider set, such as a ChEMBL slice, for
            a meaningful novelty claim.
        seed: Reproducibility of the fragment shuffle.

    Returns:
        A :class:`GenerationReport`. Candidates carry no activity prediction yet;
        apply a model via :func:`score_candidates`.
    """
    require_rdkit()
    policy = policy or GenerationPolicy()
    reference = list(reference_smiles) if reference_smiles is not None else list(
        known_actives
    )

    report = GenerationReport(n_input_actives=len(known_actives))
    # Audited before anything is generated: attrition figures are not
    # interpretable without knowing whether the policy accepts the chemistry it
    # is being pointed at.
    report.policy_audit = audit_policy(known_actives, policy)

    fragments, decomposition_errors = decompose_to_fragments(known_actives)
    report.n_fragments = len(fragments)
    if decomposition_errors:
        report.notes.append(
            f"{len(decomposition_errors)} input structure(s) could not be "
            "decomposed; see the first few: "
            + "; ".join(decomposition_errors[:3])
        )

    if len(fragments) < 2:
        report.notes.append(
            f"only {len(fragments)} fragment(s) obtained from "
            f"{len(known_actives)} actives. BRICS recombination needs at least "
            "two fragments with compatible attachment points. With this few known "
            "actives, fragment recombination is not a usable approach for this "
            "target -- which is itself a finding worth reporting."
        )
        return report

    known_keys = {
        key for key in (inchikey_of(s) for s in reference) if key is not None
    }

    seen: set[str] = set()
    survivors: list[Candidate] = []

    for product_smiles in _build_products(
        sorted(fragments), max_generated=policy.max_generated, seed=seed
    ):
        report.n_generated += 1

        result = standardize(product_smiles)
        if not result.ok or result.smiles is None:
            report.record("failed standardisation")
            continue

        if result.inchikey is not None and result.inchikey in seen:
            report.record("duplicate of another generated structure")
            continue
        if result.inchikey is not None:
            seen.add(result.inchikey)

        mol = Chem.MolFromSmiles(result.smiles)
        if mol is None:
            report.record("failed standardisation")
            continue

        heavy = mol.GetNumHeavyAtoms()
        if heavy < policy.min_heavy_atoms:
            report.record(f"too small (under {policy.min_heavy_atoms} heavy atoms)")
            continue
        if heavy > policy.max_heavy_atoms:
            report.record(f"too large (over {policy.max_heavy_atoms} heavy atoms)")
            continue

        if policy.require_novelty and result.inchikey in known_keys:
            report.record("already a known compound")
            continue

        alerts = screen_alerts(result.smiles)
        if policy.reject_pains and alerts.pains:
            report.record("PAINS alert")
            continue
        if policy.reject_brenk and alerts.brenk:
            report.record("Brenk alert")
            continue

        sascore = synthetic_accessibility(result.smiles)
        # A candidate whose SAscore could not be computed is kept, not dropped.
        # Rejecting on an unavailable value would silently discard the whole
        # library whenever RDKit's SA_Score contrib module is missing, and an
        # absent score is not evidence of a hard synthesis.
        if (
            policy.max_sascore is not None
            and sascore.is_known
            and sascore.require() > policy.max_sascore
        ):
            report.record(f"synthetic accessibility above {policy.max_sascore}")
            continue

        survivors.append(
            Candidate(
                smiles=result.smiles,
                inchikey=result.inchikey,
                sascore=sascore,
                alerts=alerts,
            )
        )

    if not survivors:
        report.candidates = []
        return report

    novelty_verdicts = assess_novelty(
        [c.smiles for c in survivors],
        reference,
        near_duplicate_threshold=policy.near_duplicate_threshold,
    )
    with_novelty: list[Candidate] = []
    for candidate, verdict in zip(survivors, novelty_verdicts, strict=True):
        if policy.require_novelty and not verdict.is_novel:
            report.record("too similar to known chemistry")
            continue
        with_novelty.append(
            Candidate(
                smiles=candidate.smiles,
                inchikey=candidate.inchikey,
                sascore=candidate.sascore,
                alerts=candidate.alerts,
                novelty=verdict,
            )
        )

    if not with_novelty:
        report.candidates = []
        return report

    keep = set(
        diverse_subset(
            [c.smiles for c in with_novelty], threshold=policy.diversity_threshold
        )
    )
    for index, candidate in enumerate(with_novelty):
        if index in keep:
            report.candidates.append(candidate)
        else:
            report.record("near-duplicate of a retained candidate")

    return report


def score_candidates(
    report: GenerationReport,
    model,
    feature_fn,
    *,
    training_smiles: Sequence[str] | None = None,
) -> GenerationReport:
    """Attach QSAR predictions to a report's candidates.

    Args:
        report: Output of :func:`generate_candidates`.
        model: A fitted :class:`chemdisco.qsar.model.QSARModel`.
        feature_fn: Takes SMILES and returns a feature matrix in the model's
            column order. Injected so this module need not know which
            representation the model was fitted on.
        training_smiles: The model's training structures. When supplied, each
            candidate's similarity to its nearest training compound sharpens the
            applicability-domain verdict considerably -- and for generated
            candidates that verdict is the difference between a result and a
            ranked list of extrapolations.

    Returns:
        The same report with predictions attached.
    """
    if not report.candidates:
        return report

    candidate_smiles = [c.smiles for c in report.candidates]
    features = feature_fn(candidate_smiles)

    max_similarity = None
    if training_smiles:
        from ..chem.similarity import max_similarity_to_reference

        max_similarity, _ = max_similarity_to_reference(
            candidate_smiles, list(training_smiles)
        )
    else:
        report.notes.append(
            "No training structures were supplied, so the applicability domain "
            "was judged on descriptor distance alone. Fingerprint similarity to "
            "the training set is the stronger signal for generated structures; "
            "pass training_smiles to use it."
        )

    predictions = model.predict(features, max_similarity=max_similarity)

    report.candidates = [
        Candidate(
            smiles=candidate.smiles,
            inchikey=candidate.inchikey,
            sascore=candidate.sascore,
            alerts=candidate.alerts,
            novelty=candidate.novelty,
            predicted_activity=prediction,
            parent_fragments=candidate.parent_fragments,
        )
        for candidate, prediction in zip(report.candidates, predictions, strict=True)
    ]
    return report
