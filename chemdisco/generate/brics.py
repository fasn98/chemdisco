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
from collections.abc import Sequence
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
        anchor_smarts: SMARTS for the motif that engages the target, measured
            rather than assumed -- in practice the most discriminating feature
            from :mod:`chemdisco.generate.pharmacophore`. Supplying it turns on the
            constraint described in :func:`_build_products`: fragments carrying the
            motif become BRICS *seeds* and only motif-free fragments are offered as
            reagents, so a product grows from exactly one anchor.

            This exists because of a measured result, not a worry. On BACE1, every
            one of the 15 candidates that reached the reference median ligand
            efficiency carried the amidine twice -- 15 of 15. In a fragment space
            built from potent inhibitors, joining two warheads is simply how the
            builder makes a compact molecule that scores well, since Vina rewards
            buried polar contacts and each warhead brings its own.
        max_anchor_copies: Reject a product carrying the motif more often than
            this. The seeding above makes it structurally unlikely rather than
            impossible -- joining two motif-free fragments can create the motif
            across the new bond -- so products are still counted and the number
            that slip through is reported. A constraint that is only asserted is
            not a constraint.
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
    anchor_smarts: tuple[str, ...] = ()
    max_anchor_copies: int = 1

    def describe(self) -> str:
        anchor = (
            f"anchor_smarts={len(self.anchor_smarts)} pattern(s), "
            f"max_anchor_copies={self.max_anchor_copies}"
            if self.anchor_smarts
            else "anchor_smarts=none (products may carry any number of warheads)"
        )
        return (
            f"max_generated={self.max_generated}; "
            f"heavy_atoms={self.min_heavy_atoms}-{self.max_heavy_atoms}; "
            f"max_sascore={self.max_sascore}; "
            f"reject_pains={self.reject_pains}; reject_brenk={self.reject_brenk}; "
            f"require_novelty={self.require_novelty}; "
            f"diversity_threshold={self.diversity_threshold}; "
            f"{anchor}"
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
    #: How the fragment pool split for the anchor constraint, and how often the
    #: constraint was not honoured by construction. Both are reported because a
    #: structural guarantee that is never checked is only a claim.
    n_anchor_fragments: int = 0
    n_plain_fragments: int = 0
    n_anchor_escapes: int = 0
    n_anchor_lost: int = 0
    # PolicyAudit is defined below; `from __future__ import annotations` makes
    # the forward reference resolve without quoting it.
    policy_audit: PolicyAudit | None = None
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
        if self.n_anchor_fragments or self.n_plain_fragments:
            lines.append(
                f"Anchor split: {self.n_anchor_fragments} fragment(s) carry the "
                f"motif, {self.n_plain_fragments} do not."
            )
            if self.n_anchor_escapes:
                lines.append(
                    f"  {self.n_anchor_escapes} product(s) carried the motif more "
                    "than allowed despite the seeded build -- joining two "
                    "motif-free fragments can create it across the new bond. "
                    "Rejected, and counted here because a structural guarantee "
                    "that is never verified is only a claim."
                )
            if self.n_anchor_lost:
                lines.append(
                    f"  {self.n_anchor_lost} product(s) grew from an anchor "
                    "fragment and came out without the motif: BRICS cut through "
                    "it, or the reaction consumed it. Rejected."
                )
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


def _compile_anchors(anchor_smarts: Sequence[str]) -> list[object]:
    """Compile anchor SMARTS, skipping any that will not parse."""
    require_rdkit()
    compiled = []
    for smarts in anchor_smarts:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None:
            compiled.append(pattern)
    return compiled


def count_anchors(smiles: str, anchor_patterns: Sequence[object]) -> int:
    """How many times the anchoring motif appears in a structure.

    Counted, not detected. Two amidines is the signature of two inhibitors joined
    end to end, and a presence check reads that as "has the motif" and approves it.
    """
    require_rdkit()
    mol = Chem.MolFromSmiles((smiles or "").strip())
    if mol is None:
        return 0
    return sum(
        len(mol.GetSubstructMatches(pattern, uniquify=True))  # type: ignore[arg-type]
        for pattern in anchor_patterns
    )


def partition_fragments(
    fragments: Sequence[str], anchor_patterns: Sequence[object]
) -> tuple[list[str], list[str]]:
    """Split fragments into those carrying the anchoring motif and the rest.

    Returns ``(anchor_bearing, motif_free)``, each sorted so the split is
    reproducible. A fragment carries BRICS attachment-point dummies, which do not
    interfere with the substructure match.
    """
    require_rdkit()
    anchored: list[str] = []
    plain: list[str] = []
    for fragment in sorted(fragments):
        if count_anchors(fragment, anchor_patterns) > 0:
            anchored.append(fragment)
        else:
            plain.append(fragment)
    return anchored, plain


def _build_products(
    fragments: Sequence[str],
    *,
    max_generated: int,
    seed: int,
    seeds: Sequence[str] | None = None,
) -> list[str]:
    """Draw products from the BRICS builder, capped and shuffled.

    The builder enumerates in a fixed order that systematically favours
    combinations of whichever fragments come first, so an uncapped prefix of its
    output is a biased sample of the space. Shuffling the fragment list before
    building spreads the sample across the fragment set.

    The randomness here selects which structures to *propose*. It never touches a
    score or a measurement, which is why this module appears on the randomness
    allowlist in ``tests/test_no_fabrication.py``.

    **Seeding the global generator is not optional.** ``BRICS.BRICSBuild`` with
    ``scrambleReagents=True`` shuffles its reagents and reaction sets using the
    ``random`` module's *global* state, not any generator passed to it. A private
    ``random.Random(seed)`` therefore controls the fragment order and nothing
    else, and the builder's enumeration order comes from OS entropy -- so with the
    cap applied, two runs draw different products from the same fragments.

    That was invisible and expensive. Two runs of the full pipeline on an
    identical curated dataset (same curation signature, 5091 compounds, 572
    actives) produced different candidate lists; worse, the six shards of a single
    run each generated their own list while the combine step pooled them as one
    experiment. The global state is seeded here and restored afterwards, so the
    determinism is local to this function and nothing else in the process has its
    randomness changed underneath it.

    Returns a list rather than a generator for the same reason: the seeded window
    has to cover the whole enumeration, and a lazy generator would leave the
    global state seeded while the caller does unrelated work between products.

    Args:
        fragments: The reagent pool the builder draws from.
        max_generated: Ceiling on products returned.
        seed: Controls the fragment shuffle and the builder's own scrambling.
        seeds: Starting points. BRICS grows each product outward from a seed using
            the reagent pool, so passing the anchor-bearing fragments here and only
            motif-free fragments as ``fragments`` constrains every product to
            exactly one anchor **by construction**, rather than generating
            double-warhead molecules and discarding them afterwards. The
            distinction is not cosmetic: the cap is applied to what the builder
            emits, so a pool that mostly yields artefacts spends the whole budget
            on them -- on BACE1 it produced 15 artefacts and 0 candidates.
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

    seed_mols: list[object] | None = None
    if seeds:
        shuffled_seeds = list(seeds)
        rng.shuffle(shuffled_seeds)
        seed_mols = [
            mol
            for mol in (Chem.MolFromSmiles(s) for s in shuffled_seeds)
            if mol is not None
        ]
        if not seed_mols:
            return []
        # With seeds supplied the reagent pool may legitimately hold a single
        # fragment: one seed plus one reagent is a product.
        if not mols:
            return []
    elif len(mols) < 2:
        return []

    products: list[str] = []
    state = random.getstate()
    random.seed(seed)
    try:
        builder = (
            BRICS.BRICSBuild(
                mols, seeds=seed_mols, scrambleReagents=True, maxDepth=3
            )
            if seed_mols is not None
            else BRICS.BRICSBuild(mols, scrambleReagents=True, maxDepth=3)
        )
        for product in builder:
            if len(products) >= max_generated:
                break
            try:
                product.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(product)
                products.append(Chem.MolToSmiles(product))
            except Exception:
                # Recombination can produce chemically invalid structures; they
                # are skipped silently here and counted by the caller.
                continue
    except Exception:
        pass
    finally:
        random.setstate(state)
    return products


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

    # The anchor constraint, applied to the fragment pool rather than to the
    # output. Anchor-bearing fragments become seeds and only motif-free fragments
    # are offered as reagents, so a product grows outward from exactly one
    # warhead. Generating double-warhead molecules and filtering them afterwards
    # is not equivalent: the cap applies to what the builder emits, so on BACE1 the
    # whole budget went to artefacts and the shortlist came out empty.
    anchor_patterns = _compile_anchors(policy.anchor_smarts)
    reagents = sorted(fragments)
    anchor_seeds: list[str] | None = None

    if anchor_patterns:
        anchored, plain = partition_fragments(fragments, anchor_patterns)
        report.n_anchor_fragments = len(anchored)
        report.n_plain_fragments = len(plain)
        if not anchored:
            report.notes.append(
                f"No fragment carries the anchoring motif, from "
                f"{len(fragments)} fragments. Either the motif is wrong for this "
                "chemistry or BRICS cut through it -- a motif spanning a BRICS "
                "bond ends up split across two fragments and present in neither. "
                "Generation proceeds unconstrained and the products are NOT "
                "guaranteed to carry a warhead."
            )
        elif not plain:
            report.notes.append(
                f"Every one of the {len(anchored)} fragments carries the "
                "anchoring motif, so there is nothing motif-free to build with "
                "and the constraint cannot be applied structurally. Generation "
                "proceeds unconstrained; products carrying the motif more than "
                f"{policy.max_anchor_copies} time(s) are still rejected, but by a "
                "filter rather than by construction."
            )
        else:
            anchor_seeds = anchored
            reagents = plain
            report.notes.append(
                f"Anchor constraint active: {len(anchored)} motif-bearing "
                f"fragment(s) used as build seeds, {len(plain)} motif-free "
                "fragment(s) as reagents. Every product therefore grows from "
                "exactly one warhead by construction."
            )

    seen: set[str] = set()
    survivors: list[Candidate] = []

    for product_smiles in _build_products(
        reagents,
        max_generated=policy.max_generated,
        seed=seed,
        seeds=anchor_seeds,
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

        if anchor_patterns:
            copies = count_anchors(result.smiles, anchor_patterns)
            # Checked even when the seeding should have made it impossible. Joining
            # two motif-free fragments can create the motif across the new bond, and
            # a constraint nobody verifies is a claim rather than a constraint.
            if copies > policy.max_anchor_copies:
                report.n_anchor_escapes += 1
                report.record(
                    f"carries the anchoring motif {policy.max_anchor_copies + 1}+ "
                    "times (two inhibitors joined)"
                )
                continue
            if copies == 0 and anchor_seeds is not None:
                # Under the seeded build every product starts from an anchor, so a
                # product without one means BRICS cut the motif or the reaction
                # consumed it. Counted rather than assumed away.
                report.n_anchor_lost += 1
                report.record("lost the anchoring motif during recombination")
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
