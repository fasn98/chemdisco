"""Conserved pharmacophoric features: what the known actives all share.

The check that catches a specific and quiet failure mode of fragment
recombination. BRICS cuts molecules at synthetically reasonable bonds and
reassembles them, which means it can detach the group that does the binding and
produce a well-shaped molecule with no way to engage the target.

A real example from the BACE1 run. This candidate scored -9.19 kcal/mol, better
than the median known inhibitor::

    N#Cc1cc(-c2cc(F)c(F)c(-c3ccc(F)cn3)c2)cc(Cl)c1F

It is a polyfluorinated biaryl nitrile with no basic nitrogen anywhere. BACE1 is
an aspartyl protease: essentially every known inhibitor carries an amidine,
guanidine or basic amine that engages the catalytic aspartate dyad. The molecule
fills the pocket and cannot do the chemistry, and a docking score has no way to
notice -- scoring functions reward shape complementarity and buried surface, not
the presence of a warhead.

The approach here is deliberately **data-driven rather than target-specific**.
Nothing in this module knows about BACE1 or aspartyl proteases. It measures which
of a set of generic functional-group patterns appear in the known actives, treats
any pattern present in most of them as conserved for this target, and flags
candidates carrying none of them. Point it at a kinase set and it will find the
hinge-binding motifs instead, for the same reason and with no changes.

The limitation is real and worth stating: a conserved pattern is a correlation
across the actives, not a demonstrated binding requirement. A genuinely novel
chemotype that engages the target another way would be flagged here. So this
produces a warning, not a rejection, and the warning says what it is based on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..chem.standardize import RDKIT_AVAILABLE, require_rdkit

if RDKIT_AVAILABLE:  # pragma: no branch
    from rdkit import Chem

#: Generic functional-group patterns, chosen to span the features that commonly
#: anchor a ligand to its target. Deliberately not a target-specific list: which
#: of these matter is measured from the actives, not assumed.
FEATURE_PATTERNS: dict[str, str] = {
    # Basic and cationic groups, which engage acidic residues.
    "amidine": "[NX3][CX3]=[NX2]",
    "guanidine": "[NX3][CX3](=[NX2])[NX3]",
    "primary_aliphatic_amine": "[NX3;H2;!$(NC=O);!$(Nc)]",
    "secondary_aliphatic_amine": "[NX3;H1;!$(NC=O);!$(Nc)]([CX4])[CX4]",
    "tertiary_aliphatic_amine": "[NX3;H0;!$(NC=O);!$(Nc)]([CX4])([CX4])[CX4]",
    "basic_nitrogen_any": "[NX3;!$(NC=O);!$(N[a]);!$(N=*);!$([N+])]",
    "aminopyridine": "[nX2]1ccccc1",
    # Acidic groups, which engage basic residues.
    "carboxylic_acid": "[CX3](=O)[OX2H1,OX1-]",
    "tetrazole": "c1nnn[nH]1",
    "sulfonamide": "[SX4](=O)(=O)[NX3]",
    "phosphate": "[PX4](=O)([OX2H1,OX1-])[OX2H1,OX1-]",
    # Hydrogen-bonding and transition-state motifs.
    "secondary_alcohol": "[CX4;H1][OX2H1]",
    "hydroxyethylene": "[CX4][CX4;H1]([OX2H1])[CX4]",
    "amide": "[CX3](=O)[NX3]",
    "urea": "[NX3][CX3](=O)[NX3]",
    "sulfone": "[SX4](=O)(=O)[#6]",
    # Aromatic and halogen contacts.
    "aromatic_ring": "a1aaaaa1",
    "fused_aromatic": "a1aaa2aaaaa2a1",
    "halogen_on_aromatic": "[F,Cl,Br,I][c]",
    "nitrile": "[NX1]#[CX2]",
}

#: Fraction of actives a pattern must appear in before it counts as conserved.
#: Four in five is strict enough that a merely common group does not qualify, and
#: loose enough to tolerate the chemotype diversity of a real active set.
DEFAULT_CONSERVATION_THRESHOLD = 0.8

#: How much more common a feature must be among the actives than in the
#: background before it counts as discriminating.
#:
#: Conservation alone is not enough, and the first version of this module learned
#: that the hard way. Its BACE1 actives all carried an amidine -- and all carried
#: an aromatic ring and an aromatic halogen too. A warheadless polyfluorinated
#: biaryl retained the latter two and sailed through a check meant to catch
#: exactly it. An aromatic ring is in almost every drug-like molecule, so its
#: presence among the actives says nothing about binding.
#:
#: Enrichment separates the two. A feature in 100% of actives and 95% of ordinary
#: compounds discriminates nothing; one in 100% of actives and 10% of ordinary
#: compounds discriminates a great deal. 1.5 is deliberately permissive: the
#: point is removing the ubiquitous, not demanding a perfect marker.
DEFAULT_ENRICHMENT_RATIO = 1.5


@dataclass(frozen=True, slots=True)
class FeatureProfile:
    """Which features the known actives share, and how consistently.

    Attributes:
        prevalence: Fraction of actives carrying each pattern.
        conserved: Patterns that are both prevalent among the actives and, when a
            background was supplied, enriched relative to it.
        n_actives: How many actives were profiled.
        threshold: The prevalence fraction used.
        background_prevalence: Fraction of background compounds carrying each
            pattern. Empty when no background was supplied.
        n_background: How many background compounds were profiled.
        ubiquitous: Patterns prevalent among the actives but no more so than in
            the background. Reported rather than discarded, because "the actives
            all have an aromatic ring and so does everything else" is exactly the
            fact a reader needs to see.
    """

    prevalence: dict[str, float]
    conserved: tuple[str, ...]
    n_actives: int
    threshold: float
    background_prevalence: dict[str, float] = field(default_factory=dict)
    n_background: int = 0
    ubiquitous: tuple[str, ...] = ()

    @property
    def has_background(self) -> bool:
        return self.n_background >= 10

    @property
    def most_discriminating(self) -> tuple[str, ...]:
        """The conserved features that separate actives from background best.

        Conservation plus modest enrichment is a low bar, and a candidate can
        clear it on something peripheral. One BACE1 survivor retained only
        ``halogen_on_aromatic`` -- enriched enough to count, but unlikely to be
        what engages the catalytic dyad -- while carrying no basic nitrogen at
        all.

        This takes the features whose enrichment is at least half the best
        observed, which on the BACE1 set keeps the amidine and basic-amine motifs
        (absent from the background entirely) and drops the aromatic halogen.
        Derived from the measurement, not from a hand-written list of what
        matters.
        """
        if not self.has_background or not self.conserved:
            return self.conserved
        ratios: dict[str, float] = {}
        for name in self.conserved:
            ratio = self.enrichment(name)
            if ratio is None:
                continue
            # Infinite enrichment -- present in actives, absent from background --
            # is the strongest signal there is; cap it so it can be compared.
            ratios[name] = 1000.0 if ratio == float("inf") else ratio
        if not ratios:
            return self.conserved
        best = max(ratios.values())
        return tuple(
            sorted(name for name, ratio in ratios.items() if ratio >= best / 2.0)
        )

    @property
    def is_informative(self) -> bool:
        """Whether this profile can support a judgement about a candidate.

        A profile with no conserved feature cannot flag anything, and one built
        from a handful of actives describes those compounds rather than the
        target. Both are reported rather than quietly producing an empty check.
        """
        return bool(self.conserved) and self.n_actives >= 10

    def enrichment(self, name: str) -> float | None:
        """How much more common ``name`` is among the actives than in background."""
        if not self.has_background:
            return None
        background = self.background_prevalence.get(name, 0.0)
        if background <= 0.0:
            # Present in the actives and absent from the background is maximally
            # discriminating; a ratio would be infinite, so it is capped.
            return float("inf") if self.prevalence.get(name, 0.0) > 0 else None
        return self.prevalence.get(name, 0.0) / background

    def describe(self) -> str:
        lines = [
            f"Feature profile from {self.n_actives} known actives "
            f"(conserved = present in at least {self.threshold:.0%}):"
        ]
        if not self.prevalence:
            return lines[0] + " none computed"

        for name, fraction in sorted(
            self.prevalence.items(), key=lambda kv: -kv[1]
        )[:12]:
            detail = f"  {name}: {fraction:.0%} of actives"
            if self.has_background:
                background = self.background_prevalence.get(name, 0.0)
                detail += f", {background:.0%} of background"
                ratio = self.enrichment(name)
                if ratio is not None and ratio != float("inf"):
                    detail += f" ({ratio:.1f}x)"
                elif ratio == float("inf"):
                    detail += " (absent from background)"
            if name in self.conserved:
                detail += "  <- conserved"
            elif name in self.ubiquitous:
                detail += "  <- prevalent but not discriminating"
            lines.append(detail)

        if not self.has_background:
            lines.append(
                "  CAUTION: no background set was supplied, so prevalence alone "
                "decided what counts as conserved. A feature in every active may "
                "also be in every other molecule -- an aromatic ring is -- and "
                "this check cannot tell the difference without a comparison set."
            )
        if self.ubiquitous:
            lines.append(
                "  Set aside as ubiquitous: "
                + ", ".join(self.ubiquitous)
                + ". Prevalent among the actives and no rarer elsewhere, so their "
                "presence in a candidate says nothing about binding."
            )

        if not self.conserved:
            lines.append(
                "  No feature is conserved across these actives. They are too "
                "chemically diverse for this check, which is informative in "
                "itself: it means no single motif characterises binding here."
            )
        elif self.n_actives < 10:
            lines.append(
                f"  CAUTION: profiled from only {self.n_actives} actives, so this "
                "describes those compounds more than the target."
            )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class FeatureVerdict:
    """Whether one candidate retains what the actives share, and how cleanly.

    Attributes:
        present: Conserved features the candidate carries.
        missing: Conserved features it lacks.
        strong_present: Of those carried, the ones among the most discriminating.
        duplicated: Features appearing more than once, with their counts. Two
            copies of the anchoring motif is the signature of two inhibitors
            glued together rather than one molecule designed.
        heavy_atoms: Size, reported because a duplicated motif and a large count
            are the same finding seen twice.
    """

    smiles: str
    present: tuple[str, ...]
    missing: tuple[str, ...]
    strong_present: tuple[str, ...] = ()
    duplicated: dict[str, int] = field(default_factory=dict)
    heavy_atoms: int = 0
    error: str | None = None

    @property
    def retains_any(self) -> bool:
        return bool(self.present)

    @property
    def retains_strong(self) -> bool:
        """Whether it keeps a feature that actually separates actives."""
        return bool(self.strong_present)

    @property
    def looks_like_two_molecules(self) -> bool:
        """Whether this is a recombination artefact rather than a candidate."""
        return bool(self.duplicated)

    def describe(self) -> str:
        if self.error:
            return f"feature check failed: {self.error}"
        if self.looks_like_two_molecules:
            duplicated = ", ".join(
                f"{name} x{count}" for name, count in sorted(self.duplicated.items())
            )
            return (
                f"carries {duplicated} across {self.heavy_atoms} heavy atoms. "
                "Fragment recombination glues whole inhibitors together, and a "
                "molecule with two anchoring motifs is two drugs end to end "
                "rather than one designed candidate -- it will fail every "
                "developability criterion whatever it scores."
            )
        if self.retains_strong:
            return "retains " + ", ".join(self.strong_present)
        if self.retains_any:
            return (
                "retains only " + ", ".join(self.present) + ", none of which is "
                "among the features that most separate actives from background. "
                "Clearing the bar on a peripheral group is weak evidence."
            )
        return (
            "retains NONE of the conserved features ("
            + ", ".join(self.missing)
            + "). Fragment recombination can detach the group that does the "
            "binding and leave a well-shaped molecule that cannot engage the "
            "target -- and a docking score cannot notice, because it rewards "
            "shape rather than chemistry."
        )


def _compiled_patterns() -> dict[str, object]:
    require_rdkit()
    compiled: dict[str, object] = {}
    for name, smarts in FEATURE_PATTERNS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None:
            compiled[name] = pattern
    return compiled


def features_of(smiles: str) -> set[str]:
    """Which patterns a structure matches."""
    return set(feature_counts(smiles))


def feature_counts(smiles: str) -> dict[str, int]:
    """How many times each pattern matches, not merely whether it does.

    The count matters, and the BACE1 run showed why. Fragment recombination
    readily glues two complete inhibitors together: candidates appeared carrying
    an aminohydantoin *and* an aminoimidazole, or an aminoimidazole *and* an
    aminothiazine, at 58 heavy atoms and a molecular weight near 800. Each is two
    drugs stapled end to end rather than a designed molecule, and a presence
    check sees only "has the motif" and waves it through.
    """
    require_rdkit()
    mol = Chem.MolFromSmiles((smiles or "").strip())
    if mol is None:
        return {}
    counts: dict[str, int] = {}
    for name, pattern in _compiled_patterns().items():
        matches = mol.GetSubstructMatches(pattern, uniquify=True)  # type: ignore[arg-type]
        if matches:
            counts[name] = len(matches)
    return counts


def _prevalence(smiles_list: Sequence[str]) -> tuple[dict[str, float], int]:
    counts: dict[str, int] = dict.fromkeys(FEATURE_PATTERNS, 0)
    parsed = 0
    for smiles in smiles_list:
        if not smiles or not smiles.strip():
            continue
        if Chem.MolFromSmiles(smiles.strip()) is None:
            continue
        parsed += 1
        for name in features_of(smiles):
            counts[name] += 1
    if parsed == 0:
        return {}, 0
    return {name: count / parsed for name, count in counts.items()}, parsed


def profile_actives(
    smiles_list: Sequence[str],
    *,
    background_smiles: Sequence[str] | None = None,
    threshold: float = DEFAULT_CONSERVATION_THRESHOLD,
    enrichment_ratio: float = DEFAULT_ENRICHMENT_RATIO,
) -> FeatureProfile:
    """Measure which features distinguish the known actives.

    Nothing target-specific is assumed. Applied to a BACE1 set this finds the
    amidine and basic-nitrogen motifs that engage the catalytic dyad; applied to
    a kinase set it would find hinge-binding patterns instead.

    Args:
        smiles_list: Known actives.
        background_smiles: Compounds that are not known actives -- the weakly
            active end of the same curated set is ideal, since those are real
            measured compounds against the same target. Strongly recommended:
            without it, prevalence alone decides, and a feature present in every
            active may be present in everything else too.
        threshold: Minimum prevalence among the actives.
        enrichment_ratio: Minimum times more common among the actives than in
            the background.
    """
    require_rdkit()
    prevalence, parsed = _prevalence(smiles_list)
    if parsed == 0:
        return FeatureProfile({}, (), 0, threshold)

    background_prevalence: dict[str, float] = {}
    n_background = 0
    if background_smiles:
        background_prevalence, n_background = _prevalence(background_smiles)

    prevalent = [
        name for name, fraction in prevalence.items() if fraction >= threshold
    ]

    conserved: list[str] = []
    ubiquitous: list[str] = []
    if n_background >= 10:
        for name in prevalent:
            background = background_prevalence.get(name, 0.0)
            if background <= 0.0 or prevalence[name] / background >= enrichment_ratio:
                conserved.append(name)
            else:
                ubiquitous.append(name)
    else:
        conserved = list(prevalent)

    return FeatureProfile(
        prevalence=prevalence,
        conserved=tuple(sorted(conserved)),
        n_actives=parsed,
        threshold=threshold,
        background_prevalence=background_prevalence,
        n_background=n_background,
        ubiquitous=tuple(sorted(ubiquitous)),
    )


def check_candidate(smiles: str, profile: FeatureProfile) -> FeatureVerdict:
    """Whether ``smiles`` retains any feature conserved among the actives.

    Returns a warning rather than a rejection. A conserved pattern is a
    correlation across the known actives, not a demonstrated requirement, so a
    genuinely novel chemotype binding another way would be flagged here too. The
    verdict says what it is based on so a chemist can overrule it.
    """
    require_rdkit()
    if not profile.is_informative:
        return FeatureVerdict(
            smiles=smiles,
            present=(),
            missing=(),
            error="the active set yielded no usable feature profile",
        )

    mol = Chem.MolFromSmiles((smiles or "").strip())
    if mol is None:
        return FeatureVerdict(smiles, (), (), error="structure failed to parse")

    counts = feature_counts(smiles)
    present = tuple(sorted(name for name in profile.conserved if name in counts))
    missing = tuple(sorted(name for name in profile.conserved if name not in counts))
    strong = profile.most_discriminating
    strong_present = tuple(sorted(name for name in strong if name in counts))
    # Only a *discriminating* motif appearing twice indicates two molecules
    # joined. Two aromatic rings is an ordinary biaryl.
    duplicated = {
        name: counts[name]
        for name in strong
        if counts.get(name, 0) > 1
    }
    return FeatureVerdict(
        smiles=smiles,
        present=present,
        missing=missing,
        strong_present=strong_present,
        duplicated=duplicated,
        heavy_atoms=mol.GetNumHeavyAtoms(),
    )


@dataclass(frozen=True, slots=True)
class FeatureScreenResult:
    """The outcome of checking a candidate set against an active profile."""

    profile: FeatureProfile
    verdicts: tuple[FeatureVerdict, ...] = field(default_factory=tuple)

    @property
    def n_retaining(self) -> int:
        return sum(1 for verdict in self.verdicts if verdict.retains_any)

    @property
    def n_flagged(self) -> int:
        return sum(
            1
            for verdict in self.verdicts
            if not verdict.retains_any and verdict.error is None
        )

    @property
    def n_weak_only(self) -> int:
        return sum(
            1
            for verdict in self.verdicts
            if verdict.retains_any and not verdict.retains_strong
        )

    @property
    def n_duplicated(self) -> int:
        return sum(1 for verdict in self.verdicts if verdict.looks_like_two_molecules)

    def describe(self) -> str:
        lines = [self.profile.describe(), ""]
        if not self.profile.is_informative:
            lines.append(
                "No conserved-feature check was applied: the active set did not "
                "yield a usable profile."
            )
            return "\n".join(lines)

        lines.append(
            f"{self.n_retaining} of {len(self.verdicts)} candidates retain at "
            f"least one conserved feature; {self.n_flagged} retain none."
        )
        if self.n_flagged:
            lines.append(
                "  The flagged candidates may fill the pocket while lacking the "
                "chemistry that binds it. A docking score cannot distinguish the "
                "two, which is why this check exists alongside it."
            )
        if self.profile.most_discriminating:
            lines.append(
                "  Most discriminating features: "
                + ", ".join(self.profile.most_discriminating)
            )
        if self.n_weak_only:
            lines.append(
                f"  {self.n_weak_only} candidate(s) clear the bar only on a "
                "peripheral feature, which is weak evidence."
            )
        if self.n_duplicated:
            lines.append(
                f"  {self.n_duplicated} candidate(s) carry an anchoring motif more "
                "than once: fragment recombination joined whole inhibitors rather "
                "than designing one molecule."
            )
        return "\n".join(lines)


def screen_candidates(
    candidate_smiles: Sequence[str],
    active_smiles: Sequence[str],
    *,
    background_smiles: Sequence[str] | None = None,
    threshold: float = DEFAULT_CONSERVATION_THRESHOLD,
    enrichment_ratio: float = DEFAULT_ENRICHMENT_RATIO,
) -> FeatureScreenResult:
    """Profile the actives against a background, then check every candidate."""
    profile = profile_actives(
        active_smiles,
        background_smiles=background_smiles,
        threshold=threshold,
        enrichment_ratio=enrichment_ratio,
    )
    verdicts = tuple(check_candidate(smiles, profile) for smiles in candidate_smiles)
    return FeatureScreenResult(profile=profile, verdicts=verdicts)
