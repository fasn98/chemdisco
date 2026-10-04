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
    """Whether one candidate retains what the actives share."""

    smiles: str
    present: tuple[str, ...]
    missing: tuple[str, ...]
    error: str | None = None

    @property
    def retains_any(self) -> bool:
        return bool(self.present)

    def describe(self) -> str:
        if self.error:
            return f"feature check failed: {self.error}"
        if self.retains_any:
            return "retains " + ", ".join(self.present)
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
    require_rdkit()
    mol = Chem.MolFromSmiles((smiles or "").strip())
    if mol is None:
        return set()
    return {
        name
        for name, pattern in _compiled_patterns().items()
        if mol.HasSubstructMatch(pattern)  # type: ignore[arg-type]
    }


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

    found = features_of(smiles)
    present = tuple(sorted(name for name in profile.conserved if name in found))
    missing = tuple(sorted(name for name in profile.conserved if name not in found))
    return FeatureVerdict(smiles=smiles, present=present, missing=missing)


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
