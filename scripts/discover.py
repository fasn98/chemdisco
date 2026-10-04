#!/usr/bin/env python3
"""The full pipeline, end to end, on one target.

Everything the earlier validation runs established, assembled into one run. Each
stage is here because it was measured, and the measurements decide how its output
may be used:

**Curation** keeps only defensible measurements and reports what it discarded.

**QSAR** reached R-squared 0.614 [0.580, 0.646] on held-out scaffolds for BACE1 --
a working model, within its applicability domain.

**Generation** by BRICS recombination, with the filter policy audited against the
known actives first. The default policy rejected 45% of published BACE1 inhibitors,
almost all on Brenk alerts: Brenk triages HTS decks for lead-likeness, and BACE1
inhibitors are large peptidomimetics carrying the amidine and guanidine motifs it
flags. The policy here is adjusted on that evidence, not to produce output.

**Docking** separates actives from property-matched decoys at AUC 0.731
[0.617, 0.831], so it carries real information about the site. But BEDROC 0.36
and an empty top 1% say the early ranking is poor, and redocking showed the pose
ranking is not determined by the score. So docking filters against the known
actives' distribution and does not rank.

What this run can produce, stated plainly before it runs: novel structures that
are synthetically plausible, free of the alerts that matter for this chemistry,
and that occupy the binding site about as well as known inhibitors do. What it
cannot produce is a potency prediction for them -- generated candidates sit
outside the QSAR model's applicability domain almost by construction, because the
whole point of generating them is that they are new.

A shortlist for a chemist to look at is the honest output. A ranked table of
predicted IC50s would not be.

Usage:
    python scripts/discover.py --shard 0 --n-shards 6
    python scripts/discover.py --combine runs/
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.request

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from chemdisco.curate import CurationPolicy, curate  # noqa: E402
from chemdisco.data.chembl import ChEMBLClient, ChEMBLError, ResponseCache  # noqa: E402
from chemdisco.dock import (  # noqa: E402
    box_from_ligand,
    interleave_by_label,
    parse_pdb,
    prepare_receptor_pdbqt,
    screen,
    shard_by_label,
    strip_to_receptor,
    toolchain_report,
    vina_available,
    write_pdb,
)
from chemdisco.generate import (  # noqa: E402
    GenerationPolicy,
    generate_candidates,
    screen_candidates,
)

#: Generation policy for BACE1, adjusted on the audit rather than on taste.
#:
#: Brenk is off because the audit measured it rejecting 45% of published BACE1
#: inhibitors. PAINS stays on: it flags assay-interference substructures, which
#: are a problem for any target, and the audit showed it rejecting almost none of
#: the actives. The size ceiling is raised because BACE1 inhibitors are genuinely
#: large -- the default 50 heavy atoms sits below many known drugs for this target.
BACE1_POLICY = GenerationPolicy(
    max_generated=4000,
    min_heavy_atoms=16,
    max_heavy_atoms=60,
    max_sascore=6.5,
    reject_pains=True,
    reject_brenk=False,
    require_novelty=True,
    # Tightened from the 0.85 default. At 0.85 the first shortlist held five
    # members of one congeneric series -- the same aminothiazine core with four
    # different N-substituents. That is one candidate with four variations, not
    # five candidates, and it crowds out the diversity the list exists to supply.
    diversity_threshold=0.7,
)


def heading(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def fetch_pdb(pdb_id: str, cache_dir: pathlib.Path) -> str | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{pdb_id.upper()}.pdb"
    if cached.exists():
        return cached.read_text()
    for attempt in range(3):
        try:
            url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
            with urllib.request.urlopen(url, timeout=120) as response:
                text = response.read().decode()
            cached.write_text(text)
            return text
        except Exception as error:
            print(f"  attempt {attempt + 1} failed: {error}")
            if attempt < 2:
                time.sleep(2**attempt)
    return None


def build_inputs(args) -> dict | None:
    """Everything up to docking: curate, generate, and pick the reference set.

    Deterministic, so every shard derives the identical ligand list from the same
    cached ChEMBL data and the same seed. That is what makes sharding valid: each
    worker must be docking a slice of one list, not its own list.
    """
    from chemdisco.chem.standardize import inchikey_of

    heading(f"1. Curating {args.name}")
    client = ChEMBLClient(cache=ResponseCache(pathlib.Path(args.cache) / "chembl"))
    try:
        records, _ = client.fetch_records(
            args.accession, activity_types=("IC50",), max_records=args.max_records
        )
    except ChEMBLError as error:
        print(f"Retrieval failed: {error}")
        return None

    report = curate(
        records,
        policy=CurationPolicy(),
        compound_key=lambda record: inchikey_of(record.smiles)
        or f"unparseable:{record.compound_id}",
    )
    print(report.describe())

    points = sorted(report.kept, key=lambda p: -p.pactivity.require())
    if len(points) < 100:
        print("Too few curated compounds.")
        return None

    actives = [p for p in points if p.pactivity.require() >= args.active_threshold]
    print(
        f"\n  {len(actives)} actives at pIC50 >= {args.active_threshold}; "
        f"using the top {min(args.n_reference, len(actives))} as the docking "
        "reference and the top "
        f"{min(args.n_seed, len(actives))} as the fragment source"
    )
    if len(actives) < 20:
        print("Too few potent compounds to work from.")
        return None

    reference = actives[: args.n_reference]
    seeds = actives[: args.n_seed]

    # The feature-profile background must be chemically DISTINCT from the
    # actives, not merely less potent. Two sources, and the history matters:
    #
    #   `weak`      -- the weakly active end of this target's own curated data.
    #                  Tried first and measurably wrong: every feature came out
    #                  at ~1.8x enrichment, so an aryl fluoride counted as an
    #                  anchoring motif and candidates with no basic nitrogen
    #                  passed. A weak BACE1 binder is still a BACE1-series
    #                  compound carrying the same amidine. Kept only so the
    #                  failure stays reproducible.
    #   `unrelated` -- ligands of kinases, GPCRs, nuclear receptors and
    #                  unrelated enzymes. Presumed non-binders rather than
    #                  measured ones, which is the trade this check needs:
    #                  shared chemistry erases the signal, so a harder control
    #                  is the wrong control here.
    background: list[str] = []
    background_summary: dict | None = None

    if args.background_source == "unrelated":
        from chemdisco.data.background import (
            fetch_unrelated_background,
        )
        from chemdisco.data.background import (
            summarise as summarise_background,
        )

        print("\n  fetching a feature-profile background from unrelated targets")
        collected = fetch_unrelated_background(
            client,
            exclude_molecule_ids=[p.compound_id for p in points],
            per_target=max(args.background_size // 6, 50),
            max_total=args.background_size,
        )
        print("  " + collected.describe().replace("\n", "\n  "))
        background_summary = summarise_background(collected)
        if collected.is_usable:
            background = list(collected.smiles)
        else:
            print(
                "  Not used. An unusable background is worse than none: it "
                "produces enrichment ratios that look like measurements."
            )
    elif args.background_source == "weak":
        weak = [
            p for p in points if p.pactivity.require() <= args.background_threshold
        ]
        if weak:
            from chemdisco.chem.similarity import max_similarity_to_reference

            similarities, _ = max_similarity_to_reference(
                [p.smiles for p in weak], [p.smiles for p in reference]
            )
            background = [
                p.smiles
                for p, similarity in zip(weak, similarities, strict=True)
                if 0.0 <= similarity < args.background_max_similarity
            ][:400]
            print(
                f"\n  feature-profile background: {len(background)} weak binders "
                f"below {args.background_max_similarity} Tanimoto to every active, "
                f"from {len(weak)} weak compounds. This source is known not to "
                "discriminate -- see the module docstring -- and is kept for "
                "comparison only."
            )

    heading("2. Generating candidates")
    print(f"  policy: {BACE1_POLICY.describe()}")
    generation = generate_candidates(
        [p.smiles for p in seeds],
        policy=BACE1_POLICY,
        reference_smiles=[p.smiles for p in points],
        seed=args.seed,
    )
    print(generation.describe())

    if not generation.candidates:
        print(
            "\nNo candidates survived. The audit above says whether that is the "
            "policy's fault or the fragment set's."
        )
        return None

    candidates = generation.candidates[: args.n_candidates]
    print(f"\n  carrying {len(candidates)} candidates forward to docking")

    return {
        "reference_smiles": [p.smiles for p in reference],
        "reference_pactivity": [p.pactivity.require() for p in reference],
        "candidate_smiles": [c.smiles for c in candidates],
        "candidate_sascore": [
            c.sascore.value if c.sascore.is_known else None for c in candidates
        ],
        "candidate_novelty": [
            c.novelty.max_similarity if c.novelty else None for c in candidates
        ],
        "reference_feature_smiles": [p.smiles for p in reference],
        "background_smiles": background,
        "background_summary": background_summary,
        "n_fragments": generation.n_fragments,
        "n_generated": generation.n_generated,
        "policy_audit_pass_rate": (
            generation.policy_audit.pass_rate if generation.policy_audit else None
        ),
    }


def prepare_receptor(args) -> tuple[str, object] | None:
    heading(f"3. Preparing the {args.pdb.upper()} receptor")
    pdb_text = fetch_pdb(args.pdb, pathlib.Path(args.cache) / "pdb")
    if pdb_text is None:
        return None
    structure = parse_pdb(pdb_text)
    ligand = structure.best_ligand()
    if ligand is None or structure.ligand_is_peptide_fragment(ligand) is not None:
        print("This structure cannot define a binding box reliably.")
        return None
    box = box_from_ligand(ligand)
    print(box.describe())
    receptor_pdbqt, method = prepare_receptor_pdbqt(
        write_pdb(strip_to_receptor(structure), title=f"{args.pdb.upper()} receptor")
    )
    if receptor_pdbqt is None:
        print(f"Receptor preparation failed: {method}")
        return None
    print(f"  prepared with {method}")
    return receptor_pdbqt, box


def combine(directory: pathlib.Path, output: str) -> int:
    """Pool the shards and apply the triage."""
    heading("Combining shards")
    shards = sorted(directory.glob("discover_shard_*.json"))
    if not shards:
        print(f"No shard files in {directory}")
        return 1

    reference_scores: list[float] = []
    reference_efficiencies: list[float | None] = []
    candidates: list[dict] = []
    receptors: set[str] = set()
    boxes: set[str] = set()
    seconds = 0.0
    requested = 0

    for path in shards:
        payload = json.loads(path.read_text())
        reference_scores.extend(payload["reference_scores"])
        reference_efficiencies.extend(payload.get("reference_efficiencies", []))
        candidates.extend(payload["candidates"])
        receptors.add(payload["receptor_id"])
        boxes.add(payload["box_signature"])
        seconds += payload.get("elapsed_seconds", 0.0)
        requested += payload.get("n_requested", 0)
        print(
            f"  {path.name}: {len(payload['reference_scores'])} reference, "
            f"{len(payload['candidates'])} candidates"
        )

    if len(receptors) > 1 or len(boxes) > 1:
        print(
            "\nREFUSING: shards used different receptors or boxes. Vina scores "
            "are only comparable within one setup."
        )
        return 1

    print(
        f"\nPooled: {len(reference_scores)} reference actives, "
        f"{len(candidates)} candidates, {seconds / 60:.0f} CPU-minutes"
    )

    profiles = [p for p in (json.loads(s.read_text()).get("feature_profile") for s in shards) if p]
    if profiles:
        profile = profiles[0]
        heading("The feature profile the candidates were judged against")
        print(
            f"  built from {profile['n_actives']} actives against "
            f"{profile['n_background']} background compounds"
        )
        summary = profile.get("background_summary") or {}
        if summary:
            families = ", ".join(
                f"{family} ({count})"
                for family, count in sorted(
                    summary.get("families", {}).items(), key=lambda kv: -kv[1]
                )
            )
            print(f"  background: {summary.get('source', 'unspecified')}")
            print(f"    families: {families or 'none'}")
            if summary.get("n_excluded_overlap"):
                print(
                    f"    {summary['n_excluded_overlap']} compound(s) excluded for "
                    "also appearing in this target's own dataset"
                )
            if summary.get("failures"):
                print(f"    not retrieved: {', '.join(summary['failures'])}")
        print(f"  conserved: {', '.join(profile['conserved']) or 'none'}")
        print(
            f"  most discriminating: "
            f"{', '.join(profile['most_discriminating']) or 'none'}"
        )
        if profile["ubiquitous"]:
            print(f"  set aside as ubiquitous: {', '.join(profile['ubiquitous'])}")
        print("\n  prevalence (actives vs background):")
        for name in profile["most_discriminating"]:
            actives_pct = profile["prevalence"].get(name, 0.0)
            background_pct = profile["background_prevalence"].get(name, 0.0)
            ratio = (
                "absent from background"
                if background_pct == 0
                else f"{actives_pct / background_pct:.1f}x"
            )
            print(f"    {name}: {actives_pct:.0%} vs {background_pct:.0%} ({ratio})")
        print(
            "\n  CAUTION: a feature can be enriched among potent compounds without\n"
            "  being what binds. Late-stage optimisation adds fluorine, so aromatic\n"
            "  halogens correlate with potency inside an optimised series while\n"
            "  forming no specific interaction. This profile measures correlation\n"
            "  with potency, not binding mechanism, and cannot tell the two apart."
        )

    if len(reference_scores) < 10:
        print(
            "Too few reference actives docked to define a distribution. A docking "
            "score has no meaning without one."
        )
        return 1

    heading("4. Triage against the reference distribution")
    reference = np.asarray(reference_scores)
    print(
        f"  reference actives by raw score: median {np.median(reference):.2f}, "
        f"range {reference.min():.2f} to {reference.max():.2f} kcal/mol"
    )

    reference_le = [
        value for value in reference_efficiencies if value is not None
    ]
    if reference_le:
        print(
            f"  reference actives by ligand efficiency: median "
            f"{np.median(reference_le):.3f}, range {min(reference_le):.3f} to "
            f"{max(reference_le):.3f} kcal/mol/atom"
        )

    # Triage on ligand efficiency, not raw score.
    #
    # The raw-score threshold was wrong, and this package documented why three
    # files away: Vina's score grows close to linearly with molecular size, so
    # "beats the median known active" selects for being large. The way BRICS
    # produces large molecules from inhibitor fragments is by joining warheads,
    # so the threshold selected recombination artefacts with near-perfect
    # efficiency -- a run where all 13 candidates clearing it carried the
    # anchoring motif twice, and nothing survived.
    #
    # Efficiency per heavy atom removes the size term from the comparison, which
    # is the whole reason the metric exists.
    if reference_le:
        threshold = float(np.median(reference_le))
        passing_score = [
            c
            for c in candidates
            if c.get("ligand_efficiency") is not None
            and c["ligand_efficiency"] <= threshold
        ]
        print(
            f"\n  {len(passing_score)} of {len(candidates)} candidates reach a "
            f"ligand efficiency of {threshold:.3f} kcal/mol/atom or better, the "
            "median known active."
        )
    else:
        threshold = float(np.median(reference))
        passing_score = [
            c for c in candidates if c["score"] is not None and c["score"] <= threshold
        ]
        print(
            f"\n  no reference efficiencies available, falling back to raw score: "
            f"{len(passing_score)} of {len(candidates)} reach {threshold:.2f} "
            "kcal/mol. This comparison is confounded by molecular size."
        )

    # Two corrections the first shortlist needed, both following from what this
    # package already documents about docking.
    #
    # Ligand efficiency, because Vina's score grows close to linearly with size:
    # the first shortlist's best scores (-12.58, -12.12) were simply its largest
    # molecules. Comparing raw scores across 20- to 50-atom candidates ranks by
    # weight.
    #
    # The conserved-feature check, because fragment recombination can detach the
    # group that does the binding. The first shortlist's top entry by synthetic
    # accessibility was a polyfluorinated biaryl nitrile with no basic nitrogen,
    # scoring -9.19 against an aspartyl protease whose inhibitors all need one.
    # Withholding a check means no verdict, not a failing verdict. When the
    # profile could not discriminate -- as it cannot against a target's own
    # data -- every candidate comes back with retains_strong False, and treating
    # that as a failure condemned all 17 survivors of a run on the strength of a
    # check the module had already declared unusable.
    #
    # A background is required, not optional. Without one, `most_discriminating`
    # falls back to every conserved feature, and "conserved" then means nothing
    # more than "common" -- which is how an aromatic ring, present in almost every
    # drug-like molecule, once passed as an anchoring motif.
    profile_usable = bool(
        profiles
        and profiles[0].get("has_background")
        and profiles[0].get("most_discriminating")
    )
    if profile_usable:
        with_features = [
            c for c in passing_score if c.get("retains_strong_feature") is not False
        ]
        dropped_features = len(passing_score) - len(with_features)
        if dropped_features:
            print(
                f"  {dropped_features} of those retain none of the features that "
                "most separate actives from background, and are set aside: "
                "fragment recombination can leave a well-shaped molecule with no "
                "way to engage the target, and a docking score cannot tell the "
                "difference."
            )
    else:
        with_features = list(passing_score)
        dropped_features = 0
        reason = (
            "no feature stood out against the background"
            if profiles and profiles[0].get("has_background")
            else "no usable background was available, so prevalence alone decided "
            "what counts as conserved -- which cannot separate a binding motif "
            "from an aromatic ring"
        )
        print(
            f"  The conserved-feature check was WITHHELD: {reason}, so it could "
            "not tell a binding motif from an "
            "optimisation artefact. No candidate is credited or condemned by it, "
            "and the survivors below have NOT been checked for the chemistry that "
            "binds -- which is a real gap in this shortlist, not a formality."
        )

    # A candidate carrying the anchoring motif twice is two inhibitors joined,
    # not one molecule. The BACE1 run produced several at 51-59 heavy atoms with
    # two complete warheads apiece -- they will fail every developability
    # criterion whatever they score.
    single_molecule = [c for c in with_features if not c.get("duplicated_motifs")]
    dropped_duplicates = len(with_features) - len(single_molecule)
    if dropped_duplicates:
        print(
            f"  {dropped_duplicates} carry an anchoring motif more than once and "
            "are set aside as recombination artefacts: two drugs glued end to "
            "end rather than one designed candidate."
        )

    survivors = single_molecule
    efficiencies = [
        c["ligand_efficiency"]
        for c in survivors
        if c.get("ligand_efficiency") is not None
    ]
    if efficiencies:
        print(
            f"\n  ligand efficiency across survivors: "
            f"{min(efficiencies):.3f} to {max(efficiencies):.3f} kcal/mol/atom"
        )

    print(
        "\n  This is a filter, not a ranking. Docking on this target separates "
        "actives from decoys (AUC 0.731) but does not order them reliably "
        "(BEDROC 0.36, empty top 1%), so the survivors are listed by ligand "
        "efficiency -- which at least corrects for the size bias -- and that "
        "order still carries far less information than it appears to."
    )

    survivors.sort(key=lambda c: (c.get("ligand_efficiency") or 0.0))

    heading("The shortlist")
    if not survivors:
        reasons = []
        if len(passing_score) == 0:
            reasons.append(
                "no candidate reached the reference efficiency threshold"
            )
        if dropped_features:
            reasons.append(
                f"{dropped_features} lost the chemistry that binds"
            )
        if dropped_duplicates:
            reasons.append(
                f"{dropped_duplicates} were recombination artefacts carrying the "
                "anchoring motif twice"
            )
        print(
            "Nothing survived: "
            + ("; ".join(reasons) if reasons else "every candidate was filtered")
            + "."
        )
    else:
        checked = (
            "retain a feature the known actives share, and "
            if profile_usable
            else ""
        )
        print(
            f"{len(survivors)} structures that are novel, pass the filters "
            f"calibrated for this target, {checked}occupy the site about as well "
            "as known inhibitors do per heavy atom.\n"
        )
        if not profile_usable:
            print(
                "  Note: none has been checked for the binding chemistry. The "
                "feature profile could not discriminate against this background, "
                "so that check was withheld.\n"
            )
        for index, candidate in enumerate(survivors[:15], start=1):
            print(f"{index:>3}. {candidate['smiles']}")
            details = [f"docking {candidate['score']:.2f} kcal/mol"]
            efficiency = candidate.get("ligand_efficiency")
            if efficiency is not None:
                details.append(f"LE {efficiency:.3f}")
            if candidate.get("heavy_atoms"):
                details.append(f"{candidate['heavy_atoms']} heavy atoms")
            sascore = candidate.get("sascore")
            if sascore is not None:
                details.append(f"SAscore {sascore:.2f}")
            novelty = candidate.get("novelty")
            if novelty is not None:
                details.append(f"Tanimoto {novelty:.2f}")
            print("     " + " | ".join(details))
            conserved = candidate.get("conserved_features") or []
            if conserved:
                print(f"     retains: {', '.join(conserved)}")

    heading("What these are, and what they are not")
    if not survivors:
        print(
            "There is no shortlist from this run, which is itself the finding.\n"
            "\n"
            "The attrition above says where the candidates went. If most were set\n"
            "aside as recombination artefacts, the fragment set and the threshold\n"
            "together are selecting for molecules joined end to end rather than\n"
            "for designs -- more fragments or a different generator would be the\n"
            "response, not a looser filter.\n"
            "\n"
            "An empty list is a usable answer. A list assembled by relaxing the\n"
            "checks until something appeared would not be."
        )
    else:
        print(
            "These structures have NO predicted potency. Generated candidates sit\n"
            "outside the QSAR model's applicability domain almost by construction --\n"
            "the reason to generate them is that they are new, and new is exactly\n"
            "where the model has no basis to predict. Attaching an IC50 to them would\n"
            "be inventing a number.\n"
            "\n"
            "What the evidence supports: they are novel against the curated ChEMBL\n"
            "set, synthetically plausible by SAscore, free of PAINS alerts, and they\n"
            "occupy the BACE1 site about as well as known inhibitors do in the same\n"
            "receptor and box.\n"
            "\n"
            "That is a shortlist worth a chemist's hour. It is not a result, and the\n"
            "only way to find out whether any of them bind is to make them and test\n"
            "them."
        )

    if output:
        path = pathlib.Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "n_candidates_docked": len(candidates),
                    "n_passing_score": len(passing_score),
                    "n_dropped_no_conserved_feature": dropped_features,
                    "n_survivors": len(survivors),
                    "reference_median": threshold,
                    "reference_n": len(reference_scores),
                    "shortlist": survivors[:50],
                },
                indent=2,
            )
        )
        print(f"\nWritten to {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accession", default="P56817")
    parser.add_argument("--name", default="BACE1")
    parser.add_argument("--pdb", default="4FRS")
    parser.add_argument("--active-threshold", type=float, default=8.0)
    parser.add_argument(
        "--background-source",
        choices=("unrelated", "weak", "none"),
        default="unrelated",
        help=(
            "Where the feature-profile background comes from. 'unrelated': "
            "ligands of unrelated targets, presumed non-binders. 'weak': this "
            "target's own weak binders, kept because it is a reproducible "
            "failure, not because it works. 'none': prevalence only, which "
            "cannot tell a binding motif from an aromatic ring"
        ),
    )
    parser.add_argument(
        "--background-size",
        type=int,
        default=1200,
        help="Ceiling on the unrelated-target background",
    )
    parser.add_argument(
        "--background-max-similarity",
        type=float,
        default=0.4,
        help=(
            "Maximum Tanimoto to any active for a compound to serve as "
            "feature-profile background. Same-series weak binders carry the "
            "actives' motifs and erase the signal"
        ),
    )
    parser.add_argument(
        "--background-threshold",
        type=float,
        default=6.0,
        help=(
            "Maximum pActivity for a compound to serve as feature-profile "
            "background. Weak measured binders, not presumed inactives"
        ),
    )
    parser.add_argument("--n-reference", type=int, default=30)
    parser.add_argument("--n-seed", type=int, default=50)
    parser.add_argument("--n-candidates", type=int, default=60)
    parser.add_argument("--max-records", type=int, default=8000)
    parser.add_argument("--exhaustiveness", type=int, default=4)
    parser.add_argument("--time-budget", type=float, default=2400.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", default=".cache")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--combine", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if args.combine:
        return combine(pathlib.Path(args.combine), args.output)

    started = time.monotonic()
    heading("0. Toolchain")
    print(toolchain_report())
    if not vina_available():
        print("\nVina is not installed; this cannot run.")
        return 2

    from chemdisco.chem.standardize import silence_rdkit_logs

    silence_rdkit_logs()

    inputs = build_inputs(args)
    if inputs is None:
        return 1

    prepared = prepare_receptor(args)
    if prepared is None:
        return 1
    receptor_pdbqt, box = prepared

    heading("4. Docking reference actives and candidates together")
    # Label 1 marks a reference active, 0 a candidate. Both go into the same
    # receptor and box in the same run, which is the only way their scores are
    # comparable -- and the reference distribution is what gives a candidate's
    # score any meaning at all.
    ligands = inputs["reference_smiles"] + inputs["candidate_smiles"]
    labels = [1] * len(inputs["reference_smiles"]) + [0] * len(
        inputs["candidate_smiles"]
    )
    ligands, labels = interleave_by_label(ligands, labels)
    if args.n_shards > 1:
        ligands, labels = shard_by_label(
            ligands, labels, shard=args.shard, n_shards=args.n_shards
        )
        print(
            f"  shard {args.shard + 1}/{args.n_shards}: {len(ligands)} ligands "
            f"({sum(labels)} reference, {len(labels) - sum(labels)} candidates)"
        )

    sascore_by_smiles = dict(
        zip(inputs["candidate_smiles"], inputs["candidate_sascore"], strict=True)
    )
    novelty_by_smiles = dict(
        zip(inputs["candidate_smiles"], inputs["candidate_novelty"], strict=True)
    )

    # Which features the known actives share, measured from them rather than
    # assumed. BRICS can detach the group that does the binding and leave a
    # well-shaped molecule a docking score cannot fault.
    heading("3b. Profiling the conserved features of the known actives")
    # The background is the weakly active end of the same curated set: real
    # measured compounds against this target, which is what distinguishes a
    # feature that marks binding from one that marks being a drug-like molecule.
    feature_screen = screen_candidates(
        inputs["candidate_smiles"],
        inputs["reference_feature_smiles"],
        background_smiles=inputs["background_smiles"],
    )
    print(feature_screen.describe())
    feature_by_smiles = {
        verdict.smiles: {
            "present": list(verdict.present),
            "strong_present": list(verdict.strong_present),
            "retains_any": verdict.retains_any,
            "retains_strong": verdict.retains_strong,
            "duplicated": dict(verdict.duplicated),
        }
        for verdict in feature_screen.verdicts
    }

    def progress(done: int, total: int, _result) -> None:
        if done % 10 == 0 or done == total:
            print(f"    {done}/{total}", flush=True)

    result = screen(
        ligands,
        receptor_pdbqt,
        box,
        labels=labels,
        receptor_id=args.pdb.upper(),
        progress=progress,
        exhaustiveness=args.exhaustiveness,
        n_poses=3,
        time_budget_seconds=args.time_budget,
    )
    print("\n" + result.describe())

    reference_scores: list[float] = []
    reference_efficiencies: list[float | None] = []
    candidate_rows: list[dict] = []
    for docked, label in zip(result.results, result.labels, strict=True):
        if not docked.ok or docked.best_score is None:
            continue
        if label == 1:
            reference_scores.append(docked.best_score)
            efficiency = docked.ligand_efficiency()
            reference_efficiencies.append(
                efficiency.value if efficiency.is_known else None
            )
        else:
            features = feature_by_smiles.get(docked.smiles, {})
            candidate_rows.append(
                {
                    "smiles": docked.smiles,
                    "score": docked.best_score,
                    "heavy_atoms": docked.n_heavy_atoms,
                    "ligand_efficiency": (
                        docked.ligand_efficiency().value
                        if docked.ligand_efficiency().is_known
                        else None
                    ),
                    "sascore": sascore_by_smiles.get(docked.smiles),
                    "novelty": novelty_by_smiles.get(docked.smiles),
                    "retains_conserved_feature": features.get("retains_any"),
                    "retains_strong_feature": features.get("retains_strong"),
                    "conserved_features": features.get("strong_present")
                    or features.get("present", []),
                    "duplicated_motifs": features.get("duplicated", {}),
                }
            )

    path = pathlib.Path(args.output or f"runs/discover_shard_{args.shard}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "shard": args.shard,
                "reference_scores": reference_scores,
                "reference_efficiencies": reference_efficiencies,
                "candidates": candidate_rows,
                "receptor_id": args.pdb.upper(),
                "box_signature": f"{box.center}|{box.size}",
                "n_requested": len(ligands),
                "elapsed_seconds": result.elapsed_seconds,
                "feature_profile": {
                    "conserved": list(feature_screen.profile.conserved),
                    "most_discriminating": list(
                        feature_screen.profile.most_discriminating
                    ),
                    "ubiquitous": list(feature_screen.profile.ubiquitous),
                    "n_actives": feature_screen.profile.n_actives,
                    "n_background": feature_screen.profile.n_background,
                    # Recorded explicitly rather than inferred downstream. With
                    # no background, `most_discriminating` falls back to every
                    # conserved feature -- which is how an aromatic ring once
                    # counted as an anchoring motif.
                    "has_background": feature_screen.profile.has_background,
                    "can_discriminate": feature_screen.profile.can_discriminate,
                    "background_summary": inputs.get("background_summary"),
                    "prevalence": {
                        name: round(value, 3)
                        for name, value in feature_screen.profile.prevalence.items()
                        if value > 0
                    },
                    "background_prevalence": {
                        name: round(value, 3)
                        for name, value in
                        feature_screen.profile.background_prevalence.items()
                        if value > 0
                    },
                },
                "n_fragments": inputs["n_fragments"],
                "policy_audit_pass_rate": inputs["policy_audit_pass_rate"],
            },
            indent=2,
        )
    )
    print(f"\nShard written to {path}")
    print(f"Elapsed: {time.monotonic() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
