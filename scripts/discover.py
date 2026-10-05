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
import hashlib
import json
import pathlib
import sys
import time
import urllib.request
from dataclasses import replace

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from chemdisco.curate import CurationPolicy, curate  # noqa: E402
from chemdisco.data.chembl import ChEMBLClient, ChEMBLError, ResponseCache  # noqa: E402
from chemdisco.dock import (  # noqa: E402
    VINA_ERROR_KCAL,
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

    # The explicit tie-break makes the ranking independent of this file. Curation
    # already sorts its output by (target_id, compound_id) and Python's sort is
    # stable, so potency alone is in fact reproducible -- but ties are frequent,
    # reported activities cluster on round numbers, and the top 30 of this list
    # become the docking reference while the top 50 become the fragment seeds. The
    # ranking should not depend on a guarantee made two modules away.
    #
    # The open question this does *not* answer: two runs of this pipeline on the
    # same target with the same seed docked 48 and 47 candidates, of which 12 and
    # 22 cleared the same efficiency threshold. Vina is seeded and curation is
    # order-independent, so neither explains it. The ligand signature recorded in
    # each shard is the instrument for localising it -- within a run first.
    points = sorted(
        report.kept, key=lambda p: (-p.pactivity.require(), p.compound_id)
    )
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

    # The anchoring motif is MEASURED here, not declared. The feature profile is
    # built before generation so the generator can be constrained by whatever
    # separates this target's actives from unrelated chemistry -- amidine on BACE1,
    # something else on a kinase, nothing at all where no feature discriminates.
    #
    # Without the constraint, the previous run docked 60 candidates, 15 reached the
    # reference median ligand efficiency, and all 15 carried the amidine twice. In a
    # fragment space built from potent inhibitors, joining two warheads is how the
    # builder makes a compact molecule that scores well.
    from chemdisco.generate.pharmacophore import FEATURE_PATTERNS, profile_actives

    anchor_smarts: tuple[str, ...] = ()
    profile = profile_actives(
        [p.smiles for p in reference], background_smiles=background
    )
    if profile.can_discriminate:
        anchor_smarts = tuple(
            FEATURE_PATTERNS[name]
            for name in profile.most_discriminating
            if name in FEATURE_PATTERNS
        )
        print(
            "  anchoring motif(s) measured from the actives: "
            + ", ".join(profile.most_discriminating)
        )
    else:
        print(
            "  no anchoring motif could be measured, so generation is NOT "
            "constrained to carry one. Products may fill the pocket with nothing "
            "to bind with, and nothing downstream will catch it either -- the "
            "conserved-feature check is withheld for the same reason."
        )

    # The control switch, and it changes exactly one thing: whether the measured
    # motif constrains the fragment pool. The profile above is still measured, still
    # recorded, still handed to the feature screen, and still decides whether the
    # conserved-feature check is usable downstream -- all untouched. Anything wider
    # than this makes the control worthless, which is why `--background-source none`
    # is not the way to get it: that would disable the profile as well and move two
    # variables at once.
    #
    # Why the switch exists: the triage threshold was calibrated while watching a
    # double-warhead candidate pool that the constrained generator no longer emits.
    # Reproducing that pool on this machine, at this exhaustiveness, is the only way
    # to measure how far the distribution moved rather than argue that it must have.
    constraint_applied = bool(anchor_smarts) and not args.no_anchor_constraint
    if anchor_smarts and not constraint_applied:
        print(
            "  CONTROL RUN: the motif above was measured and is recorded, but it is "
            "NOT constraining the fragment pool. Products may carry it twice, which "
            "is the population the inherited threshold was calibrated against."
        )
    policy = replace(
        BACE1_POLICY, anchor_smarts=anchor_smarts if constraint_applied else ()
    )
    print(f"  policy: {policy.describe()}")
    generation = generate_candidates(
        [p.smiles for p in seeds],
        policy=policy,
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

    # Two signatures, because they separate two different failures. The ligand
    # signature says whether the docked list is the same; the curation signature
    # says whether the dataset underneath it was. If the second differs, the
    # retrieval changed -- a new ChEMBL release, a truncated fetch, a cache miss --
    # and chasing the generator would be chasing the wrong thing.
    ligand_signature = hashlib.sha256(
        "\n".join(
            [p.smiles for p in reference] + [c.smiles for c in candidates]
        ).encode()
    ).hexdigest()[:16]
    curation_signature = hashlib.sha256(
        "\n".join(
            f"{p.compound_id}:{p.pactivity.require():.4f}" for p in points
        ).encode()
    ).hexdigest()[:16]

    return {
        "ligand_signature": ligand_signature,
        "curation_signature": curation_signature,
        "n_curated": len(points),
        "n_actives": len(actives),
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
        # Driven by the MEASUREMENT, not by the flag, so a constrained run and its
        # control carry identical motif provenance and compare field by field.
        "anchor_motifs": list(profile.most_discriminating) if anchor_smarts else [],
        # The one field that distinguishes them. Without it two shards with the same
        # `anchor_motifs` would be indistinguishable, and the control would be
        # unidentifiable from its own record.
        "anchor_constraint_applied": constraint_applied,
        "n_anchor_fragments": generation.n_anchor_fragments,
        "n_plain_fragments": generation.n_plain_fragments,
        "n_anchor_escapes": generation.n_anchor_escapes,
        "n_anchor_lost": generation.n_anchor_lost,
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
    ligand_signatures: set[str] = set()
    curation_signatures: set[str] = set()
    settings: set[str] = set()
    seconds = 0.0
    requested = 0

    for path in shards:
        payload = json.loads(path.read_text())
        reference_scores.extend(payload["reference_scores"])
        reference_efficiencies.extend(payload.get("reference_efficiencies", []))
        candidates.extend(payload["candidates"])
        receptors.add(payload["receptor_id"])
        boxes.add(payload["box_signature"])
        if payload.get("ligand_signature"):
            ligand_signatures.add(payload["ligand_signature"])
        if payload.get("curation_signature"):
            curation_signatures.add(payload["curation_signature"])
        settings.add(
            f"exhaustiveness={payload.get('exhaustiveness', '?')},"
            f"cpu={payload.get('cpu', '?')}"
        )
        seconds += payload.get("elapsed_seconds", 0.0)
        requested += payload.get("n_requested", 0)
        print(
            f"  {path.name}: {len(payload['reference_scores'])} reference, "
            f"{len(payload['candidates'])} candidates, ligands "
            f"{payload.get('ligand_signature', 'unrecorded')}, curation "
            f"{payload.get('curation_signature', 'unrecorded')}"
        )

    if len(receptors) > 1 or len(boxes) > 1:
        print(
            "\nREFUSING: shards used different receptors or boxes. Vina scores "
            "are only comparable within one setup."
        )
        return 1

    if len(settings) > 1:
        print(
            "\nREFUSING: shards docked with different settings "
            f"({', '.join(sorted(settings))}). Vina scores produced at different "
            "exhaustiveness are not comparable, and a thread count left at 0 is "
            "whatever each machine happened to have."
        )
        return 1

    if len(ligand_signatures) > 1:
        print(
            "\nREFUSING: shards derived different ligand lists "
            f"({', '.join(sorted(ligand_signatures))}). Sharding is valid only "
            "when every worker docks a slice of one list; pooling slices of "
            "different lists is not one experiment, and the shortlist would be "
            "assembled from incomparable parts. Each shard fetches ChEMBL for "
            "itself, so the likely causes are a dataset that changed mid-run or a "
            "retrieval that returned a different number of records -- compare the "
            "curation counts the shard logs print."
        )
        if len(curation_signatures) > 1:
            print(
                "  The curation signatures differ too "
                f"({', '.join(sorted(curation_signatures))}), so the datasets "
                "themselves were not the same. The retrieval is where to look, "
                "not the generator."
            )
        else:
            print(
                "  The curation signatures AGREE, so every shard curated the same "
                "dataset and the divergence happened after it -- in generation or "
                "in candidate selection, which are supposed to be deterministic "
                "given the seed."
            )
        return 1

    print(
        f"\nPooled: {len(reference_scores)} reference actives, "
        f"{len(candidates)} candidates, {seconds / 60:.0f} CPU-minutes"
    )

    first = json.loads(shards[0].read_text())
    # Absent in shards written before the control flag existed, and there a
    # measured motif did imply an applied constraint -- so that is the default
    # rather than a guess, and an older shard still reads correctly.
    constraint_applied = first.get(
        "anchor_constraint_applied", bool(first.get("anchor_motifs"))
    )
    if first.get("anchor_motifs") and not constraint_applied:
        heading("How generation was constrained: IT WAS NOT (control run)")
        print(
            "  anchoring motif(s), measured from the actives and recorded so this "
            "run compares field by field with the constrained one: "
            + ", ".join(first["anchor_motifs"])
        )
        print(
            "  The motif did NOT constrain the fragment pool. Products could carry "
            "it more than once, which reproduces the candidate population the "
            "triage threshold was calibrated against. Read the attrition below as "
            "the control arm, not as a pipeline result: a shortlist from this run "
            "would be assembled from recombination artefacts by construction."
        )
        print(
            f"  fragment pool: all {first.get('n_fragments', 0)} fragments offered "
            "as reagents, none reserved as seeds."
        )
    elif first.get("anchor_motifs"):
        heading("How generation was constrained")
        print(
            "  anchoring motif(s), measured from the actives: "
            + ", ".join(first["anchor_motifs"])
        )
        print(
            f"  fragment pool: {first.get('n_anchor_fragments', 0)} carrying the "
            f"motif, used as build seeds; {first.get('n_plain_fragments', 0)} "
            "motif-free, used as reagents. Every product therefore grows from "
            "exactly one warhead by construction rather than being filtered "
            "afterwards -- the filter version spends the generation budget on "
            "artefacts, which is how a previous run reached 15 artefacts and an "
            "empty shortlist."
        )
        if first.get("n_anchor_escapes"):
            print(
                f"  {first['n_anchor_escapes']} product(s) carried the motif twice "
                "anyway and were rejected: joining two motif-free fragments can "
                "create it across the new bond. Counted because a structural "
                "guarantee that is never verified is only a claim."
            )
        if first.get("n_anchor_lost"):
            print(
                f"  {first['n_anchor_lost']} product(s) grew from an anchor "
                "fragment and lost the motif in the process, and were rejected."
            )
    else:
        print(
            "\n  Generation was NOT constrained to carry an anchoring motif: none "
            "could be measured from the actives against the background."
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
    # Whether the threshold can discriminate at all at this method's precision,
    # measured before it is applied rather than assumed because the number looks
    # reasonable. Ligand efficiency is per heavy atom, so a margin only becomes
    # comparable to Vina's error after multiplying back by the atom count.
    #
    # Measured on BACE1: not one of the 22 candidates clearing the reference
    # median beat it by more than 2.5 kcal/mol -- margins ran a median of 0.49 and
    # a maximum of 1.17 -- and the unconstrained control gave the same answer. The
    # threshold has never discriminated at this precision in either population, so
    # naming a "passing" subset reports as a filter something that does not filter.
    triage_withheld = False
    n_distinguishable = 0
    if reference_le:
        threshold = float(np.median(reference_le))
        passing_score = [
            c
            for c in candidates
            if c.get("ligand_efficiency") is not None
            and c["ligand_efficiency"] <= threshold
        ]
        # Distinguishable AND better. Being able to reject a clearly bad candidate
        # is not evidence the threshold can identify a good one.
        distinguishable = [
            c
            for c in passing_score
            if c.get("heavy_atoms")
            and abs(c["ligand_efficiency"] - threshold) * c["heavy_atoms"]
            > VINA_ERROR_KCAL
        ]
        n_distinguishable = len(distinguishable)
        triage_withheld = not distinguishable

        if triage_withheld:
            margins = [
                abs(c["ligand_efficiency"] - threshold) * c["heavy_atoms"]
                for c in passing_score
                if c.get("heavy_atoms")
            ]
            print(
                f"\n  TRIAGE WITHHELD. {len(passing_score)} of {len(candidates)} "
                f"candidates reach a ligand efficiency of {threshold:.3f} "
                "kcal/mol/atom, the median known active -- but not one of them "
                f"reaches it by more than Vina's own {VINA_ERROR_KCAL} kcal/mol "
                "method error"
                + (
                    f" (margins median {np.median(margins):.2f}, max "
                    f"{max(margins):.2f} kcal/mol)"
                    if margins
                    else ""
                )
                + "."
            )
            print(
                "  So the cut runs through a part of the distribution narrower "
                "than the method can resolve. Those candidates are "
                "indistinguishable from a median known active at Vina's "
                "precision, and so are most of the "
                f"{len(candidates) - len(passing_score)} above the line. A subset "
                "called 'passing' would report as a filter something that does "
                "not filter."
            )
            print(
                f"  All {len(candidates)} candidates are reported below with their "
                "efficiencies, unranked and uncut. The threshold stays a stated "
                "reference point."
            )
            # Everything that could be assessed, because a check that cannot
            # separate does not get to name a subset. Same move the
            # conserved-feature check makes when nothing clears its floor.
            passing_score = [
                c for c in candidates if c.get("ligand_efficiency") is not None
            ]
        else:
            print(
                f"\n  {len(passing_score)} of {len(candidates)} candidates reach a "
                f"ligand efficiency of {threshold:.3f} kcal/mol/atom or better, the "
                "median known active."
            )
            print(
                f"  {n_distinguishable} of them beat it by more than the "
                f"{VINA_ERROR_KCAL} kcal/mol method error, so the comparison "
                "separates something and the cut is entitled to act."
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

    if triage_withheld:
        print(
            "\n  NOT RANKED, and not a shortlist. The triage was withheld above, "
            "so these are reported in the order they were docked. Sorting them by "
            "ligand efficiency would imply an ordering the method cannot support: "
            "the whole set spans less than Vina's error, so the first entry is "
            "not better evidence than the last."
        )
    else:
        print(
            "\n  This is a filter, not a ranking. Docking on this target separates "
            "actives from decoys (AUC 0.731) but does not order them reliably "
            "(BEDROC 0.36, empty top 1%), so the survivors are listed by ligand "
            "efficiency -- which at least corrects for the size bias -- and that "
            "order still carries far less information than it appears to."
        )
        survivors.sort(key=lambda c: (c.get("ligand_efficiency") or 0.0))

    heading("Candidates, with the triage withheld" if triage_withheld
            else "The shortlist")
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
        if triage_withheld:
            print(
                f"{len(survivors)} structures that are novel, pass the filters "
                f"calibrated for this target, and {checked}are INDISTINGUISHABLE "
                "from a median known active at Vina's precision -- which is a "
                "weaker and more accurate statement than 'reach the median'.\n"
                "\n"
                "This is not a ranked shortlist and no subset of it passed a "
                "test. The efficiency threshold is printed beside each entry as a "
                "reference point, not as a line anything cleared.\n"
            )
        else:
            print(
                f"{len(survivors)} structures that are novel, pass the filters "
                f"calibrated for this target, {checked}occupy the site about as "
                "well as known inhibitors do per heavy atom.\n"
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
                    # Recorded explicitly, by the criterion this project applies
                    # to new fields: add one only when the record lacking it can
                    # be read unambiguously. It cannot be here. With the triage
                    # withheld, `n_passing_score` equals every assessed candidate
                    # and `shortlist` is not a shortlist -- a reader given only
                    # those numbers would conclude every candidate passed a test
                    # that was never applied. The verdict is not inferable from
                    # the counts, so it is stated.
                    "triage_withheld": triage_withheld,
                    "triage_n_distinguishable": n_distinguishable,
                    "vina_error_kcal": VINA_ERROR_KCAL,
                    # Named for what it is on each branch. "shortlist" would be a
                    # false label for an uncut, unranked set.
                    (
                        "candidates_unranked"
                        if triage_withheld
                        else "shortlist"
                    ): survivors[:50],
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
    parser.add_argument(
        "--cpu",
        type=int,
        default=0,
        help=(
            "Vina thread count. 0 lets Vina use what it finds, which makes the "
            "run machine dependent -- a 2-core runner and a 16-core workstation "
            "do not necessarily produce the same score. Pin it for any set of "
            "numbers meant to be compared; scripts/probe_cpu_determinism.py "
            "measures whether it matters on a given machine"
        ),
    )
    parser.add_argument(
        "--no-anchor-constraint",
        action="store_true",
        help=(
            "Measure the anchoring motif as usual but do NOT use it to constrain "
            "the fragment pool, so products may carry it more than once. This "
            "reproduces the candidate population the triage threshold was "
            "calibrated against, which is the control for asking how far the "
            "constrained generator moved the ligand-efficiency distribution. It "
            "changes the fragment pool and nothing else: the feature profile is "
            "still measured and recorded, the feature screen still runs, and the "
            "conserved-feature check follows the same path either way"
        ),
    )
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
        cpu=args.cpu,
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
                "ligand_signature": inputs["ligand_signature"],
                # Recorded so a pooled or compared score carries the settings it
                # was produced under, including the thread count, which is
                # machine dependent when left at 0.
                "exhaustiveness": args.exhaustiveness,
                "cpu": args.cpu,
                "curation_signature": inputs["curation_signature"],
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
                "anchor_motifs": inputs["anchor_motifs"],
                "anchor_constraint_applied": inputs["anchor_constraint_applied"],
                "n_anchor_fragments": inputs["n_anchor_fragments"],
                "n_plain_fragments": inputs["n_plain_fragments"],
                "n_anchor_escapes": inputs["n_anchor_escapes"],
                "n_anchor_lost": inputs["n_anchor_lost"],
                "policy_audit_pass_rate": inputs["policy_audit_pass_rate"],
            },
            indent=2,
        )
    )
    print(f"\nShard written to {path}")
    print(f"Elapsed: {time.monotonic() - started:.0f}s")

    # Printed last, and deliberately so: the only readable channel out of a
    # sharded run in some environments is the tail of its log. A signature that
    # exists in the JSON and nowhere else cannot be compared between shards or
    # between runs, which is the whole reason it is computed.
    heading("Reproducibility")
    print(f"  ligand signature:   {inputs['ligand_signature']}")
    print(f"  curation signature: {inputs['curation_signature']}")
    print(
        f"  derived from {inputs['n_curated']} curated compounds, "
        f"{inputs['n_actives']} actives, {inputs['n_generated']} generated, "
        f"{len(inputs['candidate_smiles'])} carried forward, "
        f"{len(ligands)} in this shard's slice"
    )
    print(
        "  Two shards of one run, or two runs of this pipeline, must print the "
        "same two signatures. They are the evidence that a pooled result is one "
        "experiment rather than several averaged together."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
