#!/usr/bin/env python3
"""Measure whether docking separates known actives from property-matched decoys.

The question that decides whether docking is worth running on a target. A
scoring function that cannot rank known actives above matched decoys adds
nothing to a candidate triage, and running it anyway produces a list whose order
is noise.

The design, and why each piece is there:

**Actives come from curated ChEMBL data**, the same pipeline the QSAR model uses,
filtered to genuinely potent compounds. Potency matters: a compound at pIC50 5.5
binds weakly enough that a docking function has little to find.

**Decoys are property-matched**, not random. Vina's score grows close to linearly
with molecular size, and actives for most targets are larger than random library
compounds, so a screen against random decoys can report excellent enrichment
while measuring nothing but molecular weight. The decoys here are drawn from the
weakly-active end of the same ChEMBL set and matched on weight, lipophilicity,
hydrogen bonding, flexibility and charge, then filtered to be topologically
dissimilar from every active.

Using weak binders as decoys is a deliberate and conservative choice. They are
real measured compounds against this target rather than presumed inactives, so
there is no contamination from untested binders -- but they are also a *harder*
control than DUD-E's property-matched library compounds, because weak binders
often share the actives' chemotype. Enrichment measured this way will be lower
than a published DUD-E figure for the same target, and that is the honest
direction for the bias to run.

**A negative result is a result.** If docking does not separate the groups here,
that is worth knowing and goes in the report. It is a common outcome: docking
enrichment genuinely fails on many targets.

Usage:
    python scripts/validate_enrichment.py
    python scripts/validate_enrichment.py --n-actives 40 --decoys-per-active 3
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
    analyse_enrichment,
    box_from_ligand,
    describe_property_gap,
    parse_pdb,
    prepare_receptor_pdbqt,
    property_gap,
    screen,
    select_decoys,
    strip_to_receptor,
    toolchain_report,
    vina_available,
    write_pdb,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accession", default="P56817", help="UniProt accession")
    parser.add_argument("--name", default="BACE1")
    parser.add_argument("--pdb", default="4FRS", help="Receptor structure")
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=8.0,
        help="Minimum pActivity to count as an active",
    )
    parser.add_argument(
        "--decoy-threshold",
        type=float,
        default=5.5,
        help="Maximum pActivity to be eligible as a decoy",
    )
    parser.add_argument("--n-actives", type=int, default=30)
    parser.add_argument("--decoys-per-active", type=int, default=5)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--max-records", type=int, default=8000)
    parser.add_argument("--cache", default=".cache")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    started = time.monotonic()
    summary: dict[str, object] = {"target": args.name, "pdb": args.pdb.upper()}

    heading("0. Toolchain")
    print(toolchain_report())
    if not vina_available():
        print("\nVina is not installed; this cannot run.")
        return 2

    from chemdisco.chem.descriptors import compute_descriptors
    from chemdisco.chem.similarity import similarity_matrix
    from chemdisco.chem.standardize import inchikey_of, silence_rdkit_logs

    silence_rdkit_logs()

    # -- 1. Curated activity data ----------------------------------------
    heading(f"1. Curating {args.name} activity data")
    cache = pathlib.Path(args.cache)
    client = ChEMBLClient(cache=ResponseCache(cache / "chembl"))
    try:
        records, provenance = client.fetch_records(
            args.accession, activity_types=("IC50",), max_records=args.max_records
        )
    except ChEMBLError as error:
        print(f"Retrieval failed: {error}")
        return 1

    report = curate(
        records,
        policy=CurationPolicy(),
        compound_key=lambda record: inchikey_of(record.smiles)
        or f"unparseable:{record.compound_id}",
    )
    print(report.describe())
    print(f"\nChEMBL release: {client.release() or 'unknown'}")

    points = list(report.kept)
    if len(points) < 100:
        print("Too few curated compounds for a meaningful screen.")
        return 1

    # -- 2. Actives and the decoy pool -----------------------------------
    heading("2. Splitting into actives and a decoy pool by potency")
    actives = [p for p in points if p.pactivity.require() >= args.active_threshold]
    pool = [p for p in points if p.pactivity.require() <= args.decoy_threshold]
    actives.sort(key=lambda p: -p.pactivity.require())
    actives = actives[: args.n_actives]

    print(
        f"  actives: {len(actives)} compounds at pIC50 >= {args.active_threshold}"
    )
    print(
        f"  decoy pool: {len(pool)} compounds at pIC50 <= {args.decoy_threshold}"
    )
    if not actives or len(pool) < 10:
        print(
            "\n  Not enough compounds at the two ends of the potency range. "
            "Loosen the thresholds or raise --max-records."
        )
        return 1

    print(
        "\n  Note: these decoys are measured weak binders against this target, "
        "not presumed inactives. That removes any contamination from untested "
        "binders, but makes a harder control than a property-matched library -- "
        "weak binders often share the actives' chemotype. Enrichment here will "
        "read lower than a published DUD-E figure, which is the honest direction."
    )

    # -- 3. Property matching --------------------------------------------
    heading("3. Matching decoys to actives on properties")
    wanted = (
        "molecular_weight", "logp", "hbd", "hba", "rotatable_bonds", "formal_charge",
    )

    def descriptors_for(entries) -> tuple[list[dict[str, float]], list[int]]:
        values: list[dict[str, float]] = []
        keep: list[int] = []
        for index, point in enumerate(entries):
            result = compute_descriptors(point.smiles)
            if result.ok and result.values is not None:
                values.append({name: result.values[name] for name in wanted})
                keep.append(index)
        return values, keep

    active_props, active_keep = descriptors_for(actives)
    pool_props, pool_keep = descriptors_for(pool)
    actives = [actives[i] for i in active_keep]
    pool = [pool[i] for i in pool_keep]
    print(f"  descriptors computed for {len(actives)} actives, {len(pool)} pool")

    similarity = similarity_matrix(
        [p.smiles for p in pool], [p.smiles for p in actives]
    )
    similarity = np.where(similarity < 0, 0.0, similarity)

    selection = select_decoys(
        active_props,
        pool_props,
        similarity,
        decoys_per_active=args.decoys_per_active,
    )
    print(selection.describe())

    if selection.n_selected < 10:
        print(
            "\n  Too few matched decoys for a meaningful enrichment estimate. "
            "The pool is too small or too different from the actives."
        )
        return 1

    decoys = [pool[i] for i in selection.decoy_indices]
    decoy_props = [pool_props[i] for i in selection.decoy_indices]

    gaps = property_gap(active_props, decoy_props)
    gap_text = describe_property_gap(gaps)
    print("\n" + gap_text)
    gap_warning = gap_text if "WARNING" in gap_text else ""
    summary["decoys"] = {
        "n_actives": len(actives),
        "n_decoys": len(decoys),
        "property_gap": gaps,
        "adequate": selection.is_adequate,
    }

    # -- 4. Receptor ------------------------------------------------------
    heading(f"4. Preparing the {args.pdb.upper()} receptor")
    pdb_text = fetch_pdb(args.pdb, cache / "pdb")
    if pdb_text is None:
        print("Could not retrieve the structure.")
        return 1
    structure = parse_pdb(pdb_text)
    ligand = structure.best_ligand()
    if ligand is None:
        print("No co-crystallised ligand, so no evidence-based box.")
        return 1
    if structure.ligand_is_peptide_fragment(ligand) is not None:
        print(
            f"{ligand.name} is a fragment of a peptide ligand chain; this "
            "structure cannot define a box reliably."
        )
        return 1

    box = box_from_ligand(ligand)
    print(box.describe())
    receptor_pdbqt, method = prepare_receptor_pdbqt(
        write_pdb(strip_to_receptor(structure), title=f"{args.pdb.upper()} receptor")
    )
    if receptor_pdbqt is None:
        print(f"Receptor preparation failed: {method}")
        return 1
    print(f"  prepared with {method}")

    # -- 5. Screening -----------------------------------------------------
    heading("5. Docking actives and decoys into the same receptor and box")
    ligands = [p.smiles for p in actives] + [p.smiles for p in decoys]
    labels = [1] * len(actives) + [0] * len(decoys)
    print(f"  {len(ligands)} ligands at exhaustiveness {args.exhaustiveness}")

    def progress(done: int, total: int, _result) -> None:
        if done % 25 == 0 or done == total:
            print(f"    {done}/{total}", flush=True)

    screen_result = screen(
        ligands,
        receptor_pdbqt,
        box,
        labels=labels,
        receptor_id=args.pdb.upper(),
        progress=progress,
        exhaustiveness=args.exhaustiveness,
        n_poses=3,
    )
    print("\n" + screen_result.describe())

    # -- 6. Enrichment ----------------------------------------------------
    heading("6. Did the score separate them?")
    scored_labels, scores = screen_result.scored()
    if sum(scored_labels) < 5 or len(scored_labels) - sum(scored_labels) < 5:
        print("Too few surviving compounds in one group to measure enrichment.")
        return 1

    enrichment = analyse_enrichment(
        scored_labels, scores, property_gap_warning=gap_warning
    )
    print(enrichment.describe())

    active_scores = [s for s, label in zip(scores, scored_labels) if label == 1]
    decoy_scores = [s for s, label in zip(scores, scored_labels) if label == 0]
    print(
        f"\n  mean score: actives {np.mean(active_scores):.2f}, "
        f"decoys {np.mean(decoy_scores):.2f} kcal/mol "
        f"(difference {np.mean(active_scores) - np.mean(decoy_scores):+.2f})"
    )

    summary["enrichment"] = {
        "auc": enrichment.auc.estimate,
        "auc_interval": [enrichment.auc.low, enrichment.auc.high],
        "ef1": enrichment.ef1,
        "ef5": enrichment.ef5,
        "bedroc": enrichment.bedroc,
        "separates": enrichment.separates,
        "mean_active_score": float(np.mean(active_scores)),
        "mean_decoy_score": float(np.mean(decoy_scores)),
    }

    # -- Verdict ----------------------------------------------------------
    heading("Verdict")
    if enrichment.separates:
        print(
            f"Docking separates actives from matched decoys on {args.name} "
            f"(AUC {enrichment.auc.label()}). It can be used to discard "
            "candidates that clearly do not fit the pocket -- as a filter against "
            "the actives' score distribution, not as a ranking."
        )
    else:
        print(
            f"Docking does NOT demonstrably separate actives from matched decoys "
            f"on {args.name} (AUC {enrichment.auc.label()}, reaching below "
            "random). On this evidence it should not be used to triage "
            "candidates here at all.\n"
            "This is a finding, not a failure. Docking enrichment fails on many "
            "targets, and the alternative -- a ranked candidate list ordered by a "
            "score that carries no signal -- is worse than no list."
        )
    print(f"\nElapsed: {time.monotonic() - started:.0f}s")

    if args.output:
        path = pathlib.Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, default=str))
        print(f"Summary written to {path}")

    # A negative result is reported, not failed: the run did its job.
    return 0


if __name__ == "__main__":
    sys.exit(main())
