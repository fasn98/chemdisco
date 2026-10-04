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
    balance_selection,
    box_from_ligand,
    describe_property_gap,
    interleave_by_label,
    parse_pdb,
    prepare_receptor_pdbqt,
    property_gap,
    screen,
    select_decoys,
    shard_by_label,
    strip_to_receptor,
    toolchain_report,
    vina_available,
    write_pdb,
)


def combine(directory: pathlib.Path, output: str) -> int:
    """Analyse the pooled output of several docking shards.

    Each shard docks a slice of the same ligand list into the same receptor and
    box, so their scores are directly comparable -- which is the condition that
    makes sharding valid at all. The check below enforces it rather than
    assuming it.
    """
    heading("Combining shard results")
    shards = sorted(directory.glob("shard_*.json"))
    if not shards:
        print(f"No shard files found in {directory}")
        return 1

    labels: list[int] = []
    scores: list[float] = []
    receptors: set[str] = set()
    boxes: set[str] = set()
    gap_warnings: set[str] = set()
    total_requested = 0
    total_docked = 0
    seconds = 0.0

    for path in shards:
        payload = json.loads(path.read_text())
        labels.extend(payload["labels"])
        scores.extend(payload["scores"])
        receptors.add(payload.get("receptor_id", ""))
        boxes.add(payload.get("box_signature", ""))
        if payload.get("gap_warning"):
            gap_warnings.add(payload["gap_warning"])
        total_requested += payload.get("n_requested", 0)
        total_docked += len(payload["scores"])
        seconds += payload.get("elapsed_seconds", 0.0)
        print(
            f"  {path.name}: {len(payload['scores'])} docked "
            f"({sum(payload['labels'])} actives)"
        )

    if len(receptors) > 1 or len(boxes) > 1:
        print(
            f"\nREFUSING: the shards used {len(receptors)} receptor(s) and "
            f"{len(boxes)} box(es). Vina scores are only comparable within one "
            "receptor and one box; pooling them would produce a ranking ordered "
            "mostly by which setup each ligand happened to get."
        )
        return 1

    print(
        f"\nPooled: {total_docked} of {total_requested} ligands, "
        f"{sum(labels)} actives and {len(labels) - sum(labels)} decoys, "
        f"{seconds / 60:.0f} CPU-minutes across {len(shards)} shard(s)"
    )

    if sum(labels) < 5 or len(labels) - sum(labels) < 5:
        print("Too few in one group to measure enrichment.")
        return 1

    heading("Did the score separate them?")
    enrichment = analyse_enrichment(
        labels, scores, property_gap_warning="; ".join(sorted(gap_warnings))
    )
    print(enrichment.describe())

    active_scores = [s for s, label in zip(scores, labels, strict=True) if label == 1]
    decoy_scores = [s for s, label in zip(scores, labels, strict=True) if label == 0]
    print(
        f"\n  mean score: actives {np.mean(active_scores):.2f}, "
        f"decoys {np.mean(decoy_scores):.2f} kcal/mol "
        f"(difference {np.mean(active_scores) - np.mean(decoy_scores):+.2f})"
    )

    heading("Verdict")
    if enrichment.verdict == "separates":
        print(
            f"Docking separates actives from matched decoys (AUC "
            f"{enrichment.auc.label()}). It can filter candidates that clearly "
            "do not fit the pocket -- against the actives' score distribution, "
            "not as a ranking."
        )
    elif enrichment.verdict == "does not separate":
        print(
            f"Docking does not separate actives from matched decoys (AUC "
            f"{enrichment.auc.label()}). It should not be used to triage "
            "candidates on this target.\nThis is a finding, not a failure."
        )
    else:
        needed = enrichment.compounds_needed()
        print(
            f"INCONCLUSIVE (AUC {enrichment.auc.label()}). This sample cannot "
            "settle the question in either direction, and saying otherwise "
            "would turn absence of evidence into evidence of absence."
        )
        if needed:
            print(f"Roughly {needed} compounds per group would be needed.")

    if output:
        path = pathlib.Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "n_actives": enrichment.n_actives,
                    "n_decoys": enrichment.n_decoys,
                    "auc": enrichment.auc.estimate,
                    "auc_interval": [enrichment.auc.low, enrichment.auc.high],
                    "verdict": enrichment.verdict,
                    "ef1": enrichment.ef1,
                    "bedroc": enrichment.bedroc,
                },
                indent=2,
            )
        )
        print(f"\nSummary written to {path}")
    return 0


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
    parser.add_argument("--n-actives", type=int, default=20)
    parser.add_argument("--decoys-per-active", type=int, default=3)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument(
        "--time-budget",
        type=float,
        default=1500.0,
        help=(
            "Seconds to spend docking before stopping and reporting what was "
            "done. A screen killed by its environment's limit returns nothing"
        ),
    )
    parser.add_argument("--max-records", type=int, default=8000)
    parser.add_argument("--cache", default=".cache")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--shard", type=int, default=0, help="This worker's index, 0-based"
    )
    parser.add_argument(
        "--n-shards",
        type=int,
        default=1,
        help=(
            "Split the docking across this many workers. The power calculation "
            "from the first run asked for ~68 compounds per group, which is "
            "hours of docking on one runner and minutes across several"
        ),
    )
    parser.add_argument(
        "--combine",
        default="",
        help="Directory of shard outputs to analyse instead of docking",
    )
    args = parser.parse_args()

    if args.combine:
        return combine(pathlib.Path(args.combine), args.output)

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

    # Per-pair matching is not enough: the first run had every decoy within
    # 25 Da of its matched active while the group means differed by 34.7 Da,
    # because the matcher could not fill every quota and the decoys it did find
    # skewed small.
    balanced, balance_note = balance_selection(active_props, decoy_props)
    print("\n" + balance_note)
    decoys = [decoys[i] for i in balanced]
    decoy_props = [decoy_props[i] for i in balanced]

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
    # Interleaved so a truncated run stays balanced. In blocks, a run cut short
    # by the time budget would hold every active and no decoys.
    ligands, labels = interleave_by_label(ligands, labels)

    if args.n_shards > 1:
        ligands, labels = shard_by_label(
            ligands, labels, shard=args.shard, n_shards=args.n_shards
        )
        print(
            f"  shard {args.shard + 1} of {args.n_shards}: {len(ligands)} ligands "
            f"({sum(labels)} actives, {len(labels) - sum(labels)} decoys)"
        )
    print(
        f"  {len(ligands)} ligands at exhaustiveness {args.exhaustiveness}, "
        f"interleaved, budget {args.time_budget:.0f}s"
    )

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
        time_budget_seconds=args.time_budget,
    )
    print("\n" + screen_result.describe())

    if args.n_shards > 1:
        # A shard writes its scores and stops. Analysing a shard alone would
        # report an interval from a fraction of the data as though it were the
        # whole screen.
        scored_labels, scores = screen_result.scored()
        path = pathlib.Path(args.output or f"runs/shard_{args.shard}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "shard": args.shard,
                    "labels": scored_labels,
                    "scores": scores,
                    "receptor_id": args.pdb.upper(),
                    "box_signature": f"{box.center}|{box.size}",
                    "gap_warning": gap_warning,
                    "n_requested": len(ligands),
                    "elapsed_seconds": screen_result.elapsed_seconds,
                },
                indent=2,
            )
        )
        print(
            f"\nShard written to {path}. Combine with --combine once every "
            "shard has finished; a single shard cannot support a verdict."
        )
        return 0

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

    active_scores = [s for s, label in zip(scores, scored_labels, strict=True) if label == 1]
    decoy_scores = [s for s, label in zip(scores, scored_labels, strict=True) if label == 0]
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
