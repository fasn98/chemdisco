#!/usr/bin/env python3
"""Pick a chemotype-diverse set of BACE1 structures for cross-docking.

Why this is a script and not a list of PDB ids
----------------------------------------------
Two separate failures it exists to prevent.

**Guessing the structure.** A hand-picked PDB id costs a whole docking run when
the "ligand" turns out to be a peptide deposited as a polymer chain. 1FKN did
exactly that here, which is why ``validate_docking.py --survey`` exists. The
candidate list comes from an RCSB query on the UniProt accession, and every entry
is surveyed before it is used.

**Guessing the diversity.** Cross-docking two analogues of one congeneric series
is very nearly redocking: the receptor was induced by a molecule almost identical
to the one being placed, so a good RMSD demonstrates nothing. Chemotype
distinctness is the whole point of the experiment, so it is measured by
fingerprint similarity rather than eyeballed from ligand codes.

4FRS is pinned into the set regardless of what the diversity selection prefers,
because the diagonal of the cross-docking matrix is a redocking control and 4FRS
is the one structure whose redocking result is already on record (correct pose
third at 1.82 A, top-ranked pose 4.19 A). If the diagonal does not reproduce
that, the fault is in the cross-docking arrangement and not in cross-docking.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from chemdisco.dock import parse_pdb  # noqa: E402
from scripts.validate_docking import fetch_ligand_smiles, fetch_pdb  # noqa: E402

#: BACE1. The query is by accession so the candidate list is the database's, not
#: a remembered one.
BACE1_ACCESSION = "P56817"

#: Always included: the diagonal control, with a published redocking result.
PINNED = "4FRS"

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"


def query_structures(accession: str, max_resolution: float, rows: int) -> list[str]:
    """PDB entries for one UniProt accession, best resolution first."""
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_container_identifiers"
                        ".reference_sequence_identifiers.database_accession",
                        "operator": "exact_match",
                        "value": accession,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_container_identifiers"
                        ".reference_sequence_identifiers.database_name",
                        "operator": "exact_match",
                        "value": "UniProt",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": max_resolution,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": rows},
            "sort": [
                {
                    "sort_by": "rcsb_entry_info.resolution_combined",
                    "direction": "asc",
                }
            ],
        },
    }
    request = urllib.request.Request(
        SEARCH_URL,
        data=json.dumps(query).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    return [hit["identifier"] for hit in payload.get("result_set", [])]


def inspect(pdb_id: str, cache: pathlib.Path) -> dict | None:
    """Survey one entry and fetch its ligand chemistry. None if unusable."""
    text = fetch_pdb(pdb_id, cache)
    if text is None:
        return None
    structure = parse_pdb(text)
    ligand = structure.best_ligand()
    if ligand is None:
        return None
    if structure.ligand_is_peptide_fragment(ligand) is not None:
        return None
    smiles, source = fetch_ligand_smiles(ligand.name, cache)
    if smiles is None:
        return None
    return {
        "pdb_id": pdb_id,
        "ligand_code": ligand.name,
        "ligand_key": ligand.key,
        "n_heavy_atoms": ligand.n_heavy_atoms,
        "resolution": structure.resolution,
        "smiles": smiles,
        "smiles_source": source,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accession", default=BACE1_ACCESSION)
    parser.add_argument("--max-resolution", type=float, default=2.0)
    parser.add_argument("--rows", type=int, default=40, help="candidates to query")
    parser.add_argument("--inspect", type=int, default=24, help="candidates to survey")
    parser.add_argument("--n-select", type=int, default=5)
    parser.add_argument(
        "--max-similarity",
        type=float,
        default=0.45,
        help=(
            "Reject a candidate whose ligand is more similar than this to one "
            "already selected. Cross-docking within a congeneric series is "
            "nearly redocking and tests nothing"
        ),
    )
    parser.add_argument("--min-heavy-atoms", type=int, default=20)
    parser.add_argument("--max-heavy-atoms", type=int, default=40)
    parser.add_argument("--cache", default=".cache/pdb")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    from chemdisco.chem.similarity import tanimoto

    cache = pathlib.Path(args.cache)

    print("=" * 78)
    print(f"1. Candidates from RCSB for {args.accession} at <= {args.max_resolution} A")
    print("=" * 78)
    candidates = query_structures(args.accession, args.max_resolution, args.rows)
    print(f"  {len(candidates)} entries returned, best resolution first")
    if PINNED not in candidates:
        candidates.append(PINNED)
        print(f"  {PINNED} appended: it is the diagonal control and must be present")

    print("\n" + "=" * 78)
    print(f"2. Surveying the first {args.inspect}, plus the pinned control")
    print("=" * 78)
    to_inspect = candidates[: args.inspect]
    if PINNED not in to_inspect:
        to_inspect.append(PINNED)

    usable: list[dict] = []
    for pdb_id in to_inspect:
        record = inspect(pdb_id, cache)
        if record is None:
            print(f"  {pdb_id}: unusable (no ligand, peptide fragment, or no chemistry)")
            continue
        if not (args.min_heavy_atoms <= record["n_heavy_atoms"] <= args.max_heavy_atoms):
            print(
                f"  {pdb_id}: {record['ligand_code']} has "
                f"{record['n_heavy_atoms']} heavy atoms, outside "
                f"{args.min_heavy_atoms}-{args.max_heavy_atoms}"
            )
            continue
        usable.append(record)
        print(
            f"  {pdb_id}: {record['ligand_code']}, {record['n_heavy_atoms']} atoms, "
            f"{record['resolution']:.2f} A"
        )

    if len(usable) < args.n_select:
        print(f"\nOnly {len(usable)} usable structures; need {args.n_select}.")
        return 1

    print("\n" + "=" * 78)
    print(f"3. Selecting {args.n_select} chemically distinct ligands")
    print("=" * 78)
    print(
        f"  Rejecting any candidate above {args.max_similarity} Tanimoto to one\n"
        "  already chosen. Two analogues of one series would make cross-docking\n"
        "  into redocking with extra steps."
    )

    # The control first, so diversity is measured against it rather than it being
    # squeezed in afterwards.
    pinned = next((r for r in usable if r["pdb_id"] == PINNED), None)
    if pinned is None:
        print(f"\n  {PINNED} is not usable, which breaks the diagonal control.")
        return 1
    selected = [pinned]
    print(f"\n  {PINNED} ({pinned['ligand_code']}) selected as the diagonal control")

    for record in usable:
        if len(selected) >= args.n_select:
            break
        if record["pdb_id"] == PINNED:
            continue
        similarities = [
            tanimoto(record["smiles"], chosen["smiles"]) for chosen in selected
        ]
        worst = max((s for s in similarities if s is not None), default=None)
        if worst is None:
            print(f"  {record['pdb_id']}: similarity could not be computed, skipped")
            continue
        if worst > args.max_similarity:
            print(
                f"  {record['pdb_id']} ({record['ligand_code']}): rejected, "
                f"{worst:.2f} Tanimoto to an already-selected ligand"
            )
            continue
        selected.append(record)
        print(
            f"  {record['pdb_id']} ({record['ligand_code']}): selected, "
            f"max {worst:.2f} Tanimoto to the set"
        )

    print("\n" + "=" * 78)
    print("4. The selected set, and its pairwise similarity")
    print("=" * 78)
    for record in selected:
        print(
            f"  {record['pdb_id']}  {record['ligand_code']:>4}  "
            f"{record['n_heavy_atoms']:>2} atoms  {record['resolution']:.2f} A  "
            f"{record['smiles']}"
        )
    print("\n  pairwise Tanimoto:")
    header = "        " + "".join(f"{r['pdb_id']:>8}" for r in selected)
    print(header)
    worst_pair = 0.0
    for row in selected:
        cells = []
        for column in selected:
            if row["pdb_id"] == column["pdb_id"]:
                cells.append(f"{'-':>8}")
                continue
            value = tanimoto(row["smiles"], column["smiles"])
            cells.append(f"{value:>8.2f}" if value is not None else f"{'n/a':>8}")
            if value is not None:
                worst_pair = max(worst_pair, value)
        print(f"  {row['pdb_id']:>6}" + "".join(cells))
    print(f"\n  most similar pair in the set: {worst_pair:.2f} Tanimoto")
    if worst_pair > args.max_similarity:
        print("  WARNING: above the threshold; the set is less diverse than intended")
    else:
        print(
            "  Every pair is below the threshold, so no pair of cross-docking runs\n"
            "  is a disguised redocking."
        )

    payload = {
        "accession": args.accession,
        "max_resolution": args.max_resolution,
        "n_candidates_queried": len(candidates),
        "n_usable": len(usable),
        "max_similarity_allowed": args.max_similarity,
        "most_similar_pair": worst_pair,
        "pinned_control": PINNED,
        "selected": selected,
    }
    if args.output:
        path = pathlib.Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        print(f"\nWritten to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
