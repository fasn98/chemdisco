#!/usr/bin/env python3
"""Cross-docking: does this setup place a DIFFERENT ligand correctly?

What redocking established, and what it did not
-----------------------------------------------
Redocking puts a ligand back into the receptor conformation that same ligand
induced. It is the easiest version of the problem and the only one measured here
so far: on 4FRS the correct pose came third at 1.82 A with the top-ranked pose
4.19 A out. That showed the mechanics are right -- box, preparation, scoring,
RMSD -- and nothing more.

Cross-docking asks the question that matters for a generated candidate: given a
receptor conformation induced by some *other* molecule, does the setup place this
one where the crystal says it goes? Every candidate on the shortlist was docked
into 4FRS, a conformation induced by 0V6, and none of them is 0V6.

What a negative result would and would not invalidate
----------------------------------------------------
This distinction decides what survives, so it is stated before the numbers.

**Depends on pose accuracy:** the shortlist's poses, and therefore the scores read
off them, and therefore the ligand efficiencies computed from those scores. If
the setup cannot place a different ligand, those efficiencies rest on geometry
that may be wrong.

**Does NOT depend on pose accuracy:** the enrichment result, AUC 0.731
[0.617, 0.831]. Enrichment measures whether actives score better than
property-matched decoys. It is a statement about the *separation of scores*
between two populations, not about any individual pose being right, and a scoring
function can rank groups correctly while placing individual molecules badly --
which is exactly what the redocking result already showed on this target.

Method
------
For every ordered pair (ligand of X, receptor of Y):

1. Build the box from **Y's own** co-crystal ligand. That is the realistic case:
   the conformation available is the one some other molecule induced.
2. Dock X's ligand into Y's prepared receptor.
3. Superpose Y's receptor onto X's, and carry the docked pose into X's frame.
4. RMSD against X's ligand as observed in **X's own crystal**.

The diagonal is redocking and is the control. If it does not reproduce the 1.82 A
already on record for 4FRS, the fault is in this arrangement rather than in
cross-docking, and no off-diagonal number means anything.

Step 3 is where this fails silently -- see :mod:`chemdisco.dock.superpose`. Two
crystals of one protein sit wherever their unit cells put them, so without
reconciling the frames the RMSD measures the gap between two origins and still
looks like a pose error. Every transform used here reports its own residual and
is refused if that residual approaches the 2 A criterion it is meant to judge.

The 2 A threshold is the redocking threshold, for the reason already recorded in
the README: it is the conventional criterion, and the RMSD used is
``nearest_neighbour_rmsd``, a documented *lower bound* on the symmetry-corrected
value. A pose failing at this threshold would fail a stricter one too.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from chemdisco.dock import (  # noqa: E402
    box_from_ligand,
    dock,
    match_alpha_carbons,
    nearest_neighbour_rmsd,
    parse_pdb,
    prepare_receptor_pdbqt,
    strip_to_receptor,
    superpose,
    toolchain_report,
    vina_available,
    write_pdb,
)
from scripts.validate_docking import fetch_pdb  # noqa: E402

#: The redocking criterion, unchanged and for the same reason.
RMSD_THRESHOLD_A = 2.0


def heading(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)


def load_structure(pdb_id: str, cache: pathlib.Path):
    text = fetch_pdb(pdb_id, cache)
    if text is None:
        return None
    structure = parse_pdb(text)
    ligand = structure.best_ligand()
    if ligand is None:
        return None
    return structure, ligand


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default="runs/crossdock_set.json")
    parser.add_argument("--exhaustiveness", type=int, default=16)
    parser.add_argument("--n-poses", type=int, default=9)
    parser.add_argument("--padding", type=float, default=8.0)
    parser.add_argument("--cpu", type=int, default=6)
    parser.add_argument("--cache", default=".cache/pdb")
    parser.add_argument("--output", default="runs/crossdocking.json")
    args = parser.parse_args()

    cache = pathlib.Path(args.cache)
    selection = json.loads(pathlib.Path(args.set).read_text())
    entries = selection["selected"]
    ids = [entry["pdb_id"] for entry in entries]
    smiles_of = {entry["pdb_id"]: entry["smiles"] for entry in entries}

    heading("0. Toolchain and set")
    print(toolchain_report())
    if not vina_available():
        print("\nVina is not installed; this cannot run.")
        return 2
    print(f"\n  {len(ids)} structures: {', '.join(ids)}")
    print(f"  most similar ligand pair in the set: {selection['most_similar_pair']:.2f} Tanimoto")
    print(f"  settings: exhaustiveness {args.exhaustiveness}, {args.n_poses} poses, "
          f"cpu {args.cpu}, box padding {args.padding} A")

    heading("1. Loading structures and preparing receptors")
    loaded: dict[str, dict] = {}
    for pdb_id in ids:
        found = load_structure(pdb_id, cache)
        if found is None:
            print(f"  {pdb_id}: could not load or has no ligand -- cannot continue")
            return 1
        structure, ligand = found
        receptor_atoms = strip_to_receptor(structure)
        prepared, method = prepare_receptor_pdbqt(
            write_pdb(receptor_atoms, title=f"{pdb_id} receptor")
        )
        if prepared is None:
            print(f"  {pdb_id}: receptor preparation failed: {method}")
            return 1
        loaded[pdb_id] = {
            "structure": structure,
            "ligand": ligand,
            "receptor_atoms": receptor_atoms,
            "pdbqt": prepared,
            "box": box_from_ligand(ligand, padding=args.padding),
            "crystal": [atom.coordinates for atom in ligand.heavy_atoms],
        }
        print(
            f"  {pdb_id}: ligand {ligand.name} ({ligand.n_heavy_atoms} atoms, "
            f"chain {ligand.chain}), {len(receptor_atoms)} receptor atoms, "
            f"prepared with {method}",
            flush=True,
        )

    heading("2. Aligning every receptor pair, BEFORE docking anything")
    print(
        "  A transform is checked before it is trusted. Residue numbers are not\n"
        "  comparable across PDB entries, so the numbering offset is measured by\n"
        "  residue-identity agreement rather than assumed."
    )
    transforms: dict[tuple[str, str], object] = {}
    alignment_rows: list[dict] = []
    for receptor_id in ids:
        for ligand_id in ids:
            if receptor_id == ligand_id:
                continue
            mobile = loaded[receptor_id]
            target = loaded[ligand_id]
            points_a, points_b, correspondence = match_alpha_carbons(
                mobile["receptor_atoms"],
                target["receptor_atoms"],
                mobile_chain=mobile["ligand"].chain,
                target_chain=target["ligand"].chain,
            )
            if not correspondence.is_trustworthy or len(points_a) < 3:
                print(
                    f"  {receptor_id} -> {ligand_id}: REFUSED. "
                    f"{correspondence.describe()}"
                )
                alignment_rows.append({
                    "receptor": receptor_id, "ligand_source": ligand_id,
                    "trustworthy": False, "reason": correspondence.describe(),
                })
                continue
            result = superpose(points_a, points_b)
            ok = result.is_trustworthy
            print(
                f"  {receptor_id} -> {ligand_id}: offset {correspondence.offset:+d}, "
                f"identity {correspondence.identity_agreement:.3f}, "
                f"{result.n_matched} CA, residual {result.residual_rmsd:.3f} A "
                f"-- {'OK' if ok else 'REFUSED'}",
                flush=True,
            )
            alignment_rows.append({
                "receptor": receptor_id, "ligand_source": ligand_id,
                "offset": correspondence.offset,
                "identity_agreement": correspondence.identity_agreement,
                "n_matched": result.n_matched,
                "residual_rmsd": result.residual_rmsd,
                "trustworthy": ok,
            })
            if ok:
                transforms[(receptor_id, ligand_id)] = result

    usable = sum(1 for row in alignment_rows if row["trustworthy"])
    print(f"\n  {usable} of {len(alignment_rows)} off-diagonal alignments usable")
    if usable == 0:
        print("  No cross-docking pair can be measured. Stopping.")
        return 1

    heading("3. Docking every pair")
    print(
        "  Each cell: ligand of ROW docked into receptor of COLUMN, RMSD against\n"
        "  the row ligand's own crystal pose. Diagonal is redocking.",
    )
    cells: list[dict] = []
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def flush_partial() -> None:
        """Write after every cell, so a teardown costs one pair and not the run."""
        output.write_text(json.dumps({
            "set": selection,
            "settings": {
                "exhaustiveness": args.exhaustiveness,
                "n_poses": args.n_poses,
                "padding": args.padding,
                "cpu": args.cpu,
                "rmsd_threshold_a": RMSD_THRESHOLD_A,
                "rmsd_metric": "nearest_neighbour_rmsd (lower bound; see engine.py)",
            },
            "alignments": alignment_rows,
            "cells": cells,
            "complete": len(cells) == len(ids) * len(ids),
        }, indent=2))

    started = time.monotonic()
    for ligand_id in ids:
        for receptor_id in ids:
            is_diagonal = ligand_id == receptor_id
            label = f"{ligand_id} ligand -> {receptor_id} receptor"
            if not is_diagonal and (receptor_id, ligand_id) not in transforms:
                print(f"  {label}: SKIPPED, alignment refused", flush=True)
                cells.append({
                    "ligand_source": ligand_id, "receptor": receptor_id,
                    "diagonal": is_diagonal, "skipped": "alignment refused",
                })
                flush_partial()
                continue

            result = dock(
                smiles_of[ligand_id],
                loaded[receptor_id]["pdbqt"],
                loaded[receptor_id]["box"],
                receptor_id=receptor_id,
                exhaustiveness=args.exhaustiveness,
                n_poses=args.n_poses,
                cpu=args.cpu,
            )
            if not result.ok:
                print(f"  {label}: docking FAILED -- {result.error}", flush=True)
                cells.append({
                    "ligand_source": ligand_id, "receptor": receptor_id,
                    "diagonal": is_diagonal, "error": result.error,
                })
                flush_partial()
                continue

            crystal = loaded[ligand_id]["crystal"]
            transform = None if is_diagonal else transforms[(receptor_id, ligand_id)]
            poses: list[dict] = []
            for pose in result.poses:
                coordinates = pose.coordinates()
                if not coordinates:
                    continue
                # Carry the pose from the receptor's frame into the ligand
                # source's frame. The diagonal needs no transform by construction.
                if transform is not None:
                    coordinates = transform.apply(coordinates)
                poses.append({
                    "rank": pose.rank,
                    "score": pose.score,
                    "rmsd": nearest_neighbour_rmsd(crystal, coordinates),
                })
            if not poses:
                print(f"  {label}: no pose coordinates", flush=True)
                cells.append({
                    "ligand_source": ligand_id, "receptor": receptor_id,
                    "diagonal": is_diagonal, "error": "no pose coordinates",
                })
                flush_partial()
                continue

            top = poses[0]
            best = min(poses, key=lambda row: row["rmsd"])
            cells.append({
                "ligand_source": ligand_id,
                "receptor": receptor_id,
                "diagonal": is_diagonal,
                "top_pose_rmsd": top["rmsd"],
                "top_pose_score": top["score"],
                "best_pose_rmsd": best["rmsd"],
                "best_pose_rank": best["rank"],
                "best_pose_score": best["score"],
                "n_poses": len(poses),
                "poses": poses,
                "alignment_residual": None if transform is None else transform.residual_rmsd,
            })
            verdict = "within" if best["rmsd"] < RMSD_THRESHOLD_A else "OUTSIDE"
            print(
                f"  {label}: top {top['rmsd']:.2f} A, best {best['rmsd']:.2f} A "
                f"(rank {best['rank']}) -- {verdict} {RMSD_THRESHOLD_A} A",
                flush=True,
            )
            flush_partial()

    print(f"\n  docking took {(time.monotonic() - started) / 60:.1f} minutes")
    flush_partial()
    print(f"\nWritten to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
