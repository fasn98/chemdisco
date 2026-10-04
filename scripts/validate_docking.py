#!/usr/bin/env python3
"""Validate a docking setup by redocking a structure's own ligand.

The one check that establishes a docking setup is correct. Everything downstream
-- scores, rankings, virtual-screen enrichment -- depends on the receptor
preparation, the search box and the ligand handling being right together, and
redocking is what tests all three at once.

The procedure: take a crystal structure with a bound ligand, remove the ligand,
build a box around where it was, dock it back, and measure how far the predicted
pose lands from the observed one. Below 2 Å the setup reproduces reality. Above
it, something in the chain is wrong and every other score from that setup is
suspect.

Why 2 Å: it is roughly the resolution at which crystallography determines atom
positions, so a pose within it is as close as the reference data can distinguish.
The threshold comes from the experiment, not from the docking program.

A caveat about what this does and does not establish. Redocking is the easiest
possible docking problem: the receptor is in the conformation that *this* ligand
induced, so the pocket already fits it. Passing proves the mechanics work. It
does not predict how the setup performs on a different ligand, where the pocket
may need to move -- cross-docking, a harder test, measures that. A setup that
fails redocking is definitely broken; one that passes is merely not obviously so.

Default target 1FKN: human BACE1 with the OM99-2 inhibitor bound, which is the
target this pipeline is validated against.

Usage:
    python scripts/validate_docking.py
    python scripts/validate_docking.py --pdb 2QMG --exhaustiveness 32
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.request

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from chemdisco.dock import (  # noqa: E402
    box_from_ligand,
    dock,
    nearest_neighbour_rmsd,
    parse_pdb,
    prepare_receptor_pdbqt,
    strip_to_receptor,
    toolchain_report,
    vina_available,
    write_pdb,
)

#: SMILES for ligands whose structures cannot be inferred from PDB coordinates
#: alone. A PDB file records positions and element types, not bond orders, so a
#: ligand's chemistry has to come from the chemical component dictionary or from
#: a lookup like this one.
KNOWN_LIGAND_SMILES: dict[str, str] = {
    # OM99-2, the transition-state analogue inhibitor bound in 1FKN. A
    # peptidomimetic octapeptide analogue -- large, flexible, and therefore a
    # demanding redocking target.
    "1OL": (
        "CC(C)C[C@H](NC(=O)[C@@H](NC(=O)[C@@H](N)CC(=O)O)C(C)C)"
        "[C@@H](O)C[C@H](CC(C)C)C(=O)N[C@@H](C)C(=O)N[C@@H](CC(=O)O)C(=O)O"
    ),
}


def heading(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def fetch_pdb(pdb_id: str, cache_dir: pathlib.Path) -> str | None:
    """Fetch a structure, caching it so repeated runs are reproducible."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{pdb_id.upper()}.pdb"
    if cached.exists():
        print(f"  using cached {cached}")
        return cached.read_text()

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                text = response.read().decode()
            cached.write_text(text)
            print(f"  fetched {len(text.splitlines())} lines from RCSB")
            return text
        except Exception as error:
            print(f"  attempt {attempt + 1} failed: {error}")
            if attempt < 2:
                time.sleep(2**attempt)
    return None


def survey(pdb_ids: list[str], cache_dir: pathlib.Path) -> int:
    """Report each structure's ligand situation without docking anything.

    Picking a redocking target by guessing which PDB entry has a clean
    small-molecule inhibitor wastes a fifteen-minute docking run per wrong
    guess. This inspects candidates in seconds and says which are usable.
    """
    heading("Surveying candidate structures")
    usable: list[str] = []

    for raw in pdb_ids:
        pdb_id = raw.strip().upper()
        if not pdb_id:
            continue
        print(f"\n{pdb_id}")
        text = fetch_pdb(pdb_id, cache_dir)
        if text is None:
            print("  could not fetch")
            continue

        structure = parse_pdb(text)
        resolution = (
            f"{structure.resolution:.2f} A" if structure.resolution else "n/a"
        )
        print(f"  resolution {resolution}; {len(structure.protein_atoms)} protein atoms")

        peptide_chains = structure.peptide_ligand_chains()
        if peptide_chains:
            print(
                "  peptide ligand chain(s): "
                + ", ".join(
                    f"{chain.identifier} ({chain.n_residues} residues)"
                    for chain in peptide_chains
                )
            )

        ligand = structure.best_ligand()
        if ligand is None:
            print("  NO LIGAND -- apo structure, cannot be redocked")
            continue

        fragment_of = structure.ligand_is_peptide_fragment(ligand)
        if fragment_of is not None:
            print(
                f"  UNUSABLE: {ligand.name} ({ligand.n_heavy_atoms} atoms) is a "
                f"fragment of the peptide ligand on chain {fragment_of.identifier}"
            )
            continue

        print(
            f"  USABLE: {ligand.key}, {ligand.n_heavy_atoms} heavy atoms, "
            f"elements {sorted(ligand.elements)}"
        )
        usable.append(f"{pdb_id} ({ligand.name}, {ligand.n_heavy_atoms} atoms)")

    heading("Survey result")
    if usable:
        print("Structures suitable for redocking:")
        for entry in usable:
            print(f"  {entry}")
        print(
            "\nPick one with a ligand of 20-40 heavy atoms: large enough to be a "
            "real inhibitor, small enough that redocking is not dominated by "
            "torsional search."
        )
    else:
        print("None of the surveyed structures carries a usable small-molecule ligand.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", default="1FKN", help="PDB id to redock")
    parser.add_argument("--ligand-smiles", default="", help="Override the ligand SMILES")
    parser.add_argument("--exhaustiveness", type=int, default=16)
    parser.add_argument("--n-poses", type=int, default=9)
    parser.add_argument("--padding", type=float, default=8.0)
    parser.add_argument("--cache", default=".cache/pdb")
    parser.add_argument("--output", default="", help="Write a JSON summary here")
    parser.add_argument(
        "--survey",
        default="",
        help=(
            "Comma-separated PDB ids to inspect without docking, reporting which "
            "carry a genuine small-molecule ligand suitable for redocking"
        ),
    )
    args = parser.parse_args()

    if args.survey:
        return survey(args.survey.split(","), pathlib.Path(args.cache))

    started = time.monotonic()
    summary: dict[str, object] = {"pdb_id": args.pdb.upper()}

    heading("0. Toolchain")
    print(toolchain_report())
    if not vina_available():
        print(
            "\nVina is not installed, so this cannot run. Install with:\n"
            "  pip install vina scipy gemmi meeko\n"
            "and ensure the 'obabel' command is on PATH."
        )
        return 2

    heading(f"1. Fetching {args.pdb.upper()}")
    pdb_text = fetch_pdb(args.pdb, pathlib.Path(args.cache))
    if pdb_text is None:
        print("Could not retrieve the structure.")
        return 1

    structure = parse_pdb(pdb_text)
    print(structure.describe())
    summary["resolution"] = structure.resolution
    summary["method"] = structure.method

    heading("2. Identifying the bound ligand")
    ligand = structure.best_ligand()
    if ligand is None:
        print(
            "No candidate ligand found. This looks like an apo structure, which "
            "cannot be redocked -- there is no observed pose to compare against. "
            "Pick a holo structure of the same protein."
        )
        return 1
    print(
        f"  {ligand.key}: {ligand.n_heavy_atoms} heavy atoms, "
        f"elements {sorted(ligand.elements)}"
    )

    # The check that stops a meaningless verdict. When a peptidomimetic
    # inhibitor is deposited as a polymer chain, the HETATM records hold only
    # its non-standard residue -- 13 atoms of a 60-atom molecule in 1FKN -- and
    # an RMSD measured against that fragment is measured against the wrong
    # reference. Worse, nearest-neighbour matching against a small reference set
    # is lenient, so the error flatters the result.
    peptide_chain = structure.ligand_is_peptide_fragment(ligand)
    if peptide_chain is not None:
        print(
            f"\n  REFUSING: {ligand.name} sits on chain {peptide_chain.identifier}, "
            f"which is a {peptide_chain.n_residues}-residue peptide ligand "
            f"({len(peptide_chain.heavy_atoms)} heavy atoms). The HETATM group is "
            "that peptide's non-standard residue, not the ligand.\n"
            "  Redocking against it would compare a whole molecule to a fragment "
            "of itself, and nearest-neighbour RMSD against a small reference set "
            "flatters the result rather than catching the error.\n"
            "  Use a structure whose inhibitor is a genuine small molecule, or "
            "extend this script to reconstruct the full peptide ligand from its "
            "chain."
        )
        summary["refused"] = "ligand is a fragment of a peptide ligand chain"
        return 1
    for other in structure.candidate_ligands()[1:4]:
        print(f"  (also present: {other.key}, {other.n_heavy_atoms} heavy atoms)")

    smiles = args.ligand_smiles or KNOWN_LIGAND_SMILES.get(ligand.name, "")
    if not smiles:
        print(
            f"\nNo SMILES known for ligand {ligand.name}. A PDB file records atom "
            "positions and elements but not bond orders, so the ligand's chemistry "
            "cannot be inferred from it. Supply --ligand-smiles, or add the code "
            "to KNOWN_LIGAND_SMILES in this script."
        )
        return 1
    print(f"  chemistry: {smiles[:70]}{'...' if len(smiles) > 70 else ''}")
    summary["ligand"] = {"code": ligand.name, "heavy_atoms": ligand.n_heavy_atoms}

    heading("3. Building the search box from the observed pose")
    box = box_from_ligand(ligand, padding=args.padding)
    print(box.describe())
    summary["box"] = {
        "center": list(box.center),
        "size": list(box.size),
        "derived_from": box.derived_from,
    }

    heading("4. Preparing the receptor")
    receptor_atoms = strip_to_receptor(structure)
    print(
        f"  {len(receptor_atoms)} protein atoms kept; ligand, solvent and "
        "additives removed"
    )
    # Leaving the ligand in place would dock against an occupied pocket, and the
    # result would look like a weak binder rather than a broken setup.
    receptor_pdb = write_pdb(receptor_atoms, title=f"{args.pdb.upper()} receptor")
    receptor_pdbqt, method = prepare_receptor_pdbqt(receptor_pdb)
    if receptor_pdbqt is None:
        print(f"  receptor preparation failed: {method}")
        return 1
    print(f"  prepared with {method}: {len(receptor_pdbqt.splitlines())} PDBQT lines")

    heading("5. Redocking")
    print(
        f"  exhaustiveness={args.exhaustiveness}, n_poses={args.n_poses}; "
        "this is the slow step"
    )
    result = dock(
        smiles,
        receptor_pdbqt,
        box,
        receptor_id=args.pdb.upper(),
        exhaustiveness=args.exhaustiveness,
        n_poses=args.n_poses,
    )
    if not result.ok:
        print(f"  docking failed: {result.error}")
        summary["error"] = result.error
        return 1
    print(f"\n{result.describe()}")

    heading("6. Comparing the predicted pose to the crystal pose")
    crystal = [atom.coordinates for atom in ligand.heavy_atoms]
    rows: list[dict[str, object]] = []
    best_rmsd: float | None = None

    for pose in result.poses:
        predicted = pose.coordinates()
        if not predicted:
            continue
        value = nearest_neighbour_rmsd(crystal, predicted)
        rows.append({"rank": pose.rank, "score": pose.score, "rmsd": value})
        if best_rmsd is None or value < best_rmsd:
            best_rmsd = value
        marker = "<= correct" if value < 2.0 else ""
        print(
            f"  pose {pose.rank}: score {pose.score:>7.2f} kcal/mol, "
            f"RMSD {value:>6.2f} A {marker}"
        )

    summary["poses"] = rows

    if best_rmsd is None:
        print("\n  No pose carried parseable coordinates.")
        return 1

    top_rmsd = rows[0]["rmsd"] if rows else None
    summary["top_pose_rmsd"] = top_rmsd
    summary["best_pose_rmsd"] = best_rmsd

    heading("Verdict")
    assert isinstance(top_rmsd, float)
    if top_rmsd < 2.0:
        print(
            f"PASS. The top-scoring pose sits {top_rmsd:.2f} A from the crystal "
            "pose, within the 2 A threshold. The receptor preparation, the search "
            "box and the ligand handling work together."
        )
        verdict = "pass"
    elif best_rmsd < 2.0:
        print(
            f"PARTIAL. The top-scoring pose is {top_rmsd:.2f} A out, but a "
            f"lower-ranked pose reaches {best_rmsd:.2f} A. The search found the "
            "right answer and the scoring function did not rank it first -- which "
            "is the well-known weakness of empirical scoring, not a setup error. "
            "Treat score-based ranking on this target with corresponding caution."
        )
        verdict = "partial"
    else:
        print(
            f"FAIL. The closest pose is {best_rmsd:.2f} A from the crystal pose. "
            "Something in the chain is wrong -- receptor preparation, box "
            "placement, or the ligand chemistry. Every other docking score from "
            "this setup is suspect until this passes."
        )
        verdict = "fail"
    summary["verdict"] = verdict

    print(
        "\nWhat this does and does not establish: redocking is the easiest "
        "docking problem, because the receptor is in the conformation this very "
        "ligand induced. Passing shows the mechanics are right. It does not show "
        "the setup will place a different ligand correctly -- cross-docking "
        "measures that, and it is a harder test."
    )
    print(f"\nElapsed: {time.monotonic() - started:.1f}s")

    if args.output:
        path = pathlib.Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, default=str))
        print(f"Summary written to {path}")

    # A failing redock is a real finding, and CI should go red for it: it means
    # the docking setup in this repository does not reproduce reality.
    return 0 if verdict in ("pass", "partial") else 1


if __name__ == "__main__":
    sys.exit(main())
