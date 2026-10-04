"""Does a Vina score depend on how many threads it ran with?

A measurement with two honest outcomes, run on whatever machine is about to be
trusted with the docking.

The question matters because of where the numbers come from. Every score in this
project was produced on a 2-core GitHub runner with ``cpu=0``, which means "Vina,
use what you find". Run the same pipeline on a 16-core workstation and Vina finds
16. If its output depends on the thread count, scores from the two machines are not
comparable -- and the comparison the triage makes is exactly that: a candidate's
ligand efficiency against the median of the reference actives.

Vina's documentation implies the seed is sufficient and the thread count only
changes how the work is divided. User reports disagree. Rather than pick a side,
this docks one ligand repeatedly at several thread counts, seed and exhaustiveness
held fixed, and prints what actually happens.

If the scores are identical: nothing to do, and the uncertainty is retired with a
number instead of an opinion.

If they differ: every run that is meant to be comparable has to pin ``cpu`` to the
same value, and the mixed-machine results already recorded need a caveat. The
magnitude matters too -- a 0.01 kcal/mol spread is noise against Vina's own ~2.5
kcal/mol error, while 0.5 would mean the triage threshold moves with the hardware.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from chemdisco.dock import (  # noqa: E402
    box_from_ligand,
    dock,
    parse_pdb,
    prepare_receptor_pdbqt,
    strip_to_receptor,
    toolchain_report,
    vina_available,
    write_pdb,
)

#: A small, rigid, uncontroversial ligand. The point is reproducibility of the
#: number, not whether the pose is interesting.
DEFAULT_LIGAND = "CC(=O)Oc1ccccc1C(=O)O"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", default="4FRS")
    parser.add_argument("--ligand", default=DEFAULT_LIGAND)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cpus",
        default="1,2,4,0",
        help="Comma-separated thread counts to try; 0 means 'let Vina decide'",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cache", default=".cache")
    args = parser.parse_args()

    print(toolchain_report())
    if not vina_available():
        print("\nVina is not installed; this cannot run.")
        return 2
    print(f"\nThis machine reports {os.cpu_count()} CPU(s).")

    cached = pathlib.Path(args.cache) / "pdb" / f"{args.pdb.upper()}.pdb"
    if not cached.exists():
        print(
            f"\n{cached} not found. Run the discover or validate script once so the "
            "structure is cached, or place the PDB file there by hand."
        )
        return 1

    structure = parse_pdb(cached.read_text())
    ligand = structure.best_ligand()
    if ligand is None:
        print(f"\n{args.pdb} has no ligand to define a box from.")
        return 1
    box = box_from_ligand(ligand)
    print(f"\nBox: {box.describe()}")

    receptor_atoms = strip_to_receptor(structure)
    receptor_pdb = write_pdb(receptor_atoms, title=f"{args.pdb.upper()} receptor")
    prepared, method = prepare_receptor_pdbqt(receptor_pdb)
    if prepared is None:
        print(f"Receptor preparation failed: {method}")
        return 1
    print(f"Receptor: {len(receptor_atoms)} atoms, prepared with {method}")

    cpus = [int(value) for value in args.cpus.split(",") if value.strip()]
    observations: list[dict] = []

    print(
        f"\nDocking {args.ligand} at exhaustiveness {args.exhaustiveness}, "
        f"seed {args.seed}, {args.repeats} repeat(s) per thread count."
    )
    for cpu in cpus:
        for repeat in range(args.repeats):
            result = dock(
                args.ligand,
                prepared,
                box,
                receptor_id=args.pdb.upper(),
                exhaustiveness=args.exhaustiveness,
                seed=args.seed,
                cpu=cpu,
            )
            if not result.ok:
                print(f"  cpu={cpu} repeat={repeat}: FAILED -- {result.error}")
                continue
            score = result.best_score
            print(
                f"  cpu={cpu:<2} repeat={repeat}: {score:+.4f} kcal/mol "
                f"({len(result.poses)} poses)"
            )
            observations.append({"cpu": cpu, "repeat": repeat, "score": score})

    if len(observations) < 2:
        print("\nToo few successful runs to compare.")
        return 1

    scores = [o["score"] for o in observations]
    spread = max(scores) - min(scores)

    # Same thread count twice: if these differ, the seed is not controlling the
    # search at all and the thread count is beside the point.
    by_cpu: dict[int, list[float]] = {}
    for observation in observations:
        by_cpu.setdefault(observation["cpu"], []).append(observation["score"])
    within = max(
        (max(values) - min(values) for values in by_cpu.values() if len(values) > 1),
        default=0.0,
    )

    print("\n=== Verdict ===")
    print(f"  spread across all runs:      {spread:.4f} kcal/mol")
    print(f"  spread at a fixed cpu count: {within:.4f} kcal/mol")

    if spread == 0.0:
        print(
            "\n  REPRODUCIBLE. The score does not depend on the thread count, so "
            "numbers from a 2-core runner and a many-core workstation are "
            "comparable and `cpu` needs no pinning."
        )
    elif within > 0.0:
        print(
            "\n  NOT REPRODUCIBLE EVEN AT A FIXED THREAD COUNT. The seed is not "
            "controlling the search, which is a larger problem than the thread "
            "count: no two runs of this pipeline agree, on any machine. Nothing "
            "should be pooled or compared until that is understood."
        )
    else:
        print(
            "\n  THREAD-DEPENDENT. Each thread count is reproducible with itself "
            "and they disagree with each other, so `cpu` must be pinned to one "
            "value for any set of scores meant to be compared -- and results "
            "already produced on different machines carry this as a caveat.\n"
            f"  Scale: the spread is {spread:.3f} kcal/mol against Vina's own "
            "~2.5 kcal/mol method error. Judge whether that moves the triage "
            "threshold before deciding how much it matters."
        )

    print("\n=== JSON ===")
    print(
        json.dumps(
            {
                "cpu_count_reported": os.cpu_count(),
                "exhaustiveness": args.exhaustiveness,
                "seed": args.seed,
                "observations": observations,
                "spread_kcal": round(spread, 4),
                "spread_within_fixed_cpu_kcal": round(within, 4),
                "thread_dependent": spread > 0.0 and within == 0.0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
