#!/usr/bin/env python3
"""Probe which parts of a docking toolchain actually install and run.

Written before the docking module, not after, because the AutoDock toolchain has
several independent ways to fail on a clean machine and the documentation does
not say which ones apply to a given environment. Guessing produces a thousand
lines of code built on an import that was never going to work.

What is being checked, in the order the pipeline needs it:

1. **The Vina engine.** The ``vina`` package ships compiled bindings; whether a
   wheel exists for the runner's Python version decides the whole approach.
2. **Ligand preparation.** Vina consumes PDBQT, not SMILES. Meeko is the current
   RDKit-based route; the older alternative is an OpenBabel conversion, which
   handles protonation differently and is worth knowing about as a fallback.
3. **Receptor preparation.** The historically fragile step. ``prepare_receptor``
   came from the ADFR suite, which is not on PyPI; Meeko has since grown its own
   receptor path with its own dependencies.
4. **A real docking run.** All three working separately proves nothing. This
   docks a small ligand into a real binding site and reports the score.

Output goes to stdout and is replayed as a CI annotation, since job logs and
artifacts are not retrievable from every environment.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import traceback
from typing import Any

RESULTS: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    RESULTS.append((name, status, detail))
    marker = {"ok": "OK  ", "fail": "FAIL", "warn": "WARN"}.get(status, "??  ")
    print(f"[{marker}] {name}" + (f" -- {detail}" if detail else ""))


def try_import(module: str, label: str | None = None) -> Any | None:
    label = label or module
    try:
        imported = importlib.import_module(module)
    except Exception as error:
        record(label, "fail", f"{type(error).__name__}: {error}")
        return None
    version = getattr(imported, "__version__", "unknown version")
    record(label, "ok", str(version))
    return imported


def try_command(command: list[str], label: str) -> bool:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=120
        )
    except FileNotFoundError:
        record(label, "fail", "command not found")
        return False
    except Exception as error:
        record(label, "fail", f"{type(error).__name__}: {error}")
        return False
    output = (completed.stdout + completed.stderr).strip().splitlines()
    first = output[0][:160] if output else ""
    if completed.returncode == 0:
        record(label, "ok", first)
        return True
    record(label, "warn", f"exit {completed.returncode}: {first}")
    return False


def probe_imports() -> dict[str, Any]:
    print("\n--- imports ---")
    return {
        "rdkit": try_import("rdkit"),
        "vina": try_import("vina"),
        "meeko": try_import("meeko"),
        "openbabel": try_import("openbabel", "openbabel (python bindings)"),
        "prody": try_import("prody"),
        "scrubber": try_import("scrubber", "scrubber (protonation)"),
    }


def probe_commands() -> None:
    print("\n--- command-line tools ---")
    try_command(["obabel", "-V"], "obabel")
    try_command(["vina", "--version"], "vina binary")
    try_command(["mk_prepare_ligand.py", "--help"], "mk_prepare_ligand.py")
    try_command(["mk_prepare_receptor.py", "--help"], "mk_prepare_receptor.py")


def probe_ligand_preparation(modules: dict[str, Any]) -> str | None:
    """Turn a SMILES into PDBQT, which is what Vina actually consumes."""
    print("\n--- ligand preparation ---")
    if modules.get("rdkit") is None:
        record("ligand prep", "fail", "RDKit missing; nothing else can proceed")
        return None

    from rdkit import Chem
    from rdkit.Chem import AllChem

    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin: small, flexible enough to be real
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    status = AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    if status != 0:
        record("3D embedding", "fail", f"EmbedMolecule returned {status}")
        return None
    AllChem.MMFFOptimizeMolecule(mol)
    record("3D embedding + MMFF optimisation", "ok", f"{mol.GetNumAtoms()} atoms")

    if modules.get("meeko") is None:
        record("meeko ligand prep", "fail", "meeko not importable")
        return None

    try:
        from meeko import MoleculePreparation

        preparator = MoleculePreparation()
        # Meeko's API changed between 0.5 and 0.6: prepare() used to mutate the
        # preparator and is now expected to return setups. Both are tried,
        # because which one a runner installs is not knowable in advance.
        setups = preparator.prepare(mol)
        if setups is None:
            pdbqt = preparator.write_pdbqt_string()
        else:
            from meeko import PDBQTWriterLegacy

            pdbqt, success, error = PDBQTWriterLegacy.write_string(setups[0])
            if not success:
                record("meeko ligand prep", "fail", str(error)[:160])
                return None
        record("meeko ligand prep", "ok", f"{len(pdbqt.splitlines())} PDBQT lines")
        return pdbqt
    except Exception:
        record("meeko ligand prep", "fail", traceback.format_exc(limit=2)[-200:])
        return None


def probe_receptor_preparation() -> str | None:
    """Fetch a real structure and try to turn it into a PDBQT receptor.

    1FKN is a BACE1 structure with a co-crystallised inhibitor -- the target this
    pipeline is validated against, and small enough to fetch quickly.
    """
    print("\n--- receptor preparation ---")
    try:
        import requests

        response = requests.get(
            "https://files.rcsb.org/download/1FKN.pdb", timeout=60
        )
        if response.status_code != 200:
            record("RCSB fetch", "fail", f"HTTP {response.status_code}")
            return None
        pdb_text = response.text
        record("RCSB fetch", "ok", f"{len(pdb_text.splitlines())} lines of 1FKN")
    except Exception as error:
        record("RCSB fetch", "fail", f"{type(error).__name__}: {error}")
        return None

    with open("receptor.pdb", "w") as handle:
        handle.write(pdb_text)

    # Count what is actually in the file, which decides how preparation must go.
    atoms = sum(1 for line in pdb_text.splitlines() if line.startswith("ATOM"))
    hetatms = [
        line[17:20].strip()
        for line in pdb_text.splitlines()
        if line.startswith("HETATM")
    ]
    ligands = sorted({name for name in hetatms if name not in ("HOH", "WAT")})
    record(
        "structure contents",
        "ok",
        f"{atoms} protein atoms, HETATM residues: {ligands[:8]}",
    )

    # Route 1: Meeko's receptor preparation.
    try:
        completed = subprocess.run(
            [
                "mk_prepare_receptor.py",
                "--read_pdb",
                "receptor.pdb",
                "-o",
                "receptor_meeko",
                "-p",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        detail = (completed.stdout + completed.stderr).strip().splitlines()
        record(
            "mk_prepare_receptor.py run",
            "ok" if completed.returncode == 0 else "warn",
            f"exit {completed.returncode}: {detail[-1][:160] if detail else ''}",
        )
    except FileNotFoundError:
        record("mk_prepare_receptor.py run", "fail", "not installed")
    except Exception as error:
        record("mk_prepare_receptor.py run", "fail", f"{type(error).__name__}: {error}")

    # Route 2: OpenBabel, the older and more forgiving conversion.
    if try_command(
        ["obabel", "receptor.pdb", "-xr", "-O", "receptor_ob.pdbqt"],
        "obabel receptor -> pdbqt",
    ):
        try:
            with open("receptor_ob.pdbqt") as handle:
                text = handle.read()
            record(
                "obabel receptor output",
                "ok" if "ATOM" in text else "warn",
                f"{len(text.splitlines())} lines",
            )
            return "receptor_ob.pdbqt"
        except Exception as error:
            record("obabel receptor output", "fail", str(error))
    return None


def probe_docking_run(receptor_pdbqt: str | None, ligand_pdbqt: str | None) -> None:
    """The step that matters: does a docking actually produce a score?"""
    print("\n--- docking run ---")
    if receptor_pdbqt is None or ligand_pdbqt is None:
        record(
            "vina docking",
            "fail",
            "skipped: receptor or ligand preparation did not produce PDBQT",
        )
        return

    try:
        from vina import Vina
    except Exception as error:
        record("vina docking", "fail", f"import failed: {error}")
        return

    try:
        with open("ligand.pdbqt", "w") as handle:
            handle.write(ligand_pdbqt)

        engine = Vina(sf_name="vina", cpu=2, seed=42, verbosity=0)
        engine.set_receptor(receptor_pdbqt)
        engine.set_ligand_from_file("ligand.pdbqt")
        # A box over the 1FKN inhibitor site. Centre taken from the
        # co-crystallised ligand's rough position; exactness does not matter for
        # a toolchain probe.
        engine.compute_vina_maps(center=[18.0, 25.0, 15.0], box_size=[24, 24, 24])
        engine.dock(exhaustiveness=4, n_poses=3)
        energies = engine.energies(n_poses=3)
        best = float(energies[0][0])
        record(
            "vina docking",
            "ok",
            f"best score {best:.2f} kcal/mol over {len(energies)} pose(s)",
        )
    except Exception:
        record("vina docking", "fail", traceback.format_exc(limit=3)[-300:])


def main() -> int:
    print("Probing the docking toolchain\n" + "=" * 60)
    print(f"Python {sys.version.split()[0]}")

    modules = probe_imports()
    probe_commands()
    ligand_pdbqt = probe_ligand_preparation(modules)
    receptor_pdbqt = probe_receptor_preparation()
    probe_docking_run(receptor_pdbqt, ligand_pdbqt)

    print("\n" + "=" * 60)
    ok = sum(1 for _, status, _ in RESULTS if status == "ok")
    print(f"{ok} of {len(RESULTS)} checks passed.")
    print("\nVERDICT:")
    names = {name: status for name, status, _ in RESULTS}
    if names.get("vina docking") == "ok":
        print("  A full docking run works. Build the module on this toolchain.")
    else:
        failures = [n for n, s, _ in RESULTS if s == "fail"]
        print(
            "  No end-to-end docking. Blocking failures: "
            + (", ".join(failures) if failures else "none recorded")
        )
        print(
            "  Report this rather than writing a module around an engine that "
            "does not run here."
        )
    # Always 0: a probe reporting total failure is a successful diagnostic.
    return 0


if __name__ == "__main__":
    sys.exit(main())
