"""The Vina docking engine, and an honest account of what its scores mean.

**What a docking score is.** AutoDock Vina's scoring function is an empirical sum
of steric, hydrophobic, hydrogen-bonding and torsional terms, fitted to
reproduce measured binding affinities across a training set of protein-ligand
complexes. It reports kcal/mol and it is a real calculation on real geometry --
unlike the predecessor project's "interaction energy", which was
``-5.0 - molecular_weight/100 + random.uniform(-2, 2)``.

**What a docking score is not.** It is not a predicted binding affinity, and the
gap matters more than anything else in this module:

* Vina's own reported error against measured affinities is around 2-3 kcal/mol.
  Three kcal/mol is a factor of roughly 150 in Kd. Two compounds scoring -9.5 and
  -8.0 cannot be ordered by this number.
* Correlation between docking score and measured affinity, across diverse
  ligands on one target, is typically 0.3-0.5 Pearson. That is enough to enrich a
  virtual screen above random selection and nowhere near enough to rank a
  shortlist.
* The function has well-known systematic biases. It rewards molecular size
  almost linearly, so large ligands score better regardless of fit --
  :func:`ligand_efficiency` exists to see past that. It handles metal
  coordination, buried waters and charged interactions poorly.
* It scores a single rigid-receptor pose. Binding is an ensemble process with
  entropy and solvation terms this function only approximates.

So docking answers "could this molecule physically occupy this pocket, and in
what orientation" far better than it answers "how tightly does it bind". This
module reports scores as ``PREDICTED`` quantities carrying those caveats, and
:meth:`DockingResult.is_rankable` refuses to let a score alone order candidates
when the differences between them fall inside the method's error.

**Validation.** The one check that tells you the setup is right is redocking: take
the ligand out of the crystal structure, dock it back, and measure the RMSD
against where it actually sits. Below 2 Å the receptor preparation, the box and
the engine are working together. Above it, something is wrong, and every other
score from that setup is suspect. :func:`redock_validation` performs it, and the
pipeline is meant to run it before trusting any result on a new target.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..provenance import Origin, Quantity
from .box import Box

#: Vina's approximate standard error against measured affinities, kcal/mol.
#: Two scores closer together than this are not distinguishable.
VINA_ERROR_KCAL = 2.5

#: Exhaustiveness. Vina's default is 8; 16 roughly doubles the runtime and
#: measurably improves pose reproducibility, which matters more here than speed
#: because an irreproducible pose makes the score meaningless.
DEFAULT_EXHAUSTIVENESS = 16

#: Poses kept per ligand. More than one because the top-scoring pose is not
#: reliably the correct one, and the spread across poses is itself informative.
DEFAULT_N_POSES = 9


class DockingError(RuntimeError):
    """Raised when a docking run cannot be completed."""


def vina_available() -> bool:
    """Whether the Vina Python bindings import."""
    try:
        import vina  # noqa: F401

        return True
    except Exception:
        return False


def obabel_available() -> bool:
    """Whether the OpenBabel command line is on PATH."""
    return shutil.which("obabel") is not None


def meeko_available() -> bool:
    """Whether Meeko and its undeclared dependencies are importable.

    Meeko does not declare scipy or gemmi in its install requirements, and both
    are hard requirements. A probe of a clean runner failed on each in turn, so
    this imports rather than checking a version.
    """
    try:
        import meeko  # noqa: F401

        return True
    except Exception:
        return False


def toolchain_report() -> str:
    """What is installed, and what each absence costs."""
    lines = []
    for label, present, consequence in (
        ("vina", vina_available(), "no docking at all"),
        (
            "meeko",
            meeko_available(),
            "ligand preparation falls back to OpenBabel, which protonates by "
            "simple rules at a fixed pH rather than per-microspecies",
        ),
        ("obabel", obabel_available(), "no receptor preparation"),
    ):
        mark = "present" if present else "MISSING"
        lines.append(f"  {label}: {mark}" + ("" if present else f" -- {consequence}"))
    return "Docking toolchain:\n" + "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Pose:
    """One predicted binding pose."""

    rank: int
    score: float
    rmsd_lower_bound: float = 0.0
    rmsd_upper_bound: float = 0.0
    pdbqt: str = ""

    def coordinates(self) -> list[tuple[float, float, float]]:
        """Heavy-atom coordinates parsed from this pose's PDBQT block."""
        points: list[tuple[float, float, float]] = []
        for line in self.pdbqt.splitlines():
            if not line.startswith(("ATOM", "HETATM")):
                continue
            element = line[76:78].strip().upper()
            if element == "H" or element.startswith("H"):
                continue
            try:
                points.append(
                    (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                )
            except ValueError:
                continue
        return points


@dataclass(slots=True)
class DockingResult:
    """The outcome of docking one ligand into one receptor.

    Attributes:
        smiles: The ligand docked.
        poses: Predicted poses, best score first.
        box: Where the search ran, carrying how that position was justified.
        receptor_id: Which structure was used.
        n_heavy_atoms: Ligand size, needed for ligand efficiency.
        error: Why the run failed, if it did.
        warnings: Caveats a reader needs before using the score.
    """

    smiles: str
    poses: list[Pose] = field(default_factory=list)
    box: Box | None = None
    receptor_id: str = ""
    n_heavy_atoms: int = 0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.poses)

    @property
    def pose_ranking_is_determined(self) -> bool:
        """Whether the score actually separates the top pose from the rest.

        Measured, not assumed. Redocking the 4FRS inhibitor produced nine poses
        spanning 1.23 kcal/mol -- entirely inside Vina's own 2.5 kcal/mol error --
        with the crystallographically correct pose ranked third. Quadrupling
        exhaustiveness from 16 to 64 found more correct poses but still ranked a
        4.2 A pose first, which settles the question: the limitation is the
        scoring function, not the search.

        When this is False, the top pose is the top pose by a margin the method
        cannot resolve, and treating it as the predicted binding mode asserts a
        precision the calculation does not have.
        """
        if len(self.poses) < 2:
            return False
        spread = self.score_spread
        return spread is not None and spread > VINA_ERROR_KCAL

    @property
    def best_score(self) -> float | None:
        return self.poses[0].score if self.poses else None

    @property
    def score_spread(self) -> float | None:
        """Range across the retained poses.

        A wide spread means the search found several very different
        arrangements and preferred one only slightly -- so the top pose is not
        well determined, whatever its score says.
        """
        if len(self.poses) < 2:
            return None
        scores = [pose.score for pose in self.poses]
        return max(scores) - min(scores)

    def score_quantity(self) -> Quantity:
        """The docking score as a provenance-carrying quantity.

        ``PREDICTED``, with the method's standard error attached as its
        uncertainty, so a reader cannot see -9.2 without also seeing +/- 2.5.
        """
        if not self.ok or self.best_score is None:
            return Quantity.unknown(
                "kcal/mol", self.error or "docking produced no pose"
            )

        notes = [
            "AutoDock Vina empirical scoring function",
            f"box: {self.box.derived_from if self.box else 'unspecified'}",
            "this is a docking score, not a predicted binding affinity: Vina's "
            f"error against measured affinities is about {VINA_ERROR_KCAL} kcal/mol, "
            "which is a factor of ~150 in Kd",
            "the function rewards molecular size, so compare ligand efficiency "
            "rather than raw score across molecules of different sizes",
        ]
        if self.score_spread is not None and self.score_spread < VINA_ERROR_KCAL:
            notes.append(
                f"all {len(self.poses)} poses fall within {self.score_spread:.2f} "
                f"kcal/mol, inside the method's own {VINA_ERROR_KCAL} kcal/mol "
                "error. The pose ranking is therefore not determined by the "
                "score: the top pose is not meaningfully better than the others."
            )
        notes.extend(self.warnings)

        return Quantity(
            value=self.best_score,
            unit="kcal/mol",
            origin=Origin.PREDICTED,
            source=f"AutoDock Vina against {self.receptor_id or 'receptor'}",
            uncertainty=VINA_ERROR_KCAL,
            # Vina has no applicability domain in the QSAR sense. Marking it
            # False would imply extrapolation; marking it True would claim a
            # check that was never made. None is the honest state, and it keeps
            # the score out of any ranked list by itself.
            in_domain=None,
            notes=notes,
        )

    def ligand_efficiency(self) -> Quantity:
        """Score per heavy atom, which corrects for the size bias.

        Vina's score grows close to linearly with ligand size, so a large
        molecule outscores a small one that fits better. Dividing by heavy-atom
        count is the standard correction and is the number to compare when
        candidates differ in size. A fragment at -6 kcal/mol over 15 heavy atoms
        is a far better starting point than a 50-atom molecule at -10.
        """
        if not self.ok or self.best_score is None or self.n_heavy_atoms <= 0:
            return Quantity.unknown(
                "kcal/mol/atom", "no score or no heavy-atom count"
            )
        return Quantity.predicted(
            self.best_score / self.n_heavy_atoms,
            "kcal/mol/atom",
            "Vina score divided by heavy-atom count",
            notes=[
                "ligand efficiency corrects for Vina's near-linear reward for "
                "molecular size; compare this rather than raw score across "
                "molecules of different sizes",
            ],
        )

    def describe(self) -> str:
        if not self.ok:
            return f"docking failed: {self.error}"
        lines = [
            f"{self.smiles[:60]} against {self.receptor_id}",
            f"  best score: {self.score_quantity().label(digits=2)}",
            f"  ligand efficiency: {self.ligand_efficiency().label(digits=3)}",
            f"  {len(self.poses)} pose(s) retained"
            + (
                f", spanning {self.score_spread:.2f} kcal/mol"
                if self.score_spread is not None
                else ""
            ),
        ]
        if self.ok and not self.pose_ranking_is_determined and len(self.poses) > 1:
            lines.append(
                "  the poses are not separated by more than the method's error, "
                "so which one ranks first is not determined by the score"
            )
        lines.extend(f"  WARNING: {warning}" for warning in self.warnings)
        return "\n".join(lines)


def prepare_ligand_pdbqt(smiles: str, *, seed: int = 0xF00D) -> tuple[str | None, str]:
    """Turn a SMILES into a docking-ready PDBQT ligand.

    Generates a 3D conformer with RDKit's ETKDG, optimises it with MMFF, then
    converts to PDBQT with Meeko, falling back to OpenBabel.

    The conformer matters less than it appears to: Vina searches torsions itself,
    so the starting conformation only needs to be chemically sensible, not
    bioactive. Ring conformations and stereochemistry, which Vina does **not**
    search, are fixed here and do matter.

    Returns:
        ``(pdbqt, method)``, with ``pdbqt`` ``None`` on failure and ``method``
        naming what was used or what went wrong.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return None, "RDKit not installed"

    mol = Chem.MolFromSmiles((smiles or "").strip())
    if mol is None:
        return None, "SMILES failed to parse"

    mol = Chem.AddHs(mol)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    if AllChem.EmbedMolecule(mol, parameters) != 0:
        # Retry with random coordinates: ETKDG fails on strained macrocycles and
        # highly constrained ring systems, where the fallback usually succeeds.
        parameters.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, parameters) != 0:
            return None, "3D embedding failed even with random coordinates"
    # Optimisation failure is tolerable: the embedded geometry is already
    # chemically reasonable, and Vina searches torsions itself, so the starting
    # conformation only has to be sensible rather than optimal.
    with contextlib.suppress(Exception):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

    if meeko_available():
        try:
            from meeko import MoleculePreparation, PDBQTWriterLegacy

            preparator = MoleculePreparation()
            setups = preparator.prepare(mol)
            if setups:
                pdbqt, success, error = PDBQTWriterLegacy.write_string(setups[0])
                if success:
                    return pdbqt, "meeko"
                return None, f"meeko write failed: {error}"
        except Exception as error:
            # Fall through to OpenBabel rather than failing: Meeko's API has
            # changed shape between releases.
            last = f"meeko raised {type(error).__name__}: {error}"
        else:
            last = "meeko produced no setup"
    else:
        last = "meeko unavailable"

    if not obabel_available():
        return None, f"{last}; OpenBabel not installed either"

    with tempfile.TemporaryDirectory() as workdir:
        sdf_path = os.path.join(workdir, "ligand.sdf")
        pdbqt_path = os.path.join(workdir, "ligand.pdbqt")
        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()
        completed = subprocess.run(
            ["obabel", sdf_path, "-O", pdbqt_path],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if completed.returncode != 0 or not os.path.exists(pdbqt_path):
            return None, f"{last}; obabel failed: {completed.stderr[:160]}"
        with open(pdbqt_path) as handle:
            return handle.read(), "openbabel (fallback)"


def prepare_receptor_pdbqt(pdb_text: str) -> tuple[str | None, str]:
    """Convert receptor PDB text to PDBQT.

    OpenBabel's ``-xr`` rigid-receptor conversion. Meeko's receptor path needs
    gemmi and chemical templates and warns its way through unknown residues; a
    probe found OpenBabel the route that works unattended.

    The caller is responsible for having stripped solvent and the ligand first --
    see :func:`chemdisco.dock.pdb.strip_to_receptor`. Converting a structure with
    its ligand still in place produces a receptor whose pocket is already
    occupied, and every subsequent score is of a molecule being pushed into a
    full site.
    """
    if not obabel_available():
        return None, "OpenBabel not installed"

    with tempfile.TemporaryDirectory() as workdir:
        pdb_path = os.path.join(workdir, "receptor.pdb")
        pdbqt_path = os.path.join(workdir, "receptor.pdbqt")
        with open(pdb_path, "w") as handle:
            handle.write(pdb_text)
        completed = subprocess.run(
            ["obabel", pdb_path, "-xr", "-O", pdbqt_path],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if completed.returncode != 0 or not os.path.exists(pdbqt_path):
            return None, f"obabel failed: {completed.stderr[:200]}"
        with open(pdbqt_path) as handle:
            text = handle.read()
    if "ATOM" not in text:
        return None, "conversion produced no atom records"
    return text, "openbabel"


def dock(
    smiles: str,
    receptor_pdbqt: str,
    box: Box,
    *,
    receptor_id: str = "",
    exhaustiveness: int = DEFAULT_EXHAUSTIVENESS,
    n_poses: int = DEFAULT_N_POSES,
    seed: int = 42,
    cpu: int = 0,
) -> DockingResult:
    """Dock one ligand into a prepared receptor.

    Args:
        smiles: Ligand to dock.
        receptor_pdbqt: Prepared receptor, from :func:`prepare_receptor_pdbqt`.
        box: Search box, from :mod:`chemdisco.dock.box`.
        receptor_id: Structure identifier, recorded in the result's provenance.
        exhaustiveness: Vina's search effort. Higher is more reproducible.
        n_poses: Poses to keep.
        seed: Vina's random seed. Fixed so a reported score can be reproduced;
            Vina's search is stochastic and an unseeded run gives a different
            answer each time.
        cpu: Threads; 0 lets Vina decide.

    Returns:
        A :class:`DockingResult`. Failures are returned rather than raised: a
        virtual screen will contain ligands that cannot be prepared, and that is
        ordinary rather than exceptional.
    """
    result = DockingResult(smiles=smiles, box=box, receptor_id=receptor_id)

    if not vina_available():
        result.error = (
            "the vina package is not installed. Install it with 'pip install "
            "vina'; note it also needs scipy and gemmi if Meeko is used for "
            "ligand preparation."
        )
        return result

    ligand_pdbqt, method = prepare_ligand_pdbqt(smiles)
    if ligand_pdbqt is None:
        result.error = f"ligand preparation failed: {method}"
        return result
    if method.startswith("openbabel"):
        result.warnings.append(
            "ligand prepared with OpenBabel rather than Meeko: protonation was "
            "assigned by simple rules at a fixed pH rather than per-microspecies, "
            "which matters for basic amines and acids"
        )

    try:
        from rdkit import Chem

        parsed = Chem.MolFromSmiles(smiles)
        result.n_heavy_atoms = parsed.GetNumHeavyAtoms() if parsed else 0
    except Exception:
        result.n_heavy_atoms = 0

    if box.is_blind:
        result.warnings.append(
            "the search box is large enough to constitute blind docking; the "
            "scoring function was not calibrated for that volume"
        )
    result.warnings.extend(box.warnings)

    try:
        from vina import Vina

        with tempfile.TemporaryDirectory() as workdir:
            receptor_path = os.path.join(workdir, "receptor.pdbqt")
            ligand_path = os.path.join(workdir, "ligand.pdbqt")
            with open(receptor_path, "w") as handle:
                handle.write(receptor_pdbqt)
            with open(ligand_path, "w") as handle:
                handle.write(ligand_pdbqt)

            engine = Vina(sf_name="vina", cpu=cpu, seed=seed, verbosity=0)
            engine.set_receptor(receptor_path)
            engine.set_ligand_from_file(ligand_path)
            engine.compute_vina_maps(
                center=list(box.center), box_size=list(box.size)
            )
            engine.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)

            energies = engine.energies(n_poses=n_poses)
            pose_text = engine.poses(n_poses=n_poses)
            blocks = _split_pose_blocks(pose_text)

            for index, row in enumerate(energies):
                result.poses.append(
                    Pose(
                        rank=index + 1,
                        score=float(row[0]),
                        rmsd_lower_bound=float(row[1]) if len(row) > 1 else 0.0,
                        rmsd_upper_bound=float(row[2]) if len(row) > 2 else 0.0,
                        pdbqt=blocks[index] if index < len(blocks) else "",
                    )
                )
    except Exception as error:
        result.error = f"{type(error).__name__}: {error}"
        return result

    if not result.poses:
        result.error = "Vina returned no poses"
    return result


def _split_pose_blocks(pose_text: str) -> list[str]:
    """Split Vina's multi-pose PDBQT output into one block per model."""
    blocks: list[str] = []
    current: list[str] = []
    for line in pose_text.splitlines():
        if line.startswith("MODEL"):
            current = []
            continue
        if line.startswith("ENDMDL"):
            blocks.append("\n".join(current))
            current = []
            continue
        current.append(line)
    if current and not blocks:
        blocks.append("\n".join(current))
    return blocks


@dataclass(frozen=True, slots=True)
class RedockValidation:
    """Whether a docking setup reproduces a known crystal pose.

    The only check that tells you a setup is right. Everything else -- scores,
    rankings, enrichment -- is downstream of the receptor, the box and the
    preparation being correct together, and redocking is what tests all three at
    once.
    """

    ligand_name: str
    rmsd: float | None
    score: float | None
    passed: bool
    detail: str = ""

    def describe(self) -> str:
        if self.rmsd is None:
            return f"redocking could not be completed: {self.detail}"
        verdict = "PASS" if self.passed else "FAIL"
        text = (
            f"[{verdict}] redocked {self.ligand_name} to "
            f"{self.rmsd:.2f} A RMSD from the crystal pose"
        )
        if self.score is not None:
            text += f", scoring {self.score:.2f} kcal/mol"
        if not self.passed:
            text += (
                "\n  Above the 2 A threshold. The receptor preparation, the box "
                "or the ligand handling is wrong, and every other score from this "
                "setup is suspect. Fix this before docking anything else."
            )
        return text


def redock_validation(
    receptor_pdbqt: str,
    box: Box,
    crystal_ligand_smiles: str,
    crystal_coordinates: Sequence[Sequence[float]],
    *,
    ligand_name: str = "ligand",
    **dock_kwargs: Any,
) -> RedockValidation:
    """Dock a structure's own ligand back and measure the displacement.

    Args:
        receptor_pdbqt: Receptor with the ligand already removed.
        box: Search box, normally derived from the crystal ligand itself.
        crystal_ligand_smiles: The ligand's structure.
        crystal_coordinates: Its observed heavy-atom positions.
        ligand_name: For the report.

    Returns:
        A :class:`RedockValidation`. Below 2 Å passes, the conventional
        threshold -- which comes from the resolution at which crystallography
        determines atom positions, not from any property of the docking program.

    Note:
        RMSD here is computed between unordered point sets by nearest-neighbour
        matching, because a redocked pose's atom order need not match the crystal
        file's. That makes it a lower bound on the true symmetry-corrected RMSD:
        good enough to tell 1 Å from 6 Å, which is the decision being made, and
        not a substitute for a proper symmetry-aware calculation when reporting a
        benchmark.
    """
    result = dock(
        crystal_ligand_smiles, receptor_pdbqt, box, **dock_kwargs
    )
    if not result.ok:
        return RedockValidation(
            ligand_name=ligand_name,
            rmsd=None,
            score=None,
            passed=False,
            detail=result.error or "no poses produced",
        )

    predicted = result.poses[0].coordinates()
    if not predicted:
        return RedockValidation(
            ligand_name=ligand_name,
            rmsd=None,
            score=result.best_score,
            passed=False,
            detail="the top pose carried no parseable coordinates",
        )

    value = nearest_neighbour_rmsd(crystal_coordinates, predicted)
    return RedockValidation(
        ligand_name=ligand_name,
        rmsd=value,
        score=result.best_score,
        passed=value < 2.0,
        detail=f"{len(predicted)} predicted atoms against "
        f"{len(crystal_coordinates)} crystal atoms",
    )


def nearest_neighbour_rmsd(
    reference: Sequence[Sequence[float]], predicted: Sequence[Sequence[float]]
) -> float:
    """RMSD between unordered point sets, matching each reference to its nearest.

    A lower bound on the true symmetry-corrected RMSD, since several reference
    atoms may match the same predicted atom. It is used here because a redocked
    pose's atom order does not match the crystal file's, and the decision it
    supports -- did the pose land in roughly the right place -- is robust to that
    looseness. Do not quote it as a benchmark figure.
    """
    if not reference or not predicted:
        raise ValueError("both point sets must be non-empty")
    total = 0.0
    for point in reference:
        best = min(
            (point[0] - other[0]) ** 2
            + (point[1] - other[1]) ** 2
            + (point[2] - other[2]) ** 2
            for other in predicted
        )
        total += best
    return (total / len(reference)) ** 0.5


def scores_are_distinguishable(a: float, b: float) -> bool:
    """Whether two docking scores differ by more than the method's error.

    Called before presenting any ranked list. Vina's error against measured
    affinities is around 2.5 kcal/mol, so -9.5 and -8.0 are the same number as
    far as this method can tell, and a list ordered by that difference conveys a
    precision the calculation does not have.
    """
    return abs(a - b) > VINA_ERROR_KCAL
