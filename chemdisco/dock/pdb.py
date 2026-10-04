"""Reading PDB files: atoms, chains, and which heteroatom is the real ligand.

Parsed directly from the fixed-column format rather than through a structure
library. The parsing is forty lines, it carries no dependency, and -- the reason
that matters -- it stays pure, so the logic that decides *which* heteroatom is a
ligand can be tested without a chemistry toolkit or a network.

That decision is the substantive one in this module. A crystal structure's
HETATM records are a mixture of the bound ligand, the water, the ions, and
whatever was in the crystallisation buffer: glycerol, ethylene glycol,
sulfates, PEG fragments, DMSO. Docking into a box centred on a glycerol molecule
is a silent failure that produces perfectly plausible scores for a site that is
not the binding site. So the exclusion list below is long, specific, and
documented, and anything that survives it is sized before being accepted.

The PDB fixed-column layout, for reference:

    COLUMNS  FIELD
     1 -  6  record name ("ATOM  " / "HETATM")
     7 - 11  serial number
    13 - 16  atom name
    18 - 20  residue name
    22       chain identifier
    23 - 26  residue sequence number
    31 - 38  x
    39 - 46  y
    47 - 54  z
    55 - 60  occupancy
    61 - 66  temperature factor
    77 - 78  element symbol

Fields are positional, not whitespace-delimited: an atom name can contain
spaces, and a four-character name starts one column earlier than a shorter one.
Splitting on whitespace works until it meets a real structure.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

#: Residues that are solvent, ions, or crystallisation additives rather than a
#: bound ligand. Docking into a box centred on any of these is a silent failure,
#: so the list is deliberately broad: a missed real ligand is reported as "no
#: ligand found", while a buffer molecule accepted as one produces confident
#: nonsense.
NON_LIGAND_RESIDUES: frozenset[str] = frozenset(
    {
        # Water.
        "HOH", "WAT", "DOD", "H2O",
        # Monatomic ions.
        "NA", "K", "MG", "CA", "ZN", "FE", "FE2", "MN", "CU", "CU1", "NI", "CO",
        "CD", "HG", "CS", "RB", "SR", "BA", "LI", "AL", "CL", "BR", "IOD", "F",
        # Common buffer and cryoprotectant molecules.
        "GOL",  # glycerol -- the single most common false ligand
        "EDO",  # ethylene glycol
        "PEG", "PG4", "PGE", "1PE", "2PE", "P6G", "XPE", "7PE",  # PEG fragments
        "MPD",  # 2-methyl-2,4-pentanediol
        "DMS",  # dimethyl sulfoxide
        "SO4", "PO4", "NO3", "CO3", "ACT", "ACY",  # ions and acetate
        "FMT",  # formate
        "TRS",  # Tris buffer
        "MES", "EPE", "BTB", "CIT", "TAR", "MLA", "MLI",  # buffers
        "IMD", "BME", "DTT",  # imidazole, reducing agents
        "NH4", "AZI", "CYN", "SCN",
        "URE",  # urea
        "ETH", "IPA", "MOH", "EOH",  # small alcohols
        "SIN", "OXL",
        # Sugars and lipids from expression or purification, not binders.
        "NAG", "NDG", "BMA", "MAN", "FUC", "GAL", "GLC", "BGC", "XYS",
        "PLM", "MYR", "OLA", "STE",
    }
)

#: Minimum heavy atoms for a heteroatom group to be plausibly a drug-like
#: ligand. Below about eight, a fragment is as likely to be an unlisted additive
#: as a binder, and a box centred on it would be too small to be useful anyway.
MIN_LIGAND_HEAVY_ATOMS = 8


@dataclass(frozen=True, slots=True)
class Atom:
    """One atom from an ATOM or HETATM record."""

    serial: int
    name: str
    residue_name: str
    chain: str
    residue_seq: int
    x: float
    y: float
    z: float
    element: str
    is_hetatm: bool
    occupancy: float = 1.0
    b_factor: float = 0.0
    alt_loc: str = ""

    @property
    def is_hydrogen(self) -> bool:
        return self.element.upper() == "H"

    @property
    def coordinates(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True, slots=True)
class Residue:
    """A group of atoms sharing a residue identity -- here, a candidate ligand."""

    name: str
    chain: str
    sequence: int
    atoms: tuple[Atom, ...]

    @property
    def key(self) -> str:
        return f"{self.name}_{self.chain}_{self.sequence}"

    @property
    def heavy_atoms(self) -> tuple[Atom, ...]:
        return tuple(atom for atom in self.atoms if not atom.is_hydrogen)

    @property
    def n_heavy_atoms(self) -> int:
        return len(self.heavy_atoms)

    @property
    def elements(self) -> set[str]:
        return {atom.element.upper() for atom in self.heavy_atoms}

    @property
    def looks_drug_like(self) -> bool:
        """Whether this heteroatom group could plausibly be a bound ligand.

        Deliberately crude: enough heavy atoms to be a real molecule, and
        containing carbon, which rules out the inorganic additives that slip
        past a name-based exclusion list.
        """
        return self.n_heavy_atoms >= MIN_LIGAND_HEAVY_ATOMS and "C" in self.elements


@dataclass(slots=True)
class Structure:
    """A parsed PDB structure.

    Attributes:
        pdb_id: Four-character code, when the file declared one.
        atoms: Every atom record kept.
        title: The TITLE record, useful for sanity-checking what was fetched.
        resolution: Crystallographic resolution in angstroms, when present.
            ``None`` for NMR and predicted models, which is itself informative:
            a structure with no resolution is not an X-ray structure.
        method: Experimental method from the EXPDTA record.
        warnings: Anything a caller should know before docking into this.
    """

    pdb_id: str = ""
    atoms: list[Atom] = field(default_factory=list)
    title: str = ""
    resolution: float | None = None
    method: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def protein_atoms(self) -> list[Atom]:
        return [atom for atom in self.atoms if not atom.is_hetatm]

    @property
    def hetatms(self) -> list[Atom]:
        return [atom for atom in self.atoms if atom.is_hetatm]

    @property
    def chains(self) -> set[str]:
        return {atom.chain for atom in self.protein_atoms}

    def heteroatom_residues(self) -> list[Residue]:
        """Every distinct heteroatom group, solvent and additives included."""
        grouped: dict[tuple[str, str, int], list[Atom]] = defaultdict(list)
        for atom in self.hetatms:
            grouped[(atom.residue_name, atom.chain, atom.residue_seq)].append(atom)
        return [
            Residue(name=name, chain=chain, sequence=sequence, atoms=tuple(atoms))
            for (name, chain, sequence), atoms in grouped.items()
        ]

    def candidate_ligands(self) -> list[Residue]:
        """Heteroatom groups that could be the bound ligand, largest first.

        Two filters: the residue name is not a known solvent, ion or additive,
        and the group is large enough and organic enough to be a real molecule.
        Sorted by size because when a structure holds several genuine ligands the
        largest is almost always the one of interest.
        """
        candidates = [
            residue
            for residue in self.heteroatom_residues()
            if residue.name.strip().upper() not in NON_LIGAND_RESIDUES
            and residue.looks_drug_like
        ]
        return sorted(candidates, key=lambda residue: -residue.n_heavy_atoms)

    def best_ligand(self) -> Residue | None:
        """The most likely bound ligand, or ``None`` if the structure has none.

        ``None`` is a real and common answer -- apo structures exist -- and it
        must not be papered over by falling back to the largest heteroatom group
        regardless of what it is.
        """
        candidates = self.candidate_ligands()
        return candidates[0] if candidates else None

    def describe(self) -> str:
        lines = [
            f"{self.pdb_id or 'structure'}: {self.title[:70]}",
            f"  method={self.method or 'unknown'}, "
            f"resolution={f'{self.resolution:.2f} A' if self.resolution else 'n/a'}",
            f"  {len(self.protein_atoms)} protein atoms over "
            f"{len(self.chains)} chain(s)",
        ]
        candidates = self.candidate_ligands()
        if candidates:
            lines.append("  candidate ligands:")
            for residue in candidates[:5]:
                lines.append(
                    f"    {residue.key}: {residue.n_heavy_atoms} heavy atoms"
                )
        else:
            lines.append(
                "  no candidate ligand -- this looks like an apo structure, so a "
                "binding box must be defined some other way"
            )
        lines.extend(f"  WARNING: {warning}" for warning in self.warnings)
        return "\n".join(lines)


def _parse_float(text: str) -> float | None:
    try:
        return float(text.strip())
    except (ValueError, AttributeError):
        return None


def parse_pdb(text: str, *, keep_hydrogens: bool = False) -> Structure:
    """Parse PDB text into a :class:`Structure`.

    Args:
        text: Contents of a PDB file.
        keep_hydrogens: Keep hydrogen atoms. Off by default: crystal structures
            rarely resolve them, docking engines add their own, and keeping a
            partial set is worse than keeping none.

    Returns:
        A structure. Malformed coordinate lines are skipped and counted in
        ``warnings`` rather than raising -- real PDB files contain surprises, and
        one bad line should not lose the other nine thousand.

    Notes:
        Only the first model of a multi-model file is read. NMR ensembles hold
        twenty-odd models of the same molecule; docking into all of them at once
        would be meaningless, and silently merging them would put every atom in
        the structure at twenty slightly different places.
    """
    structure = Structure()
    malformed = 0
    alt_locs_seen: set[str] = set()
    in_later_model = False

    for line in text.splitlines():
        record = line[:6]

        if record == "MODEL ":
            serial = line[10:14].strip()
            if serial not in ("", "1"):
                in_later_model = True
            continue
        if record == "ENDMDL":
            if in_later_model:
                break
            in_later_model = False
            continue
        if in_later_model:
            continue

        if record == "HEADER":
            structure.pdb_id = line[62:66].strip()
            continue
        if record == "TITLE ":
            structure.title = (structure.title + " " + line[10:80].strip()).strip()
            continue
        if record == "EXPDTA":
            structure.method = (structure.method + " " + line[10:79].strip()).strip()
            continue
        if line.startswith("REMARK   2 RESOLUTION."):
            value = _parse_float(line[23:30])
            if value is not None:
                structure.resolution = value
            continue

        if record not in ("ATOM  ", "HETATM"):
            continue

        x = _parse_float(line[30:38])
        y = _parse_float(line[38:46])
        z = _parse_float(line[46:54])
        if x is None or y is None or z is None:
            malformed += 1
            continue

        element = line[76:78].strip()
        if not element:
            # Older files omit the element column. The atom name's first two
            # columns encode it, with the element right-justified in them.
            element = line[12:14].strip().lstrip("0123456789")[:1]

        alt_loc = line[16:17].strip()
        if alt_loc:
            alt_locs_seen.add(alt_loc)
            # Keep only the first conformer. Keeping several would place the same
            # atom at two positions and inflate any geometry computed from it.
            if alt_loc not in ("A", "1"):
                continue

        if not keep_hydrogens and element.upper() == "H":
            continue

        structure.atoms.append(
            Atom(
                serial=int(_parse_float(line[6:11]) or 0),
                name=line[12:16].strip(),
                residue_name=line[17:20].strip(),
                chain=line[21:22].strip() or "A",
                residue_seq=int(_parse_float(line[22:26]) or 0),
                x=x,
                y=y,
                z=z,
                element=element,
                is_hetatm=(record == "HETATM"),
                occupancy=_parse_float(line[54:60]) or 1.0,
                b_factor=_parse_float(line[60:66]) or 0.0,
                alt_loc=alt_loc,
            )
        )

    if malformed:
        structure.warnings.append(
            f"{malformed} coordinate line(s) could not be parsed and were skipped"
        )
    if alt_locs_seen:
        structure.warnings.append(
            f"alternate conformations present ({', '.join(sorted(alt_locs_seen))}); "
            "only the first was kept"
        )
    if not structure.protein_atoms:
        structure.warnings.append(
            "no protein atoms found -- this is not a receptor structure"
        )
    if structure.resolution is None and "X-RAY" in structure.method.upper():
        structure.warnings.append(
            "X-ray structure with no resolution record; quality cannot be judged"
        )
    if structure.resolution is not None and structure.resolution > 3.0:
        structure.warnings.append(
            f"resolution {structure.resolution:.2f} A is low. Side-chain positions "
            "at this resolution are poorly determined, and docking depends on "
            "exactly those."
        )
    return structure


def write_pdb(atoms: list[Atom], *, title: str = "") -> str:
    """Serialise atoms back to PDB text, for a stripped receptor or a ligand.

    Writes the same fixed columns it reads. Element symbols are right-justified
    in columns 77-78, which several downstream tools require and which a naive
    writer gets wrong.
    """
    lines: list[str] = []
    if title:
        lines.append(f"TITLE     {title[:70]}")
    for index, atom in enumerate(atoms, start=1):
        record = "HETATM" if atom.is_hetatm else "ATOM  "
        # A four-character atom name starts in column 13; shorter names start in
        # column 14, which is how PDB distinguishes CA the calcium ion from CA
        # the alpha carbon.
        name = atom.name
        formatted_name = f"{name:<4}" if len(name) >= 4 else f" {name:<3}"
        lines.append(
            f"{record}{index:>5} {formatted_name}{' '}"
            f"{atom.residue_name:>3} {atom.chain:>1}{atom.residue_seq:>4}    "
            f"{atom.x:>8.3f}{atom.y:>8.3f}{atom.z:>8.3f}"
            f"{atom.occupancy:>6.2f}{atom.b_factor:>6.2f}"
            f"{'':>10}{atom.element:>2}"
        )
    lines.append("END")
    return "\n".join(lines) + "\n"


def strip_to_receptor(
    structure: Structure, *, keep_chains: set[str] | None = None
) -> list[Atom]:
    """Protein atoms only: solvent, ions, additives and the ligand removed.

    Removing the co-crystallised ligand is not optional. Docking into a site that
    still contains its original occupant scores the new molecule against a
    pocket that has no room for it, and the result looks like a weak binder
    rather than a broken setup.
    """
    atoms = structure.protein_atoms
    if keep_chains:
        atoms = [atom for atom in atoms if atom.chain in keep_chains]
    return atoms
