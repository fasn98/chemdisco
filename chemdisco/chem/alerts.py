"""Structural alerts and synthetic accessibility: the filters a generator needs.

A fragment-recombination generator will happily propose structures that no chemist
would make and no assay could interpret. Three classes of problem, each with its
own filter:

**PAINS** (pan-assay interference compounds). Substructures that produce apparent
activity across unrelated targets through assay artefacts -- redox cycling, protein
aggregation, covalent reaction, fluorescence. A PAINS hit in a screening deck
wastes follow-up work. The filters here are Baell and Holloway's catalogues A, B
and C as shipped with RDKit.

A caveat that matters for honesty: the PAINS filters are substructure patterns
derived from one set of AlphaScreen campaigns, and they produce false positives.
Several marketed drugs contain PAINS substructures. So a hit is a flag for
scrutiny, not a verdict, and this module reports which pattern matched rather than
returning a bare pass or fail.

**Brenk alerts.** Substructures associated with toxicity, metabolic instability or
poor pharmacokinetics -- nitro groups, azides, aliphatic halides. More permissive
than PAINS and aimed at developability rather than assay artefacts.

**Synthetic accessibility.** The Ertl-Schuffenhauer SAscore, 1 (easy) to 10 (hard),
estimated from fragment contributions calibrated against PubChem plus a complexity
penalty. The predecessor project replaced this with a count of rings and
stereocentres, which does not distinguish a trivially accessible polycyclic natural
product from an intractable one. SAscore is the real implementation, shipped in
RDKit's Contrib directory.

SAscore is a heuristic and is returned as ``Origin.HEURISTIC`` accordingly. It
correlates with chemist judgement at roughly 0.7-0.8, which is useful for
triaging thousands of candidates and insufficient for deciding any single one.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from ..provenance import Quantity
from .standardize import RDKIT_AVAILABLE, require_rdkit

if RDKIT_AVAILABLE:  # pragma: no branch
    from rdkit import Chem, RDConfig
    from rdkit.Chem import FilterCatalog
    from rdkit.Chem.FilterCatalog import FilterCatalogParams

_SASCORER = None
_SASCORER_ERROR: str | None = None


def _load_sascorer():
    """Import RDKit's SAscore implementation from its Contrib directory.

    It is not importable as a normal module -- RDKit ships it as a script outside
    the package namespace -- so the path has to be added explicitly. Failure is
    recorded rather than raised, because SAscore is useful but not essential and a
    missing Contrib directory should not stop a pipeline.
    """
    global _SASCORER, _SASCORER_ERROR
    if _SASCORER is not None or _SASCORER_ERROR is not None:
        return _SASCORER

    require_rdkit()
    try:
        contrib_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if contrib_path not in sys.path:
            sys.path.append(contrib_path)
        import sascorer  # type: ignore[import-not-found]

        _SASCORER = sascorer
    except Exception as error:
        _SASCORER_ERROR = (
            f"RDKit's SA_Score contrib module could not be loaded ({error}). "
            "Synthetic accessibility will be reported as unavailable rather than "
            "estimated by a substitute, which would not be comparable to "
            "published SAscore values."
        )
    return _SASCORER


@dataclass(frozen=True, slots=True)
class AlertHit:
    """One structural alert matched by a molecule."""

    catalogue: str
    name: str
    description: str = ""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.catalogue}:{self.name}"


@dataclass(frozen=True, slots=True)
class AlertReport:
    """Every alert a molecule triggered, kept in full.

    Deliberately not reduced to a boolean. Which pattern matched determines
    whether it matters: a catechol flagged as PAINS in a kinase programme is a
    genuine concern, while the same flag on a natural-product scaffold being
    pursued deliberately is not. Collapsing that to pass/fail discards the
    information needed to make the call.
    """

    smiles: str
    pains: tuple[AlertHit, ...] = ()
    brenk: tuple[AlertHit, ...] = ()
    nih: tuple[AlertHit, ...] = ()
    error: str | None = None

    @property
    def n_alerts(self) -> int:
        return len(self.pains) + len(self.brenk) + len(self.nih)

    @property
    def clean(self) -> bool:
        return self.error is None and self.n_alerts == 0

    def describe(self) -> str:
        if self.error:
            return f"alert screening failed: {self.error}"
        if self.clean:
            return "no structural alerts"
        parts: list[str] = []
        if self.pains:
            parts.append(
                "PAINS: " + ", ".join(hit.name for hit in self.pains)
                + " (substructure patterns from AlphaScreen campaigns; they do "
                "produce false positives, so treat this as a reason to scrutinise "
                "rather than a disqualification)"
            )
        if self.brenk:
            parts.append("Brenk: " + ", ".join(hit.name for hit in self.brenk))
        if self.nih:
            parts.append("NIH: " + ", ".join(hit.name for hit in self.nih))
        return "; ".join(parts)


@lru_cache(maxsize=1)
def _catalogues() -> dict[str, "FilterCatalog.FilterCatalog"]:
    """Build the filter catalogues once; construction is expensive."""
    require_rdkit()
    built: dict[str, FilterCatalog.FilterCatalog] = {}

    pains_params = FilterCatalogParams()
    for catalogue in (
        FilterCatalogParams.FilterCatalogs.PAINS_A,
        FilterCatalogParams.FilterCatalogs.PAINS_B,
        FilterCatalogParams.FilterCatalogs.PAINS_C,
    ):
        pains_params.AddCatalog(catalogue)
    built["PAINS"] = FilterCatalog.FilterCatalog(pains_params)

    brenk_params = FilterCatalogParams()
    brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    built["Brenk"] = FilterCatalog.FilterCatalog(brenk_params)

    nih_params = FilterCatalogParams()
    nih_params.AddCatalog(FilterCatalogParams.FilterCatalogs.NIH)
    built["NIH"] = FilterCatalog.FilterCatalog(nih_params)

    return built


def screen_alerts(smiles: str) -> AlertReport:
    """Screen one structure against PAINS, Brenk and NIH alert sets."""
    require_rdkit()
    text = (smiles or "").strip()
    mol = Chem.MolFromSmiles(text) if text else None
    if mol is None:
        return AlertReport(smiles=text, error="structure failed to parse")

    try:
        catalogues = _catalogues()
        collected: dict[str, list[AlertHit]] = {"PAINS": [], "Brenk": [], "NIH": []}
        for name, catalogue in catalogues.items():
            for match in catalogue.GetMatches(mol):
                collected[name].append(
                    AlertHit(
                        catalogue=name,
                        name=match.GetDescription(),
                    )
                )
        return AlertReport(
            smiles=text,
            pains=tuple(collected["PAINS"]),
            brenk=tuple(collected["Brenk"]),
            nih=tuple(collected["NIH"]),
        )
    except Exception as error:
        return AlertReport(
            smiles=text, error=f"{type(error).__name__}: {error}"
        )


def synthetic_accessibility(smiles: str) -> Quantity:
    """Ertl-Schuffenhauer SAscore, 1 (easy) to 10 (hard), as a heuristic quantity.

    Returned with ``Origin.HEURISTIC`` because that is what it is: a
    fragment-frequency estimate calibrated against PubChem, not a measurement and
    not a model fitted to synthesis outcomes. It agrees with chemist judgement at
    around r=0.7-0.8, which supports triaging a large candidate list and does not
    support a decision about any individual molecule.
    """
    sascorer = _load_sascorer()
    text = (smiles or "").strip()
    if sascorer is None:
        return Quantity.unknown(None, _SASCORER_ERROR or "SAscore unavailable")

    mol = Chem.MolFromSmiles(text) if text else None
    if mol is None:
        return Quantity.unknown(None, "structure failed to parse")

    try:
        score = float(sascorer.calculateScore(mol))
    except Exception as error:
        return Quantity.unknown(None, f"SAscore calculation failed: {error}")

    notes = ["Ertl-Schuffenhauer SAscore, 1 (easy) to 10 (hard)"]
    if score > 6.0:
        notes.append(
            "above 6 suggests a demanding synthesis; worth a chemist's opinion "
            "before committing to this candidate"
        )
    return Quantity.heuristic(
        score, None, "RDKit Contrib sascorer (Ertl & Schuffenhauer 2009)", notes=notes
    )


def lipinski_violations(smiles: str) -> Quantity:
    """Count of Lipinski rule-of-five violations, as a heuristic.

    Four criteria: molecular weight above 500, cLogP above 5, hydrogen-bond donors
    above 5, acceptors above 10.

    Worth being clear about what this is for. Lipinski's analysis described the
    properties of compounds that had *survived* to late-stage clinical trials in
    the 1990s; it was never a predictor of activity and it is routinely violated
    by successful drugs, particularly kinase inhibitors and anything targeting
    the central nervous system via active transport. Use it to notice that a
    candidate is unusual, not to reject it.
    """
    require_rdkit()
    from .descriptors import compute_descriptors

    result = compute_descriptors(smiles)
    if not result.ok or result.values is None:
        return Quantity.unknown(None, result.error or "descriptors unavailable")

    values = result.values
    violations = 0
    breached: list[str] = []
    if values["molecular_weight"] > 500:
        violations += 1
        breached.append(f"MW {values['molecular_weight']:.0f} > 500")
    if values["logp"] > 5:
        violations += 1
        breached.append(f"cLogP {values['logp']:.1f} > 5")
    if values["hbd"] > 5:
        violations += 1
        breached.append(f"HBD {values['hbd']:.0f} > 5")
    if values["hba"] > 10:
        violations += 1
        breached.append(f"HBA {values['hba']:.0f} > 10")

    notes = ["Lipinski rule of five"]
    if breached:
        notes.append("; ".join(breached))
    notes.append(
        "descriptive of 1990s late-stage clinical compounds, not predictive of "
        "activity; many approved drugs violate it"
    )
    return Quantity.heuristic(
        float(violations), None, "Lipinski rule of five", notes=notes
    )


def screen_many(smiles_list: Sequence[str]) -> list[AlertReport]:
    """Screen a list of structures."""
    return [screen_alerts(smiles) for smiles in smiles_list]
