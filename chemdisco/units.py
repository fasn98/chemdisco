"""Concentration units and the conversion to pActivity.

The predecessor project converted activity values with a single line:

    return -np.log10(value * 1e-9)  # Assuming nM units

The comment is the bug. ChEMBL reports ``standard_units`` as nM, uM, mM, M,
ug.mL-1, mg.kg-1, %, and roughly a hundred other things. Assuming nanomolar
turns a micromolar IC50 into a value three log units too potent and a
percent-inhibition readout into numerical noise -- and because the result still
looks like a plausible pIC50, nothing downstream can detect it.

This module converts only what is unambiguously convertible, and refuses the
rest by returning an unknown :class:`~chemdisco.provenance.Quantity` that names
the reason.

Scientific conventions implemented:

* pActivity = -log10(C / 1 M) for a concentration ``C``. An IC50 of 1 nM is
  pIC50 9.0; 1 uM is 6.0.
* Mass-per-volume units (ug.mL-1) need a molecular weight to become molar. The
  conversion is offered but must be given the weight explicitly; it is never
  guessed.
* Non-concentration readouts (% inhibition, ratios, mg.kg-1 doses) have no
  pActivity. They are refused, not coerced.
* Only the relation ``=`` yields a point value. ``>``/``>=`` on an IC50 is a
  lower bound on concentration and therefore an *upper* bound on pActivity --
  censored data that belongs in classification or survival-style handling, not
  in a regression target.
"""

from __future__ import annotations

import math
from typing import Final

from .provenance import Origin, Quantity

#: Multiplicative factor converting a unit into molar.
#: Keys are normalised by :func:`normalise_unit`.
_TO_MOLAR: Final[dict[str, float]] = {
    "m": 1.0,
    "mm": 1e-3,
    "um": 1e-6,
    "nm": 1e-9,
    "pm": 1e-12,
    "fm": 1e-15,
}

#: Units expressing mass per volume. Convertible to molar only with a
#: molecular weight, so they are kept separate from :data:`_TO_MOLAR`.
#: Values convert the unit into grams per litre.
_TO_GRAMS_PER_LITRE: Final[dict[str, float]] = {
    "g.l-1": 1.0,
    "mg.l-1": 1e-3,
    "ug.l-1": 1e-6,
    "ng.l-1": 1e-9,
    "g.ml-1": 1000.0,
    "mg.ml-1": 1.0,
    "ug.ml-1": 1e-3,
    "ng.ml-1": 1e-6,
}

#: Readouts that are not concentrations at all. Listed explicitly so that an
#: unrecognised unit is reported as unrecognised rather than lumped in here --
#: silence about an unknown unit is how bad data gets in.
_NON_CONCENTRATION: Final[frozenset[str]] = frozenset(
    {
        "%",
        "percent",
        "ratio",
        "mg.kg-1",
        "ug.kg-1",
        "mol.kg-1",
        "ml.min-1.kg-1",
        "h",
        "hr",
        "min",
        "s",
        "l.kg-1",
        "ug",
        "mg",
        "counts",
        "fold",
        "ph",
        "degrees c",
        "kcal/mol",
        "kj/mol",
    }
)

#: Activity types whose value decreases with potency, so a low concentration is
#: a potent compound and the sign of the log is flipped as usual.
POTENCY_TYPES: Final[frozenset[str]] = frozenset(
    {"IC50", "EC50", "XC50", "AC50", "Ki", "Kd", "Potency", "ED50", "GI50", "MIC"}
)

#: Activity types already reported on a logarithmic scale, where no conversion
#: is needed and applying one would double-log the value.
ALREADY_LOG_TYPES: Final[frozenset[str]] = frozenset(
    {"pIC50", "pEC50", "pKi", "pKd", "pA2", "pKB", "Log IC50", "Log Ki"}
)


def normalise_unit(unit: str | None) -> str:
    """Normalise a ChEMBL-style unit string for lookup.

    ChEMBL is inconsistent about case and about the micro sign: ``uM``, ``um``,
    ``µM`` and ``μM`` (U+00B5 and U+03BC are different code points) all occur.
    Collapsing them here means the conversion table is matched reliably instead
    of falling through to "unrecognised" on a formatting difference.
    """
    if unit is None:
        return ""
    text = unit.strip().lower()
    text = text.replace("µ", "u").replace("μ", "u")
    text = text.replace(" ", "")
    # ChEMBL writes both "ug.mL-1" and "ug/mL".
    text = text.replace("/l", ".l-1").replace("/ml", ".ml-1").replace("/kg", ".kg-1")
    return text


def is_concentration_unit(unit: str | None) -> bool:
    """Whether ``unit`` is a molar concentration this module can convert."""
    return normalise_unit(unit) in _TO_MOLAR


def to_molar(value: float, unit: str | None) -> float | None:
    """Convert ``value`` in ``unit`` to molar, or ``None`` if impossible.

    Returns ``None`` rather than raising, because an unconvertible unit is an
    ordinary and expected condition when sweeping a public database -- not a
    programming error.
    """
    factor = _TO_MOLAR.get(normalise_unit(unit))
    if factor is None:
        return None
    return value * factor


def mass_per_litre_to_molar(
    value: float, unit: str | None, molecular_weight: float
) -> float | None:
    """Convert a mass-per-volume concentration to molar using ``molecular_weight``.

    Args:
        value: Numeric concentration.
        unit: A unit present in :data:`_TO_GRAMS_PER_LITRE`.
        molecular_weight: Grams per mole. Must be positive; this function will
            not invent a typical drug mass to make the conversion succeed.

    Returns:
        Molar concentration, or ``None`` if the unit is not mass-per-volume.
    """
    factor = _TO_GRAMS_PER_LITRE.get(normalise_unit(unit))
    if factor is None:
        return None
    if molecular_weight <= 0:
        raise ValueError("molecular_weight must be positive to convert mass to moles")
    return (value * factor) / molecular_weight


def pactivity(
    value: float | None,
    unit: str | None,
    *,
    activity_type: str,
    source: str,
    relation: str = "=",
    molecular_weight: float | None = None,
) -> Quantity:
    """Convert a reported activity into pActivity, or explain the refusal.

    This is the single entry point for building a QSAR regression target. It
    returns a :class:`~chemdisco.provenance.Quantity` in every case: a known
    ``DERIVED`` value when the conversion is sound, and an unknown quantity
    carrying the reason when it is not. Callers filter on ``is_known`` instead
    of receiving a number of unclear meaning.

    Args:
        value: The reported magnitude.
        unit: The reported unit, as the source database spells it.
        activity_type: ``standard_type`` from the source, e.g. ``"IC50"``.
        source: Citation for the measurement, e.g. a ChEMBL activity id.
        relation: ``standard_relation``. Only ``"="`` produces a point value;
            censored relations are refused with the bound recorded in notes.
        molecular_weight: Required only to convert mass-per-volume units.

    Returns:
        A pActivity quantity (dimensionless), or an unknown quantity.
    """
    if value is None:
        return Quantity.unknown(None, f"{source}: no value reported")

    rel = (relation or "=").strip()
    if rel not in ("=", "=="):
        bound = "upper bound on pActivity" if rel in (">", ">=") else "lower bound on pActivity"
        return Quantity.unknown(
            None,
            f"{source}: censored measurement (relation '{rel}'), {bound}; "
            "not usable as a regression target",
        )

    atype = (activity_type or "").strip()

    if atype in ALREADY_LOG_TYPES:
        return Quantity.derived(
            value=float(value),
            unit=None,
            source=f"{source}: {atype} reported on log scale, no conversion applied",
            notes=(f"activity_type={atype}",),
        )

    norm = normalise_unit(unit)

    if norm in _NON_CONCENTRATION:
        return Quantity.unknown(
            None,
            f"{source}: unit '{unit}' is not a concentration; "
            f"{atype or 'this readout'} has no pActivity",
        )

    molar: float | None = to_molar(float(value), unit)

    if molar is None and norm in _TO_GRAMS_PER_LITRE:
        if molecular_weight is None or molecular_weight <= 0:
            return Quantity.unknown(
                None,
                f"{source}: unit '{unit}' needs a molecular weight to become "
                "molar and none was supplied",
            )
        molar = mass_per_litre_to_molar(float(value), unit, molecular_weight)

    if molar is None:
        return Quantity.unknown(
            None, f"{source}: unrecognised unit '{unit}', refusing to guess"
        )

    if molar <= 0:
        return Quantity.unknown(
            None,
            f"{source}: non-positive concentration {value} {unit}; "
            "log is undefined",
        )

    p = -math.log10(molar)

    notes = [f"activity_type={atype or 'unspecified'}", f"from {value} {unit}"]
    # A pActivity outside roughly 2-12 is almost always a unit error in the
    # source record rather than a genuine femtomolar binder or a molar-potency
    # compound. Flag it loudly but keep the value: curation decides, not this
    # conversion function.
    if not 2.0 <= p <= 12.0:
        notes.append(
            f"implausible pActivity {p:.2f}; suspect a unit error in the source record"
        )

    return Quantity(
        value=p,
        unit=None,
        origin=Origin.DERIVED,
        source=f"{source}: -log10({molar:.3e} M)",
        notes=tuple(notes),
    )


def pactivity_is_plausible(quantity: Quantity) -> bool:
    """Whether a pActivity falls in the range real assays can report."""
    if not quantity.is_known:
        return False
    return 2.0 <= float(quantity.value) <= 12.0  # type: ignore[arg-type]
