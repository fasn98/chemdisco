"""Fingerprint similarity, and the novelty check that depends on it.

Two jobs here, both feeding decisions made elsewhere:

**Similarity to the training set**, which the applicability domain in
:mod:`chemdisco.qsar.applicability` needs to judge whether a prediction is
interpolation or extrapolation.

**Novelty against known chemistry**, which decides whether a generated candidate
is actually new. A generator that rediscovers marketed drugs is working correctly
but producing nothing of interest, and without this check its output looks like
success.

Tanimoto on ECFP4 is the standard measure. Useful reference points for reading the
numbers: above 0.85 is usually the same chemical series with a minor change; 0.6-0.85
is a recognisable analogue; 0.4-0.6 shares some features; below 0.3 means the two
molecules have little in common. These are conventions rather than physical
constants, and they shift with fingerprint type and bit count -- a Tanimoto of 0.5
on ECFP4 at 2048 bits is not the same as 0.5 on a MACCS key.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .descriptors import DEFAULT_N_BITS, DEFAULT_RADIUS, fingerprint
from .standardize import RDKIT_AVAILABLE, inchikey_of, require_rdkit

if RDKIT_AVAILABLE:  # pragma: no branch
    from rdkit import DataStructs

#: Above this, two structures are near-identical analogues rather than distinct
#: candidates. Used to collapse a generated library onto its distinct members.
NEAR_DUPLICATE_TANIMOTO = 0.85

#: Below this, a candidate has no meaningful neighbour in the reference set, so a
#: QSAR prediction about it is extrapolation.
NOVELTY_TANIMOTO = 0.4


def tanimoto(smiles_a: str, smiles_b: str) -> float | None:
    """Tanimoto similarity between two structures, or ``None`` if either fails."""
    require_rdkit()
    fp_a = fingerprint(smiles_a)
    fp_b = fingerprint(smiles_b)
    if fp_a is None or fp_b is None:
        return None
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def similarity_matrix(
    query_smiles: Sequence[str],
    reference_smiles: Sequence[str],
    *,
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_N_BITS,
) -> np.ndarray:
    """Pairwise Tanimoto between two sets of structures.

    Returns:
        Array of shape ``(n_query, n_reference)``. Rows for unparseable queries
        are filled with ``-1.0`` rather than ``0.0``, so a parse failure is
        distinguishable from genuine dissimilarity -- zero similarity is a real
        and meaningful value.
    """
    require_rdkit()
    query_fps = [fingerprint(s, radius=radius, n_bits=n_bits) for s in query_smiles]
    reference_fps = [
        fingerprint(s, radius=radius, n_bits=n_bits) for s in reference_smiles
    ]
    usable_reference = [fp for fp in reference_fps if fp is not None]

    matrix = np.full((len(query_smiles), len(reference_smiles)), -1.0, dtype=float)
    if not usable_reference:
        return matrix

    reference_positions = [i for i, fp in enumerate(reference_fps) if fp is not None]

    for row, query_fp in enumerate(query_fps):
        if query_fp is None:
            continue
        similarities = DataStructs.BulkTanimotoSimilarity(query_fp, usable_reference)
        for position, value in zip(reference_positions, similarities, strict=True):
            matrix[row, position] = float(value)
    return matrix


def max_similarity_to_reference(
    query_smiles: Sequence[str],
    reference_smiles: Sequence[str],
    *,
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_N_BITS,
) -> tuple[np.ndarray, list[int | None]]:
    """For each query, its similarity to the single nearest reference structure.

    This is the array to hand to
    :meth:`chemdisco.qsar.applicability.ApplicabilityDomain.assess` as
    ``max_similarity``.

    Returns:
        ``(similarities, nearest_indices)``. A query that failed to parse gets
        ``-1.0`` and ``None``.
    """
    matrix = similarity_matrix(
        query_smiles, reference_smiles, radius=radius, n_bits=n_bits
    )
    similarities = np.full(matrix.shape[0], -1.0, dtype=float)
    nearest: list[int | None] = []

    for row in range(matrix.shape[0]):
        valid = matrix[row] >= 0.0
        if not np.any(valid):
            nearest.append(None)
            continue
        candidate_indices = np.where(valid)[0]
        best = candidate_indices[int(np.argmax(matrix[row, candidate_indices]))]
        similarities[row] = matrix[row, best]
        nearest.append(int(best))

    return similarities, nearest


@dataclass(frozen=True, slots=True)
class NoveltyVerdict:
    """Whether a candidate is genuinely new relative to a reference set.

    Attributes:
        smiles: The candidate.
        is_exact_match: ``True`` when the candidate's InChIKey appears in the
            reference set -- it is a known compound, not a new one.
        max_similarity: Tanimoto to the nearest reference structure.
        nearest_reference: The reference structure it most resembles.
        is_novel: Combined verdict: not an exact match, and below the
            near-duplicate similarity threshold.
        notes: Interpretation for a reader.
    """

    smiles: str
    is_exact_match: bool
    max_similarity: float
    nearest_reference: str | None
    is_novel: bool
    notes: tuple[str, ...] = ()

    def describe(self) -> str:
        if self.is_exact_match:
            return (
                "already known: this exact structure is in the reference set, so "
                "it is a rediscovery rather than a proposal"
            )
        if self.max_similarity < 0:
            return "novelty could not be assessed: the structure failed to parse"
        text = f"nearest known structure at Tanimoto {self.max_similarity:.3f}"
        if self.max_similarity >= NEAR_DUPLICATE_TANIMOTO:
            text += " -- a minor variation on a known compound, not a new chemotype"
        elif self.max_similarity < NOVELTY_TANIMOTO:
            text += (
                " -- genuinely distinct from the reference set, which also means "
                "any activity prediction about it is extrapolation"
            )
        else:
            text += " -- a recognisable analogue of known chemistry"
        return text


def assess_novelty(
    candidate_smiles: Sequence[str],
    known_smiles: Sequence[str],
    *,
    near_duplicate_threshold: float = NEAR_DUPLICATE_TANIMOTO,
) -> list[NoveltyVerdict]:
    """Judge each candidate's novelty against a set of known structures.

    Exact-match detection uses standardised InChIKeys, so a candidate matching a
    known compound's salt form or a differently written tautomer is still caught
    as known. String comparison of SMILES would miss all of those.

    Args:
        candidate_smiles: Generated or proposed structures.
        known_smiles: Reference set -- the training compounds, a ChEMBL slice, a
            vendor catalogue. The verdict is only as good as this set's coverage,
            and a candidate called novel against a narrow reference may be an
            ordinary known compound.
        near_duplicate_threshold: Similarity at or above which a candidate counts
            as a minor variation rather than a new structure.
    """
    require_rdkit()
    known_keys = {
        key for key in (inchikey_of(s) for s in known_smiles) if key is not None
    }
    similarities, nearest = max_similarity_to_reference(candidate_smiles, known_smiles)

    verdicts: list[NoveltyVerdict] = []
    for index, smiles in enumerate(candidate_smiles):
        key = inchikey_of(smiles)
        exact = key is not None and key in known_keys
        similarity = float(similarities[index])
        nearest_index = nearest[index]

        notes: list[str] = []
        if not known_keys:
            notes.append(
                "the reference set produced no usable structures, so this verdict "
                "is not meaningful"
            )
        if key is None:
            notes.append(
                "no InChIKey could be computed, so exact-match detection fell back "
                "to similarity alone and may miss a known compound"
            )

        verdicts.append(
            NoveltyVerdict(
                smiles=smiles,
                is_exact_match=exact,
                max_similarity=similarity,
                nearest_reference=(
                    known_smiles[nearest_index] if nearest_index is not None else None
                ),
                is_novel=(
                    not exact and 0.0 <= similarity < near_duplicate_threshold
                ),
                notes=tuple(notes),
            )
        )
    return verdicts


def diverse_subset(
    smiles_list: Sequence[str], *, threshold: float = NEAR_DUPLICATE_TANIMOTO
) -> list[int]:
    """Greedily select structures no two of which exceed ``threshold`` similarity.

    A fragment-recombination generator produces large families of near-identical
    structures, and a ranked list of fifty variations on one molecule is not fifty
    candidates. This reduces a library to its distinct members.

    Greedy sphere exclusion in input order: deterministic, and O(n^2) in
    fingerprint comparisons, which is acceptable into the low tens of thousands.

    Returns:
        Indices of the selected structures.
    """
    require_rdkit()
    fingerprints = [fingerprint(s) for s in smiles_list]
    selected: list[int] = []
    selected_fps: list[object] = []

    for index, fp in enumerate(fingerprints):
        if fp is None:
            continue
        if selected_fps:
            similarities = DataStructs.BulkTanimotoSimilarity(fp, selected_fps)
            if max(similarities) >= threshold:
                continue
        selected.append(index)
        selected_fps.append(fp)

    return selected
