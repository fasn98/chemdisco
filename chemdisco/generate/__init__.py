"""Candidate generation. Honest about what fragment recombination can reach."""

from .brics import (
    Candidate,
    GenerationPolicy,
    GenerationReport,
    decompose_to_fragments,
    generate_candidates,
    score_candidates,
)

__all__ = [
    "Candidate",
    "GenerationPolicy",
    "GenerationReport",
    "decompose_to_fragments",
    "generate_candidates",
    "score_candidates",
]
