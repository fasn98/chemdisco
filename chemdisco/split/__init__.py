"""Dataset splitting that measures generalisation rather than memorisation."""

from .scaffold import (
    Split,
    SplitError,
    coverage,
    group_by_scaffold,
    iter_scaffold_splits,
    random_split,
    scaffold_split,
    verify_disjoint,
)

__all__ = [
    "Split",
    "SplitError",
    "coverage",
    "group_by_scaffold",
    "iter_scaffold_splits",
    "random_split",
    "scaffold_split",
    "verify_disjoint",
]
