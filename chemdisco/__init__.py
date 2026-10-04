"""chemdisco: an auditable computational drug-discovery pipeline.

Design premise: in computational drug discovery the expensive mistake is not a
crash, it is a believable number with nothing behind it. A heuristic score
rendered to two decimal places is indistinguishable from a measured binding
constant unless the code makes the difference structural.

So the central type is :class:`chemdisco.provenance.Quantity`, which cannot hold
a value without declaring where it came from, and the layering keeps the
scientific logic -- curation, unit conversion, splitting, evaluation -- as pure
functions that are testable without a chemistry toolkit or a network.
"""

from .provenance import Origin, ProvenanceError, Quantity

__all__ = ["Quantity", "Origin", "ProvenanceError", "__version__"]
__version__ = "0.1.0"
