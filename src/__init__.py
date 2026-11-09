"""
Spherical-IR: Production Directional Semantic Retrieval & Spherical Vector Indexing Engine.
"""

from .engine import (
    DenseCosineIndex,
    SearchResult,
    compute_angular_distance,
    compute_cosine_similarity,
    safe_l2_normalize,
    verify_euclidean_isomorphism,
)

__all__ = [
    "DenseCosineIndex",
    "SearchResult",
    "compute_angular_distance",
    "compute_cosine_similarity",
    "safe_l2_normalize",
    "verify_euclidean_isomorphism",
]
