"""
Dual-Threshold Hierarchical Search Engine with Near-Duplicate Disambiguation.

Implements two-stage angular thresholding on the unit hypersphere S^(D-1):
1. Semantic Relevance Boundary:
   theta_query <= theta_rel (default 70 deg, cos_sim >= ~0.342)
   Retrieves all candidate documents aligned with the query topic.

2. Duplicate Identity Boundary:
   theta_inter <= theta_dup (default 30 deg, cos_sim >= ~0.866)
   Clusters near-duplicate documents within the candidate pool,
   electing the highest-ranked document as canonical and nesting
   subordinate duplicates with term-level contribution explanations.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from .engine import DenseCosineIndex, safe_l2_normalize
from .vectorizer import SublinearTfidfVectorizer


@dataclass
class DuplicateMatch:
    """Represents a subordinate duplicate document."""

    doc_id: str
    doc_index: int
    similarity_to_canonical: float
    angle_to_canonical_deg: float
    token_contributions: list[tuple[str, float, float]] = field(default_factory=list)


@dataclass
class HierarchicalSearchResult:
    """Represents a canonical search result with nested duplicates."""

    canonical_id: str
    canonical_index: int
    query_similarity: float
    query_angle_deg: float
    query_angular_distance: float
    duplicates: list[DuplicateMatch] = field(default_factory=list)


class DualThresholdSearchEngine:
    """
    Search engine that disambiguates and groups near-duplicates from search results.
    """

    def __init__(
        self,
        index: DenseCosineIndex,
        vectorizer: SublinearTfidfVectorizer,
        relevance_threshold_deg: float = 70.0,
        duplicate_threshold_deg: float = 30.0,
    ):
        """
        Args:
            index: Pre-populated DenseCosineIndex instance.
            vectorizer: Fitted SublinearTfidfVectorizer instance.
            relevance_threshold_deg: Maximum allowable angle to query
                (default 70 deg, cos_sim >= ~0.342).
            duplicate_threshold_deg: Maximum angle between documents to be
                considered duplicates (default 30 deg, cos_sim >= ~0.866).
        """
        self.index = index
        self.vectorizer = vectorizer
        self.relevance_threshold_rad = math.radians(relevance_threshold_deg)
        self.duplicate_threshold_rad = math.radians(duplicate_threshold_deg)
        self.duplicate_threshold_deg = duplicate_threshold_deg

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        max_candidates: int = 50,
    ) -> list[HierarchicalSearchResult]:
        """
        Executes hierarchical dual-threshold retrieval and deduplication.

        Args:
            query_text: Raw query string.
            top_k: Number of unique canonical results to return.
            max_candidates: Maximum initial candidate documents to pull from index.

        Returns:
            List of HierarchicalSearchResult objects.
        """
        # 1. Transform query text to non-negative vector
        q_raw = self.vectorizer.transform([query_text])
        if np.all(q_raw == 0.0):
            return []

        q_hat = safe_l2_normalize(q_raw, axis=1)[0]

        # 2. Retrieve candidates within semantic relevance boundary
        raw_results = self.index.search(
            query=q_hat,
            top_k=max_candidates,
            angular_threshold=self.relevance_threshold_rad,
        )

        n_candidates = len(raw_results.doc_indices)
        if n_candidates == 0:
            return []

        # 3. Extract normalized candidate embeddings from index matrix
        cand_indices = raw_results.doc_indices
        assert self.index._doc_matrix is not None
        cand_matrix = self.index._doc_matrix[cand_indices]  # Shape: (n_cand, D)

        # 4. Compute pairwise inter-candidate cosine similarities:
        #    S = D_cand @ D_cand^T
        inter_sim_matrix = np.dot(cand_matrix, cand_matrix.T)
        inter_sim_matrix = np.clip(inter_sim_matrix, -1.0, 1.0)
        inter_thetas = np.arccos(inter_sim_matrix)

        # 5. Greedy canonical clustering and duplicate suppression
        canonical_results: list[HierarchicalSearchResult] = []
        assigned_to_canonical = [False] * n_candidates

        for i in range(n_candidates):
            if assigned_to_canonical[i]:
                continue

            # Document i is elected as a canonical search result
            canonical_entry = HierarchicalSearchResult(
                canonical_id=raw_results.doc_ids[i],
                canonical_index=int(raw_results.doc_indices[i]),
                query_similarity=float(raw_results.cosine_similarities[i]),
                query_angle_deg=float(raw_results.angles_degrees[i]),
                query_angular_distance=float(raw_results.angular_distances[i]),
                duplicates=[],
            )
            assigned_to_canonical[i] = True

            # Check remaining lower-ranked candidates for near-duplicate identity
            for j in range(i + 1, n_candidates):
                if assigned_to_canonical[j]:
                    continue

                inter_angle = inter_thetas[i, j]
                if inter_angle <= self.duplicate_threshold_rad:
                    assigned_to_canonical[j] = True
                    inter_sim = float(inter_sim_matrix[i, j])
                    inter_deg = float(math.degrees(inter_angle))

                    # Compute explainable token contributions
                    contributions = self.vectorizer.explain_similarity(
                        cand_matrix[i], cand_matrix[j], top_n=5
                    )

                    canonical_entry.duplicates.append(
                        DuplicateMatch(
                            doc_id=raw_results.doc_ids[j],
                            doc_index=int(raw_results.doc_indices[j]),
                            similarity_to_canonical=inter_sim,
                            angle_to_canonical_deg=inter_deg,
                            token_contributions=contributions,
                        )
                    )

            canonical_results.append(canonical_entry)
            if len(canonical_results) >= top_k:
                break

        return canonical_results
