"""
Core Mathematical Vector Engine & Angular Metric Implementation.

Implements directional vector geometry on the unit hypersphere S^(n-1)
as derived in 'Angles and Cosine Similarity.md':
- Magnitude invariance factoring: cos_sim(c*u, v) = cos_sim(u, v)
- Safe L2 unit normalization: u_hat = u / (||u||_2 + eps)
- Pre-normalized inner product equivalence: cos_sim(u, v) = u_hat^T v_hat
- Cauchy-Schwarz numerical boundary clamping: clip(s, -1.0, 1.0)
- Angular metric extraction: theta = arccos(s), D_ang = theta / pi
- Isomorphism to Euclidean distance: ||u_hat - v_hat||_2^2 = 2 * (1 - cos_sim)
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SearchResult:
    """Container for top-k directional retrieval results."""

    doc_indices: np.ndarray  # Shape: (k,) integer indices
    doc_ids: list[str]  # List of document string IDs
    cosine_similarities: np.ndarray  # Shape: (k,) in range [-1.0, 1.0]
    angular_distances: np.ndarray  # Shape: (k,) normalized in range [0.0, 1.0]
    angles_degrees: np.ndarray  # Shape: (k,) geometric angle theta in degrees


def safe_l2_normalize(
    v: np.ndarray,
    axis: int = -1,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Projects vectors onto the unit hypersphere S^(n-1) with division-by-zero safeguard.

    Mathematical Definition:
        u_hat = u / max(||u||_2, eps)

    If ||u||_2 < eps (e.g. zero vector), the zero vector is preserved without
    generating NaN or Inf values.

    Args:
        v: Input vector or matrix (float32 or float64).
        axis: Dimension along which to calculate L2 norm (-1 for row vectors).
        eps: Small non-zero floor to prevent division-by-zero.

    Returns:
        L2-normalized array of the same shape with unit length (or zero if norm < eps).
    """
    v = np.asarray(v, dtype=np.float32)
    norms = np.linalg.norm(v, ord=2, axis=axis, keepdims=True)
    # Mask zero or sub-epsilon norms to prevent division by zero
    zero_mask = norms < eps
    safe_norms = np.where(zero_mask, 1.0, norms)
    v_norm = v / safe_norms
    # Zero out vectors whose initial norm was below epsilon
    v_norm = np.where(zero_mask, 0.0, v_norm)
    return np.ascontiguousarray(v_norm, dtype=np.float32)


def compute_cosine_similarity(
    u: np.ndarray,
    v: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Computes exact cosine similarity between two raw vectors (with magnitudes intact).

    Formula (Equation 44):
        cos_sim(u, v) = (u^T v) / (||u||_2 * ||v||_2 + eps)
    """
    u = np.asarray(u, dtype=np.float32).ravel()
    v = np.asarray(v, dtype=np.float32).ravel()
    dot = float(np.dot(u, v))
    norm_u = float(np.linalg.norm(u, ord=2))
    norm_v = float(np.linalg.norm(v, ord=2))
    denom = norm_u * norm_v
    if denom < eps:
        return 0.0
    raw_sim = dot / denom
    # Clamp to [-1.0, 1.0] per Cauchy-Schwarz bound
    return float(np.clip(raw_sim, -1.0, 1.0))


def compute_angular_distance(
    cosine_sim: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Transforms cosine similarity into geometric angle theta and true metric angular distance.

    Formulas:
        theta = arccos(clip(s, -1.0, 1.0))  # in [0, pi]
        D_ang = theta / pi                   # in [0, 1]

    Boundary Safeguard:
        Floating-point imprecision can produce values such as 1.0000000000000002.
        Passing unclipped values to arccos returns NaN. Strict clipping to [-1.0, 1.0]
        preserves Cauchy-Schwarz validity.
    """
    clipped = np.clip(cosine_sim, -1.0, 1.0)
    theta = np.arccos(clipped)
    angular_dist = theta / np.pi
    return theta, angular_dist


def verify_euclidean_isomorphism(
    u_hat: np.ndarray,
    v_hat: np.ndarray,
) -> float:
    """
    Verifies Section 8 Derivation 2 from the notes:
        ||u_hat - v_hat||_2^2 = 2 * (1 - cos_sim(u, v))

    Returns:
        Absolute discrepancy between squared Euclidean distance and 2*(1 - dot).
    """
    u_hat = np.asarray(u_hat, dtype=np.float32).ravel()
    v_hat = np.asarray(v_hat, dtype=np.float32).ravel()
    sq_euclidean = float(np.sum((u_hat - v_hat) ** 2))
    cos_sim = float(np.dot(u_hat, v_hat))
    expected_sq_euclidean = 2.0 * (1.0 - cos_sim)
    return abs(sq_euclidean - expected_sq_euclidean)


class DenseCosineIndex:
    """
    Pre-Normalized Dense Matrix Retrieval Engine.

    Encapsulates Level-2 BLAS matrix-vector evaluation on pre-normalized
    vectors on the unit hypersphere S^(n-1).

    At ingest time:
        D_hat = D / ||D||_2
        Complexity: O(N * D) offline precalculation.

    At query time:
        q_hat = q / ||q||_2
        s = D_hat @ q_hat
        Complexity: O(N * D) via single-threaded Level-2 BLAS GEMV.
        Top-k: O(N + k log k) via np.argpartition partial heap sort.
    """

    def __init__(self, dim: int, eps: float = 1e-12):
        self.dim = dim
        self.eps = eps
        self._doc_matrix: np.ndarray | None = None  # Contiguous float32 (N, D)
        self._doc_ids: list[str] = []
        self._num_docs: int = 0

    @property
    def num_docs(self) -> int:
        return self._num_docs

    def add_documents(
        self,
        docs: np.ndarray,
        doc_ids: list[str] | None = None,
    ) -> None:
        """
        Normalizes and indexes a batch of document vectors.

        Args:
            docs: Array of shape (N, D) representing raw document representations.
            doc_ids: Optional list of unique string IDs for each document.
        """
        docs = np.asarray(docs, dtype=np.float32)
        if docs.ndim == 1:
            docs = docs.reshape(1, -1)
        if docs.shape[1] != self.dim:
            raise ValueError(
                f"Dimension mismatch: expected D={self.dim}, got D={docs.shape[1]}"
            )

        n_new = docs.shape[0]
        if doc_ids is None:
            generated_ids = [f"doc_{self._num_docs + i}" for i in range(n_new)]
        else:
            if len(doc_ids) != n_new:
                raise ValueError(
                    f"doc_ids length ({len(doc_ids)}) must match document count ({n_new})"
                )
            generated_ids = list(doc_ids)

        # Pre-normalize onto the unit hypersphere S^(dim-1)
        normalized_batch = safe_l2_normalize(docs, axis=1, eps=self.eps)

        if self._doc_matrix is None:
            self._doc_matrix = normalized_batch
        else:
            self._doc_matrix = np.vstack([self._doc_matrix, normalized_batch])

        self._doc_ids.extend(generated_ids)
        self._num_docs = self._doc_matrix.shape[0]

    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
        angular_threshold: float | None = None,
    ) -> SearchResult:
        """
        Executes exact MIPS (Maximum Inner Product Search) on the unit hypersphere.

        Args:
            query: Query vector of shape (D,).
            top_k: Number of highest-similarity documents to return.
            angular_threshold: Optional maximum allowed angle theta (in radians).
                               Filters out documents with angle > threshold.

        Returns:
            SearchResult dataclass containing indices, IDs, scores, and angles.
        """
        if self._doc_matrix is None or self._num_docs == 0:
            raise RuntimeError("Index is empty. Add documents before searching.")

        query = np.asarray(query, dtype=np.float32).ravel()
        if query.shape[0] != self.dim:
            raise ValueError(
                f"Query dimension mismatch: expected D={self.dim}, got D={query.shape[0]}"
            )

        # 1. Project query vector to unit hypersphere
        q_hat = safe_l2_normalize(query, axis=0, eps=self.eps)

        # 2. Level-2 BLAS GEMV: s = D_hat @ q_hat
        # Directly yields cosine similarities because ||d_i|| = ||q|| = 1.0
        scores = np.dot(self._doc_matrix, q_hat)

        # 3. Cauchy-Schwarz clamping to strictly enforce [-1.0, 1.0]
        scores = np.clip(scores, -1.0, 1.0)

        # 4. Determine retrieval depth k
        k = min(top_k, self._num_docs)

        # 5. O(N + k log k) top-k selection via partial sorting
        if k == self._num_docs:
            top_indices = np.argsort(-scores)
        else:
            partitioned_idx = np.argpartition(-scores, k)[:k]
            # Sort only the partitioned top-k candidates
            top_indices = partitioned_idx[np.argsort(-scores[partitioned_idx])]

        top_scores = scores[top_indices]

        # 6. Compute exact geometric angles and true metric angular distances
        thetas, angular_dists = compute_angular_distance(top_scores)
        angles_deg = np.degrees(thetas)

        # 7. Optional angular metric radius filtering
        if angular_threshold is not None:
            mask = thetas <= angular_threshold
            top_indices = top_indices[mask]
            top_scores = top_scores[mask]
            angular_dists = angular_dists[mask]
            angles_deg = angles_deg[mask]

        retrieved_ids = [self._doc_ids[idx] for idx in top_indices]

        return SearchResult(
            doc_indices=top_indices,
            doc_ids=retrieved_ids,
            cosine_similarities=top_scores,
            angular_distances=angular_dists,
            angles_degrees=angles_deg,
        )
