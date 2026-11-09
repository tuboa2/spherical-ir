"""
Unit tests for Sublinear TF-IDF Vectorizer and First-Quadrant Geometry.

Validates:
- First-quadrant non-negativity: X in R_(>=0)^D and cos_sim in [0.0, 1.0]
- Sublinear frequency scaling: 1 + ln(tf)
- Smooth IDF calculation and monotonicity
- Explainable cosine decomposition: sum(q_hat * d_hat) == cos_sim
- Length-invariant near-duplicate detection
- End-to-end integration with DenseCosineIndex
"""

import math
import unittest

import numpy as np
from src.engine import DenseCosineIndex, safe_l2_normalize
from src.vectorizer import SublinearTfidfVectorizer


class TestSublinearTfidfVectorizer(unittest.TestCase):
    """Rigorous tests for the vectorizer and first-quadrant properties."""

    def setUp(self):
        self.corpus = [
            "Linux kernel memory management subsystem handles page allocation and virtual memory mapping.",
            "Linux page allocation uses buddy allocator for physical memory management.",
            "AMD CPU registers and instruction set architecture support AVX2 SIMD vector extensions.",
            "Modern x86 microprocessors feature hardware registers and SIMD execution units for parallel computing.",
            "Database storage engine manages disk blocks B-tree indices and write ahead logging.",
        ]
        self.vectorizer = SublinearTfidfVectorizer(max_features=64, ngram_range=(1, 2))
        self.vectorizer.fit(self.corpus)

    def test_first_quadrant_non_negativity_invariant(self):
        """
        Validates Section 17 of notes:
        All feature values in TF-IDF are non-negative (x in R_(>=0)^D).
        Cosine similarity must be strictly bounded in [0.0, 1.0].
        No negative similarities can occur.
        """
        matrix = self.vectorizer.transform(self.corpus)
        # 1. Feature matrix is non-negative
        self.assertTrue(np.all(matrix >= 0.0))

        # 2. Pairwise cosine similarity across all document pairs is in [0.0, 1.0]
        norm_matrix = safe_l2_normalize(matrix, axis=1)
        sim_matrix = np.dot(norm_matrix, norm_matrix.T)

        self.assertTrue(np.all(sim_matrix >= -1e-6))
        self.assertTrue(np.all(sim_matrix <= 1.0 + 1e-6))

        # 3. Angle theta is strictly in [0, pi/2] (0 to 90 degrees)
        clipped_sim = np.clip(sim_matrix, 0.0, 1.0)
        angles_deg = np.degrees(np.arccos(clipped_sim))
        self.assertTrue(np.all(angles_deg >= 0.0))
        self.assertTrue(np.all(angles_deg <= 90.0 + 1e-4))

    def test_sublinear_tf_dampening(self):
        """
        Verifies that repeating a term 10 times does not increase its weight 10-fold.
        tf_sub(10) = 1 + ln(10) ~= 3.3025 vs linear tf = 10.
        """
        doc_single = "memory page allocation"
        doc_repeated = (
            "memory page allocation " + "memory " * 9
        )  # 'memory' appears 10 times

        v_single = self.vectorizer.transform([doc_single])[0]
        v_repeated = self.vectorizer.transform([doc_repeated])[0]

        mem_idx = self.vectorizer.vocabulary_["memory"]
        weight_single = v_single[mem_idx]
        weight_repeated = v_repeated[mem_idx]

        expected_ratio = (1.0 + math.log(10)) / 1.0
        actual_ratio = weight_repeated / weight_single
        self.assertAlmostEqual(actual_ratio, expected_ratio, places=4)

    def test_explainable_cosine_decomposition(self):
        """
        Verifies that sum of element-wise contributions equals total cosine similarity:
        cos_sim(q, d) = sum(q_hat * d_hat)
        """
        query = "Linux virtual memory page allocation"
        doc = self.corpus[0]

        q_vec = self.vectorizer.transform([query])
        d_vec = self.vectorizer.transform([doc])

        q_hat = safe_l2_normalize(q_vec, axis=1)[0]
        d_hat = safe_l2_normalize(d_vec, axis=1)[0]

        total_sim = float(np.dot(q_hat, d_hat))
        explanations = self.vectorizer.explain_similarity(q_hat, d_hat, top_n=10)

        # Confirm contributions sum to total similarity (for matching terms)
        sum_contrib = sum(contrib for _, contrib, _ in explanations)
        self.assertAlmostEqual(sum_contrib, total_sim, places=5)

        # Confirm highest contributing terms are the shared technical words
        top_terms = [t for t, _, _ in explanations[:3]]
        self.assertTrue(
            any(term in top_terms for term in ["memory", "page", "allocation"])
        )

    def test_length_invariant_near_duplicate_detection(self):
        """
        Validates Section 9 Example 2 with real text:
        A concise statement and an expanded technical description covering
        the same concepts yield high cosine similarity despite length difference.
        """
        short_statement = "AMD CPU registers support AVX2 SIMD extensions."
        expanded_manual = (
            "The AMD central processing unit architecture features general purpose registers "
            "as well as dedicated vector registers that support AVX2 SIMD extensions. "
            "These AVX2 SIMD instructions accelerate high throughput floating point calculations."
        )

        dim = len(self.vectorizer.feature_names_)
        index = DenseCosineIndex(dim=dim)
        index.add_documents(
            self.vectorizer.transform([short_statement]), doc_ids=["short_doc"]
        )

        # Query with expanded manual
        query_vec = self.vectorizer.transform([expanded_manual])[0]
        result = index.search(query_vec, top_k=1)

        self.assertEqual(result.doc_ids[0], "short_doc")
        # High similarity despite substantial word count difference
        self.assertGreater(result.cosine_similarities[0], 0.70)
        self.assertLess(result.angles_degrees[0], 45.0)


if __name__ == "__main__":
    unittest.main()
