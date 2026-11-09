"""
Unit tests for Spherical-IR Core Mathematical Engine.

Validates the mathematical properties derived in 'Angles and Cosine Similarity.md':
- Magnitude Invariance: cos_sim(c*u, v) = cos_sim(u, v)
- Cauchy-Schwarz bounds: -1.0 <= cos_sim <= 1.0
- IEEE-754 overflow safeguard: arccos(1.0 + 1e-16) does not produce NaN
- Zero vector epsilon safety
- Euclidean-to-Cosine isomorphism: ||u_hat - v_hat||^2 = 2 * (1 - cos_sim)
- Dense index top-k exactness and angular metric filtering
"""

import math
import unittest

import numpy as np
from spherical_ir.engine import (
    DenseCosineIndex,
    compute_angular_distance,
    compute_cosine_similarity,
    safe_l2_normalize,
    verify_euclidean_isomorphism,
)


class TestMathematicalInvariants(unittest.TestCase):
    """Rigorous tests proving the mathematical theorems from the notes."""

    def test_example_1_from_notes(self):
        """
        Validates Section 9 Example 1:
        u = [1, 2, 2]^T, v = [2, 1, -1]^T
        Expected cos_sim = 2 / (3 * sqrt(6)) ~= 0.2721655
        Expected theta ~= 1.2945 rad (74.17 degrees)
        """
        u = np.array([1.0, 2.0, 2.0], dtype=np.float32)
        v = np.array([2.0, 1.0, -1.0], dtype=np.float32)

        sim = compute_cosine_similarity(u, v)
        expected_sim = 2.0 / (3.0 * math.sqrt(6.0))
        self.assertAlmostEqual(sim, expected_sim, places=5)

        theta, _ = compute_angular_distance(sim)
        self.assertAlmostEqual(theta, math.acos(expected_sim), places=5)
        self.assertAlmostEqual(math.degrees(theta), 74.17, delta=0.05)

    def test_magnitude_invariance_example_2(self):
        """
        Validates Section 9 Example 2 (Magnitude Invariance):
        Sentence A: a = [0.8, 0.6]^T (norm = 1.0)
        Sentence B: b = [4.0, 3.0]^T (norm = 5.0, b = 5*a)
        Euclidean distance = 4.0, but cos_sim must be exactly 1.0.
        """
        a = np.array([0.8, 0.6], dtype=np.float32)
        b = np.array([4.0, 3.0], dtype=np.float32)

        # Confirm Euclidean distance is non-zero
        euclid_dist = np.linalg.norm(a - b)
        self.assertAlmostEqual(euclid_dist, 4.0, places=5)

        # Confirm Cosine similarity factors out magnitude
        sim = compute_cosine_similarity(a, b)
        self.assertAlmostEqual(sim, 1.0, places=6)

        theta, ang_dist = compute_angular_distance(sim)
        self.assertAlmostEqual(theta, 0.0, places=6)
        self.assertAlmostEqual(ang_dist, 0.0, places=6)

    def test_scale_invariance_across_orders_of_magnitude(self):
        """
        Tests positive scale invariance across extreme multipliers:
        cos_sim(c * u, v) == cos_sim(u, v) for c in [1e-4, 1e5].
        """
        u = np.array([3.14, -1.59, 2.65, -3.58], dtype=np.float32)
        v = np.array([-2.71, 8.28, 1.82, 0.45], dtype=np.float32)
        base_sim = compute_cosine_similarity(u, v)

        scales = [1e-4, 1e-2, 0.1, 2.0, 50.0, 1e3, 1e5]
        for c in scales:
            scaled_sim = compute_cosine_similarity(c * u, v)
            self.assertAlmostEqual(
                scaled_sim,
                base_sim,
                places=5,
                msg=f"Scale invariance failed at scalar c={c}",
            )

    def test_euclidean_cosine_isomorphism_derivation_2(self):
        """
        Validates Section 8 Derivation 2:
        For unit vectors, ||u_hat - v_hat||^2 = 2 * (1 - cos_sim).
        """
        rng = np.random.default_rng(42)
        for _ in range(50):
            dim = 128
            u = rng.standard_normal(dim).astype(np.float32)
            v = rng.standard_normal(dim).astype(np.float32)
            u_hat = safe_l2_normalize(u)
            v_hat = safe_l2_normalize(v)

            discrepancy = verify_euclidean_isomorphism(u_hat, v_hat)
            self.assertLess(
                discrepancy,
                1e-5,
                msg=f"Euclidean-cosine isomorphism violated with delta={discrepancy}",
            )

    def test_cauchy_schwarz_clamping_and_nan_prevention(self):
        """
        Validates Section 14 & 17:
        Values slightly exceeding 1.0 due to floating-point drift (1.0000000000000002)
        must not produce NaN in arccos.
        """
        drifted_val = 1.0000000000000002
        theta, ang_dist = compute_angular_distance(drifted_val)
        self.assertFalse(np.isnan(theta))
        self.assertAlmostEqual(theta, 0.0, places=6)
        self.assertAlmostEqual(ang_dist, 0.0, places=6)

        drifted_negative = -1.0000000000000004
        theta_neg, ang_dist_neg = compute_angular_distance(drifted_negative)
        self.assertFalse(np.isnan(theta_neg))
        self.assertAlmostEqual(theta_neg, math.pi, places=6)
        self.assertAlmostEqual(ang_dist_neg, 1.0, places=6)

    def test_zero_vector_stability(self):
        """
        Validates Section 17 edge case:
        Zero vectors must not throw ZeroDivisionError or produce NaN/Inf.
        """
        zero_v = np.zeros(64, dtype=np.float32)
        norm_v = safe_l2_normalize(zero_v)
        self.assertTrue(np.all(norm_v == 0.0))
        self.assertFalse(np.any(np.isnan(norm_v)))

        nonzero_v = np.ones(64, dtype=np.float32)
        sim = compute_cosine_similarity(zero_v, nonzero_v)
        self.assertEqual(sim, 0.0)


class TestDenseCosineIndex(unittest.TestCase):
    """Tests for the pre-normalized dense retrieval engine."""

    def setUp(self):
        self.dim = 32
        self.index = DenseCosineIndex(dim=self.dim)
        self.rng = np.random.default_rng(1337)

    def test_exact_top_k_retrieval(self):
        # Create 100 random documents
        corpus = self.rng.standard_normal((100, self.dim)).astype(np.float32)
        self.index.add_documents(corpus, doc_ids=[f"doc_{i}" for i in range(100)])

        # Query identical to document 42 (with random positive scaling factor)
        query = corpus[42] * 3.7
        result = self.index.search(query, top_k=5)

        # Top 1 must be doc_42 with similarity ~1.0
        self.assertEqual(result.doc_ids[0], "doc_42")
        self.assertAlmostEqual(result.cosine_similarities[0], 1.0, places=5)
        # Note on float32 precision: d/dx arccos(x) -> infinity as x -> 1.
        # For float32 eps ~= 1.19e-7, the minimum angular resolution near theta=0 is
        # sqrt(2 * eps) ~= 0.00049 rad ~= 0.028 degrees. delta=0.03 accounts for this.
        self.assertAlmostEqual(result.angles_degrees[0], 0.0, delta=0.03)

    def test_angular_radius_filtering(self):
        corpus = self.rng.standard_normal((50, self.dim)).astype(np.float32)
        self.index.add_documents(corpus)

        query = self.rng.standard_normal(self.dim).astype(np.float32)
        # Filter where theta <= 45 degrees (pi / 4 radians)
        threshold_rad = math.pi / 4.0
        result = self.index.search(query, top_k=50, angular_threshold=threshold_rad)

        for angle in result.angles_degrees:
            self.assertLessEqual(angle, 45.0 + 1e-4)


if __name__ == "__main__":
    unittest.main()
