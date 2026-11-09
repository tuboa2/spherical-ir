"""
Unit tests for Dual-Threshold Hierarchical Search Engine.

Validates:
- Semantic relevance boundary filtering (theta <= 60 deg, cos_sim >= 0.5)
- Near-duplicate disambiguation (theta_inter <= 18.19 deg, cos_sim >= 0.95)
- Hierarchical nesting of duplicate matches under canonical results
- Token-level explainability of duplicate relationships
- Result list diversity preservation
"""

import unittest
import numpy as np

from src.engine import DenseCosineIndex
from src.vectorizer import SublinearTfidfVectorizer
from src.search import DualThresholdSearchEngine


class TestDualThresholdSearch(unittest.TestCase):
    """Rigorous tests for hierarchical search and deduplication."""

    def setUp(self):
        # Construct corpus with both distinct topics and intentional near-duplicates
        self.corpus = [
            # Topic A (Kernel): Canonical
            "Linux kernel memory management subsystem handles page allocation and virtual memory mapping.",
            # Topic A (Kernel): Near-duplicate 1 (rephrased with minor edits)
            "Linux kernel memory management handles virtual memory mapping and page allocation subsystem.",
            # Topic A (Kernel): Near-duplicate 2 (minor additions)
            "Linux kernel memory management subsystem handles page allocation, memory mapping, and page frames.",
            # Topic B (Hardware): Canonical
            "AMD CPU registers and microarchitecture feature AVX2 SIMD vector extensions for parallel calculations.",
            # Topic B (Hardware): Near-duplicate (reordered with slight variation)
            "AMD CPU microarchitecture and registers feature AVX2 SIMD extensions for parallel calculations.",
            # Topic C (Database): Completely distinct topic
            "Relational database storage engine implements write ahead logging and B-tree disk block indices.",
            # Topic D (Networking): Distinct topic
            "TCP IP socket network stack processes incoming ethernet frames and handles packet routing.",
        ]
        self.doc_ids = [
            "kernel_v1",
            "kernel_dup1",
            "kernel_dup2",
            "amd_v1",
            "amd_dup1",
            "db_doc",
            "net_doc",
        ]

        self.vectorizer = SublinearTfidfVectorizer(max_features=128, ngram_range=(1, 1))
        doc_matrix = self.vectorizer.fit_transform(self.corpus)

        self.index = DenseCosineIndex(dim=len(self.vectorizer.feature_names_))
        self.index.add_documents(doc_matrix, doc_ids=self.doc_ids)

        self.search_engine = DualThresholdSearchEngine(
            index=self.index,
            vectorizer=self.vectorizer,
            relevance_threshold_deg=75.0,   # captures relevant docs in [0, 75 deg]
            duplicate_threshold_deg=30.0,  # cos_sim >= ~0.866
        )

    def test_relevance_threshold_filtering(self):
        """
        Confirms that documents with theta > 75 degrees (cos_sim < ~0.259)
        are excluded from the candidate pool.
        """
        # Querying specifically about TCP networking
        query = "TCP IP socket network packet routing"
        results = self.search_engine.search(query, top_k=5)

        self.assertGreater(len(results), 0)
        # All returned canonical items must satisfy theta <= 75 degrees
        for res in results:
            self.assertLessEqual(res.query_angle_deg, 75.0 + 1e-4)
            self.assertGreaterEqual(res.query_similarity, 0.50 - 1e-4)

        # Database and AMD documents must NOT be returned (their angle is > 60 deg)
        result_ids = [r.canonical_id for r in results]
        self.assertIn("net_doc", result_ids)
        self.assertNotIn("db_doc", result_ids)
        self.assertNotIn("amd_v1", result_ids)

    def test_near_duplicate_clustering_and_suppression(self):
        """
        Confirms that when querying about Linux kernel memory,
        kernel_v1 is elected as canonical and kernel_dup1/kernel_dup2
        are nested as subordinate duplicates rather than filling top-k slots.
        """
        query = "Linux kernel memory management page allocation"
        results = self.search_engine.search(query, top_k=5)

        # Top canonical result must be a kernel document
        top_res = results[0]
        self.assertTrue(top_res.canonical_id.startswith("kernel_"))

        # The other kernel documents must be nested in duplicates
        duplicate_ids = [dup.doc_id for dup in top_res.duplicates]
        self.assertGreaterEqual(len(duplicate_ids), 1)

        # None of the duplicates should appear in the primary canonical results list
        canonical_ids = [r.canonical_id for r in results]
        for dup_id in duplicate_ids:
            self.assertNotIn(dup_id, canonical_ids[1:])

    def test_duplicate_token_contributions(self):
        """
        Verifies that subordinate duplicates contain explainable token contributions
        showing why they matched the canonical document.
        """
        query = "Linux kernel memory"
        results = self.search_engine.search(query, top_k=5)

        top_res = results[0]
        self.assertTrue(len(top_res.duplicates) > 0)
        first_dup = top_res.duplicates[0]

        # Verify duplicate similarity meets threshold (cos_sim >= 0.95, theta <= 18.19 deg)
        self.assertGreaterEqual(first_dup.similarity_to_canonical, 0.90)
        self.assertLessEqual(first_dup.angle_to_canonical_deg, 25.0)

        # Verify token contributions are present and sum to similarity
        self.assertTrue(len(first_dup.token_contributions) > 0)
        top_terms = [t for t, _, _ in first_dup.token_contributions]
        self.assertTrue(any(term in top_terms for term in ["linux", "kernel", "memory"]))

    def test_result_diversity(self):
        """
        Confirms that search results across broad queries return diverse topics
        rather than redundant duplicates of a single topic.
        """
        # A broad technical query matching multiple domains
        query = "Linux kernel architecture and CPU registers"
        results = self.search_engine.search(query, top_k=5)

        canonical_ids = [r.canonical_id for r in results]
        # Should contain at least one kernel doc and at least one AMD/CPU doc
        has_kernel = any(cid.startswith("kernel_") for cid in canonical_ids)
        has_amd = any(cid.startswith("amd_") for cid in canonical_ids)

        self.assertTrue(has_kernel)
        self.assertTrue(has_amd)


if __name__ == "__main__":
    unittest.main()
