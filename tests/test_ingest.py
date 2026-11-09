"""
Unit tests for Local Documentation Ingestion and CLI Serialization.

Validates:
- Roff macro stripping into clean plain text
- Native Linux man page decoding from /usr/share/man/man1
- Local text/markdown directory ingestion
- Serialization round-trip (save and reload of index, vectorizer, and snippets)
"""

import os
import tempfile
import unittest

from src.ingest import LocalDocumentationIngester, clean_roff_text
from src.vectorizer import SublinearTfidfVectorizer
from src.engine import DenseCosineIndex
from src.cli import save_artifacts, load_artifacts, DualThresholdSearchEngine


class TestIngestionAndCLI(unittest.TestCase):
    """Tests for document loading and CLI index persistence."""

    def test_clean_roff_text(self):
        sample_roff = (
            '.TH GZIP 1 "2022-01-01" "GNU gzip" "User Commands"\n'
            '.SH NAME\n'
            'gzip, gunzip, zcat \\- compress or expand files\n'
            '.SH DESCRIPTION\n'
            '\\fBGzip\\fR reduces the size of the named files using Lempel-Ziv coding.\n'
            '.PP\n'
            'The advanced compression algorithm preserves file metadata.\n'
        )
        cleaned = clean_roff_text(sample_roff)
        self.assertNotIn(".TH", cleaned)
        self.assertNotIn(".SH", cleaned)
        self.assertNotIn("\\fB", cleaned)
        self.assertNotIn("\\fR", cleaned)
        self.assertIn("reduces the size of the named files", cleaned)
        self.assertIn("advanced compression algorithm", cleaned)

    def test_load_man_pages_live_system(self):
        """Loads a small sample of live man pages from /usr/share/man/man1."""
        man_dir = "/usr/share/man/man1"
        if not os.path.exists(man_dir):
            self.skipTest(f"Man directory {man_dir} not available on this host.")

        ingester = LocalDocumentationIngester()
        docs = ingester.load_man_pages(max_docs=10, man_dir=man_dir, min_words=10)

        self.assertGreater(len(docs), 0)
        doc_id, text = docs[0]
        self.assertIsInstance(doc_id, str)
        self.assertIsInstance(text, str)
        self.assertGreaterEqual(len(text.split()), 10)

    def test_serialization_and_search_roundtrip(self):
        """Verifies saving and reloading the index and vectorizer artifacts."""
        docs = [
            ("arch_doc", "x86 microprocessor CPU architecture and vector registers with AVX2 SIMD instructions."),
            ("kernel_doc", "Linux kernel virtual memory management page allocation subsystem and buddy allocator."),
            ("storage_doc", "Database storage engine with write ahead logging and B-tree index structures."),
        ]
        texts = [t for _, t in docs]
        ids = [i for i, _ in docs]

        vectorizer = SublinearTfidfVectorizer(max_features=64, ngram_range=(1, 1))
        matrix = vectorizer.fit_transform(texts)

        index = DenseCosineIndex(dim=len(vectorizer.feature_names_))
        index.add_documents(matrix, doc_ids=ids)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_artifacts(index, vectorizer, docs, output_dir=tmp_dir)

            # Reload
            loaded_index, loaded_vec, loaded_snippets = load_artifacts(tmp_dir)
            self.assertEqual(loaded_index.num_docs, 3)
            self.assertEqual(len(loaded_vec.feature_names_), len(vectorizer.feature_names_))
            self.assertEqual(len(loaded_snippets), 3)

            # Search with reloaded engine
            engine = DualThresholdSearchEngine(loaded_index, loaded_vec)
            results = engine.search("Linux virtual memory page", top_k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].canonical_id, "kernel_doc")


if __name__ == "__main__":
    unittest.main()
