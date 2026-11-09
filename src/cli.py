"""
Interactive Command-Line Interface for Spherical-IR.

Provides zero-download offline commands:
- build-index: Ingests native Linux man pages and creates spherical index.
- search: Executes dual-threshold hierarchical search with duplicate grouping.
- explain: Decomposes inner product into exact token contributions.
- audit-duplicates: Scans entire corpus for near-duplicate clusters.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

from .engine import DenseCosineIndex, compute_angular_distance, safe_l2_normalize
from .ingest import LocalDocumentationIngester
from .search import DualThresholdSearchEngine
from .vectorizer import SublinearTfidfVectorizer

INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "index")


def save_artifacts(
    index: DenseCosineIndex,
    vectorizer: SublinearTfidfVectorizer,
    doc_texts: list[tuple[str, str]],
    output_dir: str = INDEX_DIR,
) -> None:
    """Serializes the index matrix, vocabulary, and document metadata."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save normalized document matrix as compressed float32 npz
    assert index._doc_matrix is not None
    matrix_path = os.path.join(output_dir, "doc_matrix.npz")
    np.savez_compressed(matrix_path, doc_matrix=index._doc_matrix)

    # 2. Save vectorizer vocabulary and IDF weights
    meta_path = os.path.join(output_dir, "vectorizer_meta.json")
    meta = {
        "max_features": vectorizer.max_features,
        "feature_names": vectorizer.feature_names_,
        "idf_weights": vectorizer.idf_weights_.tolist()
        if vectorizer.idf_weights_ is not None
        else [],
        "doc_ids": index._doc_ids,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 3. Save text snippet store for snippet previews
    snippets_path = os.path.join(output_dir, "doc_snippets.json")
    snippet_dict = {
        doc_id: (text[:250] + "..." if len(text) > 250 else text)
        for doc_id, text in doc_texts
    }
    with open(snippets_path, "w", encoding="utf-8") as f:
        json.dump(snippet_dict, f, indent=2)


def load_artifacts(
    index_dir: str = INDEX_DIR,
) -> tuple[DenseCosineIndex, SublinearTfidfVectorizer, dict]:
    """Loads serialized index, vectorizer, and snippets from disk."""
    matrix_path = os.path.join(index_dir, "doc_matrix.npz")
    meta_path = os.path.join(index_dir, "vectorizer_meta.json")
    snippets_path = os.path.join(index_dir, "doc_snippets.json")

    if not os.path.exists(matrix_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Index artifacts not found in {index_dir}. Run 'build-index' first."
        )

    # Load matrix
    with np.load(matrix_path) as data:
        matrix = data["doc_matrix"]

    # Load metadata
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Reconstitute vectorizer
    vectorizer = SublinearTfidfVectorizer(max_features=meta["max_features"])
    vectorizer.feature_names_ = meta["feature_names"]
    vectorizer.vocabulary_ = {
        term: idx for idx, term in enumerate(meta["feature_names"])
    }
    vectorizer.idf_weights_ = np.array(meta["idf_weights"], dtype=np.float32)
    vectorizer._is_fitted = True

    # Reconstitute index
    dim = len(vectorizer.feature_names_)
    index = DenseCosineIndex(dim=dim)
    index._doc_matrix = matrix
    index._doc_ids = meta["doc_ids"]
    index._num_docs = matrix.shape[0]

    snippets = {}
    if os.path.exists(snippets_path):
        with open(snippets_path, "r", encoding="utf-8") as f:
            snippets = json.load(f)

    return index, vectorizer, snippets


def cmd_build_index(args: argparse.Namespace) -> None:
    """Builds and caches a spherical index from local system documentation."""
    print("=" * 65)
    print("Spherical-IR: Native System Documentation Ingestion")
    print("=" * 65)
    t0 = time.time()

    man_dir = args.man_dir
    print(f"[*] Ingesting Linux manual pages from: {man_dir}")
    print(f"[*] Max document limit: {args.max_docs}")

    ingester = LocalDocumentationIngester()
    try:
        raw_docs = ingester.load_man_pages(max_docs=args.max_docs, man_dir=man_dir)
    except FileNotFoundError as err:
        print(f"[!] Error: {err}")
        sys.exit(1)

    if not raw_docs:
        print("[!] No valid manual pages found.")
        sys.exit(1)

    print(f"[+] Loaded {len(raw_docs)} clean manual pages in {time.time() - t0:.2f}s")

    # Fit vectorizer
    t_v = time.time()
    vectorizer = SublinearTfidfVectorizer(
        max_features=args.max_features, ngram_range=(1, 2)
    )
    texts = [text for _, text in raw_docs]
    doc_ids = [doc_id for doc_id, _ in raw_docs]

    print(
        f"[*] Fitting sublinear TF-IDF vectorizer (vocab size D={args.max_features})..."
    )
    doc_matrix = vectorizer.fit_transform(texts)
    print(f"[+] Vectorized {len(raw_docs)} documents in {time.time() - t_v:.2f}s")

    # Build DenseCosineIndex
    t_i = time.time()
    dim = len(vectorizer.feature_names_)
    index = DenseCosineIndex(dim=dim)
    index.add_documents(doc_matrix, doc_ids=doc_ids)
    print(
        f"[+] Projected and indexed onto unit hypersphere in {time.time() - t_i:.2f}s"
    )

    # Save artifacts
    save_dir = args.output_dir or INDEX_DIR
    print(f"[*] Serializing index artifacts to: {save_dir}")
    save_artifacts(index, vectorizer, raw_docs, output_dir=save_dir)

    total_time = time.time() - t0
    index_size = os.path.getsize(os.path.join(save_dir, "doc_matrix.npz")) / (
        1024 * 1024
    )
    print("-" * 65)
    print(f"[SUCCESS] Index built: {index.num_docs} documents, D={dim} dimensions")
    print(f"[SUCCESS] Total footprint on disk: {index_size:.2f} MB")
    print(f"[SUCCESS] Total elapsed time: {total_time:.2f}s")
    print("=" * 65)


def cmd_search(args: argparse.Namespace) -> None:
    """Executes dual-threshold hierarchical search on the local index."""
    index_dir = args.index_dir or INDEX_DIR
    try:
        index, vectorizer, snippets = load_artifacts(index_dir)
    except FileNotFoundError as err:
        print(f"[!] {err}")
        sys.exit(1)

    engine = DualThresholdSearchEngine(
        index=index,
        vectorizer=vectorizer,
        relevance_threshold_deg=args.relevance_deg,
        duplicate_threshold_deg=args.duplicate_deg,
    )

    t0 = time.time()
    results = engine.search(
        args.query, top_k=args.top_k, max_candidates=args.candidates
    )
    elapsed_ms = (time.time() - t0) * 1000.0

    print("=" * 70)
    print(f'Query: "{args.query}"  [Evaluated in {elapsed_ms:.2f} ms]')
    print(
        f"Thresholds: Relevance <= {args.relevance_deg:.1f} deg | Duplicates <= {args.duplicate_deg:.1f} deg"
    )
    print("=" * 70)

    if not results:
        print("[!] No documents found within the semantic relevance boundary.")
        return

    for rank, res in enumerate(results, start=1):
        snippet = snippets.get(res.canonical_id, "")
        dup_count = len(res.duplicates)
        dup_badge = (
            f" [+{dup_count} duplicate{'s' if dup_count != 1 else ''}]"
            if dup_count > 0
            else ""
        )

        print(f"\n[{rank}] CANONICAL: {res.canonical_id}{dup_badge}")
        print(
            f"    Similarity: {res.query_similarity:.4f}  |  Angle: {res.query_angle_deg:.2f} deg  |  Metric D_ang: {res.query_angular_distance:.4f}"
        )
        if snippet:
            print(f"    Excerpt: {snippet}")

        # Print nested duplicates
        for dup in res.duplicates:
            print(f"      ↳ DUPLICATE: {dup.doc_id}")
            print(
                f"          Angle to canonical: {dup.angle_to_canonical_deg:.2f} deg  (similarity: {dup.similarity_to_canonical:.4f})"
            )
            if dup.token_contributions:
                terms_str = ", ".join(
                    f"{t}: {pct:.1f}%" for t, _, pct in dup.token_contributions[:4]
                )
                print(f"          Shared drivers: {terms_str}")

    print("\n" + "=" * 70)


def cmd_explain(args: argparse.Namespace) -> None:
    """Decomposes the cosine similarity between a query and a document into token drivers."""
    index_dir = args.index_dir or INDEX_DIR
    try:
        index, vectorizer, snippets = load_artifacts(index_dir)
    except FileNotFoundError as err:
        print(f"[!] {err}")
        sys.exit(1)

    if args.doc_id not in index._doc_ids:
        print(f"[!] Document ID '{args.doc_id}' not found in index.")
        sys.exit(1)

    doc_idx = index._doc_ids.index(args.doc_id)
    assert index._doc_matrix is not None
    d_hat = index._doc_matrix[doc_idx]

    q_raw = vectorizer.transform([args.query])
    q_hat = safe_l2_normalize(q_raw, axis=1)[0]

    cos_sim = float(np.dot(q_hat, d_hat))
    theta, d_ang = compute_angular_distance(cos_sim)
    theta_deg = math.degrees(theta)

    explanations = vectorizer.explain_similarity(q_hat, d_hat, top_n=args.top_n)

    print("=" * 65)
    print("Explainable Cosine Similarity Decomposition")
    print(f'Query: "{args.query}"')
    print(f"Document: {args.doc_id}")
    print(f"Total Cosine Similarity: {cos_sim:.5f}")
    print(f"Geometric Angle (theta): {theta_deg:.2f} deg  |  Metric D_ang: {d_ang:.5f}")
    print("-" * 65)
    print(f"{'Token Term':<25} | {'Inner Product Contribution':<28} | {'% Total':<8}")
    print("-" * 65)

    sum_contrib = 0.0
    for term, contrib, pct in explanations:
        sum_contrib += contrib
        print(f"{term:<25} | {contrib:<28.6f} | {pct:>6.2f}%")

    print("-" * 65)
    print(f"Sum of Listed Contributions: {sum_contrib:.6f}")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="spherical-ir",
        description="Production Directional Semantic Retrieval & Spherical Indexing Engine.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: build-index
    build_parser = subparsers.add_parser(
        "build-index", help="Build index from local man pages."
    )
    build_parser.add_argument(
        "--man-dir", default="/usr/share/man/man1", help="Path to manual directory."
    )
    build_parser.add_argument(
        "--max-docs", type=int, default=500, help="Max documents to index."
    )
    build_parser.add_argument(
        "--max-features", type=int, default=1024, help="Vocabulary feature dimension."
    )
    build_parser.add_argument(
        "--output-dir", default=INDEX_DIR, help="Output directory for index files."
    )

    # Subcommand: search
    search_parser = subparsers.add_parser(
        "search", help="Execute dual-threshold hierarchical search."
    )
    search_parser.add_argument("query", help="Text search query.")
    search_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of canonical results."
    )
    search_parser.add_argument(
        "--relevance-deg",
        type=float,
        default=80.0,
        help="Max angle for relevance (default 80.0 deg for high-dim spaces).",
    )
    search_parser.add_argument(
        "--duplicate-deg",
        type=float,
        default=30.0,
        help="Max angle for duplicate grouping.",
    )
    search_parser.add_argument(
        "--candidates", type=int, default=50, help="Max candidates to evaluate."
    )
    search_parser.add_argument(
        "--index-dir", default=INDEX_DIR, help="Path to index directory."
    )

    # Subcommand: explain
    explain_parser = subparsers.add_parser(
        "explain", help="Explain inner product between query and doc."
    )
    explain_parser.add_argument("query", help="Search query string.")
    explain_parser.add_argument("doc_id", help="Target document ID to explain.")
    explain_parser.add_argument(
        "--top-n", type=int, default=10, help="Number of top contributing terms."
    )
    explain_parser.add_argument(
        "--index-dir", default=INDEX_DIR, help="Path to index directory."
    )

    args = parser.parse_args()

    if args.command == "build-index":
        cmd_build_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "explain":
        cmd_explain(args)


if __name__ == "__main__":
    main()
