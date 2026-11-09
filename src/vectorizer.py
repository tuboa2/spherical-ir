"""
Sublinear TF-IDF Feature Extractor with Explainable First-Quadrant Geometry.

Implements lexical feature projection onto the first hyper-quadrant R_(>=0)^D:
- Logarithmic sublinear term frequency: tf_sub = 1 + ln(tf) if tf > 0 else 0
- Smooth inverse document frequency: idf = ln((1 + N) / (1 + df)) + 1
- First-quadrant geometric bounding: all x_ij >= 0, proving cos_sim in [0.0, 1.0]
- Explainable cosine decomposition: cos_sim(q, d) = sum(q_hat_j * d_hat_j)
  allowing exact term-level auditing of retrieval decisions.
"""

import math
import re

import numpy as np


class SublinearTfidfVectorizer:
    """
    Sublinear TF-IDF Vectorizer with Bounded Vocabulary and Explainable Decomposition.

    Maps unstructured text into non-negative feature vectors x in R_(>=0)^D.
    """

    def __init__(
        self,
        max_features: int = 1024,
        min_df: int = 1,
        max_df_ratio: float = 0.95,
        ngram_range: tuple[int, int] = (1, 2),
    ):
        """
        Args:
            max_features: Maximum dimension D of the feature space.
            min_df: Minimum absolute document frequency for term inclusion.
            max_df_ratio: Maximum document frequency ratio to filter stopwords.
            ngram_range: Tuple (min_n, max_n) for unigram and bigram extraction.
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.ngram_range = ngram_range

        self.vocabulary_: dict[str, int] = {}
        self.feature_names_: list[str] = []
        self.idf_weights_: np.ndarray | None = None  # Shape: (D,)
        self._is_fitted: bool = False

    def _tokenize(self, text: str) -> list[str]:
        """Extracts alphanumeric tokens and generates bounded n-grams."""
        raw_tokens = re.findall(r"[a-z0-9_]+", text.lower())
        tokens = [t for t in raw_tokens if len(t) >= 2]

        if self.ngram_range == (1, 1) or len(tokens) == 0:
            return tokens

        ngrams = list(tokens)
        min_n, max_n = self.ngram_range
        for n in range(max(2, min_n), max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = "_".join(tokens[i : i + n])
                ngrams.append(ngram)
        return ngrams

    def fit(self, raw_documents: list[str]) -> "SublinearTfidfVectorizer":
        """
        Fits vocabulary and calculates smoothed IDF weights across the corpus.

        Smooth IDF Formula:
            idf(t) = ln((1 + N) / (1 + df(t))) + 1.0
        """
        n_docs = len(raw_documents)
        if n_docs == 0:
            raise ValueError("Cannot fit vectorizer on an empty document list.")

        df_counts: dict[str, int] = {}
        for doc in raw_documents:
            unique_terms = set(self._tokenize(doc))
            for term in unique_terms:
                df_counts[term] = df_counts.get(term, 0) + 1

        max_df = int(n_docs * self.max_df_ratio)
        valid_terms = [
            (term, count)
            for term, count in df_counts.items()
            if count >= self.min_df and count <= max_df
        ]

        if len(valid_terms) < 1:
            valid_terms = list(df_counts.items())

        valid_terms.sort(key=lambda item: (-item[1], item[0]))

        selected_terms = valid_terms[: self.max_features]
        self.vocabulary_ = {term: idx for idx, (term, _) in enumerate(selected_terms)}
        self.feature_names_ = [term for term, _ in selected_terms]

        d = len(self.feature_names_)
        self.idf_weights_ = np.zeros(d, dtype=np.float32)
        for idx, term in enumerate(self.feature_names_):
            df = df_counts[term]
            self.idf_weights_[idx] = math.log((1.0 + n_docs) / (1.0 + df)) + 1.0

        self._is_fitted = True
        return self

    def transform(self, raw_documents: list[str]) -> np.ndarray:
        """
        Transforms text documents into sublinear TF-IDF vectors in R_(>=0)^D.

        Formula:
            w_(t, d) = (1 + ln(count(t, d))) * idf(t)  if count > 0 else 0
        """
        if not self._is_fitted or self.idf_weights_ is None:
            raise RuntimeError("Vectorizer must be fitted before calling transform.")

        n_docs = len(raw_documents)
        d = len(self.feature_names_)
        matrix = np.zeros((n_docs, d), dtype=np.float32)

        for doc_idx, doc in enumerate(raw_documents):
            tokens = self._tokenize(doc)
            term_counts: dict[str, int] = {}
            for t in tokens:
                if t in self.vocabulary_:
                    term_counts[t] = term_counts.get(t, 0) + 1

            for term, count in term_counts.items():
                col_idx = self.vocabulary_[term]
                tf_sub = 1.0 + math.log(count)
                matrix[doc_idx, col_idx] = tf_sub * self.idf_weights_[col_idx]

        return matrix

    def fit_transform(self, raw_documents: list[str]) -> np.ndarray:
        """Convenience method to fit and transform in a single pass."""
        return self.fit(raw_documents).transform(raw_documents)

    def explain_similarity(
        self,
        query_norm: np.ndarray,
        doc_norm: np.ndarray,
        top_n: int = 5,
    ) -> list[tuple[str, float, float]]:
        """
        Decomposes the cosine similarity into exact term-by-term contributions.

        Formula:
            cos_sim(q, d) = q_hat^T d_hat = sum_j (q_hat_j * d_hat_j)

        Returns:
            List of (term, absolute_contribution, percentage_of_total_similarity)
            sorted by largest contribution first.
        """
        if not self._is_fitted:
            raise RuntimeError("Vectorizer must be fitted.")

        q_hat = np.asarray(query_norm, dtype=np.float32).ravel()
        d_hat = np.asarray(doc_norm, dtype=np.float32).ravel()

        elementwise = q_hat * d_hat
        total_sim = float(np.sum(elementwise))

        if total_sim <= 1e-12:
            return []

        active_indices = np.where(elementwise > 1e-9)[0]
        sorted_indices = active_indices[np.argsort(-elementwise[active_indices])]

        explanations = []
        for idx in sorted_indices[:top_n]:
            term = self.feature_names_[idx]
            contrib = float(elementwise[idx])
            pct = (contrib / total_sim) * 100.0
            explanations.append((term, contrib, pct))

        return explanations
