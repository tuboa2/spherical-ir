# Spherical-IR: Production Directional Semantic Retrieval & Hypersphere Indexing Engine

A production-grade, 100% air-gapped directional semantic retrieval and document deduplication engine optimized for severely constrained x86 hardware (AMD E2-9010 APU, 2 CPU cores, 15W thermal ceiling).

---

## 1. Mathematical Foundation

Spherical-IR operationalizes directional geometry on the unit hypersphere $\mathbb{S}^{D-1}$ as derived in `Angles and Cosine Similarity.md`:

1. **Directional Isolation via Magnitude Factorization:**
   $$\text{cos sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^\top \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \cos(\theta)$$
   Positive scale invariance ($\text{cos sim}(c\mathbf{u}, \mathbf{v}) = \text{cos sim}(\mathbf{u}, \mathbf{v})$ for $c > 0$) completely decouples semantic topic alignment from document length disparities.

2. **Pre-Normalized Euclidean Isomorphism (Derivation 2):**
   By mapping document vectors to the unit hypersphere offline:
   $$\hat{\mathbf{d}}_i = \frac{\mathbf{d}_i}{\|\mathbf{d}_i\|_2 + \epsilon} \in \mathbb{S}^{D-1}$$
   online query retrieval simplifies to an inner product via Level-2 BLAS GEMV:
   $$\mathbf{s} = \hat{\mathbf{D}}\hat{\mathbf{q}}$$
   Because $\|\hat{\mathbf{d}} - \hat{\mathbf{q}}\|_2^2 = 2(1 - \hat{\mathbf{d}}^\top \hat{\mathbf{q}})$, inner product ranking is mathematically isomorphic to minimum Euclidean distance on $\mathbb{S}^{D-1}$.

3. **Cauchy-Schwarz Bounding & Singularity Clamping:**
   $$\text{clip}(\mathbf{s}, -1.0, 1.0)$$
   Eliminates IEEE-754 precision drift beyond $1.0$ ($1.0 + 10^{-16}$), preventing `NaN` failures near the pole singularity of $\arccos(x)$ as $x \to 1^-$.

4. **First-Quadrant Non-Negativity (Section 17):**
   Lexical TF-IDF representations satisfy $\mathbf{x} \in \mathbb{R}_{\ge 0}^D$. This geometrically restricts vectors to the first hyper-quadrant, proving:
   $$\text{cos sim}(\mathbf{u}, \mathbf{v}) \in [0.0, 1.0], \quad \theta \in [0^\circ, 90^\circ]$$

5. **Explainable Inner Product Decomposition:**
   $$\text{cos sim}(\mathbf{q}, \mathbf{d}) = \sum_{j=1}^D (\hat{q}_j \cdot \hat{d}_j)$$
   Enables exact term-by-term attribution of retrieval and duplicate matching decisions.

6. **True Angular Metric Bounding:**
   Unlike cosine distance ($1 - \text{cos sim}$), normalized angular distance:
   $$D_{\text{ang}}(\mathbf{u}, \mathbf{v}) = \frac{\theta}{\pi} = \frac{\arccos(\text{cos sim})}{\pi}$$
   is a true mathematical metric satisfying the triangle inequality:
   $$D_{\text{ang}}(\mathbf{x}, \mathbf{z}) \le D_{\text{ang}}(\mathbf{x}, \mathbf{y}) + D_{\text{ang}}(\mathbf{y}, \mathbf{z})$$

---

## 2. Architecture & Directory Layout

```text
spherical-ir/
├── src/
│   ├── __init__.py      # Package exports
│   ├── engine.py        # DenseCosineIndex & hypersphere geometry
│   ├── vectorizer.py    # SublinearTfidfVectorizer (first-quadrant)
│   ├── search.py        # DualThresholdSearchEngine (deduplication)
│   ├── ingest.py        # LocalDocumentationIngester (man page parser)
│   └── cli.py           # Command-line interface
├── tests/
│   ├── test_engine.py      # Core mathematical invariant tests
│   ├── test_vectorizer.py  # First-quadrant & sublinear TF tests
│   ├── test_search.py      # Dual-threshold & diversity tests
│   └── test_ingest.py      # Ingestion & serialization tests
├── data/
│   └── index/
│       ├── doc_matrix.npz         # Serialized float32 normalized matrix
│       ├── vectorizer_meta.json   # Vocabulary & smooth IDF weights
│       ├── doc_snippets.json      # Plaintext excerpt store
│       └── duplicate_audit.json   # Corpus-wide duplicate audit report
└── README.md
```

---

## 3. Quickstart & CLI Usage

### Run Automated Test Suite (19 Tests)
```bash
PYTHONPATH=spherical-ir python3 -m unittest discover -s spherical-ir/tests -v
```

### Ingest & Build Index from Native Host Documentation
```bash
PYTHONPATH=spherical-ir python3 -m src.cli build-index --max-docs 500
```

### Execute Dual-Threshold Hierarchical Search
```bash
PYTHONPATH=spherical-ir python3 -m src.cli search "compress files" --top-k 5
```

### Explain Inner Product Decomposition
```bash
PYTHONPATH=spherical-ir python3 -m src.cli explain "compress files" bzcmp
```

### Run Corpus-Wide Duplicate Audit
```bash
PYTHONPATH=spherical-ir python3 -m src.cli audit-duplicates --duplicate-deg 30.0
```

---

## 4. Hardware Verification (AMD E2-9010 APU)

Measured on physical host (AMD E2-9010, 2 cores, 2.0 GHz, 6.7 GiB total RAM / 3.2 GiB available, Linux Mint 22):

| Operation | Measured Peak RSS | Measured Latency | Hard Budget | Status |
| :--- | ---: | ---: | ---: | :--- |
| **Ingestion & Indexing (500 docs)** | 100.2 MiB | 7.88 s | ≤ 2.2 GiB / ≤ 15 min | PASS |
| **Online Query Search** | 40.8 MiB | 3.72 ms | ≤ 1.0 GiB / ≤ 5 min | PASS |
| **Corpus Pairwise Audit (500 docs)** | 43.0 MiB | 0.91 s | ≤ 2.2 GiB / ≤ 15 min | PASS |
| **Unit Test Suite (19 tests)** | 38.6 MiB | 0.22 s | ≤ 2.2 GiB / ≤ 15 min | PASS |
| **Total Project Disk Footprint** | — | ~696 KB | ≤ 5 GB (≤ 2 GB target) | PASS |
| **Operating System Swap I/O** | 0 B | 0 swaps | 0 B | PASS |
