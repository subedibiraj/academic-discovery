# Reproducibility Statement

This document describes what can be reproduced from this repository
without any proprietary data or API keys, and what requires additional
steps (model download, re-scraping).

---

## What is fully reproducible without any downloads

All of the following run with `pip install -r requirements.txt` only:

| Script | What it produces | Input |
|--------|-----------------|-------|
| `analysis/significance.py` | Bootstrap CIs + Bonferroni tests (Table 6) | `data/final/comparison_results.json`, `data/final/relevance_labels.json` |
| `analysis/multi_query_eval.py` | 5-query NDCG table (Table 8), TF-IDF significance | `data/final/all_professors.json`, `data/final/comparison_results.json`, both label files |
| `analysis/arxiv_experiment.py` | Controlled arXiv experiment (Table 5) | `data/final/arxiv_impact.json`, `data/final/relevance_labels.json` |
| `analysis/learn_to_rank.py` | LTR NDCG/MAP + feature importances | `data/final/comparison_results.json`, `data/final/relevance_labels.json` |
| `analysis/ablation.py` | Field contribution study (Table 4) | `data/final/all_professors_embedded.json` (**not in repo**), `data/final/relevance_labels.json` |

---

## What requires a one-time model download (~90 MB)

```bash
pip install sentence-transformers
python analysis/embed_queries.py   # downloads all-MiniLM-L6-v2 once
python analysis/multi_query_eval.py  # now includes semantic for Q2-Q5
```

This generates `data/final/query_embeddings.json` and unlocks semantic,
hybrid, and reranked scores for Q2–Q5 in the multi-query evaluation.

---

## What requires re-scraping (hours, network dependent)

Re-scraping faculty pages is time-dependent — pages change and some
universities use JavaScript rendering (Playwright required).
The archived corpus (`data/final/all_professors.json`, 768 professors,
April–May 2026) is the authoritative dataset for this paper.

```bash
# Only needed to extend the corpus or re-scrape from scratch
python crawler/<university>_scraper.py
python data/merge.py
python embeddings/embed.py          # re-embed after scraping
python matcher/compare.py           # re-run comparison
```

---

## Key versioning information

| Component | Version / Identifier |
|-----------|---------------------|
| Embedding model | `all-MiniLM-L6-v2` (sentence-transformers) |
| BM25 | `rank-bm25` library, k1=1.5, b=0.75 |
| Scraping date | April–May 2026 |
| arXiv fetch date | April–May 2026 |
| Python | 3.9+ |
| Bootstrap seed (significance.py) | 42 (resampling), 0 (random baseline permutations) |
| Bootstrap seed (multi_query_eval.py) | 42 (topic-level resampling) |

---

## Label files

| File | Description | Entries |
|------|-------------|---------|
| `data/final/relevance_labels.json` | Q1 graded labels (grade 0/1/2) | 67 |
| `data/final/multi_query_labels.json` | Q2–Q5 graded labels | 95 (Q2/Q3/Q4: 25 each, Q5: 20) |

Both files are committed to the repository. The `grade` field (0/1/2) is
used for NDCG computation; `relevant` (True/False, grade ≥ 1) is used
for Precision and Recall.

---

## Hybrid weight provenance

Weights are module-level constants in `matcher/compare.py`, set before
label collection began. The block comment above `HYBRID_ALPHA` in
`compare.py` documents the a-priori rationale and confirms the weights
were not tuned on the collected labels.

---

## Generating the embedded professors file (required for ablation.py)

`analysis/ablation.py` requires `data/final/all_professors_embedded.json`,
which is **not committed to the repository** because it contains 768 × 384
float vectors (~10 MB). Generate it by running the embedding pipeline:

```bash
# Requires sentence-transformers and HuggingFace access (~90 MB download)
pip install sentence-transformers
python embeddings/embed.py
```

This produces `data/final/all_professors_embedded.json` (and
`data/final/comparison_results.json` if the relevance label file exists).
After generating it, `analysis/ablation.py` can be run to reproduce
Table 4 in the paper.

The committed `data/final/ablation_results.json` is the authoritative
output from the original run and correctly matches the paper's reported
values (Full Model NDCG=0.5934, −Biography=0.4632, etc.).

## Known non-reproducible elements

- **Faculty page content**: pages change continuously; re-scraping will
  yield a different corpus than the one used in the paper.
- **arXiv paper set**: new papers added daily; re-fetching will yield
  different paper counts per professor.
- **Model weights**: `all-MiniLM-L6-v2` is pinned by name; if the model
  card is updated on HuggingFace, cosine similarities may differ by a
  small epsilon. Use the cached model from first download.
