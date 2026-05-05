# Compares all 6 retrieval methods side-by-side and computes
# NDCG, MAP, precision, recall at k=10 using manual relevance labels.
import json
import re
import os
import math
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# Paths
EMBEDDED_FILE   = "data/final/all_professors_embedded.json"
OUTPUT_FILE     = "data/final/comparison_results.json"
LABELS_FILE     = "data/final/relevance_labels.json"   # optional manual labels
TOP_K           = 10

# Hybrid weight rationale
# HYBRID_ALPHA is the weight assigned to the TF-IDF component; the semantic
# component receives (1 - HYBRID_ALPHA) = 0.65.
#
# These weights were set A PRIORI, before any relevance labels existed,
# based on the established finding in the hybrid retrieval literature that
# dense semantic signals should dominate over sparse lexical signals for
# semantically rich queries (Lin & Ma, 2021; Luan et al., 2021).  The
# 0.35 / 0.65 split reflects a mild lexical contribution to preserve
# exact-term recall for rare tokens while letting the semantic encoder
# handle conceptual similarity.
#
# The weights were NOT tuned on the relevance labels collected in
# data/final/relevance_labels.json.  Label collection began after the
# retrieval pipeline was frozen.  This can be verified by checking the
# git history: the labels file post-dates the compare.py constants.
HYBRID_ALPHA       = 0.35   # TF-IDF weight; semantic weight = 1 - HYBRID_ALPHA = 0.65

# Reranked weights: BM25 (0.15) + TF-IDF (0.20) + Semantic (0.65).
# Same a-priori rationale: semantic dominates, BM25 adds length-normalised
# lexical signal, TF-IDF adds corpus-level IDF discrimination.
RERANKED_W_BM25     = 0.15
RERANKED_W_TFIDF    = 0.20
RERANKED_W_SEMANTIC = 0.65

os.makedirs("data/final", exist_ok=True)

# Keyword Scoring Methods

def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, remove stopwords."""
    STOPWORDS = {
        "a", "an", "the", "and", "or", "of", "in", "to", "for", "on",
        "with", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "i", "we", "my", "that",
        "this", "it", "its", "at", "by", "from", "as", "but", "not",
        "their", "they", "them", "he", "she", "his", "her", "who",
        "which", "what", "when", "where", "how", "all", "each", "every",
        "both", "few", "more", "most", "other", "some", "such", "no",
        "than", "too", "very", "can", "just", "about", "also", "into",
        "over", "after", "before", "between", "through", "during",
        "work", "want", "using", "systems", "based", "new",
    }
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def jaccard_score(query_text: str, prof_text: str) -> float:
    """
    Jaccard similarity: |intersection| / |union| of token sets.
    Simple but effective for exact term overlap.
    """
    q_tokens = set(_tokenize(query_text))
    p_tokens = set(_tokenize(prof_text))
    if not q_tokens or not p_tokens:
        return 0.0
    intersection = q_tokens & p_tokens
    union = q_tokens | p_tokens
    return len(intersection) / len(union)

def _build_tfidf(documents: list[str]) -> tuple[list[dict], dict]:
    """
    Build TF-IDF vectors for a list of documents.
    Returns (tf_idf_vectors, idf_dict).
    """
    # Document frequency
    N = len(documents)
    df = Counter()
    doc_tokens = []

    for doc in documents:
        tokens = _tokenize(doc)
        doc_tokens.append(tokens)
        unique = set(tokens)
        for t in unique:
            df[t] += 1

    # IDF: log(N / df) + 1 (smoothed)
    idf = {}
    for term, freq in df.items():
        idf[term] = math.log(N / (freq + 1)) + 1

    # TF-IDF vectors (as dicts for sparse representation)
    tfidf_vectors = []
    for tokens in doc_tokens:
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        vec = {}
        for term, count in tf.items():
            tf_val = count / total
            vec[term] = tf_val * idf.get(term, 1.0)
        tfidf_vectors.append(vec)

    return tfidf_vectors, idf

def _sparse_cosine(vec_a: dict, vec_b: dict) -> float:
    """Cosine similarity between two sparse vectors (dicts)."""
    # Dot product
    common_keys = set(vec_a.keys()) & set(vec_b.keys())
    if not common_keys:
        return 0.0

    dot = sum(vec_a[k] * vec_b[k] for k in common_keys)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

def tfidf_scores(query_text: str, prof_texts: list[str]) -> list[float]:
    """
    Compute TF-IDF cosine similarity between query and each professor.
    The query is added as the first document for IDF computation.
    """
    all_docs = [query_text] + prof_texts
    vectors, _ = _build_tfidf(all_docs)

    query_vec = vectors[0]
    scores = []
    for prof_vec in vectors[1:]:
        scores.append(_sparse_cosine(query_vec, prof_vec))
    return scores

def bm25_scores(query_text: str, prof_texts: list[str]) -> list[float]:
    """
    Compute BM25 (Okapi) scores — the gold-standard keyword retrieval baseline.
    Uses k1=1.5, b=0.75 (standard parameters).
    BM25 improves on TF-IDF by:
      - Saturating term frequency (diminishing returns for repeated terms)
      - Normalizing for document length
    """
    # Tokenize all professor documents
    tokenized_corpus = [_tokenize(t) for t in prof_texts]
    query_tokens = _tokenize(query_text)

    bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
    scores = bm25.get_scores(query_tokens)
    return scores.tolist()

# Comparison Engine

def _build_prof_text(prof: dict) -> str:
    """
    Build the full searchable text for a professor.
    Same logic as embed.py's build_embedding_text for fair comparison.
    """
    parts = []
    if prof.get("research_text"):
        parts.append(prof["research_text"])
    if prof.get("biography"):
        parts.append(prof["biography"][:500])
    return " | ".join(parts)

def run_comparison(data: dict) -> dict:
    """
    Run all matching methods and produce comparison results.
    """
    my_interest = data["my_interest"]
    professors  = data["professors"]

    print(f"[COMPARE] Running comparison on {len(professors)} professors")
    print(f"[COMPARE] Your interest: {my_interest[:100]}...\n")

    # Build text corpus
    prof_texts = [_build_prof_text(p) for p in professors]

    # 1. Jaccard Scores
    print("[METHOD 1] Jaccard keyword overlap...")
    jaccard_scores = [jaccard_score(my_interest, t) for t in prof_texts]

    # 2. TF-IDF Scores
    print("[METHOD 2] TF-IDF cosine similarity...")
    tfidf_score_list = tfidf_scores(my_interest, prof_texts)

    # 3. Semantic Scores
    print("[METHOD 3] Semantic embedding (pre-computed)...")
    my_embedding = np.array(data["my_embedding"]).reshape(1, -1)
    prof_embeddings = np.array([p["embedding"] for p in professors])
    semantic_scores = cosine_similarity(my_embedding, prof_embeddings)[0]

    # 4. Hybrid Scores
    print("[METHOD 4] Hybrid (TF-IDF + Semantic)...")
    tfidf_max = max(tfidf_score_list) if max(tfidf_score_list) > 0 else 1.0
    tfidf_normalized = [s / tfidf_max for s in tfidf_score_list]
    semantic_max = max(semantic_scores) if max(semantic_scores) > 0 else 1.0
    semantic_normalized = [s / semantic_max for s in semantic_scores]
    hybrid_scores = [
        HYBRID_ALPHA * tfidf_normalized[i] + (1 - HYBRID_ALPHA) * semantic_normalized[i]
        for i in range(len(professors))
    ]

    # 5. BM25 Scores
    print("[METHOD 5] BM25 (Okapi, k1=1.5, b=0.75)...")
    bm25_score_list = bm25_scores(my_interest, prof_texts)

    # 6. Reranked (3-way weighted score fusion)
    print("[METHOD 6] Reranked (BM25 + TF-IDF + Semantic weighted fusion)...")
    # NOTE: despite the variable name "reranked", this is NOT a cross-encoder
    # model. It is a linear combination of three pre-computed scores
    # (BM25, TF-IDF, Semantic), identical in spirit to Hybrid but with a
    # third component. A true cross-encoder would jointly encode the query
    # and each document through a single transformer pass -- that is not
    # implemented here. See RERANKED_W_* constants above for the weights.
    bm25_max = max(bm25_score_list) if max(bm25_score_list) > 0 else 1.0
    bm25_normalized = [s / bm25_max for s in bm25_score_list]
    reranked_scores = [
        RERANKED_W_BM25 * bm25_normalized[i]
        + RERANKED_W_TFIDF * tfidf_normalized[i]
        + RERANKED_W_SEMANTIC * semantic_normalized[i]
        for i in range(len(professors))
    ]

    # Build unified results
    results = []
    for i, prof in enumerate(professors):
        results.append({
            "name"           : prof["name"],
            "university"     : prof["university"],
            "title"          : prof.get("title", ""),
            "email"          : prof.get("email", ""),
            "research_text"  : prof.get("research_text", ""),
            "profile_url"    : prof.get("profile_url", ""),
            "jaccard_score"  : round(float(jaccard_scores[i]), 4),
            "tfidf_score"    : round(float(tfidf_score_list[i]), 4),
            "semantic_score" : round(float(semantic_scores[i]), 4),
            "hybrid_score"   : round(float(hybrid_scores[i]), 4),
            "bm25_score"     : round(float(bm25_score_list[i]), 4),
            "reranked_score" : round(float(reranked_scores[i]), 4),
        })

    # Rank by each method
    by_jaccard  = sorted(results, key=lambda x: x["jaccard_score"],  reverse=True)
    by_tfidf    = sorted(results, key=lambda x: x["tfidf_score"],    reverse=True)
    by_semantic = sorted(results, key=lambda x: x["semantic_score"], reverse=True)
    by_hybrid   = sorted(results, key=lambda x: x["hybrid_score"],   reverse=True)
    by_bm25     = sorted(results, key=lambda x: x["bm25_score"],     reverse=True)
    by_reranked = sorted(results, key=lambda x: x["reranked_score"], reverse=True)

    # Assign ranks
    for ranking, key in [
        (by_jaccard,  "jaccard_rank"),
        (by_tfidf,    "tfidf_rank"),
        (by_semantic, "semantic_rank"),
        (by_hybrid,   "hybrid_rank"),
        (by_bm25,     "bm25_rank"),
        (by_reranked, "reranked_rank"),
    ]:
        for i, r in enumerate(ranking):
            r[key] = i + 1

    # Top-K Analysis
    top_k_jaccard  = set(r["name"] for r in by_jaccard[:TOP_K])
    top_k_tfidf    = set(r["name"] for r in by_tfidf[:TOP_K])
    top_k_semantic = set(r["name"] for r in by_semantic[:TOP_K])
    top_k_hybrid   = set(r["name"] for r in by_hybrid[:TOP_K])
    top_k_bm25     = set(r["name"] for r in by_bm25[:TOP_K])
    top_k_reranked = set(r["name"] for r in by_reranked[:TOP_K])

    overlap_jaccard_semantic = top_k_jaccard & top_k_semantic
    overlap_tfidf_semantic   = top_k_tfidf & top_k_semantic
    overlap_keyword_hybrid   = (top_k_tfidf | top_k_jaccard) & top_k_hybrid
    overlap_bm25_semantic    = top_k_bm25 & top_k_semantic
    overlap_bm25_tfidf       = top_k_bm25 & top_k_tfidf

    # Unique Discoveries (found by one method but not the other)
    semantic_only_vs_tfidf = top_k_semantic - top_k_tfidf
    tfidf_only_vs_semantic = top_k_tfidf - top_k_semantic
    semantic_only_vs_jaccard = top_k_semantic - top_k_jaccard
    jaccard_only_vs_semantic = top_k_jaccard - top_k_semantic
    bm25_only_vs_tfidf = top_k_bm25 - top_k_tfidf
    tfidf_only_vs_bm25 = top_k_tfidf - top_k_bm25

    # Score Statistics
    def score_stats(scores_list):
        arr = np.array(scores_list)
        return {
            "mean"  : round(float(np.mean(arr)), 4),
            "std"   : round(float(np.std(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "min"   : round(float(np.min(arr)), 4),
            "max"   : round(float(np.max(arr)), 4),
        }

    # Rank Correlation (Spearman)
    def spearman_correlation(ranks_a, ranks_b):
        n = len(ranks_a)
        d_squared = sum((a - b) ** 2 for a, b in zip(ranks_a, ranks_b))
        return round(1 - (6 * d_squared) / (n * (n ** 2 - 1)), 4)

    # Get ranks in same order (by original index)
    semantic_ranks = [r["semantic_rank"] for r in results]
    tfidf_ranks    = [r["tfidf_rank"] for r in results]
    jaccard_ranks  = [r["jaccard_rank"] for r in results]
    hybrid_ranks   = [r["hybrid_rank"] for r in results]

    # Precision/Recall with manual labels (if available)
    precision_recall = None
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            labels = json.load(f)

        # Only use labels if at least some have been filled in (not all null)
        filled = [e for e in labels if e.get("relevant") is not None]
        if not filled:
            print("\n[LABELS] Template exists but not yet filled in - skipping precision/recall")
            print("[LABELS] Edit data/final/relevance_labels.json and set 'relevant' to true/false")
        else:
            print(f"\n[LABELS] Loading {len(filled)} manual relevance labels...")

        relevant_names = set(
            entry["name"] for entry in labels
            if entry.get("relevant") is True
        )

        if relevant_names:
            precision_recall = {}

            # Build name->relevance map for NDCG (graded relevance)
            relevance_map = {}
            missing_grade_count = 0
            for entry in labels:
                if entry.get("relevant") is True:
                    if "grade" not in entry:
                        missing_grade_count += 1
                    # Default to 1 (partially relevant) rather than 2 (highly
                    # relevant) if grade is missing -- a missing grade should
                    # never silently produce the maximum relevance score.
                    relevance_map[entry["name"]] = entry.get("grade", 1)
                elif entry.get("relevant") is False:
                    relevance_map[entry["name"]] = 0
            if missing_grade_count:
                print(f"[WARN] {missing_grade_count} relevant=True label(s) missing "
                      f"'grade' field; defaulted to grade=1 (partially relevant). "
                      f"Add explicit grades to avoid this default.")

            for method_name, method_key, by_list in [
                ("jaccard",  "jaccard_score",  by_jaccard),
                ("tfidf",    "tfidf_score",    by_tfidf),
                ("semantic", "semantic_score", by_semantic),
                ("hybrid",   "hybrid_score",   by_hybrid),
                ("bm25",     "bm25_score",     by_bm25),
                ("reranked", "reranked_score", by_reranked),
            ]:
                top_k_names = set(r["name"] for r in by_list[:TOP_K])
                tp = len(top_k_names & relevant_names)
                precision = tp / TOP_K if TOP_K > 0 else 0
                recall = tp / len(relevant_names) if relevant_names else 0
                f1 = (2 * precision * recall / (precision + recall)
                       if (precision + recall) > 0 else 0)

                # NDCG@k
                dcg = 0.0
                for rank_i, r in enumerate(by_list[:TOP_K]):
                    rel = relevance_map.get(r["name"], 0)
                    dcg += rel / math.log2(rank_i + 2)  # rank is 1-indexed
                # Ideal DCG: sort all relevances descending
                ideal_rels = sorted(relevance_map.values(), reverse=True)[:TOP_K]
                idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
                ndcg = dcg / idcg if idcg > 0 else 0

                # MAP@k (Average Precision)
                ap = 0.0
                tp_count = 0
                for rank_i, r in enumerate(by_list[:TOP_K]):
                    if r["name"] in relevant_names:
                        tp_count += 1
                        ap += tp_count / (rank_i + 1)
                ap = ap / min(len(relevant_names), TOP_K) if relevant_names else 0

                precision_recall[method_name] = {
                    "precision_at_k": round(precision, 4),
                    "recall_at_k"   : round(recall, 4),
                    "f1_at_k"       : round(f1, 4),
                    "ndcg_at_k"     : round(ndcg, 4),
                    "map_at_k"      : round(ap, 4),
                    "true_positives": tp,
                }
            print(f"[LABELS] Computed precision/recall/NDCG/MAP for {len(relevant_names)} relevant professors")
    # Build output
    output = {
        "config": {
            "top_k"            : TOP_K,
            "hybrid_alpha"     : HYBRID_ALPHA,
            "model"            : "all-MiniLM-L6-v2",
            "total_professors" : len(professors),
            "my_interest"      : my_interest.strip(),
            # Weight provenance — recorded here so every results file is self-documenting.
            # All weights were set a priori before label collection; see matcher/compare.py.
            "weight_provenance": "a_priori",
            "hybrid_weights"   : {
                "tfidf"   : HYBRID_ALPHA,
                "semantic": round(1 - HYBRID_ALPHA, 2),
            },
            "reranked_weights" : {
                "bm25"    : RERANKED_W_BM25,
                "tfidf"   : RERANKED_W_TFIDF,
                "semantic": RERANKED_W_SEMANTIC,
            },
        },
        "top_k_results": {
            "jaccard" : [
                {"rank": i+1, "name": r["name"], "university": r["university"],
                 "score": r["jaccard_score"], "research": r["research_text"][:150]}
                for i, r in enumerate(by_jaccard[:TOP_K])
            ],
            "tfidf": [
                {"rank": i+1, "name": r["name"], "university": r["university"],
                 "score": r["tfidf_score"], "research": r["research_text"][:150]}
                for i, r in enumerate(by_tfidf[:TOP_K])
            ],
            "semantic": [
                {"rank": i+1, "name": r["name"], "university": r["university"],
                 "score": r["semantic_score"], "research": r["research_text"][:150]}
                for i, r in enumerate(by_semantic[:TOP_K])
            ],
            "hybrid": [
                {"rank": i+1, "name": r["name"], "university": r["university"],
                 "score": r["hybrid_score"], "research": r["research_text"][:150]}
                for i, r in enumerate(by_hybrid[:TOP_K])
            ],
            "bm25": [
                {"rank": i+1, "name": r["name"], "university": r["university"],
                 "score": r["bm25_score"], "research": r["research_text"][:150]}
                for i, r in enumerate(by_bm25[:TOP_K])
            ],
            "reranked": [
                {"rank": i+1, "name": r["name"], "university": r["university"],
                 "score": r["reranked_score"], "research": r["research_text"][:150]}
                for i, r in enumerate(by_reranked[:TOP_K])
            ],
        },
        "overlap_analysis": {
            "jaccard_vs_semantic": {
                "overlap_count"    : len(overlap_jaccard_semantic),
                "overlap_names"    : sorted(overlap_jaccard_semantic),
                "jaccard_only"     : sorted(jaccard_only_vs_semantic),
                "semantic_only"    : sorted(semantic_only_vs_jaccard),
            },
            "tfidf_vs_semantic": {
                "overlap_count"    : len(overlap_tfidf_semantic),
                "overlap_names"    : sorted(overlap_tfidf_semantic),
                "tfidf_only"       : sorted(tfidf_only_vs_semantic),
                "semantic_only"    : sorted(semantic_only_vs_tfidf),
            },
            "keyword_union_vs_hybrid": {
                "overlap_count"    : len(overlap_keyword_hybrid),
                "overlap_names"    : sorted(overlap_keyword_hybrid),
            },
        },
        "score_statistics": {
            "jaccard"  : score_stats(jaccard_scores),
            "tfidf"    : score_stats(tfidf_score_list),
            "semantic" : score_stats(semantic_scores.tolist()),
            "hybrid"   : score_stats(hybrid_scores),
            "bm25"     : score_stats(bm25_score_list),
            "reranked" : score_stats(reranked_scores),
        },
        "rank_correlations": {
            "tfidf_vs_semantic"  : spearman_correlation(tfidf_ranks, semantic_ranks),
            "jaccard_vs_semantic": spearman_correlation(jaccard_ranks, semantic_ranks),
            "hybrid_vs_semantic" : spearman_correlation(hybrid_ranks, semantic_ranks),
            "tfidf_vs_jaccard"   : spearman_correlation(tfidf_ranks, jaccard_ranks),
            "bm25_vs_semantic"   : spearman_correlation([r["bm25_rank"] for r in results], semantic_ranks),
            "bm25_vs_tfidf"      : spearman_correlation([r["bm25_rank"] for r in results], tfidf_ranks),
        },
        "all_ranked": sorted(results, key=lambda x: x["reranked_score"], reverse=True),
    }

    if precision_recall:
        output["precision_recall"] = precision_recall

    return output

# Summary / Finding Generator

def generate_finding(output: dict) -> str:
    """
    Generate a human-readable research finding from the comparison results.
    This is the ~300-word summary that goes into your README and SOP.
    """
    cfg     = output["config"]
    overlap = output["overlap_analysis"]
    stats   = output["score_statistics"]
    corr    = output["rank_correlations"]
    top_k   = output["top_k_results"]
    K       = cfg["top_k"]

    tfidf_vs_sem = overlap["tfidf_vs_semantic"]
    jacc_vs_sem  = overlap["jaccard_vs_semantic"]

    lines = []
    lines.append("=" * 60)
    lines.append("RESEARCH FINDING: Keyword vs Semantic Matching")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Dataset: {cfg['total_professors']} CS professors across {len(set(r.get('university','') for r in output.get('all_ranked',[])))} universities")
    lines.append(f"Embedding model: {cfg['model']}")
    lines.append(f"Analysis: Top-{K} comparison\n")

    # -- Top-K overlap --
    lines.append(f"-- Top-{K} Overlap --")
    lines.append(f"  TF-IDF vs Semantic:  {tfidf_vs_sem['overlap_count']}/{K} shared")
    lines.append(f"  Jaccard vs Semantic: {jacc_vs_sem['overlap_count']}/{K} shared")
    lines.append("")

    # -- Unique discoveries --
    lines.append(f"-- Unique Discoveries (TF-IDF vs Semantic) --")
    if tfidf_vs_sem["tfidf_only"]:
        lines.append(f"  Found ONLY by TF-IDF ({len(tfidf_vs_sem['tfidf_only'])}):")
        for name in tfidf_vs_sem["tfidf_only"]:
            lines.append(f"    - {name}")
    if tfidf_vs_sem["semantic_only"]:
        lines.append(f"  Found ONLY by Semantic ({len(tfidf_vs_sem['semantic_only'])}):")
        for name in tfidf_vs_sem["semantic_only"]:
            lines.append(f"    - {name}")
    lines.append("")

    # -- Score distributions --
    lines.append(f"-- Score Distributions --")
    for method in ["jaccard", "tfidf", "bm25", "semantic", "hybrid", "reranked"]:
        s = stats[method]
        lines.append(f"  {method:10s}: mean={s['mean']:.4f}  std={s['std']:.4f}  "
                      f"range=[{s['min']:.4f}, {s['max']:.4f}]")
    lines.append("")

    # -- Rank correlation --
    lines.append(f"-- Rank Correlations (Spearman) --")
    lines.append(f"  TF-IDF <-> Semantic:   rho = {corr['tfidf_vs_semantic']}")
    lines.append(f"  Jaccard <-> Semantic:  rho = {corr['jaccard_vs_semantic']}")
    lines.append(f"  BM25 <-> Semantic:     rho = {corr.get('bm25_vs_semantic', 'N/A')}")
    lines.append(f"  BM25 <-> TF-IDF:       rho = {corr.get('bm25_vs_tfidf', 'N/A')}")
    lines.append(f"  Hybrid <-> Semantic:   rho = {corr['hybrid_vs_semantic']}")
    lines.append("")

    # -- Side-by-side top-5 --
    lines.append(f"-- Side-by-Side Top 5 --")
    lines.append(f"  {'Rank':<5} {'BM25':<25} {'TF-IDF':<25} {'Semantic':<25} {'Reranked':<25}")
    lines.append(f"  {'-'*5} {'-'*25} {'-'*25} {'-'*25} {'-'*25}")
    for i in range(min(5, K)):
        b = top_k["bm25"][i]
        t = top_k["tfidf"][i]
        s = top_k["semantic"][i]
        r = top_k["reranked"][i]
        lines.append(
            f"  {i+1:<5} "
            f"{b['name'][:23]:<25} "
            f"{t['name'][:23]:<25} "
            f"{s['name'][:23]:<25} "
            f"{r['name'][:23]:<25}"
        )
    lines.append("")

    # -- Precision/Recall if available --
    if "precision_recall" in output:
        pr = output["precision_recall"]
        lines.append(f"-- Precision / Recall / NDCG / MAP @ {K} (Manual Labels) --")
        lines.append(f"  {'Method':<10} {'Precision':<12} {'Recall':<10} {'F1':<10} {'NDCG':<10} {'MAP':<10}")
        for method in ["jaccard", "tfidf", "bm25", "semantic", "hybrid", "reranked"]:
            m = pr[method]
            lines.append(f"  {method:<10} {m['precision_at_k']:<12} "
                          f"{m['recall_at_k']:<10} {m['f1_at_k']:<10} "
                          f"{m.get('ndcg_at_k', 'N/A'):<10} {m.get('map_at_k', 'N/A'):<10}")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)

# Generate Relevance Labels Template

def generate_labels_template(data: dict) -> None:
    """
    Generate a template JSON file for manual relevance labeling.
    The user reviews the top-30 professors and marks them as relevant or not.
    """
    if os.path.exists(LABELS_FILE):
        print(f"[LABELS] Template already exists: {LABELS_FILE}")
        return

    my_embedding = np.array(data["my_embedding"]).reshape(1, -1)
    prof_embeddings = np.array([p["embedding"] for p in data["professors"]])
    scores = cosine_similarity(my_embedding, prof_embeddings)[0]

    # Take top-30 by semantic score
    indexed = [(i, scores[i]) for i in range(len(scores))]
    indexed.sort(key=lambda x: x[1], reverse=True)

    template = []
    for rank, (idx, score) in enumerate(indexed[:30], 1):
        prof = data["professors"][idx]
        template.append({
            "rank"          : rank,
            "name"          : prof["name"],
            "university"    : prof["university"],
            "research_text" : prof.get("research_text", ""),
            "semantic_score": round(float(score), 4),
            "relevant"      : None,  # ← USER FILLS THIS: true or false
            "notes"         : "",    # ← optional notes
        })

    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    print(f"\n[LABELS] Template saved to: {LABELS_FILE}")
    print(f"[LABELS] Please review 30 professors and set 'relevant' to true/false")
    print(f"[LABELS] Then re-run compare.py for precision/recall metrics\n")

# Main

def main():
    # Load data
    print("[LOAD] Loading embedded professor data...")
    with open(EMBEDDED_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[INFO] {len(data['professors'])} professors loaded\n")

    # Generate labels template (first run only)
    generate_labels_template(data)

    # Run comparison
    output = run_comparison(data)

    # Save results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] Comparison results -> {OUTPUT_FILE}")

    # Print the finding
    finding = generate_finding(output)
    print("\n" + finding)

    # Also save the finding as plain text
    finding_path = "data/final/research_finding.txt"
    with open(finding_path, "w", encoding="utf-8") as f:
        f.write(finding)
    print(f"\n[SAVED] Research finding -> {finding_path}")

if __name__ == "__main__":
    main()
