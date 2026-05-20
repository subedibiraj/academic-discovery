# Evaluates all 6 methods across 5 queries with bootstrap significance.
#
# Multi-query evaluation for the academic advisor discovery system.
#
# PURPOSE

# Single-query evaluation (n=1 topic) cannot reliably rank IR systems
# (Voorhees & Buckley, 2002).  This module adds 4 additional queries,
# giving a 5-query evaluation set.
#
# RETRIEVAL SCORES FOR Q2-Q5

# Semantic scores for Q2-Q5 require running all-MiniLM-L6-v2 locally
# (HuggingFace download, ~90MB).  Run:
#   python analysis/embed_queries.py   # generates data/final/query_embeddings.json
# before running this script for full semantic results.
#
# When query_embeddings.json is absent, this script computes Jaccard,
# TF-IDF, and BM25 scores from professor profile texts (no model needed)
# and marks semantic/hybrid/reranked as N/A for Q2-Q5.  The keyword
# method comparison across 5 queries is still valid.
#
# QUERY DESIGN

# Queries Q2-Q5 reflect realistic CS graduate applicant research interest
# statements, grounded in (a) the corpus research area distribution and
# (b) common profiles on r/gradadmissions and r/MachineLearning
# (paraphrased and generalized; not copied verbatim from any post).
#
import json
import math
import os
import re
import numpy as np
from collections import Counter

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("[WARN] rank_bm25 not installed — BM25 scores will be 0")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] sklearn not installed — TF-IDF scores will be 0")

PROFESSORS_FILE    = "data/final/all_professors.json"
COMPARE_FILE       = "data/final/comparison_results.json"
LABELS_FILE        = "data/final/relevance_labels.json"
MQ_LABELS_FILE     = "data/final/multi_query_labels.json"
QUERY_EMBED_FILE   = "data/final/query_embeddings.json"   # optional
OUTPUT_FILE        = "data/final/multi_query_results.json"
TOP_K              = 10
N_BOOTSTRAP        = 10_000
HYBRID_ALPHA       = 0.35   # TF-IDF weight; semantic weight = 1 - HYBRID_ALPHA. Must match matcher/compare.py.
RERANKED_W_BM25     = 0.15  # Must match matcher/compare.py RERANKED_W_BM25.
RERANKED_W_TFIDF    = 0.20  # Must match matcher/compare.py RERANKED_W_TFIDF.
RERANKED_W_SEMANTIC = 0.65  # Must match matcher/compare.py RERANKED_W_SEMANTIC.

os.makedirs("data/final", exist_ok=True)

# QUERY DEFINITIONS
QUERIES = {
    "Q1_web_ir_nlp": {
        "persona": "Web IR & NLP Engineer",
        "source":  "Author's own research interest (original Q1 query — embed.py version)",
        "text": (
            "I am interested in large-scale web data collection and information extraction "
            "systems, web crawling and browser automation for structured data acquisition, "
            "data engineering pipelines for cleaning and transforming heterogeneous data "
            "formats (JSON, CSV, Parquet). I have experience applying NLP and machine "
            "learning to real-world problems including speech recognition with Whisper "
            "for low-resource languages and computer vision for image colorization. "
            "I want to work on intelligent systems that automatically discover, extract, "
            "and organize knowledge from unstructured web and text data, with applications "
            "in information retrieval, natural language processing, and applied machine "
            "learning for data science."
        ),
    },
    "Q2_cv_robotics": {
        "persona": "Computer Vision & Robot Learning",
        "source":  (
            "Composite of r/gradadmissions profiles: applicants describing "
            "interests in visual perception, robot manipulation, and embodied AI"
        ),
        "text": (
            "I want to work on computer vision and robot learning — specifically "
            "teaching robots to perceive and interact with their physical environment "
            "using visual inputs. My interests include deep learning for 3D scene "
            "understanding, object detection and tracking, robotic manipulation and "
            "grasping, reinforcement learning for continuous control, and building "
            "autonomous agents that can generalize across environments. I am drawn to "
            "labs that combine perception and action in real physical systems rather "
            "than purely simulation."
        ),
    },
    "Q3_security_privacy": {
        "persona": "Systems Security & Applied Cryptography",
        "source":  (
            "Composite of r/netsec and r/gradadmissions profiles: applicants "
            "describing interests in offensive/defensive security and "
            "privacy-preserving systems"
        ),
        "text": (
            "My research interests are in systems security and applied cryptography. "
            "I am interested in understanding and defending against real-world "
            "attacks on networked systems, including vulnerability discovery, malware "
            "analysis, and network intrusion detection. On the cryptographic side I "
            "am drawn to privacy-preserving computation, secure multiparty protocols, "
            "and cryptographic techniques that can be deployed in practical systems. "
            "I am also interested in the security of mobile and IoT devices, and in "
            "using machine learning to detect anomalous behavior in large-scale networks."
        ),
    },
    "Q4_systems_distributed": {
        "persona": "Distributed Systems & Cloud Infrastructure",
        "source":  (
            "Composite of r/cscareerquestions and r/gradadmissions profiles: "
            "applicants with systems engineering background targeting distributed "
            "systems or cloud computing research"
        ),
        "text": (
            "I am interested in the design and implementation of large-scale "
            "distributed systems — the kind of infrastructure that powers cloud "
            "computing platforms at Google, Amazon, and Microsoft. My specific "
            "interests include distributed storage systems, consistency and "
            "replication protocols, fault tolerance and reliability in the presence "
            "of failures, operating system kernels and resource scheduling, and the "
            "intersection of systems software with machine learning workloads. I want "
            "to build systems that are fast, reliable, and easy for developers to "
            "reason about."
        ),
    },
    "Q5_bio_health": {
        "persona": "Computational Biology & Health Informatics",
        "source":  (
            "Composite of r/bioinformatics and r/gradadmissions profiles: "
            "applicants from biology/CS backgrounds targeting computational "
            "genomics or medical AI research"
        ),
        "text": (
            "I come from a background in both computer science and molecular "
            "biology and I want to apply machine learning and statistical methods "
            "to biological and medical problems. My interests include computational "
            "genomics, protein structure prediction, single-cell RNA sequencing "
            "analysis, and using deep learning to interpret clinical data and medical "
            "imaging. I am particularly drawn to research that connects algorithmic "
            "advances to real clinical impact — drug target discovery, precision "
            "medicine, and early disease detection using multi-modal patient data."
        ),
    },
}

# RELEVANCE LABELS  Q2-Q5
# Pooling: top-20 by semantic (Q1 run) + keyword matching; all graded 0/1/2.
MULTI_QUERY_LABELS = {
    "Q2_cv_robotics": [
        {"name": "Sergey Levine",         "grade": 2, "relevant": True},
        {"name": "Trevor Darrell",        "grade": 2, "relevant": True},
        {"name": "Stella Yu",             "grade": 2, "relevant": True},
        {"name": "Pieter Abbeel",         "grade": 2, "relevant": True},
        {"name": "Scott Niekum",          "grade": 2, "relevant": True},
        {"name": "Donghyun Kim",          "grade": 2, "relevant": True},
        {"name": "Vangelis Kalogerakis",  "grade": 2, "relevant": True},
        {"name": "Ruoshi Liu",            "grade": 2, "relevant": True},
        {"name": "Ruohan Gao",            "grade": 2, "relevant": True},
        {"name": "Roberto Martin-Martin", "grade": 2, "relevant": True},
        {"name": "Qixing Huang",          "grade": 2, "relevant": True},
        {"name": "Philipp Krähenbühl",    "grade": 2, "relevant": True},
        {"name": "Roger Eastman",         "grade": 2, "relevant": True},
        {"name": "Subhransu Maji",        "grade": 2, "relevant": True},
        {"name": "Hao Zhang",             "grade": 2, "relevant": True},
        {"name": "Peter Stone",           "grade": 1, "relevant": True},
        {"name": "Shawna Thomas",         "grade": 1, "relevant": True},
        {"name": "Bruno Castro da Silva", "grade": 1, "relevant": True},
        {"name": "Soheil Feizi",          "grade": 1, "relevant": True},
        {"name": "Tom Goldstein",         "grade": 1, "relevant": True},
        {"name": "Ali Ghodsi",            "grade": 0, "relevant": False},
        {"name": "Joseph Gonzalez",       "grade": 0, "relevant": False},
        {"name": "Stuart J. Russell",     "grade": 0, "relevant": False},
        {"name": "Tim Davis",             "grade": 0, "relevant": False},
        {"name": "Mingyuan Zhou",         "grade": 0, "relevant": False},
    ],
    "Q3_security_privacy": [
        {"name": "Bradley Reaves",        "grade": 2, "relevant": True},
        {"name": "Yizheng Chen",          "grade": 2, "relevant": True},
        {"name": "Nitesh Saxena",         "grade": 2, "relevant": True},
        {"name": "David Wagner",          "grade": 2, "relevant": True},
        {"name": "Amir Rahmati",          "grade": 2, "relevant": True},
        {"name": "Alessandra Scafuro",    "grade": 2, "relevant": True},
        {"name": "Raluca Ada Popa",       "grade": 2, "relevant": True},
        {"name": "Nick Nikiforakis",      "grade": 2, "relevant": True},
        {"name": "Radu Sion",             "grade": 2, "relevant": True},
        {"name": "William Enck",          "grade": 2, "relevant": True},
        {"name": "Wenjing Lou",           "grade": 2, "relevant": True},
        {"name": "Marina Blanton,PhD",    "grade": 2, "relevant": True},
        {"name": "Ian Miers",             "grade": 2, "relevant": True},
        {"name": "Pubali Datta",          "grade": 2, "relevant": True},
        {"name": "Marius Minea",          "grade": 2, "relevant": True},
        {"name": "Brent Waters",          "grade": 1, "relevant": True},
        {"name": "Michelle Mazurek",      "grade": 1, "relevant": True},
        {"name": "Jianqing Liu",          "grade": 1, "relevant": True},
        {"name": "Yanlai Wu",             "grade": 1, "relevant": True},
        {"name": "Wujie Wen",             "grade": 1, "relevant": True},
        {"name": "Pete Keleher",          "grade": 0, "relevant": False},
        {"name": "Michael Marsh",         "grade": 0, "relevant": False},
        {"name": "William Regli",         "grade": 0, "relevant": False},
        {"name": "Seth Nielson",          "grade": 0, "relevant": False},
        {"name": "Shravan Narayan",       "grade": 0, "relevant": False},
    ],
    "Q4_systems_distributed": [
        {"name": "Frank Mueller",         "grade": 2, "relevant": True},
        {"name": "Erez Zadok",            "grade": 2, "relevant": True},
        {"name": "Xiaohui (Helen) Gu",    "grade": 2, "relevant": True},
        {"name": "Jiajia Li",             "grade": 2, "relevant": True},
        {"name": "Ali Ghodsi",            "grade": 2, "relevant": True},
        {"name": "Ion Stoica",            "grade": 2, "relevant": True},
        {"name": "Matei Zaharia",         "grade": 2, "relevant": True},
        {"name": "Prashant Shenoy",       "grade": 2, "relevant": True},
        {"name": "Dilma Da Silva",        "grade": 2, "relevant": True},
        {"name": "Marco Serafini",        "grade": 2, "relevant": True},
        {"name": "Huaicheng Li",          "grade": 2, "relevant": True},
        {"name": "Ki Hwan Yum",           "grade": 2, "relevant": True},
        {"name": "Daniel Abadi",          "grade": 2, "relevant": True},
        {"name": "Xipeng (Gracen) Shen",  "grade": 2, "relevant": True},
        {"name": 'Eun Jung "EJ" Kim',     "grade": 2, "relevant": True},
        {"name": "Michael Ferdman",       "grade": 1, "relevant": True},
        {"name": "Michael Bender",        "grade": 1, "relevant": True},
        {"name": "Yuriy Brun",            "grade": 1, "relevant": True},
        {"name": "Man-Ki Yoon",           "grade": 1, "relevant": True},
        {"name": "Hyunyoung Lee",         "grade": 1, "relevant": True},
        {"name": "Tom Goldstein",         "grade": 0, "relevant": False},
        {"name": "Ranga Raju Vatsavai",   "grade": 0, "relevant": False},
        {"name": "Joseph Gonzalez",       "grade": 0, "relevant": False},
        {"name": "Kemafor Anyanwu Ogan",  "grade": 0, "relevant": False},
        {"name": "Ruozhou Yu",            "grade": 0, "relevant": False},
    ],
    "Q5_bio_health": [
        {"name": "Anna Green",            "grade": 2, "relevant": True},
        {"name": "Mihai Pop",             "grade": 2, "relevant": True},
        {"name": "Ilias Georgakopoulos-Soares", "grade": 2, "relevant": True},
        {"name": "Nilah Ioannidis",       "grade": 2, "relevant": True},
        {"name": "Irene Chen",            "grade": 2, "relevant": True},
        {"name": "Richard M. Karp",       "grade": 2, "relevant": True},
        {"name": "Hava Siegelmann",       "grade": 2, "relevant": True},
        {"name": "Steffen Heber",         "grade": 2, "relevant": True},
        {"name": "Zhaozheng Yin",         "grade": 2, "relevant": True},
        {"name": "Debswapna Bhattacharya","grade": 2, "relevant": True},
        {"name": "Jing Li",               "grade": 2, "relevant": True},
        {"name": "Lenore Cowen",          "grade": 2, "relevant": True},
        {"name": "Tamer Kahveci",         "grade": 1, "relevant": True},
        {"name": "Ruohan Gao",            "grade": 1, "relevant": True},
        {"name": "Vagelis Papalexakis",   "grade": 1, "relevant": True},
        {"name": "Pieter Abbeel",         "grade": 0, "relevant": False},
        {"name": "Anna Rumshisky",        "grade": 0, "relevant": False},
        {"name": "Hao Zhang",             "grade": 0, "relevant": False},
        {"name": "Sergey Levine",         "grade": 0, "relevant": False},
        {"name": "Stuart J. Russell",     "grade": 0, "relevant": False},
    ],
}

# Scoring helpers  (Jaccard, TF-IDF, BM25 — no model required)

def tokenize(text):
    """Mirror of matcher/compare.py _tokenize — must stay in sync."""
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

def jaccard(query_tokens, doc_tokens):
    q, d = set(query_tokens), set(doc_tokens)
    return len(q & d) / len(q | d) if (q | d) else 0.0

def build_embedding_text(prof):
    parts = []
    if prof.get("research_text"):
        parts.append(prof["research_text"])
    if prof.get("biography"):
        parts.append(prof["biography"][:500])
    return " | ".join(parts)

def score_all_methods(query_text, professors, prof_texts, query_id,
                      semantic_scores=None):
    """
    Returns dict name→{jaccard, tfidf, bm25, semantic, hybrid, reranked}.
    semantic_scores: optional dict name→float from precomputed embeddings.
    """
    q_tokens = tokenize(query_text)
    doc_tokens_list = [tokenize(t) for t in prof_texts]

    # Jaccard
    jac_scores = [jaccard(q_tokens, dt) for dt in doc_tokens_list]

    # TF-IDF
    if HAS_SKLEARN:
        vec = TfidfVectorizer(sublinear_tf=True, max_features=50000)
        corpus = prof_texts + [query_text]
        tfidf_mat = vec.fit_transform(corpus)
        tfidf_sim = sk_cosine(tfidf_mat[-1], tfidf_mat[:-1])[0]
        tfidf_scores = tfidf_sim.tolist()
    else:
        tfidf_scores = [0.0] * len(professors)

    # BM25
    if HAS_BM25:
        bm25 = BM25Okapi(doc_tokens_list, k1=1.5, b=0.75)
        bm25_raw = bm25.get_scores(q_tokens)
        bm25_scores = bm25_raw.tolist()
    else:
        bm25_scores = [0.0] * len(professors)

    # Normalize helpers
    def minmax(scores):
        mn, mx = min(scores), max(scores)
        if mx == mn:
            return [0.0] * len(scores)
        return [(s - mn) / (mx - mn) for s in scores]

    tfidf_n = minmax(tfidf_scores)
    sem_n   = minmax(semantic_scores) if semantic_scores else [0.0]*len(professors)
    bm25_n  = minmax(bm25_scores)

    hybrid_sc   = [HYBRID_ALPHA * tfidf_n[i] + (1 - HYBRID_ALPHA) * sem_n[i]
                   for i in range(len(professors))]
    reranked_sc = [RERANKED_W_BM25 * bm25_n[i] + RERANKED_W_TFIDF * tfidf_n[i]
                   + RERANKED_W_SEMANTIC * sem_n[i]
                   for i in range(len(professors))]

    results = {}
    for i, prof in enumerate(professors):
        results[prof["name"]] = {
            "jaccard":  jac_scores[i],
            "tfidf":    tfidf_scores[i],
            "bm25":     bm25_scores[i],
            "semantic": semantic_scores[i] if semantic_scores else None,
            "hybrid":   hybrid_sc[i]   if semantic_scores else None,
            "reranked": reranked_sc[i] if semantic_scores else None,
        }
    return results

# Metric helpers

def compute_ndcg(name_scores, relevance_map, k=TOP_K):
    # NOTE: must check `is not None` explicitly, not `x[1] or -1` --
    # a genuine score of 0.0 is falsy in Python and would be silently
    # treated as -1 (same as a missing/None score) under `or`. In this
    # codebase's actual call pattern this never changes results (see
    # evaluate_query_from_scores, which skips compute_ndcg entirely when
    # all scores for a method are None), but the explicit check is
    # correct regardless of how this function is called in the future.
    ranked = sorted(name_scores.items(),
                     key=lambda x: x[1] if x[1] is not None else -1,
                     reverse=True)
    dcg  = sum(relevance_map.get(n, 0) / math.log2(i + 2)
               for i, (n, _) in enumerate(ranked[:k]))
    ideal= sorted(relevance_map.values(), reverse=True)[:k]
    idcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

def compute_map(name_scores, relevant_set, k=TOP_K):
    ranked = sorted(name_scores.items(),
                     key=lambda x: x[1] if x[1] is not None else -1,
                     reverse=True)
    ap, tp = 0.0, 0
    for i, (n, _) in enumerate(ranked[:k]):
        if n in relevant_set:
            tp += 1
            ap += tp / (i + 1)
    return ap / min(len(relevant_set), k) if relevant_set else 0.0

def evaluate_query_from_scores(name_score_dict, label_list):
    """Evaluate all methods for one query given precomputed name→scores dict."""
    rel_map = {}
    rel_set = set()
    for entry in label_list:
        name  = entry["name"]
        grade = int(entry.get("grade", 0))
        if entry.get("relevant") is True:
            rel_map[name] = grade
            rel_set.add(name)
        else:
            rel_map[name] = 0

    methods = ["jaccard", "tfidf", "bm25", "semantic", "hybrid", "reranked"]
    results = {}
    for m in methods:
        m_scores = {n: v[m] for n, v in name_score_dict.items()}
        if all(v is None for v in m_scores.values()):
            results[m] = {"ndcg_at_k": None, "map_at_k": None}
        else:
            results[m] = {
                "ndcg_at_k": round(compute_ndcg(m_scores, rel_map), 4),
                "map_at_k":  round(compute_map(m_scores, rel_set), 4),
                "n_relevant": len(rel_set),
                "n_labelled": len(label_list),
            }
    return results

# Bootstrap

def bootstrap_multi_query(per_query_scores, n_bootstrap=N_BOOTSTRAP):
    methods   = ["jaccard", "tfidf", "bm25"]   # keyword-only (always available)
    query_ids = list(per_query_scores.keys())
    n_queries = len(query_ids)

    # Check if semantic available
    q1_sem = per_query_scores["Q1_web_ir_nlp"].get("semantic", {}).get("ndcg_at_k")
    if q1_sem is not None:
        methods = ["jaccard", "tfidf", "bm25", "semantic", "hybrid", "reranked"]

    # Topic-level stats
    topic_stats = {}
    for method in methods:
        scores = [per_query_scores[q][method]["ndcg_at_k"]
                  for q in query_ids
                  if per_query_scores[q][method]["ndcg_at_k"] is not None]
        # SAFETY: if fewer than n_queries data points are available (e.g.
        # semantic data exists for Q1 only, not Q2-Q5, because
        # query_embeddings.json was not generated), do NOT silently report
        # a partial mean labeled as if it were the full multi-query
        # aggregate -- that would misleadingly understate the topic count
        # behind the number. Null out the aggregate in that case instead.
        if scores and len(scores) < n_queries:
            print(f"[WARN] {method}: only {len(scores)}/{n_queries} queries have "
                  f"data; aggregate mean would be misleading -- reporting as "
                  f"incomplete (None) rather than a partial mean.")
            topic_stats[method] = {
                "mean_ndcg": None, "std_ndcg": None,
                "min_ndcg": None, "max_ndcg": None,
                "per_query": {q: per_query_scores[q][method]["ndcg_at_k"]
                              for q in query_ids},
                "incomplete": True,
                "n_available": len(scores),
            }
            continue
        topic_stats[method] = {
            "mean_ndcg":  round(float(np.mean(scores)), 4) if scores else None,
            "std_ndcg":   round(float(np.std(scores, ddof=1)), 4) if len(scores)>1 else None,
            "min_ndcg":   round(float(np.min(scores)), 4) if scores else None,
            "max_ndcg":   round(float(np.max(scores)), 4) if scores else None,
            "per_query":  {q: per_query_scores[q][method]["ndcg_at_k"] for q in query_ids},
        }

    # Topic bootstrap
    rng = np.random.RandomState(42)
    boot_ndcg = {m: [] for m in methods}
    for _ in range(n_bootstrap):
        sampled = rng.choice(query_ids, size=n_queries, replace=True)
        for method in methods:
            scores = [per_query_scores[q][method]["ndcg_at_k"]
                      for q in sampled
                      if per_query_scores[q][method]["ndcg_at_k"] is not None]
            boot_ndcg[method].append(float(np.mean(scores)) if scores else 0.0)

    ci_stats = {}
    for method in methods:
        arr = np.array(boot_ndcg[method])
        ci_stats[method] = {
            "ci_low":  round(float(np.percentile(arr, 2.5)), 4),
            "ci_high": round(float(np.percentile(arr, 97.5)), 4),
        }

    return topic_stats, ci_stats, boot_ndcg, methods

def paired_test(boot_a, boot_b):
    diffs = np.array(boot_a) - np.array(boot_b)
    md    = float(np.mean(diffs))
    p_raw = float(np.mean(diffs <= 0)) * 2 if md > 0 else float(np.mean(diffs >= 0)) * 2
    return md, min(p_raw, 1.0)

# Main

def main():
    print("=" * 70)
    print("  MULTI-QUERY EVALUATION  (5 queries, Q1-Q5)")
    print("=" * 70)

    # Load professors
    with open(PROFESSORS_FILE, "r", encoding="utf-8") as f:
        professors = json.load(f)
    prof_texts = [build_embedding_text(p) for p in professors]
    print(f"[DATA] {len(professors)} professors loaded")

    # Load Q1 scores from comparison_results (already computed with model)
    with open(COMPARE_FILE, "r", encoding="utf-8") as f:
        compare = json.load(f)
    q1_all_ranked = {r["name"]: r for r in compare["all_ranked"]}

    # Load optional precomputed query embeddings (semantic for Q2-Q5)
    query_semantic = {}
    if os.path.exists(QUERY_EMBED_FILE):
        with open(QUERY_EMBED_FILE, "r") as f:
            qe = json.load(f)
        query_semantic = qe.get("query_scores", {})
        print(f"[EMBED] Loaded semantic scores from {QUERY_EMBED_FILE}")
    else:
        print(f"[EMBED] {QUERY_EMBED_FILE} not found — "
              f"semantic/hybrid/reranked will be N/A for Q2-Q5")
        print(f"         Run: python analysis/embed_queries.py  to generate it")

    # Save multi-query label file
    with open(MQ_LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(MULTI_QUERY_LABELS, f, indent=2)

    # Load Q1 labels
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        q1_labels_raw = json.load(f)
    q1_labels = [l for l in q1_labels_raw if l.get("relevant") is not None]

    all_labels = {
        "Q1_web_ir_nlp":        q1_labels,
        "Q2_cv_robotics":       MULTI_QUERY_LABELS["Q2_cv_robotics"],
        "Q3_security_privacy":  MULTI_QUERY_LABELS["Q3_security_privacy"],
        "Q4_systems_distributed": MULTI_QUERY_LABELS["Q4_systems_distributed"],
        "Q5_bio_health":        MULTI_QUERY_LABELS["Q5_bio_health"],
    }

    # Per-query scoring & evaluation
    per_query_scores = {}

    for qid, query_info in QUERIES.items():
        query_text = query_info["text"]

        if qid == "Q1_web_ir_nlp":
            # Use pre-computed scores from comparison_results
            name_score_dict = {
                name: {
                    "jaccard":  r["jaccard_score"],
                    "tfidf":    r["tfidf_score"],
                    "bm25":     r["bm25_score"],
                    "semantic": r["semantic_score"],
                    "hybrid":   r["hybrid_score"],
                    "reranked": r["reranked_score"],
                }
                for name, r in q1_all_ranked.items()
            }
        else:
            # Compute keyword scores fresh; semantic from embed file if available
            sem_scores = None
            if qid in query_semantic:
                prof_name_order = [p["name"] for p in professors]
                sem_map  = query_semantic[qid]
                sem_scores = [sem_map.get(n, 0.0) for n in prof_name_order]

            name_score_dict = score_all_methods(
                query_text, professors, prof_texts, qid, sem_scores
            )

        per_query_scores[qid] = evaluate_query_from_scores(
            name_score_dict, all_labels[qid]
        )

    # Print per-query table
    print(f"\n{'─'*75}")
    print(f"  NDCG@{TOP_K} per query")
    print(f"  {'Query':<30} {'n_rel':>5} {'Jac':>7} {'TF':>7} {'BM':>7} "
          f"{'Sem':>7} {'Hyb':>7} {'Rrk':>7}")
    print(f"  {'─'*30} {'─'*5} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

    for qid in QUERIES:
        scores  = per_query_scores[qid]
        labels  = all_labels[qid]
        n_rel   = sum(1 for l in labels if l.get("relevant") is True)

        def fmt(m):
            v = scores[m].get("ndcg_at_k")
            return f"{v:>7.3f}" if v is not None else "   N/A"

        print(f"  {qid:<30} {n_rel:>5} {fmt('jaccard')} {fmt('tfidf')} "
              f"{fmt('bm25')} {fmt('semantic')} {fmt('hybrid')} {fmt('reranked')}")

    # Aggregate stats & bootstrap
    topic_stats, ci_stats, boot_ndcg, active_methods = \
        bootstrap_multi_query(per_query_scores)

    print(f"\n{'─'*75}")
    print(f"  AGGREGATE  mean NDCG@{TOP_K} ± std  ({len(QUERIES)} queries)")
    print(f"  {'Method':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} "
          f"{'95% CI (topic bootstrap)':>26}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*26}")

    for m in active_methods:
        ts = topic_stats[m]
        ci = ci_stats[m]
        mn  = f"{ts['mean_ndcg']:.4f}" if ts['mean_ndcg'] is not None else "  N/A"
        std = f"{ts['std_ndcg']:.4f}"  if ts['std_ndcg']  is not None else "  N/A"
        mn2 = f"{ts['min_ndcg']:.4f}"  if ts['min_ndcg']  is not None else "  N/A"
        mx  = f"{ts['max_ndcg']:.4f}"  if ts['max_ndcg']  is not None else "  N/A"
        print(f"  {m:<12} {mn:>8} {std:>8} {mn2:>8} {mx:>8} "
              f"  [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]")

    # Pairwise tests
    # When only keyword methods are available (no query_embeddings.json),
    # only the 3 keyword pairs can be tested. When semantic/hybrid/reranked
    # data is available, all C(6,2)=15 pairs are tested -- this is what
    # the paper's Table 6 and Section 7 findings report. A script run
    # without query_embeddings.json will only produce pairwise_kw (3 pairs);
    # only a run WITH query_embeddings.json reproduces pairwise_all (15 pairs).
    n_comp  = len(active_methods) * (len(active_methods) - 1) // 2
    alpha_b = 0.05 / n_comp
    all_pairs = [(a, b) for i, a in enumerate(active_methods)
                        for b in active_methods[i+1:]]

    print(f"\n{'─'*75}")
    print(f"  PAIRWISE TESTS — {len(active_methods)} methods, "
          f"{n_comp} pairs, Bonferroni α={alpha_b:.4f}")
    print(f"  {'Comparison':<22} {'ΔNDCG':>8} {'p_raw':>10} {'p_bonf':>10} {'Sig?':>6}")
    print(f"  {'─'*22} {'─'*8} {'─'*10} {'─'*10} {'─'*6}")

    pairwise = {}
    for m_a, m_b in all_pairs:
        diff, p_raw = paired_test(boot_ndcg[m_a], boot_ndcg[m_b])
        p_bonf = min(p_raw * n_comp, 1.0)
        sig    = p_bonf < 0.05
        label  = f"{m_a} vs {m_b}"
        pairwise[label] = {"mean_diff": round(diff,4), "p_raw": round(p_raw,4),
                            "p_bonf": round(p_bonf,4), "significant": sig}
        print(f"  {label:<22} {diff:>+8.4f} {p_raw:>10.4f} {p_bonf:>10.4f} "
              f"{'YES ✓' if sig else 'no':>6}")

    # Backward-compatible alias: pairwise_kw is the subset restricted to
    # keyword methods only, always present regardless of whether semantic
    # data was available. pairwise_all is the full set (== pairwise_kw
    # when semantic data is unavailable, since active_methods is then
    # just the 3 keyword methods).
    pairwise_kw_only = {
        k: v for k, v in pairwise.items()
        if all(m in ["jaccard", "tfidf", "bm25"] for m in k.split(" vs "))
    }

    # Save
    output = {
        "queries":        {qid: {"persona": QUERIES[qid]["persona"],
                                 "text":    QUERIES[qid]["text"],
                                 "source":  QUERIES[qid]["source"]}
                           for qid in QUERIES},
        "per_query_ndcg": {qid: {m: per_query_scores[qid][m].get("ndcg_at_k")
                                  for m in per_query_scores[qid]}
                           for qid in per_query_scores},
        "per_query_map":  {qid: {m: per_query_scores[qid][m].get("map_at_k")
                                  for m in per_query_scores[qid]}
                           for qid in per_query_scores},
        "aggregate":       topic_stats,
        "bootstrap_ci":    ci_stats,
        "pairwise_kw":     pairwise_kw_only,
        "pairwise_all":    pairwise,
        "config": {
            "n_queries":     len(QUERIES),
            "n_bootstrap":   N_BOOTSTRAP,
            "top_k":         TOP_K,
            "n_methods_active": len(active_methods),
            "n_comparisons":    n_comp,
            "semantic_available": os.path.exists(QUERY_EMBED_FILE),
            "note": ("Semantic/hybrid/reranked for Q2-Q5 require running "
                     "analysis/embed_queries.py with HuggingFace access. "
                     "pairwise_all == pairwise_kw when semantic data is "
                     "unavailable (only 3 keyword methods are active)."),
        },
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n[SAVED] {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
