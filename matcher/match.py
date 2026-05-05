# Multi-score ranking: computes jaccard, tfidf, semantic, and hybrid scores
# for each professor against my research interests.
import json
import sys
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path so we can import compare module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matcher.compare import tfidf_scores, _build_prof_text

INPUT_FILE  = "data/final/all_professors_embedded.json"
OUTPUT_FILE = "data/final/ranked_results.json"
TOP_N       = 15
HYBRID_ALPHA = 0.35  # weight for TF-IDF; (1-alpha) for semantic

def main():
    print("[LOAD] Loading embedded data...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    my_embedding = np.array(data["my_embedding"]).reshape(1, -1)
    my_interest  = data["my_interest"]
    professors   = data["professors"]

    print(f"[INFO] Ranking {len(professors)} professors...\n")

    # -- Semantic scores (from pre-computed embeddings) --
    prof_embeddings = np.array([p["embedding"] for p in professors])
    semantic_scores = cosine_similarity(my_embedding, prof_embeddings)[0]

    # -- TF-IDF keyword scores --
    prof_texts = [_build_prof_text(p) for p in professors]
    tfidf_score_list = tfidf_scores(my_interest, prof_texts)

    # -- Normalize for hybrid combination --
    tfidf_max = max(tfidf_score_list) if max(tfidf_score_list) > 0 else 1.0
    tfidf_norm = [s / tfidf_max for s in tfidf_score_list]

    sem_max = max(semantic_scores) if max(semantic_scores) > 0 else 1.0
    sem_norm = [s / sem_max for s in semantic_scores]

    # -- Build results --
    results = []
    for i, prof in enumerate(professors):
        hybrid = HYBRID_ALPHA * tfidf_norm[i] + (1 - HYBRID_ALPHA) * sem_norm[i]
        results.append({
            "rank"           : 0,
            "semantic_score" : round(float(semantic_scores[i]), 4),
            "tfidf_score"    : round(float(tfidf_score_list[i]), 4),
            "hybrid_score"   : round(float(hybrid), 4),
            "name"           : prof["name"],
            "university"     : prof["university"],
            "title"          : prof.get("title", ""),
            "email"          : prof.get("email", ""),
            "research"       : prof.get("research_text", ""),
            "profile_url"    : prof.get("profile_url", ""),
        })

    # Sort by hybrid score (best overall ranking)
    results.sort(key=lambda x: x["hybrid_score"], reverse=True)

    # -- Print ranked output --
    print(f"{'='*70}")
    print(f"YOUR INTEREST:")
    print(f"{my_interest[:200]}...")
    print(f"{'='*70}")
    print(f"\nTOP {TOP_N} PROFESSOR MATCHES (ranked by hybrid score)\n")

    for i, r in enumerate(results[:TOP_N], 1):
        r["rank"] = i
        print(f"{i:2d}. {r['name']} - {r['university']}")
        print(f"    Hybrid: {r['hybrid_score']:.4f}  |  "
              f"Semantic: {r['semantic_score']:.4f}  |  "
              f"TF-IDF: {r['tfidf_score']:.4f}")
        print(f"    Title   : {r['title']}")
        print(f"    Email   : {r['email']}")
        print(f"    Research: {r['research'][:100]}{'...' if len(r['research']) > 100 else ''}")
        print()

    # -- Save ranked results --
    output = {
        "my_interest"      : my_interest,
        "total_professors" : len(professors),
        "ranking_method"   : "hybrid (0.35*tfidf + 0.65*semantic)",
        "top_matches"      : results[:TOP_N],
        "all_ranked"       : results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] Full ranked list -> {OUTPUT_FILE}")
    print(f"[DONE]  {len(professors)} professors ranked.")

if __name__ == "__main__":
    main()