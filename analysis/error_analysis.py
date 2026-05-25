# Error analysis: finds false positives and negatives
# for each method to understand failure patterns.
import json
import os

COMPARE_FILE = "data/final/comparison_results.json"
LABELS_FILE  = "data/final/relevance_labels.json"
OUTPUT_FILE  = "data/final/error_analysis.json"
TOP_K        = 10

os.makedirs("data/final", exist_ok=True)

def main():
    print("=" * 60)
    print("  ERROR ANALYSIS: Why Methods Fail")
    print("=" * 60)

    with open(COMPARE_FILE, "r", encoding="utf-8") as f:
        compare = json.load(f)
    all_ranked = compare["all_ranked"]
    prof_map = {r["name"]: r for r in all_ranked}

    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        labels = json.load(f)

    relevant = set()
    irrelevant = set()
    for entry in labels:
        if entry.get("relevant") is True:
            relevant.add(entry["name"])
        elif entry.get("relevant") is False:
            irrelevant.add(entry["name"])

    print(f"[DATA] {len(relevant)} relevant, {len(irrelevant)} irrelevant\n")

    methods = {
        "jaccard":  "jaccard_score",
        "tfidf":    "tfidf_score",
        "bm25":     "bm25_score",
        "semantic": "semantic_score",
        "hybrid":   "hybrid_score",
        "reranked": "reranked_score",
    }

    results = {}

    for method_name, score_key in methods.items():
        ranked = sorted(all_ranked, key=lambda x: x.get(score_key, 0), reverse=True)
        top_k_names = set(r["name"] for r in ranked[:TOP_K])

        # False positives: in top-K but labeled irrelevant
        fp = top_k_names & irrelevant
        # False negatives: relevant but NOT in top-K
        fn = relevant - top_k_names
        # True positives
        tp = top_k_names & relevant

        print(f"\n{'='*50}")
        print(f"  {method_name.upper()} (score_key={score_key})")
        print(f"{'='*50}")
        print(f"  TP={len(tp)}, FP={len(fp)}, FN={len(fn)}")

        # Analyze false positives
        fp_details = []
        if fp:
            print(f"\n  -- False Positives (ranked high but irrelevant) --")
            for name in sorted(fp):
                p = prof_map.get(name, {})
                rank = next((i+1 for i, r in enumerate(ranked) if r["name"] == name), "?")
                snippet = (p.get("research_text", "") or "")[:100]
                print(f"    Rank {rank}: {name}")
                print(f"      Score: {p.get(score_key, 0):.4f}")
                print(f"      Research: {snippet}...")
                fp_details.append({
                    "name": name, "rank": rank,
                    "score": round(p.get(score_key, 0), 4),
                    "research": snippet,
                    "university": p.get("university", ""),
                })

        # Analyze false negatives
        fn_details = []
        if fn:
            print(f"\n  -- False Negatives (relevant but missed) --")
            for name in sorted(fn):
                p = prof_map.get(name, {})
                rank = next((i+1 for i, r in enumerate(ranked) if r["name"] == name), "?")
                snippet = (p.get("research_text", "") or "")[:100]
                print(f"    Rank {rank}: {name}")
                print(f"      Score: {p.get(score_key, 0):.4f}")
                print(f"      Research: {snippet}...")
                fn_details.append({
                    "name": name, "rank": rank,
                    "score": round(p.get(score_key, 0), 4),
                    "research": snippet,
                    "university": p.get("university", ""),
                })

        # Failure patterns
        fp_unis = [d["university"] for d in fp_details]
        fn_unis = [d["university"] for d in fn_details]

        results[method_name] = {
            "tp": len(tp), "fp": len(fp), "fn": len(fn),
            "false_positives": fp_details,
            "false_negatives": fn_details,
            "fp_universities": list(set(fp_unis)),
            "fn_universities": list(set(fn_unis)),
        }

    # Cross-method analysis
    print(f"\n\n{'='*60}")
    print("  CROSS-METHOD ERROR PATTERNS")
    print(f"{'='*60}")

    # Find professors that ALL methods get wrong
    all_fn = None
    for m in methods:
        ranked = sorted(all_ranked, key=lambda x: x.get(methods[m], 0), reverse=True)
        top_names = set(r["name"] for r in ranked[:TOP_K])
        fn = relevant - top_names
        all_fn = fn if all_fn is None else all_fn & fn

    print(f"\n  Relevant professors missed by ALL methods ({len(all_fn)}):")
    for name in sorted(all_fn):
        p = prof_map.get(name, {})
        snippet = (p.get("research_text", "") or "")[:80]
        print(f"    - {name} ({p.get('university', '')})")
        print(f"      Research: {snippet}...")

    results["cross_method"] = {
        "missed_by_all": sorted(all_fn),
        "count_missed_by_all": len(all_fn),
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
