# Exports pipeline data to JSON files for the web dashboard.
import json
import os

INPUT_EMBEDDED  = "data/final/all_professors_embedded.json"
INPUT_ARXIV_PROFS = "data/final/all_professors_arxiv.json"
INPUT_COMPARE   = "data/final/comparison_results.json"
INPUT_STATS     = "data/final/statistics.json"
INPUT_ARXIV     = "data/final/arxiv_impact.json"
INPUT_CLUSTERS  = "data/final/clusters.json"
INPUT_EXPLAIN   = "data/final/explanations.json"
INPUT_MULTIQ    = "data/final/multi_query_results.json"
INPUT_LTR       = "data/final/ltr_results.json"
INPUT_LDA       = "data/final/lda_topics.json"
INPUT_ABLATION  = "data/final/ablation_results.json"
OUTPUT_DIR      = "app/data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # 1. Professors (no embeddings)
    print("[EXPORT] Loading embedded data...")
    with open(INPUT_EMBEDDED, "r", encoding="utf-8") as f:
        data = json.load(f)

    professors = []
    for p in data["professors"]:
        prof = {
            "name": p["name"],
            "university": p["university"],
            "title": p.get("title", ""),
            "email": p.get("email", ""),
            "research_text": p.get("research_text", ""),
            "biography": (p.get("biography", "") or "")[:800],
            "profile_url": p.get("profile_url", ""),
            "google_scholar": p.get("google_scholar", ""),
            "personal_website": p.get("personal_website", ""),
            "education": p.get("education", []),
            "research_areas": p.get("research_areas") or p.get("research_interests", []),
            "arxiv_papers": p.get("arxiv_papers", []),
        }
        professors.append(prof)

    # Merge arXiv papers from enriched dataset
    if os.path.exists(INPUT_ARXIV_PROFS):
        with open(INPUT_ARXIV_PROFS, "r", encoding="utf-8") as f:
            arxiv_profs = json.load(f)
        arxiv_map = {f"{p['name']}|{p['university']}": p.get("arxiv_papers", [])
                     for p in arxiv_profs}
        merged = 0
        for prof in professors:
            key = f"{prof['name']}|{prof['university']}"
            if key in arxiv_map and arxiv_map[key]:
                prof["arxiv_papers"] = arxiv_map[key]
                merged += 1
        print(f"[MERGE] Added arXiv papers for {merged} professors")

    # Add scores from comparison results
    if os.path.exists(INPUT_COMPARE):
        with open(INPUT_COMPARE, "r", encoding="utf-8") as f:
            compare = json.load(f)

        # Build name->scores mapping from all_ranked list
        score_map = {}
        for entry in compare.get("all_ranked", []):
            name = entry["name"]
            score_map[name] = {
                "jaccard_score": entry.get("jaccard_score", 0),
                "tfidf_score": entry.get("tfidf_score", 0),
                "semantic_score": entry.get("semantic_score", 0),
                "hybrid_score": entry.get("hybrid_score", 0),
                "bm25_score": entry.get("bm25_score", 0),
                "reranked_score": entry.get("reranked_score", 0),
            }

        for prof in professors:
            if prof["name"] in score_map:
                prof.update(score_map[prof["name"]])

    ui_data = {
        "my_interest": data["my_interest"],
        "total": len(professors),
        "professors": professors,
    }

    out_path = os.path.join(OUTPUT_DIR, "professors.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ui_data, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {out_path} ({len(professors)} professors)")

    # 2. Comparison results (already lightweight)
    if os.path.exists(INPUT_COMPARE):
        with open(INPUT_COMPARE, "r", encoding="utf-8") as f:
            compare = json.load(f)
        out_path = os.path.join(OUTPUT_DIR, "comparison.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(compare, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] {out_path}")

    # 3. Statistics
    if os.path.exists(INPUT_STATS):
        with open(INPUT_STATS, "r", encoding="utf-8") as f:
            stats = json.load(f)
        out_path = os.path.join(OUTPUT_DIR, "statistics.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] {out_path}")

    # 4. arXiv impact
    if os.path.exists(INPUT_ARXIV):
        with open(INPUT_ARXIV, "r", encoding="utf-8") as f:
            arxiv = json.load(f)
        out_path = os.path.join(OUTPUT_DIR, "arxiv_impact.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(arxiv, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] {out_path}")

    # 5. Clusters
    if os.path.exists(INPUT_CLUSTERS):
        with open(INPUT_CLUSTERS, "r", encoding="utf-8") as f:
            clusters = json.load(f)
        out_path = os.path.join(OUTPUT_DIR, "clusters.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(clusters, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] {out_path}")

    # 6. Explanations
    if os.path.exists(INPUT_EXPLAIN):
        with open(INPUT_EXPLAIN, "r", encoding="utf-8") as f:
            explain = json.load(f)
        out_path = os.path.join(OUTPUT_DIR, "explanations.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(explain, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] {out_path}")

    # 7. Multi-query evaluation results (5-query comparison)
    if os.path.exists(INPUT_MULTIQ):
        with open(INPUT_MULTIQ, "r", encoding="utf-8") as f:
            multiq = json.load(f)
        out_path = os.path.join(OUTPUT_DIR, "multi_query_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(multiq, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] {out_path}")

    # 8. LTR results (feature importances, NDCG/MAP)
    if os.path.exists(INPUT_LTR):
        with open(INPUT_LTR, "r", encoding="utf-8") as f:
            ltr = json.load(f)
        out_path = os.path.join(OUTPUT_DIR, "ltr_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ltr, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] {out_path}")

    # 9. LDA topic model output
    if os.path.exists(INPUT_LDA):
        with open(INPUT_LDA, "r", encoding="utf-8") as f:
            lda = json.load(f)
        out_path = os.path.join(OUTPUT_DIR, "lda_topics.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(lda, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] {out_path}")

    # 10. Ablation study results
    if os.path.exists(INPUT_ABLATION):
        with open(INPUT_ABLATION, "r", encoding="utf-8") as f:
            ablation = json.load(f)
        out_path = os.path.join(OUTPUT_DIR, "ablation_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ablation, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] {out_path}")

    print(f"\n[DONE] UI data exported to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
