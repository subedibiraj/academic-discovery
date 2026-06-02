# Ablation study: measures how removing each text component
# (bio, research areas, arxiv) affects retrieval quality.
import json
import math
import os
import sys
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
EMBEDDED_FILE = "data/final/all_professors_embedded.json"
LABELS_FILE   = "data/final/relevance_labels.json"
OUTPUT_FILE   = "data/final/ablation_results.json"
TOP_K         = 10

os.makedirs("data/final", exist_ok=True)

# Text Builders — each ablation builds text differently

def build_full_text(prof):
    """Full model: research_text + biography + arXiv abstracts."""
    parts = []
    if prof.get("research_text"):
        parts.append(prof["research_text"])
    if prof.get("biography"):
        parts.append(prof["biography"][:500])
    if prof.get("arxiv_papers"):
        for paper in prof["arxiv_papers"][:3]:
            if paper.get("abstract"):
                parts.append(paper["abstract"][:300])
    return " | ".join(parts)

def build_no_biography(prof):
    """Remove biography — keep research_text + arXiv."""
    parts = []
    if prof.get("research_text"):
        parts.append(prof["research_text"])
    if prof.get("arxiv_papers"):
        for paper in prof["arxiv_papers"][:3]:
            if paper.get("abstract"):
                parts.append(paper["abstract"][:300])
    return " | ".join(parts)

def build_no_research_areas(prof):
    """Remove structured research areas — keep biography + arXiv."""
    parts = []
    bio = prof.get("biography", "")
    if bio:
        parts.append(bio[:500])
    if prof.get("arxiv_papers"):
        for paper in prof["arxiv_papers"][:3]:
            if paper.get("abstract"):
                parts.append(paper["abstract"][:300])
    return " | ".join(parts)

def build_no_arxiv(prof):
    """Remove arXiv — keep research_text + biography."""
    parts = []
    if prof.get("research_text"):
        parts.append(prof["research_text"])
    if prof.get("biography"):
        parts.append(prof["biography"][:500])
    return " | ".join(parts)

def build_profile_only(prof):
    """Profile text only — no biography, no arXiv."""
    return prof.get("research_text", "")

def build_research_areas_only(prof):
    """Research areas only — minimal structured signal."""
    areas = prof.get("research_areas", [])
    if areas:
        return ", ".join(areas)
    # Fallback: extract first line of research_text
    rt = prof.get("research_text", "")
    return rt[:200] if rt else ""

# NDCG Computation

def compute_ndcg(ranked_names, relevance_map, k):
    """Compute NDCG@k given a ranked list of names and relevance map."""
    dcg = 0.0
    for i, name in enumerate(ranked_names[:k]):
        rel = relevance_map.get(name, 0)
        dcg += rel / math.log2(i + 2)

    ideal_rels = sorted(relevance_map.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0

def compute_map(ranked_names, relevant_set, k):
    """Compute MAP@k (Average Precision)."""
    ap = 0.0
    tp = 0
    for i, name in enumerate(ranked_names[:k]):
        if name in relevant_set:
            tp += 1
            ap += tp / (i + 1)
    return ap / min(len(relevant_set), k) if relevant_set else 0

# Main Ablation Runner

def main():
    print("=" * 60)
    print("  ABLATION STUDY: Component Contribution Analysis")
    print("=" * 60)

    # Load data
    with open(EMBEDDED_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    professors = data["professors"]
    my_interest = data["my_interest"]

    # Load relevance labels
    if not os.path.exists(LABELS_FILE):
        print("[ERROR] No relevance labels found. Run compare.py first.")
        sys.exit(1)

    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        labels = json.load(f)

    relevance_map = {}
    relevant_set = set()
    for entry in labels:
        if entry.get("relevant") is True:
            # Default to 1 (partially relevant), not 2, if grade is missing --
            # see matcher/compare.py for rationale.
            relevance_map[entry["name"]] = entry.get("grade", 1)
            relevant_set.add(entry["name"])
        elif entry.get("relevant") is False:
            relevance_map[entry["name"]] = 0

    if not relevant_set:
        print("[ERROR] No relevant labels found.")
        sys.exit(1)

    print(f"[DATA] {len(professors)} professors, {len(relevant_set)} relevant labels\n")

    # Load embedding model
    print("[MODEL] Loading sentence-transformers/all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Define ablations
    ablations = [
        ("Full Model",           build_full_text),
        ("-Biography",           build_no_biography),
        ("-Research Areas",      build_no_research_areas),
        ("-arXiv Papers",        build_no_arxiv),
        ("-Bio -arXiv (profile only)", build_profile_only),
        ("Research Areas Only",  build_research_areas_only),
    ]

    results = []

    for name, text_builder in ablations:
        print(f"\n[ABLATION] {name}")

        # Build texts
        prof_texts = [text_builder(p) for p in professors]
        empty_count = sum(1 for t in prof_texts if not t.strip())
        print(f"  Empty docs: {empty_count}/{len(professors)}")

        # Embed
        query_embedding = model.encode([my_interest])
        prof_embeddings = model.encode(prof_texts, show_progress_bar=False)

        # Semantic scores
        scores = cosine_similarity(query_embedding, prof_embeddings)[0]

        # Rank
        ranked_indices = np.argsort(scores)[::-1]
        ranked_names = [professors[i]["name"] for i in ranked_indices]

        # Compute metrics
        ndcg = compute_ndcg(ranked_names, relevance_map, TOP_K)
        map_score = compute_map(ranked_names, relevant_set, TOP_K)

        # Top-5 for inspection
        top5 = [(professors[i]["name"], round(float(scores[i]), 4))
                for i in ranked_indices[:5]]

        result = {
            "ablation": name,
            "ndcg_at_k": round(ndcg, 4),
            "map_at_k": round(map_score, 4),
            "empty_docs": empty_count,
            "top_5": top5,
        }
        results.append(result)
        print(f"  NDCG@{TOP_K}: {ndcg:.4f}  |  MAP@{TOP_K}: {map_score:.4f}")

    # Compute deltas from full model
    full_ndcg = results[0]["ndcg_at_k"]
    full_map = results[0]["map_at_k"]
    for r in results:
        r["ndcg_delta"] = round(r["ndcg_at_k"] - full_ndcg, 4)
        r["map_delta"] = round(r["map_at_k"] - full_map, 4)

    # Save results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"ablations": results, "config": {"top_k": TOP_K}}, f, indent=2)
    print(f"\n[SAVED] {OUTPUT_FILE}")

    # Print summary table
    print("\n" + "=" * 70)
    print("  ABLATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Ablation':<28} {'NDCG@10':<10} {'Delta NDCG':<10} {'MAP@10':<10} {'Delta MAP':<10}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for r in results:
        delta_ndcg = f"{r['ndcg_delta']:+.4f}" if r['ndcg_delta'] != 0 else "baseline"
        delta_map = f"{r['map_delta']:+.4f}" if r['map_delta'] != 0 else "baseline"
        print(f"  {r['ablation']:<28} {r['ndcg_at_k']:<10.4f} {delta_ndcg:<10} "
              f"{r['map_at_k']:<10.4f} {delta_map:<10}")
    print("=" * 70)

if __name__ == "__main__":
    main()
