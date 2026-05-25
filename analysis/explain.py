# Generates explainable recommendation cards.
# Shows matching keywords, shared topics, and score drivers.
import json
import os
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Config
INPUT_EMBEDDED  = "data/final/all_professors_embedded.json"
INPUT_COMPARE   = "data/final/comparison_results.json"
INPUT_CLUSTERS  = "data/final/clusters.json"
OUTPUT_FILE     = "data/final/explanations.json"
TOP_N           = 50   # explain top N professors

os.makedirs("data/final", exist_ok=True)

def load_data():
    """Load embedded data and comparison results."""
    print("[LOAD] Loading data...")

    with open(INPUT_EMBEDDED, "r", encoding="utf-8") as f:
        embedded = json.load(f)

    with open(INPUT_COMPARE, "r", encoding="utf-8") as f:
        compare = json.load(f)

    clusters = {}
    if os.path.exists(INPUT_CLUSTERS):
        with open(INPUT_CLUSTERS, "r", encoding="utf-8") as f:
            clusters = json.load(f)

    return embedded, compare, clusters

def get_matching_keywords(user_text, prof_text, top_k=6):
    """
    Find the most important shared keywords between user and professor
    using TF-IDF feature importance.
    """
    if not user_text or not prof_text:
        return []

    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=500,
            ngram_range=(1, 2),
            min_df=1,
        )
        tfidf_matrix = vectorizer.fit_transform([user_text, prof_text])
        feature_names = vectorizer.get_feature_names_out()

        # Get TF-IDF scores for both docs
        user_scores = tfidf_matrix[0].toarray().flatten()
        prof_scores = tfidf_matrix[1].toarray().flatten()

        # Find shared important terms (product of scores = high means both use it)
        shared_importance = user_scores * prof_scores
        top_indices = shared_importance.argsort()[::-1][:top_k]

        keywords = []
        for idx in top_indices:
            if shared_importance[idx] > 0:
                keywords.append({
                    "term": feature_names[idx],
                    "relevance": round(float(shared_importance[idx]), 4),
                })

        return keywords
    except Exception:
        return []

def get_shared_topics(user_interests_text, prof_interests):
    """Find research topics that overlap between user and professor."""
    if not user_interests_text or not prof_interests:
        return []

    user_lower = user_interests_text.lower()
    shared = []

    # Check each professor interest against user text
    for interest in prof_interests:
        interest_clean = interest.strip().lower()
        if len(interest_clean) < 3:
            continue

        # Check for direct match or key phrase overlap
        interest_words = set(re.findall(r'\b\w{3,}\b', interest_clean))
        user_words = set(re.findall(r'\b\w{3,}\b', user_lower))

        overlap = interest_words & user_words
        # If >40% of the interest's words appear in user text, it's a match
        if interest_words and len(overlap) / len(interest_words) >= 0.4:
            shared.append(interest.strip())

    return shared[:6]

def compute_score_breakdown(prof_scores):
    """Break down the hybrid score into component contributions."""
    semantic = prof_scores.get("semantic_score", 0)
    tfidf = prof_scores.get("tfidf_score", 0)
    hybrid = prof_scores.get("hybrid_score", 0)
    jaccard = prof_scores.get("jaccard_score", 0)

    # Determine primary driver
    if semantic > tfidf * 2:
        driver = "semantic"
        explanation = "Conceptually similar research (embedding match)"
    elif tfidf > semantic * 1.5:
        driver = "keyword"
        explanation = "Strong keyword overlap in research descriptions"
    else:
        driver = "balanced"
        explanation = "Both keyword and conceptual similarity contribute"

    return {
        "semantic_score": round(semantic, 4),
        "tfidf_score": round(tfidf, 4),
        "hybrid_score": round(hybrid, 4),
        "jaccard_score": round(jaccard, 4),
        "primary_driver": driver,
        "explanation": explanation,
        "semantic_pct": round(semantic / hybrid * 100, 1) if hybrid > 0 else 0,
    }

def main():
    embedded, compare, clusters = load_data()

    my_interest = embedded["my_interest"]
    professors = embedded["professors"]
    all_ranked = compare.get("all_ranked", [])

    # Build score map
    score_map = {}
    for entry in all_ranked:
        score_map[entry["name"]] = entry

    # Build cluster map
    cluster_map = {}
    cluster_info = clusters.get("clusters", {})
    for cp in clusters.get("professors", []):
        cluster_map[cp["name"]] = cp.get("cluster_id")

    # Sort by hybrid score
    ranked = sorted(all_ranked, key=lambda x: x.get("hybrid_score", 0), reverse=True)

    print(f"[INFO] Generating explanations for top {TOP_N} professors...")

    explanations = {}
    for i, entry in enumerate(ranked[:TOP_N]):
        name = entry["name"]

        # Find full professor data
        prof = next((p for p in professors if p["name"] == name), None)
        if not prof:
            continue

        # Get research text
        prof_research = prof.get("research_text", "")
        prof_interests = prof.get("research_interests", [])
        prof_bio = prof.get("biography", "")
        full_prof_text = f"{prof_research} {prof_bio}"

        # 1. Matching keywords
        keywords = get_matching_keywords(my_interest, full_prof_text)

        # 2. Shared topics
        shared_topics = get_shared_topics(my_interest, prof_interests)

        # 3. Score breakdown
        scores = compute_score_breakdown(entry)

        # 4. Cluster context
        cid = cluster_map.get(name)
        cluster_label = ""
        if cid is not None and str(cid) in cluster_info:
            cluster_label = cluster_info[str(cid)].get("label", "")

        explanations[name] = {
            "rank": i + 1,
            "matching_keywords": keywords,
            "shared_topics": shared_topics,
            "score_breakdown": scores,
            "cluster_id": cid,
            "cluster_label": cluster_label,
            "summary": generate_summary(name, keywords, shared_topics, scores, cluster_label),
        }

        if i < 5:
            print(f"\n  [{i+1}] {name}")
            print(f"      Keywords: {', '.join(k['term'] for k in keywords[:3])}")
            print(f"      Topics:   {', '.join(shared_topics[:3]) or 'none'}")
            print(f"      Driver:   {scores['primary_driver']} ({scores['explanation']})")

    # Save
    output = {
        "my_interest": my_interest,
        "total_explained": len(explanations),
        "explanations": explanations,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {OUTPUT_FILE} ({len(explanations)} professors explained)")

def generate_summary(name, keywords, shared_topics, scores, cluster_label):
    """Generate a human-readable explanation sentence."""
    parts = []

    # Score driver
    if scores["primary_driver"] == "semantic":
        parts.append("strong conceptual alignment with your research interests")
    elif scores["primary_driver"] == "keyword":
        parts.append("significant keyword overlap with your research description")
    else:
        parts.append("both keyword and conceptual similarity to your interests")

    # Keywords
    if keywords:
        top_kw = [k["term"] for k in keywords[:3]]
        parts.append(f"matching on terms like {', '.join(top_kw)}")

    # Shared topics
    if shared_topics:
        parts.append(f"shared interest in {', '.join(shared_topics[:2])}")

    # Cluster
    if cluster_label:
        parts.append(f"part of the {cluster_label} research cluster")

    return f"Recommended due to {'; '.join(parts)}."

if __name__ == "__main__":
    main()
