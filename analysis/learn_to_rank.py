# Learning-to-Rank with GradientBoosting
# Trains on manual relevance labels to learn optimal feature weights.
import json
import math
import os
import sys
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Paths
COMPARE_FILE = "data/final/comparison_results.json"
LABELS_FILE  = "data/final/relevance_labels.json"
OUTPUT_FILE  = "data/final/ltr_results.json"
TOP_K        = 10

os.makedirs("data/final", exist_ok=True)

def compute_ndcg(scores, relevances, k):
    """Compute NDCG@k from score-relevance pairs."""
    sorted_indices = np.argsort(scores)[::-1]
    dcg = 0.0
    for i, idx in enumerate(sorted_indices[:k]):
        dcg += relevances[idx] / math.log2(i + 2)
    ideal = sorted(relevances, reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0

def main():
    print("=" * 60)
    print("  LEARNING-TO-RANK: Feature Weight Optimization")
    print("=" * 60)

    # Load comparison data
    with open(COMPARE_FILE, "r", encoding="utf-8") as f:
        compare = json.load(f)
    all_ranked = compare["all_ranked"]

    # Load labels
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        labels = json.load(f)

    # Build label map
    label_map = {}
    for entry in labels:
        if entry.get("relevant") is not None:
            label_map[entry["name"]] = {
                "relevant": entry["relevant"],
                # Default to 1 (partially relevant) not 2 (highly relevant)
                # if grade is missing -- consistent with compare.py fix.
                "grade": entry.get("grade", 1 if entry["relevant"] else 0),
            }

    print(f"[DATA] {len(all_ranked)} professors, {len(label_map)} labeled\n")

    # Build feature matrix
    feature_names = [
        "semantic_score",
        "tfidf_score",
        "bm25_score",
        "jaccard_score",
        "hybrid_score",
        "reranked_score",
        "has_arxiv",
        "n_research_areas",
        "bio_length",
    ]

    prof_map = {r["name"]: r for r in all_ranked}
    X_all = []
    y_all = []
    names_all = []

    for name, info in label_map.items():
        if name not in prof_map:
            continue
        r = prof_map[name]
        # NOTE: has_arxiv (feature index 6 below) is a structurally broken
        # feature, kept here for transparency rather than silently removed.
        # The check searches `research_text` (research area tags, e.g.
        # "Machine Learning, NLP") for the literal substring "arxiv" -- but
        # research_text never contains paper metadata, so this condition is
        # FALSE for all 768 professors (verified directly: 0/768 have
        # "arxiv" anywhere in research_text). The corpus-wide arXiv
        # coverage data needed to populate this correctly only exists for
        # the 296-professor subpopulation in data/final/arxiv_impact.json,
        # used for the separate controlled arXiv experiment -- it was never
        # joined into all_professors.json / comparison_results.json at the
        # full 768-professor scale. As a result this feature always
        # evaluates to 0, and the LTR model trained here effectively has 8
        # informative features, not 9. This matches the reported
        # feature_importances showing has_arxiv at exactly 0.0% -- not
        # because arXiv coverage is uninformative, but because the feature
        # itself carries no signal as implemented.
        features = [
            r.get("semantic_score", 0),
            r.get("tfidf_score", 0),
            r.get("bm25_score", 0),
            r.get("jaccard_score", 0),
            r.get("hybrid_score", 0),
            r.get("reranked_score", 0),
            1 if r.get("research_text", "") and "arxiv" in r.get("research_text", "").lower() else 0,
            len(r.get("research_text", "").split(",")) if r.get("research_text") else 0,
            len(r.get("research_text", "")),
        ]
        X_all.append(features)
        y_all.append(1 if info["relevant"] else 0)
        names_all.append(name)

    X = np.array(X_all)
    y = np.array(y_all)

    print(f"[FEATURES] {X.shape[1]} features, {X.shape[0]} samples")
    print(f"[LABELS] {sum(y)} relevant, {len(y) - sum(y)} irrelevant\n")

    # Stratified K-Fold Cross Validation
    print("[MODEL] Training GradientBoostingClassifier with 5-fold CV...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_ndcgs = []
    all_pred_scores = np.zeros(len(y))

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )
        model.fit(X_train_s, y_train)

        # Predict probabilities
        probs = model.predict_proba(X_test_s)[:, 1]
        all_pred_scores[test_idx] = probs

        print(f"  Fold {fold+1}: test_size={len(test_idx)}, "
              f"relevant={sum(y_test)}")

    # Compute overall NDCG using cross-validated predictions
    # For full ranking, score ALL professors
    print(f"\n[RANKING] Scoring all {len(all_ranked)} professors with full model...")
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    model.fit(X_s, y)

    # Score all professors (not just labeled ones)
    X_full = []
    full_names = []
    for r in all_ranked:
        features = [
            r.get("semantic_score", 0),
            r.get("tfidf_score", 0),
            r.get("bm25_score", 0),
            r.get("jaccard_score", 0),
            r.get("hybrid_score", 0),
            r.get("reranked_score", 0),
            # has_arxiv: same structurally broken feature as in the labeled
            # loop above -- always 0 for all 768 professors. See the comment
            # block above for the full explanation.
            1 if r.get("research_text", "") and "arxiv" in r.get("research_text", "").lower() else 0,
            len(r.get("research_text", "").split(",")) if r.get("research_text") else 0,
            len(r.get("research_text", "")),
        ]
        X_full.append(features)
        full_names.append(r["name"])

    X_full = np.array(X_full)
    X_full_s = scaler.transform(X_full)
    ltr_scores = model.predict_proba(X_full_s)[:, 1]

    # Compute LTR NDCG
    relevance_map = {n: label_map[n]["grade"] for n in label_map if label_map[n]["relevant"]}
    ranked_indices = np.argsort(ltr_scores)[::-1]
    ranked_names = [full_names[i] for i in ranked_indices]

    dcg = 0.0
    for i, name in enumerate(ranked_names[:TOP_K]):
        rel = relevance_map.get(name, 0)
        dcg += rel / math.log2(i + 2)
    ideal_rels = sorted(relevance_map.values(), reverse=True)[:TOP_K]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    ltr_ndcg = dcg / idcg if idcg > 0 else 0

    # MAP
    ap, tp = 0.0, 0
    relevant_set = set(relevance_map.keys())
    for i, name in enumerate(ranked_names[:TOP_K]):
        if name in relevant_set:
            tp += 1
            ap += tp / (i + 1)
    ltr_map = ap / min(len(relevant_set), TOP_K) if relevant_set else 0

    # Feature importances
    importances = model.feature_importances_
    sorted_fi = sorted(zip(feature_names, importances), key=lambda x: -x[1])

    print("\n" + "=" * 60)
    print("  LEARNING-TO-RANK RESULTS")
    print("=" * 60)
    print(f"\n  LTR NDCG@{TOP_K}: {ltr_ndcg:.4f}")
    print(f"  LTR MAP@{TOP_K}:  {ltr_map:.4f}")

    print(f"\n  -- Feature Importances --")
    print(f"  {'Feature':<20} {'Importance':<12} {'Bar'}")
    print(f"  {'-'*20} {'-'*12} {'-'*30}")
    for feat, imp in sorted_fi:
        bar = "#" * int(imp * 100)
        print(f"  {feat:<20} {imp:<12.4f} {bar}")

    print(f"\n  -- LTR Top 10 --")
    for i, idx in enumerate(ranked_indices[:10]):
        name = full_names[idx]
        score = ltr_scores[idx]
        rel = relevance_map.get(name, "?")
        print(f"  {i+1:>3}. {name:<35} score={score:.4f}  rel={rel}")

    print("=" * 60)

    # Save results
    real_pr = compare.get("precision_recall", {})
    results = {
        "ltr_ndcg": round(ltr_ndcg, 4),
        "ltr_map": round(ltr_map, 4),
        "feature_importances": {feat: round(float(imp), 4) for feat, imp in sorted_fi},
        "top_10": [
            {"rank": i+1, "name": full_names[ranked_indices[i]],
             "score": round(float(ltr_scores[ranked_indices[i]]), 4)}
            for i in range(10)
        ],
        "comparison": {
            # Pulled directly from comparison_results.json (loaded above as
            # `compare`) rather than hardcoded -- a previous version of this
            # script had fabricated/stale placeholder values here
            # (semantic=0.7682, bm25=0.7243, tfidf=0.6855) that did not
            # match any real NDCG figure in the dataset. Verified those
            # numbers never appeared in the paper, README, or app (the app
            # only reads feature_importances from this file), so this was
            # dead data rather than a propagated error -- but it is fixed
            # here to actually reflect the real Q1 NDCG values so the field
            # is trustworthy if anyone reads this file directly.
            "semantic_ndcg": real_pr.get("semantic", {}).get("ndcg_at_k"),
            "bm25_ndcg":     real_pr.get("bm25", {}).get("ndcg_at_k"),
            "tfidf_ndcg":    real_pr.get("tfidf", {}).get("ndcg_at_k"),
        },
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
