# Bootstrap significance testing with Bonferroni correction.
# Bootstrap significance testing (B=10,000) with Bonferroni correction.
#
# CHANGES vs original:
#   - Bootstrap resamples only the 67 LABELLED professors (not all 768).
#     Resampling unlabelled professors adds no signal and miscalibrates CIs.
#   - Bonferroni correction applied across all 15 pairwise comparisons.
#   - Significance test named explicitly: paired bootstrap permutation test.
#   - Oracle and random baselines added.
#   - grade field (0/1/2) used throughout for graded NDCG.
#
import json
import math
import os
import numpy as np

COMPARE_FILE = "data/final/comparison_results.json"
LABELS_FILE  = "data/final/relevance_labels.json"
OUTPUT_FILE  = "data/final/significance_results.json"
TOP_K        = 10
N_BOOTSTRAP  = 10_000
ALPHA        = 0.05   # family-wise error rate (Bonferroni target)

os.makedirs("data/final", exist_ok=True)

# Metric helpers
def compute_ndcg(ranked_names, relevance_map, k=TOP_K):
    """NDCG@k with graded relevance (grade in {0,1,2}). Unlabelled = 0."""
    dcg  = sum(relevance_map.get(n, 0) / math.log2(i + 2)
               for i, n in enumerate(ranked_names[:k]))
    ideal = sorted(relevance_map.values(), reverse=True)[:k]
    idcg  = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

def compute_map(ranked_names, relevant_set, k=TOP_K):
    """MAP@k (binary relevance)."""
    ap, tp = 0.0, 0
    for i, name in enumerate(ranked_names[:k]):
        if name in relevant_set:
            tp += 1
            ap += tp / (i + 1)
    return ap / min(len(relevant_set), k) if relevant_set else 0.0

# Bootstrap
def bootstrap_metrics(all_ranked, labeled, n_bootstrap=N_BOOTSTRAP, k=TOP_K):
    """
    Paired bootstrap resampling TEST (Efron & Tibshirani, 1993).

    We resample ONLY the 67 labelled professors (with replacement).
    Unlabelled professors are treated as irrelevant (grade=0) and are
    fixed — they contribute nothing to NDCG regardless of rank, so
    including them in the resample would inflate effective sample size
    without adding information.
    """
    methods = {
        "jaccard":  "jaccard_score",
        "tfidf":    "tfidf_score",
        "bm25":     "bm25_score",
        "semantic": "semantic_score",
        "hybrid":   "hybrid_score",
        "reranked": "reranked_score",
    }

    # Pre-sort full ranking by each method (fixed — not resampled)
    sorted_by = {}
    for method_name, score_key in methods.items():
        ranked = sorted(all_ranked, key=lambda x: x.get(score_key, 0), reverse=True)
        sorted_by[method_name] = [r["name"] for r in ranked]

    labeled_arr = np.array(labeled, dtype=object)
    rng = np.random.RandomState(42)

    ndcg_samples = {m: [] for m in methods}
    map_samples  = {m: [] for m in methods}

    print(f"[BOOTSTRAP] Paired bootstrap test, B={n_bootstrap}, "
          f"resampling {len(labeled)} labelled professors...")

    for b in range(n_bootstrap):
        # Resample the 67 labelled professors with replacement
        indices = rng.choice(len(labeled_arr), size=len(labeled_arr), replace=True)
        sample  = labeled_arr[indices]

        # Build relevance maps from this resample
        rel_map = {}
        rel_set = set()
        for entry in sample:
            name  = entry["name"]
            grade = int(entry.get("grade", 0))
            if entry.get("relevant") is True:
                # Take max grade if name appears multiple times in resample
                rel_map[name] = max(rel_map.get(name, 0), grade)
                rel_set.add(name)
            else:
                rel_map[name] = rel_map.get(name, 0)  # stays 0

        if not rel_set:
            continue

        for method_name in methods:
            ndcg_samples[method_name].append(
                compute_ndcg(sorted_by[method_name], rel_map, k)
            )
            map_samples[method_name].append(
                compute_map(sorted_by[method_name], rel_set, k)
            )

        if (b + 1) % 2000 == 0:
            print(f"  ... {b + 1}/{n_bootstrap}")

    return ndcg_samples, map_samples

# Significance test
def paired_bootstrap_test(samples_a, samples_b):
    """
    Paired bootstrap permutation test (two-sided).
    Returns: (mean_diff, raw_p_value)
    Interpretation: fraction of bootstrap samples where A <= B (when mean A>B).
    """
    diffs = np.array(samples_a) - np.array(samples_b)
    mean_diff = float(np.mean(diffs))
    if mean_diff > 0:
        p_raw = float(np.mean(diffs <= 0)) * 2   # two-sided
    else:
        p_raw = float(np.mean(diffs >= 0)) * 2
    return mean_diff, min(p_raw, 1.0)

# Baselines
def compute_oracle_random(all_ranked, relevance_map, relevant_names, k=TOP_K):
    """
    Oracle: place all relevant professors in the first k positions.
    Random: expected NDCG under a random permutation of all 768 professors.
    """
    # Oracle ranking: relevant first (sorted by grade desc), then rest
    relevant_sorted = sorted(
        [n for n in relevant_names],
        key=lambda n: relevance_map.get(n, 0), reverse=True
    )
    rest = [r["name"] for r in all_ranked if r["name"] not in relevant_names]
    oracle_names = relevant_sorted + rest
    oracle_ndcg  = compute_ndcg(oracle_names, relevance_map, k)
    oracle_map   = compute_map(oracle_names, relevant_names, k)

    # Random expected NDCG: average over 10,000 random permutations
    rng  = np.random.RandomState(0)
    all_names = [r["name"] for r in all_ranked]
    random_ndcgs, random_maps = [], []
    for _ in range(10_000):
        perm = rng.permutation(all_names).tolist()
        random_ndcgs.append(compute_ndcg(perm, relevance_map, k))
        random_maps.append(compute_map(perm, relevant_names, k))

    return {
        "oracle": {
            "ndcg_at_k": round(oracle_ndcg, 4),
            "map_at_k":  round(oracle_map, 4),
        },
        "random": {
            "ndcg_at_k": round(float(np.mean(random_ndcgs)), 4),
            "ndcg_std":  round(float(np.std(random_ndcgs)),  4),
            "map_at_k":  round(float(np.mean(random_maps)),  4),
        },
    }

# Main
def main():
    print("=" * 70)
    print("  STATISTICAL SIGNIFICANCE: Paired Bootstrap + Bonferroni Correction")
    print("=" * 70)

    with open(COMPARE_FILE, "r", encoding="utf-8") as f:
        compare = json.load(f)
    all_ranked = compare["all_ranked"]

    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        labels = json.load(f)

    labeled = [l for l in labels if l.get("relevant") is not None]

    # Build authoritative relevance maps from full label set
    relevance_map  = {}
    relevant_names = set()
    for entry in labeled:
        name  = entry["name"]
        grade = int(entry.get("grade", 0))
        if entry.get("relevant") is True:
            relevance_map[name] = grade
            relevant_names.add(name)
        else:
            relevance_map[name] = 0

    print(f"[DATA] {len(all_ranked)} professors total, "
          f"{len(labeled)} labelled ({len(relevant_names)} relevant)")
    print(f"[CONFIG] B={N_BOOTSTRAP}, k={TOP_K}, "
          f"family-wise alpha={ALPHA} (Bonferroni)\n")

    # Oracle and random baselines
    print("[BASELINES] Computing oracle and random baselines...")
    baselines = compute_oracle_random(all_ranked, relevance_map, relevant_names)
    print(f"  Oracle NDCG@{TOP_K} = {baselines['oracle']['ndcg_at_k']:.4f}  "
          f"MAP@{TOP_K} = {baselines['oracle']['map_at_k']:.4f}")
    print(f"  Random NDCG@{TOP_K} = {baselines['random']['ndcg_at_k']:.4f} "
          f"(±{baselines['random']['ndcg_std']:.4f})\n")

    # Bootstrap
    ndcg_samples, map_samples = bootstrap_metrics(all_ranked, labeled)

    # Confidence intervals
    results = {
        "baselines": baselines,
        "confidence_intervals": {},
        "pairwise_tests": {},
    }

    print("\n" + "=" * 70)
    print("  95% CONFIDENCE INTERVALS  (bootstrap over 67 labelled professors)")
    print("=" * 70)
    print(f"  {'Method':<12} {'NDCG@10':>10} {'95% CI':>22}  "
          f"{'MAP@10':>10} {'95% CI':>22}")
    print(f"  {'-'*12} {'-'*10} {'-'*22}  {'-'*10} {'-'*22}")

    for method in ["jaccard", "tfidf", "bm25", "semantic", "hybrid", "reranked"]:
        nd = np.array(ndcg_samples[method])
        mp = np.array(map_samples[method])

        ci = {
            "ndcg_mean":    round(float(np.mean(nd)), 4),
            "ndcg_ci_low":  round(float(np.percentile(nd, 2.5)), 4),
            "ndcg_ci_high": round(float(np.percentile(nd, 97.5)), 4),
            "ndcg_std":     round(float(np.std(nd)), 4),
            "map_mean":     round(float(np.mean(mp)), 4),
            "map_ci_low":   round(float(np.percentile(mp, 2.5)), 4),
            "map_ci_high":  round(float(np.percentile(mp, 97.5)), 4),
            "map_std":      round(float(np.std(mp)), 4),
        }
        results["confidence_intervals"][method] = ci

        print(f"  {method:<12} {ci['ndcg_mean']:>10.4f} "
              f"[{ci['ndcg_ci_low']:.4f}, {ci['ndcg_ci_high']:.4f}]  "
              f"{ci['map_mean']:>10.4f} "
              f"[{ci['map_ci_low']:.4f}, {ci['map_ci_high']:.4f}]")

    # Pairwise tests with Bonferroni
    # All 15 pairs (C(6,2)) — Bonferroni threshold = ALPHA / 15
    all_methods = ["jaccard", "tfidf", "bm25", "semantic", "hybrid", "reranked"]
    all_pairs   = [(a, b) for i, a in enumerate(all_methods)
                          for b in all_methods[i+1:]]
    n_comparisons   = len(all_pairs)
    alpha_bonferroni = ALPHA / n_comparisons

    print(f"\n{'=' * 70}")
    print(f"  PAIRWISE TESTS — Paired Bootstrap Permutation Test (two-sided)")
    print(f"  {n_comparisons} comparisons; Bonferroni α = "
          f"{ALPHA}/{n_comparisons} = {alpha_bonferroni:.4f}")
    print(f"{'=' * 70}")
    print(f"  {'Comparison':<28} {'ΔNDCG':>8}  {'p (raw)':>10}  "
          f"{'p (Bonf.)':>10}  {'Sig?':>6}")
    print(f"  {'-'*28} {'-'*8}  {'-'*10}  {'-'*10}  {'-'*6}")

    for method_a, method_b in all_pairs:
        diff, p_raw = paired_bootstrap_test(
            ndcg_samples[method_a], ndcg_samples[method_b]
        )
        p_bonf = min(p_raw * n_comparisons, 1.0)  # Bonferroni-adjusted p
        sig    = p_bonf < ALPHA
        label  = f"{method_a} vs {method_b}"
        results["pairwise_tests"][label] = {
            "mean_diff":        round(diff, 4),
            "p_value_raw":      round(p_raw, 4),
            "p_value_bonf":     round(p_bonf, 4),
            "n_comparisons":    n_comparisons,
            "alpha_bonferroni": round(alpha_bonferroni, 4),
            "significant_bonf": sig,
        }
        sig_str = "YES ✓" if sig else "no"
        print(f"  {label:<28} {diff:>+8.4f}  {p_raw:>10.4f}  "
              f"{p_bonf:>10.4f}  {sig_str:>6}")

    print("=" * 70)
    print(f"\n  NOTE: After Bonferroni correction (α={alpha_bonferroni:.4f}), "
          f"pairs marked 'no' should be treated as\n"
          f"  statistically indistinguishable at this label-set size (n=67).")

    results["config"] = {
        "n_bootstrap":      N_BOOTSTRAP,
        "top_k":            TOP_K,
        "alpha_family_wise": ALPHA,
        "alpha_per_test":   round(alpha_bonferroni, 4),
        "n_comparisons":    n_comparisons,
        "n_labeled":        len(labeled),
        "n_relevant":       len(relevant_names),
        "bootstrap_population": "labelled_only_67",
        "significance_test": "paired_bootstrap_permutation_two_sided",
        "correction": "bonferroni",
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
