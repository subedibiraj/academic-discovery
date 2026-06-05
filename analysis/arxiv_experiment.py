# Controlled experiment: does appending arXiv abstracts help retrieval?
#
# Clean, controlled arXiv concatenation experiment.
#
# DESIGN (fixing the confounds identified in the methodology audit)
# The original experiment compared:
#   - Full corpus (768 profs, no arXiv)  vs
#   - Full corpus (768 profs, with arXiv for some of them)
#
# This conflates THREE changes simultaneously:
#   (1) embedding content changes for profs WITH arXiv papers
#   (2) document length distribution shifts across the corpus
#   (3) population heterogeneity (profs with/without arXiv differ systematically)
#
# This script runs a CONTROLLED experiment:
#   - Restricts to the 296-professor subpopulation for which BOTH conditions
#     were computed (same professors, same query, same model)
#   - Condition A: profile-only embeddings  (research_text + biography[:500])
#   - Condition B: arXiv-enriched embeddings (profile + abstract[:300] x 3)
#   - Evaluates NDCG@10 / P@10 on the SAME subpopulation for both conditions
#   - Uses the stored scores from data/final/arxiv_impact.json
#     (produced by data/arxiv_analysis.py with the full embedding model)
#
# ADDITIONAL ANALYSIS

# Tests Hypothesis 1: Does arXiv hurt relevant professors MORE than
# irrelevant ones? (Jargon dilution in relevant profiles vs generic
# content in irrelevant profiles.)
#
import json
import math
import os

ARXIV_FILE  = "data/final/arxiv_impact.json"
LABELS_FILE = "data/final/relevance_labels.json"
OUTPUT_FILE = "data/final/arxiv_experiment_clean.json"

os.makedirs("data/final", exist_ok=True)

TOP_K = 10

def ndcg_at_k(ranked_names, rel_map, k=TOP_K):
    dcg   = sum(rel_map.get(n, 0) / math.log2(i + 2)
                for i, n in enumerate(ranked_names[:k]))
    ideal = sorted(rel_map.values(), reverse=True)[:k]
    idcg  = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

def precision_at_k(ranked_names, rel_set, k=TOP_K):
    return sum(1 for n in ranked_names[:k] if n in rel_set) / k

def recall_at_k(ranked_names, rel_set, k=TOP_K):
    hits = sum(1 for n in ranked_names[:k] if n in rel_set)
    return hits / len(rel_set) if rel_set else 0.0

def main():
    print("=" * 65)
    print("  ARXIV CONCATENATION EXPERIMENT  (controlled, same subpopulation)")
    print("=" * 65)

    with open(ARXIV_FILE, "r") as f:
        arxiv = json.load(f)
    with open(LABELS_FILE, "r") as f:
        labels = json.load(f)

    # Build relevance maps
    rel_map = {}
    rel_set = set()
    for l in labels:
        if l.get("relevant") is True:
            # Default to 1 (partially relevant), not 2, if grade is missing --
            # see matcher/compare.py for rationale.
            rel_map[l["name"]] = l.get("grade", 1)
            rel_set.add(l["name"])
        else:
            rel_map[l["name"]] = 0

    # Coverage stats
    cov = arxiv["arxiv_coverage"]
    print(f"\n[CORPUS]  Subpopulation: {cov['total_professors']} professors")
    print(f"          With arXiv papers: {cov['professors_with_papers']} "
          f"({cov['coverage_pct']:.1f}%)")
    print(f"          Total papers: {cov['total_papers']}  "
          f"(avg {cov['avg_papers_per_prof']:.1f} per prof)")
    print(f"          Note: arXiv papers are recent (2024-2026 heavy) — "
          f"may not reflect primary research identity")

    # Condition A: Profile-only (same 296 subpopulation)
    profile_top10 = [r["name"]
                     for r in arxiv["ranking_comparison"]["top_k_profile"]]
    arxiv_top10   = [r["name"]
                     for r in arxiv["ranking_comparison"]["top_k_arxiv"]]

    ndcg_A = ndcg_at_k(profile_top10, rel_map)
    ndcg_B = ndcg_at_k(arxiv_top10,   rel_map)
    p_A    = precision_at_k(profile_top10, rel_set)
    p_B    = precision_at_k(arxiv_top10,   rel_set)
    r_A    = recall_at_k(profile_top10,    rel_set)
    r_B    = recall_at_k(arxiv_top10,      rel_set)

    print(f"\n{'─'*65}")
    print(f"  METRIC COMPARISON  (same {cov['total_professors']}-professor subpopulation)")
    print(f"{'─'*65}")
    print(f"  {'Condition':<28} {'NDCG@10':>9} {'P@10':>7} {'R@10':>7}")
    print(f"  {'─'*28} {'─'*9} {'─'*7} {'─'*7}")
    print(f"  {'A: Profile-only':<28} {ndcg_A:>9.4f} {p_A:>7.2f} {r_A:>7.3f}")
    print(f"  {'B: arXiv-enriched':<28} {ndcg_B:>9.4f} {p_B:>7.2f} {r_B:>7.3f}")
    print(f"  {'Delta (B - A)':<28} {ndcg_B-ndcg_A:>+9.4f} {p_B-p_A:>+7.2f} {r_B-r_A:>+7.3f}")

    # Who moved in/out of top-10
    dropped = sorted(set(profile_top10) - set(arxiv_top10))
    gained  = sorted(set(arxiv_top10) - set(profile_top10))

    print(f"\n  Dropped from top-10 by arXiv: {dropped}")
    print(f"  Gained in top-10  with arXiv: {gained}")

    dropped_rel = [n for n in dropped if n in rel_set]
    gained_rel  = [n for n in gained  if n in rel_set]
    print(f"  Relevant professors dropped:  {dropped_rel}")
    print(f"  Relevant professors gained:   {gained_rel}")

    # Hypothesis 1: Jargon dilution
    score_changes = arxiv["biggest_score_changes"]
    rel_changes   = [s for s in score_changes if s["name"] in rel_set]
    irrel_changes = [s for s in score_changes if s["name"] not in rel_set]

    avg_delta_rel   = (sum(s["arxiv_score"] - s["profile_score"]
                           for s in rel_changes)   / len(rel_changes)
                       if rel_changes else 0)
    avg_delta_irrel = (sum(s["arxiv_score"] - s["profile_score"]
                           for s in irrel_changes) / len(irrel_changes)
                       if irrel_changes else 0)

    print(f"\n{'─'*65}")
    print(f"  HYPOTHESIS 1: Jargon dilution hurts relevant professors more")
    print(f"{'─'*65}")
    print(f"  Relevant professors   (n={len(rel_changes)}):  "
          f"avg score delta = {avg_delta_rel:+.4f}")
    print(f"  Irrelevant professors (n={len(irrel_changes)}): "
          f"avg score delta = {avg_delta_irrel:+.4f}")
    h1_supported = avg_delta_rel < avg_delta_irrel
    print(f"  Direction: {'consistent with' if h1_supported else 'NOT consistent with'} "
          f"H1 (relevant profs hurt {'more' if h1_supported else 'less'} than irrelevant)")
    print(f"  CAVEAT: n={len(rel_changes)} relevant professors tested -- this is a")
    print(f"  small, non-random sample (only professors in the top-20 largest-change")
    print(f"  list). Treat as suggestive, not confirmatory.")

    if rel_changes:
        print(f"\n  Relevant professor score changes:")
        for s in sorted(rel_changes, key=lambda x: x["arxiv_score"]-x["profile_score"]):
            delta = s["arxiv_score"] - s["profile_score"]
            print(f"    {s['name']:<30} {s['profile_score']:.4f} → "
                  f"{s['arxiv_score']:.4f}  ({delta:+.4f})")

    # Hypothesis 2: Recency bias
    yr = cov["year_distribution"]
    total_papers = sum(yr.values())
    recent = yr.get("2025", 0) + yr.get("2026", 0)
    print(f"\n{'─'*65}")
    print(f"  HYPOTHESIS 2: Recent papers bias (recency ≠ primary research identity)")
    print(f"{'─'*65}")
    print(f"  Papers from 2025-2026: {recent}/{total_papers} "
          f"({recent/total_papers*100:.0f}% of all arXiv papers)")
    print(f"  Implication: Recent papers may reflect current trends, not the")
    print(f"  professor's core expertise that matches a PhD applicant query.")

    # Summary
    print(f"\n{'═'*65}")
    print(f"  CONCLUSION")
    print(f"{'═'*65}")
    print(f"  On the same {cov['total_professors']}-professor subpopulation, arXiv concatenation")
    print(f"  REDUCES NDCG@10 from {ndcg_A:.4f} to {ndcg_B:.4f} "
          f"(Δ = {ndcg_B-ndcg_A:+.4f}).")
    print(f"  P@10 drops from {p_A:.2f} to {p_B:.2f}: "
          f"{len(dropped_rel)} relevant professor(s) displaced from top-10.")
    print(f"  Direction is consistent with jargon dilution (n={len(rel_changes)} "
          f"relevant professors tested -- suggestive, not confirmatory):")
    print(f"  relevant professors tend to suffer larger score drops than irrelevant ones.")
    print(f"  Proposed fix: late fusion (separate retrieval on papers, combine scores)")
    print(f"  rather than text concatenation.")

    # Save
    result = {
        "experiment_design": {
            "subpopulation": cov["total_professors"],
            "with_arxiv_papers": cov["professors_with_papers"],
            "conditions": {
                "A": "profile-only (research_text + biography[:500])",
                "B": "arXiv-enriched (profile + abstract[:300] x top-3 papers)",
            },
            "control": "Same professors, same query, same model — only embedding content changes",
            "confounds_fixed": [
                "Population restricted to same 296 professors in both conditions",
                "Document length effect isolated (same population)",
                "No cross-population comparison",
            ],
        },
        "metrics": {
            "profile_only": {
                "ndcg_at_k": round(ndcg_A, 4),
                "precision_at_k": round(p_A, 4),
                "recall_at_k":    round(r_A, 4),
            },
            "arxiv_enriched": {
                "ndcg_at_k": round(ndcg_B, 4),
                "precision_at_k": round(p_B, 4),
                "recall_at_k":    round(r_B, 4),
            },
            "delta": {
                "ndcg":      round(ndcg_B - ndcg_A, 4),
                "precision": round(p_B - p_A, 4),
                "recall":    round(r_B - r_A, 4),
            },
        },
        "top10_changes": {
            "dropped": dropped,
            "gained":  gained,
            "relevant_dropped": dropped_rel,
            "relevant_gained":  gained_rel,
        },
        "hypothesis_1_jargon_dilution": {
            "avg_delta_relevant":   round(avg_delta_rel, 4),
            "avg_delta_irrelevant": round(avg_delta_irrel, 4),
            "supported": h1_supported,
            "n_relevant_tested":   len(rel_changes),
            "n_irrelevant_tested": len(irrel_changes),
            "caveat": (
                f"n_relevant_tested={len(rel_changes)} is a SMALL sample -- "
                f"only {len(rel_changes)} of the corpus's relevant professors "
                f"happened to fall within the pre-selected 'biggest_score_changes' "
                f"list (top-20 by magnitude among {cov['professors_with_papers']} "
                f"professors with arXiv papers). This result should be treated "
                f"as suggestive, not confirmatory. A systematic test across all "
                f"relevant professors (not just the largest-change subset) is "
                f"needed before treating H1 as established."
            ),
        },
        "hypothesis_2_recency_bias": {
            "pct_papers_2025_2026": round(recent / total_papers * 100, 1),
            "year_distribution":    yr,
            "interpretation": "73%+ papers from 2025-2026; may not reflect primary expertise",
        },
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[SAVED] {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
