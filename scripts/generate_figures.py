# Regenerates all figures for the paper.
#
# Regenerates ALL figures used in the research paper.
# Run this from the repo root whenever data changes.
#
# Usage:
#   python scripts/generate_figures.py
#
# Output:
#   docs/figures/ndcg_comparison.png   ← included in Overleaf / LaTeX
#   docs/figures/umap_clusters.png     ← included in Overleaf / LaTeX
#   data/final/figures/ndcg_comparison.png  ← copy for archive
#
# Requirements:
#   pip install matplotlib numpy
#   (umap_clusters.png requires data/final/clusters.json from analysis/cluster.py)

import json
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_FIG = os.path.join(ROOT, "docs", "figures")
DATA_FIG = os.path.join(ROOT, "data", "final", "figures")
os.makedirs(DOCS_FIG, exist_ok=True)
os.makedirs(DATA_FIG, exist_ok=True)

# Figure 1 — NDCG Comparison (3-panel)
# Reads: data/final/multi_query_results.json

def generate_ndcg_figure():
    mq_path = os.path.join(ROOT, "data", "final", "multi_query_results.json")
    if not os.path.exists(mq_path):
        print(f"  SKIP: {mq_path} not found")
        return

    with open(mq_path) as f:
        mq = json.load(f)

    methods  = ["jaccard", "tfidf", "bm25", "semantic", "hybrid", "reranked"]
    labels   = ["Jaccard", "TF-IDF", "BM25", "Semantic", "Hybrid", "Reranked"]
    colors   = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b"]

    # Q1 metrics (single-query, from comparison_results)
    q1_ndcg  = [0.1396, 0.0948, 0.4537, 0.5934, 0.3469, 0.5581]
    q1_map   = [0.0665, 0.0250, 0.2750, 0.3877, 0.1733, 0.3463]

    # 5-query aggregates from multi_query_results
    agg      = mq.get("aggregate", {})
    ci       = mq.get("bootstrap_ci", {})
    mean5    = [agg.get(m, {}).get("mean_ndcg", 0) for m in methods]
    std5     = [agg.get(m, {}).get("std_ndcg",  0) for m in methods]

    # Per-query heatmap
    pq       = mq.get("per_query_ndcg", {})
    q_ids    = ["Q1_web_ir_nlp", "Q2_cv_robotics", "Q3_security_privacy",
                "Q4_systems_distributed", "Q5_bio_health"]
    q_labels = ["Q1\nWeb IR", "Q2\nCV/Robot", "Q3\nSecurity",
                "Q4\nDistrib.", "Q5\nBio/Health"]
    data     = np.array([[pq.get(q, {}).get(m, 0) or 0
                          for m in methods] for q in q_ids])

    x = np.arange(len(methods))
    w = 0.38

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Retrieval Method Comparison — Academic Advisor Discovery",
                 fontsize=13, fontweight="bold", y=1.01)

    # Panel 1: Q1 NDCG + MAP
    ax = axes[0]
    b1 = ax.bar(x - w/2, q1_ndcg, w, label="NDCG@10", color=colors, alpha=0.88)
    ax.bar(x + w/2,  q1_map,  w, label="MAP@10",  color=colors, alpha=0.45,
           hatch="//")
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 0.76)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Q1 Single-Query\n(NDCG@10 & MAP@10)", fontsize=10)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: 5-query mean ± std
    ax = axes[1]
    best = mean5.index(max(mean5)) if mean5 else 0
    bars = ax.bar(x, mean5, color=colors, alpha=0.88, width=0.55,
                  yerr=std5, capsize=4,
                  error_kw={"linewidth": 1.2, "capthick": 1.2})
    ax.bar(best, mean5[best], color=colors[best], alpha=0.88, width=0.55,
           edgecolor="gold", linewidth=2.5)
    for bar, m in zip(bars, mean5):
        if m:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.022,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.annotate("TF-IDF < all others\n(p_bonf < 0.001)",
                xy=(1, 0.246), xytext=(3.2, 0.12), fontsize=8, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 0.76)
    ax.set_ylabel("Mean NDCG@10", fontsize=10)
    ax.set_title("5-Query Mean NDCG@10\n(±1 std, gold border = best)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 3: per-query heatmap
    ax = axes[2]
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=0.75)
    for i in range(len(q_labels)):
        for j in range(len(labels)):
            v  = data[i, j]
            fw = "bold" if v == data[i].max() else "normal"
            col = "white" if v < 0.15 or v > 0.68 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color=col, fontweight=fw)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(q_labels)))
    ax.set_yticklabels(q_labels, fontsize=9)
    ax.set_title("NDCG@10 Per Query\n(bold = best method per row)", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="NDCG@10")

    plt.tight_layout()

    out_docs = os.path.join(DOCS_FIG, "ndcg_comparison.png")
    out_data = os.path.join(DATA_FIG, "ndcg_comparison.png")
    plt.savefig(out_docs, dpi=150, bbox_inches="tight", facecolor="white")
    plt.savefig(out_data, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ ndcg_comparison.png saved → docs/figures/ and data/final/figures/")

# Figure 2 — UMAP Cluster Map
# Reads: data/final/clusters.json  (from analysis/cluster.py)

def generate_umap_figure():
    clusters_path = os.path.join(ROOT, "data", "final", "clusters.json")
    if not os.path.exists(clusters_path):
        print(f"  SKIP: {clusters_path} not found — run analysis/cluster.py first")
        return

    with open(clusters_path, encoding="utf-8") as f:
        data = json.load(f)

    profs  = data["professors"]
    x      = [p["umap_x"]    for p in profs]
    y      = [p["umap_y"]    for p in profs]
    clust  = [p["cluster_id"] for p in profs]
    my_pos = data["my_position"]

    cluster_labels = {
        0: "PL / Architecture / SE",
        1: "Scientific Computing / AI / Graphics",
        2: "AI / Agents / NLP",
        3: "AI / ML / Robotics",
        4: "ML / Algorithms / Bio",
        5: "OS / Networking / Cloud",
        6: "CS Education / HCI",
        7: "Quantum / Theory",
        8: "ML / NLP / Data Science",
        9: "Security / Privacy",
    }

    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(figsize=(11, 7))

    for cid in range(10):
        mask = [i for i, c in enumerate(clust) if c == cid]
        ax.scatter([x[i] for i in mask], [y[i] for i in mask],
                   c=[palette[cid]], label=f"C{cid}: {cluster_labels[cid]}",
                   alpha=0.6, s=18)

    ax.scatter(my_pos["umap_x"], my_pos["umap_y"],
               c="red", s=180, zorder=5, marker="*",
               label="Your Research Interest")

    ax.set_title("Research Landscape — UMAP + KMeans "
                 "(768 CS Faculty, 10 Clusters)", fontsize=12)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.85,
              ncol=1, markerscale=1.2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out_docs = os.path.join(DOCS_FIG, "umap_clusters.png")
    out_data = os.path.join(DATA_FIG, "umap_clusters.png")
    plt.savefig(out_docs, dpi=150, bbox_inches="tight", facecolor="white")
    plt.savefig(out_data, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ umap_clusters.png saved → docs/figures/ and data/final/figures/")

# Main

if __name__ == "__main__":
    print("Generating paper figures...")
    print()
    generate_ndcg_figure()
    generate_umap_figure()
    print()
    print("Done. Upload docs/figures/ to Overleaf alongside research_paper.tex")
