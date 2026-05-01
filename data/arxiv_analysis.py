# Analyzes how adding arXiv papers changes rankings.
import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Paths
PROFILE_EMBEDDED  = "data/final/all_professors_embedded.json"
ARXIV_DATA        = "data/final/all_professors_arxiv.json"
OUTPUT_FILE       = "data/final/arxiv_impact.json"
FIGURES_DIR       = "data/final/figures"

os.makedirs(FIGURES_DIR, exist_ok=True)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Dark theme colors (same as analyze.py)
COLORS = {
    "bg": "#0d1117", "card_bg": "#161b22", "text": "#e6edf3",
    "text_muted": "#8b949e", "accent1": "#58a6ff", "accent2": "#3fb950",
    "accent3": "#d29922", "accent4": "#f78166", "accent5": "#bc8cff",
    "grid": "#21262d",
}

def apply_dark_style(fig, ax):
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["card_bg"])
    ax.tick_params(colors=COLORS["text_muted"], labelsize=9)
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.title.set_color(COLORS["text"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
    ax.grid(True, alpha=0.15, color=COLORS["text_muted"])

def main():
    # Load data
    if not os.path.exists(ARXIV_DATA):
        print("[ERROR] arXiv data not found. Run crawler/arxiv_fetcher.py first.")
        return

    print("[LOAD] Loading arXiv-enriched professor data...")
    with open(ARXIV_DATA, "r", encoding="utf-8") as f:
        professors = json.load(f)
    print(f"[INFO] {len(professors)} professors loaded")

    print("[LOAD] Loading current embeddings (profile-only)...")
    with open(PROFILE_EMBEDDED, "r", encoding="utf-8") as f:
        embedded_data = json.load(f)

    my_interest = embedded_data["my_interest"]
    my_embedding = np.array(embedded_data["my_embedding"]).reshape(1, -1)
    profile_profs = embedded_data["professors"]

    # arXiv Coverage Stats
    with_papers = [p for p in professors if p.get("arxiv_papers")]
    total_papers = sum(len(p.get("arxiv_papers", [])) for p in professors)
    paper_counts = [len(p.get("arxiv_papers", [])) for p in professors]

    years = []
    categories = []
    for p in professors:
        for paper in p.get("arxiv_papers", []):
            years.append(paper.get("year", 0))
            categories.extend(paper.get("categories", []))

    year_dist = Counter(years)
    cat_dist = Counter(categories)

    print(f"\n[STATS] arXiv Coverage:")
    print(f"  Professors with papers : {len(with_papers)}/{len(professors)} "
          f"({100*len(with_papers)//len(professors)}%)")
    print(f"  Total papers collected : {total_papers}")
    print(f"  Avg papers per prof    : {total_papers/max(len(with_papers),1):.1f}")
    print(f"  Year range             : {min(years) if years else 'N/A'} - {max(years) if years else 'N/A'}")

    # Profile-only scores (already have them)
    profile_embeddings = np.array([p["embedding"] for p in profile_profs])
    profile_scores = cosine_similarity(my_embedding, profile_embeddings)[0]

    # Profile+arXiv scores (need to compute)
    # We need to embed with arXiv text. Import the function.
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from embeddings.embed import build_embedding_text
    from sentence_transformers import SentenceTransformer

    print("\n[MODEL] Loading model for arXiv-enriched embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Build name-to-professor mapping for alignment
    name_to_idx = {}
    for i, p in enumerate(profile_profs):
        key = f"{p['name']}|{p['university']}"
        name_to_idx[key] = i

    # Build arXiv-enriched texts
    arxiv_texts = []
    aligned_indices = []
    for p in professors:
        key = f"{p['name']}|{p['university']}"
        if key in name_to_idx:
            arxiv_texts.append(build_embedding_text(p, use_arxiv=True))
            aligned_indices.append(name_to_idx[key])

    print(f"[EMBED] Embedding {len(arxiv_texts)} professors with arXiv data...")
    arxiv_embeddings = model.encode(
        arxiv_texts, normalize_embeddings=True,
        batch_size=32, show_progress_bar=True
    )

    arxiv_scores_full = np.zeros(len(profile_profs))
    for i, orig_idx in enumerate(aligned_indices):
        arxiv_scores_full[orig_idx] = cosine_similarity(
            my_embedding, arxiv_embeddings[i].reshape(1, -1)
        )[0][0]

    # Compare Rankings
    TOP_K = 10

    profile_ranked = sorted(
        [(i, profile_scores[i]) for i in range(len(profile_profs))],
        key=lambda x: x[1], reverse=True
    )
    arxiv_ranked = sorted(
        [(i, arxiv_scores_full[i]) for i in range(len(profile_profs))],
        key=lambda x: x[1], reverse=True
    )

    top_k_profile = set(profile_profs[i]["name"] for i, _ in profile_ranked[:TOP_K])
    top_k_arxiv = set(profile_profs[i]["name"] for i, _ in arxiv_ranked[:TOP_K])

    overlap = top_k_profile & top_k_arxiv
    profile_only = top_k_profile - top_k_arxiv
    arxiv_only = top_k_arxiv - top_k_profile

    # Score changes for top professors
    score_changes = []
    for i in range(len(profile_profs)):
        change = float(arxiv_scores_full[i] - profile_scores[i])
        if abs(change) > 0.001:  # only meaningful changes
            score_changes.append({
                "name": profile_profs[i]["name"],
                "university": profile_profs[i]["university"],
                "profile_score": round(float(profile_scores[i]), 4),
                "arxiv_score": round(float(arxiv_scores_full[i]), 4),
                "change": round(change, 4),
                "has_papers": bool(professors[i].get("arxiv_papers") if i < len(professors) else False),
            })

    score_changes.sort(key=lambda x: abs(x["change"]), reverse=True)

    # Build Output
    output = {
        "arxiv_coverage": {
            "professors_with_papers": len(with_papers),
            "total_professors": len(professors),
            "coverage_pct": round(100 * len(with_papers) / len(professors), 1),
            "total_papers": total_papers,
            "avg_papers_per_prof": round(total_papers / max(len(with_papers), 1), 1),
            "year_distribution": dict(sorted(year_dist.items())),
            "top_categories": [
                {"category": cat, "count": cnt}
                for cat, cnt in cat_dist.most_common(15)
            ],
        },
        "ranking_comparison": {
            "top_k": TOP_K,
            "overlap_count": len(overlap),
            "overlap_names": sorted(overlap),
            "profile_only": sorted(profile_only),
            "arxiv_enriched_only": sorted(arxiv_only),
            "top_k_profile": [
                {"rank": r+1, "name": profile_profs[i]["name"],
                 "university": profile_profs[i]["university"],
                 "score": round(float(profile_scores[i]), 4)}
                for r, (i, _) in enumerate(profile_ranked[:TOP_K])
            ],
            "top_k_arxiv": [
                {"rank": r+1, "name": profile_profs[i]["name"],
                 "university": profile_profs[i]["university"],
                 "score": round(float(arxiv_scores_full[i]), 4)}
                for r, (i, _) in enumerate(arxiv_ranked[:TOP_K])
            ],
        },
        "biggest_score_changes": score_changes[:20],
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] arXiv impact analysis -> {OUTPUT_FILE}")

    # Print Summary
    print(f"\n{'='*55}")
    print(f"ARXIV INTEGRATION IMPACT")
    print(f"{'='*55}")
    print(f"  Top-{TOP_K} overlap: {len(overlap)}/{TOP_K} professors shared")
    print(f"  Ranking changed for {TOP_K - len(overlap)} positions")

    if profile_only:
        print(f"\n  Dropped from top-{TOP_K} (profile-only had, arXiv didn't):")
        for name in sorted(profile_only):
            print(f"    - {name}")

    if arxiv_only:
        print(f"\n  New in top-{TOP_K} (arXiv surfaced):")
        for name in sorted(arxiv_only):
            print(f"    + {name}")

    print(f"\n  Biggest score changes:")
    for c in score_changes[:5]:
        direction = "+" if c["change"] > 0 else ""
        print(f"    {c['name']}: {c['profile_score']:.4f} -> {c['arxiv_score']:.4f} "
              f"({direction}{c['change']:.4f})")

    print(f"{'='*55}")

    # Charts
    if HAS_MPL and years:
        # 1. Year distribution
        fig, ax = plt.subplots(figsize=(8, 4.5))
        apply_dark_style(fig, ax)
        sorted_years = sorted(year_dist.keys())
        counts = [year_dist[y] for y in sorted_years]
        ax.bar(sorted_years, counts, color=COLORS["accent1"], alpha=0.85,
               edgecolor=COLORS["bg"], linewidth=1)
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel("Number of Papers", fontsize=11)
        ax.set_title("arXiv Papers by Year", fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "arxiv_year_dist.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [CHART] arxiv_year_dist.png")

        # 2. Score change scatter
        fig, ax = plt.subplots(figsize=(7, 7))
        apply_dark_style(fig, ax)
        ax.scatter(profile_scores, arxiv_scores_full,
                   c=COLORS["accent5"], alpha=0.4, s=20, edgecolors='none')
        lims = [0, max(max(profile_scores), max(arxiv_scores_full)) * 1.05]
        ax.plot(lims, lims, '--', color=COLORS["accent4"], alpha=0.5, linewidth=1)
        ax.set_xlabel("Profile-Only Score", fontsize=11)
        ax.set_ylabel("Profile+arXiv Score", fontsize=11)
        ax.set_title("Score Change: Profile vs Profile+arXiv",
                      fontsize=14, fontweight='bold', pad=15)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "arxiv_score_scatter.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [CHART] arxiv_score_scatter.png")

        # 3. Papers per professor histogram
        fig, ax = plt.subplots(figsize=(8, 4.5))
        apply_dark_style(fig, ax)
        ax.hist(paper_counts, bins=range(0, max(paper_counts)+2),
                color=COLORS["accent2"], alpha=0.85, edgecolor=COLORS["bg"], linewidth=1,
                align='left')
        ax.set_xlabel("Number of arXiv Papers Found", fontsize=11)
        ax.set_ylabel("Number of Professors", fontsize=11)
        ax.set_title("arXiv Papers per Professor", fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "arxiv_papers_hist.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [CHART] arxiv_papers_hist.png")

if __name__ == "__main__":
    main()
