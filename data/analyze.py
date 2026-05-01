# Dataset statistics and charts.
# Generates summary stats and matplotlib visualizations.
import json
import os
import re
from collections import Counter

# Paths
INPUT_FILE     = "data/final/all_professors.json"
STATS_FILE     = "data/final/statistics.json"
FIGURES_DIR    = "data/final/figures"
COMPARISON_FILE = "data/final/comparison_results.json"

os.makedirs(FIGURES_DIR, exist_ok=True)

# Try to import matplotlib — graceful fallback if not installed
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARN] matplotlib not installed — skipping chart generation")
    print("[WARN] Install with: pip install matplotlib\n")

# Chart Styling

# Dark premium color palette
COLORS = {
    "bg"        : "#0d1117",
    "card_bg"   : "#161b22",
    "text"      : "#e6edf3",
    "text_muted": "#8b949e",
    "accent1"   : "#58a6ff",   # blue
    "accent2"   : "#3fb950",   # green
    "accent3"   : "#d29922",   # amber
    "accent4"   : "#f78166",   # orange
    "accent5"   : "#bc8cff",   # purple
    "accent6"   : "#ff7b72",   # red
    "grid"      : "#21262d",
}

PALETTE = [
    COLORS["accent1"], COLORS["accent2"], COLORS["accent3"],
    COLORS["accent4"], COLORS["accent5"], COLORS["accent6"],
]

def apply_dark_style(fig, ax):
    """Apply consistent dark theme to a matplotlib figure."""
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["card_bg"])
    ax.tick_params(colors=COLORS["text_muted"], labelsize=9)
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.title.set_color(COLORS["text"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
    ax.grid(True, alpha=0.15, color=COLORS["text_muted"])

# Statistics Computation

def compute_statistics(professors: list) -> dict:
    """Compute comprehensive dataset statistics."""

    total = len(professors)

    # University breakdown
    uni_counts = Counter(p["university"] for p in professors)

    # Data coverage
    has_email      = sum(1 for p in professors if p.get("email"))
    has_bio        = sum(1 for p in professors if p.get("biography"))
    has_research   = sum(1 for p in professors if p.get("research_text"))
    has_education  = sum(1 for p in professors if p.get("education"))
    has_awards     = sum(1 for p in professors if p.get("awards"))
    has_pubs       = sum(1 for p in professors if p.get("publications"))
    has_website    = sum(1 for p in professors if p.get("personal_website"))
    has_scholar    = sum(1 for p in professors if p.get("google_scholar"))
    has_phone      = sum(1 for p in professors if p.get("phone"))
    has_office     = sum(1 for p in professors if p.get("office"))

    coverage = {
        "email"            : {"count": has_email,    "pct": round(100*has_email/total, 1)},
        "biography"        : {"count": has_bio,      "pct": round(100*has_bio/total, 1)},
        "research_text"    : {"count": has_research,  "pct": round(100*has_research/total, 1)},
        "education"        : {"count": has_education, "pct": round(100*has_education/total, 1)},
        "awards"           : {"count": has_awards,    "pct": round(100*has_awards/total, 1)},
        "publications"     : {"count": has_pubs,      "pct": round(100*has_pubs/total, 1)},
        "personal_website" : {"count": has_website,   "pct": round(100*has_website/total, 1)},
        "google_scholar"   : {"count": has_scholar,   "pct": round(100*has_scholar/total, 1)},
        "phone"            : {"count": has_phone,     "pct": round(100*has_phone/total, 1)},
        "office"           : {"count": has_office,    "pct": round(100*has_office/total, 1)},
    }

    # Research areas
    all_areas = []
    for p in professors:
        areas = p.get("research_interests", [])
        if isinstance(areas, list):
            all_areas.extend(areas)

    area_counts = Counter(all_areas)

    # Normalize similar areas (case-insensitive dedup + strip abbreviations)
    normalized_areas = Counter()
    area_map = {}  # normalized_key → canonical form (shortest clean version)
    for area, count in area_counts.items():
        # Strip parenthetical abbreviations like (AI), (DBMS), (HCI)
        clean = re.sub(r'\s*\([A-Z]{2,}\)', '', area).strip()
        key = clean.lower().strip()
        if key in area_map:
            normalized_areas[area_map[key]] += count
        else:
            area_map[key] = clean
            normalized_areas[clean] = count

    # Research text length stats
    research_lengths = [len(p.get("research_text", "")) for p in professors]
    bio_lengths = [len(p.get("biography", "")) for p in professors]

    def length_stats(lengths):
        if not lengths:
            return {}
        import statistics
        return {
            "mean"   : round(statistics.mean(lengths), 1),
            "median" : round(statistics.median(lengths), 1),
            "stdev"  : round(statistics.stdev(lengths), 1) if len(lengths) > 1 else 0,
            "min"    : min(lengths),
            "max"    : max(lengths),
            "empty"  : sum(1 for l in lengths if l == 0),
        }

    # Title distribution
    title_categories = Counter()
    for p in professors:
        title = p.get("title", "").lower()
        if "emerit" in title:
            title_categories["Emeritus"] += 1
        elif "assistant" in title:
            title_categories["Assistant Professor"] += 1
        elif "associate" in title:
            title_categories["Associate Professor"] += 1
        elif "professor" in title or "faculty" in title:
            title_categories["Full Professor"] += 1
        elif "lecturer" in title:
            title_categories["Lecturer"] += 1
        else:
            title_categories["Other"] += 1

    # Per-university coverage
    uni_coverage = {}
    for uni_name in uni_counts:
        uni_profs = [p for p in professors if p["university"] == uni_name]
        n = len(uni_profs)
        uni_coverage[uni_name] = {
            "total"         : n,
            "has_email"     : sum(1 for p in uni_profs if p.get("email")),
            "has_bio"       : sum(1 for p in uni_profs if p.get("biography")),
            "has_research"  : sum(1 for p in uni_profs if p.get("research_text")),
            "has_pubs"      : sum(1 for p in uni_profs if p.get("publications")),
            "has_scholar"   : sum(1 for p in uni_profs if p.get("google_scholar")),
        }

    return {
        "total_professors"     : total,
        "universities"         : dict(uni_counts.most_common()),
        "coverage"             : coverage,
        "research_text_length" : length_stats(research_lengths),
        "biography_length"     : length_stats(bio_lengths),
        "total_unique_areas"   : len(normalized_areas),
        "total_area_mentions"  : sum(normalized_areas.values()),
        "top_30_areas"         : [
            {"area": area, "count": count}
            for area, count in normalized_areas.most_common(30)
        ],
        "title_distribution"   : dict(title_categories.most_common()),
        "per_university"       : uni_coverage,
    }

# Chart Generation

def chart_university_breakdown(stats: dict) -> None:
    """Bar chart: professors per university."""
    if not HAS_MATPLOTLIB:
        return

    unis = stats["universities"]
    names = list(unis.keys())
    counts = list(unis.values())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    apply_dark_style(fig, ax)

    bars = ax.barh(names, counts, color=PALETTE[:len(names)], height=0.5,
                    edgecolor=COLORS["bg"], linewidth=1.5)

    # Value labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                str(count), va='center', ha='left',
                color=COLORS["text"], fontsize=11, fontweight='bold')

    ax.set_xlabel("Number of Professors", fontsize=11)
    ax.set_title("Professors per University", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, max(counts) * 1.15)
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "university_breakdown.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [CHART] {path}")

def chart_data_coverage(stats: dict) -> None:
    """Horizontal bar chart: data field coverage percentages."""
    if not HAS_MATPLOTLIB:
        return

    coverage = stats["coverage"]
    # Order by percentage descending
    items = sorted(coverage.items(), key=lambda x: x[1]["pct"], reverse=True)
    labels = [k.replace("_", " ").title() for k, _ in items]
    pcts = [v["pct"] for _, v in items]

    fig, ax = plt.subplots(figsize=(8, 5))
    apply_dark_style(fig, ax)

    bars = ax.barh(labels, pcts, color=PALETTE[0], height=0.55, alpha=0.85,
                    edgecolor=COLORS["bg"], linewidth=1)

    for bar, pct in zip(bars, pcts):
        color = COLORS["accent2"] if pct >= 80 else (
            COLORS["accent3"] if pct >= 50 else COLORS["accent6"])
        bar.set_color(color)
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{pct:.0f}%", va='center', ha='left',
                color=COLORS["text_muted"], fontsize=9)

    ax.set_xlabel("Coverage (%)", fontsize=11)
    ax.set_title("Data Field Coverage", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, 105)
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "data_coverage.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [CHART] {path}")

def chart_research_areas(stats: dict) -> None:
    """Horizontal bar chart: top 15 research areas."""
    if not HAS_MATPLOTLIB:
        return

    top_areas = stats["top_30_areas"][:15]
    labels = [a["area"] for a in reversed(top_areas)]
    counts = [a["count"] for a in reversed(top_areas)]

    fig, ax = plt.subplots(figsize=(9, 6))
    apply_dark_style(fig, ax)

    # Gradient colors based on count
    max_count = max(counts)
    colors = [plt.cm.cool(0.3 + 0.6 * c / max_count) for c in counts]

    bars = ax.barh(labels, counts, color=colors, height=0.6,
                    edgecolor=COLORS["bg"], linewidth=1)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(count), va='center', ha='left',
                color=COLORS["text_muted"], fontsize=9)

    ax.set_xlabel("Number of Professors", fontsize=11)
    ax.set_title("Top 15 Research Areas", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, max(counts) * 1.15)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "research_areas.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [CHART] {path}")

def chart_title_distribution(stats: dict) -> None:
    """Donut chart: faculty rank distribution."""
    if not HAS_MATPLOTLIB:
        return

    titles = stats["title_distribution"]
    labels = list(titles.keys())
    counts = list(titles.values())

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(COLORS["bg"])

    wedges, texts, autotexts = ax.pie(
        counts, labels=None, autopct='%1.0f%%',
        colors=PALETTE[:len(labels)],
        startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.45, edgecolor=COLORS["bg"], linewidth=2),
    )

    for t in autotexts:
        t.set_color(COLORS["text"])
        t.set_fontsize(10)
        t.set_fontweight('bold')

    # Legend
    legend = ax.legend(
        wedges, [f"{l} ({c})" for l, c in zip(labels, counts)],
        loc="center", fontsize=9,
        frameon=False,
    )
    for text in legend.get_texts():
        text.set_color(COLORS["text_muted"])

    ax.set_title("Faculty Rank Distribution", fontsize=14,
                  fontweight='bold', color=COLORS["text"], pad=20)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "title_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [CHART] {path}")

def chart_score_distribution(comparison_file: str) -> None:
    """Histogram: score distributions for all matching methods."""
    if not HAS_MATPLOTLIB:
        return
    if not os.path.exists(comparison_file):
        print(f"  [SKIP] No comparison file for score distribution chart")
        return

    with open(comparison_file, "r", encoding="utf-8") as f:
        comp = json.load(f)

    all_ranked = comp.get("all_ranked", [])
    if not all_ranked:
        return

    methods = [
        ("semantic_score", "Semantic", COLORS["accent1"]),
        ("tfidf_score",    "TF-IDF",   COLORS["accent2"]),
        ("jaccard_score",  "Jaccard",  COLORS["accent3"]),
        ("hybrid_score",   "Hybrid",   COLORS["accent5"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("Score Distributions by Method", fontsize=16,
                  fontweight='bold', color=COLORS["text"], y=0.98)

    for ax, (key, label, color) in zip(axes.flatten(), methods):
        apply_dark_style(fig, ax)
        scores = [r[key] for r in all_ranked]

        ax.hist(scores, bins=30, color=color, alpha=0.8,
                edgecolor=COLORS["bg"], linewidth=0.5)
        ax.set_title(label, fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel("Score", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)

        # Add mean line
        import statistics
        mean_val = statistics.mean(scores)
        ax.axvline(mean_val, color=COLORS["accent6"], linestyle='--',
                    linewidth=1.5, alpha=0.7)
        ax.text(mean_val, ax.get_ylim()[1] * 0.9, f'μ={mean_val:.3f}',
                color=COLORS["accent6"], fontsize=8, ha='left')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(FIGURES_DIR, "score_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [CHART] {path}")

def chart_per_university_coverage(stats: dict) -> None:
    """Grouped bar chart: per-university data coverage."""
    if not HAS_MATPLOTLIB:
        return

    uni_cov = stats["per_university"]
    unis = list(uni_cov.keys())
    # Shorten names
    short_names = []
    for u in unis:
        if "Texas" in u:
            short_names.append("TAMU")
        elif "Berkeley" in u:
            short_names.append("Berkeley")
        elif "UMass" in u:
            short_names.append("UMass")
        else:
            short_names.append(u[:12])

    fields = ["has_email", "has_bio", "has_research", "has_pubs", "has_scholar"]
    field_labels = ["Email", "Biography", "Research", "Publications", "Scholar"]

    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 5))
    apply_dark_style(fig, ax)

    x = np.arange(len(unis))
    width = 0.15

    for i, (field, label) in enumerate(zip(fields, field_labels)):
        pcts = []
        for uni in unis:
            total = uni_cov[uni]["total"]
            val = uni_cov[uni].get(field, 0)
            pcts.append(100 * val / total if total > 0 else 0)

        bars = ax.bar(x + i * width, pcts, width, label=label,
                       color=PALETTE[i], alpha=0.85,
                       edgecolor=COLORS["bg"], linewidth=0.8)

    ax.set_xlabel("University", fontsize=11)
    ax.set_ylabel("Coverage (%)", fontsize=11)
    ax.set_title("Data Coverage by University", fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(short_names)
    ax.set_ylim(0, 115)

    legend = ax.legend(fontsize=8, loc='upper right', framealpha=0.3)
    for text in legend.get_texts():
        text.set_color(COLORS["text_muted"])
    legend.get_frame().set_facecolor(COLORS["card_bg"])

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "per_university_coverage.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [CHART] {path}")

# Main

def main():
    # Load data
    print("[LOAD] Loading professor data...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        professors = json.load(f)

    print(f"[INFO] {len(professors)} professors loaded\n")

    # Compute statistics
    print("[STATS] Computing dataset statistics...")
    stats = compute_statistics(professors)

    # Save statistics
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] Statistics → {STATS_FILE}\n")

    # Generate charts
    print("[CHARTS] Generating visualizations...")
    chart_university_breakdown(stats)
    chart_data_coverage(stats)
    chart_research_areas(stats)
    chart_title_distribution(stats)
    chart_per_university_coverage(stats)
    chart_score_distribution(COMPARISON_FILE)

    # Print summary
    print(f"\n{'='*55}")
    print(f"DATASET SUMMARY")
    print(f"{'='*55}")
    print(f"  Total professors     : {stats['total_professors']}")
    print(f"  Universities         : {len(stats['universities'])}")
    for uni, count in stats['universities'].items():
        print(f"    • {uni}: {count}")
    print(f"  Unique research areas: {stats['total_unique_areas']}")
    print(f"\n  Data Coverage:")
    for field, info in stats['coverage'].items():
        bar = "█" * int(info['pct'] / 5) + "░" * (20 - int(info['pct'] / 5))
        print(f"    {field:20s} {bar} {info['pct']:5.1f}%  ({info['count']})")
    print(f"\n  Top 5 Research Areas:")
    for item in stats['top_30_areas'][:5]:
        print(f"    • {item['area']}: {item['count']} professors")
    print(f"\n  Faculty Ranks:")
    for title, count in stats['title_distribution'].items():
        print(f"    • {title}: {count}")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
