# UMAP dimensionality reduction + KMeans clustering.
# Groups professors into research communities.
import json
import os
import re
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Config
INPUT_FILE   = "data/final/all_professors_embedded.json"
OUTPUT_FILE  = "data/final/clusters.json"
N_CLUSTERS   = 10
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST  = 0.2
RANDOM_STATE   = 42

os.makedirs("data/final", exist_ok=True)

def load_data():
    """Load embedded professor data."""
    print("[LOAD] Loading embedded professor data...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    professors = data["professors"]
    my_interest = data["my_interest"]
    my_embedding = data.get("my_embedding", None)

    # Extract embeddings matrix
    embeddings = []
    valid_profs = []
    for p in professors:
        emb = p.get("embedding")
        if emb and len(emb) > 0:
            embeddings.append(emb)
            valid_profs.append(p)

    print(f"[INFO] {len(valid_profs)} professors with embeddings")
    return valid_profs, np.array(embeddings), my_interest, my_embedding

def run_umap(embeddings, my_embedding=None):
    """Reduce embeddings to 2D using UMAP."""
    print(f"[UMAP] Reducing {embeddings.shape[0]} × {embeddings.shape[1]} → 2D...")

    import umap

    # If we have user embedding, append it for projection
    if my_embedding is not None:
        all_embeddings = np.vstack([embeddings, [my_embedding]])
    else:
        all_embeddings = embeddings

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=RANDOM_STATE,
    )

    coords_2d = reducer.fit_transform(all_embeddings)

    if my_embedding is not None:
        prof_coords = coords_2d[:-1]
        user_coords = coords_2d[-1].tolist()
    else:
        prof_coords = coords_2d
        user_coords = None

    print(f"[UMAP] Done. X range: [{prof_coords[:,0].min():.2f}, {prof_coords[:,0].max():.2f}]")
    return prof_coords, user_coords

def run_kmeans(embeddings, n_clusters=N_CLUSTERS):
    """Cluster professors using KMeans on full-dimensional embeddings."""
    print(f"[KMEANS] Clustering into {n_clusters} groups...")

    # Normalize embeddings for better clustering
    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=10,
        max_iter=300,
    )
    labels = kmeans.fit_predict(scaled)

    # Compute cluster sizes
    cluster_sizes = Counter(labels)
    for cid, size in sorted(cluster_sizes.items()):
        print(f"  Cluster {cid}: {size} professors")

    return labels

def label_clusters(professors, cluster_labels):
    """Auto-label clusters using most frequent research terms."""
    print("[LABEL] Auto-labeling clusters...")

    cluster_terms = {}
    n_clusters = max(cluster_labels) + 1

    for cid in range(n_clusters):
        # Get all research areas for this cluster
        members = [p for p, c in zip(professors, cluster_labels) if c == cid]
        all_terms = []

        for p in members:
            # Use research_interests or parse research_text
            interests = p.get("research_interests", [])
            if interests:
                all_terms.extend(interests)
            elif p.get("research_text"):
                all_terms.extend([
                    t.strip() for t in p["research_text"].split(",")
                    if len(t.strip()) > 2
                ])

        # Clean and normalize terms
        cleaned = []
        for term in all_terms:
            # Strip parenthetical abbreviations
            clean = re.sub(r'\s*\([A-Z]{2,}\)', '', term).strip()
            if clean and len(clean) > 2:
                cleaned.append(clean.lower())

        term_counts = Counter(cleaned)
        top_terms = [t for t, _ in term_counts.most_common(3)]

        # Create readable label
        if top_terms:
            label = " / ".join(t.title() for t in top_terms)
        else:
            label = f"Cluster {cid}"

        cluster_terms[cid] = {
            "label": label,
            "top_terms": [t for t, _ in term_counts.most_common(8)],
            "size": len(members),
            "universities": dict(Counter(m["university"] for m in members)),
        }
        print(f"  Cluster {cid}: {label} ({len(members)} members)")

    return cluster_terms

def compute_cluster_stats(professors, cluster_labels, embeddings):
    """Compute statistics for each cluster."""
    n_clusters = max(cluster_labels) + 1
    stats = {}

    for cid in range(n_clusters):
        mask = cluster_labels == cid
        cluster_embs = embeddings[mask]

        # Compute centroid and average distance
        centroid = cluster_embs.mean(axis=0)
        distances = np.linalg.norm(cluster_embs - centroid, axis=1)

        stats[cid] = {
            "avg_distance": float(distances.mean()),
            "std_distance": float(distances.std()),
            "cohesion": float(1.0 / (1.0 + distances.mean())),  # higher = tighter
        }

    return stats

def main():
    # 1. Load data
    professors, embeddings, my_interest, my_embedding = load_data()

    # 2. Run UMAP
    coords_2d, user_coords = run_umap(embeddings, my_embedding)

    # 3. Run KMeans
    cluster_labels = run_kmeans(embeddings)

    # 4. Auto-label clusters
    cluster_info = label_clusters(professors, cluster_labels)

    # 5. Compute cluster stats
    cluster_stats = compute_cluster_stats(professors, cluster_labels, embeddings)

    # 6. Build output
    professor_clusters = []
    for i, (prof, label) in enumerate(zip(professors, cluster_labels)):
        professor_clusters.append({
            "name": prof["name"],
            "university": prof["university"],
            "cluster_id": int(label),
            "umap_x": float(coords_2d[i, 0]),
            "umap_y": float(coords_2d[i, 1]),
        })

    output = {
        "config": {
            "n_clusters": N_CLUSTERS,
            "umap_neighbors": UMAP_NEIGHBORS,
            "umap_min_dist": UMAP_MIN_DIST,
            "total_professors": len(professors),
        },
        "my_interest": my_interest,
        "my_position": {
            "umap_x": user_coords[0] if user_coords else None,
            "umap_y": user_coords[1] if user_coords else None,
        },
        "clusters": {
            str(k): {**v, **cluster_stats.get(k, {})}
            for k, v in cluster_info.items()
        },
        "professors": professor_clusters,
    }

    # 7. Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {OUTPUT_FILE}")

    # 8. Summary
    print(f"\n{'='*55}")
    print(f"CLUSTERING SUMMARY")
    print(f"{'='*55}")
    print(f"  Professors   : {len(professors)}")
    print(f"  Clusters     : {N_CLUSTERS}")
    print(f"  UMAP dims    : 2D (from {embeddings.shape[1]}D)")
    if user_coords:
        print(f"  Your position: ({user_coords[0]:.2f}, {user_coords[1]:.2f})")

    # Find which cluster user falls into
    if user_coords:
        user_np = np.array(user_coords)
        cluster_centers_2d = {}
        for i, (_, label) in enumerate(zip(professors, cluster_labels)):
            cid = int(label)
            if cid not in cluster_centers_2d:
                cluster_centers_2d[cid] = []
            cluster_centers_2d[cid].append(coords_2d[i])

        best_cluster = min(
            cluster_centers_2d.keys(),
            key=lambda c: np.linalg.norm(
                np.mean(cluster_centers_2d[c], axis=0) - user_np
            )
        )
        print(f"  Nearest cluster: {best_cluster} — {cluster_info[best_cluster]['label']}")

    print(f"\n  Cluster breakdown:")
    for cid in sorted(cluster_info.keys()):
        info = cluster_info[cid]
        print(f"    [{cid:2d}] {info['label']:<45} ({info['size']} profs)")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
