# LDA topic modeling: discovers latent research topics
# as an alternative view to embedding-based clusters.
import json
import os
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Paths
EMBEDDED_FILE = "data/final/all_professors_embedded.json"
OUTPUT_FILE   = "data/final/lda_topics.json"
N_TOPICS      = 12   # number of latent topics
TOP_WORDS     = 10   # words per topic
MAX_DF        = 0.85  # ignore words in >85% of docs
MIN_DF        = 3     # ignore words in <3 docs

os.makedirs("data/final", exist_ok=True)

def build_text(prof):
    """Build clean text for LDA from professor data."""
    parts = []
    if prof.get("research_text"):
        parts.append(prof["research_text"])
    if prof.get("biography"):
        parts.append(prof["biography"][:500])
    text = " ".join(parts).lower()
    # Remove URLs, emails, special chars
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Custom stop words for academic CS text
CS_STOP_WORDS = [
    'professor', 'assistant', 'associate', 'university', 'department',
    'computer', 'science', 'research', 'phd', 'dr', 'degree', 'received',
    'interests', 'include', 'including', 'work', 'working', 'also',
    'new', 'based', 'using', 'approach', 'approaches', 'problems',
    'students', 'faculty', 'teaching', 'courses', 'joined', 'year',
    'years', 'current', 'currently', 'group', 'lab', 'published',
    'papers', 'journal', 'conference', 'award', 'awards', 'fellow',
    'member', 'institute', 'center', 'program', 'school', 'college',
]

def main():
    print("=" * 60)
    print("  TOPIC MODELING: Latent Dirichlet Allocation")
    print("=" * 60)

    # Load data
    with open(EMBEDDED_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    professors = data["professors"]
    print(f"[DATA] {len(professors)} professors loaded\n")

    # Build text corpus
    texts = [build_text(p) for p in professors]
    names = [p["name"] for p in professors]
    universities = [p["university"] for p in professors]

    # Remove empty documents
    valid = [(i, t) for i, t in enumerate(texts) if len(t.split()) >= 5]
    valid_indices = [v[0] for v in valid]
    valid_texts = [v[1] for v in valid]
    print(f"[FILTER] {len(valid_texts)}/{len(texts)} professors with sufficient text\n")

    # Vectorize
    print("[VECTORIZE] Building document-term matrix...")
    # Build combined stop words list
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    all_stop_words = list(ENGLISH_STOP_WORDS | set(CS_STOP_WORDS))

    vectorizer = CountVectorizer(
        max_df=MAX_DF,
        min_df=MIN_DF,
        stop_words=all_stop_words,
        max_features=5000,
    )

    dtm = vectorizer.fit_transform(valid_texts)
    feature_names = vectorizer.get_feature_names_out()
    print(f"  Vocabulary: {len(feature_names)} terms")
    print(f"  Matrix: {dtm.shape[0]} docs x {dtm.shape[1]} terms\n")

    # Fit LDA
    print(f"[LDA] Fitting {N_TOPICS} topics...")
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        max_iter=50,
        learning_method='online',
        random_state=42,
        n_jobs=-1,
    )
    doc_topic_dist = lda.fit_transform(dtm)
    print(f"  Perplexity: {lda.perplexity(dtm):.2f}")
    print(f"  Log-likelihood: {lda.score(dtm):.2f}\n")

    # Extract topics
    topics = []
    print("=" * 60)
    print("  DISCOVERED TOPICS")
    print("=" * 60)

    for topic_idx, topic in enumerate(lda.components_):
        top_word_indices = topic.argsort()[:-TOP_WORDS-1:-1]
        top_words = [feature_names[i] for i in top_word_indices]
        top_weights = [round(float(topic[i]), 2) for i in top_word_indices]

        # Find professors most associated with this topic
        topic_profs = []
        for j, dist in enumerate(doc_topic_dist):
            if dist[topic_idx] > 0.3:  # threshold for strong association
                orig_idx = valid_indices[j]
                topic_profs.append({
                    "name": names[orig_idx],
                    "university": universities[orig_idx],
                    "weight": round(float(dist[topic_idx]), 3),
                })
        topic_profs.sort(key=lambda x: -x["weight"])

        # Auto-label: use top 3 words
        label = " / ".join(w.title() for w in top_words[:3])

        topic_info = {
            "id": topic_idx,
            "label": label,
            "top_words": top_words,
            "top_weights": top_weights,
            "n_professors": len(topic_profs),
            "top_professors": topic_profs[:5],
        }
        topics.append(topic_info)

        print(f"\n  Topic {topic_idx}: {label}")
        print(f"  Words: {', '.join(top_words)}")
        print(f"  Professors: {len(topic_profs)}")
        if topic_profs:
            for tp in topic_profs[:3]:
                print(f"    - {tp['name']} ({tp['university']}) [{tp['weight']:.3f}]")

    # Build per-professor topic distribution
    prof_topics = []
    for j, dist in enumerate(doc_topic_dist):
        orig_idx = valid_indices[j]
        dominant_topic = int(np.argmax(dist))
        prof_topics.append({
            "name": names[orig_idx],
            "university": universities[orig_idx],
            "dominant_topic": dominant_topic,
            "dominant_topic_label": topics[dominant_topic]["label"],
            "topic_distribution": [round(float(d), 4) for d in dist],
        })

    # University-topic heatmap data
    uni_set = sorted(set(universities))
    uni_topic_counts = {}
    for uni in uni_set:
        counts = [0] * N_TOPICS
        for pt in prof_topics:
            if pt["university"] == uni:
                counts[pt["dominant_topic"]] += 1
        uni_topic_counts[uni] = counts

    # Summary
    print(f"\n\n{'='*60}")
    print("  UNIVERSITY-TOPIC DISTRIBUTION")
    print(f"{'='*60}")
    header = f"  {'University':<30}" + "".join(f"T{i:<3}" for i in range(N_TOPICS))
    print(header)
    print("  " + "-" * (30 + N_TOPICS * 4))
    for uni in uni_set:
        counts = uni_topic_counts[uni]
        row = f"  {uni:<30}" + "".join(f"{c:<4}" for c in counts)
        print(row)

    # Save
    output = {
        "config": {
            "n_topics": N_TOPICS,
            "n_professors": len(valid_texts),
            "vocabulary_size": len(feature_names),
            "perplexity": round(float(lda.perplexity(dtm)), 2),
        },
        "topics": topics,
        "professor_topics": prof_topics,
        "university_topic_counts": uni_topic_counts,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
