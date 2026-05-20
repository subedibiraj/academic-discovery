# Encodes Q2-Q5 query strings with sentence-transformers.
#
# Generates semantic similarity scores for Q2-Q5 using all-MiniLM-L6-v2.
# Requires HuggingFace access (~90MB download on first run, then cached).
#
# Usage:
#   python analysis/embed_queries.py
#
# Output:
#   data/final/query_embeddings.json   <- consumed by multi_query_eval.py
#
# After running this, re-run multi_query_eval.py to get full 5-query results.
#
import json, os, sys
import numpy as np

PROFESSORS_FILE = "data/final/all_professors.json"
OUTPUT_FILE     = "data/final/query_embeddings.json"
MODEL_NAME      = "all-MiniLM-L6-v2"
# batch_size=32 matches embeddings/embed.py exactly. Note: batch_size is a
# throughput/hardware parameter only -- sentence-transformers processes each
# input independently (attention masking prevents cross-example leakage
# within a batch), so it does not change numerical output. Kept identical
# to embed.py anyway to remove any doubt about Q1 vs Q2-Q5 comparability.

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers not installed.")
    print("Run: pip install sentence-transformers")
    sys.exit(1)

# Import queries from multi_query_eval
sys.path.insert(0, os.path.dirname(__file__))
from multi_query_eval import QUERIES

def build_embedding_text(prof):
    parts = []
    if prof.get("research_text"): parts.append(prof["research_text"])
    if prof.get("biography"):     parts.append(prof["biography"][:500])
    return " | ".join(parts)

def main():
    print(f"[MODEL] Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print("[MODEL] Ready")

    with open(PROFESSORS_FILE, "r", encoding="utf-8") as f:
        professors = json.load(f)
    prof_texts = [build_embedding_text(p) for p in professors]
    prof_names = [p["name"] for p in professors]
    print(f"[DATA] {len(professors)} professors")

    print(f"[EMBED] Encoding professor profiles...")
    prof_embeddings = model.encode(
        prof_texts, normalize_embeddings=True,
        batch_size=32, show_progress_bar=True
    )

    query_scores = {}
    # Skip Q1 — already in comparison_results.json
    for qid, qinfo in QUERIES.items():
        if qid == "Q1_web_ir_nlp":
            continue
        print(f"[EMBED] Query {qid}...")
        q_emb = model.encode(qinfo["text"], normalize_embeddings=True)
        sims  = (prof_embeddings @ q_emb).tolist()
        query_scores[qid] = {name: round(float(s), 6)
                              for name, s in zip(prof_names, sims)}

    out = {"model": MODEL_NAME, "query_scores": query_scores}
    os.makedirs("data/final", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[DONE] Saved → {OUTPUT_FILE}")
    print("Now run: python analysis/multi_query_eval.py")

if __name__ == "__main__":
    main()
