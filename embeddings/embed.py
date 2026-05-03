import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import argparse

# Paths
INPUT_PROFILE  = "data/final/all_professors.json"
INPUT_ARXIV    = "data/final/all_professors_arxiv.json"
OUTPUT_FILE    = "data/final/all_professors_embedded.json"

os.makedirs("data/final", exist_ok=True)

# -- YOUR research interest -- edit this --------------------------------------
MY_INTEREST = """
I am interested in large-scale web data collection and information extraction 
systems, web crawling and browser automation for structured data acquisition, 
data engineering pipelines for cleaning and transforming heterogeneous data 
formats (JSON, CSV, Parquet). I have experience applying NLP and machine 
learning to real-world problems including speech recognition with Whisper 
for low-resource languages and computer vision for image colorization. 
I want to work on intelligent systems that automatically discover, extract, 
and organize knowledge from unstructured web and text data, with applications 
in information retrieval, natural language processing, and applied machine 
learning for data science.
"""

def build_embedding_text(prof: dict, use_arxiv: bool = False) -> str:
    """
    Combine all research-relevant fields into one rich text for embedding.
    research_text alone is sometimes too short -- biography adds context.
    When use_arxiv=True, also includes recent paper abstracts.
    """
    parts = []

    if prof.get("research_text"):
        parts.append(prof["research_text"])

    # First 500 chars of biography adds useful context without noise
    if prof.get("biography"):
        parts.append(prof["biography"][:500])

    # arXiv paper abstracts (weighted lower -- first 300 chars each, max 3)
    if use_arxiv and prof.get("arxiv_papers"):
        abstracts = []
        for paper in prof["arxiv_papers"][:3]:
            abstract = paper.get("abstract", "")
            if abstract:
                abstracts.append(abstract[:300])
        if abstracts:
            parts.append("Recent papers: " + " | ".join(abstracts))

    return " | ".join(parts)

def main():
    parser = argparse.ArgumentParser(description="Embed professor profiles")
    parser.add_argument("--arxiv", action="store_true",
                        help="Include arXiv paper abstracts in embedding text")
    args = parser.parse_args()

    use_arxiv = args.arxiv

    # Choose input file
    if use_arxiv and os.path.exists(INPUT_ARXIV):
        input_file = INPUT_ARXIV
        print(f"[MODE] Profile + arXiv (using {INPUT_ARXIV})")
    else:
        input_file = INPUT_PROFILE
        if use_arxiv:
            print(f"[WARN] --arxiv flag set but {INPUT_ARXIV} not found, using profile-only")
        print(f"[MODE] Profile-only (using {INPUT_PROFILE})")

    print("[LOAD] Loading professors...")
    with open(input_file, "r", encoding="utf-8") as f:
        professors = json.load(f)

    print(f"[INFO] {len(professors)} professors loaded")

    if use_arxiv:
        with_papers = sum(1 for p in professors if p.get("arxiv_papers"))
        print(f"[INFO] {with_papers} professors have arXiv papers")

    # Load model -- downloads once, cached locally after
    print("\n[MODEL] Loading sentence-transformers model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("[MODEL] Ready")

    # Embed YOUR interest
    print("\n[EMBED] Embedding your research interest...")
    my_embedding = model.encode(MY_INTEREST, normalize_embeddings=True)
    print(f"[EMBED] Your embedding shape: {my_embedding.shape}")

    # Embed each professor
    print(f"\n[EMBED] Embedding {len(professors)} professors...")
    texts = [build_embedding_text(p, use_arxiv=use_arxiv) for p in professors]

    # Batch encode (much faster than one at a time)
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True
    )

    # Attach embeddings to professors
    for i, prof in enumerate(professors):
        prof["embedding"] = embeddings[i].tolist()
        prof["embedding_text"] = texts[i]  # so you can debug what was embedded

    # Save
    result = {
        "my_interest"   : MY_INTEREST.strip(),
        "my_embedding"  : my_embedding.tolist(),
        "embedding_mode": "profile+arxiv" if use_arxiv else "profile-only",
        "professors"    : professors
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Saved to {OUTPUT_FILE}")
    print(f"[INFO] Mode: {'profile+arxiv' if use_arxiv else 'profile-only'}")
    print(f"[INFO] Each professor now has an 'embedding' field")

if __name__ == "__main__":
    main()