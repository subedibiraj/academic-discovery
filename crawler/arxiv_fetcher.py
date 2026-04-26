# Fetches recent papers from arXiv API for each professor.
# Cached locally to avoid re-fetching. 5s rate limit between requests.
import json
import os
import re
import time
import random
import arxiv

# Paths
INPUT_FILE  = "data/final/all_professors.json"
OUTPUT_FILE = "data/final/all_professors_arxiv.json"
CACHE_FILE  = "data/final/arxiv_cache.json"

MAX_PAPERS    = 5       # papers per professor
DELAY         = 8.0     # seconds between arXiv requests (API rate limit)
MIN_YEAR      = 2020    # only papers from this year onward
MAX_RETRIES   = 3       # retries on HTTP 429

# CS arXiv categories -- used to filter out non-CS papers
CS_CATEGORIES = {
    "cs.AI", "cs.CL", "cs.CV", "cs.DB", "cs.DC", "cs.DS", "cs.IR",
    "cs.LG", "cs.MA", "cs.NE", "cs.NI", "cs.PL", "cs.RO", "cs.SE",
    "cs.SI", "cs.SY", "cs.CR", "cs.CG", "cs.CC", "cs.CE", "cs.CY",
    "cs.DL", "cs.DM", "cs.ET", "cs.FL", "cs.GL", "cs.GR", "cs.GT",
    "cs.HC", "cs.IT", "cs.LO", "cs.MM", "cs.MS", "cs.NA", "cs.OH",
    "cs.OS", "cs.PF", "cs.SC", "cs.SD",
    "stat.ML", "stat.ME",  # many ML papers land here
    "eess.SP", "eess.AS",  # signal processing, audio/speech
}

os.makedirs("data/final", exist_ok=True)

def _normalize_name(name: str) -> str:
    """
    Normalize a professor name for arXiv author search.
    arXiv author search works best with 'LastName, FirstName' or 'LastName FirstInitial'.
    """
    # Remove quotes, parenthetical nicknames: 'Anxiao "Andrew" Jiang' -> 'Anxiao Jiang'
    name = re.sub(r'"[^"]*"', '', name).strip()
    name = re.sub(r'\([^)]*\)', '', name).strip()
    # Remove middle initials and extra whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def _build_query(name: str) -> str:
    """
    Build an arXiv search query for a professor.
    Uses author search + CS category filter.
    """
    clean = _normalize_name(name)
    parts = clean.split()
    if len(parts) < 2:
        return f"au:{clean}"

    last_name = parts[-1]
    first_name = parts[0]

    # Try "LastName, FirstName" format (most reliable for arXiv)
    return f"au:\"{last_name} {first_name}\""

def fetch_papers(name: str, max_results: int = MAX_PAPERS) -> list:
    """
    Fetch recent CS papers for a professor from arXiv.
    Returns list of paper dicts with title, abstract, year, url, categories.
    Retries with exponential backoff on HTTP 429.
    """
    query = _build_query(name)

    for attempt in range(MAX_RETRIES):
        search = arxiv.Search(
            query=query,
            max_results=max_results * 3,  # fetch extra, we'll filter
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        client = arxiv.Client(
            page_size=10,
            delay_seconds=DELAY,
            num_retries=5,
        )

        papers = []
        try:
            for result in client.results(search):
                # Filter: must have at least one CS category
                cats = set(result.categories)
                if not cats & CS_CATEGORIES:
                    continue

                # Filter: recent papers only
                year = result.published.year
                if year < MIN_YEAR:
                    continue

                papers.append({
                    "title"      : result.title,
                    "abstract"   : result.summary.strip(),
                    "year"       : year,
                    "url"        : result.entry_id,
                    "categories" : result.categories,
                    "authors"    : [a.name for a in result.authors[:5]],
                })

                if len(papers) >= max_results:
                    break

            return papers  # success

        except Exception as e:
            if "429" in str(e) and attempt < MAX_RETRIES - 1:
                wait = DELAY * (3 ** (attempt + 1))  # exponential backoff
                print(f"    [RETRY] Rate limited, waiting {wait:.0f}s (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                print(f"    [ERROR] arXiv query failed: {e}")
                return []

    return []

def load_cache() -> dict:
    """Load cached arXiv results."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache: dict) -> None:
    """Save arXiv results cache."""
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def main():
    # Load professors
    print("[LOAD] Loading professor data...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        professors = json.load(f)
    print(f"[INFO] {len(professors)} professors loaded\n")

    # Load cache
    cache = load_cache()
    cached_count = len(cache)
    print(f"[CACHE] {cached_count} professors already cached\n")

    # Clean cache: remove empty entries from previous rate-limited runs
    cleaned = 0
    keys_to_remove = [k for k, v in cache.items() if v == [] or v is None]
    for k in keys_to_remove:
        del cache[k]
        cleaned += 1
    if cleaned:
        print(f"[CACHE] Cleaned {cleaned} empty entries from previous failed runs")
        save_cache(cache)

    # Fetch papers
    total = len(professors)
    hits = 0
    errors = 0
    skipped = 0

    for i, prof in enumerate(professors, 1):
        name = prof["name"]
        uni  = prof["university"]

        # Check cache first
        cache_key = f"{name}|{uni}"
        if cache_key in cache:
            prof["arxiv_papers"] = cache[cache_key]
            if cache[cache_key]:
                hits += 1
            skipped += 1
            continue

        print(f"[{i}/{total}] {name} ({uni})")

        try:
            papers = fetch_papers(name)
            prof["arxiv_papers"] = papers
            # Only cache non-empty results or genuinely empty (not errors)
            cache[cache_key] = papers

            if papers:
                hits += 1
                print(f"    Found {len(papers)} papers")
                print(f"    Latest: {papers[0]['title'][:70]}... ({papers[0]['year']})")
            else:
                print(f"    No CS papers found")

        except Exception as e:
            print(f"    [ERROR] {e}")
            prof["arxiv_papers"] = []
            errors += 1

        # Checkpoint cache every 20 professors
        if i % 20 == 0:
            save_cache(cache)
            print(f"    [CHECKPOINT] Cache saved ({len(cache)} entries)\n")

        # Extra delay between requests
        time.sleep(DELAY + random.uniform(0.5, 2.5))

    # Final cache save
    save_cache(cache)

    # Save enriched data
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(professors, f, indent=2, ensure_ascii=False)

    # Summary
    with_papers = sum(1 for p in professors if p.get("arxiv_papers"))
    total_papers = sum(len(p.get("arxiv_papers", [])) for p in professors)

    print(f"\n{'='*55}")
    print(f"ARXIV ENRICHMENT SUMMARY")
    print(f"{'='*55}")
    print(f"  Total professors     : {total}")
    print(f"  With arXiv papers    : {with_papers} ({100*with_papers//total}%)")
    print(f"  Total papers fetched : {total_papers}")
    print(f"  Avg papers/professor : {total_papers/max(with_papers,1):.1f}")
    print(f"  Skipped (cached)     : {skipped}")
    print(f"  Errors               : {errors}")
    print(f"  Saved to             : {OUTPUT_FILE}")
    print(f"  Cache                : {CACHE_FILE}")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
