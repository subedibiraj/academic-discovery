import json
import re
import os

# Input files (auto-discover all *_professors.json in data/processed/)
import glob
INPUT_FILES = sorted(glob.glob("data/processed/*_professors.json"))

OUTPUT_FILE = "data/final/all_professors.json"

os.makedirs("data/final", exist_ok=True)

def clean_text(text: str) -> str:
    """Remove extra whitespace and newlines."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def clean_list(items: list) -> list:
    """Clean each item in a list, remove duplicates and empty strings."""
    cleaned = []
    seen = set()
    for item in items:
        item = clean_text(str(item))
        if item and item not in seen:
            cleaned.append(item)
            seen.add(item)
    return cleaned

def normalize(prof: dict) -> dict:
    """
    Normalize a professor record into a single consistent schema.
    """

    # Name
    name = clean_text(prof.get("name", ""))

    # University & Department
    university = clean_text(prof.get("university", ""))
    department = clean_text(prof.get("department", ""))

    # Title — handle both "title" (str) and "titles" (list)
    title_raw = prof.get("title") or prof.get("titles") or ""
    if isinstance(title_raw, list):
        title = clean_text(title_raw[0]) if title_raw else ""
    else:
        title = clean_text(str(title_raw))

    # Contact
    email  = clean_text(prof.get("email", ""))
    phone  = clean_text(prof.get("phone", ""))
    office = clean_text(prof.get("office", ""))

    # Profile URL — handle both field names
    profile_url = clean_text(
        prof.get("profile_url") or prof.get("source_url") or ""
    )

    # Research Interests
    raw_interests = prof.get("research_interests", [])

    if isinstance(raw_interests, list):
        # Fix TAMU bug: first item often concatenates all others
        # Detect: if an item contains another item as substring, skip it
        filtered = []
        for item in raw_interests:
            item_clean = clean_text(str(item))
            if not item_clean:
                continue
            # Skip if this item is clearly a concatenation of multiple items
            # (contains another item in the list as a substring)
            is_concat = any(
                item_clean != clean_text(str(other)) and
                clean_text(str(other)) in item_clean
                for other in raw_interests
                if other != item
            )
            if not is_concat:
                filtered.append(item_clean)
        research_interests = clean_list(filtered)
    else:
        research_interests = []

    # Build research_text (used for embeddings)
    research_text = ", ".join(research_interests)

    # Education
    edu_raw = prof.get("education", [])
    if isinstance(edu_raw, list):
        education = clean_list(edu_raw)
    elif isinstance(edu_raw, str):
        # Berkeley stores as semicolon-separated string
        education = [clean_text(e) for e in edu_raw.split(";") if clean_text(e)]
    else:
        education = []

    # Biography
    biography = clean_text(prof.get("biography", ""))

    # Awards
    awards_raw = prof.get("awards", [])
    awards = clean_list(awards_raw) if isinstance(awards_raw, list) else []

    # Publications
    pubs_raw = prof.get("selected_publications", [])
    publications = clean_list(pubs_raw) if isinstance(pubs_raw, list) else []

    # Other fields
    website = clean_text(
        prof.get("personal_website") or prof.get("website") or ""
    )
    scholar = clean_text(prof.get("google_scholar", ""))
    lab     = clean_text(prof.get("lab", ""))

    return {
        "name"              : name,
        "university"        : university,
        "department"        : department,
        "title"             : title,
        "email"             : email,
        "phone"             : phone,
        "office"            : office,
        "profile_url"       : profile_url,
        "personal_website"  : website,
        "google_scholar"    : scholar,
        "lab"               : lab,
        "research_interests": research_interests,
        "research_text"     : research_text,   # ← THIS is what embeddings use
        "education"         : education,
        "biography"         : biography,
        "awards"            : awards,
        "publications"      : publications,
    }

def main():
    all_professors = []
    seen_names = set()

    for filepath in INPUT_FILES:
        if not os.path.exists(filepath):
            print(f"[SKIP] File not found: {filepath}")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"[LOAD] {filepath} → {len(data)} records")

        for prof in data:
            # Skip error records
            if "error" in prof:
                continue

            normalized = normalize(prof)

            # Skip if no name
            if not normalized["name"]:
                continue

            # Skip if no research data at all
            if not normalized["research_text"] and not normalized["biography"]:
                print(f"  [SKIP] No research data: {normalized['name']}")
                continue

            # Deduplicate by name + university
            key = f"{normalized['name']}|{normalized['university']}"
            if key in seen_names:
                continue
            seen_names.add(key)

            all_professors.append(normalized)

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_professors, f, indent=2, ensure_ascii=False)

    # Report
    print(f"\n{'='*50}")
    print(f"[DONE] Total merged     : {len(all_professors)}")
    print(f"       With research    : {len([p for p in all_professors if p['research_text']])}")
    print(f"       With biography   : {len([p for p in all_professors if p['biography']])}")
    print(f"       With email       : {len([p for p in all_professors if p['email']])}")
    print(f"       Saved to         : {OUTPUT_FILE}")
    print(f"{'='*50}")

    # Show universities breakdown
    from collections import Counter
    unis = Counter(p["university"] for p in all_professors)
    print("\n[BREAKDOWN]")
    for uni, count in unis.most_common():
        print(f"  {uni}: {count} professors")

    # Sample
    sample = all_professors[0]
    print(f"\n[SAMPLE] {sample['name']}")
    print(f"  Research: {sample['research_text'][:100]}")

if __name__ == "__main__":
    main()