import json
import requests
from bs4 import BeautifulSoup
import time
import os
import re

# Config
FACULTY_LIST_URL = "https://www.cs.utexas.edu/people"
BASE_URL         = "https://www.cs.utexas.edu"
OUTPUT_JSON      = "data/processed/utaustin_professors.json"
DELAY            = 1.5

HEADERS = {
    "User-Agent"               : "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:150.0) Gecko/20100101 Firefox/150.0",
    "Accept"                   : "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language"          : "en-US,en;q=0.9",
    "Accept-Encoding"          : "gzip, deflate, br, zstd",
    "Connection"               : "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest"           : "document",
    "Sec-Fetch-Mode"           : "navigate",
    "Sec-Fetch-Site"           : "none",
    "Sec-Fetch-User"           : "?1",
    "Pragma"                   : "no-cache",
    "Cache-Control"            : "no-cache",
}

os.makedirs("data/processed", exist_ok=True)

def parse_faculty_list(html: str) -> list:
    """
    Parse the main faculty listing page at /people.

    Structure (from HTML):
      .views-row
        .views-field-field-faculty-photo  → photo (skip)
        .views-field-title > a            → name + profile URL
        .views-field-field-contact-faculty-title → title
        .views-field-field-research-groups → research area links

    There are two view blocks:
      - block_1  → main faculty
      - block_2  → affiliated faculty (separate section)
    Both use the same card structure, so we handle both.
    """
    soup = BeautifulSoup(html, "html.parser")
    professors = []
    seen_urls  = set()

    rows = soup.select(".views-row")
    print(f"  [PARSE] {len(rows)} total rows found across all view blocks")

    for row in rows:
        # Profile link + Name
        title_field = row.select_one(".views-field-title .field-content a")
        if not title_field:
            continue

        href = title_field.get("href", "")
        profile_url = BASE_URL + href if href.startswith("/") else href

        if profile_url in seen_urls:
            continue
        seen_urls.add(profile_url)

        name = title_field.get_text(strip=True)
        if not name:
            continue

        # Title
        title_div = row.select_one(".views-field-field-contact-faculty-title .field-content")
        title = title_div.get_text(strip=True) if title_div else ""

        # Research areas (from listing page links)
        research_div = row.select_one(".views-field-field-research-groups .field-content")
        research_areas = []
        if research_div:
            research_areas = [a.get_text(strip=True) for a in research_div.find_all("a")]

        professors.append({
            "name"              : name,
            "university"        : "University of Texas at Austin",
            "department"        : "Computer Science",
            "title"             : title,
            "email"             : "",
            "phone"             : "",
            "office"            : "",
            "profile_url"       : profile_url,
            "personal_website"  : "",
            "google_scholar"    : "",
            "lab"               : "",
            "research_interests": research_areas,   # refined on profile page
            "research_text"     : ", ".join(research_areas),
            "education"         : [],
            "biography"         : "",
            "awards"            : [],
            "publications"      : [],
        })

    print(f"[LIST] Extracted {len(professors)} faculty entries")
    return professors

def parse_profile_page(html: str) -> dict:
    """
    Parse an individual faculty profile page.

    Structure (from HTML):
      .faculty-hero
        .faculty-name          → name (already have it)
        .faculty-title         → short title
        .faculty-caption       → biography blurb

      .faculty-details
        #toggled-section-research
          .research-areas li a → research area labels
          ul li                → research interest bullets
        #toggled-section-publications
          .pub-*               → structured publication fields
        #toggled-section-awards
          .faculty-awards li   → award strings

      .faculty-contact
        .contact-title         → full title
        .contact-homepage a    → personal website
        .contact-phone         → phone
        .contact-email a       → email
        .contact-office        → office location
        .contact-cv a          → CV link (skip — not in schema)
    """
    soup = BeautifulSoup(html, "html.parser")
    result = {
        "title"             : "",
        "email"             : "",
        "phone"             : "",
        "office"            : "",
        "personal_website"  : "",
        "google_scholar"    : "",
        "research_interests": [],
        "research_text"     : "",
        "biography"         : "",
        "awards"            : [],
        "publications"      : [],
    }

    # Biography blurb (faculty-caption)
    caption = soup.select_one(".faculty-caption")
    if caption:
        result["biography"] = re.sub(r'\s+', ' ', caption.get_text(strip=True))

    # Full title (from contact block — more complete than hero)
    contact_title = soup.select_one(".contact-title")
    if contact_title:
        result["title"] = contact_title.get_text(strip=True)

    # Personal website
    homepage = soup.select_one(".contact-homepage a")
    if homepage:
        result["personal_website"] = homepage.get("href", "").strip()

    # Phone
    phone_div = soup.select_one(".contact-phone")
    if phone_div:
        result["phone"] = phone_div.get_text(strip=True)

    # Email
    email_link = soup.select_one(".contact-email a")
    if email_link:
        result["email"] = email_link.get("href", "").replace("mailto:", "").strip()

    # Office
    office_div = soup.select_one(".contact-office")
    if office_div:
        result["office"] = office_div.get_text(strip=True)

    # Research areas + interests
    # The research section has two parts:
    #   1. .research-areas  → broad area labels (links)
    #   2. The <ul> that follows → specific research interest bullets
    research_section = soup.select_one("#toggled-section-research")
    interests = []
    if research_section:
        # Specific interest bullets (the second <ul>, after .research-areas)
        all_uls = research_section.find_all("ul")
        for ul in all_uls:
            if "research-areas" in (ul.get("class") or []):
                continue  # skip the broad-area list
            for li in ul.find_all("li"):
                text = li.get_text(strip=True)
                if text:
                    interests.append(text)

        # Fall back to research area names if no specific bullets
        if not interests:
            for a in research_section.select(".research-areas li a"):
                text = a.get_text(strip=True)
                if text:
                    interests.append(text)

    result["research_interests"] = interests
    result["research_text"]      = ", ".join(interests)

    # Awards
    awards_section = soup.select_one("#toggled-section-awards")
    if awards_section:
        result["awards"] = [
            li.get_text(strip=True)
            for li in awards_section.select(".faculty-awards li")
            if li.get_text(strip=True)
        ]

    # Publications
    # Each pub is a <div> containing <span class="pub-*"> elements.
    # We reconstruct a readable citation string from them.
    pubs_section = soup.select_one("#toggled-section-publications")
    pubs = []
    if pubs_section:
        for pub_div in pubs_section.find_all("div", recursive=False):
            parts = {}
            for span in pub_div.find_all("span", class_=True):
                for cls in span.get("class", []):
                    if cls.startswith("pub-"):
                        key = cls[4:]   # strip "pub-" prefix
                        parts[key] = span.get_text(strip=True)

            # Build citation: Author (Date). Title. Publication. Alt.
            citation_parts = []
            if parts.get("author"):
                citation_parts.append(parts["author"])
            if parts.get("date"):
                citation_parts.append(f"({parts['date']})")
            if parts.get("title"):
                citation_parts.append(parts["title"] + ".")
            if parts.get("pub"):
                citation_parts.append(parts["pub"] + ".")
            if parts.get("loc"):
                citation_parts.append(parts["loc"] + ".")
            if parts.get("alt"):
                citation_parts.append(parts["alt"])

            citation = " ".join(citation_parts).strip()
            if citation:
                pubs.append(citation)

    result["publications"] = pubs

    return result

def scrape_profiles(professors: list) -> list:
    """Visit each professor's profile page and enrich their data."""
    for i, prof in enumerate(professors, 1):
        url = prof.get("profile_url", "")
        if not url:
            print(f"  [SKIP] No profile URL for {prof['name']}")
            continue

        print(f"[{i}/{len(professors)}] {prof['name']}")
        print(f"  → {url}")

        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            resp.encoding = "utf-8"
        except requests.RequestException as e:
            print(f"  [ERROR] {e}")
            time.sleep(DELAY)
            continue

        extras = parse_profile_page(resp.text)

        # Profile page is authoritative — override listing-page values
        if extras["title"]:
            prof["title"] = extras["title"]
        if extras["email"]:
            prof["email"] = extras["email"]
        if extras["phone"]:
            prof["phone"] = extras["phone"]
        if extras["office"]:
            prof["office"] = extras["office"]
        if extras["personal_website"]:
            prof["personal_website"] = extras["personal_website"]
        if extras["research_interests"]:
            prof["research_interests"] = extras["research_interests"]
            prof["research_text"]      = extras["research_text"]
        if extras["biography"]:
            prof["biography"] = extras["biography"]
        if extras["awards"]:
            prof["awards"] = extras["awards"]
        if extras["publications"]:
            prof["publications"] = extras["publications"]

        print(f"  Research : {prof['research_text'][:80]}{'...' if len(prof['research_text']) > 80 else ''}")
        print(f"  Bio      : {'Yes' if prof['biography'] else 'No'}")
        print(f"  Email    : {prof['email'] or '[none]'}")
        print(f"  Awards   : {len(prof['awards'])}")
        print(f"  Pubs     : {len(prof['publications'])}")

        # Checkpoint every 10 profiles
        if i % 10 == 0:
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(professors, f, indent=2, ensure_ascii=False)
            print(f"  [CHECKPOINT] {i} saved.\n")

        time.sleep(DELAY)

    return professors

def main():
    # 1. Fetch faculty listing page
    print(f"[FETCH] Faculty list: {FACULTY_LIST_URL}\n")
    try:
        resp = requests.get(FACULTY_LIST_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        resp.encoding = "utf-8"
    except requests.RequestException as e:
        print(f"[ABORT] Could not fetch faculty list: {e}")
        return

    # 2. Parse listing
    professors = parse_faculty_list(resp.text)

    if not professors:
        print("[ABORT] No faculty found — check HTML selectors.")
        return

    # Quick save before profile scraping
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(professors, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {len(professors)} faculty (before profile enrichment)\n")

    # 3. Scrape individual profiles
    print(f"[PROFILES] Scraping {len(professors)} individual pages...\n")
    professors = scrape_profiles(professors)

    # 4. Final save
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(professors, f, indent=2, ensure_ascii=False)

    # 5. Summary
    with_research = [p for p in professors if p.get("research_text")]
    with_bio      = [p for p in professors if p.get("biography")]
    with_email    = [p for p in professors if p.get("email")]
    with_awards   = [p for p in professors if p.get("awards")]
    with_pubs     = [p for p in professors if p.get("publications")]

    print(f"\n{'='*50}")
    print(f"[DONE] Total scraped    : {len(professors)}")
    print(f"       With research    : {len(with_research)}")
    print(f"       With biography   : {len(with_bio)}")
    print(f"       With email       : {len(with_email)}")
    print(f"       With awards      : {len(with_awards)}")
    print(f"       With publications: {len(with_pubs)}")
    print(f"       Saved to         : {OUTPUT_JSON}")
    print(f"{'='*50}")

    # Sample output
    if professors:
        s = professors[0]
        print(f"\n── Sample ──────────────────────────────────────────")
        print(f"  Name     : {s['name']}")
        print(f"  Title    : {s['title']}")
        print(f"  Email    : {s['email']}")
        print(f"  Office   : {s['office']}")
        print(f"  Website  : {s['personal_website']}")
        print(f"  Research : {s['research_text'][:100]}")
        print(f"  Bio      : {s['biography'][:100] + '...' if s['biography'] else '[none]'}")
        print(f"  Awards   : {len(s['awards'])} entries")
        print(f"  Pubs     : {len(s['publications'])} entries")

if __name__ == "__main__":
    main()