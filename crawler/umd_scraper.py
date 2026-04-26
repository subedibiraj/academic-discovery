import json
import requests
from bs4 import BeautifulSoup
import time
import os
import re

# Config
FACULTY_LIST_URL = "https://www.cs.umd.edu/people/faculty"
BASE_URL         = "https://www.cs.umd.edu"
OUTPUT_JSON      = "data/processed/umd_professors.json"
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
    Parse the faculty listing page at /people/faculty.

    Structure:
      #views-bootstrap-grid-1
        .row
          .media.col-xs-12.col-sm-6.col-md-4.col-lg-4   (one per faculty)
            .views-field-nothing-1
              .media-body
                h4.media-heading > a   → name + /people/<slug> href
                (text nodes)           → title(s)
                a[href^="/research-area/"] → research area labels
    """
    soup = BeautifulSoup(html, "html.parser")
    professors = []
    seen_urls  = set()

    grid = soup.find("div", id="views-bootstrap-grid-1")
    if not grid:
        print("  [WARN] Could not find #views-bootstrap-grid-1 — check HTML")
        return professors

    cards = grid.select("div.media.col-xs-12")
    print(f"  [PARSE] {len(cards)} faculty cards found")

    for card in cards:
        # Profile link + Name
        heading = card.select_one("h4.media-heading a")
        if not heading:
            continue

        href = heading.get("href", "")
        profile_url = BASE_URL + href if href.startswith("/") else href

        if profile_url in seen_urls:
            continue
        seen_urls.add(profile_url)

        name = heading.get_text(strip=True)
        if not name:
            continue

        # Research areas from listing card
        body = card.select_one(".media-body")
        research_areas = []
        if body:
            research_areas = [
                a.get_text(strip=True)
                for a in body.find_all("a", href=lambda h: h and "/research-area/" in h)
            ]

        professors.append({
            "name"              : name,
            "university"        : "University of Maryland",
            "department"        : "Computer Science",
            "title"             : "",        # filled from profile page
            "email"             : "",
            "phone"             : "",
            "office"            : "",
            "profile_url"       : profile_url,
            "personal_website"  : "",
            "google_scholar"    : "",
            "lab"               : "",
            "research_interests": research_areas,
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
      .view-display-id-block_1   (main info block)
        .views-field-field-faculty-title h2               → primary title
        .views-field-field-profile-faculty-sec-title h2   → secondary title
        .views-field-field-profile-phone .field-content   → phone
        .views-field-field-profile-location .field-content → office/location
        .views-field-field-profile-website a              → personal website
        .views-field-field-profile-gscholar a             → google scholar
        .views-field-field-faculty-education ul li        → education entries
        .views-field-field-research-areas .field-content a → research areas
        .views-field-field-biography .field-content       → biography

      .view-display-id-block_3 / .view-id-awards  (awards table)
        table tbody tr
          td[0] → year
          td[1] → type
          td[2] → org
          td[3] → name (strong)

    Note: Email uses a contact form (/user/uid/contact), not a mailto link,
    so we leave email blank (not scrapable without JS session).
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
        "education"         : [],
        "biography"         : "",
        "awards"            : [],
    }

    # Main info block (view block_1)
    main_block = soup.select_one(".view-display-id-block_1")
    if not main_block:
        return result

    # Titles
    primary   = main_block.select_one(".views-field-field-faculty-title h2")
    secondary = main_block.select_one(".views-field-field-profile-faculty-sec-title h2")
    titles = []
    if primary   and primary.get_text(strip=True):
        titles.append(primary.get_text(strip=True))
    if secondary and secondary.get_text(strip=True):
        titles.append(secondary.get_text(strip=True))
    result["title"] = ", ".join(titles)

    # Phone
    phone_div = main_block.select_one(".views-field-field-profile-phone .field-content")
    if phone_div:
        result["phone"] = phone_div.get_text(strip=True)

    # Office/Location
    loc_div = main_block.select_one(".views-field-field-profile-location .field-content")
    if loc_div:
        result["office"] = loc_div.get_text(strip=True)

    # Personal Website
    web_div = main_block.select_one(".views-field-field-profile-website .field-content a")
    if web_div:
        result["personal_website"] = web_div.get("href", "").strip()

    # Google Scholar
    # The link text is just the user ID; reconstruct the full URL
    scholar_div = main_block.select_one(".views-field-field-profile-gscholar .field-content a")
    if scholar_div:
        href = scholar_div.get("href", "").strip()
        if href:
            # Already a full URL in the HTML
            result["google_scholar"] = href if href.startswith("http") else f"https://scholar.google.com/citations?user={href}"

    # Education
    edu_div = main_block.select_one(".views-field-field-faculty-education .field-content")
    if edu_div:
        result["education"] = [
            li.get_text(strip=True)
            for li in edu_div.find_all("li")
            if li.get_text(strip=True)
        ]

    # Research Areas
    ra_div = main_block.select_one(".views-field-field-research-areas .field-content")
    if ra_div:
        areas = [
            a.get_text(strip=True)
            for a in ra_div.find_all("a")
            if a.get_text(strip=True)
        ]
        result["research_interests"] = areas
        result["research_text"]      = ", ".join(areas)

    # Biography
    bio_div = main_block.select_one(".views-field-field-biography .field-content")
    if bio_div:
        result["biography"] = re.sub(r'\s+', ' ', bio_div.get_text(separator=" ", strip=True)).strip()

    # Awards (structured table)
    # Format each row as: "YEAR - TYPE - ORG: Name"
    awards_table = soup.select_one(".view-id-awards tbody")
    awards = []
    if awards_table:
        for row in awards_table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            year  = cells[0].get_text(strip=True)
            atype = cells[1].get_text(strip=True)
            org   = cells[2].get_text(strip=True)
            aname_tag = cells[3].find("strong")
            aname = aname_tag.get_text(strip=True) if aname_tag else cells[3].get_text(strip=True)

            parts = [year]
            if atype:
                parts.append(atype)
            if org:
                parts.append(org)
            entry = " - ".join(parts) + f": {aname}" if aname else " - ".join(parts)
            if entry.strip(" -:"):
                awards.append(entry)

    result["awards"] = awards

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
        if extras["phone"]:
            prof["phone"] = extras["phone"]
        if extras["office"]:
            prof["office"] = extras["office"]
        if extras["personal_website"]:
            prof["personal_website"] = extras["personal_website"]
        if extras["google_scholar"]:
            prof["google_scholar"] = extras["google_scholar"]
        if extras["education"]:
            prof["education"] = extras["education"]
        if extras["research_interests"]:
            prof["research_interests"] = extras["research_interests"]
            prof["research_text"]      = extras["research_text"]
        if extras["biography"]:
            prof["biography"] = extras["biography"]
        if extras["awards"]:
            prof["awards"] = extras["awards"]

        print(f"  Title    : {prof['title'][:70]}")
        print(f"  Research : {prof['research_text'][:80]}{'...' if len(prof['research_text']) > 80 else ''}")
        print(f"  Bio      : {'Yes' if prof['biography'] else 'No'}")
        print(f"  Awards   : {len(prof['awards'])}")
        print(f"  Scholar  : {prof['google_scholar'] or '[none]'}")

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
    with_scholar  = [p for p in professors if p.get("google_scholar")]
    with_awards   = [p for p in professors if p.get("awards")]
    with_edu      = [p for p in professors if p.get("education")]

    print(f"\n{'='*50}")
    print(f"[DONE] Total scraped      : {len(professors)}")
    print(f"       With research      : {len(with_research)}")
    print(f"       With biography     : {len(with_bio)}")
    print(f"       With Google Scholar: {len(with_scholar)}")
    print(f"       With awards        : {len(with_awards)}")
    print(f"       With education     : {len(with_edu)}")
    print(f"       Saved to           : {OUTPUT_JSON}")
    print(f"{'='*50}")

    # Sample output
    if professors:
        s = professors[0]
        print(f"\n── Sample ──────────────────────────────────────────")
        print(f"  Name     : {s['name']}")
        print(f"  Title    : {s['title']}")
        print(f"  Phone    : {s['phone']}")
        print(f"  Office   : {s['office']}")
        print(f"  Website  : {s['personal_website']}")
        print(f"  Scholar  : {s['google_scholar']}")
        print(f"  Research : {s['research_text'][:100]}")
        print(f"  Education: {s['education']}")
        print(f"  Bio      : {s['biography'][:100] + '...' if s['biography'] else '[none]'}")
        print(f"  Awards   : {len(s['awards'])} entries")

if __name__ == "__main__":
    main()