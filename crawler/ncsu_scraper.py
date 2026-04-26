import json
import requests
from bs4 import BeautifulSoup
import time
import os
import re

# Config
# We scrape Professors and Associate Professors only.
# Assistant professors, lecturers, adjuncts etc. are skipped.
LISTING_PAGES = [
    ("https://csc.ncsu.edu/group/faculty/professors/",  "Professor"),
    ("https://csc.ncsu.edu/group/faculty/associate/",   "Associate Professor"),
    ("https://csc.ncsu.edu/group/faculty/assistant/",   "Assistant Professor"),
]
BASE_URL    = "https://csc.ncsu.edu"
OUTPUT_JSON = "data/processed/ncstate_professors.json"
DELAY       = 1.5

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

def parse_listing_page(html: str, source_url: str) -> list:
    """
    Parse one faculty group listing page.

    Structure (from HTML):
      .person-card
        .person-card__info-container
          a.name[href]          → profile URL + display name
          p.title               → title (redundant on detail page but useful)
          a.phone               → phone
          a.email               → email

    Note: the listing page already has phone and email inline,
    so we capture them here and let the detail page override if richer.
    """
    soup = BeautifulSoup(html, "html.parser")
    professors = []
    seen_urls  = set()

    cards = soup.select(".person-card")
    print(f"  [PARSE] {source_url} → {len(cards)} cards")

    for card in cards:
        info = card.select_one(".person-card__info-container")
        if not info:
            continue

        name_a = info.select_one("a.name")
        if not name_a:
            continue

        href = name_a.get("href", "").strip()
        profile_url = href if href.startswith("http") else BASE_URL + href
        if not profile_url or profile_url in seen_urls:
            continue
        seen_urls.add(profile_url)

        name = name_a.get_text(strip=True)
        if not name:
            continue

        title_p = info.select_one("p.title")
        title   = title_p.get_text(strip=True) if title_p else ""

        phone_a = info.select_one("a.phone")
        phone   = phone_a.get_text(strip=True) if phone_a else ""

        email_a = info.select_one("a.email")
        email   = email_a.get("href", "").replace("mailto:", "").strip() if email_a else ""

        professors.append({
            "name"              : name,
            "university"        : "North Carolina State University",
            "department"        : "Computer Science",
            "title"             : title,
            "email"             : email,
            "phone"             : phone,
            "office"            : "",
            "profile_url"       : profile_url,
            "personal_website"  : "",
            "google_scholar"    : "",
            "lab"               : "",
            "research_interests": [],
            "research_text"     : "",
            "education"         : [],
            "biography"         : "",
            "awards"            : [],
            "publications"      : [],
        })

    return professors

def parse_profile_page(html: str) -> dict:
    """
    Parse an individual faculty profile page (WordPress / ncstate-people plugin).

    Structure (from HTML):
      .person-card__info-container
        p.title          → title
        p.office         → office room
        a.phone          → phone
        a.email          → mailto: link
        a.website        → personal website

      .ncst-people-person__biography          → biography (HTML blob)
      .ncst-people-person__research-description → research areas (plain text)

      .ncst-people-person__publications ul li  → recent publications
        a[href]  → citation link
        text     → title + venue

      #ncst-accordion-item-awards-and-honors ul li → awards

    Note: Google Scholar is NOT a structured field here — it sometimes appears
    as a link inside the biography text, so we scan for it there.
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

    # Contact card (on detail page)
    card_info = soup.select_one(".person-card__info-container")
    if card_info:
        title_p = card_info.select_one("p.title")
        if title_p:
            result["title"] = title_p.get_text(strip=True)

        office_p = card_info.select_one("p.office")
        if office_p:
            result["office"] = office_p.get_text(strip=True)

        phone_a = card_info.select_one("a.phone")
        if phone_a:
            result["phone"] = phone_a.get_text(strip=True)

        email_a = card_info.select_one("a.email")
        if email_a:
            result["email"] = email_a.get("href", "").replace("mailto:", "").strip()

        website_a = card_info.select_one("a.website")
        if website_a:
            result["personal_website"] = website_a.get("href", "").strip()

    # Biography
    bio_el = soup.select_one(".ncst-people-person__biography")
    if bio_el:
        bio_text = re.sub(r'\s+', ' ', bio_el.get_text(separator=" ", strip=True))
        result["biography"] = bio_text

        # Scan biography links for Google Scholar
        for a in bio_el.find_all("a", href=True):
            if "scholar.google" in a["href"]:
                result["google_scholar"] = a["href"].strip()
                break

    # Research / Areas of Expertise
    research_el = soup.select_one(".ncst-people-person__research-description")
    if research_el:
        raw = research_el.get_text(separator="\n", strip=True)
        # Split on newlines and <br> (already converted to \n by separator)
        areas = [line.strip() for line in raw.splitlines() if line.strip()]
        result["research_interests"] = areas
        result["research_text"]      = ", ".join(areas)

    # Publications
    pubs_ul = soup.select_one(".ncst-people-person__publications")
    pubs = []
    if pubs_ul:
        for li in pubs_ul.find_all("li"):
            link_a = li.select_one("a")
            title  = ""
            venue  = ""
            if link_a:
                title = link_a.get_text(strip=True)
                # Venue text sits after the </a>, strip leading comma/space
                after = link_a.next_sibling
                if after:
                    venue = str(after).strip().lstrip(",").strip()
            elif li.get_text(strip=True):
                title = li.get_text(strip=True)

            if title:
                citation = title
                if venue:
                    citation = f"{title}, {venue}"
                pubs.append(citation)

    result["publications"] = pubs

    # Awards
    awards_div = soup.select_one("#ncst-accordion-item-awards-and-honors")
    if awards_div:
        result["awards"] = [
            li.get_text(strip=True)
            for li in awards_div.select("ul li")
            if li.get_text(strip=True)
        ]

    return result

def scrape_all_listings() -> list:
    """Fetch all configured listing pages and deduplicate entries."""
    all_professors = []
    seen_urls = set()

    for listing_url, rank_label in LISTING_PAGES:
        print(f"\n[FETCH] {listing_url}")
        try:
            resp = requests.get(listing_url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            resp.encoding = "utf-8"
        except requests.RequestException as e:
            print(f"  [ERROR] {e}")
            continue

        entries = parse_listing_page(resp.text, listing_url)
        for entry in entries:
            if entry["profile_url"] not in seen_urls:
                seen_urls.add(entry["profile_url"])
                all_professors.append(entry)

        time.sleep(DELAY)

    print(f"\n[LIST] Total unique faculty: {len(all_professors)}")
    return all_professors

def scrape_profiles(professors: list) -> list:
    """Visit each professor's profile page and enrich their data."""
    for i, prof in enumerate(professors, 1):
        url = prof.get("profile_url", "")
        if not url:
            print(f"  [SKIP] No profile URL for {prof['name']}")
            continue

        print(f"[{i}/{len(professors)}] {prof['name']}")
        print(f"  → {url}")

        # Use Referer matching the professors listing (NC State checks referrers)
        headers = {**HEADERS,
                   "Referer"        : "https://csc.ncsu.edu/group/faculty/professors/",
                   "Sec-Fetch-Site" : "same-origin"}

        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            resp.encoding = "utf-8"
        except requests.RequestException as e:
            print(f"  [ERROR] {e}")
            time.sleep(DELAY)
            continue

        extras = parse_profile_page(resp.text)

        # Profile page is authoritative — override listing values
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
        if extras["google_scholar"]:
            prof["google_scholar"] = extras["google_scholar"]
        if extras["research_interests"]:
            prof["research_interests"] = extras["research_interests"]
            prof["research_text"]      = extras["research_text"]
        if extras["biography"]:
            prof["biography"] = extras["biography"]
        if extras["awards"]:
            prof["awards"] = extras["awards"]
        if extras["publications"]:
            prof["publications"] = extras["publications"]

        print(f"  Title    : {prof['title'][:70]}")
        print(f"  Email    : {prof['email'] or '[none]'}")
        print(f"  Research : {prof['research_text'][:80]}{'...' if len(prof['research_text']) > 80 else ''}")
        print(f"  Pubs     : {len(prof['publications'])}")
        print(f"  Awards   : {len(prof['awards'])}")

        # Checkpoint every 10 profiles
        if i % 10 == 0:
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(professors, f, indent=2, ensure_ascii=False)
            print(f"  [CHECKPOINT] {i} saved.\n")

        time.sleep(DELAY)

    return professors

def main():
    # 1. Fetch all listing pages (professors + associate professors)
    professors = scrape_all_listings()

    if not professors:
        print("[ABORT] No faculty found — check HTML selectors.")
        return

    # Quick save before profile scraping
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(professors, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {len(professors)} faculty (before profile enrichment)\n")

    # 2. Scrape individual profiles
    print(f"[PROFILES] Scraping {len(professors)} individual pages...\n")
    professors = scrape_profiles(professors)

    # 3. Final save
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(professors, f, indent=2, ensure_ascii=False)

    # 4. Summary
    with_research = [p for p in professors if p.get("research_text")]
    with_bio      = [p for p in professors if p.get("biography")]
    with_email    = [p for p in professors if p.get("email")]
    with_pubs     = [p for p in professors if p.get("publications")]
    with_awards   = [p for p in professors if p.get("awards")]
    with_scholar  = [p for p in professors if p.get("google_scholar")]

    print(f"\n{'='*50}")
    print(f"[DONE] Total scraped      : {len(professors)}")
    print(f"       With research      : {len(with_research)}")
    print(f"       With biography     : {len(with_bio)}")
    print(f"       With email         : {len(with_email)}")
    print(f"       With publications  : {len(with_pubs)}")
    print(f"       With awards        : {len(with_awards)}")
    print(f"       With Google Scholar: {len(with_scholar)}")
    print(f"       Saved to           : {OUTPUT_JSON}")
    print(f"{'='*50}")

    # Sample output
    if professors:
        s = professors[0]
        print(f"\n── Sample ──────────────────────────────────────────")
        print(f"  Name     : {s['name']}")
        print(f"  Title    : {s['title']}")
        print(f"  Email    : {s['email']}")
        print(f"  Phone    : {s['phone']}")
        print(f"  Office   : {s['office']}")
        print(f"  Website  : {s['personal_website']}")
        print(f"  Scholar  : {s['google_scholar']}")
        print(f"  Research : {s['research_text'][:100]}")
        print(f"  Bio      : {s['biography'][:100] + '...' if s['biography'] else '[none]'}")
        print(f"  Pubs     : {len(s['publications'])} entries")
        print(f"  Awards   : {len(s['awards'])} entries")

if __name__ == "__main__":
    main()