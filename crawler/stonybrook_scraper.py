import json
import requests
from bs4 import BeautifulSoup
import time
import os
import re

# Config
FACULTY_LIST_URL = "https://www.cs.stonybrook.edu/people/faculty"
BASE_URL         = "https://www.cs.stonybrook.edu"
OUTPUT_JSON      = "data/processed/stonybrook_professors.json"
DELAY            = 1.5

# Only scrape these tabs — skip Affiliated, Emeritus, In Memoriam
TARGET_TABS = ["nav-core"]  # add "nav-research" if you want research faculty too

HEADERS = {
    "User-Agent"     : "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:150.0) Gecko/20100101 Firefox/150.0",
    "Accept"         : "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection"     : "keep-alive",
}

os.makedirs("data/processed", exist_ok=True)

def decode_spamspan(element) -> str:
    """
    Decode spamspan email format:
    <span class="u">amiri</span> [at] <span class="d">cs.stonybrook.edu</span>
    """
    spamspan = element.find("span", class_="spamspan")
    if not spamspan:
        return ""
    u = spamspan.find("span", class_="u")
    d = spamspan.find("span", class_="d")
    if u and d:
        domain = re.sub(r'\s*\[dot\]\s*', '.', d.get_text())
        domain = re.sub(r'\s+', '', domain)
        return f"{u.get_text(strip=True)}@{domain}"
    return ""

def parse_faculty_list(html: str) -> list:
    """
    Parse the faculty listing page.
    Only extracts Core Faculty (nav-core tab).
    Returns list of dicts with name, title, interests snippet, profile_url.
    """
    soup = BeautifulSoup(html, "html.parser")
    professors = []
    seen_urls = set()

    for tab_id in TARGET_TABS:
        tab = soup.find("div", id=tab_id)
        if not tab:
            print(f"  [WARN] Tab '{tab_id}' not found on page")
            continue

        cards = tab.find_all("div", class_=lambda c: c and "card" in c and "h-100" in c)
        print(f"  [PARSE] Tab '{tab_id}': {len(cards)} cards found")

        for card in cards:
            # Profile link
            name_p = card.find("p", class_="faculty-name-field")
            if not name_p:
                continue
            link_tag = name_p.find("a")
            if not link_tag or not link_tag.get("href"):
                continue

            href = link_tag["href"]
            profile_url = BASE_URL + href if href.startswith("/") else href

            if profile_url in seen_urls:
                continue
            seen_urls.add(profile_url)

            # Name
            name = link_tag.get_text(strip=True)
            if not name:
                continue

            # Title
            title_p = card.find("p", class_="faculty-jobtitle-field")
            title = title_p.get_text(strip=True) if title_p else ""

            # Research interests snippet (often truncated on list page)
            interests_p = card.find("p", class_="faculty-interests-field")
            interests_snippet = ""
            if interests_p:
                # Remove the "more" link text
                more_link = interests_p.find("a", class_="views-more-link")
                if more_link:
                    more_link.decompose()
                interests_snippet = interests_p.get_text(strip=True).rstrip("…").strip()

            professors.append({
                "name"              : name,
                "university"        : "Stony Brook University",
                "department"        : "Computer Science",
                "title"             : title,
                "email"             : "",        # filled from profile page
                "phone"             : "",
                "office"            : "",
                "personal_website"  : "",
                "research_interests": [interests_snippet] if interests_snippet else [],
                "research_text"     : interests_snippet,
                "education"         : [],
                "biography"         : "",
                "research_summary"  : "",        # full research paragraph
                "awards"            : [],
                "teaching"          : "",
                "profile_url"       : profile_url,
            })

    print(f"[LIST] Extracted {len(professors)} core faculty")
    return professors

def parse_profile_page(html: str) -> dict:
    """
    Parse individual professor profile page.
    Drupal field classes are very clean and consistent.
    """
    soup = BeautifulSoup(html, "html.parser")
    result = {
        "email"             : "",
        "office"            : "",
        "personal_website"  : "",
        "research_interests": [],
        "research_text"     : "",
        "biography"         : "",
        "research_summary"  : "",
        "awards"            : "",
        "teaching"          : "",
    }

    # Email (spamspan format)
    email_div = soup.find("div", class_=lambda c: c and "field--name-field-protected-email" in c)
    if email_div:
        result["email"] = decode_spamspan(email_div)

    # Office/Address
    addr_div = soup.find("div", class_=lambda c: c and "field--name-field-facultyaddress" in c)
    if addr_div:
        result["office"] = addr_div.get_text(separator=" ", strip=True)
        result["office"] = re.sub(r'\s+', ' ', result["office"]).strip()

    # Personal Website
    web_div = soup.find("div", class_=lambda c: c and "field--name-field-facultywebsite" in c)
    if web_div:
        web_link = web_div.find("a")
        if web_link:
            result["personal_website"] = web_link.get("href", "").strip()

    # Research Interests (clean, comma-separated on profile)
    interests_div = soup.find("div", class_=lambda c: c and "field--name-field-interests" in c)
    if interests_div:
        field_item = interests_div.find("div", class_="field__item")
        if field_item:
            raw = field_item.get_text(strip=True)
            # Split by comma and clean each item
            interests = [i.strip() for i in raw.split(",") if i.strip()]
            result["research_interests"] = interests
            result["research_text"] = ", ".join(interests)

    # Biography
    bio_div = soup.find("div", class_=lambda c: c and "field--name-field-biography" in c)
    if bio_div:
        field_item = bio_div.find("div", class_="field__item")
        if field_item:
            result["biography"] = field_item.get_text(separator=" ", strip=True)
            result["biography"] = re.sub(r'\s+', ' ', result["biography"]).strip()

    # Research Summary (longer research description)
    research_div = soup.find("div", class_=lambda c: c and "field--name-field-research" in c)
    if research_div:
        field_item = research_div.find("div", class_="field__item")
        if field_item:
            result["research_summary"] = field_item.get_text(separator=" ", strip=True)
            result["research_summary"] = re.sub(r'\s+', ' ', result["research_summary"]).strip()

    # Awards
    awards_div = soup.find("div", class_=lambda c: c and "field--name-field-awards" in c)
    if awards_div:
        field_item = awards_div.find("div", class_="field__item")
        if field_item:
            result["awards"] = field_item.get_text(strip=True)

    # Teaching Summary
    teaching_div = soup.find("div", class_=lambda c: c and "field--name-field-teachingsummary" in c)
    if teaching_div:
        field_item = teaching_div.find("div", class_="field__item")
        if field_item:
            result["teaching"] = field_item.get_text(strip=True)

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

        # Merge — profile page is authoritative for all fields
        if extras["email"]:
            prof["email"] = extras["email"]
        if extras["office"]:
            prof["office"] = extras["office"]
        if extras["personal_website"]:
            prof["personal_website"] = extras["personal_website"]
        if extras["research_interests"]:
            prof["research_interests"] = extras["research_interests"]
            prof["research_text"] = extras["research_text"]
        if extras["biography"]:
            prof["biography"] = extras["biography"]
        if extras["research_summary"]:
            prof["research_summary"] = extras["research_summary"]
        if extras["awards"]:
            prof["awards"] = extras["awards"]
        if extras["teaching"]:
            prof["teaching"] = extras["teaching"]

        print(f"  Research : {prof['research_text'][:80]}{'...' if len(prof['research_text']) > 80 else ''}")
        print(f"  Bio      : {'Yes' if prof['biography'] else 'No'}")
        print(f"  Email    : {prof['email'] or '[none]'}")

        # Checkpoint every 10
        if i % 10 == 0:
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(professors, f, indent=2, ensure_ascii=False)
            print(f"  [CHECKPOINT] {i} saved.\n")

        time.sleep(DELAY)

    return professors

def main():
    # 1. Fetch faculty list (single page — no pagination on Stony Brook)
    print(f"[FETCH] Faculty list: {FACULTY_LIST_URL}\n")
    try:
        resp = requests.get(FACULTY_LIST_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        resp.encoding = "utf-8"
    except requests.RequestException as e:
        print(f"[ABORT] Could not fetch faculty list: {e}")
        return

    # 2. Parse listing page
    professors = parse_faculty_list(resp.text)

    if not professors:
        print("[ABORT] No faculty found.")
        return

    # Quick save before profile scraping
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(professors, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {len(professors)} faculty (before profile enrichment)")

    # 3. Scrape individual profiles
    print(f"\n[PROFILES] Scraping {len(professors)} individual pages...\n")
    professors = scrape_profiles(professors)

    # 4. Final save
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(professors, f, indent=2, ensure_ascii=False)

    # 5. Summary
    with_research = [p for p in professors if p.get("research_text")]
    with_bio      = [p for p in professors if p.get("biography")]
    with_email    = [p for p in professors if p.get("email")]

    print(f"\n{'='*50}")
    print(f"[DONE] Total scraped    : {len(professors)}")
    print(f"       With research    : {len(with_research)}")
    print(f"       With biography   : {len(with_bio)}")
    print(f"       With email       : {len(with_email)}")
    print(f"       Saved to         : {OUTPUT_JSON}")
    print(f"{'='*50}")

    # Sample
    if professors:
        s = professors[0]
        print(f"\n── Sample ──────────────────────────────────────────")
        print(f"  Name     : {s['name']}")
        print(f"  Title    : {s['title']}")
        print(f"  Email    : {s['email']}")
        print(f"  Research : {s['research_text'][:100]}")
        print(f"  Bio      : {s['biography'][:100] + '...' if s['biography'] else '[none]'}")

if __name__ == "__main__":
    main()