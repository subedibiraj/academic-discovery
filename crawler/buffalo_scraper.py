import json
import requests
from bs4 import BeautifulSoup
import time
import os
import re

# Config
FACULTY_LIST_URL = "https://engineering.buffalo.edu/computer-science-engineering/people/faculty-directory.html"
BASE_URL         = "https://engineering.buffalo.edu"
OUTPUT_JSON      = "data/processed/ubuffalo_professors.json"
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

def _clean_phone(raw: str) -> str:
    """Strip 'Phone: ' prefix and any trailing junk from phone strings."""
    return re.sub(r'^Phone:\s*', '', raw, flags=re.IGNORECASE).strip()

def parse_faculty_list(html: str) -> list:
    """
    Parse the faculty listing page.

    UB renders ALL core faculty inline in the initial HTML inside:
      .ubcms-init-rendered-tab-container
        ul.list-style-teaser
          li > .profileinfo-teaser

    Each card contains:
      .profileinfo-teaser-name  a.title[href]         → detail URL + name
      .profileinfo-teaser-degree                       → degree/education hint
      .profileinfo-teaser-dept-title (first one)       → title
      .profileinfo-teaser-interests span + text        → research topics
      .profileinfo-teaser-contact  p[0]                → office
                                   p[1] (Phone:...)    → phone
                                   a.longtext          → email

    The detail URL pattern is:
      /computer-science-engineering/people/faculty-directory/full-time.host.html/
      content/shared/engineering/.../profiles/faculty/.../NAME.detail.html
    """
    soup = BeautifulSoup(html, "html.parser")
    professors = []
    seen_urls  = set()

    # The active tab content is in the first .ubcms-init-rendered-tab-container
    tab_container = soup.select_one(".ubcms-init-rendered-tab-container")
    if not tab_container:
        print("  [WARN] Could not find .ubcms-init-rendered-tab-container")
        return professors

    cards = tab_container.select(".profileinfo-teaser")
    print(f"  [PARSE] {len(cards)} faculty cards found")

    for card in cards:
        # Name + Profile URL
        name_a = card.select_one(".profileinfo-teaser-name a.title")
        if not name_a:
            continue

        href = name_a.get("href", "").strip()
        profile_url = BASE_URL + href if href.startswith("/") else href
        if not profile_url or profile_url in seen_urls:
            continue
        seen_urls.add(profile_url)

        # Strip the trailing comma/span from the name
        name = name_a.get_text(strip=True).rstrip(",").strip()
        if not name:
            continue

        # Title (first .profileinfo-teaser-dept-title)
        title_div = card.select_one(".profileinfo-teaser-dept-title")
        title = title_div.get_text(strip=True) if title_div else ""

        # Research topics
        interests_p = card.select_one(".profileinfo-teaser-interests p")
        research_text = ""
        research_interests = []
        if interests_p:
            # Remove the label span ("Research Topics:" / "Research Interests:")
            label = interests_p.select_one(".profileinfo-teaser-interests-title")
            if label:
                label.extract()
            raw = interests_p.get_text(strip=True).lstrip(":").strip()
            # Split on semicolons, commas, or newlines
            research_interests = [t.strip() for t in re.split(r'[;,\n]', raw) if t.strip()]
            research_text = ", ".join(research_interests)

        # Contact block (hide-in-narrow version is most complete)
        contact_div = card.select_one(".profileinfo-teaser-contact.hide-in-narrow")
        if not contact_div:
            contact_div = card.select_one(".profileinfo-teaser-contact")

        office = ""
        phone  = ""
        email  = ""

        if contact_div:
            paras = contact_div.find_all("p")
            for p in paras:
                txt = p.get_text(strip=True)
                if "Phone:" in txt or re.match(r'\(?\d{3}\)?[\s\-]\d{3}', txt):
                    phone = _clean_phone(txt)
                elif "@" in txt:
                    email_a = p.select_one("a")
                    if email_a:
                        email = email_a.get("href", "").replace("mailto:", "").strip()
                elif txt and not office:
                    office = txt   # first non-phone, non-email paragraph

        professors.append({
            "name"              : name,
            "university"        : "University at Buffalo",
            "department"        : "Computer Science and Engineering",
            "title"             : title,
            "email"             : email,
            "phone"             : phone,
            "office"            : office,
            "profile_url"       : profile_url,
            "personal_website"  : "",
            "google_scholar"    : "",
            "lab"               : "",
            "research_interests": research_interests,
            "research_text"     : research_text,
            "education"         : [],
            "biography"         : "",
            "awards"            : [],
            "publications"      : [],
        })

    print(f"[LIST] Extracted {len(professors)} faculty entries")
    return professors

def parse_profile_page(html: str) -> dict:
    """
    Parse an individual faculty profile (detail) page.

    Structure (from HTML):
      .profileinfo-dept-title                   → title (already have it)
      .profileinfo-interest h3 + p              → research topics (already have)
      .profileinfo-address-inner p              → office + city + email
      .profileinfo-links .calltoaction a        → may include CV and Google Scholar

      Inside the Biography tab (initial active tab):
        .ubcms-init-rendered-tab-container
          h3 "Education"  → next ul li  → education entries

    The research tab is loaded async and NOT in the initial HTML,
    so we skip it (we already have research from the listing page).
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
    }

    # Title
    title_div = soup.select_one(".profileinfo-dept-title")
    if title_div:
        result["title"] = title_div.get_text(strip=True)

    # Contact info
    addr_inner = soup.select_one(".profileinfo-address-inner")
    if addr_inner:
        paras = addr_inner.find_all("p")
        office_parts = []
        for p in paras:
            email_a = p.select_one("a[href^='mailto:']")
            if email_a:
                result["email"] = email_a.get("href", "").replace("mailto:", "").strip()
            else:
                txt = p.get_text(strip=True)
                if txt and not re.search(r'@|Phone:', txt):
                    office_parts.append(txt)
        result["office"] = ", ".join(office_parts)

    # Related Links — CV / Google Scholar
    for a in soup.select(".profileinfo-links .calltoaction a"):
        href  = a.get("href", "").strip()
        label = a.get_text(strip=True).lower()
        if "scholar.google" in href:
            result["google_scholar"] = href
        elif "scholar" in label or "google scholar" in label:
            result["google_scholar"] = href
        elif href.startswith("http") and not result["personal_website"]:
            # Anything else that's external and not Scholar treat as homepage
            if "buffalo.edu" not in href and "dam/engineering" not in href:
                result["personal_website"] = href

    # Education (Biography tab — present in initial HTML)
    tab_container = soup.select_one(".ubcms-init-rendered-tab-container")
    if tab_container:
        edu_section = None
        for h3 in tab_container.find_all("h3"):
            if "education" in h3.get_text(strip=True).lower():
                edu_section = h3
                break

        if edu_section:
            # The education list is the next <ul> sibling
            sib = edu_section.find_next_sibling()
            while sib:
                if sib.name == "ul":
                    result["education"] = [
                        li.get_text(strip=True)
                        for li in sib.find_all("li")
                        if li.get_text(strip=True)
                    ]
                    break
                elif sib.name in ("h3", "h2"):
                    break
                sib = sib.find_next_sibling()

    # Research topics (from .profileinfo-interest on detail page)
    interest_div = soup.select_one(".profileinfo-interest p")
    if interest_div:
        raw = interest_div.get_text(strip=True).lstrip(":").strip()
        areas = [t.strip() for t in re.split(r'[;,\n]', raw) if t.strip()]
        if areas:
            result["research_interests"] = areas
            result["research_text"]      = ", ".join(areas)

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

        headers = {**HEADERS,
                   "Referer"       : FACULTY_LIST_URL,
                   "Alt-Used"      : "engineering.buffalo.edu",
                   "Sec-Fetch-Site": "same-origin"}

        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            resp.encoding = "utf-8"
        except requests.RequestException as e:
            print(f"  [ERROR] {e}")
            time.sleep(DELAY)
            continue

        extras = parse_profile_page(resp.text)

        # Profile page is authoritative
        if extras["title"]:
            prof["title"] = extras["title"]
        if extras["email"]:
            prof["email"] = extras["email"]
        if extras["office"]:
            prof["office"] = extras["office"]
        if extras["personal_website"]:
            prof["personal_website"] = extras["personal_website"]
        if extras["google_scholar"]:
            prof["google_scholar"] = extras["google_scholar"]
        if extras["research_interests"]:
            prof["research_interests"] = extras["research_interests"]
            prof["research_text"]      = extras["research_text"]
        if extras["education"]:
            prof["education"] = extras["education"]

        print(f"  Title    : {prof['title'][:70]}")
        print(f"  Email    : {prof['email'] or '[none]'}")
        print(f"  Research : {prof['research_text'][:80]}{'...' if len(prof['research_text']) > 80 else ''}")
        print(f"  Education: {len(prof['education'])} entries")
        print(f"  Scholar  : {prof['google_scholar'] or '[none]'}")

        # Checkpoint every 10 profiles
        if i % 10 == 0:
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(professors, f, indent=2, ensure_ascii=False)
            print(f"  [CHECKPOINT] {i} saved.\n")

        time.sleep(DELAY)

    return professors

def main():
    # 1. Fetch listing page
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
    with_edu      = [p for p in professors if p.get("education")]
    with_email    = [p for p in professors if p.get("email")]
    with_scholar  = [p for p in professors if p.get("google_scholar")]

    print(f"\n{'='*50}")
    print(f"[DONE] Total scraped      : {len(professors)}")
    print(f"       With research      : {len(with_research)}")
    print(f"       With email         : {len(with_email)}")
    print(f"       With education     : {len(with_edu)}")
    print(f"       With Google Scholar: {len(with_scholar)}")
    print(f"       Saved to           : {OUTPUT_JSON}")
    print(f"{'='*50}")

    if professors:
        s = professors[0]
        print(f"\n── Sample ──────────────────────────────────────────")
        print(f"  Name     : {s['name']}")
        print(f"  Title    : {s['title']}")
        print(f"  Email    : {s['email']}")
        print(f"  Phone    : {s['phone']}")
        print(f"  Office   : {s['office']}")
        print(f"  Scholar  : {s['google_scholar']}")
        print(f"  Research : {s['research_text'][:100]}")
        print(f"  Education: {s['education']}")

if __name__ == "__main__":
    main()