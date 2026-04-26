import json
import requests
from bs4 import BeautifulSoup
import time
import os
import re

# Config
FACULTY_LIST_URL = "https://website.cs.vt.edu/people/faculty.html"
BASE_URL         = "https://website.cs.vt.edu"
OUTPUT_JSON      = "data/processed/vtech_professors.json"
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
    Parse the faculty listing page at /people/faculty.html.

    The complete faculty list is embedded in the subnav sidebar of every
    page on the site. The listing page itself renders:

      ul.vt-subnav-children
        li.vt-subnav-droplist-item
          a[href]   → full profile URL, link text = faculty name

    This is the cleanest source of all faculty URLs — no pagination needed.
    """
    soup = BeautifulSoup(html, "html.parser")
    professors = []
    seen_urls  = set()

    subnav = soup.select_one("ul.vt-subnav-children")
    if not subnav:
        print("  [WARN] Could not find ul.vt-subnav-children — check HTML")
        return professors

    items = subnav.select("li.vt-subnav-droplist-item a")
    print(f"  [PARSE] {len(items)} faculty links found in subnav")

    for a in items:
        href = a.get("href", "").strip()
        if not href or "/people/faculty/" not in href:
            continue

        profile_url = href if href.startswith("http") else BASE_URL + href
        if profile_url in seen_urls:
            continue
        seen_urls.add(profile_url)

        name = a.get_text(strip=True)
        if not name:
            continue

        professors.append({
            "name"              : name,
            "university"        : "Virginia Tech",
            "department"        : "Computer Science",
            "title"             : "",
            "email"             : "",
            "phone"             : "",
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

    print(f"[LIST] Extracted {len(professors)} faculty entries")
    return professors

def parse_profile_page(html: str) -> dict:
    """
    Parse an individual faculty profile page.

    Structure (from HTML):
      .vt-bio-name                    → name (already have it)
      .vt-person-title                → primary title
      .vt-bio-address address         → office address (multi-line)
      .vt-bio-contact-email           → email (mailto: link)
      .vt-bio-phone-link              → phone (tel: link)

      .vt-bodycol-content .vt-text    → free-form body; contains:
        h3 "Research interests"  → next <ul> has research interest bullets
        h3 with Google Scholar / Homepage links → parse hrefs
        h3 "Education"           → next <ul> has education bullets

    VT uses free-form HTML in the body column rather than structured fields,
    so we walk the DOM section by section using h3 headings as delimiters.
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

    # Title
    title_div = soup.select_one(".vt-person-title")
    if title_div:
        result["title"] = title_div.get_text(strip=True)

    # Office address
    addr = soup.select_one(".vt-bio-address address")
    if addr:
        result["office"] = re.sub(r'\s+', ' ', addr.get_text(separator=" ", strip=True))

    # Email
    email_a = soup.select_one("a.vt-bio-contact-email")
    if email_a:
        result["email"] = email_a.get("href", "").replace("mailto:", "").strip()

    # Phone
    phone_a = soup.select_one("a.vt-bio-phone-link")
    if phone_a:
        # href is "tel:(540) 231-6504" — strip the scheme and decode %20
        raw = phone_a.get("href", "").replace("tel:", "").strip()
        result["phone"] = requests.utils.unquote(raw)

    # Body content — walk h3-delimited sections
    body = soup.select_one(".vt-bodycol-content .vt-text")
    if body:
        _parse_body_sections(body, result)

    return result

def _parse_body_sections(body_el, result: dict) -> None:
    """
    Walk the free-form .vt-text body, using <h3> tags as section headers.

    Recognised sections (case-insensitive):
      "research interests"  → next <ul> → research_interests list
      "google scholar" / "homepage" links inside any h3 → parsed directly
      "education"           → next <ul> → education list
      "biography" / "about" → paragraph text → biography

    The Google Scholar / Homepage h3 is special: it contains inline <a> tags
    rather than a following list, so we handle it separately.
    """
    children = list(body_el.children)

    i = 0
    while i < len(children):
        node = children[i]

        if getattr(node, 'name', None) == 'h3':
            heading_text = node.get_text(strip=True).lower()

            # Google Scholar / Homepage links (inline in h3)
            for a in node.find_all("a"):
                href  = a.get("href", "").strip()
                label = a.get_text(strip=True).lower()
                if "scholar.google" in href:
                    result["google_scholar"] = href
                elif "homepage" in label or (
                    href and "scholar" not in href and href.startswith("http")
                    and not any(x in href for x in ["vt.edu", "linkedin", "twitter", "facebook"])
                ):
                    # Only override if not already set and looks like a personal site
                    if not result["personal_website"]:
                        result["personal_website"] = href

            # Research interests
            if "research interest" in heading_text:
                j = i + 1
                while j < len(children):
                    sib = children[j]
                    if getattr(sib, 'name', None) == 'ul':
                        items = [li.get_text(strip=True) for li in sib.find_all("li") if li.get_text(strip=True)]
                        result["research_interests"] = items
                        result["research_text"]      = ", ".join(items)
                        break
                    elif getattr(sib, 'name', None) == 'h3':
                        break
                    j += 1

            # Education
            elif "education" in heading_text:
                j = i + 1
                while j < len(children):
                    sib = children[j]
                    if getattr(sib, 'name', None) == 'ul':
                        items = [li.get_text(strip=True) for li in sib.find_all("li") if li.get_text(strip=True)]
                        result["education"] = items
                        break
                    elif getattr(sib, 'name', None) == 'h3':
                        break
                    j += 1

            # Biography / About
            elif any(kw in heading_text for kw in ("bio", "about", "overview")):
                j = i + 1
                bio_parts = []
                while j < len(children):
                    sib = children[j]
                    if getattr(sib, 'name', None) == 'h3':
                        break
                    if getattr(sib, 'name', None) in ('p', 'div'):
                        text = sib.get_text(separator=" ", strip=True)
                        if text:
                            bio_parts.append(text)
                    j += 1
                if bio_parts:
                    result["biography"] = re.sub(r'\s+', ' ', " ".join(bio_parts))

        i += 1

    # Fallback: if no biography section found, try plain <p> tags
    if not result["biography"]:
        paras = []
        for p in body_el.find_all("p"):
            text = p.get_text(separator=" ", strip=True)
            # Skip very short lines (likely labels) and lines that look like links
            if text and len(text) > 40 and not text.startswith("http"):
                paras.append(text)
        if paras:
            result["biography"] = re.sub(r'\s+', ' ', " ".join(paras))

def scrape_profiles(professors: list) -> list:
    """Visit each professor's profile page and enrich their data."""
    for i, prof in enumerate(professors, 1):
        url = prof.get("profile_url", "")
        if not url:
            print(f"  [SKIP] No profile URL for {prof['name']}")
            continue

        print(f"[{i}/{len(professors)}] {prof['name']}")
        print(f"  → {url}")

        # Use Referer matching the listing page (VT checks this sometimes)
        headers = {**HEADERS, "Referer": FACULTY_LIST_URL, "Sec-Fetch-Site": "same-origin"}

        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            resp.encoding = "utf-8"
        except requests.RequestException as e:
            print(f"  [ERROR] {e}")
            time.sleep(DELAY)
            continue

        extras = parse_profile_page(resp.text)

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
        if extras["education"]:
            prof["education"] = extras["education"]
        if extras["biography"]:
            prof["biography"] = extras["biography"]

        print(f"  Title    : {prof['title']}")
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
    # 1. Fetch faculty listing page
    print(f"[FETCH] Faculty list: {FACULTY_LIST_URL}\n")
    try:
        resp = requests.get(FACULTY_LIST_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        resp.encoding = "utf-8"
    except requests.RequestException as e:
        print(f"[ABORT] Could not fetch faculty list: {e}")
        return

    # 2. Parse listing — all URLs come from the subnav
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
    with_scholar  = [p for p in professors if p.get("google_scholar")]
    with_edu      = [p for p in professors if p.get("education")]

    print(f"\n{'='*50}")
    print(f"[DONE] Total scraped      : {len(professors)}")
    print(f"       With research      : {len(with_research)}")
    print(f"       With biography     : {len(with_bio)}")
    print(f"       With email         : {len(with_email)}")
    print(f"       With Google Scholar: {len(with_scholar)}")
    print(f"       With education     : {len(with_edu)}")
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
        print(f"  Education: {s['education']}")
        print(f"  Bio      : {s['biography'][:100] + '...' if s['biography'] else '[none]'}")

if __name__ == "__main__":
    main()