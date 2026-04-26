import json
import requests
from bs4 import BeautifulSoup
import time
import os
import re

# Config
FACULTY_LIST_URL = "https://www2.eecs.berkeley.edu/Faculty/Lists/CS/faculty.html"
BASE_URL         = "https://www2.eecs.berkeley.edu"
OUTPUT_JSON      = "data/processed/berkeley_professors.json"
DELAY            = 1.5

HEADERS_LIST = {
    "User-Agent"     : "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:149.0) Gecko/20100101 Firefox/149.0",
    "Accept"         : "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer"        : "https://www.google.com/",
    "Connection"     : "keep-alive",
}

HEADERS_PROFILE = {
    "User-Agent"     : "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:149.0) Gecko/20100101 Firefox/149.0",
    "Accept"         : "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer"        : FACULTY_LIST_URL,
    "Connection"     : "keep-alive",
}

os.makedirs("data/processed", exist_ok=True)

import sys
sys.stdout.reconfigure(encoding='utf-8')

def parse_faculty_list(html: str) -> list:
    """
    Parse the faculty list page.
    Each faculty card already has: name, title, email, office, 
    research interests, education, and profile link.
    """
    soup = BeautifulSoup(html, "html.parser")
    professors = []

    cards = soup.find_all("div", class_="cc-image-list__item")
    print(f"[PARSE] Found {len(cards)} faculty cards on list page")

    for card in cards:
        # Name + profile link
        name_tag = card.find("h3")
        if not name_tag:
            continue
        name = name_tag.get_text(strip=True)
        link_tag = name_tag.find("a")
        profile_path = link_tag["href"] if link_tag else ""
        profile_url = BASE_URL + profile_path if profile_path else ""

        # Content paragraph
        content_div = card.find("div", class_="cc-image-list__item__content")
        if not content_div:
            continue
        p_tag = content_div.find("p")
        if not p_tag:
            continue

        p_html = str(p_tag)
        p_text = p_tag.get_text(separator="|", strip=True)

        # Title/rank
        title = ""
        strong_tags = p_tag.find_all("strong")
        if strong_tags:
            # First strong tag is always the title
            title = strong_tags[0].get_text(strip=True)

        # Parse the paragraph text by label
        # Structure: Title \n Office/Email \n Research: ... \n Education: ... \n Office Hours: ...
        research_text  = ""
        education_text = ""
        office         = ""
        email          = ""

        # Get all text lines split by <br>
        lines = [line.strip() for line in p_tag.get_text(separator="\n").split("\n") if line.strip()]

        # Research interests — extract from links to /Research/Areas/
        research_links = p_tag.find_all("a", href=lambda h: h and "/Research/Areas/" in h)
        research_interests = [a.get_text(strip=True) for a in research_links]
        research_text = ", ".join(research_interests)

        # Email — look for mailto link
        email_tag = p_tag.find("a", href=lambda h: h and "mailto:" in h)
        if email_tag:
            email = email_tag.get_text(strip=True)

        # Education — line after "Education:" label
        for i, line in enumerate(lines):
            if "Education:" in line:
                # Education text follows on the same line or next
                edu_part = line.replace("Education:", "").strip()
                if edu_part:
                    education_text = edu_part
                elif i + 1 < len(lines):
                    education_text = lines[i + 1]
                break

        # Office — first line that has a hall/room reference (before email)
        # Usually format: "746 Sutardja Dai Hall, (510) 642-7034; email@..."
        for line in lines[1:4]:  # skip title, check next few lines
            if "@" not in line and ("Hall" in line or "Soda" in line or "Cory" in line
                                     or "Sutardja" in line or "Warren" in line
                                     or "Stanley" in line or "Evans" in line):
                # Extract just the office part before the phone number
                office_part = line.split(",")[0].strip()
                if office_part and len(office_part) < 80:
                    office = office_part
                break

        professors.append({
            "name"               : name,
            "university"         : "UC Berkeley",
            "department"         : "EECS — Computer Science",
            "title"              : title,
            "email"              : email,
            "office"             : office,
            "research_interests" : research_interests,
            "research_text"      : research_text,
            "education"          : education_text,
            "profile_url"        : profile_url,
            "biography"          : "",          # filled by profile scrape
            "awards"             : [],          # filled by profile scrape
        })

    return professors

def parse_profile_page(html: str) -> dict:
    """
    Parse individual professor profile page.
    Fixes: email, office, awards extraction from Kadence block structure.
    """
    soup = BeautifulSoup(html, "html.parser", from_encoding="utf-8")
    result = {"biography": "", "awards": [], "email": "", "office": ""}

    # Email
    # Find any mailto link anywhere on the page
    for a in soup.find_all("a", href=True):
        if "mailto:" in a["href"]:
            email = a.get_text(strip=True)
            # Skip assistant/staff emails (usually in "Research Support" section)
            # Keep the first one that looks like a faculty email
            if email and "@" in email:
                result["email"] = email
                break

    # Office
    # On profile pages, office is in a kt-adv-heading paragraph
    # It contains building names like "Hall", "Soda", "Cory", "Sutardja"
    BUILDING_KEYWORDS = ["Hall", "Soda", "Cory", "Sutardja", "Evans",
                         "Stanley", "Warren", "Hearst", "Way West", "South Hall"]
    for p in soup.find_all(["p", "h2", "h3", "h4"]):
        text = p.get_text(strip=True)
        if any(kw in text for kw in BUILDING_KEYWORDS):
            # Make sure it's short (an office address, not a paragraph)
            if len(text) < 100 and "@" not in text and "Research" not in text:
                result["office"] = text
                break

    # Biography
    # Find h4 with "Biography" text, then get the next paragraph sibling
    for heading in soup.find_all(["h4", "h3", "h2"]):
        if "biography" in heading.get_text(strip=True).lower():
            # Walk up to find containing column div, then get next sibling paragraph
            container = heading.find_parent("div", class_=lambda c: c and "kt-inside-inner-col" in c)
            if container:
                bio_p = container.find("p")
                if bio_p:
                    bio_text = bio_p.get_text(strip=True)
                    if len(bio_text) > 100:
                        result["biography"] = bio_text
                        break

    # Awards
    # Structure: button.kt-blocks-accordion-header → parent div.wp-block-kadence-pane
    #            → sibling div.kt-accordion-panel → div.kt-accordion-panel-inner
    #            → ul.kt-svg-icon-list → li → span.kt-svg-icon-list-text
    for btn in soup.find_all("button", class_="kt-blocks-accordion-header"):
        btn_text = btn.get_text(strip=True).lower()
        if "award" in btn_text or "honor" in btn_text:
            # Go up to the pane container
            pane = btn.find_parent("div", class_=lambda c: c and "kt-accordion-pane" in c)
            if pane:
                # Find the panel div inside the pane
                panel = pane.find("div", class_=lambda c: c and "kt-accordion-panel" in c)
                if panel:
                    award_spans = panel.find_all("span", class_="kt-svg-icon-list-text")
                    result["awards"] = [
                        s.get_text(strip=True)
                        for s in award_spans
                        if s.get_text(strip=True)
                    ]
            break

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
            resp = requests.get(url, headers=HEADERS_PROFILE, timeout=15)
            resp.encoding = 'utf-8'
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  [ERROR] {e}")
            time.sleep(DELAY)
            continue

        extras = parse_profile_page(resp.text)
        prof.update(extras)

        print(f"  Research: {prof['research_text'][:80] + '...' if len(prof['research_text']) > 80 else prof['research_text'] or '[NONE]'}")
        print(f"  Bio found: {'Yes' if prof['biography'] else 'No'}")

        # Checkpoint every 10
        if i % 10 == 0:
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(professors, f, indent=2, ensure_ascii=False)
            print(f"  [CHECKPOINT] {i} profiles saved.")

        time.sleep(DELAY)

    return professors

def main():
    # Step 1: Fetch faculty list
    print(f"[FETCH] Faculty list from:\n  {FACULTY_LIST_URL}\n")
    try:
        resp = requests.get(FACULTY_LIST_URL, headers=HEADERS_LIST, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[ABORT] Could not fetch faculty list: {e}")
        return

    # Step 2: Parse list page
    professors = parse_faculty_list(resp.text)
    print(f"\n[LIST] Extracted {len(professors)} professors from list page")

    if not professors:
        print("[ABORT] No professors found. Check HTML structure.")
        return

    # Quick save before profile scraping
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(professors, f, indent=2, ensure_ascii=False)

    # Step 3: Enrich with individual profile pages
    print(f"\n[PROFILES] Now scraping individual pages...\n")
    professors = scrape_profiles(professors)

    # Step 4: Final save
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(professors, f, indent=2, ensure_ascii=False)

    # Step 5: Summary
    with_research = [p for p in professors if p.get("research_text")]
    with_bio      = [p for p in professors if p.get("biography")]

    print(f"\n{'='*50}")
    print(f"[DONE] Total professors  : {len(professors)}")
    print(f"       With research     : {len(with_research)}")
    print(f"       With biography    : {len(with_bio)}")
    print(f"       Saved to          : {OUTPUT_JSON}")
    print(f"{'='*50}\n")

    # Sample output
    if professors:
        s = professors[0]
        print(f"── Sample ─────────────────────────────────────────")
        print(f"  Name     : {s['name']}")
        print(f"  Title    : {s['title']}")
        print(f"  Email    : {s['email']}")
        print(f"  Research : {s['research_text'][:100]}")
        print(f"  Bio      : {s['biography'][:100] + '...' if s['biography'] else '[none]'}")

if __name__ == "__main__":
    main()