import json
import requests
from bs4 import BeautifulSoup
import time
import os
import re

# Config
BASE_URL       = "https://www.cics.umass.edu"
# Faculty-only filter (351 = Faculty profile type)
DIRECTORY_URL  = "https://www.cics.umass.edu/about/directory?s=&field_person__profile_type_ref_target_id%5B351%5D=351"
OUTPUT_JSON    = "data/processed/umass_professors.json"
DELAY          = 1.5

HEADERS = {
    "User-Agent"     : "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:150.0) Gecko/20100101 Firefox/150.0",
    "Accept"         : "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection"     : "keep-alive",
}

# Faculty title keywords — filter out staff
FACULTY_TITLES = [
    "professor", "lecturer", "scientist", "faculty",
    "researcher", "instructor", "fellow"
]

os.makedirs("data/processed", exist_ok=True)

def is_faculty(role: str) -> bool:
    """Return True if the role string indicates a faculty member."""
    return any(kw in role.lower() for kw in FACULTY_TITLES)

def get_all_directory_pages() -> list[str]:
    """
    Fetch all paginated pages of the directory.
    Returns list of HTML strings.
    """
    pages = []
    url = DIRECTORY_URL
    page_num = 0

    while url:
        print(f"[FETCH] Directory page {page_num}: {url}")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            resp.encoding = "utf-8"
            pages.append(resp.text)
        except requests.RequestException as e:
            print(f"  [ERROR] {e}")
            break

        # Check for "next page" link
        soup = BeautifulSoup(resp.text, "html.parser")
        next_link = soup.find("a", {"rel": "next"})
        if not next_link:
            # Also check for pagination links with text "next" or "›"
            next_link = soup.find("a", title="Go to next page")
            if not next_link:
                # Try li.pager__item--next > a
                next_item = soup.find("li", class_=lambda c: c and "pager__item--next" in c)
                next_link = next_item.find("a") if next_item else None

        if next_link and next_link.get("href"):
            href = next_link["href"]
            url = BASE_URL + href if href.startswith("/") else href
            page_num += 1
            time.sleep(DELAY)
        else:
            print(f"[PAGES] No more pages found. Total pages: {page_num + 1}")
            url = None

    return pages

def parse_directory_pages(pages: list[str]) -> list[dict]:
    """
    Parse all directory pages and extract faculty cards.
    Returns list of professor dicts with basic info + profile URL.
    """
    professors = []
    seen_urls = set()

    for page_html in pages:
        soup = BeautifulSoup(page_html, "html.parser")
        cards = soup.find_all("div", class_=lambda c: c and "person--card-long" in c)
        print(f"  [PARSE] Found {len(cards)} cards on this page")

        for card in cards:
            # Profile URL
            link_tag = card.find("a", class_="person__image")
            if not link_tag or not link_tag.get("href"):
                continue
            profile_path = link_tag["href"]
            profile_url = BASE_URL + profile_path if profile_path.startswith("/") else profile_path

            # Skip duplicates
            if profile_url in seen_urls:
                continue
            seen_urls.add(profile_url)

            # Name
            name_div = card.find("div", class_="person__title")
            name = name_div.get_text(strip=True) if name_div else ""
            if not name:
                continue

            # Role/Title
            role_div = card.find("div", class_="person__role")
            role = role_div.get_text(strip=True) if role_div else ""

            # Filter: skip non-faculty (staff, admins, etc.)
            if role and not is_faculty(role):
                print(f"  [SKIP] {name} — role: '{role}'")
                continue

            # Email
            email = ""
            email_tag = card.find("a", class_="person__email")
            if email_tag:
                email = email_tag.get_text(strip=True)
            else:
                # Try spamspan format
                spamspan = card.find("span", class_="spamspan")
                if spamspan:
                    u = spamspan.find("span", class_="u")
                    d = spamspan.find("span", class_="d")
                    if u and d:
                        domain = re.sub(r'\[dot\]', '.', d.get_text())
                        domain = re.sub(r'\s+', '', domain)
                        email = f"{u.get_text(strip=True)}@{domain}"

            # Phone
            phone = ""
            phone_div = card.find("div", class_="person__phone")
            if phone_div:
                phone = phone_div.get_text(strip=True)

            # Location
            office = ""
            loc_div = card.find("div", class_="person__campus-loc")
            if loc_div:
                office = loc_div.get_text(separator=" ", strip=True)
                office = re.sub(r'\s+', ' ', office).strip()

            # Research Areas from card (may be incomplete — "first + N more")
            research_interests = []
            tags_div = card.find("div", class_="person__tags")
            if tags_div:
                research_interests = [
                    a.get_text(strip=True)
                    for a in tags_div.find_all("a")
                ]

            # Check if there are more tags ("+N more" link)
            tags_extra = card.find("a", class_="person__tags-extra")
            has_more_tags = tags_extra is not None

            professors.append({
                "name"              : name,
                "university"        : "UMass Amherst",
                "department"        : "Manning College of Information & Computer Sciences",
                "title"             : role,
                "email"             : email,
                "phone"             : phone,
                "office"            : office,
                "research_interests": research_interests,  # may be partial
                "research_text"     : ", ".join(research_interests),
                "has_more_tags"     : has_more_tags,
                "education"         : [],
                "biography"         : "",
                "profile_url"       : profile_url,
            })

    print(f"[DIRECTORY] Total faculty extracted: {len(professors)}")
    return professors

def parse_profile_page(html: str) -> dict:
    """
    Parse the individual professor profile page.
    Gets: full research areas, biography, education, lab affiliations.
    """
    soup = BeautifulSoup(html, "html.parser")
    result = {
        "research_interests": [],
        "biography"         : "",
        "education"         : [],
        "lab"               : "",
        "email"             : "",
        "phone"             : "",
        "office"            : "",
    }

    # Email (spamspan format on profile)
    spamspan = soup.find("span", class_="spamspan")
    if spamspan:
        u = spamspan.find("span", class_="u")
        d = spamspan.find("span", class_="d")
        if u and d:
            domain = re.sub(r'\[dot\]', '.', d.get_text())
            domain = re.sub(r'\s+', '', domain)
            result["email"] = f"{u.get_text(strip=True)}@{domain}"

    # Phone
    phone_a = soup.find("div", class_="page-header__phone")
    if phone_a:
        result["phone"] = phone_a.get_text(strip=True)

    # Office / Location
    loc_div = soup.find("div", class_="page-header__location")
    if loc_div:
        result["office"] = loc_div.get_text(separator=" ", strip=True)
        result["office"] = re.sub(r'\s+', ' ', result["office"]).strip()

    # Full Research Areas
    research_section = soup.find("article", class_="research-areas-tags")
    if research_section:
        result["research_interests"] = [
            a.get_text(strip=True)
            for a in research_section.find_all("a")
        ]

    # Biography
    about_section = soup.find("article", class_="about")
    if about_section:
        text_area = about_section.find("div", class_="text-area")
        if text_area:
            result["biography"] = text_area.get_text(separator=" ", strip=True)
            result["biography"] = re.sub(r'\s+', ' ', result["biography"]).strip()

    # Education
    edu_section = soup.find("article", class_="person__education")
    if edu_section:
        result["education"] = [
            li.get_text(strip=True)
            for li in edu_section.find_all("li")
            if li.get_text(strip=True)
        ]

    # Primary Lab
    lab_section = soup.find("article", class_="primary-lab")
    if lab_section:
        lab_link = lab_section.find("a")
        if lab_link:
            result["lab"] = lab_link.get_text(strip=True)

    return result

def scrape_profiles(professors: list) -> list:
    """Visit each professor's profile page to enrich data."""
    for i, prof in enumerate(professors, 1):
        url = prof.get("profile_url", "")
        if not url:
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

        # Merge: profile page is authoritative (has full research areas)
        if extras["research_interests"]:
            prof["research_interests"] = extras["research_interests"]
            prof["research_text"] = ", ".join(extras["research_interests"])

        if extras["biography"]:
            prof["biography"] = extras["biography"]

        if extras["education"]:
            prof["education"] = extras["education"]

        if extras["email"] and not prof["email"]:
            prof["email"] = extras["email"]

        if extras["phone"] and not prof["phone"]:
            prof["phone"] = extras["phone"]

        if extras["office"] and not prof["office"]:
            prof["office"] = extras["office"]

        if extras["lab"]:
            prof["lab"] = extras["lab"]

        # Remove helper field
        prof.pop("has_more_tags", None)

        print(f"  Research: {prof['research_text'][:80]}{'...' if len(prof['research_text']) > 80 else ''}")
        print(f"  Bio: {'Yes' if prof['biography'] else 'No'}")

        # Checkpoint every 10
        if i % 10 == 0:
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(professors, f, indent=2, ensure_ascii=False)
            print(f"  [CHECKPOINT] {i} saved.\n")

        time.sleep(DELAY)

    return professors

def main():
    # 1. Fetch all directory pages (handles pagination)
    pages = get_all_directory_pages()
    print(f"\n[PAGES] Fetched {len(pages)} page(s) total\n")

    # 2. Parse faculty cards from all pages
    professors = parse_directory_pages(pages)

    if not professors:
        print("[ABORT] No faculty found.")
        return

    # Quick save before profile scraping
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(professors, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {len(professors)} faculty to {OUTPUT_JSON}")

    # 3. Enrich with individual profile pages
    print(f"\n[PROFILES] Scraping {len(professors)} individual pages...\n")
    professors = scrape_profiles(professors)

    # 4. Final save
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(professors, f, indent=2, ensure_ascii=False)

    # 5. Summary
    with_research = [p for p in professors if p.get("research_text")]
    with_bio      = [p for p in professors if p.get("biography")]

    print(f"\n{'='*50}")
    print(f"[DONE] Total faculty    : {len(professors)}")
    print(f"       With research    : {len(with_research)}")
    print(f"       With biography   : {len(with_bio)}")
    print(f"       Saved to         : {OUTPUT_JSON}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()