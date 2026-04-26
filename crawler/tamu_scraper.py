import json
import requests
from bs4 import BeautifulSoup
import time
import os

# Config
PROFILE_DATA_URL = "https://engineering.tamu.edu/profile-data.json"
OUTPUT_JSON      = "data/processed/tamu_professors.json"
BASE_URL         = "https://engineering.tamu.edu"
TARGET_DEPT      = "csce"
TARGET_ROLE      = "Faculty"
DELAY            = 1.5

# Two sets of headers — one for the JSON API, one for profile pages
API_HEADERS = {
    "User-Agent"       : "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:149.0) Gecko/20100101 Firefox/149.0",
    "Accept"           : "application/json, text/javascript, */*; q=0.01",
    "Accept-Language"  : "en-US,en;q=0.9",
    "Referer"          : "https://engineering.tamu.edu/cse/profiles/index.html",
    "X-Requested-With" : "XMLHttpRequest",
    "Connection"       : "keep-alive",
    "Pragma"           : "no-cache",
    "Cache-Control"    : "no-cache",
}

PROFILE_HEADERS = {
    "User-Agent"     : "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:149.0) Gecko/20100101 Firefox/149.0",
    "Accept"         : "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection"     : "keep-alive",
}

os.makedirs("data/processed", exist_ok=True)

def fetch_faculty_list() -> list:
    """
    Hit the TAMU profile data API and return only CSE Faculty.
    """
    print(f"[FETCH] Pulling fresh data from:\n  {PROFILE_DATA_URL}\n")

    response = requests.get(PROFILE_DATA_URL, headers=API_HEADERS, timeout=30)
    response.raise_for_status()

    raw_data = response.json()

    # Handle if wrapped in a key
    if isinstance(raw_data, dict):
        for key in ["people", "faculty", "data", "results"]:
            if key in raw_data:
                raw_data = raw_data[key]
                break

    # Filter: must have TARGET_DEPT and TARGET_ROLE in tags
    filtered = [
        p for p in raw_data
        if TARGET_DEPT.lower() in [t.lower() for t in p.get("tag", [])]
        and TARGET_ROLE.lower() in [t.lower() for t in p.get("tag", [])]
    ]

    print(f"[FILTER] Total records: {len(raw_data)}")
    print(f"[FILTER] CSE Faculty only: {len(filtered)}\n")
    return filtered

def build_url(link: str) -> str:
    return BASE_URL + link

def parse_profile(html: str, url: str, base_info: dict) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    # Name
    name_tag = soup.find("h1", class_="headline-group")
    name = name_tag.get_text(strip=True) if name_tag else base_info.get("name", "")

    # Titles
    titles = []
    titles_div = soup.find("div", class_="profile__titles")
    if titles_div:
        titles = [li.get_text(strip=True) for li in titles_div.find_all("li")]

    # Contact
    email, phone, office, website = "", "", "", ""
    contact_div = soup.find("div", class_="profile__contact")
    if contact_div:
        for li in contact_div.find_all("li"):
            label_tag = li.find("span", class_="profile__contact-label")
            if not label_tag:
                continue
            label = label_tag.get_text(strip=True).lower()
            value = li.get_text(strip=True).replace(label_tag.get_text(strip=True), "").strip()

            if "email" in label:
                a = li.find("a")
                email = a.get_text(strip=True) if a else value
            elif "phone" in label:
                phone = value
            elif "office" in label:
                office = value
            elif "website" in label:
                a = li.find("a")
                website = a["href"] if a and a.get("href") else value

    # Research Interests
    research_interests = []
    research_section = soup.find("div", id="researchinterest")
    if research_section:
        research_interests = [
            li.get_text(strip=True)
            for li in research_section.find_all("li")
            if li.get_text(strip=True)
        ]

    # Education
    education = []
    edu_section = soup.find("div", id="educationbackground")
    if edu_section:
        education = [
            li.get_text(strip=True)
            for li in edu_section.find_all("li")
            if li.get_text(strip=True)
        ]

    # Awards
    awards = []
    awards_section = soup.find("div", id="awards")
    if awards_section:
        awards = [
            li.get_text(strip=True)
            for li in awards_section.find_all("li")
            if li.get_text(strip=True)
        ]

    # Publications
    publications = []
    pub_section = soup.find("div", id="publications")
    if pub_section:
        publications = [
            li.get_text(strip=True)
            for li in pub_section.find_all("li")
            if li.get_text(strip=True)
        ]

    # Google Scholar
    scholar_url = ""
    for a in soup.find_all("a", href=True):
        if "scholar.google.com" in a["href"]:
            scholar_url = a["href"]
            break

    return {
        "name"                 : name,
        "university"           : "Texas A&M University",
        "department"           : "Computer Science & Engineering",
        "titles"               : titles,
        "email"                : email,
        "phone"                : phone,
        "office"               : office,
        "personal_website"     : website,
        "google_scholar"       : scholar_url,
        "research_interests"   : research_interests,
        "research_text"        : ", ".join(research_interests),  # used for embeddings
        "education"            : education,
        "awards"               : awards,
        "selected_publications": publications,
        "source_url"           : url,
    }

def scrape_all(faculty_list: list) -> list:
    results = []

    for i, person in enumerate(faculty_list, 1):
        link = person.get("link", "")
        if not link:
            print(f"  [SKIP] No link for {person.get('name')}")
            continue

        url = build_url(link)
        print(f"[{i}/{len(faculty_list)}] {person['name']}")
        print(f"  → {url}")

        try:
            response = requests.get(url, headers=PROFILE_HEADERS, timeout=15)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"  [ERROR] {e}")
            results.append({"name": person.get("name"), "source_url": url, "error": str(e)})
            time.sleep(DELAY)
            continue

        profile = parse_profile(response.text, url, person)

        # Quick feedback
        ri = profile["research_text"]
        print(f"  Research: {ri[:80] + '...' if len(ri) > 80 else ri or '[NONE FOUND]'}")

        results.append(profile)

        # Save checkpoint every 10
        if i % 10 == 0:
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  [CHECKPOINT] {i} profiles saved.\n")

        time.sleep(DELAY)

    return results

def main():
    # 1. Fetch fresh faculty list from API
    faculty = fetch_faculty_list()

    if not faculty:
        print("[ABORT] No faculty found. Check TARGET_DEPT and TARGET_ROLE.")
        return

    # 2. Scrape each profile page
    results = scrape_all(faculty)

    # 3. Save final output
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 4. Summary
    success = [r for r in results if "error" not in r and r.get("research_text")]
    empty   = [r for r in results if "error" not in r and not r.get("research_text")]
    errors  = [r for r in results if "error" in r]

    print(f"\n{'='*50}")
    print(f"[DONE] Total scraped  : {len(results)}")
    print(f"       With research  : {len(success)}")
    print(f"       Empty research : {len(empty)}")  # profile exists but no research listed
    print(f"       Errors         : {len(errors)}")
    print(f"       Saved to       : {OUTPUT_JSON}")
    print(f"{'='*50}")

    if empty:
        print("\n[WARN] These professors had no research interests listed:")
        for r in empty:
            print(f"  - {r['name']} → {r['source_url']}")

if __name__ == "__main__":
    main()