# Shared helpers for the scrapers.
# Shared utilities for university scrapers.
# Currently a placeholder — common helpers (headers, delay, spamspan decode)
# can be extracted here as the scraper library grows.

import re
import time
import random

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:150.0) Gecko/20100101 Firefox/150.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

def polite_delay(base: float = 1.5) -> None:
    """Sleep for base + small random jitter to avoid hammering servers."""
    time.sleep(base + random.uniform(0.2, 0.8))

def decode_spamspan(element) -> str:
    """
    Decode spamspan email format:
    <span class="u">user</span> [at] <span class="d">domain [dot] com</span>
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
