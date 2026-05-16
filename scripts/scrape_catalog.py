"""
SHL Catalog Scraper
Scrapes https://www.shl.com/solutions/products/product-catalog/
Restricted to Individual Test Solutions (type=1) only.

Run: python scripts/scrape_catalog.py
Output: data/catalog.json
"""
import json
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path

BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/solutions/products/product-catalog/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; SHL-Recommender-Bot/1.0; "
        "+https://github.com/your-handle/shl-recommender)"
    )
}

# SHL test type codes
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "M": "Motivation",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def fetch_page(start: int = 0, type_filter: int = 1) -> BeautifulSoup:
    """Fetch one page of the catalog."""
    params = {"start": start, "type": type_filter}
    resp = requests.get(CATALOG_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def parse_assessment_row(row) -> dict | None:
    """Parse a single <tr> row from the catalog table into an assessment dict."""
    cells = row.find_all("td")
    if not cells:
        return None

    # Column 0: Name + link
    name_cell = cells[0]
    a_tag = name_cell.find("a")
    if not a_tag:
        return None

    name = a_tag.get_text(strip=True)
    relative_url = a_tag.get("href", "")
    url = BASE_URL + relative_url if relative_url.startswith("/") else relative_url

    # Column 1+: test type flags (each cell has a span if the flag is active)
    # SHL catalog columns: Remote Testing | Adaptive/IRT | A | B | C | D | E | K | M | P | S
    type_flags = []
    type_codes = ["A", "B", "C", "D", "E", "K", "M", "P", "S"]

    remote_testing = False
    adaptive = False

    for i, cell in enumerate(cells[1:], start=1):
        # Check for a bullet/checkmark indicating this flag is set
        has_flag = bool(cell.find("span") or cell.get_text(strip=True))
        if i == 1:
            remote_testing = has_flag
        elif i == 2:
            adaptive = has_flag
        elif i - 3 < len(type_codes) and has_flag:
            type_flags.append(type_codes[i - 3])

    # Fallback: if parsing failed to find types, derive from URL or name
    if not type_flags:
        name_lower = name.lower()
        if any(k in name_lower for k in ["personality", "opq", "mqm", "motiv"]):
            type_flags = ["P"]
        elif any(k in name_lower for k in ["java", "python", "sql", "excel", ".net", "c++", "c#"]):
            type_flags = ["K"]
        elif any(k in name_lower for k in ["numerical", "verbal", "inductive", "deductive", "reasoning"]):
            type_flags = ["A"]
        else:
            type_flags = ["A"]

    # primary test type = first flag
    primary_type = type_flags[0] if type_flags else "A"

    return {
        "name": name,
        "url": url,
        "test_type": primary_type,
        "test_types": type_flags,
        "remote_testing": remote_testing,
        "adaptive": adaptive,
        "description": "",  # populated by detail scrape
    }


def fetch_assessment_detail(url: str) -> dict:
    """Fetch the detail page to get description."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Extract description from the product detail page
        description = ""

        # Try common selectors for description
        for selector in [
            ".product-hero__description",
            ".product-description",
            ".hero-description",
            "meta[name='description']",
            ".content-intro",
        ]:
            el = soup.select_one(selector)
            if el:
                if el.name == "meta":
                    description = el.get("content", "")
                else:
                    description = el.get_text(strip=True, separator=" ")
                if description:
                    break

        # Fallback: grab first substantial paragraph
        if not description:
            for p in soup.find_all("p"):
                text = p.get_text(strip=True)
                if len(text) > 80:
                    description = text
                    break

        # Duration hint
        duration_min = None
        full_text = soup.get_text(" ", strip=True)
        import re
        match = re.search(r"(\d+)\s*(?:to\s*(\d+)\s*)?minutes?", full_text, re.I)
        if match:
            duration_min = int(match.group(2) or match.group(1))

        return {"description": description[:500], "duration_minutes": duration_min}
    except Exception:
        return {"description": "", "duration_minutes": None}


def scrape_catalog(
    fetch_details: bool = True,
    detail_delay: float = 0.5,
) -> list[dict]:
    """
    Scrape entire Individual Test Solutions catalog.

    Args:
        fetch_details: If True, visit each assessment page for description.
        detail_delay: Seconds to wait between detail page requests (be polite).
    """
    assessments = []
    start = 0
    page_size = 12  # SHL returns 12 per page

    print("Scraping catalog pages...")
    while True:
        soup = fetch_page(start=start, type_filter=1)

        # Find catalog table
        table = soup.find("table", {"class": lambda c: c and "product" in c.lower()})
        if not table:
            # Try generic table
            table = soup.find("table")

        if not table:
            print(f"  No table found at start={start}, stopping.")
            break

        rows = table.find_all("tr")[1:]  # skip header
        if not rows:
            print(f"  No rows at start={start}, done.")
            break

        parsed_count = 0
        for row in rows:
            assessment = parse_assessment_row(row)
            if assessment:
                assessments.append(assessment)
                parsed_count += 1

        print(f"  start={start}: parsed {parsed_count} assessments (total: {len(assessments)})")

        if parsed_count < page_size:
            break

        start += page_size
        time.sleep(0.3)

    if fetch_details and assessments:
        print(f"\nFetching detail pages for {len(assessments)} assessments...")
        for i, a in enumerate(assessments):
            detail = fetch_assessment_detail(a["url"])
            a.update(detail)
            if i % 10 == 0:
                print(f"  {i}/{len(assessments)} details fetched")
            time.sleep(detail_delay)

    return assessments


def save_catalog(assessments: list[dict], path: str = "data/catalog.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(assessments)} assessments to {path}")


if __name__ == "__main__":
    catalog = scrape_catalog(fetch_details=True, detail_delay=0.5)
    save_catalog(catalog)
    print(f"\nDone. Total assessments: {len(catalog)}")
    print("Sample:", json.dumps(catalog[:2], indent=2))
