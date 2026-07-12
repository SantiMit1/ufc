import json
import asyncio
from datetime import datetime

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup


EVENTS_URL = "http://ufcstats.com/statistics/events/completed?page=all"
OUTPUT_PATH = "data/events_index.json"


async def fetch_page(url: str) -> str:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")
        content = await page.content()
        await browser.close()
    return content


def parse_events(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")

    events = []
    for row in soup.select("table tbody tr"):
        tds = row.find_all("td")
        if len(tds) < 2:
            continue

        link = row.find("a")
        if not link or not link.get("href"):
            continue

        name = link.get_text(strip=True)
        url = link["href"]

        td0_text = tds[0].get_text(strip=True)
        date_str = td0_text.replace(name, "", 1).strip()

        dt = datetime.strptime(date_str, "%B %d, %Y")
        events.append({
            "name": name,
            "date": dt.strftime("%Y-%m-%d"),
            "url": url,
            "scrapped": False,
        })

    events.sort(key=lambda e: e["date"])
    return events


def main():
    existing = {}
    try:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for ev in json.load(f):
                existing[ev["url"]] = ev
    except FileNotFoundError:
        pass

    html = asyncio.run(fetch_page(EVENTS_URL))
    events = parse_events(html)

    new_count = 0
    for ev in events:
        if ev["url"] not in existing:
            existing[ev["url"]] = ev
            new_count += 1

    merged = sorted(existing.values(), key=lambda e: e["date"])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Index saved to {OUTPUT_PATH} with {len(merged)} events ({new_count} new)")


if __name__ == "__main__":
    main()
