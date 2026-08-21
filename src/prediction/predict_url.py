"""
Scrape a UFCstats event page and predict every fight with the model.

Usage:
  .venv/bin/python src/prediction/predict_url.py http://ufcstats.com/event-details/8a0a35e7c74bebcc
  .venv/bin/python src/prediction/predict_url.py --url http://ufcstats.com/event-details/... --model-path ...

Results:
- Fights whose fighters are NOT in data/fighters_cache.json OR that have 0 UFC
  fights before the event are skipped.
- Fighters with fewer than 3 fights before the event are marked with '*' next
  to their name; each fighter's UFC record (W-L-D, from fights.json) is shown
  in parentheses next to their name.
- Scheduled rounds: 5 for title fights (belt icon) and for the first fight on
  the page; 3 otherwise.
"""
import sys
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from config import MODEL_PATH, FEATURE_COLS_PATH
from fighter_engine import build_historical_context, is_debut, make_initial_state, predict_fight
from stats_utils import load_fights, load_fighter_cache

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def parse_event_page(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    h2 = soup.find("h2", class_="b-content__title")
    event_name = ""
    if h2:
        span = h2.find("span", class_="b-content__title-highlight")
        event_name = (span or h2).get_text(strip=True)

    event_date = None
    for li in soup.select("li.b-list__box-list-item"):
        text = li.get_text(" ", strip=True)
        if text.startswith("Date:"):
            date_str = text.replace("Date:", "").strip()
            try:
                event_date = datetime.strptime(date_str, "%B %d, %Y")
            except ValueError:
                event_date = None

    fights = []
    table = soup.find("table", class_="b-fight-details__table")
    if table:
        for row in table.select("tbody tr.b-fight-details__table-row"):
            cells = row.find_all("td")
            if len(cells) < 7:
                continue
            fighter_cell = cells[1]
            links = fighter_cell.find_all("a")
            if len(links) < 2:
                continue
            weight_cell = cells[6]
            category = weight_cell.get_text(strip=True)
            title_fight = bool(
                weight_cell.find("img") and "belt" in (weight_cell.find("img").get("src", "") or "")
            )
            fights.append({
                "fighter_1": links[0].get_text(strip=True),
                "fighter_2": links[1].get_text(strip=True),
                "category": category,
                "title_fight": title_fight,
            })

    return {"event_name": event_name, "event_date": event_date, "fights": fights}


def main():
    parser = argparse.ArgumentParser(description="Scrape a UFCstats event and predict all fights")
    parser.add_argument("url", nargs="?", help="UFCstats event URL, e.g. http://ufcstats.com/event-details/...")
    parser.add_argument("--url", dest="url_alt", help="Alternative --url flag")
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--features-path", default=str(FEATURE_COLS_PATH))
    args = parser.parse_args()

    url = args.url or args.url_alt
    if not url:
        parser.error("provide an event URL as positional arg or --url")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=USER_AGENT)
        page = context.new_page()
        try:
            page.goto(url, wait_until="commit", timeout=30000)
            page.wait_for_selector("table.b-fight-details__table", timeout=20000)
            page.wait_for_timeout(1500)
            html = page.content()
        finally:
            context.close()
            browser.close()

    event = parse_event_page(html)
    page_fights = event["fights"]
    event_date = event["event_date"]

    if not page_fights:
        print("No fights found on the page.")
        sys.exit(1)
    if event_date is None:
        print("Could not parse event date from the page.")
        sys.exit(1)

    fights = load_fights()
    fighters_cache = load_fighter_cache()

    model = joblib.load(args.model_path)
    feature_meta = joblib.load(args.features_path)

    fighter_states, priors = build_historical_context(fights, fighters_cache, event_date)

    results = []
    skipped = []
    for i, pf in enumerate(page_fights):
        f1, f2 = pf["fighter_1"], pf["fighter_2"]
        category = pf["category"]
        rounds = 5 if (i == 0 or pf["title_fight"]) else 3

        if f1 not in fighters_cache or f2 not in fighters_cache:
            skipped.append((f1, f2, category, "fighter not in cache"))
            continue

        state1 = fighter_states.get(f1, make_initial_state())
        state2 = fighter_states.get(f2, make_initial_state())
        tf1, tf2 = state1["total_fights"], state2["total_fights"]
        event_date_str = event_date.strftime("%Y-%m-%d")

        if is_debut(f1, fighter_states, fighters_cache, event_date_str) or is_debut(f2, fighter_states, fighters_cache, event_date_str):
            skipped.append((f1, f2, category, "0 UFC fights before the event"))
            continue

        prob_a, prob_b = predict_fight(f1, f2, category, fighter_states,
                                       fighters_cache, model, feature_meta,
                                       event_date, priors=priors)
        rec1 = (state1["wins"], state1["losses"], state1["draws"])
        rec2 = (state2["wins"], state2["losses"], state2["draws"])
        results.append((f1, f2, category, prob_a, prob_b, tf1, tf2, rounds, rec1, rec2))

    print()
    print("=" * 120)
    print(f"  {event['event_name'] or url}  ({event_date.strftime('%B %d, %Y')})")
    print("=" * 120)
    header = (f"  {'#':<4s} {'Fighter A':<32s} {'Fighter B':<32s} {'Category':<20s} "
              f"{'Rnd':<4s} {'Prob A':<8s} {'Prob B':<8s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, (f1, f2, cat, prob_a, prob_b, tf1, tf2, rounds, rec1, rec2) in enumerate(results, 1):
        r1 = f"({rec1[0]}-{rec1[1]}-{rec1[2]})"
        r2 = f"({rec2[0]}-{rec2[1]}-{rec2[2]})"
        n1 = f"{f1} {r1}{'*' if tf1 < 3 else ''}"
        n2 = f"{f2} {r2}{'*' if tf2 < 3 else ''}"
        rnd = "5" if rounds == 5 else "3"
        print(f"  {i:<4d} {n1:<32s} {n2:<32s} {cat:<20s} {rnd:<4s} "
              f"{prob_a*100:6.1f}% {prob_b*100:6.1f}%")

    print("=" * 120)

    low = [(f1, f2, tf1, tf2, rec1, rec2)
           for f1, f2, _, _, _, tf1, tf2, _, rec1, rec2 in results if tf1 < 3 or tf2 < 3]
    if low:
        print("\n  * Fighter has fewer than 3 UFC fights before this event — prediction may be unreliable.")
        for f1, f2, tf1, tf2, rec1, rec2 in low:
            r1 = f"({rec1[0]}-{rec1[1]}-{rec1[2]})"
            r2 = f"({rec2[0]}-{rec2[1]}-{rec2[2]})"
            print(f"    - {f1} {r1}{'*' if tf1 < 3 else ''} vs {f2} {r2}{'*' if tf2 < 3 else ''}")

    if skipped:
        print(f"\n  SKIPPED ({len(skipped)} fight(s)):")
        for f1, f2, cat, reason in skipped:
            print(f"    - {f1} vs {f2} ({cat}): {reason}")

    print()


if __name__ == "__main__":
    main()