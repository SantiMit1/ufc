import json
import asyncio
import random
import re
import os
import time
from datetime import datetime

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from tqdm import tqdm

EVENTS_PATH = "data/events_index.json"
FIGHTS_PATH = "data/fights.json"
FIGHTERS_CACHE_PATH = "data/fighters_cache.json"


def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if not content.strip():
        return []
    return json.loads(content)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


async def fetch_page(context, url, label=""):
    page = await context.new_page()
    try:
        await page.goto(url, wait_until="commit", timeout=30000)
        await page.wait_for_selector("body.b-page, table, h2", timeout=15000)
        await page.wait_for_timeout(1500)
        content = await page.content()
        return content
    except Exception as e:
        tqdm.write(f"    [WARN] fetch_page {label}: {e}")
        return None
    finally:
        await page.close()


def split_fighter_values(value):
    if not value:
        return ("", "")

    if " of " in value:
        parts = value.split(" of ")
        if len(parts) == 3:
            a, bcd, d = parts
            bcd = bcd.strip()
            for split_pos in range(1, len(bcd)):
                b, c = bcd[:split_pos], bcd[split_pos:]
                if b.isdigit() and c.isdigit():
                    if int(a) <= int(b) and int(c) <= int(d):
                        return (f"{a} of {b}", f"{c} of {d}")
            mid = len(bcd) // 2
            b, c = bcd[:mid], bcd[mid:]
            return (f"{a} of {b}", f"{c} of {d}")

    if "%" in value:
        parts = value.split("%")
        if len(parts) >= 3:
            return (parts[0] + "%", parts[1] + "%")
        elif len(parts) == 2:
            return (parts[0] + "%", parts[1])

    if ":" in value:
        mid = len(value) // 2
        return (value[:mid], value[mid:])

    if value and set(value) <= {"-"}:
        mid = len(value) // 2
        return (value[:mid], value[mid:])

    if len(value) % 2 == 0:
        mid = len(value) // 2
        return (value[:mid], value[mid:])

    return (value, "")


def get_cell_values(cell):
    """Return (fighter_1_value, fighter_2_value) for a stats cell.

    ufcstats.com renders each fighter's value in a separate
    <p class="b-fight-details__table-text"> element. Splitting the
    concatenated cell text with split_fighter_values() is fragile because
    two fighter values can share digits ambiguously (e.g. '1 of 11' + '2 of
    12' -> '1 of 112 of 12'), so read the <p> elements directly and only
    fall back to text-splitting if that structure is missing.
    """
    ps = cell.find_all("p", class_="b-fight-details__table-text")
    if len(ps) >= 2:
        return ps[0].get_text(strip=True), ps[1].get_text(strip=True)
    return split_fighter_values(cell.get_text(strip=True))


def parse_stats_value(raw):
    raw = raw.strip()
    if not raw or raw == "---" or raw == "--":
        return {"landed": None, "attempted": None}

    m = re.match(r"(\d+) of (\d+)", raw)
    if m:
        return {"landed": int(m.group(1)), "attempted": int(m.group(2))}

    m = re.match(r"(\d+)%", raw)
    if m:
        return int(m.group(1))

    m = re.match(r"(\d+):(\d+)", raw)
    if m:
        minutes = int(m.group(1))
        seconds = int(m.group(2))
        return minutes * 60 + seconds

    return raw


def parse_event_page(html, event_name, event_date):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="b-fight-details__table")
    if not table:
        print(f"  [WARN] No fight table found for {event_name}")
        return []

    fights = []
    rows = table.select("tbody tr.b-fight-details__table-row")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 10:
            continue

        wl_text = cells[0].get_text(strip=True)
        fight_link = cells[0].find("a")
        fight_url = fight_link["href"] if fight_link else None

        fighter_cell = cells[1]
        fighter_links = fighter_cell.find_all("a")
        f1_name = fighter_links[0].get_text(strip=True) if len(fighter_links) > 0 else ""
        f2_name = fighter_links[1].get_text(strip=True) if len(fighter_links) > 1 else ""
        f1_url = fighter_links[0]["href"] if len(fighter_links) > 0 else None
        f2_url = fighter_links[1]["href"] if len(fighter_links) > 1 else None

        weight_cell = cells[6]
        category = weight_cell.get_text(strip=True)


        method = cells[7].get_text(strip=True)
        round_num = cells[8].get_text(strip=True)
        time_str = cells[9].get_text(strip=True)

        if not round_num.isdigit():
            round_num = None

        wl_lower = wl_text.lower()
        if wl_lower == "win":
            winner = f1_name
        elif wl_lower == "loss":
            winner = f2_name
        elif "draw" in wl_lower:
            winner = "Draw"
        elif "nc" in wl_lower or "no contest" in wl_lower:
            winner = "No Contest"
        elif "Overturned" in method:
            winner = "No Contest"
        else:
            winner = None

        fight = {
            "event_name": event_name,
            "event_date": event_date,
            "category": category,
            "fighter_1": f1_name,
            "fighter_2": f2_name,
            "fighter_1_url": f1_url,
            "fighter_2_url": f2_url,
            "winner": winner,
            "method": method,
            "round": int(round_num) if round_num else None,
            "time": time_str if time_str else None,
            "referee": None,
            "fight_url": fight_url,
        }
        fights.append(fight)

    return fights


def parse_fight_page(html, fight):
    soup = BeautifulSoup(html, "html.parser")

    content_div = soup.find("div", class_="b-fight-details__content")
    if content_div:
        text_items = content_div.find_all("i", class_="b-fight-details__text-item")
        for item in text_items:
            label_el = item.find("i", class_="b-fight-details__label")
            if not label_el:
                continue
            label = label_el.get_text(strip=True).replace(":", "").strip()
            label_el.extract()
            value = item.get_text(strip=True)
            if label == "Referee":
                fight["referee"] = value if value else None
            elif label == "Method":
                if value:
                    fight["method"] = value
            elif label == "Round":
                if value and value.isdigit():
                    fight["round"] = int(value)
            elif label == "Time":
                if value:
                    fight["time"] = value

    all_tables = soup.find_all("table")
    # ufcstats.com renders per-round tables with the 'js-fight-table' class and
    # the totals / significant-strike breakdown tables WITHOUT any class.
    tables = [t for t in all_tables if "js-fight-table" not in (t.get("class") or [])]
    stats = {"fighter_1": {}, "fighter_2": {}}

    if len(tables) >= 1:
        rows = tables[0].select("tbody tr")
        if rows:
            cells = rows[0].find_all("td")
            if len(cells) >= 10:
                col_map = {
                    1: "knockdowns",
                    2: "sig_strikes",
                    4: "total_strikes",
                    5: "takedowns",
                    7: "sub_attempts",
                    9: "control_time",
                }

                for col_idx, key in col_map.items():
                    if col_idx < len(cells):
                        f1, f2 = get_cell_values(cells[col_idx])
                        if key == "control_time":
                            parsed_1 = parse_stats_value(f1) if f1 != "--" else None
                            parsed_2 = parse_stats_value(f2) if f2 != "--" else None
                            if isinstance(parsed_1, int):
                                stats["fighter_1"]["control_time_seconds"] = parsed_1
                            if isinstance(parsed_2, int):
                                stats["fighter_2"]["control_time_seconds"] = parsed_2
                        elif key in ("knockdowns", "sub_attempts"):
                            f1v = int(f1) if f1.isdigit() else None
                            f2v = int(f2) if f2.isdigit() else None
                            stats["fighter_1"][key] = f1v
                            stats["fighter_2"][key] = f2v
                        elif key in ("sig_strikes", "total_strikes", "takedowns"):
                            parsed_1 = parse_stats_value(f1)
                            parsed_2 = parse_stats_value(f2)
                            if isinstance(parsed_1, dict):
                                stats["fighter_1"][key] = parsed_1
                            if isinstance(parsed_2, dict):
                                stats["fighter_2"][key] = parsed_2

    if len(tables) >= 2:
        rows = tables[1].select("tbody tr")
        if rows:
            cells = rows[0].find_all("td")
            if len(cells) >= 9:
                target_map = {
                    3: "head",
                    4: "body",
                    5: "leg",
                }
                position_map = {
                    6: "distance",
                    7: "clinch",
                    8: "ground",
                }

                for col_idx, key in target_map.items():
                    if col_idx < len(cells):
                        f1, f2 = get_cell_values(cells[col_idx])
                        p1 = parse_stats_value(f1)
                        p2 = parse_stats_value(f2)
                        stats["fighter_1"][key] = p1 if isinstance(p1, dict) else None
                        stats["fighter_2"][key] = p2 if isinstance(p2, dict) else None

                for col_idx, key in position_map.items():
                    if col_idx < len(cells):
                        f1, f2 = get_cell_values(cells[col_idx])
                        p1 = parse_stats_value(f1)
                        p2 = parse_stats_value(f2)
                        stats["fighter_1"][key] = p1 if isinstance(p1, dict) else None
                        stats["fighter_2"][key] = p2 if isinstance(p2, dict) else None

    fight["stats_fighter_1"] = stats["fighter_1"]
    fight["stats_fighter_2"] = stats["fighter_2"]


def parse_fighter_page(html):
    soup = BeautifulSoup(html, "html.parser")

    info_box = soup.find("div", class_="b-list__info-box")
    if not info_box:
        return {}

    data = {}

    items = info_box.find_all("li", class_="b-list__box-list-item")
    for item in items:
        label_el = item.find("i", class_="b-list__box-item-title")
        if not label_el:
            continue
        label = label_el.get_text(strip=True).replace(":", "").strip()
        label_el.extract()
        value = item.get_text(strip=True)

        if label == "Height":
            data["height_cm"] = convert_height(value)
        elif label == "Weight":
            data["weight_lbs"] = convert_weight(value)
        elif label == "Reach":
            data["reach_cm"] = convert_reach(value)
        elif label == "STANCE":
            data["stance"] = value if value and value != "--" else None
        elif label == "DOB":
            data["dob"] = convert_dob(value)

    return data


def convert_height(raw):
    if not raw or raw == "--":
        return None
    m = re.match(r"(\d+)'\s*(\d+)\"?", raw)
    if m:
        feet = int(m.group(1))
        inches = int(m.group(2))
        total_inches = feet * 12 + inches
        return round(total_inches * 2.54)
    return None


def convert_weight(raw):
    if not raw or raw == "--":
        return None
    m = re.match(r"(\d+)", raw)
    if m:
        return int(m.group(1))
    return None


def convert_reach(raw):
    if not raw or raw == "--":
        return None
    m = re.match(r"(\d+)\"?", raw)
    if m:
        inches = int(m.group(1))
        return round(inches * 2.54)
    return None


def convert_dob(raw):
    if not raw or raw == "--":
        return None
    try:
        dt = datetime.strptime(raw, "%b %d, %Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return raw


def calculate_age(dob_str, event_date_str):
    if not dob_str or not event_date_str:
        return None
    try:
        dob = datetime.strptime(dob_str, "%Y-%m-%d")
        event = datetime.strptime(event_date_str, "%Y-%m-%d")
        return event.year - dob.year - ((event.month, event.day) < (dob.month, dob.day))
    except (ValueError, TypeError):
        return None


def format_duration(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


async def get_fighter_info(context, fighter_name, fighter_url, fighters_cache, debut_date=None):
    if fighter_name in fighters_cache:
        return fighters_cache[fighter_name]

    tqdm.write(f"    Fetching fighter: {fighter_name}")
    html = await fetch_page(context, fighter_url, fighter_name)
    if not html or len(html) < 500:
        tqdm.write(f"    [WARN] Failed to fetch fighter page for {fighter_name}")
        fighters_cache[fighter_name] = {"debut_date": debut_date}
        return fighters_cache[fighter_name]

    info = parse_fighter_page(html)
    info["debut_date"] = debut_date
    fighters_cache[fighter_name] = info
    tqdm.write(f"    Cached fighter: {fighter_name}")
    return info


def enrich_fight_with_fighter_data(fight, fighters_cache):
    f1_info = fighters_cache.get(fight["fighter_1"], {})
    f2_info = fighters_cache.get(fight["fighter_2"], {})
    fight["fighter_1_age"] = calculate_age(f1_info.get("dob"), fight["event_date"])
    fight["fighter_2_age"] = calculate_age(f2_info.get("dob"), fight["event_date"])
    fight["fighter_1_height_cm"] = f1_info.get("height_cm")
    fight["fighter_1_reach_cm"] = f1_info.get("reach_cm")
    fight["fighter_2_height_cm"] = f2_info.get("height_cm")
    fight["fighter_2_reach_cm"] = f2_info.get("reach_cm")


async def main():
    events = load_json(EVENTS_PATH)
    fights = load_json(FIGHTS_PATH)
    fighters_cache = load_json(FIGHTERS_CACHE_PATH)

    if not fighters_cache:
        fighters_cache = {}

    pending = [e for e in events if not e.get("scrapped")]
    tqdm.write(f"Loaded {len(events)} events, {len(pending)} pending")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        try:
            scraped_count = sum(1 for e in events if e.get("scrapped"))
            avg_fights_per_event = round(len(fights) / scraped_count) if scraped_count and fights else 11
            t_start = time.monotonic()
            fights_done = 0

            outer = tqdm(total=len(pending), position=0, leave=True, unit="event", desc="Events", ncols=100)
            for ei, event in enumerate(pending):
                name = event["name"]
                date = event["date"]
                url = event["url"]
                outer.set_description(f"Events: {name[:55]}")
                tqdm.write(f"\nProcessing event: {name} ({date})")

                try:
                    html = await fetch_page(context, url, name)
                    if not html or len(html) < 1000:
                        tqdm.write(f"  [ERROR] Failed to fetch event page for {name}")
                        outer.update(1)
                        continue

                    event_fights = parse_event_page(html, name, date)
                    if not event_fights:
                        tqdm.write(f"  [WARN] No fights found for {name}")
                        outer.update(1)
                        continue

                    tqdm.write(f"  Found {len(event_fights)} fights")

                    inner = tqdm(event_fights, position=1, leave=False, unit="fight",
                                 desc=f"  {name[:30]}", ncols=100)
                    for i, fight in enumerate(inner):
                        inner.set_description(
                            f"  {fight['fighter_1'][:18]} vs {fight['fighter_2'][:18]}"
                        )

                        if fight["fight_url"]:
                            try:
                                fight_html = await fetch_page(context, fight["fight_url"], f"fight-{i}")
                                if fight_html and len(fight_html) > 500:
                                    parse_fight_page(fight_html, fight)
                            except Exception as e:
                                tqdm.write(f"    [WARN] Could not fetch fight details: {e}")

                        for fname, furl in [
                            (fight["fighter_1"], fight["fighter_1_url"]),
                            (fight["fighter_2"], fight["fighter_2_url"]),
                        ]:
                            if fname and fname not in fighters_cache:
                                try:
                                    await get_fighter_info(context, fname, furl, fighters_cache,
                                                           debut_date=fight["event_date"])
                                except Exception as e:
                                    tqdm.write(f"    [WARN] Could not fetch fighter {fname}: {e}")

                        enrich_fight_with_fighter_data(fight, fighters_cache)

                        clean_fight = {k: v for k, v in fight.items() if not k.endswith("_url") and k != "fight_url"}
                        fights.append(clean_fight)

                        fights_done += 1
                        elapsed = time.monotonic() - t_start
                        spf = elapsed / fights_done
                        remaining_fights = (
                            (len(pending) - ei - 1) * avg_fights_per_event
                            + (len(event_fights) - i - 1)
                        )
                        outer.set_postfix_str(f"ETA ~{format_duration(spf * remaining_fights)}")

                        await asyncio.sleep(random.uniform(1.0, 2.0))

                    inner.close()

                    save_json(FIGHTS_PATH, fights)
                    save_json(FIGHTERS_CACHE_PATH, fighters_cache)

                    event["scrapped"] = True
                    save_json(EVENTS_PATH, events)
                    tqdm.write(f"  Marked as scrapped: {name}")

                    await asyncio.sleep(random.uniform(1.0, 2.0))

                except Exception as e:
                    tqdm.write(f"  [ERROR] Failed to process event {name}: {e}")

                outer.update(1)
                outer.set_postfix_str("")
            outer.close()

        finally:
            await context.close()
            await browser.close()

    tqdm.write(f"\nDone. Fights saved to {FIGHTS_PATH}")
    tqdm.write(f"Fighters cached: {len(fighters_cache)}")


if __name__ == "__main__":
    asyncio.run(main())
