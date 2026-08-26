"""Data commands: build-index, scrape, features."""
import json
import asyncio
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cli._common import get_events_path
from config import CUTOFF_DATE, BASE_DIR


def handle_build_index(args):
    events_path = get_events_path(args)
    events_path.parent.mkdir(parents=True, exist_ok=True)

    from scraping.build_events_index import fetch_page, parse_events, EVENTS_URL

    print(f"Building events index from {EVENTS_URL} ...")
    print(f"Output: {events_path}")

    # Load existing
    existing = {}
    if events_path.exists():
        try:
            with open(events_path, "r", encoding="utf-8") as f:
                for ev in json.load(f):
                    existing[ev["url"]] = ev
        except Exception:
            existing = {}

    html = asyncio.run(fetch_page(EVENTS_URL))
    events = parse_events(html)

    new_count = 0
    for ev in events:
        if ev["url"] not in existing:
            existing[ev["url"]] = ev
            new_count += 1

    merged = sorted(existing.values(), key=lambda e: e["date"])
    with open(events_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Index saved to {events_path} with {len(merged)} events ({new_count} new)")
    return 0


def handle_scrape(args):
    from cli._common import resolve_paths

    fights_path, cache_path, _, _, _ = resolve_paths(args)
    events_path = get_events_path(args)

    # Use scraping.scrape_ufc helpers but with custom paths
    import scraping.scrape_ufc as scrape_mod

    # Patch paths in module so its save/load respect --data-dir
    orig_events = scrape_mod.EVENTS_PATH
    orig_fights = scrape_mod.FIGHTS_PATH
    orig_cache = scrape_mod.FIGHTERS_CACHE_PATH
    scrape_mod.EVENTS_PATH = str(events_path)
    scrape_mod.FIGHTS_PATH = str(fights_path)
    scrape_mod.FIGHTERS_CACHE_PATH = str(cache_path)

    # If --force, we need to reset scrapped flags before pending selection.
    # Instead of hacking main(), we handle by temporarily patching the pending logic:
    # we will run a wrapper that respects limit/force.

    # Load events to compute limited pending
    try:
        with open(events_path, "r", encoding="utf-8") as f:
            events = json.load(f)
    except FileNotFoundError:
        print(f"error: events index not found at {events_path}", file=sys.stderr)
        print("  Run `ufc data build-index` first.", file=sys.stderr)
        return 1

    if args.force:
        # force re-scrape: treat all events as pending
        pending_count = len(events)
        if args.limit is not None:
            pending_count = min(pending_count, args.limit)
        print(f"Force mode: will re-process {pending_count} events (limit={args.limit})")
        # Reset scrapped flags for limited subset
        # We will slice and reset those
        # For simplicity, run original main but with events truncated
        # Save a backup and patch file
        # Instead, we directly run loop with our pending list below (custom main)
        pass  # will use custom loop
    else:
        pending = [e for e in events if not e.get("scrapped")]
        if args.limit is not None:
            pending = pending[: args.limit]
        print(f"Found {len(events)} total events, {len(pending)} pending (limit={args.limit})")
        if not pending:
            print("No pending events.")
            return 0

    # If limit or force requested, we run a custom limited scrape loop
    # to avoid modifying the original main's pending logic.
    if args.limit is not None or args.force:
        return _run_limited_scrape(events_path, fights_path, cache_path, args.limit, args.force)

    # No limit/force: delegate to original main (which handles its own pending)
    try:
        asyncio.run(scrape_mod.main())
    finally:
        scrape_mod.EVENTS_PATH = orig_events
        scrape_mod.FIGHTS_PATH = orig_fights
        scrape_mod.FIGHTERS_CACHE_PATH = orig_cache
    return 0


def _run_limited_scrape(events_path, fights_path, cache_path, limit, force):
    """Run scrape for a limited subset, reusing scrape_ufc helpers."""
    import scraping.scrape_ufc as scrape_mod
    from tqdm import tqdm
    import asyncio as aio

    # Load data
    events = scrape_mod.load_json(str(events_path))
    fights = scrape_mod.load_json(str(fights_path))
    fighters_cache = scrape_mod.load_json(str(cache_path))
    if not fighters_cache:
        fighters_cache = {}

    if force:
        pending = events[:limit] if limit else events
        # reset scrapped for those to force re-scrape
        for e in pending:
            e["scrapped"] = False
    else:
        pending = [e for e in events if not e.get("scrapped")]
        if limit:
            pending = pending[:limit]

    print(f"Processing {len(pending)} events (limited)")

    async def _limited_main():
        from playwright.async_api import async_playwright
        import random as rnd
        import time

        p = await async_playwright().__aenter__() if False else None  # placeholder
        # Use same pattern as original main but with our pending
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
                existing_keys: set[tuple] = set()
                for f in fights:
                    existing_keys.add((
                        f.get("event_name"), f.get("fighter_1"), f.get("fighter_2"),
                        f.get("round"), f.get("time"), f.get("winner"), f.get("method"),
                    ))
                    if f.get("fight_url"):
                        existing_keys.add((f.get("event_name"), f.get("fight_url")))

                outer = tqdm(total=len(pending), position=0, leave=True, unit="event", desc="Events", ncols=100)
                for ei, event in enumerate(pending):
                    name = event["name"]
                    date = event["date"]
                    url = event["url"]
                    outer.set_description(f"Events: {name[:55]}")
                    tqdm.write(f"\nProcessing event: {name} ({date})")
                    try:
                        html = await scrape_mod.fetch_page(context, url, name)
                        if not html or len(html) < 1000:
                            tqdm.write(f"  [ERROR] Failed to fetch event page for {name}")
                            outer.update(1)
                            continue
                        event_fights = scrape_mod.parse_event_page(html, name, date)
                        if not event_fights:
                            tqdm.write(f"  [WARN] No fights found for {name}")
                            outer.update(1)
                            continue
                        tqdm.write(f"  Found {len(event_fights)} fights")
                        inner = tqdm(event_fights, position=1, leave=False, unit="fight", desc=f"  {name[:30]}", ncols=100)
                        for i, fight in enumerate(inner):
                            inner.set_description(f"  {fight['fighter_1'][:18]} vs {fight['fighter_2'][:18]}")
                            if fight["fight_url"]:
                                try:
                                    fight_html = await scrape_mod.fetch_page(context, fight["fight_url"], f"fight-{i}")
                                    if fight_html and len(fight_html) > 500:
                                        scrape_mod.parse_fight_page(fight_html, fight)
                                except Exception as e:
                                    tqdm.write(f"    [WARN] Could not fetch fight details: {e}")
                            for fname, furl in [
                                (fight["fighter_1"], fight["fighter_1_url"]),
                                (fight["fighter_2"], fight["fighter_2_url"]),
                            ]:
                                if fname and fname not in fighters_cache:
                                    try:
                                        await scrape_mod.get_fighter_info(context, fname, furl, fighters_cache, debut_date=fight["event_date"])
                                    except Exception as e:
                                        tqdm.write(f"    [WARN] Could not fetch fighter {fname}: {e}")
                            scrape_mod.enrich_fight_with_fighter_data(fight, fighters_cache)
                            clean_fight = {k: v for k, v in fight.items() if k == "fight_url" or not k.endswith("_url")}
                            key = (
                                clean_fight.get("event_name"), clean_fight.get("fighter_1"),
                                clean_fight.get("fighter_2"), clean_fight.get("round"),
                                clean_fight.get("time"), clean_fight.get("winner"),
                                clean_fight.get("method"),
                            )
                            url_key = (clean_fight.get("event_name"), clean_fight.get("fight_url"))
                            if key in existing_keys or url_key in existing_keys:
                                tqdm.write(f"    [SKIP] duplicate {clean_fight['fighter_1']} vs {clean_fight['fighter_2']} @ {clean_fight['event_name']}")
                            else:
                                fights.append(clean_fight)
                                existing_keys.add(key)
                                if fight.get("fight_url"):
                                    existing_keys.add(url_key)
                            fights_done += 1
                            elapsed = time.monotonic() - t_start
                            spf = elapsed / fights_done if fights_done else 0
                            remaining_fights = (
                                (len(pending) - ei - 1) * avg_fights_per_event
                                + (len(event_fights) - i - 1)
                            )
                            outer.set_postfix_str(f"ETA ~{scrape_mod.format_duration(spf * remaining_fights)}")
                            await aio.sleep(rnd.uniform(1.0, 2.0))
                        inner.close()
                        scrape_mod.save_json(str(fights_path), fights)
                        scrape_mod.save_json(str(cache_path), fighters_cache)
                        event["scrapped"] = True
                        # persist to original events_path (which is same list object as `events`)
                        scrape_mod.save_json(str(events_path), events)
                        tqdm.write(f"  Marked as scrapped: {name}")
                        await aio.sleep(rnd.uniform(1.0, 2.0))
                    except Exception as e:
                        tqdm.write(f"  [ERROR] Failed to process event {name}: {e}")
                    outer.update(1)
                    outer.set_postfix_str("")
                outer.close()
            finally:
                await context.close()
                await browser.close()
        tqdm.write(f"\nDone. Fights saved to {fights_path}")
        tqdm.write(f"Fighters cached: {len(fighters_cache)}")

    asyncio.run(_limited_main())
    return 0


def handle_features(args):
    from cli._common import resolve_paths
    fights_path, cache_path, dataset_path, _, _ = resolve_paths(args)

    seed = args.seed
    random.seed(seed)

    print(f"Loading fights from {fights_path} ...")
    from stats_utils import load_fights as _load_fights, load_fighter_cache as _load_cache, PriorAccumulator, COMPOSITE_GROUPS, COMPOSITE_SIGNS
    from fighter_engine import FightStateEngine, compute_stats_from_state, classify_method, compute_feature_diffs

    # Override load to use custom paths if --data-dir supplied
    if str(fights_path) != str(BASE_DIR / "data" / "fights.json") or str(cache_path) != str(BASE_DIR / "data" / "fighters_cache.json"):
        # monkey-patch paths via direct load
        import json as _json
        from pathlib import Path as _Path

        def _load_custom(p):
            with open(p, "r", encoding="utf-8") as f:
                return _json.load(f)
        try:
            fights = _load_custom(fights_path)
            fighters_cache = _load_custom(cache_path)
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    else:
        fights = _load_fights()
        fighters_cache = _load_cache()

    total_raw = len(fights)
    prior_accum = PriorAccumulator()
    engine = FightStateEngine(fights)
    discarded = total_raw - len(engine.filtered)
    print(f"Discarded {discarded} fights before {CUTOFF_DATE.date()}")

    def _build_stub_composite_params():
        means = {g: {f: 0.0 for f in feats} for g, feats in COMPOSITE_GROUPS.items()}
        stds = {g: {f: 1.0 for f in feats} for g, feats in COMPOSITE_GROUPS.items()}
        weights = {}
        for g, feats in COMPOSITE_GROUPS.items():
            signs = COMPOSITE_SIGNS.get(g, {})
            mag = 1.0 / max(len(feats), 1)
            weights[g] = {f: float(signs.get(f, 1) * mag) for f in feats}
        return {"means": means, "stds": stds, "weights": weights, "method": "stub-equal", "version": 1}

    stub_params = _build_stub_composite_params()
    rows = []
    debut_a_count = 0
    debut_b_count = 0
    unknown_stance_count = 0

    for idx, fight in enumerate(engine, 1):
        f1_name = fight["fighter_1"]
        f2_name = fight["fighter_2"]
        fighter_state = engine.state
        if random.random() < 0.5:
            a_name, b_name = f1_name, f2_name
            a_is_f1 = True
            b_is_f1 = False
        else:
            a_name, b_name = f2_name, f1_name
            a_is_f1 = False
            b_is_f1 = True

        priors = prior_accum.priors()
        feat_a = compute_stats_from_state(fighter_state[a_name], a_name, fighters_cache, fight["_parsed_date"], category=fight["category"], priors=priors, fight=fight, is_fighter_1=a_is_f1, composite_params=stub_params)
        feat_b = compute_stats_from_state(fighter_state[b_name], b_name, fighters_cache, fight["_parsed_date"], category=fight["category"], priors=priors, fight=fight, is_fighter_1=b_is_f1, composite_params=stub_params)
        prior_accum.add(fight)

        if feat_a["is_debut"]:
            debut_a_count += 1
        if feat_b["is_debut"]:
            debut_b_count += 1
        if feat_a["stance"] == "Unknown":
            unknown_stance_count += 1
        if feat_b["stance"] == "Unknown":
            unknown_stance_count += 1

        height_key_a = "fighter_1_height_cm" if a_is_f1 else "fighter_2_height_cm"
        reach_key_a = "fighter_1_reach_cm" if a_is_f1 else "fighter_2_reach_cm"
        height_a = fight.get(height_key_a)
        if height_a is None:
            height_a = fighters_cache.get(a_name, {}).get("height_cm")
        height_a = float(height_a) if height_a is not None else np.nan
        reach_a = fight.get(reach_key_a)
        if reach_a is None:
            reach_a = fighters_cache.get(a_name, {}).get("reach_cm")
        reach_a = float(reach_a) if reach_a is not None else np.nan

        height_key_b = "fighter_1_height_cm" if b_is_f1 else "fighter_2_height_cm"
        reach_key_b = "fighter_1_reach_cm" if b_is_f1 else "fighter_2_reach_cm"
        height_b = fight.get(height_key_b)
        if height_b is None:
            height_b = fighters_cache.get(b_name, {}).get("height_cm")
        height_b = float(height_b) if height_b is not None else np.nan
        reach_b = fight.get(reach_key_b)
        if reach_b is None:
            reach_b = fighters_cache.get(b_name, {}).get("reach_cm")
        reach_b = float(reach_b) if reach_b is not None else np.nan

        winner_name = fight["winner"]
        if winner_name == a_name:
            winner_label = 1
        elif winner_name == b_name:
            winner_label = 2
        elif winner_name == "Draw":
            winner_label = "Draw"
        elif winner_name == "No Contest":
            winner_label = "No Contest"
        else:
            winner_label = winner_name

        _, _, finish_type_target = classify_method(fight["method"], fight["winner"], fight["fighter_1"], fight["fighter_2"])
        fight_round_raw = fight.get("round")
        finish_round_val = int(fight_round_raw) if fight_round_raw is not None else 0

        row = {
            "fight_id": f"fight_{idx:05d}",
            "event_date": fight["event_date"],
            "category": fight.get("category", ""),
            "fighter_a_name": a_name,
            "fighter_b_name": b_name,
            "age_a": feat_a["age"],
            "age_b": feat_b["age"],
            "age_diff": feat_a["age"] - feat_b["age"] if not (np.isnan(feat_a["age"]) or np.isnan(feat_b["age"])) else np.nan,
            "stance_a": feat_a["stance"],
            "stance_b": feat_b["stance"],
            "height_diff": height_a - height_b if not (np.isnan(height_a) or np.isnan(height_b)) else np.nan,
            "reach_diff": reach_a - reach_b if not (np.isnan(reach_a) or np.isnan(reach_b)) else np.nan,
        }
        row.update(compute_feature_diffs(feat_a, feat_b))
        row.update({
            "winner": winner_label,
            "finish_type": finish_type_target if finish_type_target is not None else "OTHER",
            "finish_round": finish_round_val,
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dataset_path, index=False)
    print(f"\nFinal rows generated: {len(df)}")
    print(f"Fights with is_debut_a=True: {debut_a_count}")
    print(f"Fights with is_debut_b=True: {debut_b_count}")
    print(f"Stance 'Unknown' entries: {unknown_stance_count}")
    if len(df) > 0:
        null_counts = df.isnull().sum()
        null_pct = (df.isnull().sum() / len(df)) * 100
        null_df = pd.DataFrame({"null_count": null_counts, "null_pct": null_pct})
        nz = null_df[null_df["null_count"] > 0]
        if not nz.empty:
            print(f"\nNull summary per column:\n{nz.to_string()}")
    print(f"\nDataset saved to {dataset_path}")
    return 0
