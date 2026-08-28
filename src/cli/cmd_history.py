"""History command — fighter Elo history."""

import json
import json as _json
import sys
from datetime import datetime
from pathlib import Path

from cli._common import resolve_paths
from config import FIGHTS_PATH as _default_fights
from fighter_history import build_history, print_table, select_fighter
from fighter_history import load_fights as _load


def handle_history(args):
    as_json = args.json
    fighter_arg = args.fighter

    # Load fights respecting --data-dir
    fights_path, cache_path, _, _, _ = resolve_paths(args)

    # Use same load as fighter_history.py but with custom path

    if str(fights_path) != str(_default_fights):
        with open(fights_path, "r", encoding="utf-8") as f:
            fights = _json.load(f)
    else:
        fights = _load(Path(fights_path))

    # All names for interactive
    all_names = set()
    for f in fights:
        all_names.add(f["fighter_1"])
        all_names.add(f["fighter_2"])

    # If fighter provided via CLI arg, do single run
    if fighter_arg:
        fighter = fighter_arg.strip()
        # fuzzy match like interactive: find closest case-insensitive
        if fighter not in all_names:
            # try case-insensitive exact
            lower_map = {n.lower(): n for n in all_names}
            if fighter.lower() in lower_map:
                fighter = lower_map[fighter.lower()]
            else:
                # substring search
                q = fighter.lower()
                matches = [n for n in all_names if q in n.lower()]
                if not matches:
                    print(f"No fighter found matching '{fighter_arg}'", file=sys.stderr)
                    return 1
                if len(matches) == 1:
                    fighter = matches[0]
                else:
                    print(f"Multiple matches for '{fighter_arg}':", file=sys.stderr)
                    for m in sorted(matches)[:10]:
                        print(f"  - {m}", file=sys.stderr)
                    return 1


        history, _ = build_history(fighter, fights)
        if not history:
            print(f"No fights found for '{fighter}'", file=sys.stderr)
            return 1

        if as_json:
            # convert datetimes to strings
            def _ser(o):
                if isinstance(o, datetime):
                    return o.strftime("%Y-%m-%d")
                return str(o)
            serial = []
            for h in history:
                nh = {k: (v.strftime("%Y-%m-%d") if isinstance(v, datetime) else v) for k, v in h.items()}
                serial.append(nh)
            print(json.dumps({"fighter": fighter, "history": serial}, ensure_ascii=False, indent=2, default=_ser))
        else:
            print_table(history, fighter)
        return 0

    # Interactive mode (no arg)

    print("=" * 60)
    print("  HISTORIAL ELO DE PELEADOR")
    print("=" * 60)
    print(f"\n  {len(fights)} fights loaded")
    print(f"  {len(all_names)} fighters")

    while True:
        fighter = select_fighter("Enter fighter name", all_names)
        if fighter is None:
            print("\nExiting.")
            break
        history, _ = build_history(fighter, fights)
        if not history:
            print(f"'{fighter}' no fights found.")
            continue
        if as_json:
            def _ser2(o):
                if isinstance(o, datetime):
                    return o.strftime("%Y-%m-%d")
                return str(o)
            serial = [{k: (v.strftime("%Y-%m-%d") if isinstance(v, datetime) else v) for k, v in h.items()} for h in history]
            print(json.dumps({"fighter": fighter, "history": serial}, ensure_ascii=False, indent=2, default=_ser2))
        else:
            print_table(history, fighter)
    return 0
