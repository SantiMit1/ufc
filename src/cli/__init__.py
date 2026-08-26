"""Unified CLI for UFC predictor — entry point `ufc = "cli:main"`."""

import argparse
import sys
from pathlib import Path
from importlib.metadata import version as pkg_version, PackageNotFoundError

# Ensure src is on sys.path when run as `python src/cli.py` without install
try:
    import cli._common  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cli._common  # noqa: F401


def _get_version():
    try:
        return pkg_version("ufc-predictor")
    except PackageNotFoundError:
        return "0.1.0 (dev)"


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ufc",
        description="UFC fight predictor — scraping, features, training and predictions",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s { _get_version() }")
    parser.add_argument("--model", default=None, help="path to model .pkl (default: models/ufc_stacking_ensemble.pkl)")
    parser.add_argument("--features", default=None, help="path to features meta .pkl (default: models/ufc_stacking_ensemble_meta.pkl)")
    parser.add_argument("--data-dir", default=None, help="base data directory (default: data/)")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-q", "--quiet", action="store_true", help="quiet output")

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # --- data ---
    p_data = sub.add_parser("data", help="data pipeline (scraping + features)")
    data_sub = p_data.add_subparsers(dest="data_command", metavar="<data-command>")
    data_sub.required = True

    p_bi = data_sub.add_parser("build-index", help="build events index from ufcstats.com")
    p_bi.set_defaults(func=_handle_data_build_index)

    p_scrape = data_sub.add_parser("scrape", help="scrape fights and fighters")
    p_scrape.add_argument("--limit", type=int, default=None, help="limit number of events to scrape (for testing)")
    p_scrape.add_argument("--force", action="store_true", help="re-scrape even if already marked scrapped")
    p_scrape.set_defaults(func=_handle_data_scrape)

    p_feat = data_sub.add_parser("features", help="build dataset.csv from fights")
    p_feat.add_argument("--seed", type=int, default=42, help="random seed for A/B assignment (default: 42)")
    p_feat.set_defaults(func=_handle_data_features)

    # --- train ---
    p_train = sub.add_parser("train", help="train stacking ensemble")
    p_train.add_argument("--quick", action="store_true", help="quick run with fewer trials (for testing)")
    p_train.add_argument("--trials-lgb", type=int, default=None, help="override LGBM trials (default 50, quick 5)")
    p_train.add_argument("--trials-xgb", type=int, default=None, help="override XGB trials (default 20, quick 3)")
    p_train.add_argument("--no-plot", action="store_true", help="skip feature importance plot")
    p_train.set_defaults(func=_handle_train)

    # --- predict ---
    p_pred = sub.add_parser("predict", help="predict fights")
    pred_sub = p_pred.add_subparsers(dest="predict_command", metavar="<predict-command>")
    pred_sub.required = True

    p_fight = pred_sub.add_parser("fight", help="predict a single fight (interactive if no args)")
    p_fight.add_argument("--fighter-a", default=None, help="first fighter name")
    p_fight.add_argument("--fighter-b", default=None, help="second fighter name")
    p_fight.add_argument("--weight-class", default=None, help="weight class (e.g. Lightweight)")
    p_fight.add_argument("--rounds", type=int, choices=[3, 5], default=None, help="scheduled rounds (3 or 5)")
    p_fight.add_argument("--json", action="store_true", help="output JSON instead of table")
    p_fight.add_argument("--explain", action="store_true", help="include SHAP explanation (if available)")
    p_fight.set_defaults(func=_handle_predict_fight)

    p_event = pred_sub.add_parser("event", help="predict all fights for an event by name")
    p_event.add_argument("--event", required=True, help="event name, e.g. 'UFC 328: Chimaev vs. Strickland'")
    p_event.add_argument("--exact", action="store_true", help="require exact name match (default: substring)")
    p_event.add_argument("--json", action="store_true", help="output JSON (default: table)")
    p_event.set_defaults(func=_handle_predict_event)

    p_url = pred_sub.add_parser("url", help="scrape ufcstats event URL and predict")
    p_url.add_argument("url", nargs="?", help="ufcstats event URL")
    p_url.add_argument("--url", dest="url_flag", default=None, help="alternative --url flag")
    p_url.add_argument("--json", action="store_true", help="output JSON")
    p_url.set_defaults(func=_handle_predict_url)

    # --- backtest ---
    p_bt = sub.add_parser("backtest", help="backtest without lookahead")
    p_bt.add_argument("--start", required=True, help="start date YYYY-MM-DD inclusive")
    p_bt.add_argument("--end", default=None, help="end date YYYY-MM-DD inclusive (default: today)")
    p_bt.add_argument("--json", action="store_true", help="output JSON")
    p_bt.set_defaults(func=_handle_backtest)

    # --- history ---
    p_hist = sub.add_parser("history", help="show fighter Elo history")
    p_hist.add_argument("fighter", nargs="?", default=None, help="fighter name (if omitted, interactive)")
    p_hist.add_argument("--json", action="store_true", help="output JSON")
    p_hist.set_defaults(func=_handle_history)

    # --- pipeline ---
    p_pipe = sub.add_parser("pipeline", help="run data pipeline: build-index -> scrape -> features")
    p_pipe.add_argument("--skip-train", action="store_true", help="do not run train after pipeline")
    p_pipe.add_argument("--seed", type=int, default=42, help="seed for features step")
    p_pipe.set_defaults(func=_handle_pipeline)

    # --- info ---
    p_info = sub.add_parser("info", help="show data/model status")
    p_info.add_argument("--json", action="store_true", help="output JSON")
    p_info.set_defaults(func=_handle_info)

    return parser


# --- handlers (lazy imports to keep startup fast) ---

def _handle_data_build_index(args):
    from cli.cmd_data import handle_build_index
    return handle_build_index(args)


def _handle_data_scrape(args):
    from cli.cmd_data import handle_scrape
    return handle_scrape(args)


def _handle_data_features(args):
    from cli.cmd_data import handle_features
    return handle_features(args)


def _handle_train(args):
    from cli.cmd_train import handle_train
    return handle_train(args)


def _handle_predict_fight(args):
    from cli.cmd_predict import handle_fight
    return handle_fight(args)


def _handle_predict_event(args):
    from cli.cmd_predict import handle_event
    return handle_event(args)


def _handle_predict_url(args):
    from cli.cmd_predict import handle_url
    return handle_url(args)


def _handle_backtest(args):
    from cli.cmd_backtest import handle_backtest
    return handle_backtest(args)


def _handle_history(args):
    from cli.cmd_history import handle_history
    return handle_history(args)


def _handle_pipeline(args):
    from cli.cmd_pipeline import handle_pipeline
    return handle_pipeline(args)


def _handle_info(args):
    from cli.cmd_info import handle_info
    return handle_info(args)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    # global --data-dir handling etc. is done in _common per command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except SystemExit:
        raise
    except Exception as e:
        if getattr(args, "verbose", False):
            raise
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
