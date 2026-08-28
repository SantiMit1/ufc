"""Pipeline command."""
from cli.cmd_data import handle_build_index, handle_features, handle_scrape


def handle_pipeline(args):

    print("=" * 60)
    print("  PIPELINE: build-index -> scrape -> features")
    print("=" * 60)

    # Step 1: build-index
    print("\n[1/3] data build-index")
    rc = handle_build_index(args)
    if rc != 0:
        print(f"build-index failed with code {rc}", file=__import__("sys").stderr)
        return rc

    # Step 2: scrape
    print("\n[2/3] data scrape")
    # For pipeline we don't force limit; use same args but without limit/force
    # Ensure args has limit/force attrs (from data scrape parser they exist, but pipeline args doesn't have them)
    # So we create a simple namespace with needed fields
    class _ScrapeArgs:
        pass
    sargs = _ScrapeArgs()
    sargs.data_dir = getattr(args, "data_dir", None)
    sargs.model = getattr(args, "model", None)
    sargs.features = getattr(args, "features", None)
    sargs.verbose = getattr(args, "verbose", False)
    sargs.quiet = getattr(args, "quiet", False)
    sargs.limit = None
    sargs.force = False
    # copy over data_dir etc.
    rc = handle_scrape(sargs)
    if rc != 0:
        print(f"scrape failed with code {rc}", file=__import__("sys").stderr)
        return rc

    # Step 3: features
    print("\n[3/3] data features")
    class _FeatArgs:
        pass
    fargs = _FeatArgs()
    fargs.data_dir = getattr(args, "data_dir", None)
    fargs.model = getattr(args, "model", None)
    fargs.features = getattr(args, "features", None)
    fargs.verbose = getattr(args, "verbose", False)
    fargs.quiet = getattr(args, "quiet", False)
    fargs.seed = getattr(args, "seed", 42)
    rc = handle_features(fargs)
    if rc != 0:
        print(f"features failed with code {rc}", file=__import__("sys").stderr)
        return rc

    if not getattr(args, "skip_train", False):
        print("\n[4/4] train")
        print("  Note: training is slow (50 LGB + 20 XGB trials). Use --skip-train to skip, or run `ufc train --quick` separately.")
        print("  Skipping train (run `ufc train` manually). Use `ufc train` to train the model.")
    else:
        print("\nSkipping train (--skip-train)")

    print("\nPipeline completed.")
    return 0
