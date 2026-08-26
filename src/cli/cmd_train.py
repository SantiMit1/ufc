"""Train command."""

def handle_train(args):
    from cli._common import resolve_paths

    _, _, dataset_path, model_path, features_path = resolve_paths(args)

    n_lgb = args.trials_lgb
    n_xgb = args.trials_xgb
    quick = args.quick
    no_plot = args.no_plot

    # defaults
    if quick:
        n_lgb = n_lgb if n_lgb is not None else 5
        n_xgb = n_xgb if n_xgb is not None else 3
    else:
        n_lgb = n_lgb if n_lgb is not None else 50
        n_xgb = n_xgb if n_xgb is not None else 20

    print(f"Training with dataset={dataset_path} model={model_path} features={features_path}")
    print(f"Trials: LGBM={n_lgb} XGB={n_xgb} quick={quick} no_plot={no_plot}")

    # Import here (heavy deps)
    from train_model import main as train_main

    train_main(
        n_lgb_trials=n_lgb,
        n_xgb_trials=n_xgb,
        no_plot=no_plot,
        dataset_path=str(dataset_path),
        model_path=str(model_path),
        features_path=str(features_path),
    )
    return 0
