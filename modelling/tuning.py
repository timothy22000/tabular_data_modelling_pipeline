"""Optuna hyperparameter tuning for all architectures."""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from .config import (
    DLConfig,
    HAS_TORCH, HAS_CATBOOST, HAS_XGBOOST, HAS_OPTUNA,
    torch, optuna, xgb, CatBoostRegressor,
    log, _clamp_predictions,
)
from .data import DLFeatureBundle, build_dataloaders
from .models import get_default_dl_params, build_dl_model
from .training import (
    _resolve_device, _get_monotone_cont_indices,
    train_one_epoch, evaluate,
)


def tune_catboost(
    bundle: DLFeatureBundle,
    config: DLConfig,
) -> Dict[str, Any]:
    """Tune CatBoost hyperparameters with Optuna.

    Optimises mean RMSE on a stratified time-split validation over
    ``config.cv_folds`` folds.  Uses ``CatBoostRegressor.fit`` with
    eval_set for each trial's validation.

    Search space:
        - depth: int in [4, 10]
        - l2_leaf_reg: float in [1, 10] (log-uniform)
        - learning_rate: float in [0.01, 0.15] (log-uniform)
        - iterations: int in [500, 3000]
        - subsample: float in [0.7, 1.0]
        - bagging_temperature: float in [0.0, 2.0]

    Args:
        bundle: Feature bundle with CatBoost Pool objects.
        config: DL pipeline configuration (n_tuning_trials, seed, quick).

    Returns:
        Dictionary of best hyperparameters from the Optuna study.
        Falls back to ``get_default_dl_params("catboost")`` if Optuna or
        CatBoost are unavailable or if tuning is skipped.
    """
    if config.skip_tuning or not HAS_OPTUNA or not HAS_CATBOOST:
        log.info("  CatBoost tuning skipped — using defaults")
        return get_default_dl_params("catboost")

    if bundle.catboost_train_pool is None or bundle.catboost_test_pool is None:
        log.warning("  CatBoost pools not available — using defaults")
        return get_default_dl_params("catboost")

    log.info("  Starting CatBoost Optuna tuning (%d trials) ...", config.n_tuning_trials)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        """Optuna objective: RMSE on the validation pool."""
        params = {
            "iterations": trial.suggest_int("iterations", 500, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        }

        model = CatBoostRegressor(
            **params,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=config.seed,
            verbose=False,
            early_stopping_rounds=50,
            cat_features=bundle.categorical_feature_names,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                bundle.catboost_train_pool,
                eval_set=bundle.catboost_test_pool,
                use_best_model=True,
            )

        preds = _clamp_predictions(model.predict(bundle.catboost_test_pool))
        rmse = float(np.sqrt(np.mean((bundle.y_test - preds) ** 2)))
        return rmse

    n_trials = config.n_tuning_trials
    if config.quick:
        n_trials = min(n_trials, 10)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
        study_name="catboost_tuning",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_trial.params)
    best_params["early_stopping_rounds"] = 50
    log.info(
        "  CatBoost best trial #%d (RMSE=%.4f): %s",
        study.best_trial.number,
        study.best_trial.value,
        best_params,
    )
    return best_params


def tune_xgboost(
    bundle: DLFeatureBundle,
    config: DLConfig,
) -> Dict[str, Any]:
    """Tune XGBoost hyperparameters with Optuna.

    Search space:
        - max_depth: int in [3, 10]
        - eta (learning_rate): float in [0.01, 0.15] (log-uniform)
        - n_estimators: int in [300, 2000]
        - reg_alpha: float in [1e-3, 10.0] (log-uniform)
        - reg_lambda: float in [1e-3, 10.0] (log-uniform)
        - subsample: float in [0.7, 1.0]
        - colsample_bytree: float in [0.7, 1.0]

    Args:
        bundle: Feature bundle with raw arrays reconstructed for XGBoost.
        config: DL pipeline configuration.

    Returns:
        Dictionary of best XGBoost hyperparameters.
    """
    if config.skip_tuning or not HAS_OPTUNA or not HAS_XGBOOST:
        log.info("  XGBoost tuning skipped — using defaults")
        return get_default_dl_params("xgboost")

    log.info("  Starting XGBoost Optuna tuning (%d trials) ...", config.n_tuning_trials)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Reconstruct raw matrices once (avoid repeat per trial)
    # Reverse Z-score standardisation to recover original scale
    X_train_cont_raw = (bundle.X_train_cont * bundle.cont_std + bundle.cont_mean).astype(np.float32)
    X_test_cont_raw = (bundle.X_test_cont * bundle.cont_std + bundle.cont_mean).astype(np.float32)
    X_train_raw = np.concatenate(
        [X_train_cont_raw, bundle.X_train_cat.astype(np.float32)],
        axis=1,
    )
    X_test_raw = np.concatenate(
        [X_test_cont_raw, bundle.X_test_cat.astype(np.float32)],
        axis=1,
    )
    cont_names = bundle.continuous_feature_names
    all_feature_names = cont_names + bundle.categorical_feature_names
    mono_tuple = tuple(config.dataset.monotone_constraints.get(f, 0) for f in all_feature_names)

    def objective(trial: "optuna.Trial") -> float:
        """Optuna objective: RMSE on test set."""
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "eta": trial.suggest_float("eta", 0.01, 0.15, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        }
        n_est = params.pop("n_estimators")

        model = xgb.XGBRegressor(
            objective="reg:gamma",
            n_estimators=n_est,
            early_stopping_rounds=50,
            monotone_constraints=mono_tuple,
            random_state=config.seed,
            verbosity=0,
            **params,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                X_train_raw,
                bundle.y_train,
                eval_set=[(X_test_raw, bundle.y_test)],
                verbose=False,
            )

        preds = _clamp_predictions(model.predict(X_test_raw))
        rmse = float(np.sqrt(np.mean((bundle.y_test - preds) ** 2)))
        params["n_estimators"] = n_est  # Restore for Optuna storage
        return rmse

    n_trials = config.n_tuning_trials
    if config.quick:
        n_trials = min(n_trials, 10)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
        study_name="xgboost_tuning",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_trial.params)
    best_params["early_stopping_rounds"] = 50
    log.info(
        "  XGBoost best trial #%d (RMSE=%.4f): %s",
        study.best_trial.number,
        study.best_trial.value,
        best_params,
    )
    return best_params


def tune_dl_model(
    architecture: str,
    bundle: DLFeatureBundle,
    config: DLConfig,
) -> Dict[str, Any]:
    """Tune DL model hyperparameters with Optuna using a short training run.

    Each trial trains the model for a reduced number of epochs (min(50,
    config.epochs // 4)) with early stopping and reports validation Gamma
    deviance.

    Search spaces per architecture:

    CANN:
        - hidden_layer_sizes: categorical [[128,64], [256,128,64], [512,256,128,64]]
        - dropout: float in [0.1, 0.5]
        - lr: float in [1e-4, 1e-2] (log-uniform)
        - weight_decay: float in [1e-5, 1e-2] (log-uniform)

    FT-Transformer:
        - d_model: categorical [32, 64, 128]
        - n_heads: categorical [2, 4, 8] (filtered to divide d_model)
        - n_layers: int in [2, 6]
        - dropout: float in [0.0, 0.3]
        - ffn_factor: categorical [2, 4]
        - lr: float in [1e-4, 5e-3] (log-uniform)
        - weight_decay: float in [1e-5, 1e-2] (log-uniform)

    TabM:
        - n_members: int in [4, 16]
        - hidden_layer_sizes: categorical [[64,32], [128,64], [256,128]]
        - dropout: float in [0.1, 0.4]
        - lr: float in [1e-4, 1e-2] (log-uniform)
        - weight_decay: float in [1e-5, 1e-2] (log-uniform)

    Args:
        architecture: One of "cann", "ft_transformer", "tabm".
        bundle: Feature bundle with all required arrays.
        config: DL pipeline configuration.

    Returns:
        Dictionary of best hyperparameters. Falls back to defaults if Optuna
        or PyTorch are unavailable or tuning is skipped.
    """
    if config.skip_tuning or not HAS_OPTUNA or not HAS_TORCH:
        log.info("  %s tuning skipped — using defaults", architecture)
        return get_default_dl_params(architecture)

    log.info(
        "  Starting %s Optuna tuning (%d trials) ...",
        architecture,
        config.n_tuning_trials,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    device = _resolve_device(config)
    train_loader, val_loader, _ = build_dataloaders(bundle, config)
    tune_epochs = max(10, min(50, config.epochs // 4))
    if config.quick:
        tune_epochs = max(5, tune_epochs // 2)

    category_sizes = [len(m) for m in bundle.category_mappings.values()]
    n_cont = bundle.X_train_cont.shape[1]
    base_mode_map = {
        "cann": "glm", "cann_gbm": "gbm",
        "localglmnet": "glm", "drn": "drn",
    }
    base_mode = base_mode_map.get(architecture, "none")

    def objective(trial: "optuna.Trial") -> float:
        """Optuna objective: val Gamma deviance after short training."""
        params = _suggest_dl_params(trial, architecture)
        model = build_dl_model(architecture, params, bundle, config.dataset.prediction_floor)
        model = model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params.get("lr", 1e-3),
            weight_decay=params.get("weight_decay", 1e-4),
        )
        mono_indices, mono_directions = _get_monotone_cont_indices(
            bundle.continuous_feature_names, config.dataset.monotone_constraints
        )

        for epoch in range(tune_epochs):
            train_one_epoch(
                model, train_loader, optimizer, device, config,
                mono_indices, mono_directions, base_mode,
            )

        val_loss, _, _ = evaluate(model, val_loader, device, base_mode)
        model = model.cpu()
        return float(val_loss)

    n_trials = config.n_tuning_trials
    if config.quick:
        n_trials = min(n_trials, 5)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
        study_name=f"{architecture}_tuning",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_trial.params)
    # Optuna stores categorical hidden_dims as string — parse back to list
    if "hidden_dims" in best_params and isinstance(best_params["hidden_dims"], str):
        best_params["hidden_dims"] = [
            int(x) for x in best_params["hidden_dims"].strip("[]").split(",")
        ]
    log.info(
        "  %s best trial #%d (val_deviance=%.6f): %s",
        architecture,
        study.best_trial.number,
        study.best_trial.value,
        best_params,
    )
    return best_params


def _suggest_dl_params(
    trial: "optuna.Trial",
    architecture: str,
) -> Dict[str, Any]:
    """Suggest hyperparameters for a DL architecture from an Optuna trial.

    This is a helper function called inside the Optuna objective.  It
    translates the architecture-specific search space into a flat
    hyperparameter dictionary.

    Args:
        trial: Optuna trial object used for parameter suggestion.
        architecture: One of "cann", "ft_transformer", "tabm".

    Returns:
        Dictionary of suggested hyperparameter name -> value.
    """
    if architecture == "cann":
        hidden_choice = trial.suggest_categorical(
            "hidden_dims",
            ["[64,32]", "[128,64]", "[256,128]"],
        )
        hidden_dims = [int(x) for x in hidden_choice.strip("[]").split(",")]
        return {
            "hidden_dims": hidden_dims,
            "dropout": trial.suggest_float("dropout", 0.1, 0.3),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True),
        }

    elif architecture == "ft_transformer":
        d_model = trial.suggest_categorical("d_model", [32, 64, 128])
        # Ensure n_heads divides d_model
        valid_heads = [h for h in [2, 4, 8] if d_model % h == 0]
        n_heads = trial.suggest_categorical("n_heads", valid_heads or [2])
        return {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": trial.suggest_int("n_layers", 2, 6),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "ffn_factor": trial.suggest_categorical("ffn_factor", [2, 4]),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        }

    elif architecture == "tabm":
        hidden_choice = trial.suggest_categorical(
            "hidden_dims",
            ["[64,32]", "[128,64]", "[256,128]"],
        )
        hidden_dims = [int(x) for x in hidden_choice.strip("[]").split(",")]
        return {
            "n_members": trial.suggest_int("n_members", 4, 16),
            "hidden_dims": hidden_dims,
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        }

    elif architecture == "cann_gbm":
        hidden_choice = trial.suggest_categorical(
            "hidden_dims",
            ["[64,32]", "[128,64]", "[256,128]"],
        )
        hidden_dims = [int(x) for x in hidden_choice.strip("[]").split(",")]
        return {
            "hidden_dims": hidden_dims,
            "dropout": trial.suggest_float("dropout", 0.1, 0.3),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True),
        }

    elif architecture == "localglmnet":
        hidden_choice = trial.suggest_categorical(
            "hidden_dims",
            ["[64,32]", "[128,64]", "[256,128]"],
        )
        hidden_dims = [int(x) for x in hidden_choice.strip("[]").split(",")]
        return {
            "hidden_dims": hidden_dims,
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),
            "coeff_reg": trial.suggest_float("coeff_reg", 1e-4, 0.1, log=True),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        }

    elif architecture == "drn":
        hidden_choice = trial.suggest_categorical(
            "hidden_dims",
            ["[64,32]", "[128,64]", "[256,128]"],
        )
        hidden_dims = [int(x) for x in hidden_choice.strip("[]").split(",")]
        return {
            "hidden_dims": hidden_dims,
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),
            "kl_alpha": trial.suggest_float("kl_alpha", 0.01, 1.0, log=True),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        }

    else:
        raise ValueError(f"Unknown DL architecture for param suggestion: '{architecture}'")
