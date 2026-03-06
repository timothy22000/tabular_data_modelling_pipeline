"""CatBoost training wrapper."""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

from ..config import (
    DLConfig, HAS_CATBOOST, CatBoostRegressor,
    np, log, time, MONOTONE_CONSTRAINTS, _clamp_predictions, _compute_metrics,
)
from ..data import DLFeatureBundle


def get_default_dl_params(architecture: str) -> Dict[str, Any]:
    """Import and delegate to models/__init__.py — avoids circular import."""
    from . import get_default_dl_params as _get
    return _get(architecture)


def train_catboost(
    bundle: DLFeatureBundle,
    config: DLConfig,
    best_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train a CatBoostRegressor with monotonicity constraints.

    Uses the CatBoost Pool objects from the bundle for efficient training.
    Monotone constraints are applied to the five actuarially motivated
    features (NCD_CAPPED, MILEAGE_K, VEHICLE_VALUE, CLM_NUM_L5Y, CREDIT_SCORE).
    Eval set enables CatBoost's internal early stopping.

    Args:
        bundle: DLFeatureBundle containing Pool objects and feature metadata.
        config: DL pipeline configuration.
        best_params: Tuned hyperparameters (None = use defaults from
            ``get_default_dl_params("catboost")``).

    Returns:
        Dictionary containing:
            - "model": Fitted CatBoostRegressor.
            - "train_preds": Training predictions (clamped).
            - "test_preds": Test predictions (clamped).
            - "metrics_train": Metric dict for training split.
            - "metrics_test": Metric dict for test split.
            - "training_time": Wall-clock seconds.
    """
    if not HAS_CATBOOST:
        raise ImportError("CatBoost is required for train_catboost().")

    log.info("-" * 60)
    log.info("3a: Training CatBoost")
    log.info("-" * 60)

    if best_params is None:
        best_params = get_default_dl_params("catboost")

    # Build monotone constraint map for CatBoost
    # CatBoost expects a dict {feature_name: direction} or a list aligned to all features
    all_feature_names = bundle.continuous_feature_names + bundle.categorical_feature_names
    mono_constraints: Dict[str, int] = {}
    for feat, direction in MONOTONE_CONSTRAINTS.items():
        if feat in all_feature_names:
            mono_constraints[feat] = direction

    iterations = best_params.pop("iterations", config.catboost_iterations)
    learning_rate = best_params.pop("learning_rate", 0.05)
    depth = best_params.pop("depth", 6)
    l2_leaf_reg = best_params.pop("l2_leaf_reg", 3.0)
    subsample = best_params.pop("subsample", 0.85)
    bagging_temperature = best_params.pop("bagging_temperature", 1.0)
    early_stopping_rounds = best_params.pop("early_stopping_rounds", 50)

    if config.quick:
        iterations = min(iterations, 200)
        log.info("  [quick] CatBoost iterations capped at %d", iterations)

    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        subsample=subsample,
        bagging_temperature=bagging_temperature,
        loss_function="RMSE",
        eval_metric="RMSE",
        monotone_constraints=mono_constraints,
        random_seed=config.seed,
        verbose=100,
        early_stopping_rounds=early_stopping_rounds,
        cat_features=bundle.categorical_feature_names,
    )

    t0 = time.time()
    model.fit(
        bundle.catboost_train_pool,
        eval_set=bundle.catboost_test_pool,
        use_best_model=True,
    )
    elapsed = time.time() - t0

    log.info(
        "  CatBoost trained in %.1fs | best_iteration=%d",
        elapsed,
        model.best_iteration_,
    )

    train_preds = _clamp_predictions(model.predict(bundle.catboost_train_pool))
    test_preds = _clamp_predictions(model.predict(bundle.catboost_test_pool))

    metrics_train = _compute_metrics(bundle.y_train, train_preds, "train")
    metrics_test = _compute_metrics(bundle.y_test, test_preds, "test")

    log.info(
        "  CatBoost — Train Gini: %.4f | Test Gini: %.4f | Test MAE: %.0f | A/E: %.4f",
        metrics_train["gini"],
        metrics_test["gini"],
        metrics_test["mae"],
        metrics_test["ae_ratio"],
    )

    return {
        "model": model,
        "train_preds": train_preds,
        "test_preds": test_preds,
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "training_time": elapsed,
    }
