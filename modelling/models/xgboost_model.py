"""XGBoost training wrapper."""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

from ..config import (
    DLConfig, HAS_XGBOOST, xgb,
    np, log, time, _clamp_predictions, _compute_metrics,
)
from ..data import DLFeatureBundle


# Map DatasetConfig.family -> XGBoost objective string.
_XGB_OBJECTIVE_BY_FAMILY: Dict[str, str] = {
    "gaussian": "reg:squarederror",
    "gamma": "reg:gamma",
    "tweedie": "reg:tweedie",
    "poisson": "count:poisson",
}


def get_default_dl_params(architecture: str) -> Dict[str, Any]:
    """Import and delegate to models/__init__.py — avoids circular import."""
    from . import get_default_dl_params as _get
    return _get(architecture)


def train_xgboost(
    bundle: DLFeatureBundle,
    config: DLConfig,
    best_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train an XGBRegressor with a family-appropriate objective and monotone constraints.

    The objective is chosen from ``config.dataset.family``:

    - ``gaussian`` -> ``reg:squarederror``
    - ``gamma``    -> ``reg:gamma``
    - ``tweedie``  -> ``reg:tweedie``
    - ``poisson``  -> ``count:poisson``

    Unrecognised families fall back to ``reg:squarederror`` with a warning.

    Uses the RAW (non-standardised) feature matrices, label-encoded
    categoricals, and aligns the monotone_constraints tuple to the column
    order.  Early stopping is applied against the test set.

    Args:
        bundle: DLFeatureBundle with raw continuous and categorical arrays.
        config: DL pipeline configuration.
        best_params: Tuned hyperparameters from ``tune_xgboost`` (None =
            use ``get_default_dl_params("xgboost")``).

    Returns:
        Dictionary with keys: model, train_preds, test_preds, metrics_train,
        metrics_test, training_time.
    """
    if not HAS_XGBOOST:
        raise ImportError("XGBoost is required for train_xgboost().")

    log.info("-" * 60)
    log.info("3b: Training XGBoost")
    log.info("-" * 60)

    if best_params is None:
        best_params = get_default_dl_params("xgboost")

    # XGBoost uses label-encoded integers for categoricals (treat as numeric)
    # Combine raw continuous + label-encoded categoricals
    # NOTE: X_train_cont is standardised; XGBoost needs raw.
    # Reverse the Z-score standardisation to recover original scale.
    cont_names = bundle.continuous_feature_names
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

    all_feature_names = cont_names + bundle.categorical_feature_names
    mono_tuple = tuple(config.dataset.monotone_constraints.get(f, 0) for f in all_feature_names)

    params = {k: v for k, v in best_params.items()}
    n_estimators = params.pop("n_estimators", 1000)
    early_stopping_rounds = params.pop("early_stopping_rounds", 50)
    if config.quick:
        n_estimators = min(n_estimators, 300)
        log.info("  [quick] XGBoost n_estimators capped at %d", n_estimators)

    family = config.dataset.family
    objective = _XGB_OBJECTIVE_BY_FAMILY.get(family, "reg:squarederror")
    if family not in _XGB_OBJECTIVE_BY_FAMILY:
        log.warning(
            "XGBoost: family=%r not recognised; falling back to reg:squarederror.",
            family,
        )

    # For count:poisson, XGBoost's auto base_score computation can
    # produce inf and crash training. Set it explicitly to log(mean(y))
    # which is the canonical Poisson MLE intercept.
    extra_kwargs: Dict[str, Any] = {}
    if objective == "count:poisson":
        y_pos = bundle.y_train[bundle.y_train > 0]
        if len(y_pos):
            extra_kwargs["base_score"] = float(np.log(np.mean(y_pos)))

    model = xgb.XGBRegressor(
        objective=objective,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        monotone_constraints=mono_tuple,
        random_state=config.seed,
        verbosity=1,
        **extra_kwargs,
        **params,
    )

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            X_train_raw,
            bundle.y_train,
            eval_set=[(X_test_raw, bundle.y_test)],
            verbose=100,
        )
    elapsed = time.time() - t0

    log.info(
        "  XGBoost trained in %.1fs | best_iteration=%d",
        elapsed,
        model.best_iteration,
    )

    train_preds = _clamp_predictions(model.predict(X_train_raw))
    test_preds = _clamp_predictions(model.predict(X_test_raw))

    metrics_train = _compute_metrics(bundle.y_train, train_preds, "train")
    metrics_test = _compute_metrics(bundle.y_test, test_preds, "test")

    log.info(
        "  XGBoost — Train Gini: %.4f | Test Gini: %.4f | Test MAE: %.0f | A/E: %.4f",
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
