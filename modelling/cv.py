"""Cross-validation for CatBoost."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .config import (
    DLConfig,
    HAS_CATBOOST, CatBoostRegressor, Pool,
    log, MONOTONE_CONSTRAINTS, _clamp_predictions, _compute_metrics, compute_gini,
)
from .data import DLFeatureBundle
from .models import get_default_dl_params


def run_cross_validation(
    bundle: DLFeatureBundle,
    config: DLConfig,
) -> Dict[str, Any]:
    """Run k-fold cross-validation for CatBoost.

    DL models are excluded for speed; ensemble seed variance already provides
    stability information for those architectures.  For each fold, CatBoost
    is trained with the best available parameters (loaded from
    dl_metrics_summary.json if it exists, else defaults) and OOF predictions
    are collected.

    Args:
        bundle: DLFeatureBundle with feature matrices and metadata.
        config: DL pipeline configuration (cv_folds, seed, quick, output_dir).

    Returns:
        Dictionary with keys:
            - "catboost": Per-fold metrics dict and OOF predictions.
            - "cv_summary": Aggregated mean/std across folds.
    """
    from sklearn.model_selection import KFold

    log.info("=" * 72)
    log.info("SECTION 10: Cross-Validation")
    log.info("=" * 72)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cv_results: Dict[str, Any] = {}

    if not HAS_CATBOOST:
        log.warning("  CatBoost not installed — skipping cross-validation")
        return cv_results

    n_folds = 3 if config.quick else config.cv_folds
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=config.seed)

    # Load best CatBoost params from prior tuning if available
    best_params = get_default_dl_params("catboost")
    summary_path = output_dir / "dl_metrics_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r") as fh:
                summary = json.load(fh)
            if "catboost" in summary and "best_params" in summary["catboost"]:
                loaded = summary["catboost"]["best_params"]
                if loaded:
                    best_params.update(loaded)
                    log.info("  Loaded CatBoost params from dl_metrics_summary.json")
        except Exception as exc:
            log.warning("  Could not load CatBoost params: %s", exc)

    if config.quick:
        best_params["iterations"] = min(int(best_params.get("iterations", 2000)), 200)

    # Build monotone constraint dict
    all_feature_names = bundle.continuous_feature_names + bundle.categorical_feature_names
    mono_constraints: Dict[str, int] = {
        feat: direction
        for feat, direction in MONOTONE_CONSTRAINTS.items()
        if feat in all_feature_names
    }

    log.info("  Running %d-fold CV for CatBoost ...", n_folds)

    cont_names = bundle.continuous_feature_names
    cat_names = bundle.categorical_feature_names

    # Reconstruct raw DataFrame for CatBoost from de-standardised arrays + raw cats
    X_train_cont_raw = (bundle.X_train_cont * bundle.cont_std + bundle.cont_mean).astype(np.float32)
    X_raw_df = pd.DataFrame(X_train_cont_raw, columns=cont_names)
    for i, col in enumerate(cat_names):
        X_raw_df[col] = bundle.train_df[col].astype(str).fillna("UNKNOWN").values

    y_train = bundle.y_train
    cat_col_indices_cv = [
        list(X_raw_df.columns).index(c) for c in cat_names if c in X_raw_df.columns
    ]

    fold_records: List[Dict[str, Any]] = []
    oof_preds = np.zeros(len(y_train), dtype=np.float32)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_raw_df)):
        log.info(
            "  Fold %d/%d — train=%d, val=%d",
            fold_idx + 1, n_folds, len(train_idx), len(val_idx),
        )

        X_fold_train = X_raw_df.iloc[train_idx]
        X_fold_val = X_raw_df.iloc[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]

        try:
            train_pool_cv = Pool(
                data=X_fold_train,
                label=y_fold_train,
                cat_features=cat_col_indices_cv,
            )
            val_pool_cv = Pool(
                data=X_fold_val,
                label=y_fold_val,
                cat_features=cat_col_indices_cv,
            )

            fp = {k: v for k, v in best_params.items()}
            fold_model = CatBoostRegressor(
                iterations=int(fp.pop("iterations", 2000)),
                learning_rate=float(fp.pop("learning_rate", 0.05)),
                depth=int(fp.pop("depth", 6)),
                l2_leaf_reg=float(fp.pop("l2_leaf_reg", 3.0)),
                subsample=float(fp.pop("subsample", 0.85)),
                bagging_temperature=float(fp.pop("bagging_temperature", 1.0)),
                loss_function="RMSE",
                eval_metric="RMSE",
                monotone_constraints=mono_constraints,
                random_seed=config.seed + fold_idx,
                verbose=False,
                early_stopping_rounds=int(fp.pop("early_stopping_rounds", 50)),
                cat_features=cat_names,
            )
            fold_model.fit(
                train_pool_cv, eval_set=val_pool_cv, use_best_model=True
            )

            fold_train_preds = _clamp_predictions(fold_model.predict(train_pool_cv))
            fold_val_preds = _clamp_predictions(fold_model.predict(val_pool_cv))
            oof_preds[val_idx] = fold_val_preds

            m_tr = _compute_metrics(y_fold_train, fold_train_preds, "train")
            m_va = _compute_metrics(y_fold_val, fold_val_preds, "val")

            record = {
                "fold": fold_idx + 1,
                "gini_train": round(m_tr["gini"], 4),
                "gini_val": round(m_va["gini"], 4),
                "mae": round(m_va["mae"], 2),
                "ae_ratio": round(m_va["ae_ratio"], 4),
                "best_iteration": int(fold_model.best_iteration_),
            }
            fold_records.append(record)

            log.info(
                "    Fold %d — Gini(tr)=%.4f  Gini(val)=%.4f  MAE=%.0f  A/E=%.4f  iter=%d",
                fold_idx + 1,
                m_tr["gini"], m_va["gini"], m_va["mae"],
                m_va["ae_ratio"], fold_model.best_iteration_,
            )

        except Exception as exc:
            log.error("    Fold %d failed: %s", fold_idx + 1, exc)

    if fold_records:
        cv_df = pd.DataFrame(fold_records)
        oof_gini = float(compute_gini(y_train, oof_preds))
        cv_summary = {
            "gini_val_mean": float(cv_df["gini_val"].mean()),
            "gini_val_std": float(cv_df["gini_val"].std()),
            "gini_train_mean": float(cv_df["gini_train"].mean()),
            "mae_mean": float(cv_df["mae"].mean()),
            "ae_ratio_mean": float(cv_df["ae_ratio"].mean()),
            "oof_gini": oof_gini,
            "n_folds": n_folds,
        }

        log.info(
            "  CV Summary — OOF Gini=%.4f | Val Gini mean=%.4f (std=%.4f)",
            oof_gini, cv_summary["gini_val_mean"], cv_summary["gini_val_std"],
        )

        cv_path = output_dir / "cv_results.csv"
        cv_df.to_csv(cv_path, index=False)
        log.info("  CV results saved to %s", cv_path)

        cv_sum_path = output_dir / "cv_summary.json"
        with open(cv_sum_path, "w") as fh:
            json.dump(cv_summary, fh, indent=2)
        log.info("  CV summary saved to %s", cv_sum_path)

        cv_results["catboost"] = {
            "fold_records": fold_records,
            "oof_preds": oof_preds,
            "cv_summary": cv_summary,
        }
        cv_results["cv_summary"] = cv_summary
    else:
        log.warning("  No folds completed successfully.")

    return cv_results
