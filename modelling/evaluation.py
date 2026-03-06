"""Evaluation and diagnostics for all trained models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .config import (
    DLConfig,
    HAS_TORCH, HAS_CATBOOST, HAS_XGBOOST,
    CatBoostRegressor, xgb,
    log, _compute_metrics, compute_gamma_deviance, compute_decile_analysis,
)
from .data import DLFeatureBundle


def evaluate_all_models(
    results: Dict[str, Any],
    bundle: DLFeatureBundle,
    config: DLConfig,
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics for all trained models.

    Iterates over every entry in ``results`` (including stacked_ensemble if
    present), recomputes metrics via ``_compute_metrics`` and
    ``compute_decile_analysis``, derives parameter counts where possible, and
    assembles a comparison DataFrame.

    Output files saved to ``config.output_dir``:
        - ``evaluation_summary.csv``: Per-model metric table.
        - ``model_comparison.json``: Full metric dict per model.

    Args:
        results: Mapping of architecture name to training result dict.
            Each entry must have "train_preds" and "test_preds" arrays.
        bundle: DLFeatureBundle providing y_train, y_test.
        config: DL pipeline configuration (output_dir).

    Returns:
        Dictionary mapping model name to comprehensive metrics dict, plus
        a "comparison_df" key containing the summary DataFrame.
    """
    log.info("=" * 72)
    log.info("SECTION 8: Evaluation & Diagnostics")
    log.info("=" * 72)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_records: List[Dict[str, Any]] = []
    eval_full: Dict[str, Any] = {}

    for model_name, res in results.items():
        if "error" in res:
            log.warning("  Skipping %s — training error: %s", model_name, res["error"])
            continue
        if "train_preds" not in res or "test_preds" not in res:
            log.warning("  Skipping %s — missing predictions", model_name)
            continue

        train_preds = res["train_preds"]
        test_preds = res["test_preds"]

        metrics_train = _compute_metrics(bundle.y_train, train_preds, "train")
        metrics_test = _compute_metrics(bundle.y_test, test_preds, "test")

        # Decile analysis for test set
        try:
            decile_df = compute_decile_analysis(bundle.y_test, test_preds)
            ae_by_decile = decile_df["ae_ratio"].tolist() if "ae_ratio" in decile_df.columns else []
        except Exception:
            ae_by_decile = []

        # Parameter count
        n_params = _count_model_params(res, model_name)

        training_time = res.get("training_time", float("nan"))

        record = {
            "model": model_name,
            "gini_train": round(metrics_train.get("gini", float("nan")), 4),
            "gini_test": round(metrics_test.get("gini", float("nan")), 4),
            "mae": round(metrics_test.get("mae", float("nan")), 2),
            "rmse": round(metrics_test.get("rmse", float("nan")), 2),
            "cv_rmse": round(
                metrics_test.get("rmse", float("nan"))
                / max(float(np.mean(bundle.y_test)), 1.0),
                4,
            ),
            "ae_ratio": round(metrics_test.get("ae_ratio", float("nan")), 4),
            "gamma_deviance": round(
                compute_gamma_deviance(bundle.y_test, test_preds), 6
            ),
            "n_params": n_params,
            "training_time": round(training_time, 1),
        }
        eval_records.append(record)

        eval_full[model_name] = {
            "metrics_train": metrics_train,
            "metrics_test": metrics_test,
            "ae_by_decile": ae_by_decile,
            "n_params": n_params,
            "training_time": training_time,
            "best_params": {
                k: (
                    int(v) if isinstance(v, (np.integer,)) else
                    float(v) if isinstance(v, (np.floating,)) else v
                )
                for k, v in res.get("best_params", {}).items()
            },
        }

        log.info(
            "  %-22s  Gini(tr)=%.4f  Gini(te)=%.4f  MAE=%.0f  A/E=%.4f  params=%s",
            model_name,
            metrics_train.get("gini", float("nan")),
            metrics_test.get("gini", float("nan")),
            metrics_test.get("mae", float("nan")),
            metrics_test.get("ae_ratio", float("nan")),
            f"{n_params:,}" if isinstance(n_params, int) else str(n_params),
        )

    if not eval_records:
        log.warning("  No models to evaluate.")
        return {"comparison_df": pd.DataFrame()}

    comparison_df = pd.DataFrame(eval_records).sort_values(
        "gini_test", ascending=False
    )

    # Save CSV — merge with any existing results, dedup on model name (keep latest)
    csv_path = output_dir / "evaluation_summary.csv"
    summary_df = comparison_df
    if csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path)
            combined = pd.concat([existing_df, summary_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["model"], keep="last")
            summary_df = combined.sort_values("gini_test", ascending=False)
        except Exception:
            pass
    summary_df.to_csv(csv_path, index=False)
    log.info("  Evaluation summary saved to %s", csv_path)

    # Save JSON — merge with existing architecture-keyed dict
    json_path = output_dir / "model_comparison.json"
    existing_comparison: Dict[str, Any] = {}
    if json_path.exists():
        try:
            with open(json_path) as fh:
                existing_comparison = json.load(fh)
        except Exception:
            pass
    merged_comparison = {**existing_comparison, **eval_full}
    with open(json_path, "w") as fh:
        json.dump(merged_comparison, fh, indent=2, default=str)
    log.info("  Model comparison saved to %s", json_path)

    # Log ranked table
    log.info("")
    log.info("  EVALUATION SUMMARY (ranked by test Gini)")
    log.info("  " + "-" * 90)
    log.info(
        "  %-22s  %8s  %8s  %8s  %8s  %10s  %10s",
        "Model", "Gini(tr)", "Gini(te)", "MAE", "A/E", "Deviance", "Time(s)",
    )
    log.info("  " + "-" * 90)
    for _, row in comparison_df.iterrows():
        log.info(
            "  %-22s  %8.4f  %8.4f  %8.0f  %8.4f  %10.6f  %10.1f",
            row["model"],
            row["gini_train"],
            row["gini_test"],
            row["mae"],
            row["ae_ratio"],
            row["gamma_deviance"],
            row["training_time"],
        )
    log.info("  " + "-" * 90)

    eval_full["comparison_df"] = comparison_df
    return eval_full


def _count_model_params(res: Dict[str, Any], model_name: str) -> Any:
    """Estimate parameter count for a trained model result.

    Tries multiple strategies depending on the model type:
      - PyTorch nn.Module: sum of .numel() over parameters.
      - CatBoost: .tree_count_ * estimated_params_per_tree.
      - XGBoost: best_iteration from model.
      - Stacked ensemble: returns number of base learners.

    Args:
        res: Result dict for a single model.
        model_name: Architecture name for type-based dispatch.

    Returns:
        Integer parameter count, or "N/A" if not computable.
    """
    # DL ensemble: sum across members
    if "ensemble_results" in res:
        try:
            total = 0
            for tr in res["ensemble_results"]:
                if tr.model is not None and HAS_TORCH:
                    total += sum(p.numel() for p in tr.model.parameters())
            return total if total > 0 else "N/A"
        except Exception:
            pass

    # Direct model object
    model = res.get("model")
    if model is None:
        # Stacked ensemble has base_weights
        if "base_weights" in res:
            return len(res["base_weights"])
        return "N/A"

    # CatBoost
    if HAS_CATBOOST and isinstance(model, CatBoostRegressor):
        try:
            return int(model.tree_count_)
        except Exception:
            pass

    # XGBoost
    if HAS_XGBOOST and isinstance(model, xgb.XGBRegressor):
        try:
            return int(model.best_iteration)
        except Exception:
            pass

    # PyTorch
    if HAS_TORCH:
        try:
            import torch.nn as _nn
            if isinstance(model, _nn.Module):
                return sum(p.numel() for p in model.parameters())
        except Exception:
            pass

    return "N/A"
