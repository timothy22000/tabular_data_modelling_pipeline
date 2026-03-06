"""Comparison with GLM and GBM pipeline results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import DLConfig, log, compute_gini, _compute_metrics
from .data import DLFeatureBundle


def compare_with_existing_models(
    results: Dict[str, Any],
    bundle: DLFeatureBundle,
    config: DLConfig,
) -> Dict[str, Any]:
    """Build a side-by-side comparison table across all model families.

    Loads published metrics from:
      - ``data_to_be_cleaned/net/glm_results/model_summary.json``
      - ``data_to_be_cleaned/net/gbm_results/model_summary.json``

    Merges these with the current DL pipeline results to produce a unified
    comparison table.  Also computes the "double lift" of the best DL model
    over the GLM baseline.

    Saves ``full_model_comparison.json`` to ``config.output_dir``.

    Args:
        results: Mapping of architecture name to training result dict.
        bundle: DLFeatureBundle (used for GLM baseline gini from bundle).
        config: DL pipeline configuration.

    Returns:
        Dictionary with keys:
            - "comparison_table": List of per-model metric dicts.
            - "best_model": Name of the model with highest test Gini.
            - "double_lift": best_model_gini / glm_gini.
            - "glm_gini": Baseline GLM test Gini.
    """
    log.info("=" * 72)
    log.info("SECTION 11: Comparison with GLM & GBM")
    log.info("=" * 72)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison: Dict[str, Any] = {
        "comparison_table": [],
        "best_model": None,
        "double_lift": None,
        "glm_gini": None,
    }

    all_rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Load existing GLM results
    # ------------------------------------------------------------------
    glm_summary_path = Path("data_to_be_cleaned/net/glm_results/model_summary.json")
    glm_gini_test: Optional[float] = None

    if glm_summary_path.exists():
        try:
            with open(glm_summary_path, "r") as fh:
                glm_data = json.load(fh)
            # Look for a top-level "gini_test" or nested in metrics_test
            glm_gini_test = (
                glm_data.get("gini_test")
                or glm_data.get("metrics_test", {}).get("gini")
                or glm_data.get("test_gini")
            )
            glm_gini_train = (
                glm_data.get("gini_train")
                or glm_data.get("metrics_train", {}).get("gini")
            )
            glm_mae = glm_data.get("mae") or glm_data.get("metrics_test", {}).get("mae")
            glm_ae = glm_data.get("ae_ratio") or glm_data.get("metrics_test", {}).get("ae_ratio")

            if glm_gini_test is not None:
                all_rows.append({
                    "model": "GLM",
                    "family": "GLM",
                    "gini_train": round(float(glm_gini_train or 0), 4),
                    "gini_test": round(float(glm_gini_test), 4),
                    "mae": round(float(glm_mae or 0), 2),
                    "ae_ratio": round(float(glm_ae or 1.0), 4),
                    "source": "glm_results",
                })
                comparison["glm_gini"] = float(glm_gini_test)
                log.info("  GLM loaded — Test Gini=%.4f", glm_gini_test)
            else:
                log.warning("  GLM summary found but no gini_test key")
        except Exception as exc:
            log.warning("  Could not load GLM summary: %s", exc)
    else:
        # Compute from bundle GLM predictions
        try:
            glm_gini_test = float(compute_gini(bundle.y_test, bundle.glm_test_preds))
            glm_gini_train = float(compute_gini(bundle.y_train, bundle.glm_train_preds))
            m_glm = _compute_metrics(bundle.y_test, bundle.glm_test_preds, "test")
            all_rows.append({
                "model": "GLM",
                "family": "GLM",
                "gini_train": round(glm_gini_train, 4),
                "gini_test": round(glm_gini_test, 4),
                "mae": round(m_glm.get("mae", 0), 2),
                "ae_ratio": round(m_glm.get("ae_ratio", 1.0), 4),
                "source": "bundle",
            })
            comparison["glm_gini"] = glm_gini_test
            log.info("  GLM computed from bundle — Test Gini=%.4f", glm_gini_test)
        except Exception as exc:
            log.warning("  Could not compute GLM Gini from bundle: %s", exc)

    # ------------------------------------------------------------------
    # Load existing GBM results
    # ------------------------------------------------------------------
    gbm_summary_path = Path("data_to_be_cleaned/net/gbm_results/model_summary.json")
    if gbm_summary_path.exists():
        try:
            with open(gbm_summary_path, "r") as fh:
                gbm_data = json.load(fh)
            # GBM summary may be a dict of model_name -> metrics
            gbm_model_keys = [
                k for k in gbm_data.keys()
                if isinstance(gbm_data[k], dict) and "gini_test" in gbm_data[k]
            ]
            if not gbm_model_keys:
                # Try flat structure
                if "gini_test" in gbm_data:
                    gbm_model_keys = ["lightgbm"]
                    gbm_data = {"lightgbm": gbm_data}

            for gbm_key in gbm_model_keys:
                gd = gbm_data[gbm_key]
                all_rows.append({
                    "model": gbm_key,
                    "family": "GBM",
                    "gini_train": round(float(gd.get("gini_train", 0)), 4),
                    "gini_test": round(float(gd.get("gini_test", 0)), 4),
                    "mae": round(float(gd.get("mae", 0)), 2),
                    "ae_ratio": round(float(gd.get("ae_ratio", 1.0)), 4),
                    "source": "gbm_results",
                })
                log.info(
                    "  GBM (%s) loaded — Test Gini=%.4f",
                    gbm_key, gd.get("gini_test", 0),
                )
        except Exception as exc:
            log.warning("  Could not load GBM summary: %s", exc)

    # ------------------------------------------------------------------
    # Add current DL pipeline results
    # ------------------------------------------------------------------
    family_map = {
        "catboost": "GBM",
        "xgboost": "GBM",
        "cann": "DL",
        "cann_gbm": "DL",
        "ft_transformer": "DL",
        "tabm": "DL",
        "localglmnet": "DL",
        "drn": "DL",
        "stacked_ensemble": "Ensemble",
    }

    for model_name, res in results.items():
        if "error" in res or "train_preds" not in res:
            continue
        m_tr = res.get("metrics_train", {})
        m_te = res.get("metrics_test", {})
        all_rows.append({
            "model": model_name,
            "family": family_map.get(model_name, "DL"),
            "gini_train": round(float(m_tr.get("gini", 0)), 4),
            "gini_test": round(float(m_te.get("gini", 0)), 4),
            "mae": round(float(m_te.get("mae", 0)), 2),
            "ae_ratio": round(float(m_te.get("ae_ratio", 1.0)), 4),
            "source": "dl_pipeline",
        })

    # ------------------------------------------------------------------
    # Sort and annotate
    # ------------------------------------------------------------------
    all_rows.sort(key=lambda r: r["gini_test"], reverse=True)
    comparison["comparison_table"] = all_rows

    if all_rows:
        best_row = all_rows[0]
        comparison["best_model"] = best_row["model"]
        if comparison["glm_gini"] and comparison["glm_gini"] > 0:
            comparison["double_lift"] = round(
                best_row["gini_test"] / comparison["glm_gini"], 4
            )
        log.info(
            "  Best model: %s (Test Gini=%.4f)",
            best_row["model"], best_row["gini_test"],
        )
        if comparison["double_lift"] is not None:
            log.info("  Double lift over GLM: %.4f x", comparison["double_lift"])

    # Log comparison table
    log.info("")
    log.info("  FULL MODEL COMPARISON (ranked by test Gini)")
    log.info("  " + "-" * 72)
    log.info(
        "  %-22s  %-10s  %8s  %8s  %8s  %8s",
        "Model", "Family", "Gini(tr)", "Gini(te)", "MAE", "A/E",
    )
    log.info("  " + "-" * 72)
    for row in all_rows:
        log.info(
            "  %-22s  %-10s  %8.4f  %8.4f  %8.0f  %8.4f",
            row["model"], row["family"],
            row["gini_train"], row["gini_test"],
            row["mae"], row["ae_ratio"],
        )
    log.info("  " + "-" * 72)

    # Save
    full_path = output_dir / "full_model_comparison.json"
    with open(full_path, "w") as fh:
        json.dump(comparison, fh, indent=2, default=str)
    log.info("  Full comparison saved to %s", full_path)

    return comparison
