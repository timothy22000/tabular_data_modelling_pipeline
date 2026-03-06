"""Save all pipeline artefacts to disk."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .config import (
    DLConfig,
    HAS_TORCH, HAS_CATBOOST, HAS_XGBOOST,
    torch, CatBoostRegressor, xgb,
    log,
)
from .data import DLFeatureBundle
from .evaluation import _count_model_params


def save_all_outputs(
    results: Dict[str, Any],
    bundle: DLFeatureBundle,
    interp_results: Dict[str, Any],
    cv_results: Dict[str, Any],
    comparison: Dict[str, Any],
    config: DLConfig,
) -> None:
    """Save all pipeline artefacts to ``data_to_be_cleaned/net/dl_results/``.

    Artefacts saved:
      - ``model_summary.json``: Comprehensive metrics for all models.
      - ``{arch}.cbm``: CatBoost model binary.
      - ``{arch}.json``: XGBoost model JSON.
      - ``{arch}_member{i}.pt``: PyTorch state dicts for DL ensemble members.
      - ``ensemble_weights.json``: Stacked ensemble base learner weights.
      - ``feature_importance.csv``: Feature importance from interpretability.
      - ``cv_results.csv``, ``cv_summary.json``: Cross-validation outputs.
      - ``tuning_log.json``: Best hyperparameters per architecture.
      - ``model_comparison.json``: DL pipeline comparison table.
      - ``full_model_comparison.json``: Cross-family comparison table.

    Args:
        results: Mapping of architecture name to training result dict.
        bundle: DLFeatureBundle (not serialised; used for metadata).
        interp_results: Interpretability artefacts.
        cv_results: CV results dict.
        comparison: Full model comparison dict.
        config: DL pipeline configuration.
    """
    log.info("=" * 72)
    log.info("SECTION 13: Output Saving")
    log.info("=" * 72)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # model_summary.json
    # ------------------------------------------------------------------
    try:
        model_summary: Dict[str, Any] = {
            "pipeline": "dl",
            "timestamp": pd.Timestamp.now().isoformat(),
            "config": {
                "seed": config.seed,
                "n_tuning_trials": config.n_tuning_trials,
                "cv_folds": config.cv_folds,
                "quick": config.quick,
                "architectures": config.architectures,
                "epochs": config.epochs,
                "patience": config.patience,
                "batch_size": config.batch_size,
                "n_ensemble": config.n_ensemble,
                "catboost_iterations": config.catboost_iterations,
                "mono_lambda": config.mono_lambda,
            },
            "best_model": comparison.get("best_model"),
            "double_lift_vs_glm": comparison.get("double_lift"),
            "glm_gini": comparison.get("glm_gini"),
            "models": {},
        }

        for arch, res in results.items():
            if "error" in res:
                model_summary["models"][arch] = {"error": res["error"]}
                continue
            arch_entry: Dict[str, Any] = {
                "metrics_train": res.get("metrics_train", {}),
                "metrics_test": res.get("metrics_test", {}),
                "training_time": res.get("training_time", 0.0),
                "best_params": {
                    k: (
                        int(v) if isinstance(v, (np.integer,)) else
                        float(v) if isinstance(v, (np.floating,)) else v
                    )
                    for k, v in res.get("best_params", {}).items()
                },
                "n_params": _count_model_params(res, arch),
            }
            if "ensemble_results" in res:
                arch_entry["n_ensemble_members"] = len(res["ensemble_results"])
                arch_entry["best_epochs"] = [
                    tr.best_epoch for tr in res["ensemble_results"]
                ]
            if "base_weights" in res:
                arch_entry["base_weights"] = res["base_weights"]
            model_summary["models"][arch] = arch_entry

        # Merge with existing model_summary to preserve results from previous runs
        summary_path = output_dir / "model_summary.json"
        if summary_path.exists():
            try:
                with open(summary_path) as fh:
                    existing = json.load(fh)
                    existing_models = existing.get("models", {})
                    # Preserve models from previous runs, overwrite with current
                    merged = {**existing_models}
                    merged.update(model_summary["models"])
                    model_summary["models"] = merged
            except Exception:
                pass  # If existing file is corrupt, just overwrite
        with open(summary_path, "w") as fh:
            json.dump(model_summary, fh, indent=2, default=str)
        log.info("  model_summary.json saved to %s", summary_path)
    except Exception as exc:
        log.error("  model_summary.json failed: %s", exc)

    # ------------------------------------------------------------------
    # CatBoost model binary
    # ------------------------------------------------------------------
    if "catboost" in results and "error" not in results["catboost"]:
        try:
            cb_model = results["catboost"].get("model")
            if cb_model is not None and HAS_CATBOOST:
                cb_path = output_dir / "catboost.cbm"
                cb_model.save_model(str(cb_path))
                log.info("  CatBoost model saved to %s", cb_path)
        except Exception as exc:
            log.warning("  CatBoost model save failed: %s", exc)

    # ------------------------------------------------------------------
    # XGBoost model JSON
    # ------------------------------------------------------------------
    if "xgboost" in results and "error" not in results["xgboost"]:
        try:
            xgb_model = results["xgboost"].get("model")
            if xgb_model is not None and HAS_XGBOOST:
                xgb_path = output_dir / "xgboost.json"
                xgb_model.save_model(str(xgb_path))
                log.info("  XGBoost model saved to %s", xgb_path)
        except Exception as exc:
            log.warning("  XGBoost model save failed: %s", exc)

    # ------------------------------------------------------------------
    # GLM predictions and dispersion (for future runs to skip refitting)
    # ------------------------------------------------------------------
    try:
        np.save(output_dir / "glm_train_preds.npy", bundle.glm_train_preds)
        np.save(output_dir / "glm_test_preds.npy", bundle.glm_test_preds)
        with open(output_dir / "glm_dispersion.json", "w") as fh:
            json.dump({"dispersion": bundle.glm_dispersion}, fh)
        log.info("  GLM predictions and dispersion saved")
    except Exception as exc:
        log.warning("  GLM save failed: %s", exc)

    # ------------------------------------------------------------------
    # PyTorch state dicts for DL ensemble members
    # ------------------------------------------------------------------
    if HAS_TORCH:
        for arch in ["cann", "cann_gbm", "ft_transformer", "tabm", "localglmnet", "drn"]:
            if arch not in results or "error" in results[arch]:
                continue
            ensemble_res = results[arch].get("ensemble_results", [])
            for member_i, tr in enumerate(ensemble_res):
                try:
                    if tr.model is not None:
                        pt_path = output_dir / f"{arch}_member{member_i}.pt"
                        torch.save(tr.model.state_dict(), str(pt_path))
                        log.info("  %s member %d saved to %s", arch, member_i, pt_path)
                except Exception as exc:
                    log.warning("  %s member %d save failed: %s", arch, member_i, exc)

    # ------------------------------------------------------------------
    # Ensemble weights JSON
    # ------------------------------------------------------------------
    try:
        ens_res = results.get("stacked_ensemble", {})
        bw = ens_res.get("base_weights", {})
        if bw:
            ew_path = output_dir / "ensemble_weights.json"
            with open(ew_path, "w") as fh:
                json.dump(bw, fh, indent=2)
            log.info("  ensemble_weights.json saved to %s", ew_path)
    except Exception as exc:
        log.warning("  ensemble_weights.json failed: %s", exc)

    # ------------------------------------------------------------------
    # Tuning log (best params per architecture)
    # ------------------------------------------------------------------
    try:
        tuning_log: Dict[str, Any] = {}
        for arch, res in results.items():
            if "error" not in res:
                bp = res.get("best_params", {})
                tuning_log[arch] = {
                    k: (
                        int(v) if isinstance(v, (np.integer,)) else
                        float(v) if isinstance(v, (np.floating,)) else v
                    )
                    for k, v in bp.items()
                }
        tuning_path = output_dir / "tuning_log.json"
        existing_tuning = {}
        if tuning_path.exists():
            try:
                with open(tuning_path) as fh:
                    existing_tuning = json.load(fh)
            except Exception:
                pass
        merged_tuning = {**existing_tuning, **tuning_log}
        with open(tuning_path, "w") as fh:
            json.dump(merged_tuning, fh, indent=2)
        log.info("  tuning_log.json saved to %s", tuning_path)
    except Exception as exc:
        log.warning("  tuning_log.json failed: %s", exc)

    # ------------------------------------------------------------------
    # full_model_comparison.json — already saved by compare_with_existing_models,
    # but we re-save here to ensure it includes stacked_ensemble metrics.
    # ------------------------------------------------------------------
    try:
        full_path = output_dir / "full_model_comparison.json"
        with open(full_path, "w") as fh:
            json.dump(comparison, fh, indent=2, default=str)
        log.info("  full_model_comparison.json saved to %s", full_path)
    except Exception as exc:
        log.warning("  full_model_comparison.json re-save failed: %s", exc)

    log.info("  All outputs saved to %s", output_dir)
