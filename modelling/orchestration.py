"""Model training orchestration — trains all configured architectures."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .config import DLConfig, HAS_TORCH, HAS_CATBOOST, HAS_XGBOOST, log, _clamp_predictions
from .data import DLFeatureBundle, build_dataloaders
from .models import get_default_dl_params
from .models.catboost_model import train_catboost
from .models.xgboost_model import train_xgboost
from .tuning import tune_catboost, tune_xgboost, tune_dl_model
from .training import _train_dl_ensemble


def train_all_models(
    bundle: DLFeatureBundle,
    config: DLConfig,
) -> Dict[str, Any]:
    """Train all configured architectures sequentially and collect results.

    Training order: catboost -> xgboost -> cann -> ft_transformer -> tabm.
    For each architecture:
      1. Run Optuna tuning (or skip with defaults).
      2. Train the model (GBM: single fit; DL: ensemble of n_ensemble seeds).
      3. Log Gini, MAE, A/E ratio.
      4. Save metrics to JSON.

    Args:
        bundle: DLFeatureBundle with all feature matrices and metadata.
        config: DL pipeline configuration.

    Returns:
        Dictionary mapping architecture name to result dict.  Each result
        dict contains at minimum:
            - "train_preds": np.ndarray
            - "test_preds": np.ndarray
            - "metrics_train": Dict[str, Any]
            - "metrics_test": Dict[str, Any]
            - "best_params": Dict[str, Any]
            - "training_time": float
    """
    log.info("=" * 72)
    log.info("SECTION 6: Model Training")
    log.info("=" * 72)
    log.info("  Architectures to train: %s", config.architectures)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}

    # Pre-build DataLoaders once for DL architectures
    dl_loaders_built = False
    train_loader: Any = None
    val_loader: Any = None
    test_loader: Any = None

    arch_order = ["catboost", "xgboost", "cann", "cann_gbm", "ft_transformer", "tabm", "localglmnet", "drn"]
    active_archs = [a for a in arch_order if a in config.architectures]

    # DRN's architecture and loss function are derived for a Gamma response
    # distribution: it refines the Gamma (shape, rate) parameters around a
    # GLM base prediction and minimises Gamma NLL + KL(Gamma||Gamma_base).
    # Running DRN on a non-Gamma family (Poisson counts, Gaussian, etc.)
    # minimises the wrong objective and produces miscalibrated predictions
    # even when rank order looks fine. Skip DRN with an explicit warning
    # rather than silently shipping broken artefacts. A Poisson/Tweedie
    # variant would require a new architecture + loss, not a config flag.
    if "drn" in active_archs and config.dataset.family not in ("gamma", "tweedie"):
        log.warning(
            "  DRN skipped: requires gamma or tweedie family "
            "(got '%s'). The architecture refines Gamma (shape, rate) "
            "parameters and the loss is Gamma NLL; running it on '%s' "
            "data produces well-ranked but badly calibrated predictions. "
            "If you need a distributional model for this family, retrain "
            "with --architectures excluding drn, or implement a "
            "family-specific variant.",
            config.dataset.family,
            config.dataset.family,
        )
        active_archs = [a for a in active_archs if a != "drn"]

    # If cann_gbm requested but catboost not in architectures, load saved model
    if "cann_gbm" in config.architectures and "catboost" not in config.architectures:
        if bundle.gbm_train_preds is None:
            cb_path = output_dir / "catboost.cbm"
            if cb_path.exists() and bundle.catboost_train_pool is not None:
                from catboost import CatBoostRegressor as _CBR
                log.info("  Loading saved CatBoost model from %s", cb_path)
                cb_model = _CBR()
                cb_model.load_model(str(cb_path))
                train_preds = _clamp_predictions(
                    np.asarray(cb_model.predict(bundle.catboost_train_pool), dtype=np.float32)
                )
                test_preds = _clamp_predictions(
                    np.asarray(cb_model.predict(bundle.catboost_test_pool), dtype=np.float32)
                )
                bundle.gbm_train_preds = train_preds
                bundle.gbm_test_preds = test_preds
                dl_loaders_built = False
                log.info("  GBM predictions loaded (%d train, %d test)",
                         len(train_preds), len(test_preds))
            else:
                log.warning("  No saved CatBoost model at %s — CANN-GBM will be skipped", cb_path)

    for arch in active_archs:
        log.info("")
        log.info("  " + "=" * 68)
        log.info("  Training: %s", arch.upper())
        log.info("  " + "=" * 68)

        t_arch_start = time.time()

        try:
            if arch == "catboost":
                best_params = tune_catboost(bundle, config)
                result = train_catboost(bundle, config, best_params)
                result["best_params"] = best_params
                results[arch] = result
                # Populate GBM predictions for CANN-GBM
                if result.get("train_preds") is not None:
                    bundle.gbm_train_preds = result["train_preds"].astype(np.float32)
                    bundle.gbm_test_preds = result["test_preds"].astype(np.float32)
                    log.info("  GBM predictions stored for CANN-GBM (%d train, %d test)",
                             len(bundle.gbm_train_preds), len(bundle.gbm_test_preds))
                    # Reset DataLoaders so they pick up GBM predictions
                    dl_loaders_built = False

            elif arch == "xgboost":
                best_params = tune_xgboost(bundle, config)
                result = train_xgboost(bundle, config, best_params)
                result["best_params"] = best_params
                results[arch] = result

            else:
                # DL architectures require PyTorch
                if not HAS_TORCH:
                    log.warning(
                        "  PyTorch not installed — skipping %s", arch
                    )
                    continue

                # CANN-GBM requires CatBoost predictions
                if arch == "cann_gbm" and bundle.gbm_train_preds is None:
                    log.warning("  CANN-GBM requires CatBoost — skipping (CatBoost not trained)")
                    continue

                # Build DataLoaders once (shared across all DL architectures)
                if not dl_loaders_built:
                    train_loader, val_loader, test_loader = build_dataloaders(
                        bundle, config
                    )
                    dl_loaders_built = True

                if train_loader is None:
                    log.warning("  DataLoaders unavailable — skipping %s", arch)
                    continue

                best_params = tune_dl_model(arch, bundle, config)
                result = _train_dl_ensemble(
                    architecture=arch,
                    params=best_params,
                    bundle=bundle,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    config=config,
                )
                result["best_params"] = best_params
                results[arch] = result

            t_arch_elapsed = time.time() - t_arch_start
            m_test = results[arch]["metrics_test"]
            log.info(
                "  [%s] DONE in %.1fs — Test Gini=%.4f | MAE=%.0f | A/E=%.4f",
                arch,
                t_arch_elapsed,
                m_test["gini"],
                m_test["mae"],
                m_test["ae_ratio"],
            )

        except Exception as exc:
            log.error("  [%s] Training FAILED: %s", arch, exc, exc_info=True)
            results[arch] = {"error": str(exc)}

    # ----- Save all metrics to JSON ----------------------------------------
    metrics_summary: Dict[str, Any] = {}
    for arch, res in results.items():
        if "error" in res:
            metrics_summary[arch] = {"error": res["error"]}
        else:
            metrics_summary[arch] = {
                "metrics_train": res.get("metrics_train", {}),
                "metrics_test": res.get("metrics_test", {}),
                "best_params": {
                    k: (
                        int(v) if isinstance(v, (np.integer,)) else
                        float(v) if isinstance(v, (np.floating,)) else v
                    )
                    for k, v in res.get("best_params", {}).items()
                },
                "training_time": res.get("training_time", 0.0),
            }

    metrics_path = output_dir / "dl_metrics_summary.json"
    existing_metrics = {}
    if metrics_path.exists():
        try:
            with open(metrics_path) as fh:
                existing_metrics = json.load(fh)
        except Exception:
            pass
    merged_metrics = {**existing_metrics, **metrics_summary}
    with open(metrics_path, "w") as f:
        json.dump(merged_metrics, f, indent=2, default=str)
    log.info("")
    log.info("  Metrics summary saved to %s", metrics_path)

    # ----- Leaderboard summary ---------------------------------------------
    log.info("")
    log.info("  " + "=" * 68)
    log.info("  LEADERBOARD (Test Set)")
    log.info("  " + "=" * 68)
    log.info("  %-20s  %8s  %8s  %8s  %10s", "Architecture", "Gini", "MAE", "A/E", "Time(s)")
    log.info("  " + "-" * 64)
    for arch in active_archs:
        if arch in results and "error" not in results[arch]:
            m = results[arch]["metrics_test"]
            t = results[arch].get("training_time", float("nan"))
            log.info(
                "  %-20s  %8.4f  %8.0f  %8.4f  %10.1f",
                arch,
                m["gini"],
                m["mae"],
                m["ae_ratio"],
                t,
            )
        else:
            log.info("  %-20s  %s", arch, "FAILED")
    log.info("  " + "=" * 68)

    return results
