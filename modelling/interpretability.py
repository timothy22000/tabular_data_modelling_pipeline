"""Interpretability: SHAP, Captum IG, attention extraction, permutation importance."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .config import (
    DLConfig,
    HAS_TORCH, HAS_CATBOOST, HAS_XGBOOST, HAS_CAPTUM,
    torch, log,
)
from .data import DLFeatureBundle


def run_interpretability(
    results: Dict[str, Any],
    bundle: DLFeatureBundle,
    config: DLConfig,
) -> Dict[str, Any]:
    """Compute interpretability artefacts for all trained models.

    Skipped entirely when ``config.skip_interpretability`` is True.

    Methods applied per model type:
      - CatBoost / XGBoost: SHAP TreeExplainer (shap library, if available).
      - CANN / FT-Transformer / TabM: Captum IntegratedGradients (if HAS_CAPTUM).
      - FT-Transformer: Forward hook to extract attention weights.
      - CANN: NN residual distribution analysis.
      - All DL models: Permutation importance via sklearn.

    Saves ``feature_importance.csv`` to ``config.output_dir``.

    Args:
        results: Mapping of architecture name to training result dict.
        bundle: DLFeatureBundle with feature matrices and metadata.
        config: DL pipeline configuration.

    Returns:
        Dictionary mapping model name to interpretability artefacts dict.
    """
    if config.skip_interpretability:
        log.warning("  Interpretability skipped (--skip-interpretability)")
        return {}

    log.info("=" * 72)
    log.info("SECTION 9: Interpretability")
    log.info("=" * 72)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    interp_results: Dict[str, Any] = {}
    all_feature_names = (
        bundle.continuous_feature_names + bundle.categorical_feature_names
    )
    n_subsample = 200 if config.quick else 500

    rng = np.random.default_rng(config.seed)
    n_test = len(bundle.y_test)
    sub_idx = rng.choice(n_test, size=min(n_subsample, n_test), replace=False)

    # ------------------------------------------------------------------
    # 9a: CatBoost interpretability
    # ------------------------------------------------------------------
    if "catboost" in results and "error" not in results["catboost"]:
        log.info("  9a: CatBoost interpretability")
        artefacts_cb: Dict[str, Any] = {}
        cb_model = results["catboost"].get("model")

        if cb_model is not None and HAS_CATBOOST:
            try:
                fi = cb_model.get_feature_importance()
                fi_names = cb_model.feature_names_
                artefacts_cb["feature_importance"] = dict(zip(fi_names, fi.tolist()))
                log.info("    CatBoost built-in importances computed")
            except Exception as exc:
                log.warning("    CatBoost feature importance failed: %s", exc)

            try:
                import shap

                # Reconstruct raw test data matching CatBoost training format
                X_test_cont_raw_cb = (bundle.X_test_cont * bundle.cont_std + bundle.cont_mean).astype(np.float32)
                test_data_df = pd.DataFrame(X_test_cont_raw_cb, columns=bundle.continuous_feature_names)
                for _col in bundle.categorical_feature_names:
                    test_data_df[_col] = bundle.test_df[_col].astype(str).fillna("UNKNOWN").values
                test_data = test_data_df.iloc[sub_idx]
                explainer = shap.TreeExplainer(cb_model)
                shap_vals = explainer.shap_values(test_data)
                mean_abs_shap = np.abs(shap_vals).mean(axis=0)
                artefacts_cb["shap_values"] = shap_vals
                artefacts_cb["mean_abs_shap"] = dict(
                    zip(all_feature_names, mean_abs_shap.tolist())
                )
                log.info("    CatBoost SHAP values computed (%d samples)", len(sub_idx))
            except ImportError:
                log.warning("    shap not installed — skipping SHAP for CatBoost")
            except Exception as exc:
                log.warning("    CatBoost SHAP failed: %s", exc)

        interp_results["catboost"] = artefacts_cb

    # ------------------------------------------------------------------
    # 9b: XGBoost interpretability
    # ------------------------------------------------------------------
    if "xgboost" in results and "error" not in results["xgboost"]:
        log.info("  9b: XGBoost interpretability")
        artefacts_xgb: Dict[str, Any] = {}
        xgb_model = results["xgboost"].get("model")

        if xgb_model is not None and HAS_XGBOOST:
            try:
                fi_arr = xgb_model.feature_importances_
                xgb_fn = bundle.continuous_feature_names + bundle.categorical_feature_names
                artefacts_xgb["feature_importance"] = dict(zip(xgb_fn, fi_arr.tolist()))
                log.info("    XGBoost built-in importances computed")
            except Exception as exc:
                log.warning("    XGBoost feature importance failed: %s", exc)

            try:
                import shap

                X_test_cont_raw = (bundle.X_test_cont * bundle.cont_std + bundle.cont_mean).astype(np.float32)
                X_sub_raw = np.concatenate(
                    [
                        X_test_cont_raw[sub_idx],
                        bundle.X_test_cat[sub_idx].astype(np.float32),
                    ],
                    axis=1,
                )
                explainer = shap.TreeExplainer(xgb_model)
                shap_vals = explainer.shap_values(X_sub_raw)
                mean_abs_shap = np.abs(shap_vals).mean(axis=0)
                xgb_fn2 = bundle.continuous_feature_names + bundle.categorical_feature_names
                artefacts_xgb["shap_values"] = shap_vals
                artefacts_xgb["mean_abs_shap"] = dict(
                    zip(xgb_fn2, mean_abs_shap.tolist())
                )
                log.info("    XGBoost SHAP values computed (%d samples)", len(sub_idx))
            except ImportError:
                log.warning("    shap not installed — skipping SHAP for XGBoost")
            except Exception as exc:
                log.warning("    XGBoost SHAP failed: %s", exc)

        interp_results["xgboost"] = artefacts_xgb

    # ------------------------------------------------------------------
    # 9c: DL model interpretability
    # ------------------------------------------------------------------
    for arch in ["cann", "cann_gbm", "ft_transformer", "tabm", "localglmnet", "drn"]:
        if arch not in results or "error" in results[arch]:
            continue
        if not HAS_TORCH:
            log.warning("  PyTorch unavailable — skipping DL interpretability for %s", arch)
            continue

        log.info("  9c: DL interpretability — %s", arch)
        artefacts_dl: Dict[str, Any] = {}
        ensemble_res = results[arch].get("ensemble_results", [])
        first_model = ensemble_res[0].model if ensemble_res else None

        if first_model is None:
            interp_results[arch] = artefacts_dl
            continue

        first_model = first_model.to(torch.device("cpu"))
        first_model.eval()

        x_cont_sub = torch.tensor(bundle.X_test_cont[sub_idx], dtype=torch.float32)
        x_cat_sub = torch.tensor(bundle.X_test_cat[sub_idx], dtype=torch.long)
        glm_sub = torch.tensor(bundle.glm_test_preds[sub_idx], dtype=torch.float32)

        # Resolve base prediction for this architecture
        if arch == "cann_gbm" and bundle.gbm_test_preds is not None:
            base_sub = torch.tensor(bundle.gbm_test_preds[sub_idx], dtype=torch.float32)
        else:
            base_sub = glm_sub

        # Captum Integrated Gradients
        if HAS_CAPTUM:
            try:
                from captum.attr import IntegratedGradients

                _x_cat_cap = x_cat_sub
                _base_cap = base_sub
                _mdl_cap = first_model

                def _ig_wrapper(x_c: "torch.Tensor") -> "torch.Tensor":
                    pred, _ = _mdl_cap(x_c, _x_cat_cap, _base_cap)
                    return pred.unsqueeze(-1)

                ig = IntegratedGradients(_ig_wrapper)
                baseline = torch.zeros_like(x_cont_sub)
                attributions, _ = ig.attribute(
                    x_cont_sub,
                    baselines=baseline,
                    return_convergence_delta=True,
                    n_steps=50,
                )
                attr_np = attributions.detach().cpu().numpy()
                mean_abs_attr = np.abs(attr_np).mean(axis=0)
                artefacts_dl["attributions"] = attr_np
                artefacts_dl["mean_abs_attributions"] = dict(
                    zip(bundle.continuous_feature_names, mean_abs_attr.tolist())
                )
                log.info("    %s IG attributions computed (%d samples)", arch, len(sub_idx))
            except Exception as exc:
                log.warning("    %s Captum IG failed: %s", arch, exc)
        else:
            log.warning("    Captum not installed — skipping IG for %s", arch)

        # FT-Transformer attention weights
        if arch == "ft_transformer":
            try:
                with torch.no_grad():
                    tokens = first_model.tokenizer(x_cont_sub, x_cat_sub)
                    attn_weights_list: List[np.ndarray] = []
                    x_tok = tokens
                    for layer in first_model.transformer.layers:
                        attn_out, attn_w = layer.self_attn(
                            x_tok, x_tok, x_tok,
                            need_weights=True,
                            average_attn_weights=True,
                        )
                        attn_weights_list.append(attn_w.cpu().numpy())
                        x_tok = layer(x_tok)
                if attn_weights_list:
                    avg_attn = np.mean(
                        [w.mean(axis=0) for w in attn_weights_list], axis=0
                    )
                    artefacts_dl["attention_weights"] = avg_attn
                    log.info(
                        "    FT-Transformer attention extracted (%d layers)",
                        len(attn_weights_list),
                    )
            except Exception as exc:
                log.warning("    FT-Transformer attention failed: %s", exc)

        # CANN / CANN-GBM residual analysis
        if arch in ("cann", "cann_gbm"):
            try:
                with torch.no_grad():
                    _, nn_residual = first_model(x_cont_sub, x_cat_sub, base_sub)
                residuals_np = nn_residual.cpu().numpy()
                artefacts_dl[f"{arch}_residuals"] = residuals_np

                residual_by_factor: Dict[str, Dict[str, float]] = {}
                for col in bundle.categorical_feature_names[:5]:
                    col_idx = bundle.categorical_feature_names.index(col)
                    cat_codes = bundle.X_test_cat[sub_idx, col_idx]
                    rev_map = {v: k for k, v in bundle.category_mappings[col].items()}
                    fac_res: Dict[str, List[float]] = {}
                    for code, resid in zip(cat_codes, residuals_np):
                        label = rev_map.get(int(code), "UNKNOWN")
                        fac_res.setdefault(label, []).append(float(resid))
                    residual_by_factor[col] = {
                        k: float(np.mean(v)) for k, v in fac_res.items()
                    }
                artefacts_dl[f"{arch}_residuals_by_factor"] = residual_by_factor
                log.info(
                    "    %s residuals analysed (mean=%.4f, std=%.4f)",
                    arch,
                    float(residuals_np.mean()),
                    float(residuals_np.std()),
                )
            except Exception as exc:
                log.warning("    %s residual analysis failed: %s", arch, exc)

        # LocalGLMnet coefficient extraction
        if arch == "localglmnet":
            try:
                with torch.no_grad():
                    _, coeffs = first_model(x_cont_sub, x_cat_sub, base_sub)
                coeffs_np = coeffs.cpu().numpy()  # shape (n_sub, n_cont)
                artefacts_dl["localglmnet_coefficients"] = coeffs_np

                # Standardised coefficients (as-is, features are z-scored)
                coeff_df = pd.DataFrame(
                    coeffs_np,
                    columns=bundle.continuous_feature_names,
                )
                # Raw coefficients: beta_raw_k = beta_std_k / std_k
                raw_coeffs = coeffs_np / bundle.cont_std[np.newaxis, :]
                coeff_df_raw = pd.DataFrame(
                    raw_coeffs,
                    columns=[f"{c}_raw" for c in bundle.continuous_feature_names],
                )
                coeff_out = pd.concat([coeff_df, coeff_df_raw], axis=1)
                coeff_path = output_dir / "localglmnet_coefficients.csv"
                coeff_out.to_csv(coeff_path, index=False)

                # Summary statistics
                mean_coeffs = coeffs_np.mean(axis=0)
                artefacts_dl["mean_coefficients"] = dict(
                    zip(bundle.continuous_feature_names, mean_coeffs.tolist())
                )
                log.info(
                    "    LocalGLMnet coefficients extracted (%d samples, %d features) -> %s",
                    len(sub_idx),
                    len(bundle.continuous_feature_names),
                    coeff_path,
                )
            except Exception as exc:
                log.warning("    LocalGLMnet coefficient extraction failed: %s", exc)

        # DRN distributional outputs
        if arch == "drn":
            try:
                from scipy.stats import gamma as gamma_dist

                with torch.no_grad():
                    mean_pred, dist_params = first_model(x_cont_sub, x_cat_sub, base_sub)
                shape_np = dist_params[:, 0].cpu().numpy()
                rate_np = dist_params[:, 1].cpu().numpy()
                mean_np = mean_pred.cpu().numpy()

                # Per-policy distributional quantities
                cov = 1.0 / np.sqrt(np.maximum(shape_np, 1e-6))  # CoV = 1/sqrt(shape)
                scale_np = 1.0 / np.maximum(rate_np, 1e-6)  # scipy scale = 1/rate
                p95 = gamma_dist.ppf(0.95, a=shape_np, scale=scale_np)
                p99 = gamma_dist.ppf(0.99, a=shape_np, scale=scale_np)

                drn_df = pd.DataFrame({
                    "mean_pred": mean_np,
                    "shape": shape_np,
                    "rate": rate_np,
                    "cov": cov,
                    "p95": p95,
                    "p99": p99,
                })
                drn_path = output_dir / "drn_distributional_outputs.csv"
                drn_df.to_csv(drn_path, index=False)

                # Aggregate VaR
                agg_var_95 = float(np.percentile(p95, 95))
                agg_var_99 = float(np.percentile(p99, 99))
                artefacts_dl["drn_distributional"] = {
                    "mean_shape": float(shape_np.mean()),
                    "mean_cov": float(cov.mean()),
                    "agg_var_95": agg_var_95,
                    "agg_var_99": agg_var_99,
                }
                log.info(
                    "    DRN distributional outputs: mean_shape=%.3f, mean_CoV=%.3f, "
                    "VaR95=%.0f, VaR99=%.0f -> %s",
                    float(shape_np.mean()),
                    float(cov.mean()),
                    agg_var_95,
                    agg_var_99,
                    drn_path,
                )
            except ImportError:
                log.warning("    scipy not installed — skipping DRN distributional outputs")
            except Exception as exc:
                log.warning("    DRN distributional output failed: %s", exc)

        # Permutation importance
        try:
            from sklearn.inspection import permutation_importance as sk_perm_imp

            _pm_model = first_model
            _pm_x_cat = x_cat_sub
            _pm_base = base_sub

            class _DLPredictor:
                """Sklearn-compatible wrapper for DL model."""

                def predict(self, X: np.ndarray) -> np.ndarray:
                    x_c = torch.tensor(X, dtype=torch.float32)
                    with torch.no_grad():
                        pred, _ = _pm_model(x_c, _pm_x_cat, _pm_base)
                    return pred.numpy()

            predictor = _DLPredictor()
            n_pi_repeats = 3 if config.quick else 5
            perm_result = sk_perm_imp(
                predictor,
                bundle.X_test_cont[sub_idx],
                bundle.y_test[sub_idx],
                n_repeats=n_pi_repeats,
                random_state=config.seed,
                scoring="neg_mean_absolute_error",
            )
            artefacts_dl["perm_importance"] = dict(
                zip(
                    bundle.continuous_feature_names,
                    perm_result.importances_mean.tolist(),
                )
            )
            log.info(
                "    %s permutation importance computed (%d features)",
                arch,
                len(bundle.continuous_feature_names),
            )
        except Exception as exc:
            log.warning("    %s permutation importance failed: %s", arch, exc)

        interp_results[arch] = artefacts_dl

    # ------------------------------------------------------------------
    # 9d: Aggregate feature importance table
    # ------------------------------------------------------------------
    try:
        importance_rows: Dict[str, Dict[str, float]] = {}
        for model_name, artefacts_i in interp_results.items():
            if "mean_abs_shap" in artefacts_i:
                importance_rows[model_name] = artefacts_i["mean_abs_shap"]
            elif "mean_abs_attributions" in artefacts_i:
                importance_rows[model_name] = artefacts_i["mean_abs_attributions"]
            elif "perm_importance" in artefacts_i:
                importance_rows[model_name] = artefacts_i["perm_importance"]
            elif "feature_importance" in artefacts_i:
                importance_rows[model_name] = artefacts_i["feature_importance"]

        if importance_rows:
            fi_df = pd.DataFrame(importance_rows)
            fi_path = output_dir / "feature_importance.csv"
            fi_df.to_csv(fi_path)
            log.info("  Feature importance saved to %s", fi_path)
    except Exception as exc:
        log.warning("  Feature importance aggregation failed: %s", exc)

    log.info("  Interpretability complete for %d models", len(interp_results))
    return interp_results
