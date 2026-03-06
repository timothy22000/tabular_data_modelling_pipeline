"""Diagnostic figures and HTML dashboards for the DL pipeline."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .config import (
    DLConfig,
    HAS_TORCH, HAS_MATPLOTLIB,
    torch, log,
    C_PRIMARY, C_ACCENT, C_GREEN, C_GOLD, C_RED, C_PURPLE, C_TEAL,
    MONOTONE_CONSTRAINTS, _lorenz_curve, compute_decile_analysis,
)
from .data import DLFeatureBundle
from .evaluation import _count_model_params

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    pass


_STYLE_TRIED = False


def _setup_plot_style() -> None:
    """Apply matplotlib style and configure DPI once."""
    global _STYLE_TRIED
    if _STYLE_TRIED or not HAS_MATPLOTLIB:
        return
    _STYLE_TRIED = True
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass  # Use matplotlib default


def _family_color(model_name: str) -> str:
    """Return a consistent colour for a model based on its family.

    Args:
        model_name: Architecture name.

    Returns:
        Hex colour string.
    """
    name = model_name.lower()
    if name == "glm":
        return "#9E9E9E"
    if name in ("lightgbm", "lgbm"):
        return C_TEAL
    if name in ("catboost", "xgboost"):
        return C_PRIMARY
    if name == "cann":
        return C_PURPLE
    if name == "ft_transformer":
        return C_ACCENT
    if name == "tabm":
        return C_GREEN
    if "ensemble" in name or "stacked" in name:
        return C_GOLD
    return C_RED


def generate_dl_visualizations(
    results: Dict[str, Any],
    bundle: DLFeatureBundle,
    interp_results: Dict[str, Any],
    cv_results: Dict[str, Any],
    comparison: Dict[str, Any],
    config: DLConfig,
) -> None:
    """Generate all 17 diagnostic figures plus two HTML dashboards.

    All figures are saved to ``{config.output_dir}/figures/``.
    Each figure is wrapped in an individual try/except block so a single
    failure does not abort the entire visualisation pass.

    Args:
        results: Mapping of architecture name to training result dict.
        bundle: DLFeatureBundle with feature matrices, targets, and metadata.
        interp_results: Output of run_interpretability().
        cv_results: Output of run_cross_validation().
        comparison: Output of compare_with_existing_models().
        config: DL pipeline configuration.
    """
    if not HAS_MATPLOTLIB:
        log.warning("  matplotlib not installed — skipping visualizations")
        return

    log.info("=" * 72)
    log.info("SECTION 12: Visualizations")
    log.info("=" * 72)

    _setup_plot_style()
    fig_dir = Path(config.output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    DPI = 150

    # Gather all successfully trained models
    trained_models = {
        name: res for name, res in results.items()
        if "error" not in res and "test_preds" in res
    }

    comparison_table = comparison.get("comparison_table", [])

    # -----------------------------------------------------------------
    # Figure 1: Gini comparison bar chart
    # -----------------------------------------------------------------
    try:
        log.info("  Figure 1: Gini comparison")
        all_rows = comparison_table if comparison_table else [
            {
                "model": name,
                "gini_train": res["metrics_train"].get("gini", 0),
                "gini_test": res["metrics_test"].get("gini", 0),
            }
            for name, res in trained_models.items()
        ]
        models_fig1 = [r["model"] for r in all_rows]
        ginis_train = [r.get("gini_train", 0) for r in all_rows]
        ginis_test = [r.get("gini_test", 0) for r in all_rows]

        x = np.arange(len(models_fig1))
        w = 0.35
        fig, ax = plt.subplots(figsize=(max(8, len(models_fig1) * 1.1), 5))
        bars_tr = ax.bar(x - w / 2, ginis_train, w, label="Train",
                         color=[_family_color(m) for m in models_fig1], alpha=0.6)
        bars_te = ax.bar(x + w / 2, ginis_test, w, label="Test",
                         color=[_family_color(m) for m in models_fig1], alpha=0.95)
        ax.set_xticks(x)
        ax.set_xticklabels(models_fig1, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Gini Coefficient")
        ax.set_title("Model Gini Comparison (Train vs Test)", fontweight="bold")
        ax.legend()
        for bar in bars_te:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.002,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7,
            )
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_dl_01_gini_comparison.png", dpi=DPI)
        plt.close(fig)
        log.info("    Saved fig_dl_01_gini_comparison.png")
    except Exception as exc:
        log.warning("  Figure 1 failed: %s", exc)

    # -----------------------------------------------------------------
    # Figure 2: Lorenz curves
    # -----------------------------------------------------------------
    try:
        log.info("  Figure 2: Lorenz curves")
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.plot([0, 0, 1], [0, 1, 1], color="#CCCCCC", lw=1, ls=":", label="Perfect")

        for name, res in trained_models.items():
            try:
                xv, yv = _lorenz_curve(bundle.y_test, res["test_preds"])
                ax.plot(xv, yv, label=name, color=_family_color(name), lw=1.8)
            except Exception:
                pass

        ax.set_xlabel("Cumulative Share of Policies")
        ax.set_ylabel("Cumulative Share of Premium")
        ax.set_title("Lorenz Curves — All Models", fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_dl_02_lorenz_curves.png", dpi=DPI)
        plt.close(fig)
        log.info("    Saved fig_dl_02_lorenz_curves.png")
    except Exception as exc:
        log.warning("  Figure 2 failed: %s", exc)

    # -----------------------------------------------------------------
    # Figure 3: Training curves
    # -----------------------------------------------------------------
    try:
        log.info("  Figure 3: Training curves")
        dl_archs_trained = [
            name for name in ["cann", "cann_gbm", "ft_transformer", "tabm", "localglmnet", "drn"]
            if name in results and "error" not in results[name]
            and results[name].get("ensemble_results")
        ]
        if dl_archs_trained:
            n_sub = len(dl_archs_trained)
            fig, axes = plt.subplots(1, n_sub, figsize=(5 * n_sub, 4), squeeze=False)
            for col_i, arch in enumerate(dl_archs_trained):
                ax = axes[0][col_i]
                ensemble_res = results[arch]["ensemble_results"]
                for member_i, tr in enumerate(ensemble_res):
                    epochs_range = range(1, len(tr.train_losses) + 1)
                    alpha = 0.9 if member_i == 0 else 0.4
                    ax.plot(epochs_range, tr.train_losses, color=C_PRIMARY,
                            lw=1.2, alpha=alpha,
                            label="Train" if member_i == 0 else "_")
                    ax.plot(epochs_range, tr.val_losses, color=C_RED,
                            lw=1.2, alpha=alpha,
                            label="Val" if member_i == 0 else "_")
                    # Mark best epoch
                    best_ep = tr.best_epoch + 1
                    if 0 < best_ep <= len(tr.val_losses):
                        ax.axvline(best_ep, color=C_GOLD, lw=1, ls="--", alpha=0.7)
                ax.set_title(arch, fontweight="bold", fontsize=10)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Gamma Deviance")
                ax.legend(fontsize=8)
            fig.suptitle("DL Training Curves", fontweight="bold", y=1.01)
            fig.tight_layout()
            fig.savefig(fig_dir / "fig_dl_03_training_curves.png", dpi=DPI,
                        bbox_inches="tight")
            plt.close(fig)
            log.info("    Saved fig_dl_03_training_curves.png")
    except Exception as exc:
        log.warning("  Figure 3 failed: %s", exc)

    # -----------------------------------------------------------------
    # Figure 4: CatBoost feature importance
    # -----------------------------------------------------------------
    try:
        log.info("  Figure 4: CatBoost feature importance")
        cb_interp = interp_results.get("catboost", {})
        imp_source = cb_interp.get("mean_abs_shap") or cb_interp.get("feature_importance")
        if imp_source:
            top15 = sorted(imp_source.items(), key=lambda x: x[1], reverse=True)[:15]
            feat_labels = [f[0] for f in top15]
            feat_vals = [f[1] for f in top15]

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(feat_labels[::-1], feat_vals[::-1], color=C_PRIMARY)
            ax.set_xlabel("Mean |SHAP|" if "mean_abs_shap" in cb_interp else "Importance")
            ax.set_title("CatBoost Feature Importance (Top 15)", fontweight="bold")
            fig.tight_layout()
            fig.savefig(fig_dir / "fig_dl_04_catboost_importance.png", dpi=DPI)
            plt.close(fig)
            log.info("    Saved fig_dl_04_catboost_importance.png")
    except Exception as exc:
        log.warning("  Figure 4 failed: %s", exc)

    # -----------------------------------------------------------------
    # Figure 5: CANN residuals
    # -----------------------------------------------------------------
    try:
        log.info("  Figure 5: CANN residuals")
        cann_interp = interp_results.get("cann", {})
        residuals = cann_interp.get("cann_residuals")
        if residuals is not None:
            by_factor = cann_interp.get("cann_residuals_by_factor", {})
            n_factors = len(by_factor)
            ncols = min(3, max(1, n_factors))
            nrows = max(1, (n_factors + ncols - 1) // ncols)
            fig = plt.figure(figsize=(5 * ncols + 4, 4 * nrows))
            # Main distribution
            ax0 = fig.add_subplot(nrows + 1, 1, 1)
            ax0.hist(residuals, bins=60, color=C_PURPLE, alpha=0.75, edgecolor="white")
            ax0.axvline(0, color=C_RED, lw=1.5, ls="--")
            ax0.set_xlabel("NN Residual (log-scale)")
            ax0.set_ylabel("Count")
            ax0.set_title("CANN NN Residual Distribution", fontweight="bold")

            for fi, (col, factor_means) in enumerate(by_factor.items()):
                ax_f = fig.add_subplot(nrows + 1, ncols, ncols + fi + 1)
                labels = list(factor_means.keys())[:10]
                vals = [factor_means[l] for l in labels]
                colors = [C_RED if v > 0 else C_PRIMARY for v in vals]
                ax_f.barh(labels, vals, color=colors)
                ax_f.axvline(0, color="black", lw=0.8)
                ax_f.set_title(col, fontsize=9)
                ax_f.set_xlabel("Mean residual")
            fig.tight_layout()
            fig.savefig(fig_dir / "fig_dl_05_cann_residuals.png", dpi=DPI)
            plt.close(fig)
            log.info("    Saved fig_dl_05_cann_residuals.png")
    except Exception as exc:
        log.warning("  Figure 5 failed: %s", exc)

    # -----------------------------------------------------------------
    # Figure 6: FT-Transformer attention heatmap
    # -----------------------------------------------------------------
    try:
        log.info("  Figure 6: Attention heatmap")
        ftt_interp = interp_results.get("ft_transformer", {})
        attn_w = ftt_interp.get("attention_weights")
        if attn_w is not None and HAS_MATPLOTLIB:
            n_tok = attn_w.shape[0]
            # Build token labels: CLS + features
            token_labels = ["[CLS]"] + bundle.continuous_feature_names + bundle.categorical_feature_names
            token_labels = token_labels[:n_tok]

            fig, ax = plt.subplots(figsize=(min(14, n_tok * 0.6 + 2), min(12, n_tok * 0.6 + 2)))
            im = ax.imshow(attn_w, aspect="auto", cmap="Blues")
            ax.set_xticks(range(len(token_labels)))
            ax.set_yticks(range(len(token_labels)))
            ax.set_xticklabels(token_labels, rotation=90, fontsize=7)
            ax.set_yticklabels(token_labels, fontsize=7)
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title("FT-Transformer Avg Attention Weights", fontweight="bold")
            fig.tight_layout()
            fig.savefig(fig_dir / "fig_dl_06_attention_heatmap.png", dpi=DPI)
            plt.close(fig)
            log.info("    Saved fig_dl_06_attention_heatmap.png")
    except Exception as exc:
        log.warning("  Figure 6 failed: %s", exc)

    # -----------------------------------------------------------------
    # Figure 7: Integrated Gradients attribution bar
    # -----------------------------------------------------------------
    try:
        log.info("  Figure 7: Attribution bars")
        dl_with_attr = [
            name for name in ["cann", "cann_gbm", "ft_transformer", "tabm", "localglmnet", "drn"]
            if interp_results.get(name, {}).get("mean_abs_attributions")
        ]
        if dl_with_attr:
            ncols_7 = len(dl_with_attr)
            fig, axes = plt.subplots(1, ncols_7, figsize=(6 * ncols_7, 6), squeeze=False)
            for ci, arch in enumerate(dl_with_attr):
                ax = axes[0][ci]
                attr_dict = interp_results[arch]["mean_abs_attributions"]
                top10 = sorted(attr_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                feat_lb = [f[0] for f in top10]
                feat_va = [f[1] for f in top10]
                ax.barh(feat_lb[::-1], feat_va[::-1], color=_family_color(arch))
                ax.set_title(f"{arch}\nIntegrated Gradients", fontsize=9)
                ax.set_xlabel("Mean |Attribution|")
            fig.suptitle("DL Integrated Gradients — Top Features", fontweight="bold")
            fig.tight_layout()
            fig.savefig(fig_dir / "fig_dl_07_attribution_bar.png", dpi=DPI)
            plt.close(fig)
            log.info("    Saved fig_dl_07_attribution_bar.png")
    except Exception as exc:
        log.warning("  Figure 7 failed: %s", exc)

    # -----------------------------------------------------------------
    # Figure 8: Decile A/E calibration bars
    # -----------------------------------------------------------------
    try:
        log.info("  Figure 8: Decile calibration")
        model_names_8 = list(trained_models.keys())
        n_models_8 = len(model_names_8)
        if n_models_8 > 0:
            ncols_8 = min(4, n_models_8)
            nrows_8 = (n_models_8 + ncols_8 - 1) // ncols_8
            fig, axes = plt.subplots(
                nrows_8, ncols_8,
                figsize=(5 * ncols_8, 4 * nrows_8),
                squeeze=False,
            )
            for mi, mname in enumerate(model_names_8):
                row_i, col_i = divmod(mi, ncols_8)
                ax = axes[row_i][col_i]
                try:
                    dec_df = compute_decile_analysis(
                        bundle.y_test, trained_models[mname]["test_preds"]
                    )
                    dec_x = range(1, len(dec_df) + 1)
                    ax.bar(dec_x, dec_df.get("actual_mean", dec_df.iloc[:, 0]),
                           color="#9E9E9E", alpha=0.7, label="Actual")
                    ax.bar(dec_x, dec_df.get("pred_mean", dec_df.iloc[:, 1]),
                           color=_family_color(mname), alpha=0.7, label="Predicted")
                    if "ae_ratio" in dec_df.columns:
                        ax2 = ax.twinx()
                        ax2.plot(dec_x, dec_df["ae_ratio"],
                                 color=C_RED, marker="o", ms=4, lw=1.5, label="A/E")
                        ax2.axhline(1.0, color=C_RED, ls="--", lw=0.8)
                        ax2.set_ylabel("A/E", fontsize=8)
                        ax2.set_ylim(0.5, 1.5)
                except Exception:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes)
                ax.set_title(mname, fontsize=9, fontweight="bold")
                ax.set_xlabel("Decile")
                ax.set_ylabel("Mean Premium")
                ax.legend(fontsize=7)

            # Hide unused subplots
            for mi in range(n_models_8, nrows_8 * ncols_8):
                row_i, col_i = divmod(mi, ncols_8)
                axes[row_i][col_i].set_visible(False)

            fig.suptitle("Decile A/E Calibration — All Models", fontweight="bold")
            fig.tight_layout()
            fig.savefig(fig_dir / "fig_dl_08_calibration_deciles.png", dpi=DPI)
            plt.close(fig)
            log.info("    Saved fig_dl_08_calibration_deciles.png")
    except Exception as exc:
        log.warning("  Figure 8 failed: %s", exc)

    # -----------------------------------------------------------------
    # Figure 9: Actual vs predicted scatter
    # -----------------------------------------------------------------
    try:
        log.info("  Figure 9: Actual vs predicted")
        n_scatter = 2000
        rng_s = np.random.default_rng(config.seed + 1)
        scatter_idx = rng_s.choice(len(bundle.y_test), size=min(n_scatter, len(bundle.y_test)), replace=False)
        y_scatter = bundle.y_test[scatter_idx]

        model_names_9 = list(trained_models.keys())
        n_models_9 = len(model_names_9)
        if n_models_9 > 0:
            ncols_9 = min(3, n_models_9)
            nrows_9 = (n_models_9 + ncols_9 - 1) // ncols_9
            fig, axes = plt.subplots(
                nrows_9, ncols_9,
                figsize=(5 * ncols_9, 4 * nrows_9),
                squeeze=False,
            )
            for mi, mname in enumerate(model_names_9):
                row_i, col_i = divmod(mi, ncols_9)
                ax = axes[row_i][col_i]
                p_scatter = trained_models[mname]["test_preds"][scatter_idx]
                ax.scatter(y_scatter, p_scatter, s=4, alpha=0.3,
                           color=_family_color(mname))
                # 45-degree reference
                lim = max(y_scatter.max(), p_scatter.max())
                ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="y=x")
                # OLS fit
                try:
                    coeffs_ols = np.polyfit(y_scatter, p_scatter, 1)
                    x_fit = np.linspace(0, lim, 100)
                    y_fit = np.polyval(coeffs_ols, x_fit)
                    r2 = float(np.corrcoef(y_scatter, p_scatter)[0, 1] ** 2)
                    ax.plot(x_fit, y_fit, color=C_RED, lw=1.2,
                            label=f"OLS slope={coeffs_ols[0]:.2f} R²={r2:.3f}")
                except Exception:
                    pass
                ax.set_title(mname, fontsize=9, fontweight="bold")
                ax.set_xlabel("Actual Premium")
                ax.set_ylabel("Predicted Premium")
                ax.legend(fontsize=7)

            for mi in range(n_models_9, nrows_9 * ncols_9):
                row_i, col_i = divmod(mi, ncols_9)
                axes[row_i][col_i].set_visible(False)

            fig.suptitle("Actual vs Predicted (subsample=2000)", fontweight="bold")
            fig.tight_layout()
            fig.savefig(fig_dir / "fig_dl_09_actual_vs_predicted.png", dpi=DPI)
            plt.close(fig)
            log.info("    Saved fig_dl_09_actual_vs_predicted.png")
    except Exception as exc:
        log.warning("  Figure 9 failed: %s", exc)

    # -----------------------------------------------------------------
    # Figure 10: Ensemble weights
    # -----------------------------------------------------------------
    try:
        log.info("  Figure 10: Ensemble weights")
        ens_res = results.get("stacked_ensemble", {})
        base_weights = ens_res.get("base_weights", {})
        if base_weights:
            sorted_weights = sorted(base_weights.items(), key=lambda x: x[1], reverse=True)
            labels_10 = [w[0] for w in sorted_weights]
            vals_10 = [w[1] for w in sorted_weights]
            colors_10 = [_family_color(l) for l in labels_10]
            fig, ax = plt.subplots(figsize=(8, max(3, len(labels_10) * 0.5 + 1)))
            bars_10 = ax.barh(labels_10[::-1], vals_10[::-1], color=colors_10[::-1])
            for bar, val in zip(bars_10, vals_10[::-1]):
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=9)
            ax.set_xlabel("Weight (normalised)")
            ax.set_title("Stacked Ensemble — Base Learner Weights", fontweight="bold")
            ax.set_xlim(0, max(vals_10) * 1.2)
            fig.tight_layout()
            fig.savefig(fig_dir / "fig_dl_10_ensemble_weights.png", dpi=DPI)
            plt.close(fig)
            log.info("    Saved fig_dl_10_ensemble_weights.png")
    except Exception as exc:
        log.warning("  Figure 10 failed: %s", exc)

    # -----------------------------------------------------------------
    # Figure 11: Ensemble variance across seeds
    # -----------------------------------------------------------------
    try:
        log.info("  Figure 11: Ensemble variance")
        dl_archs_ens = [
            name for name in ["cann", "cann_gbm", "ft_transformer", "tabm", "localglmnet", "drn"]
            if name in results and "error" not in results[name]
            and len(results[name].get("ensemble_results", [])) > 1
        ]
        if dl_archs_ens:
            n_deciles = 10
            decile_edges = np.percentile(bundle.y_test, np.linspace(0, 100, n_deciles + 1))
            decile_labels = np.digitize(bundle.y_test, decile_edges[1:-1])

            fig, axes = plt.subplots(
                1, len(dl_archs_ens),
                figsize=(5 * len(dl_archs_ens), 5),
                squeeze=False,
            )
            for ai, arch in enumerate(dl_archs_ens):
                ax = axes[0][ai]
                ens_preds = np.stack(
                    [tr.test_preds for tr in results[arch]["ensemble_results"]], axis=0
                )  # (n_members, n_test)
                pred_std = ens_preds.std(axis=0)

                std_by_decile = [
                    pred_std[decile_labels == d] for d in range(n_deciles)
                ]
                std_by_decile = [v for v in std_by_decile if len(v) > 0]
                ax.boxplot(std_by_decile, patch_artist=True,
                           boxprops=dict(facecolor=_family_color(arch), alpha=0.6))
                ax.set_title(f"{arch}\nPrediction Std by Decile", fontsize=9)
                ax.set_xlabel("Premium Decile")
                ax.set_ylabel("Prediction Std (£)")

            fig.suptitle("Ensemble Member Prediction Variance", fontweight="bold")
            fig.tight_layout()
            fig.savefig(fig_dir / "fig_dl_11_ensemble_variance.png", dpi=DPI)
            plt.close(fig)
            log.info("    Saved fig_dl_11_ensemble_variance.png")
    except Exception as exc:
        log.warning("  Figure 11 failed: %s", exc)

    # -----------------------------------------------------------------
    # Figure 12: Partial dependence plots (top 6 continuous features)
    # -----------------------------------------------------------------
    try:
        log.info("  Figure 12: Partial dependence plots")
        # Identify top 6 continuous features from CatBoost importance
        cb_interp = interp_results.get("catboost", {})
        imp_src = cb_interp.get("feature_importance") or cb_interp.get("mean_abs_shap", {})
        cont_importance = {
            k: v for k, v in imp_src.items() if k in bundle.continuous_feature_names
        }
        top6_cont = [
            f[0] for f in sorted(cont_importance.items(), key=lambda x: x[1], reverse=True)
        ][:6]
        if not top6_cont:
            top6_cont = bundle.continuous_feature_names[:6]

        if top6_cont and trained_models:
            pdp_models_name = ["catboost"] + [
                a for a in ["cann", "cann_gbm", "ft_transformer", "tabm", "localglmnet", "drn"]
                if a in trained_models
            ]
            pdp_models_name = pdp_models_name[:2]  # Limit to 2 for speed

            fig, axes = plt.subplots(
                len(top6_cont), len(pdp_models_name),
                figsize=(5 * len(pdp_models_name), 3 * len(top6_cont)),
                squeeze=False,
            )

            for fi, feat in enumerate(top6_cont):
                feat_idx = bundle.continuous_feature_names.index(feat) if feat in bundle.continuous_feature_names else -1
                if feat_idx < 0:
                    continue

                feat_vals_raw = bundle.X_train_cont[:, feat_idx]
                grid = np.linspace(
                    np.percentile(feat_vals_raw, 5),
                    np.percentile(feat_vals_raw, 95),
                    30,
                )

                for mi, mname in enumerate(pdp_models_name):
                    ax = axes[fi][mi]
                    try:
                        # CatBoost PDP via sklearn
                        if mname == "catboost" and HAS_CATBOOST:
                            from sklearn.inspection import partial_dependence as sk_pdp

                            class _CBWrapper:
                                """Sklearn-compatible CatBoost wrapper for PDP."""
                                def __init__(self, cb: Any) -> None:
                                    self.cb = cb
                                    self.feature_names_in_ = np.array(
                                        bundle.continuous_feature_names
                                        + bundle.categorical_feature_names
                                    )

                                def predict(self, X: np.ndarray) -> np.ndarray:
                                    return self.cb.predict(X)

                            X_pdp = np.concatenate(
                                [bundle.X_train_cont, bundle.X_train_cat.astype(np.float32)],
                                axis=1,
                            )
                            wrapper = _CBWrapper(trained_models[mname]["model"])
                            pdp_result = sk_pdp(wrapper, X_pdp, features=[feat_idx], kind="average")
                            pdp_vals = pdp_result["average"][0]
                            pdp_grid = pdp_result["grid_values"][0]
                            ax.plot(pdp_grid, pdp_vals, color=C_PRIMARY, lw=2)
                        elif mname in ["cann", "ft_transformer", "tabm"] and HAS_TORCH:
                            # Manual PDP for DL
                            ens_res_dl = results[mname].get("ensemble_results", [])
                            if not ens_res_dl:
                                raise ValueError("No ensemble results")
                            dl_model = ens_res_dl[0].model.to(torch.device("cpu"))
                            dl_model.eval()

                            X_base = torch.tensor(
                                bundle.X_train_cont[:200], dtype=torch.float32
                            )
                            X_cat_base = torch.tensor(
                                bundle.X_train_cat[:200], dtype=torch.long
                            )
                            glm_base = torch.tensor(
                                bundle.glm_train_preds[:200], dtype=torch.float32
                            )

                            pdp_vals_dl = []
                            for gv in grid:
                                X_g = X_base.clone()
                                X_g[:, feat_idx] = float(gv)
                                with torch.no_grad():
                                    p_g, _ = dl_model(X_g, X_cat_base, glm_base)
                                pdp_vals_dl.append(float(p_g.mean()))
                            ax.plot(grid, pdp_vals_dl, color=_family_color(mname), lw=2)
                        else:
                            raise ValueError("Unsupported model type for PDP")
                    except Exception as pdp_exc:
                        ax.text(0.5, 0.5, f"PDP N/A\n{pdp_exc}",
                                ha="center", va="center", transform=ax.transAxes, fontsize=7)

                    if fi == 0:
                        ax.set_title(mname, fontsize=9, fontweight="bold")
                    if mi == 0:
                        ax.set_ylabel(feat, fontsize=8)
                    ax.set_xlabel("Feature value (std)")

            fig.suptitle("Partial Dependence — Top 6 Continuous Features", fontweight="bold")
            fig.tight_layout()
            fig.savefig(fig_dir / "fig_dl_12_pdp_top6.png", dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            log.info("    Saved fig_dl_12_pdp_top6.png")
    except Exception as exc:
        log.warning("  Figure 12 failed: %s", exc)

    # -----------------------------------------------------------------
    # Figure 13: Monotonicity compliance check
    # -----------------------------------------------------------------
    try:
        log.info("  Figure 13: Monotonicity check")
        mono_feats = [
            f for f in MONOTONE_CONSTRAINTS
            if f in bundle.continuous_feature_names
        ][:5]

        if mono_feats and trained_models:
            n_bins = 10
            fig, axes = plt.subplots(
                len(mono_feats), 1,
                figsize=(8, 3 * len(mono_feats)),
                squeeze=False,
            )
            for fi, feat in enumerate(mono_feats):
                ax = axes[fi][0]
                feat_idx = bundle.continuous_feature_names.index(feat)
                direction = MONOTONE_CONSTRAINTS[feat]

                feat_vals = bundle.X_test_cont[:, feat_idx]
                bins = np.percentile(feat_vals, np.linspace(0, 100, n_bins + 1))
                bin_labels = (bins[:-1] + bins[1:]) / 2
                bin_idx = np.digitize(feat_vals, bins[1:-1])

                for mname, res in list(trained_models.items())[:4]:
                    preds = res["test_preds"]
                    bin_means = np.array([
                        preds[bin_idx == b].mean() if (bin_idx == b).any() else np.nan
                        for b in range(n_bins)
                    ])
                    ax.plot(bin_labels, bin_means, marker="o", ms=4,
                            label=mname, color=_family_color(mname), lw=1.5)

                arrow = "↑" if direction > 0 else "↓"
                ax.set_title(f"{feat}  (expected: {arrow})", fontsize=9, fontweight="bold")
                ax.set_xlabel("Feature value (std)")
                ax.set_ylabel("Mean predicted premium (£)")
                ax.legend(fontsize=7)

            fig.suptitle("Monotonicity Compliance Check", fontweight="bold")
            fig.tight_layout()
            fig.savefig(fig_dir / "fig_dl_13_monotonicity_check.png", dpi=DPI)
            plt.close(fig)
            log.info("    Saved fig_dl_13_monotonicity_check.png")
    except Exception as exc:
        log.warning("  Figure 13 failed: %s", exc)

    # -----------------------------------------------------------------
    # Figure 14: CV stability
    # -----------------------------------------------------------------
    try:
        log.info("  Figure 14: CV stability")
        cb_cv = cv_results.get("catboost", {})
        fold_recs = cb_cv.get("fold_records", [])
        if fold_recs:
            fold_nums = [r["fold"] for r in fold_recs]
            gini_tr = [r["gini_train"] for r in fold_recs]
            gini_va = [r["gini_val"] for r in fold_recs]

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(fold_nums, gini_tr, "o-", color=C_PRIMARY, lw=2, label="Train Gini")
            ax.plot(fold_nums, gini_va, "s-", color=C_RED, lw=2, label="Val Gini")
            mean_val = np.mean(gini_va)
            ax.axhline(mean_val, color=C_GOLD, ls="--", lw=1.5,
                       label=f"Val mean={mean_val:.4f}")
            ax.set_xticks(fold_nums)
            ax.set_xlabel("Fold")
            ax.set_ylabel("Gini Coefficient")
            ax.set_title("CatBoost CV Stability", fontweight="bold")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / "fig_dl_14_cv_stability.png", dpi=DPI)
            plt.close(fig)
            log.info("    Saved fig_dl_14_cv_stability.png")
    except Exception as exc:
        log.warning("  Figure 14 failed: %s", exc)

    # -----------------------------------------------------------------
    # Figure 15: Model complexity vs test Gini
    # -----------------------------------------------------------------
    try:
        log.info("  Figure 15: Model complexity vs Gini")
        complexity_rows = []
        for name, res in trained_models.items():
            n_p = _count_model_params(res, name)
            gini_te = res.get("metrics_test", {}).get("gini", 0)
            if isinstance(n_p, int) and n_p > 0:
                complexity_rows.append({"model": name, "n_params": n_p, "gini_test": gini_te})

        if complexity_rows:
            fig, ax = plt.subplots(figsize=(7, 5))
            for row in complexity_rows:
                ax.scatter(
                    row["n_params"], row["gini_test"],
                    s=100, color=_family_color(row["model"]),
                    zorder=5,
                )
                ax.annotate(
                    row["model"],
                    (row["n_params"], row["gini_test"]),
                    textcoords="offset points",
                    xytext=(5, 4),
                    fontsize=8,
                )
            ax.set_xscale("log")
            ax.set_xlabel("Number of Parameters / Trees (log scale)")
            ax.set_ylabel("Test Gini")
            ax.set_title("Model Complexity vs Test Gini", fontweight="bold")
            fig.tight_layout()
            fig.savefig(fig_dir / "fig_dl_15_model_complexity.png", dpi=DPI)
            plt.close(fig)
            log.info("    Saved fig_dl_15_model_complexity.png")
    except Exception as exc:
        log.warning("  Figure 15 failed: %s", exc)

    # -----------------------------------------------------------------
    # Figure 16: Deviance residuals distribution
    # -----------------------------------------------------------------
    try:
        log.info("  Figure 16: Deviance residuals")
        fig, axes = plt.subplots(
            1, min(4, len(trained_models)),
            figsize=(4 * min(4, len(trained_models)), 4),
            squeeze=False,
        )
        for mi, (mname, res) in enumerate(list(trained_models.items())[:4]):
            ax = axes[0][mi]
            preds = res["test_preds"]
            y_t = bundle.y_test
            # Gamma deviance residual: sign(y-p) * sqrt(unit_deviance)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_safe = np.maximum(preds, 1.0)
                y_safe = np.maximum(y_t, 1.0)
                unit_dev = 2.0 * (-np.log(y_safe / p_safe) + (y_safe - p_safe) / p_safe)
                dev_resid = np.sign(y_safe - p_safe) * np.sqrt(np.maximum(unit_dev, 0))

            ax.hist(dev_resid, bins=60, color=_family_color(mname),
                    alpha=0.75, edgecolor="white")
            ax.axvline(0, color=C_RED, lw=1.5, ls="--")
            ax.set_title(mname, fontsize=9, fontweight="bold")
            ax.set_xlabel("Deviance Residual")
            if mi == 0:
                ax.set_ylabel("Count")

        fig.suptitle("Gamma Deviance Residuals", fontweight="bold")
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_dl_16_residual_distribution.png", dpi=DPI)
        plt.close(fig)
        log.info("    Saved fig_dl_16_residual_distribution.png")
    except Exception as exc:
        log.warning("  Figure 16 failed: %s", exc)

    # -----------------------------------------------------------------
    # Figure 17: Combined dashboard (2x3 summary grid)
    # -----------------------------------------------------------------
    try:
        log.info("  Figure 17: Combined dashboard")
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle("DL Pipeline Summary Dashboard", fontsize=14, fontweight="bold", y=0.98)

        # Panel 1: Gini comparison (top-left)
        ax1 = fig.add_subplot(2, 3, 1)
        rows_dash = comparison_table[:8] if comparison_table else []
        if rows_dash:
            m_labels = [r["model"] for r in rows_dash]
            m_ginis = [r.get("gini_test", 0) for r in rows_dash]
            x_d = range(len(m_labels))
            ax1.bar(x_d, m_ginis,
                    color=[_family_color(m) for m in m_labels], alpha=0.85)
            ax1.set_xticks(x_d)
            ax1.set_xticklabels(m_labels, rotation=30, ha="right", fontsize=7)
            ax1.set_ylabel("Test Gini")
            ax1.set_title("Gini Comparison", fontweight="bold")

        # Panel 2: Lorenz curves (top-middle)
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
        for mname, res in list(trained_models.items())[:5]:
            try:
                xv, yv = _lorenz_curve(bundle.y_test, res["test_preds"])
                ax2.plot(xv, yv, label=mname, color=_family_color(mname), lw=1.5)
            except Exception:
                pass
        ax2.set_title("Lorenz Curves", fontweight="bold")
        ax2.legend(fontsize=6, loc="upper left")
        ax2.set_xlabel("Cumulative policies")
        ax2.set_ylabel("Cumulative premium")

        # Panel 3: Best model calibration (top-right)
        ax3 = fig.add_subplot(2, 3, 3)
        best_name = comparison.get("best_model")
        if best_name and best_name in trained_models:
            try:
                dec_df = compute_decile_analysis(
                    bundle.y_test, trained_models[best_name]["test_preds"]
                )
                dec_x = range(1, len(dec_df) + 1)
                ax3.bar(dec_x, dec_df.get("actual_mean", dec_df.iloc[:, 0]),
                        color="#9E9E9E", alpha=0.7, label="Actual")
                ax3.bar(dec_x, dec_df.get("pred_mean", dec_df.iloc[:, 1]),
                        color=_family_color(best_name), alpha=0.7, label="Pred")
                ax3.set_title(f"Best Model Calibration\n({best_name})", fontweight="bold")
                ax3.legend(fontsize=7)
            except Exception:
                ax3.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax3.transAxes)

        # Panel 4: Best model scatter (bottom-left)
        ax4 = fig.add_subplot(2, 3, 4)
        if best_name and best_name in trained_models:
            rng_d = np.random.default_rng(config.seed + 2)
            idx_d = rng_d.choice(len(bundle.y_test), size=min(1000, len(bundle.y_test)), replace=False)
            y_d = bundle.y_test[idx_d]
            p_d = trained_models[best_name]["test_preds"][idx_d]
            ax4.scatter(y_d, p_d, s=3, alpha=0.3, color=_family_color(best_name))
            lim_d = max(y_d.max(), p_d.max())
            ax4.plot([0, lim_d], [0, lim_d], "k--", lw=0.8)
            ax4.set_xlabel("Actual")
            ax4.set_ylabel("Predicted")
            ax4.set_title(f"Actual vs Pred\n({best_name})", fontweight="bold")

        # Panel 5: Ensemble weights (bottom-middle)
        ax5 = fig.add_subplot(2, 3, 5)
        ens_res_d = results.get("stacked_ensemble", {})
        bw = ens_res_d.get("base_weights", {})
        if bw:
            sorted_bw = sorted(bw.items(), key=lambda x: x[1], reverse=True)
            l_bw = [w[0] for w in sorted_bw]
            v_bw = [w[1] for w in sorted_bw]
            ax5.barh(l_bw[::-1], v_bw[::-1],
                     color=[_family_color(l) for l in l_bw[::-1]])
            ax5.set_xlabel("Weight")
            ax5.set_title("Ensemble Weights", fontweight="bold")

        # Panel 6: Model complexity vs Gini (bottom-right)
        ax6 = fig.add_subplot(2, 3, 6)
        for mname, res in trained_models.items():
            n_p = _count_model_params(res, mname)
            gini_te = res.get("metrics_test", {}).get("gini", 0)
            if isinstance(n_p, int) and n_p > 0:
                ax6.scatter(n_p, gini_te, s=80, color=_family_color(mname), zorder=5)
                ax6.annotate(mname, (n_p, gini_te), textcoords="offset points",
                             xytext=(4, 3), fontsize=7)
        ax6.set_xscale("log")
        ax6.set_xlabel("Params (log)")
        ax6.set_ylabel("Test Gini")
        ax6.set_title("Complexity vs Gini", fontweight="bold")

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(fig_dir / "fig_dl_17_combined_dashboard.png", dpi=DPI)
        plt.close(fig)
        log.info("    Saved fig_dl_17_combined_dashboard.png")
    except Exception as exc:
        log.warning("  Figure 17 failed: %s", exc)

    # -----------------------------------------------------------------
    # HTML Dashboards (Plotly, optional)
    # -----------------------------------------------------------------
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Dashboard 1: Model comparison
        try:
            rows_html = comparison_table[:8] if comparison_table else []
            if rows_html:
                fig_html = go.Figure()
                for row in rows_html:
                    fig_html.add_trace(
                        go.Bar(
                            name=row["model"],
                            x=[row["model"]],
                            y=[row.get("gini_test", 0)],
                            text=[f"{row.get('gini_test', 0):.4f}"],
                            textposition="auto",
                        )
                    )
                fig_html.update_layout(
                    title="DL Pipeline — Model Gini Comparison",
                    yaxis_title="Test Gini",
                    showlegend=True,
                )
                html_path_1 = Path(config.output_dir) / "dashboard_dl_models.html"
                fig_html.write_html(str(html_path_1))
                log.info("  HTML dashboard saved to %s", html_path_1)
        except Exception as exc:
            log.warning("  HTML dashboard 1 failed: %s", exc)

        # Dashboard 2: Interpretability
        try:
            all_interp_feat: Dict[str, Dict[str, float]] = {}
            for mname, artefact in interp_results.items():
                src = (
                    artefact.get("mean_abs_shap")
                    or artefact.get("mean_abs_attributions")
                    or artefact.get("perm_importance")
                    or artefact.get("feature_importance")
                    or {}
                )
                if src:
                    all_interp_feat[mname] = src

            if all_interp_feat:
                fig_html2 = go.Figure()
                for mname, feat_imp in all_interp_feat.items():
                    top_items = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:15]
                    fig_html2.add_trace(
                        go.Bar(
                            name=mname,
                            x=[f[0] for f in top_items],
                            y=[f[1] for f in top_items],
                        )
                    )
                fig_html2.update_layout(
                    title="DL Pipeline — Feature Importance / Attribution",
                    barmode="group",
                    xaxis_tickangle=-45,
                )
                html_path_2 = Path(config.output_dir) / "dashboard_dl_interpretability.html"
                fig_html2.write_html(str(html_path_2))
                log.info("  HTML interpretability dashboard saved to %s", html_path_2)
        except Exception as exc:
            log.warning("  HTML dashboard 2 failed: %s", exc)

    except ImportError:
        log.warning("  plotly not installed — skipping HTML dashboards")

    log.info("  Visualizations complete. Saved to %s", fig_dir)
