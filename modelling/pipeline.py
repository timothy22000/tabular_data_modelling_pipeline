"""Top-level pipeline orchestration — runs Sections 1-13 in order."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict

from .config import DLConfig, log, parse_args
from .data import load_and_prepare_dl_data, prepare_dl_features
from .orchestration import train_all_models
from .ensemble import build_stacked_ensemble
from .evaluation import evaluate_all_models
from .interpretability import run_interpretability
from .cv import run_cross_validation
from .comparison import compare_with_existing_models
from .visualization import generate_dl_visualizations
from .output import save_all_outputs


def run_dl_pipeline(config: DLConfig) -> Dict[str, Any]:
    """Orchestrate the full DL pipeline: Sections 1-13 in order.

    Each section is timed and errors are logged without aborting the run
    (except fatal errors in data loading / feature preparation).

    Args:
        config: DL pipeline configuration.

    Returns:
        Comprehensive results dictionary with keys:
            - "results": Architecture name -> training result dict.
            - "bundle": DLFeatureBundle.
            - "interp_results": Interpretability artefacts.
            - "cv_results": Cross-validation results.
            - "comparison": Full model comparison.
            - "section_times": Dict of section name -> elapsed seconds.
    """
    pipeline_start = time.time()
    section_times: Dict[str, float] = {}

    log.info("=" * 72)
    log.info("DL PIPELINE — START")
    log.info("  config.output_dir = %s", config.output_dir)
    log.info("  config.architectures = %s", config.architectures)
    log.info("  config.quick = %s", config.quick)
    log.info("  config.skip_tuning = %s", config.skip_tuning)
    log.info("  config.skip_interpretability = %s", config.skip_interpretability)
    log.info("=" * 72)

    # ------------------------------------------------------------------
    # Section 1: Data Loading
    # ------------------------------------------------------------------
    t0 = time.time()
    train_df, test_df, cap_value = load_and_prepare_dl_data(config)
    section_times["data_loading"] = time.time() - t0

    # ------------------------------------------------------------------
    # Section 2: Feature Preparation
    # ------------------------------------------------------------------
    t0 = time.time()
    bundle = prepare_dl_features(train_df, test_df, config)
    section_times["feature_preparation"] = time.time() - t0

    # ------------------------------------------------------------------
    # Section 6: Model Training (Sections 3-5 are definitions, not calls)
    # ------------------------------------------------------------------
    t0 = time.time()
    results = train_all_models(bundle, config)
    section_times["model_training"] = time.time() - t0

    # ------------------------------------------------------------------
    # Section 7: Stacked Ensemble
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        stacked = build_stacked_ensemble(results, bundle, config)
        results["stacked_ensemble"] = stacked
        results["stacked_ensemble"]["best_params"] = {}
    except Exception as exc:
        log.error("  Section 7 (Stacked Ensemble) failed: %s", exc)
        results["stacked_ensemble"] = {"error": str(exc)}
    section_times["stacked_ensemble"] = time.time() - t0

    # ------------------------------------------------------------------
    # Section 8: Evaluation & Diagnostics
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        eval_results = evaluate_all_models(results, bundle, config)
    except Exception as exc:
        log.error("  Section 8 (Evaluation) failed: %s", exc)
        eval_results = {}
    section_times["evaluation"] = time.time() - t0

    # ------------------------------------------------------------------
    # Section 9: Interpretability
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        interp_results = run_interpretability(results, bundle, config)
    except Exception as exc:
        log.error("  Section 9 (Interpretability) failed: %s", exc)
        interp_results = {}
    section_times["interpretability"] = time.time() - t0

    # ------------------------------------------------------------------
    # Section 10: Cross-Validation
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        cv_results = run_cross_validation(bundle, config)
    except Exception as exc:
        log.error("  Section 10 (Cross-Validation) failed: %s", exc)
        cv_results = {}
    section_times["cross_validation"] = time.time() - t0

    # ------------------------------------------------------------------
    # Section 11: Comparison with GLM & GBM
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        comparison = compare_with_existing_models(results, bundle, config)
    except Exception as exc:
        log.error("  Section 11 (Comparison) failed: %s", exc)
        comparison = {
            "comparison_table": [],
            "best_model": None,
            "double_lift": None,
            "glm_gini": None,
        }
    section_times["comparison"] = time.time() - t0

    # ------------------------------------------------------------------
    # Section 12: Visualizations
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        generate_dl_visualizations(
            results, bundle, interp_results, cv_results, comparison, config
        )
    except Exception as exc:
        log.error("  Section 12 (Visualizations) failed: %s", exc)
    section_times["visualizations"] = time.time() - t0

    # ------------------------------------------------------------------
    # Section 13: Output Saving
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        save_all_outputs(results, bundle, interp_results, cv_results, comparison, config)
    except Exception as exc:
        log.error("  Section 13 (Output Saving) failed: %s", exc)
    section_times["output_saving"] = time.time() - t0

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    total_elapsed = time.time() - pipeline_start
    log.info("")
    log.info("=" * 72)
    log.info("DL PIPELINE — COMPLETE")
    log.info("  Total elapsed: %.1fs (%.1f min)", total_elapsed, total_elapsed / 60)
    log.info("  Section breakdown:")
    for section, elapsed in section_times.items():
        log.info("    %-25s  %.1fs", section, elapsed)
    log.info("=" * 72)

    best_model = comparison.get("best_model")
    if best_model and best_model in results and "error" not in results[best_model]:
        best_gini = results[best_model].get("metrics_test", {}).get("gini", 0)
        log.info("  Best model: %s  (Test Gini=%.4f)", best_model, best_gini)

    return {
        "results": results,
        "bundle": bundle,
        "eval_results": eval_results,
        "interp_results": interp_results,
        "cv_results": cv_results,
        "comparison": comparison,
        "section_times": section_times,
        "total_elapsed": total_elapsed,
    }


def main() -> None:
    """Entry point for the DL pipeline CLI.

    Parses command-line arguments, configures logging, runs the full
    pipeline, and logs the total wall-clock time.

    Usage examples::

        python train.py
        python train.py --quick --skip-tuning
        python train.py --architectures catboost xgboost
        python train.py --n-trials 50 --epochs 200 --device cpu
        python train.py --skip-tuning --skip-interpretability
    """
    # Re-configure logging to match GBM pipeline format
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    config = parse_args()

    log.info("Deep Learning & Advanced Modelling Pipeline")
    log.info("  Input:      %s", config.input_path)
    log.info("  Output dir: %s", config.output_dir)
    log.info("  Seed:       %d", config.seed)
    log.info("  Quick mode: %s", config.quick)
    log.info("  Device:     %s", config.device)

    pipeline_results = run_dl_pipeline(config)

    total = pipeline_results.get("total_elapsed", 0.0)
    log.info(
        "Pipeline finished in %.1fs (%.1f min)",
        total, total / 60,
    )


if __name__ == "__main__":
    main()
