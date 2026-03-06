"""Configuration, setup, and CLI argument parsing for the DL pipeline."""
from __future__ import annotations

# Must be set BEFORE any imports to avoid OpenMP conflicts on macOS.
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional heavy dependencies — degrade gracefully with HAS_X flags
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, random_split

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = None  # type: ignore[assignment]
    random_split = None  # type: ignore[assignment]
    HAS_TORCH = False

try:
    import catboost
    from catboost import CatBoostRegressor, Pool

    HAS_CATBOOST = True
except ImportError:
    catboost = None  # type: ignore[assignment]
    CatBoostRegressor = None  # type: ignore[assignment]
    Pool = None  # type: ignore[assignment]
    HAS_CATBOOST = False

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    xgb = None  # type: ignore[assignment]
    HAS_XGBOOST = False

try:
    import optuna

    HAS_OPTUNA = True
except ImportError:
    optuna = None  # type: ignore[assignment]
    HAS_OPTUNA = False

try:
    import captum

    HAS_CAPTUM = True
except ImportError:
    captum = None  # type: ignore[assignment]
    HAS_CAPTUM = False

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import tabm  # type: ignore[import]

    HAS_TABM = True
except ImportError:
    tabm = None  # type: ignore[assignment]
    HAS_TABM = False

# ---------------------------------------------------------------------------
# Imports from the GLM pipeline
# ---------------------------------------------------------------------------
from build_glm import (
    BASE_LEVELS,
    CATEGORICAL_FACTORS,
    GLMConfig,
    align_test_matrix,
    cap_premium,
    compute_gini,
    consolidate_categoricals,
    fit_gamma_glm,
    load_data,
    prepare_design_matrix,
)

# ---------------------------------------------------------------------------
# Imports from the GBM pipeline
# ---------------------------------------------------------------------------
from build_gbm import (
    DERIVED_CONTINUOUS,
    GLM_HYBRID_FACTORS,
    MONOTONE_CONSTRAINTS,
    NATIVE_CATEGORICALS,
    PARSIMONIOUS_FEATURES,
    RAW_CONTINUOUS,
    GBMConfig,
    _clamp_predictions,
    _compute_metrics,
    _lorenz_curve,
    compute_decile_analysis,
    compute_gamma_deviance,
    load_and_prepare_data,
    prepare_gbm_features,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour constants (extends GBM palette with DL-specific colours)
# ---------------------------------------------------------------------------
C_PRIMARY = "#1E3A5F"
C_ACCENT = "#2E6B9E"
C_GREEN = "#1D9A6C"
C_GOLD = "#C8963E"
C_RED = "#DC2626"
C_PURPLE = "#7C3AED"
C_TEAL = "#0D9488"

# ---------------------------------------------------------------------------
# DL Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class DLConfig:
    """Configuration for the deep learning and advanced modelling pipeline.

    Attributes:
        input_path: Path to the GLM-ready CSV file.
        output_dir: Directory for output artefacts.
        seed: Master random seed for reproducibility across all frameworks.
        cap_percentile: Premium winsorisation percentile (0-100).
        cap_value: Hard premium cap override in GBP (None = use percentile).
        n_tuning_trials: Number of Optuna trials per architecture.
        cv_folds: Number of cross-validation folds for GBM tuning.
        quick: Subsample 1000 training rows and reduce epochs for rapid
            iteration. Sets n_ensemble=1, catboost_iterations=200, epochs=50.
        skip_tuning: Skip Optuna tuning and use architecture defaults.
        skip_interpretability: Skip Captum attribution computation.
        architectures: List of architectures to train. Valid values are
            "catboost", "xgboost", "cann", "cann_gbm", "ft_transformer",
            "tabm", "localglmnet", "drn".
        epochs: Maximum training epochs for DL models.
        patience: Early stopping patience (epochs without val improvement).
        batch_size: Mini-batch size for DL training.
        device: Compute device — "auto" detects MPS > CUDA > CPU.
        val_fraction: Fraction of training data held out for DL validation.
        n_ensemble: Number of seed-varied DL models to average per architecture.
        catboost_iterations: Maximum boosting rounds for CatBoost.
        mono_lambda: Weight of monotonicity penalty in DL training loss.
    """

    input_path: str = "data_to_be_cleaned/net/net_glm_ready.csv"
    output_dir: str = "data_to_be_cleaned/net/dl_results"
    seed: int = 42
    cap_percentile: float = 99.5
    cap_value: Optional[float] = None
    n_tuning_trials: int = 30
    cv_folds: int = 5
    quick: bool = False
    skip_tuning: bool = False
    skip_interpretability: bool = False
    architectures: List[str] = field(
        default_factory=lambda: [
            "catboost",
            "xgboost",
            "cann",
            "ft_transformer",
            "tabm",
        ]
    )
    epochs: int = 300
    patience: int = 30
    batch_size: int = 512
    device: str = "auto"
    val_fraction: float = 0.15
    n_ensemble: int = 3
    catboost_iterations: int = 2000
    mono_lambda: float = 0.1


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def parse_args() -> DLConfig:
    """Parse command-line arguments and return a DLConfig.

    Returns:
        Populated DLConfig instance from CLI flags.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Deep learning & advanced modelling pipeline for UK motor net premium: "
            "CatBoost, XGBoost, CANN, CANN-GBM, FT-Transformer, TabM, LocalGLMnet, DRN."
        ),
    )
    parser.add_argument(
        "--input",
        default="data_to_be_cleaned/net/net_glm_ready.csv",
        help="Path to GLM-ready CSV (default: data_to_be_cleaned/net/net_glm_ready.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="data_to_be_cleaned/net/dl_results",
        help="Output directory for artefacts (default: data_to_be_cleaned/net/dl_results)",
    )
    parser.add_argument(
        "--cap",
        type=float,
        default=None,
        help="Hard premium cap in GBP (default: use percentile)",
    )
    parser.add_argument(
        "--cap-percentile",
        type=float,
        default=99.5,
        help="Percentile for premium cap (default: 99.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed (default: 42)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Optuna trials per architecture (default: 30)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Cross-validation folds for GBM tuning (default: 5)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Subsample 1000 training rows; reduce epochs/ensemble for rapid iteration",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip Optuna tuning — use architecture defaults",
    )
    parser.add_argument(
        "--skip-interpretability",
        action="store_true",
        help="Skip Captum attribution computation",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["catboost", "xgboost", "cann", "ft_transformer", "tabm"],
        choices=["catboost", "xgboost", "cann", "cann_gbm", "ft_transformer", "tabm", "localglmnet", "drn"],
        help="Architectures to train (default: catboost xgboost cann ft_transformer tabm)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Maximum DL training epochs (default: 300)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience in epochs (default: 30)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Mini-batch size for DL training (default: 512)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help='Compute device: "auto" detects MPS > CUDA > CPU (default: auto)',
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of training data for DL validation (default: 0.15)",
    )
    parser.add_argument(
        "--n-ensemble",
        type=int,
        default=3,
        help="Number of seeds to average per DL architecture (default: 3)",
    )
    parser.add_argument(
        "--catboost-iterations",
        type=int,
        default=2000,
        help="Maximum CatBoost boosting rounds (default: 2000)",
    )
    parser.add_argument(
        "--mono-lambda",
        type=float,
        default=0.1,
        help="Weight of monotonicity penalty in DL loss (default: 0.1)",
    )

    args = parser.parse_args()

    return DLConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        seed=args.seed,
        cap_percentile=args.cap_percentile,
        cap_value=args.cap,
        n_tuning_trials=args.n_trials,
        cv_folds=args.cv_folds,
        quick=args.quick,
        skip_tuning=args.skip_tuning,
        skip_interpretability=args.skip_interpretability,
        architectures=args.architectures,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        device=args.device,
        val_fraction=args.val_fraction,
        n_ensemble=args.n_ensemble,
        catboost_iterations=args.catboost_iterations,
        mono_lambda=args.mono_lambda,
    )
