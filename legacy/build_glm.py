#!/usr/bin/env python3
"""Benchmark Net Premium Gamma GLM — Ageas Direct UK Motor Insurance.

Fits a Gamma GLM with log link to AD_POLPREMIUM using stepwise factor
selection, interaction testing, and full actuarial diagnostics.

Usage:
    python build_net_premium_glm.py
    python build_net_premium_glm.py --cap 10000
    python build_net_premium_glm.py --sensitivity
    python build_net_premium_glm.py --quick
"""

# =============================================================================
# Section 0: Setup
# =============================================================================

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2

# Optional matplotlib / seaborn — not strictly needed for core GLM logic but
# imported here so that visualisation sections appended later can rely on them.
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# statsmodels is the primary fitting backend.  sklearn GammaRegressor is used
# as a fallback when statsmodels is unavailable.
try:
    import statsmodels.api as sm
    from statsmodels.genmod.families import Gamma
    from statsmodels.genmod.families.links import Log as LogLink

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

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
# Constants
# ---------------------------------------------------------------------------

EXCLUDE_COLS: List[str] = [
    "AD_POLPREMIUM",
    "LOG_AD_POLPREMIUM",
    "AGG_SOURCE",
    "ANNUAL_TOPS_PRICE",
    "HSALE",
    "source_file",
    "SPLIT",
    "CREDIT_SCORE_MISSING",
]  # plus all *_IMPUTED columns detected at runtime

CANONICAL_REGIONS = {
    "SE", "WM", "SW", "Y", "S", "NW", "EM", "EA", "N", "OL", "NL", "L", "WH", "W",
}

BASE_LEVELS: Dict[str, str] = {
    "RISK_AREA": "SE",
    "CONVICTIONS_FLAG": "N",
    "COVER_TYPE": "COMP",
    "DD_DUQ": "N",
    "CLASSOFUSEDESC": "1",
    "NCDPROTECT": "FALSE",
    "AGE_BAND": "41-50",
    "ENGINE_SIZE_BAND": "2.0-2.9",
    "VEHICLE_VALUE_BAND": "<5K",
    "VEHICLE_AGE_BAND": "11+",
    "MILEAGE_K_BAND": "5-10",
    "CREDIT_SCORE_BAND": "400-499",
    "CLM_GROUP": "0",
    "FUEL_TYPE_CAT": "1",
}

INTERACTION_PAIRS: List[Tuple[str, str]] = [
    ("AGE_BAND", "COVER_TYPE"),
    ("AGE_BAND", "NCD_CAPPED"),
    ("VEHICLE_AGE_BAND", "VEHICLE_VALUE_BAND"),
    ("CREDIT_SCORE_BAND", "AGE_BAND"),
    ("MILEAGE_K_BAND", "COVER_TYPE"),
    ("RISK_AREA", "AGE_BAND"),
]

# Colour palette (used by appended visualisation sections)
C_PRIMARY = "#1E3A5F"
C_ACCENT = "#2B7A78"
C_GREEN = "#1D9A6C"
C_AMBER = "#F59E0B"
C_RED = "#DC2626"


@dataclass
class GLMConfig:
    """Configuration for the Gamma GLM pipeline.

    Attributes:
        input_path: Path to the GLM-ready CSV file.
        output_dir: Directory for output artefacts.
        seed: Random seed for reproducibility.
        cap_percentile: Premium winsorisation percentile (0–100).
        cap_value: Hard premium cap override (None = use percentile).
        run_interactions: Whether to test pre-specified interaction pairs.
        run_sensitivity: Whether to run sensitivity analysis (Sections 11–12).
        quick: Subsample 1000 training rows for rapid iteration.
        cv_folds: Number of cross-validation folds (0 = disabled, 5 = standard).
    """

    input_path: str = "data_to_be_cleaned/net/net_glm_ready.csv"
    output_dir: str = "data_to_be_cleaned/net/glm_results"
    seed: int = 42
    cap_percentile: float = 99.5
    cap_value: Optional[float] = None
    run_interactions: bool = True
    run_sensitivity: bool = False
    quick: bool = False
    cv_folds: int = 0


# =============================================================================
# Section 1: Data Loading & Preparation
# =============================================================================


def load_data(config: GLMConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load GLM-ready CSV, split train/test, exclude leakage columns.

    Args:
        config: Pipeline configuration.

    Returns:
        Tuple of (train_df, test_df) with leakage columns removed.
    """
    log.info("Loading data from: %s", config.input_path)
    df = pd.read_csv(config.input_path, low_memory=False)
    log.info("  Loaded %d rows, %d columns", len(df), df.shape[1])

    # Detect *_IMPUTED columns to exclude
    imputed_cols = [c for c in df.columns if c.endswith("_IMPUTED")]
    all_exclude = set(EXCLUDE_COLS) | set(imputed_cols)
    log.info("  Excluding %d leakage/meta columns", len(all_exclude))

    # Split on SPLIT column
    if "SPLIT" not in df.columns:
        raise ValueError("SPLIT column not found in dataset.")

    train_df = df[df["SPLIT"] == "TRAIN"].copy()
    test_df = df[df["SPLIT"] == "TEST"].copy()
    log.info("  Train rows: %d  |  Test rows: %d", len(train_df), len(test_df))

    # Drop excluded columns — keep target in frame for now (dropped later
    # when building design matrices)
    cols_to_drop = [c for c in all_exclude if c in df.columns and c != "AD_POLPREMIUM"]
    train_df = train_df.drop(columns=cols_to_drop, errors="ignore")
    test_df = test_df.drop(columns=cols_to_drop, errors="ignore")

    # Quick mode: subsample training set
    if config.quick:
        rng = np.random.default_rng(config.seed)
        idx = rng.choice(len(train_df), size=min(1000, len(train_df)), replace=False)
        train_df = train_df.iloc[idx].copy()
        log.info("  [quick] Subsampled to %d training rows", len(train_df))

    return train_df, test_df


def cap_premium(
    train_df: pd.DataFrame, config: GLMConfig
) -> Tuple[pd.DataFrame, float]:
    """Winsorise AD_POLPREMIUM at the configured percentile.

    Args:
        train_df: Training DataFrame containing AD_POLPREMIUM.
        config: Pipeline configuration with cap_percentile and cap_value.

    Returns:
        Tuple of (updated train_df, cap_value).
    """
    if config.cap_value is not None:
        cap = float(config.cap_value)
        log.info("  Using hard cap: £%.2f", cap)
    else:
        cap = float(np.percentile(train_df["AD_POLPREMIUM"].dropna(), config.cap_percentile))
        log.info("  P%.1f cap: £%.2f", config.cap_percentile, cap)

    n_capped = int((train_df["AD_POLPREMIUM"] > cap).sum())
    train_df = train_df.copy()
    train_df["AD_POLPREMIUM_CAPPED"] = train_df["AD_POLPREMIUM"].clip(upper=cap)
    log.info("  Rows capped: %d (%.2f%%)", n_capped, 100 * n_capped / len(train_df))

    return train_df, cap


# =============================================================================
# Section 2: Categorical Consolidation
# =============================================================================


def consolidate_risk_area(val: Any) -> str:
    """Map 347 OCR-variant risk area codes to ~14 canonical regions.

    Args:
        val: Raw risk area value (may be NaN, numeric prefix + code, etc.).

    Returns:
        Canonical region string or "OTHER".
    """
    if pd.isna(val):
        return "OTHER"
    val_str = str(val).strip().upper()
    if val_str in CANONICAL_REGIONS:
        return val_str
    # Strip leading digits/decimals: "1SE" -> "SE", "2.0NW" -> "NW"
    stripped = re.sub(r"^[\d.]+\s*", "", val_str)
    if stripped in CANONICAL_REGIONS:
        return stripped
    # Check if any canonical region is embedded anywhere in the string
    for region in sorted(CANONICAL_REGIONS, key=len, reverse=True):
        if region in val_str:
            return region
    return "OTHER"


def consolidate_convictions(val: Any) -> str:
    """Map 140 OCR-variant conviction flags to N/Y/OTHER.

    Args:
        val: Raw conviction flag value.

    Returns:
        "N", "Y", or "OTHER".
    """
    if pd.isna(val):
        return "OTHER"
    val_str = str(val).strip().upper()
    if val_str == "N":
        return "N"
    if val_str == "Y":
        return "Y"
    # Compound patterns: "1 NW N" -> N, "1 EA Y" -> Y
    if val_str.endswith(" N") or (val_str.endswith("N") and len(val_str) <= 3):
        return "N"
    if val_str.endswith(" Y") or (val_str.endswith("Y") and len(val_str) <= 3):
        return "Y"
    return "OTHER"


def consolidate_class_of_use(val: Any) -> str:
    """Map 115 OCR-variant class-of-use codes to ~8 groups.

    Args:
        val: Raw class-of-use value.

    Returns:
        Canonical class string or "OTHER".
    """
    if pd.isna(val):
        return "OTHER"
    val_str = str(val).strip()
    m = re.match(r"^(\d+)", val_str)
    if m:
        code = int(m.group(1))
        if 0 <= code <= 8:
            return str(code)
        return "OTHER"
    if val_str.upper().startswith("A"):
        m2 = re.match(r"^A(\d+)", val_str, re.IGNORECASE)
        if m2:
            abi = int(m2.group(1))
            if abi == 4:
                return "A4"
            if abi == 10:
                return "A10"
        return "A_OTHER"
    return "OTHER"


def consolidate_dd_duq(val: Any) -> str:
    """Map DD/DUQ flag variants to N/Y/OTHER.

    Args:
        val: Raw DD_DUQ value.

    Returns:
        "N", "Y", or "OTHER".
    """
    if pd.isna(val):
        return "OTHER"
    val_str = str(val).strip().upper()
    if val_str == "N":
        return "N"
    if val_str == "Y":
        return "Y"
    return "OTHER"


def consolidate_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all categorical consolidation to a DataFrame.

    Mutates a copy of the input, adding cleaned / derived columns:
    - RISK_AREA, CONVICTIONS_FLAG, CLASSOFUSEDESC, DD_DUQ — cleaned strings
    - CLM_GROUP — CLM_NUM_L5Y capped at 3, as string factor
    - FUEL_TYPE_CAT — FUEL_TYPE as string factor
    - Band columns — NaN replaced with "UNKNOWN"
    - NCDPROTECT — normalised to uppercase string "TRUE"/"FALSE"

    Args:
        df: Input DataFrame (train or test).

    Returns:
        New DataFrame with consolidated categorical columns.
    """
    df = df.copy()

    if "RISK_AREA" in df.columns:
        df["RISK_AREA"] = df["RISK_AREA"].apply(consolidate_risk_area)
    if "CONVICTIONS_FLAG" in df.columns:
        df["CONVICTIONS_FLAG"] = df["CONVICTIONS_FLAG"].apply(consolidate_convictions)
    if "CLASSOFUSEDESC" in df.columns:
        df["CLASSOFUSEDESC"] = df["CLASSOFUSEDESC"].apply(consolidate_class_of_use)
    if "DD_DUQ" in df.columns:
        df["DD_DUQ"] = df["DD_DUQ"].apply(consolidate_dd_duq)

    # CLM_NUM_L5Y -> capped factor
    if "CLM_NUM_L5Y" in df.columns:
        df["CLM_GROUP"] = (
            df["CLM_NUM_L5Y"].fillna(0).clip(upper=3).astype(int).astype(str)
        )
    else:
        df["CLM_GROUP"] = "0"

    # FUEL_TYPE as string factor
    if "FUEL_TYPE" in df.columns:
        df["FUEL_TYPE_CAT"] = df["FUEL_TYPE"].fillna(1).astype(int).astype(str)
    else:
        df["FUEL_TYPE_CAT"] = "1"

    # Fill band nulls with "UNKNOWN"
    band_cols = [
        "AGE_BAND",
        "ENGINE_SIZE_BAND",
        "VEHICLE_VALUE_BAND",
        "VEHICLE_AGE_BAND",
        "MILEAGE_K_BAND",
        "CREDIT_SCORE_BAND",
    ]
    for col in band_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", "UNKNOWN")

    # NCDPROTECT: normalise to uppercase string
    if "NCDPROTECT" in df.columns:
        df["NCDPROTECT"] = (
            df["NCDPROTECT"]
            .astype(str)
            .str.upper()
            .replace({"TRUE": "TRUE", "FALSE": "FALSE", "NAN": "FALSE"})
            .fillna("FALSE")
        )

    return df


# =============================================================================
# Section 3: Design Matrix Construction
# =============================================================================

CATEGORICAL_FACTORS: List[str] = [
    "AGE_BAND",
    "ENGINE_SIZE_BAND",
    "VEHICLE_VALUE_BAND",
    "VEHICLE_AGE_BAND",
    "MILEAGE_K_BAND",
    "CREDIT_SCORE_BAND",
    "RISK_AREA",
    "CONVICTIONS_FLAG",
    "COVER_TYPE",
    "DD_DUQ",
    "CLASSOFUSEDESC",
    "NCDPROTECT",
    "CLM_GROUP",
    "FUEL_TYPE_CAT",
]

CONTINUOUS_FACTORS: List[str] = ["NCD_CAPPED"]  # DTI added optionally


def prepare_design_matrix(
    df: pd.DataFrame,
    factors: List[str],
    base_levels: Dict[str, str] = BASE_LEVELS,
) -> pd.DataFrame:
    """Build design matrix with one-hot encoding, dropping base levels.

    Categorical factors in CATEGORICAL_FACTORS receive dummy encoding with
    the base level column dropped.  Continuous factors are included as-is.

    Args:
        df: Source DataFrame (train or test).
        factors: Ordered list of factor names to include.
        base_levels: Mapping of factor -> base level string.

    Returns:
        Float design matrix (no intercept — statsmodels adds it).
    """
    parts: List[pd.DataFrame] = []

    for factor in factors:
        if factor not in df.columns:
            log.warning("Factor %s not found in DataFrame — skipped.", factor)
            continue

        if factor in CATEGORICAL_FACTORS:
            dummies = pd.get_dummies(df[factor], prefix=factor, dtype=float)
            base_col = f"{factor}_{base_levels.get(factor, '')}"
            if base_col in dummies.columns:
                dummies = dummies.drop(columns=[base_col])
            parts.append(dummies)
        else:
            # Continuous or pass-through numeric
            series = df[[factor]].astype(float)
            parts.append(series)

    if not parts:
        return pd.DataFrame(index=df.index)

    X = pd.concat(parts, axis=1)
    return X


def align_test_matrix(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    """Ensure test matrix has same columns as train.

    Columns present in train but absent in test are added with value 0.
    Columns present in test but absent in train are dropped.
    Column order is forced to match train.

    Args:
        X_train: Training design matrix.
        X_test: Test design matrix (may differ due to unseen levels).

    Returns:
        Aligned test design matrix with identical columns to X_train.
    """
    X_test = X_test.copy()
    missing = set(X_train.columns) - set(X_test.columns)
    extra = set(X_test.columns) - set(X_train.columns)

    if missing:
        log.debug("Test matrix: adding %d missing columns (set to 0): %s", len(missing), missing)
    if extra:
        log.debug("Test matrix: dropping %d extra columns: %s", len(extra), extra)

    for col in missing:
        X_test[col] = 0.0

    X_test = X_test[X_train.columns]
    return X_test


# =============================================================================
# Section 4: Model Fitting — Wrapper Classes and fit_gamma_glm
# =============================================================================


class StatsmodelsResult:
    """Uniform wrapper around a statsmodels GLMResults object.

    Exposes a stable interface used by diagnostics, stepwise selection,
    and relativity extraction regardless of the underlying backend.

    Args:
        result: Fitted statsmodels GLMResults object.
        feature_names: List of feature names (excluding the intercept).
    """

    def __init__(self, result: Any, feature_names: List[str]) -> None:
        self._result = result
        self.feature_names = feature_names

    @property
    def params(self) -> pd.Series:
        """Fitted coefficient vector including intercept."""
        return self._result.params

    @property
    def bse(self) -> pd.Series:
        """Standard errors for all coefficients."""
        return self._result.bse

    @property
    def pvalues(self) -> pd.Series:
        """Two-sided Wald p-values for all coefficients."""
        return self._result.pvalues

    @property
    def deviance(self) -> float:
        """Residual deviance of the fitted model."""
        return float(self._result.deviance)

    @property
    def pearson_chi2(self) -> float:
        """Pearson chi-squared statistic."""
        return float(self._result.pearson_chi2)

    @property
    def aic(self) -> float:
        """Akaike Information Criterion."""
        return float(self._result.aic)

    @property
    def bic(self) -> float:
        """BIC based on log-likelihood (bic_llf)."""
        return float(self._result.bic_llf)

    @property
    def df_resid(self) -> float:
        """Residual degrees of freedom."""
        return float(self._result.df_resid)

    @property
    def n_params(self) -> int:
        """Total number of estimated parameters (including intercept)."""
        return int(len(self._result.params))

    @property
    def scale(self) -> float:
        """Estimated dispersion parameter."""
        return float(self._result.scale)

    @property
    def resid_deviance(self) -> pd.Series:
        """Deviance residuals."""
        return self._result.resid_deviance

    @property
    def resid_pearson(self) -> pd.Series:
        """Pearson residuals."""
        return self._result.resid_pearson

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions on a new design matrix.

        Args:
            X: Design matrix without intercept column.

        Returns:
            Predicted values on the response scale (premium £).
        """
        X_const = sm.add_constant(X, has_constant="add")
        return np.asarray(self._result.predict(X_const))

    def summary(self) -> Any:
        """Return the statsmodels GLM summary object."""
        return self._result.summary()

    def get_influence(self) -> Any:
        """Return the statsmodels influence measures object."""
        return self._result.get_influence()


class SklearnResult:
    """Uniform wrapper around a sklearn GammaRegressor (fallback backend).

    Provides the same interface as StatsmodelsResult where possible.
    Standard errors, deviance, AIC, and BIC are unavailable and return NaN.

    Args:
        model: Fitted sklearn GammaRegressor.
        feature_names: List of feature names.
    """

    def __init__(self, model: Any, feature_names: List[str]) -> None:
        self._model = model
        self.feature_names = feature_names
        # Build params Series to mimic statsmodels interface
        coef_values = np.concatenate([[model.intercept_], model.coef_])
        coef_names = ["const"] + list(feature_names)
        self._params = pd.Series(coef_values, index=coef_names)

    @property
    def params(self) -> pd.Series:
        """Fitted coefficients as a Series with names."""
        return self._params

    @property
    def bse(self) -> pd.Series:
        """Standard errors — unavailable for sklearn, returns NaN Series."""
        return pd.Series(np.nan, index=self._params.index)

    @property
    def pvalues(self) -> pd.Series:
        """P-values — unavailable for sklearn, returns NaN Series."""
        return pd.Series(np.nan, index=self._params.index)

    @property
    def deviance(self) -> float:
        """Deviance — unavailable for sklearn, returns NaN."""
        return float("nan")

    @property
    def pearson_chi2(self) -> float:
        """Pearson chi2 — unavailable for sklearn, returns NaN."""
        return float("nan")

    @property
    def aic(self) -> float:
        """AIC — unavailable for sklearn, returns NaN."""
        return float("nan")

    @property
    def bic(self) -> float:
        """BIC — unavailable for sklearn, returns NaN."""
        return float("nan")

    @property
    def df_resid(self) -> float:
        """Residual df — unavailable for sklearn, returns NaN."""
        return float("nan")

    @property
    def n_params(self) -> int:
        """Number of parameters (intercept + coefficients)."""
        return len(self._params)

    @property
    def scale(self) -> float:
        """Dispersion — unavailable for sklearn, returns NaN."""
        return float("nan")

    @property
    def resid_deviance(self) -> pd.Series:
        """Deviance residuals — unavailable for sklearn."""
        return pd.Series(dtype=float)

    @property
    def resid_pearson(self) -> pd.Series:
        """Pearson residuals — unavailable for sklearn."""
        return pd.Series(dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the sklearn model.

        Args:
            X: Design matrix (intercept column must NOT be included).

        Returns:
            Predicted values on the response scale.
        """
        return np.asarray(self._model.predict(X.values.astype(float)))

    def summary(self) -> str:
        """Return a minimal text summary."""
        lines = ["SklearnResult (GammaRegressor fallback)"]
        lines.append(f"  n_params: {self.n_params}")
        lines.append("  Deviance/AIC/BIC: not available")
        return "\n".join(lines)

    def get_influence(self) -> None:
        """Influence measures not available for sklearn backend."""
        return None


def fit_gamma_glm(
    X: pd.DataFrame,
    y: pd.Series,
    weights: Optional[np.ndarray] = None,
) -> "StatsmodelsResult | SklearnResult":
    """Fit a Gamma GLM with log link.

    Uses statsmodels as the primary backend.  Falls back to sklearn
    GammaRegressor when statsmodels is not installed.

    When X is empty (null / intercept-only model), a single constant column
    is created so that statsmodels can still fit an intercept.

    Args:
        X: Design matrix without intercept (may be empty for null model).
        y: Response vector (positive floats, e.g. premium £).
        weights: Optional frequency weights (passed to statsmodels only).

    Returns:
        Fitted result wrapper (StatsmodelsResult or SklearnResult).
    """
    # Intercept-only null model: X is empty — create a dummy constant column
    # so that sm.add_constant has something to attach the intercept to.
    if X.shape[1] == 0:
        X_fit = pd.DataFrame({"_null_": np.zeros(len(y))}, index=X.index)
    else:
        X_fit = X.copy()

    if HAS_STATSMODELS:
        X_const = sm.add_constant(X_fit, has_constant="add")

        family = Gamma(link=LogLink())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if weights is not None:
                glm_model = sm.GLM(y, X_const, family=family, freq_weights=weights)
            else:
                glm_model = sm.GLM(y, X_const, family=family)
            result = glm_model.fit(maxiter=100, tol=1e-8)

        # Feature names: all X columns except the null placeholder
        feature_names = [c for c in X_fit.columns if c != "_null_"]
        return StatsmodelsResult(result, feature_names)

    else:
        # sklearn fallback — no intercept-only support needed; GammaRegressor
        # always fits its own intercept.
        from sklearn.linear_model import GammaRegressor

        if X.shape[1] == 0:
            X_sk = pd.DataFrame({"_null_": np.zeros(len(y))}, index=X.index)
        else:
            X_sk = X.copy()

        model = GammaRegressor(alpha=0.001, max_iter=1000, tol=1e-8)
        model.fit(X_sk.values.astype(float), y.values.astype(float))
        feature_names = [c for c in X_sk.columns if c != "_null_"]
        return SklearnResult(model, feature_names)


# =============================================================================
# Section 5: Stepwise Factor Selection
# =============================================================================

ALL_CANDIDATE_FACTORS: List[str] = [
    "AGE_BAND",
    "NCD_CAPPED",
    "CREDIT_SCORE_BAND",
    "VEHICLE_AGE_BAND",
    "RISK_AREA",
    "MILEAGE_K_BAND",
    "CONVICTIONS_FLAG",
    "VEHICLE_VALUE_BAND",
    "COVER_TYPE",
    "NCDPROTECT",
    "ENGINE_SIZE_BAND",
    "DD_DUQ",
    "CLASSOFUSEDESC",
    "CLM_GROUP",
    "FUEL_TYPE_CAT",
]


def stepwise_select(
    train_df: pd.DataFrame,
    y_train: pd.Series,
    candidate_factors: List[str],
    threshold: float = 0.05,
) -> Tuple[List[str], pd.DataFrame]:
    """Forward stepwise factor selection by deviance reduction.

    At each step the candidate that produces the largest statistically
    significant reduction in residual deviance (chi-squared LRT) is added.
    Selection stops when no candidate achieves p < threshold.

    Args:
        train_df: Training DataFrame with all consolidated columns.
        y_train: Training response vector (positive floats).
        candidate_factors: Ordered list of factors to consider.
        threshold: Significance level for the deviance LRT (default 0.05).

    Returns:
        Tuple of (selected_factors list, step_log DataFrame).
    """
    # Only consider factors that exist in the DataFrame
    available = [f for f in candidate_factors if f in train_df.columns]
    if len(available) < len(candidate_factors):
        missing_cands = set(candidate_factors) - set(available)
        log.warning("Candidate factors not found in DataFrame: %s", missing_cands)

    selected: List[str] = []
    remaining: List[str] = list(available)
    step_log: List[Dict[str, Any]] = []

    # Fit null model (intercept only)
    X_null = pd.DataFrame(index=train_df.index)
    null_result = fit_gamma_glm(X_null, y_train)
    current_deviance = null_result.deviance
    current_df_resid = null_result.df_resid

    log.info("  Null model deviance: %.2f  df_resid: %.0f", current_deviance, current_df_resid)

    for step in range(len(remaining)):
        best_factor: Optional[str] = None
        best_deviance = current_deviance
        best_pval = 1.0
        best_result: Optional[Any] = None

        for factor in remaining:
            trial_factors = selected + [factor]
            X_trial = prepare_design_matrix(train_df, trial_factors)

            try:
                trial_result = fit_gamma_glm(X_trial, y_train)
            except Exception as exc:
                log.debug("Factor %s failed to fit: %s", factor, exc)
                continue

            # Deviance LRT — only meaningful for statsmodels
            if np.isnan(trial_result.deviance):
                # sklearn fallback: can't do LRT — include all by convention
                if best_factor is None:
                    best_factor = factor
                    best_deviance = 0.0
                    best_pval = 0.0
                    best_result = trial_result
                continue

            delta_dev = current_deviance - trial_result.deviance
            delta_df = current_df_resid - trial_result.df_resid

            if delta_df <= 0:
                continue

            pval = float(1 - chi2.cdf(delta_dev, int(delta_df)))

            if pval < best_pval:
                best_factor = factor
                best_deviance = trial_result.deviance
                best_pval = pval
                best_result = trial_result

        if best_factor is not None and best_pval < threshold:
            selected.append(best_factor)
            remaining.remove(best_factor)

            # Compute Gini coefficient at this step
            X_sel = prepare_design_matrix(train_df, selected)
            if best_result is not None:
                y_pred = best_result.predict(X_sel)
            else:
                # Should not reach here, but guard defensively
                y_pred = np.full(len(y_train), y_train.mean())

            gini = compute_gini(y_train.values, np.asarray(y_pred))

            step_log.append(
                {
                    "step": step + 1,
                    "factor": best_factor,
                    "deviance": round(float(best_deviance), 2),
                    "delta_deviance": round(current_deviance - float(best_deviance), 2),
                    "p_value": best_pval,
                    "n_params": best_result.n_params if best_result else None,
                    "gini": round(gini, 4),
                }
            )

            current_deviance = float(best_deviance)
            current_df_resid = best_result.df_resid if best_result else current_df_resid

            log.info(
                "  Step %2d: +%-25s  dev=%.1f  p=%.2e  Gini=%.4f",
                step + 1,
                best_factor,
                current_deviance,
                best_pval,
                gini,
            )
        else:
            log.info(
                "  Stepwise stopped at step %d (best p=%.4f >= %.4f)",
                step + 1,
                best_pval,
                threshold,
            )
            break

    return selected, pd.DataFrame(step_log)


# =============================================================================
# Section 6: Interaction Testing
# =============================================================================


def test_interactions(
    train_df: pd.DataFrame,
    y_train: pd.Series,
    selected_factors: List[str],
    main_result: "StatsmodelsResult | SklearnResult",
) -> Tuple[List[Tuple[str, str]], pd.DataFrame]:
    """Test pre-specified interaction pairs against the main-effects model.

    For each pair in INTERACTION_PAIRS where both factors were selected,
    we augment the main-effects design matrix with the cross-product of
    non-base dummies and perform a chi-squared deviance LRT.  Interactions
    passing p < 0.01 and contributing > 0.1% deviance reduction are retained.

    Args:
        train_df: Training DataFrame.
        y_train: Training response vector.
        selected_factors: Factors selected by stepwise procedure.
        main_result: Fitted main-effects GLM result.

    Returns:
        Tuple of (significant interaction pairs list, results DataFrame).
    """
    # For sklearn fallback deviance is NaN — skip interaction testing
    if np.isnan(main_result.deviance):
        log.warning("Interaction testing skipped: deviance unavailable (sklearn backend).")
        return [], pd.DataFrame()

    base_deviance = main_result.deviance
    base_df_resid = main_result.df_resid
    results: List[Dict[str, Any]] = []
    significant: List[Tuple[str, str]] = []

    for f1, f2 in INTERACTION_PAIRS:
        if f1 not in selected_factors or f2 not in selected_factors:
            log.debug("  Interaction %s x %s: one or both factors not selected — skipped.", f1, f2)
            continue
        if f1 not in train_df.columns or f2 not in train_df.columns:
            log.debug("  Interaction %s x %s: column missing — skipped.", f1, f2)
            continue

        # Base design matrix (main effects)
        X_base = prepare_design_matrix(train_df, selected_factors)

        # Build non-base dummies for f1
        d1 = pd.get_dummies(train_df[f1], prefix=f1, dtype=float)
        base1 = f"{f1}_{BASE_LEVELS.get(f1, '')}"
        if base1 in d1.columns:
            d1 = d1.drop(columns=[base1])

        # Build non-base dummies (or continuous) for f2
        if f2 in CATEGORICAL_FACTORS:
            d2 = pd.get_dummies(train_df[f2], prefix=f2, dtype=float)
            base2 = f"{f2}_{BASE_LEVELS.get(f2, '')}"
            if base2 in d2.columns:
                d2 = d2.drop(columns=[base2])
        else:
            d2 = train_df[[f2]].astype(float)
            d2.columns = [f2]

        # Cross-product of non-base dummies
        interaction_series: List[pd.Series] = []
        for c1 in d1.columns:
            for c2 in d2.columns:
                inter_col = d1[c1] * d2[c2]
                inter_col.name = f"{c1}__x__{c2}"
                interaction_series.append(inter_col)

        if not interaction_series:
            continue

        X_inter = pd.concat([X_base] + interaction_series, axis=1)

        try:
            inter_result = fit_gamma_glm(X_inter, y_train)
        except Exception as exc:
            log.warning("  Interaction %s x %s fit failed: %s", f1, f2, exc)
            continue

        delta_dev = base_deviance - inter_result.deviance
        delta_df = base_df_resid - inter_result.df_resid

        if delta_df <= 0:
            continue

        pval = float(1 - chi2.cdf(delta_dev, int(delta_df)))
        pct_improvement = 100.0 * delta_dev / base_deviance if base_deviance > 0 else 0.0
        is_significant = bool(pval < 0.01 and pct_improvement > 0.1)

        results.append(
            {
                "interaction": f"{f1} x {f2}",
                "delta_deviance": round(delta_dev, 2),
                "pct_improvement": round(pct_improvement, 4),
                "delta_df": int(delta_df),
                "p_value": pval,
                "significant": is_significant,
            }
        )

        if is_significant:
            significant.append((f1, f2))
            log.info(
                "  Interaction %s x %s: dev_red=%.1f (%.2f%%), p=%.2e — INCLUDED",
                f1,
                f2,
                delta_dev,
                pct_improvement,
                pval,
            )
        else:
            log.info(
                "  Interaction %s x %s: dev_red=%.1f (%.2f%%), p=%.2e — not significant",
                f1,
                f2,
                delta_dev,
                pct_improvement,
                pval,
            )

    return significant, pd.DataFrame(results)


# =============================================================================
# Section 7: Diagnostics
# =============================================================================


def compute_gini(y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
    """Compute the Gini coefficient via the Lorenz curve.

    Gini = 1 - 2 * (area under Lorenz curve).  A higher Gini indicates
    greater discriminatory power of the model.

    Args:
        y_actual: Observed response values (e.g. actual premiums).
        y_predicted: Model predictions used to rank risks.

    Returns:
        Gini coefficient in [0, 1].
    """
    y_actual = np.asarray(y_actual, dtype=float)
    y_predicted = np.asarray(y_predicted, dtype=float)

    n = len(y_actual)
    if n == 0:
        return 0.0

    order = np.argsort(y_predicted)
    y_sorted = y_actual[order]
    cumulative = np.cumsum(y_sorted)
    total = cumulative[-1]

    if total == 0:
        return 0.0

    gini = float(1 - 2 * cumulative.sum() / (n * total))
    return gini


def compute_diagnostics(
    result: "StatsmodelsResult | SklearnResult",
    X: pd.DataFrame,
    y: pd.Series,
    label: str = "TRAIN",
) -> Dict[str, Any]:
    """Compute the full actuarial diagnostic suite.

    Metrics include Gini, MAE, RMSE, CV(RMSE), A/E ratio, deviance, AIC,
    and BIC.  For sklearn results, deviance/AIC/BIC are NaN.

    Args:
        result: Fitted GLM wrapper.
        X: Design matrix (no intercept).
        y: Observed response values.
        label: Label string ("TRAIN" or "TEST") for reporting.

    Returns:
        Dictionary of diagnostic metrics.
    """
    y_pred = result.predict(X)
    y_arr = np.asarray(y, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    mean_actual = float(y_arr.mean()) if len(y_arr) > 0 else 0.0
    mean_predicted = float(y_pred_arr.mean()) if len(y_pred_arr) > 0 else 0.0

    mae = float(np.mean(np.abs(y_arr - y_pred_arr)))
    rmse = float(np.sqrt(np.mean((y_arr - y_pred_arr) ** 2)))
    cv_rmse = rmse / mean_actual if mean_actual > 0 else float("nan")
    ae_ratio = float(y_arr.sum() / y_pred_arr.sum()) if y_pred_arr.sum() > 0 else float("nan")
    gini = compute_gini(y_arr, y_pred_arr)

    diag: Dict[str, Any] = {
        "split": label,
        "n": int(len(y_arr)),
        "deviance": result.deviance if hasattr(result, "deviance") else float("nan"),
        "aic": result.aic if hasattr(result, "aic") else float("nan"),
        "bic": result.bic if hasattr(result, "bic") else float("nan"),
        "n_params": result.n_params,
        "gini": round(gini, 6),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "cv_rmse": round(cv_rmse, 6),
        "ae_ratio": round(ae_ratio, 6),
        "mean_actual": round(mean_actual, 4),
        "mean_predicted": round(mean_predicted, 4),
    }

    return diag


def compute_vif(X: pd.DataFrame) -> Dict[str, float]:
    """Compute Variance Inflation Factor per column.

    Each column is regressed on all remaining columns (OLS with intercept).
    VIF = 1 / (1 - R²).  Values > 5 indicate moderate multicollinearity;
    values > 10 indicate severe multicollinearity.

    Args:
        X: Design matrix (float, no intercept, no constant column).

    Returns:
        Mapping of column name -> VIF value.
    """
    from numpy.linalg import lstsq

    vifs: Dict[str, float] = {}
    X_arr = X.values.astype(float)
    n_cols = X_arr.shape[1]

    for i, col in enumerate(X.columns):
        mask = list(range(n_cols))
        mask.pop(i)

        if len(mask) == 0:
            vifs[col] = 1.0
            continue

        X_others = X_arr[:, mask]
        y_col = X_arr[:, i]

        # Add intercept
        X_ols = np.column_stack([np.ones(len(y_col)), X_others])

        try:
            coef, _, _, _ = lstsq(X_ols, y_col, rcond=None)
            y_hat = X_ols @ coef
            ss_res = float(np.sum((y_col - y_hat) ** 2))
            ss_tot = float(np.sum((y_col - y_col.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vifs[col] = float(1.0 / (1.0 - r2)) if r2 < 1.0 else 999.0
        except Exception:
            vifs[col] = float("nan")

    return vifs


def decile_analysis(
    y_actual: pd.Series,
    y_predicted: np.ndarray,
    label: str = "TRAIN",
) -> pd.DataFrame:
    """Actual vs expected statistics by predicted premium decile.

    Args:
        y_actual: Observed response values.
        y_predicted: Model predictions used to form deciles.
        label: Split label ("TRAIN" or "TEST").

    Returns:
        DataFrame with one row per decile containing n, means, sums, A/E ratio.
    """
    y_arr = np.asarray(y_actual, dtype=float)
    y_pred_arr = np.asarray(y_predicted, dtype=float)

    df = pd.DataFrame({"actual": y_arr, "predicted": y_pred_arr})
    df["decile"] = pd.qcut(df["predicted"], 10, labels=range(1, 11), duplicates="drop")

    summary = (
        df.groupby("decile", observed=True)
        .agg(
            n=("actual", "count"),
            actual_mean=("actual", "mean"),
            predicted_mean=("predicted", "mean"),
            actual_sum=("actual", "sum"),
            predicted_sum=("predicted", "sum"),
        )
        .reset_index()
    )

    summary["ae_ratio"] = summary["actual_mean"] / summary["predicted_mean"]
    summary["split"] = label

    return summary


def factor_calibration(
    df: pd.DataFrame,
    y_actual: pd.Series,
    y_predicted: np.ndarray,
    factors: List[str],
    label: str = "TEST",
) -> pd.DataFrame:
    """Compute A/E ratio by factor level for calibration monitoring.

    Args:
        df: Source DataFrame containing factor columns.
        y_actual: Observed response values.
        y_predicted: Model predictions.
        factors: List of factor names to assess.
        label: Split label ("TRAIN" or "TEST").

    Returns:
        DataFrame with one row per (factor, level) containing n, means, A/E.
    """
    rows: List[Dict[str, Any]] = []
    y_act = np.asarray(y_actual, dtype=float)
    y_pred = np.asarray(y_predicted, dtype=float)

    for factor in factors:
        if factor not in df.columns:
            continue

        for level in sorted(df[factor].unique(), key=str):
            mask = (df[factor] == level).values
            n = int(mask.sum())
            if n == 0:
                continue

            act_mean = float(y_act[mask].mean())
            pred_mean = float(y_pred[mask].mean())
            ae = act_mean / pred_mean if pred_mean > 0 else float("nan")

            rows.append(
                {
                    "factor": factor,
                    "level": str(level),
                    "n": n,
                    "actual_mean": round(act_mean, 2),
                    "predicted_mean": round(pred_mean, 2),
                    "ae_ratio": round(ae, 4),
                    "split": label,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Section 8: Relativity Extraction
# =============================================================================


def _parse_param_name(name: str, factors: List[str]) -> Tuple[str, str]:
    """Parse a dummy-encoded column name into (factor, level).

    For example "AGE_BAND_17-21" -> ("AGE_BAND", "17-21").
    For interaction terms like "AGE_BAND_17-21__x__COVER_TYPE_TPFT" the full
    name is returned as-is with an empty level.

    Args:
        name: Parameter name from the GLM result.
        factors: List of factor names in the model.

    Returns:
        Tuple of (factor_name, level_string).
    """
    if name == "const":
        return "const", ""

    # Interaction columns contain "__x__" — return as one unit
    if "__x__" in name:
        return "interaction", name

    # Try matching against known factor prefixes (longest first to avoid partial matches)
    for factor in sorted(factors, key=len, reverse=True):
        prefix = f"{factor}_"
        if name.startswith(prefix):
            level = name[len(prefix):]
            return factor, level

    # Continuous factor or unrecognised — return name as factor, empty level
    return name, ""


def extract_relativities(
    result: "StatsmodelsResult | SklearnResult",
    train_df: pd.DataFrame,
    selected_factors: List[str],
) -> Tuple[float, pd.DataFrame]:
    """Extract multiplicative relativities from GLM log-link coefficients.

    For each non-intercept parameter:
    - relativity = exp(coefficient)
    - 95% CI = exp(coef +/- 1.96 * se)
    - credibility Z = sqrt(n / (n + 1082))  [Buhlmann credibility]

    Args:
        result: Fitted GLM wrapper.
        train_df: Training DataFrame (used for exposure counts).
        selected_factors: Factors selected during stepwise.

    Returns:
        Tuple of (base_premium_float, relativity_DataFrame).
    """
    params = result.params
    bse = result.bse if hasattr(result, "bse") else pd.Series(dtype=float)
    pvals = result.pvalues if hasattr(result, "pvalues") else pd.Series(dtype=float)

    # Intercept -> base premium on response scale
    intercept_coef = float(params.get("const", params.iloc[0]))
    base_premium = float(np.exp(intercept_coef))

    rows: List[Dict[str, Any]] = []

    for name, coef in params.items():
        if name == "const":
            continue

        coef_f = float(coef)
        se = float(bse.get(name, float("nan"))) if name in bse.index else float("nan")
        pval = float(pvals.get(name, float("nan"))) if name in pvals.index else float("nan")

        relativity = float(np.exp(coef_f))
        ci_lo = float(np.exp(coef_f - 1.96 * se)) if not np.isnan(se) else float("nan")
        ci_hi = float(np.exp(coef_f + 1.96 * se)) if not np.isnan(se) else float("nan")

        # Parse factor and level from column name
        factor, level = _parse_param_name(name, selected_factors)

        # Exposure count
        n = 0
        if factor in train_df.columns and level != "":
            n = int((train_df[factor] == level).sum())
        elif factor in train_df.columns:
            n = len(train_df)

        # Buhlmann credibility weight (k=1082 calibrated to ~50% cred at 1082 risks)
        cred_z = float(np.sqrt(n / (n + 1082))) if n > 0 else 0.0

        rows.append(
            {
                "factor": factor,
                "level": level,
                "n": n,
                "coefficient": round(coef_f, 6),
                "std_error": round(se, 6) if not np.isnan(se) else None,
                "relativity": round(relativity, 4),
                "ci_lower": round(ci_lo, 4) if not np.isnan(ci_lo) else None,
                "ci_upper": round(ci_hi, 4) if not np.isnan(ci_hi) else None,
                "p_value": round(pval, 6) if not np.isnan(pval) else None,
                "significant": bool(pval < 0.05) if not np.isnan(pval) else None,
                "credibility_z": round(cred_z, 4),
            }
        )

    return base_premium, pd.DataFrame(rows)


# =============================================================================
# Section 13: CLI Orchestrator
# =============================================================================


def _build_interaction_columns(
    df: pd.DataFrame,
    X_base: pd.DataFrame,
    included_interactions: List[Tuple[str, str]],
) -> pd.DataFrame:
    """Append interaction columns to a design matrix in-place (copy returned).

    Helper used by both the training and test matrix assembly paths to
    avoid code duplication.

    Args:
        df: Source DataFrame with factor columns.
        X_base: Main-effects design matrix to augment.
        included_interactions: List of (f1, f2) tuples to add.

    Returns:
        Augmented design matrix with interaction columns appended.
    """
    X_out = X_base.copy()

    for f1, f2 in included_interactions:
        if f1 not in df.columns or f2 not in df.columns:
            continue

        d1 = pd.get_dummies(df[f1], prefix=f1, dtype=float)
        base1 = f"{f1}_{BASE_LEVELS.get(f1, '')}"
        if base1 in d1.columns:
            d1 = d1.drop(columns=[base1])

        if f2 in CATEGORICAL_FACTORS:
            d2 = pd.get_dummies(df[f2], prefix=f2, dtype=float)
            base2 = f"{f2}_{BASE_LEVELS.get(f2, '')}"
            if base2 in d2.columns:
                d2 = d2.drop(columns=[base2])
        else:
            d2 = df[[f2]].astype(float)
            d2.columns = [f2]

        for c1 in d1.columns:
            for c2 in d2.columns:
                col_name = f"{c1}__x__{c2}"
                if c1 in d1.columns and c2 in d2.columns:
                    X_out[col_name] = d1[c1].values * d2[c2].values
                else:
                    X_out[col_name] = 0.0

    return X_out


def run_glm(config: GLMConfig) -> Dict[str, Any]:
    """Main orchestrator — runs the full GLM pipeline end to end.

    Executes Sections 1–8 in sequence:
    1. Data loading and train/test split
    2. Premium winsorisation
    3. Categorical consolidation
    4. Forward stepwise factor selection
    5. Main-effects Gamma GLM fitting
    6. Interaction testing (optional)
    7. Final model fitting (main effects + significant interactions)
    8. Diagnostics, decile analysis, calibration, relativity extraction, VIF

    All artefacts are written to config.output_dir.

    Args:
        config: Pipeline configuration object.

    Returns:
        Dictionary of all key objects (DataFrames, results, diagnostics)
        for use by appended visualisation and sensitivity sections.
    """
    t0 = time.time()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    log.info("=" * 70)
    log.info("Benchmark Net Premium Gamma GLM")
    log.info("=" * 70)

    # ------------------------------------------------------------------ 1. Load
    log.info("\n--- Data Loading ---")
    train_df, test_df = load_data(config)

    # ------------------------------------------------------------------ 2. Cap
    log.info("\n--- Premium Winsorisation ---")
    train_df, cap_value = cap_premium(train_df, config)
    y_train = train_df["AD_POLPREMIUM_CAPPED"]
    y_test = test_df["AD_POLPREMIUM"]

    # -------------------------------------------- 3. Categorical consolidation
    log.info("\n--- Categorical Consolidation ---")
    train_df = consolidate_categoricals(train_df)
    test_df = consolidate_categoricals(test_df)

    for col in ["RISK_AREA", "CONVICTIONS_FLAG", "CLASSOFUSEDESC", "DD_DUQ"]:
        if col in train_df.columns:
            n_levels = train_df[col].nunique()
            log.info("  %s: %d levels after consolidation", col, n_levels)

    # --------------------------------------- 4. Forward stepwise factor selection
    log.info("\n--- Stepwise Factor Selection ---")
    selected_factors, step_log_df = stepwise_select(
        train_df, y_train, ALL_CANDIDATE_FACTORS
    )
    step_log_df.to_csv(output_dir / "stepwise_log.csv", index=False)
    log.info("  Selected %d factors: %s", len(selected_factors), selected_factors)

    if not selected_factors:
        log.warning("No factors selected by stepwise — using all candidates as fallback.")
        selected_factors = [
            f for f in ALL_CANDIDATE_FACTORS if f in train_df.columns
        ]

    # -------------------------------------------- 5. Fit main-effects model
    log.info("\n--- Main Effects Model ---")
    X_train_main = prepare_design_matrix(train_df, selected_factors)
    main_result = fit_gamma_glm(X_train_main, y_train)
    log.info(
        "  Main effects: %d parameters, deviance=%.2f",
        main_result.n_params,
        main_result.deviance if not np.isnan(main_result.deviance) else -1,
    )

    # -------------------------------------------------- 6. Test interactions
    included_interactions: List[Tuple[str, str]] = []
    inter_df = pd.DataFrame()

    if config.run_interactions and len(selected_factors) >= 2:
        log.info("\n--- Interaction Testing ---")
        included_interactions, inter_df = test_interactions(
            train_df, y_train, selected_factors, main_result
        )
        inter_df.to_csv(output_dir / "interaction_tests.csv", index=False)
    else:
        log.info("\n--- Interaction Testing: skipped ---")

    # ----------------------- 7. Fit final model (+ significant interactions)
    if included_interactions:
        log.info("\n--- Final Model (main effects + %d interactions) ---", len(included_interactions))
        X_train_final = _build_interaction_columns(
            train_df, X_train_main, included_interactions
        )
        final_result = fit_gamma_glm(X_train_final, y_train)
        log.info(
            "  Final: %d parameters, deviance=%.2f",
            final_result.n_params,
            final_result.deviance if not np.isnan(final_result.deviance) else -1,
        )
    else:
        log.info("\n--- Final Model: main effects only ---")
        X_train_final = X_train_main
        final_result = main_result

    # --------------------------------------------------- 8a. Train diagnostics
    log.info("\n--- Train Diagnostics ---")
    train_diag = compute_diagnostics(final_result, X_train_final, y_train, "TRAIN")
    for k, v in train_diag.items():
        if isinstance(v, float) and not np.isnan(v):
            log.info("  %-20s %.4f", k, v)

    # --------------------------------------------------- 8b. Test validation
    log.info("\n--- Test Validation ---")
    X_test_main = prepare_design_matrix(test_df, selected_factors)

    if included_interactions:
        X_test_final = _build_interaction_columns(
            test_df, X_test_main, included_interactions
        )
    else:
        X_test_final = X_test_main

    X_test_final = align_test_matrix(X_train_final, X_test_final)

    test_diag = compute_diagnostics(final_result, X_test_final, y_test, "TEST")
    for k, v in test_diag.items():
        if isinstance(v, float) and not np.isnan(v):
            log.info("  %-20s %.4f", k, v)

    y_pred_test = final_result.predict(X_test_final)

    # ------------------------------------------- 8c. Decile analysis (A vs E)
    log.info("\n--- Decile Analysis ---")
    y_pred_train = final_result.predict(X_train_final)
    train_decile = decile_analysis(y_train, y_pred_train, "TRAIN")
    test_decile = decile_analysis(y_test, y_pred_test, "TEST")
    decile_combined = pd.concat([train_decile, test_decile], ignore_index=True)
    decile_combined.to_csv(output_dir / "decile_analysis.csv", index=False)
    log.info("  Saved decile_analysis.csv (%d rows)", len(decile_combined))

    # ------------------------------------------- 8d. Factor calibration
    log.info("\n--- Factor Calibration ---")
    calib_train = factor_calibration(train_df, y_train, y_pred_train, selected_factors, "TRAIN")
    calib_test = factor_calibration(test_df, y_test, y_pred_test, selected_factors, "TEST")
    calib_combined = pd.concat([calib_train, calib_test], ignore_index=True)
    calib_combined.to_csv(output_dir / "factor_calibration.csv", index=False)
    log.info("  Saved factor_calibration.csv (%d rows)", len(calib_combined))

    # ----------------------------------------------- 8e. Relativities
    log.info("\n--- Relativity Extraction ---")
    base_premium, rel_df = extract_relativities(final_result, train_df, selected_factors)
    rel_df.to_csv(output_dir / "relativity_table.csv", index=False)
    log.info("  Base premium: £%.2f", base_premium)
    log.info("  Parameters with relativities: %d", len(rel_df))
    log.info("  Unique factors: %d", rel_df["factor"].nunique())

    # Inspect high/low relativities
    if len(rel_df) > 0:
        rel_sorted = rel_df.sort_values("relativity", ascending=False)
        log.info("  Top 3 relativities:\n%s", rel_sorted[["factor", "level", "relativity"]].head(3).to_string(index=False))
        log.info("  Bottom 3 relativities:\n%s", rel_sorted[["factor", "level", "relativity"]].tail(3).to_string(index=False))

    # -------------------------------------------------------- 8f. VIF
    log.info("\n--- VIF Analysis ---")
    vifs = compute_vif(X_train_final)
    vif_df = pd.DataFrame(
        [{"column": k, "vif": round(v, 2)} for k, v in vifs.items()]
    ).sort_values("vif", ascending=False)
    vif_df.to_csv(output_dir / "vif_report.csv", index=False)
    max_vif = float(vif_df["vif"].max()) if len(vif_df) > 0 else float("nan")
    log.info(
        "  Max VIF: %.2f %s",
        max_vif,
        "(OK)" if not np.isnan(max_vif) and max_vif < 5 else "(WARNING: >5 or NaN)",
    )
    high_vif = vif_df[vif_df["vif"] > 5] if len(vif_df) > 0 else pd.DataFrame()
    if len(high_vif) > 0:
        log.warning("  Columns with VIF > 5:\n%s", high_vif.to_string(index=False))

    # ------------------------------------------- 8g. Model summary JSON
    log.info("\n--- Assembling Model Summary ---")
    gini_stability = (
        round(test_diag["gini"] / train_diag["gini"], 4)
        if train_diag["gini"] > 0
        else None
    )
    summary: Dict[str, Any] = {
        "model_type": "Gamma GLM",
        "link": "log",
        "target": "AD_POLPREMIUM",
        "backend": "statsmodels" if HAS_STATSMODELS else "sklearn",
        "premium_cap": round(cap_value, 2),
        "n_train": int(train_diag["n"]),
        "n_test": int(test_diag["n"]),
        "n_parameters": int(final_result.n_params),
        "base_premium_gbp": round(base_premium, 2),
        "train_gini": round(train_diag["gini"], 4),
        "test_gini": round(test_diag["gini"], 4),
        "gini_stability": gini_stability,
        "train_ae_ratio": round(train_diag["ae_ratio"], 4),
        "test_ae_ratio": round(test_diag["ae_ratio"], 4),
        "train_mae": round(train_diag["mae"], 2),
        "test_mae": round(test_diag["mae"], 2),
        "train_rmse": round(train_diag["rmse"], 2),
        "test_rmse": round(test_diag["rmse"], 2),
        "train_cv_rmse": round(train_diag["cv_rmse"], 4),
        "test_cv_rmse": round(test_diag["cv_rmse"], 4),
        "aic": round(train_diag.get("aic", float("nan")), 2),
        "bic": round(train_diag.get("bic", float("nan")), 2),
        "deviance": round(train_diag.get("deviance", float("nan")), 2),
        "factors_selected": selected_factors,
        "n_factors": len(selected_factors),
        "interactions_included": [f"{f1} x {f2}" for f1, f2 in included_interactions],
        "n_interactions": len(included_interactions),
        "max_vif": round(max_vif, 2) if not np.isnan(max_vif) else None,
    }

    with open(output_dir / "model_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log.info("  Saved model_summary.json")

    # Save statsmodels text summary if available
    if HAS_STATSMODELS and hasattr(final_result, "summary"):
        try:
            sm_summary_text = str(final_result.summary())
            (output_dir / "glm_summary.txt").write_text(sm_summary_text)
            log.info("  Saved glm_summary.txt")
        except Exception as exc:
            log.warning("  Could not write glm_summary.txt: %s", exc)

    # Save predictions for downstream use
    train_preds_df = pd.DataFrame(
        {"actual": y_train.values, "predicted": y_pred_train},
        index=train_df.index,
    )
    train_preds_df.to_csv(output_dir / "train_predictions.csv", index=False)

    test_preds_df = pd.DataFrame(
        {"actual": y_test.values, "predicted": y_pred_test},
        index=test_df.index,
    )
    test_preds_df.to_csv(output_dir / "test_predictions.csv", index=False)
    log.info("  Saved train_predictions.csv and test_predictions.csv")

    # ---------------------------------------------------------------- Summary
    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("BENCHMARK NET PREMIUM GAMMA GLM — RESULTS")
    print("=" * 70)
    print(f"  Backend:         {'statsmodels' if HAS_STATSMODELS else 'sklearn (fallback)'}")
    print(f"  Base Premium:    £{base_premium:,.2f}")
    print(f"  Train Gini:      {train_diag['gini']:.4f}")
    print(f"  Test Gini:       {test_diag['gini']:.4f}")
    if gini_stability is not None:
        print(f"  Gini Stability:  {gini_stability:.2%}")
    print(f"  Train A/E:       {train_diag['ae_ratio']:.4f}")
    print(f"  Test A/E:        {test_diag['ae_ratio']:.4f}")
    print(f"  Train MAE:       £{train_diag['mae']:,.2f}")
    print(f"  Test MAE:        £{test_diag['mae']:,.2f}")
    aic_val = train_diag.get("aic", float("nan"))
    print(f"  AIC:             {aic_val:.1f}" if not np.isnan(aic_val) else "  AIC:             N/A")
    print(f"  Factors:         {len(selected_factors)}")
    print(f"  Parameters:      {final_result.n_params}")
    print(f"  Interactions:    {len(included_interactions)}")
    print(f"  Max VIF:         {max_vif:.2f}" if not np.isnan(max_vif) else "  Max VIF:         N/A")
    print(f"  Premium Cap:     £{cap_value:,.0f}")
    print(f"  Elapsed:         {elapsed:.1f}s")
    print(f"  Output:          {output_dir}/")
    print("=" * 70)

    # Return everything needed by Sections 9-12
    return {
        "config": config,
        "train_df": train_df,
        "test_df": test_df,
        "y_train": y_train,
        "y_test": y_test,
        "X_train": X_train_final,
        "X_test": X_test_final,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "final_result": final_result,
        "main_result": main_result,
        "selected_factors": selected_factors,
        "included_interactions": included_interactions,
        "train_diag": train_diag,
        "test_diag": test_diag,
        "base_premium": base_premium,
        "rel_df": rel_df,
        "step_log_df": step_log_df,
        "cap_value": cap_value,
        "vif_df": vif_df,
        "summary": summary,
        "decile_df": decile_combined,
        "calib_df": calib_combined,
        "inter_df": inter_df,
        "output_dir": output_dir,
    }


# =============================================================================
# Section 9: Parsimonious Model
# =============================================================================


def fit_parsimonious_model(results: Dict[str, Any]) -> Dict[str, Any]:
    """Re-fit a Gamma GLM using only the top 6 factors by deviance contribution.

    Selects the six factors with the largest cumulative deviance reduction from
    the stepwise log, builds a main-effects-only design matrix, fits a Gamma
    GLM, computes diagnostics, and saves comparison artefacts.

    Args:
        results: Full results dict returned by ``run_glm()``.

    Returns:
        Dictionary with keys ``parsimonious_result``, ``pars_train_diag``,
        ``pars_test_diag``, ``pars_base_premium``, ``pars_rel_df``.
    """
    log.info("\n--- Section 9: Parsimonious Model ---")

    output_dir: Path = results["output_dir"]
    train_df: pd.DataFrame = results["train_df"]
    test_df: pd.DataFrame = results["test_df"]
    y_train: pd.Series = results["y_train"]
    y_test: pd.Series = results["y_test"]
    step_log_df: pd.DataFrame = results["step_log_df"]
    full_train_diag: Dict[str, Any] = results["train_diag"]
    full_test_diag: Dict[str, Any] = results["test_diag"]
    full_base_premium: float = results["base_premium"]

    # ------------------------------------------------------------------
    # Select top 6 factors by cumulative deviance contribution
    # ------------------------------------------------------------------
    if len(step_log_df) == 0:
        log.warning("  step_log_df is empty — using first 6 selected factors.")
        top6_factors: List[str] = results["selected_factors"][:6]
    else:
        sorted_log = step_log_df.sort_values("delta_deviance", ascending=False)
        top6_factors = list(sorted_log["factor"].head(6))

    log.info("  Parsimonious factors (%d): %s", len(top6_factors), top6_factors)

    if not top6_factors:
        log.warning("  No parsimonious factors identified — aborting Section 9.")
        return {}

    # ------------------------------------------------------------------
    # Build design matrices (no interactions)
    # ------------------------------------------------------------------
    X_pars_train = prepare_design_matrix(train_df, top6_factors)
    X_pars_test_raw = prepare_design_matrix(test_df, top6_factors)
    X_pars_test = align_test_matrix(X_pars_train, X_pars_test_raw)

    # ------------------------------------------------------------------
    # Fit parsimonious Gamma GLM
    # ------------------------------------------------------------------
    try:
        pars_result = fit_gamma_glm(X_pars_train, y_train)
    except Exception as exc:
        log.error("  Parsimonious model fit failed: %s", exc)
        return {}

    log.info(
        "  Parsimonious model: %d parameters, deviance=%.2f",
        pars_result.n_params,
        pars_result.deviance if not np.isnan(pars_result.deviance) else -1,
    )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    pars_train_diag = compute_diagnostics(pars_result, X_pars_train, y_train, "TRAIN")
    pars_test_diag = compute_diagnostics(pars_result, X_pars_test, y_test, "TEST")

    log.info(
        "  Parsimonious — Train Gini: %.4f  Test Gini: %.4f  AIC: %.1f",
        pars_train_diag["gini"],
        pars_test_diag["gini"],
        pars_train_diag.get("aic", float("nan")),
    )

    # ------------------------------------------------------------------
    # Comparison logging
    # ------------------------------------------------------------------
    full_aic = full_train_diag.get("aic", float("nan"))
    pars_aic = pars_train_diag.get("aic", float("nan"))
    gini_diff_train = pars_train_diag["gini"] - full_train_diag["gini"]
    gini_diff_test = pars_test_diag["gini"] - full_test_diag["gini"]
    aic_diff = pars_aic - full_aic if not (np.isnan(pars_aic) or np.isnan(full_aic)) else float("nan")

    log.info(
        "  Comparison vs full model:\n"
        "    Gini train diff: %+.4f\n"
        "    Gini test  diff: %+.4f\n"
        "    AIC diff:        %+.1f",
        gini_diff_train,
        gini_diff_test,
        aic_diff if not np.isnan(aic_diff) else -9999,
    )

    # ------------------------------------------------------------------
    # Extract relativities
    # ------------------------------------------------------------------
    pars_base_premium, pars_rel_df = extract_relativities(
        pars_result, train_df, top6_factors
    )
    log.info("  Parsimonious base premium: £%.2f  (full: £%.2f)", pars_base_premium, full_base_premium)

    # ------------------------------------------------------------------
    # Save artefacts
    # ------------------------------------------------------------------
    pars_summary: Dict[str, Any] = {
        "model_type": "Parsimonious Gamma GLM (top 6 factors)",
        "factors": top6_factors,
        "n_factors": len(top6_factors),
        "n_parameters": pars_result.n_params,
        "base_premium_gbp": round(pars_base_premium, 2),
        "train_gini": round(pars_train_diag["gini"], 4),
        "test_gini": round(pars_test_diag["gini"], 4),
        "train_ae_ratio": round(pars_train_diag["ae_ratio"], 4),
        "test_ae_ratio": round(pars_test_diag["ae_ratio"], 4),
        "train_mae": round(pars_train_diag["mae"], 2),
        "test_mae": round(pars_test_diag["mae"], 2),
        "aic": round(pars_aic, 2) if not np.isnan(pars_aic) else None,
        "comparison": {
            "full_model_n_factors": len(results["selected_factors"]),
            "full_model_aic": round(full_aic, 2) if not np.isnan(full_aic) else None,
            "full_model_train_gini": round(full_train_diag["gini"], 4),
            "full_model_test_gini": round(full_test_diag["gini"], 4),
            "full_base_premium_gbp": round(full_base_premium, 2),
            "gini_train_diff": round(gini_diff_train, 4),
            "gini_test_diff": round(gini_diff_test, 4),
            "aic_diff": round(aic_diff, 2) if not np.isnan(aic_diff) else None,
        },
    }

    with open(output_dir / "parsimonious_model_summary.json", "w") as fh:
        json.dump(pars_summary, fh, indent=2, default=str)
    log.info("  Saved parsimonious_model_summary.json")

    pars_rel_df.to_csv(output_dir / "parsimonious_relativity_table.csv", index=False)
    log.info("  Saved parsimonious_relativity_table.csv (%d rows)", len(pars_rel_df))

    return {
        "parsimonious_result": pars_result,
        "pars_train_diag": pars_train_diag,
        "pars_test_diag": pars_test_diag,
        "pars_base_premium": pars_base_premium,
        "pars_rel_df": pars_rel_df,
        "top6_factors": top6_factors,
        "X_pars_train": X_pars_train,
        "X_pars_test": X_pars_test,
    }


# =============================================================================
# Section 10: Visualisations
# =============================================================================


def generate_visualizations(results: Dict[str, Any]) -> None:
    """Generate 14 diagnostic PNG figures in ``{output_dir}/figures/``.

    Each figure is saved at 150 DPI with a white background.  The function
    is a no-op when matplotlib is unavailable.

    Args:
        results: Full results dict returned by ``run_glm()``.
    """
    if not HAS_MATPLOTLIB:
        log.warning("matplotlib not available — skipping Section 10 visualisations.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns
    from scipy import stats as scipy_stats

    log.info("\n--- Section 10: Generating Visualisations ---")

    output_dir: Path = results["output_dir"]
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    train_df: pd.DataFrame = results["train_df"]
    test_df: pd.DataFrame = results["test_df"]
    y_train: pd.Series = results["y_train"]
    y_test: pd.Series = results["y_test"]
    y_pred_train: np.ndarray = results["y_pred_train"]
    y_pred_test: np.ndarray = results["y_pred_test"]
    final_result = results["final_result"]
    step_log_df: pd.DataFrame = results["step_log_df"]
    rel_df: pd.DataFrame = results["rel_df"]
    vif_df: pd.DataFrame = results["vif_df"]
    decile_df: pd.DataFrame = results["decile_df"]
    calib_df: pd.DataFrame = results["calib_df"]
    selected_factors: List[str] = results["selected_factors"]
    included_interactions: List[Tuple[str, str]] = results["included_interactions"]

    _TITLE_KW = {"color": C_PRIMARY, "fontweight": "bold", "fontsize": 12}
    _SAVE_KW = {"dpi": 150, "bbox_inches": "tight", "facecolor": "white"}

    def _save(fig: Any, name: str) -> None:
        path = figures_dir / name
        fig.savefig(path, **_SAVE_KW)
        plt.close(fig)
        log.info("  Saved %s", name)

    # ------------------------------------------------------------------
    # Fig 01 — Target distribution
    # ------------------------------------------------------------------
    try:
        # Reload raw target from train_df — use AD_POLPREMIUM if available,
        # else fall back to the capped series.
        raw_col = "AD_POLPREMIUM" if "AD_POLPREMIUM" in train_df.columns else None
        raw_series = train_df[raw_col].dropna() if raw_col else y_train.dropna()
        raw_arr = np.asarray(raw_series, dtype=float)
        raw_arr = raw_arr[raw_arr > 0]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(
            np.log10(raw_arr),
            bins=60,
            color=C_ACCENT,
            edgecolor="white",
            linewidth=0.4,
            alpha=0.85,
        )
        mean_val = float(np.mean(raw_arr))
        median_val = float(np.median(raw_arr))
        cap_val: float = results["cap_value"]

        for val, label_txt, col in [
            (mean_val, f"Mean £{mean_val:,.0f}", C_AMBER),
            (median_val, f"Median £{median_val:,.0f}", C_GREEN),
            (cap_val, f"Cap £{cap_val:,.0f}", C_RED),
        ]:
            if val > 0:
                ax.axvline(np.log10(val), color=col, linewidth=1.8, linestyle="--", label=label_txt)

        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"£{10**x:,.0f}")
        )
        ax.set_xlabel("AD_POLPREMIUM (log scale)", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title("Fig 01 — Target Distribution (AD_POLPREMIUM)", **_TITLE_KW)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        _save(fig, "fig01_target_distribution.png")
    except Exception as exc:
        log.warning("Fig 01 failed: %s", exc)
        plt.close("all")

    # ------------------------------------------------------------------
    # Fig 02 — Deviance residuals QQ plot
    # ------------------------------------------------------------------
    try:
        resid_dev = np.asarray(final_result.resid_deviance, dtype=float)
        resid_dev = resid_dev[np.isfinite(resid_dev)]

        fig, ax = plt.subplots(figsize=(7, 6))
        if len(resid_dev) > 0:
            (osm, osr), (slope, intercept, _r) = scipy_stats.probplot(
                resid_dev, dist="norm", fit=True
            )
            ax.scatter(osm, osr, s=4, color=C_ACCENT, alpha=0.5, label="Deviance residuals")
            x_line = np.array([min(osm), max(osm)])
            ax.plot(x_line, slope * x_line + intercept, color=C_RED, linewidth=1.5, label="Theoretical line")
        else:
            ax.text(0.5, 0.5, "Deviance residuals unavailable\n(sklearn backend)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11, color="grey")

        ax.set_xlabel("Theoretical quantiles", fontsize=10)
        ax.set_ylabel("Sample quantiles", fontsize=10)
        ax.set_title("Fig 02 — Deviance Residuals QQ Plot", **_TITLE_KW)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        _save(fig, "fig02_deviance_residuals_qq.png")
    except Exception as exc:
        log.warning("Fig 02 failed: %s", exc)
        plt.close("all")

    # ------------------------------------------------------------------
    # Fig 03 — Residuals vs Fitted
    # ------------------------------------------------------------------
    try:
        resid_dev = np.asarray(final_result.resid_deviance, dtype=float)
        fitted = np.asarray(y_pred_train, dtype=float)

        fig, ax = plt.subplots(figsize=(9, 5))
        if len(resid_dev) == len(fitted) and len(resid_dev) > 0:
            mask = np.isfinite(resid_dev) & np.isfinite(fitted) & (fitted > 0)
            x_log = np.log(fitted[mask])
            y_res = resid_dev[mask]

            # Subsample if large
            n_plot = min(5000, len(x_log))
            rng = np.random.default_rng(42)
            idx = rng.choice(len(x_log), size=n_plot, replace=False)
            ax.scatter(
                x_log[idx], y_res[idx],
                s=3, color=C_ACCENT, alpha=0.35, rasterized=True
            )

            # Lowess smoother
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
                sorted_idx = np.argsort(x_log[idx])
                smoothed = sm_lowess(y_res[idx][sorted_idx], x_log[idx][sorted_idx], frac=0.25)
                ax.plot(smoothed[:, 0], smoothed[:, 1], color=C_RED, linewidth=2, label="Lowess")
                ax.legend(fontsize=9)
            except Exception:
                pass

            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"£{np.exp(x):,.0f}")
            )
        else:
            ax.text(0.5, 0.5, "Residuals unavailable (sklearn backend)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11, color="grey")

        ax.set_xlabel("Fitted values (log scale)", fontsize=10)
        ax.set_ylabel("Deviance residuals", fontsize=10)
        ax.set_title("Fig 03 — Residuals vs Fitted", **_TITLE_KW)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        _save(fig, "fig03_residuals_vs_fitted.png")
    except Exception as exc:
        log.warning("Fig 03 failed: %s", exc)
        plt.close("all")

    # ------------------------------------------------------------------
    # Fig 04 — Actual vs Expected by decile
    # ------------------------------------------------------------------
    try:
        train_dec = decile_df[decile_df["split"] == "TRAIN"].copy()
        test_dec = decile_df[decile_df["split"] == "TEST"].copy()

        fig, ax = plt.subplots(figsize=(12, 6))
        n_dec = max(len(train_dec), len(test_dec), 1)
        x = np.arange(n_dec)
        w = 0.35

        if len(train_dec) > 0:
            ax.bar(x - w / 2, train_dec["actual_mean"].values, w, label="Train Actual",
                   color=C_PRIMARY, alpha=0.85)
            ax.bar(x - w / 2 + w, train_dec["predicted_mean"].values, w, label="Train Expected",
                   color=C_ACCENT, alpha=0.85)
        if len(test_dec) > 0:
            ax.bar(x + w * 0.5, test_dec["actual_mean"].values, w, label="Test Actual",
                   color=C_AMBER, alpha=0.85)
            ax.bar(x + w * 1.5, test_dec["predicted_mean"].values, w, label="Test Expected",
                   color=C_RED, alpha=0.7)

        # A/E overlay lines
        ax2 = ax.twinx()
        if len(train_dec) > 0 and "ae_ratio" in train_dec.columns:
            ax2.plot(x - w / 2, train_dec["ae_ratio"].values, "o--",
                     color=C_PRIMARY, linewidth=1.5, markersize=5, label="Train A/E")
        if len(test_dec) > 0 and "ae_ratio" in test_dec.columns:
            ax2.plot(x + w * 0.5, test_dec["ae_ratio"].values, "s--",
                     color=C_AMBER, linewidth=1.5, markersize=5, label="Test A/E")
        ax2.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
        ax2.set_ylabel("A/E Ratio", fontsize=9, color="grey")
        ax2.set_ylim(0.8, 1.25)

        ax.set_xticks(x)
        ax.set_xticklabels([f"D{i+1}" for i in range(n_dec)], fontsize=9)
        ax.set_xlabel("Predicted premium decile", fontsize=10)
        ax.set_ylabel("Mean premium (£)", fontsize=10)
        ax.set_title("Fig 04 — Actual vs Expected by Decile (Train & Test)", **_TITLE_KW)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        _save(fig, "fig04_actual_vs_expected_decile.png")
    except Exception as exc:
        log.warning("Fig 04 failed: %s", exc)
        plt.close("all")

    # ------------------------------------------------------------------
    # Fig 05 — Gini Lorenz curves
    # ------------------------------------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(7, 6))

        def _lorenz_curve(y_act: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            order = np.argsort(y_pred)
            y_sorted = y_act[order]
            cum_actual = np.cumsum(y_sorted) / y_sorted.sum()
            cum_pop = np.linspace(0, 1, len(y_sorted))
            return cum_pop, cum_actual

        for y_act, y_pred, label_txt, col in [
            (np.asarray(y_train, float), y_pred_train, "Train", C_PRIMARY),
            (np.asarray(y_test, float), y_pred_test, "Test", C_ACCENT),
        ]:
            gini_val = compute_gini(y_act, y_pred)
            cum_pop, cum_actual = _lorenz_curve(y_act, y_pred)
            ax.plot(cum_pop, cum_actual, color=col, linewidth=2,
                    label=f"{label_txt} (Gini={gini_val:.3f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (Gini=0)")
        ax.set_xlabel("Cumulative population (ranked by predicted)", fontsize=10)
        ax.set_ylabel("Cumulative actual premium share", fontsize=10)
        ax.set_title("Fig 05 — Gini Lorenz Curves", **_TITLE_KW)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        _save(fig, "fig05_gini_lorenz.png")
    except Exception as exc:
        log.warning("Fig 05 failed: %s", exc)
        plt.close("all")

    # ------------------------------------------------------------------
    # Fig 06 — Relativity heatmap table
    # ------------------------------------------------------------------
    try:
        if len(rel_df) > 0:
            pivot_rows = rel_df[rel_df["factor"].isin(selected_factors)].copy()
            pivot_rows = pivot_rows[pivot_rows["level"] != ""].head(60)
            if len(pivot_rows) > 0:
                pivot_rows["label"] = pivot_rows["factor"] + " :: " + pivot_rows["level"].astype(str)
                pivot_rows = pivot_rows.sort_values("relativity")
                n_rows = len(pivot_rows)
                cell_h = max(0.3, min(0.5, 18.0 / n_rows))
                fig_h = max(5, n_rows * cell_h + 1.5)
                fig, ax = plt.subplots(figsize=(8, fig_h))

                rel_vals = pivot_rows["relativity"].values
                vmin, vmax = max(0.3, rel_vals.min()), min(3.5, rel_vals.max())
                colours = plt.cm.RdYlGn(  # type: ignore[attr-defined]
                    (rel_vals - vmin) / max(vmax - vmin, 1e-6)
                )

                bars = ax.barh(
                    pivot_rows["label"].values,
                    rel_vals,
                    color=colours,
                    edgecolor="white",
                    linewidth=0.4,
                )
                ax.axvline(1.0, color="black", linewidth=1.0, linestyle="--")
                for bar_obj, val in zip(bars, rel_vals):
                    ax.text(
                        val + 0.01, bar_obj.get_y() + bar_obj.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=6.5,
                    )
                ax.set_xlabel("Multiplicative relativity", fontsize=10)
                ax.set_title("Fig 06 — Multiplicative Relativities", **_TITLE_KW)
                ax.tick_params(axis="y", labelsize=7)
                fig.tight_layout()
                _save(fig, "fig06_relativity_heatmap.png")
            else:
                raise ValueError("No relativity rows with non-empty level.")
        else:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, "No relativity data available",
                    ha="center", va="center", transform=ax.transAxes)
            _save(fig, "fig06_relativity_heatmap.png")
    except Exception as exc:
        log.warning("Fig 06 failed: %s", exc)
        plt.close("all")

    # ------------------------------------------------------------------
    # Fig 07 — Factor importance (deviance reduction)
    # ------------------------------------------------------------------
    try:
        if len(step_log_df) > 0:
            slog = step_log_df.sort_values("delta_deviance", ascending=True).copy()
            total_dev = slog["delta_deviance"].sum()
            slog["cum_pct"] = slog["delta_deviance"].cumsum() / total_dev * 100

            fig, ax1 = plt.subplots(figsize=(9, 5))
            ax1.barh(slog["factor"], slog["delta_deviance"], color=C_ACCENT, alpha=0.85)
            ax1.set_xlabel("Deviance reduction", fontsize=10)
            ax1.set_title("Fig 07 — Factor Importance (Deviance Reduction)", **_TITLE_KW)
            ax1.grid(axis="x", alpha=0.3)

            ax2 = ax1.twinx()
            ax2.plot(slog["delta_deviance"].values, slog["cum_pct"].values,
                     "o-", color=C_RED, linewidth=2, markersize=5)
            ax2.set_ylabel("Cumulative % deviance explained", fontsize=9, color=C_RED)
            ax2.set_ylim(0, 110)
            ax2.axhline(95, color=C_RED, linestyle=":", linewidth=1)
            fig.tight_layout()
            _save(fig, "fig07_factor_importance.png")
        else:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, "No stepwise log data",
                    ha="center", va="center", transform=ax.transAxes)
            _save(fig, "fig07_factor_importance.png")
    except Exception as exc:
        log.warning("Fig 07 failed: %s", exc)
        plt.close("all")

    # ------------------------------------------------------------------
    # Fig 08 — A/E by factor level (top 6, test set, 2×3 grid)
    # ------------------------------------------------------------------
    try:
        top6 = (
            step_log_df.sort_values("delta_deviance", ascending=False)["factor"].head(6).tolist()
            if len(step_log_df) > 0
            else selected_factors[:6]
        )
        test_calib = calib_df[(calib_df["split"] == "TEST") & (calib_df["factor"].isin(top6))].copy()

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes_flat = axes.flatten()

        for ax_i, factor in enumerate(top6[:6]):
            ax = axes_flat[ax_i]
            fdata = test_calib[test_calib["factor"] == factor].copy()
            if len(fdata) == 0:
                ax.set_visible(False)
                continue
            fdata = fdata.sort_values("level", key=lambda s: s.astype(str))
            cols = [C_GREEN if v <= 1.02 else (C_AMBER if v <= 1.1 else C_RED)
                    for v in fdata["ae_ratio"].values]
            ax.bar(fdata["level"].astype(str), fdata["ae_ratio"], color=cols, alpha=0.85,
                   edgecolor="white")
            ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
            ax.set_title(factor, fontsize=10, color=C_PRIMARY, fontweight="bold")
            ax.set_xlabel("Level", fontsize=8)
            ax.set_ylabel("A/E", fontsize=8)
            ax.tick_params(axis="x", rotation=35, labelsize=7)
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(0.7, 1.4)

        for ax_i in range(len(top6), 6):
            axes_flat[ax_i].set_visible(False)

        fig.suptitle("Fig 08 — A/E by Factor Level (Top 6 Factors, Test Set)", **_TITLE_KW)
        fig.tight_layout()
        _save(fig, "fig08_actual_vs_expected_factors.png")
    except Exception as exc:
        log.warning("Fig 08 failed: %s", exc)
        plt.close("all")

    # ------------------------------------------------------------------
    # Fig 09 — VIF diagnostic
    # ------------------------------------------------------------------
    try:
        if len(vif_df) > 0:
            vdf = vif_df.sort_values("vif", ascending=True).tail(20)
            colours_vif = [C_RED if v > 10 else (C_AMBER if v > 5 else C_GREEN)
                           for v in vdf["vif"].values]
            fig, ax = plt.subplots(figsize=(9, max(4, len(vdf) * 0.35 + 1.5)))
            ax.barh(vdf["column"], vdf["vif"], color=colours_vif, alpha=0.85, edgecolor="white")
            ax.axvline(5, color=C_AMBER, linewidth=1.5, linestyle="--", label="VIF=5 threshold")
            ax.axvline(10, color=C_RED, linewidth=1.5, linestyle="--", label="VIF=10 threshold")
            ax.set_xlabel("Variance Inflation Factor", fontsize=10)
            ax.set_title("Fig 09 — VIF Diagnostic (top 20 features)", **_TITLE_KW)
            ax.legend(fontsize=9)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="x", alpha=0.3)
            fig.tight_layout()
            _save(fig, "fig09_vif_diagnostic.png")
        else:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, "VIF data not available",
                    ha="center", va="center", transform=ax.transAxes)
            _save(fig, "fig09_vif_diagnostic.png")
    except Exception as exc:
        log.warning("Fig 09 failed: %s", exc)
        plt.close("all")

    # ------------------------------------------------------------------
    # Fig 10 — Cook's distance index plot
    # ------------------------------------------------------------------
    try:
        infl = final_result.get_influence()
        if infl is not None:
            cooks_d, _ = infl.cooks_distance
            cooks_arr = np.asarray(cooks_d, dtype=float)
            n_obs = len(cooks_arr)
            threshold = 4.0 / n_obs if n_obs > 0 else 0.01

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.stem(
                np.arange(n_obs), cooks_arr,
                linefmt="grey", markerfmt=f"o", basefmt=" ",
            )
            # Colour top outliers red
            top10_idx = np.argsort(cooks_arr)[-10:]
            ax.scatter(top10_idx, cooks_arr[top10_idx], color=C_RED, zorder=5, s=40)
            for idx_c in top10_idx:
                if cooks_arr[idx_c] > threshold:
                    ax.annotate(
                        str(idx_c),
                        (idx_c, cooks_arr[idx_c]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7,
                    )
            ax.axhline(threshold, color=C_RED, linewidth=1.5, linestyle="--",
                       label=f"Threshold 4/n={threshold:.4f}")
            ax.set_xlabel("Observation index", fontsize=10)
            ax.set_ylabel("Cook's distance", fontsize=10)
            ax.set_title("Fig 10 — Cook's Distance", **_TITLE_KW)
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            _save(fig, "fig10_cooks_distance.png")
        else:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, "Cook's distance unavailable (sklearn backend)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11, color="grey")
            _save(fig, "fig10_cooks_distance.png")
    except Exception as exc:
        log.warning("Fig 10 failed: %s", exc)
        plt.close("all")

    # ------------------------------------------------------------------
    # Fig 11 — Coefficient forest plot (top 30 by magnitude)
    # ------------------------------------------------------------------
    try:
        params = final_result.params
        bse = final_result.bse
        pvals = final_result.pvalues

        coef_df = pd.DataFrame({
            "name": params.index,
            "coef": params.values,
            "se": bse.values if hasattr(bse, "values") else np.full(len(params), float("nan")),
            "pval": pvals.values if hasattr(pvals, "values") else np.full(len(params), float("nan")),
        })
        coef_df = coef_df[coef_df["name"] != "const"].copy()
        coef_df["abs_coef"] = coef_df["coef"].abs()
        coef_df = coef_df.sort_values("abs_coef", ascending=True).tail(30)

        fig, ax = plt.subplots(figsize=(9, max(5, len(coef_df) * 0.35 + 1.5)))
        y_pos = np.arange(len(coef_df))

        sig_mask = coef_df["pval"].fillna(1.0) < 0.05
        colours_coef = [C_ACCENT if s else "lightgrey" for s in sig_mask]
        ax.barh(y_pos, coef_df["coef"].values, color=colours_coef, alpha=0.85, edgecolor="white")

        # 95% CI whiskers
        valid_se = coef_df["se"].notna() & coef_df["se"].ne(float("nan"))
        for i, (_, row) in enumerate(coef_df.iterrows()):
            if not np.isnan(row["se"]):
                ax.errorbar(
                    row["coef"], i,
                    xerr=1.96 * row["se"],
                    fmt="none", color="black", capsize=3, linewidth=1.2,
                )

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(coef_df["name"].values, fontsize=7)
        ax.set_xlabel("Coefficient estimate (log scale)", fontsize=10)
        ax.set_title("Fig 11 — Coefficient Forest Plot (top 30 by magnitude)", **_TITLE_KW)
        ax.grid(axis="x", alpha=0.3)

        import matplotlib.patches as mpatches
        legend_items = [
            mpatches.Patch(color=C_ACCENT, label="Significant (p<0.05)"),
            mpatches.Patch(color="lightgrey", label="Not significant"),
        ]
        ax.legend(handles=legend_items, fontsize=9)
        fig.tight_layout()
        _save(fig, "fig11_coefficient_forest.png")
    except Exception as exc:
        log.warning("Fig 11 failed: %s", exc)
        plt.close("all")

    # ------------------------------------------------------------------
    # Fig 12 — Interaction effects heatmap
    # ------------------------------------------------------------------
    try:
        has_age_ncd = any(
            ("AGE_BAND" in pair and "NCD_CAPPED" in pair)
            for pair in [f"{f1}|{f2}" for f1, f2 in included_interactions]
        )

        if included_interactions:
            # Build AGE_BAND x NCD_CAPPED heatmap from relativities
            inter_rows = rel_df[rel_df["factor"] == "interaction"].copy()
            # Filter for AGE_BAND x NCD_CAPPED cross-terms
            age_ncd_rows = inter_rows[
                inter_rows["level"].str.contains("AGE_BAND", na=False) &
                inter_rows["level"].str.contains("NCD_CAPPED", na=False)
            ].copy()

            if len(age_ncd_rows) > 0:
                def _parse_age_ncd(level_str: str) -> Tuple[str, str]:
                    parts = level_str.split("__x__")
                    if len(parts) == 2:
                        age_part = parts[0].replace("AGE_BAND_", "")
                        ncd_part = parts[1].replace("NCD_CAPPED", "NCD")
                        return age_part, ncd_part
                    return level_str, "NCD"

                age_ncd_rows[["age_level", "ncd_level"]] = pd.DataFrame(
                    age_ncd_rows["level"].apply(_parse_age_ncd).tolist(),
                    index=age_ncd_rows.index,
                )
                pivot = age_ncd_rows.pivot_table(
                    index="age_level", columns="ncd_level", values="relativity", aggfunc="mean"
                )
                fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.2), max(5, len(pivot) * 0.6 + 1.5)))
                sns.heatmap(
                    pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                    center=1.0, ax=ax, linewidths=0.5,
                )
                ax.set_title("Fig 12 — Interaction Relativities: AGE_BAND x NCD_CAPPED", **_TITLE_KW)
            else:
                # Show first significant interaction as text table
                fig, ax = plt.subplots(figsize=(9, 5))
                inter_rows2 = inter_rows.head(20)
                if len(inter_rows2) > 0:
                    ax.barh(inter_rows2["level"].astype(str),
                            inter_rows2["relativity"].values,
                            color=C_ACCENT, alpha=0.85)
                    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
                    ax.set_xlabel("Relativity", fontsize=10)
                else:
                    ax.text(0.5, 0.5, f"Interactions included:\n{included_interactions}\n(No AGE_BAND×NCD_CAPPED terms)",
                            ha="center", va="center", transform=ax.transAxes, fontsize=10)
                ax.set_title("Fig 12 — Interaction Effects", **_TITLE_KW)
        else:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(
                0.5, 0.5,
                "No significant interactions included in the final model.\n"
                "All pre-specified pairs tested; none passed p<0.01 + 0.1% deviance threshold.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="grey", wrap=True,
            )
            ax.set_title("Fig 12 — Interaction Effects", **_TITLE_KW)
            ax.axis("off")

        fig.tight_layout()
        _save(fig, "fig12_interaction_effects.png")
    except Exception as exc:
        log.warning("Fig 12 failed: %s", exc)
        plt.close("all")

    # ------------------------------------------------------------------
    # Fig 13 — Train vs Test stability
    # ------------------------------------------------------------------
    try:
        train_diag: Dict[str, Any] = results["train_diag"]
        test_diag: Dict[str, Any] = results["test_diag"]

        metrics = ["gini", "ae_ratio", "mae", "rmse"]
        metric_labels = ["Gini", "A/E Ratio", "MAE (£)", "RMSE (£)"]
        train_vals = [train_diag.get(m, float("nan")) for m in metrics]
        test_vals = [test_diag.get(m, float("nan")) for m in metrics]

        fig, axes2 = plt.subplots(1, 4, figsize=(14, 5))
        for ax_j, (metric_lbl, tv, tev) in enumerate(zip(metric_labels, train_vals, test_vals)):
            ax = axes2[ax_j]
            vals = [tv, tev]
            valid = [v for v in vals if not np.isnan(v)]
            if not valid:
                ax.set_visible(False)
                continue
            bars2 = ax.bar(["Train", "Test"], vals, color=[C_PRIMARY, C_ACCENT], alpha=0.85,
                           edgecolor="white")
            for b, v in zip(bars2, vals):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, v * 1.01,
                            f"{v:.4f}", ha="center", va="bottom", fontsize=9)
            ax.set_title(metric_lbl, fontsize=10, color=C_PRIMARY, fontweight="bold")
            ax.set_ylabel(metric_lbl, fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            ymax = max(valid) * 1.2 if max(valid) > 0 else 1.0
            ax.set_ylim(0, ymax)

        fig.suptitle("Fig 13 — Train vs Test Stability", **_TITLE_KW)
        fig.tight_layout()
        _save(fig, "fig13_train_test_stability.png")
    except Exception as exc:
        log.warning("Fig 13 failed: %s", exc)
        plt.close("all")

    # ------------------------------------------------------------------
    # Fig 14 — Sensitivity analysis placeholder / results
    # ------------------------------------------------------------------
    try:
        sens_path = output_dir / "sensitivity_analysis.csv"
        if sens_path.exists():
            sens_df = pd.read_csv(sens_path)
            # 3-panel: Gini by experiment, AIC by experiment, base premium by experiment
            experiments = sens_df["experiment"].unique()
            fig, axes3 = plt.subplots(1, 3, figsize=(15, 5))
            panel_data = [
                ("gini_test", "Test Gini", axes3[0]),
                ("aic", "AIC", axes3[1]),
                ("base_premium", "Base Premium (£)", axes3[2]),
            ]
            for col_name, col_label, ax in panel_data:
                if col_name not in sens_df.columns:
                    ax.set_visible(False)
                    continue
                for exp in experiments:
                    edf = sens_df[sens_df["experiment"] == exp]
                    ax.plot(edf["variant"], edf[col_name], "o-", label=exp, linewidth=1.5)
                ax.set_title(col_label, fontsize=10, color=C_PRIMARY, fontweight="bold")
                ax.set_xlabel("Variant", fontsize=9)
                ax.tick_params(axis="x", rotation=20, labelsize=8)
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)

            fig.suptitle("Fig 14 — Sensitivity Analysis Results", **_TITLE_KW)
            fig.tight_layout()
        else:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.text(
                0.5, 0.5,
                "Fig 14 — Sensitivity Analysis\n\n"
                "Run with --sensitivity flag to populate this figure.\n"
                "Three experiments: cap sensitivity, imputation sensitivity,\n"
                "factor dropping (full vs top-6 vs without DTI).",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="grey",
            )
            ax.set_title("Fig 14 — Sensitivity Analysis (not run)", **_TITLE_KW)
            ax.axis("off")

        fig.tight_layout()
        _save(fig, "fig14_sensitivity_analysis.png")
    except Exception as exc:
        log.warning("Fig 14 failed: %s", exc)
        plt.close("all")

    log.info("  All figures saved to %s/", figures_dir)


# =============================================================================
# Section 11: HTML Dashboards
# =============================================================================

import base64 as _base64


def _img_to_base64(path: Path) -> str:
    """Read a PNG file and return a base64-encoded data URI string.

    Args:
        path: Absolute path to the image file.

    Returns:
        Base64-encoded string (no ``data:`` prefix) or empty string if the
        file does not exist.
    """
    if not path.exists():
        return ""
    with open(path, "rb") as fh:
        return _base64.b64encode(fh.read()).decode("ascii")


def generate_dashboards(results: Dict[str, Any]) -> None:
    """Generate two HTML dashboards in ``{output_dir}/figures/``.

    Produces:

    * ``glm_dashboard.html`` — Internal dashboard with embedded figures,
      download buttons, stats banner, and responsive card grid.
    * ``glm_dashboard_publish.html`` — Stakeholder dashboard with SHA-256
      password gate ("ageas2026"), Inter font, lightbox zoom, and no
      download buttons.

    Args:
        results: Full results dict returned by ``run_glm()``.
    """
    log.info("\n--- Section 11: Generating HTML Dashboards ---")

    output_dir: Path = results["output_dir"]
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    train_diag: Dict[str, Any] = results["train_diag"]
    test_diag: Dict[str, Any] = results["test_diag"]
    base_premium: float = results["base_premium"]
    selected_factors: List[str] = results["selected_factors"]
    final_result = results["final_result"]

    # ------------------------------------------------------------------
    # Figure manifest
    # ------------------------------------------------------------------
    FIGURES = [
        ("fig01_target_distribution.png", "Target Distribution"),
        ("fig02_deviance_residuals_qq.png", "Deviance Residuals QQ"),
        ("fig03_residuals_vs_fitted.png", "Residuals vs Fitted"),
        ("fig04_actual_vs_expected_decile.png", "A/E by Decile"),
        ("fig05_gini_lorenz.png", "Gini Lorenz Curves"),
        ("fig06_relativity_heatmap.png", "Relativity Heatmap"),
        ("fig07_factor_importance.png", "Factor Importance"),
        ("fig08_actual_vs_expected_factors.png", "A/E by Factor Level"),
        ("fig09_vif_diagnostic.png", "VIF Diagnostic"),
        ("fig10_cooks_distance.png", "Cook's Distance"),
        ("fig11_coefficient_forest.png", "Coefficient Forest Plot"),
        ("fig12_interaction_effects.png", "Interaction Effects"),
        ("fig13_train_test_stability.png", "Train/Test Stability"),
        ("fig14_sensitivity_analysis.png", "Sensitivity Analysis"),
    ]

    # Append CV figures if they exist (generated by run_cross_validation)
    for cv_fig in [
        ("fig15_cv_gini_boxplot.png", "CV: Gini Stability"),
        ("fig16_cv_factor_stability.png", "CV: Factor Selection Stability"),
    ]:
        if (figures_dir / cv_fig[0]).exists():
            FIGURES.append(cv_fig)

    # ------------------------------------------------------------------
    # Stats banner values
    # ------------------------------------------------------------------
    aic_val = train_diag.get("aic", float("nan"))
    aic_str = f"{aic_val:,.0f}" if not np.isnan(aic_val) else "N/A"
    train_ae = train_diag.get("ae_ratio", float("nan"))
    test_ae = test_diag.get("ae_ratio", float("nan"))
    ae_str = (
        f"{train_ae:.3f} / {test_ae:.3f}"
        if not (np.isnan(train_ae) or np.isnan(test_ae))
        else "N/A"
    )
    n_factors = len(selected_factors)
    n_params = final_result.n_params

    stats_items = [
        ("Base Premium", f"£{base_premium:,.2f}"),
        ("Train Gini", f"{train_diag.get('gini', 0):.4f}"),
        ("Test Gini", f"{test_diag.get('gini', 0):.4f}"),
        ("Factors", str(n_factors)),
        ("Parameters", str(n_params)),
        ("AIC", aic_str),
        ("A/E (Tr/Te)", ae_str),
        ("Train MAE", f"£{train_diag.get('mae', 0):,.2f}"),
        ("Test MAE", f"£{test_diag.get('mae', 0):,.2f}"),
    ]

    # Add CV metrics if available
    cv_summary = results.get("cv_summary")
    if cv_summary:
        stats_items.append(
            ("CV Gini (OOS)", f"{cv_summary['cv_gini_oos_mean']:.4f} ± {cv_summary['cv_gini_oos_std']:.4f}")
        )
        stats_items.append(("CV Folds", str(cv_summary["cv_folds"])))

    # ------------------------------------------------------------------
    # Build figure cards HTML (internal — with download buttons)
    # ------------------------------------------------------------------
    def _cards_html(include_download: bool, lightbox: bool) -> str:
        parts_html: List[str] = []
        for fname, title in FIGURES:
            img_path = figures_dir / fname
            b64 = _img_to_base64(img_path)
            if not b64:
                continue
            data_uri = f"data:image/png;base64,{b64}"
            dl_btn = (
                f'<a class="dl-btn" href="{data_uri}" download="{fname}">Download</a>'
                if include_download
                else ""
            )
            lb_open = f'onclick="openLightbox(\'{fname}\')"' if lightbox else ""
            card = f"""
    <div class="card">
      <h3 class="card-title">{title}</h3>
      <img src="{data_uri}" alt="{title}" class="fig-img" {lb_open}>
      {dl_btn}
    </div>"""
            parts_html.append(card)
        return "\n".join(parts_html)

    # ------------------------------------------------------------------
    # Build all base64 data URIs for lightbox (publish dashboard)
    # ------------------------------------------------------------------
    def _lightbox_js_data() -> str:
        entries: List[str] = []
        for fname, title in FIGURES:
            img_path = figures_dir / fname
            b64 = _img_to_base64(img_path)
            if b64:
                entries.append(
                    f'  "{fname}": {{ src: "data:image/png;base64,{b64}", title: "{title}" }}'
                )
        return "{\n" + ",\n".join(entries) + "\n}"

    # ------------------------------------------------------------------
    # Internal dashboard HTML
    # ------------------------------------------------------------------
    stats_banner_html = "\n".join(
        f'<div class="stat-item"><div class="stat-label">{lbl}</div>'
        f'<div class="stat-value">{val}</div></div>'
        for lbl, val in stats_items
    )

    internal_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ageas Direct — Net Premium GLM Dashboard</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #f0f2f5; color: #1a1a2e; }}
  .header {{
    background: linear-gradient(135deg, {C_PRIMARY} 0%, {C_ACCENT} 100%);
    color: white; padding: 24px 32px;
  }}
  .header h1 {{ font-size: 1.6rem; margin-bottom: 4px; }}
  .header p {{ font-size: 0.9rem; opacity: 0.85; }}
  .stats-banner {{
    display: flex; flex-wrap: wrap; gap: 12px;
    background: white; padding: 16px 32px;
    border-bottom: 2px solid #e5e7eb;
  }}
  .stat-item {{
    background: #f9fafb; border: 1px solid #e5e7eb;
    border-radius: 8px; padding: 10px 18px; min-width: 120px; text-align: center;
  }}
  .stat-label {{ font-size: 0.72rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }}
  .stat-value {{ font-size: 1.15rem; font-weight: 700; color: {C_PRIMARY}; margin-top: 2px; }}
  .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;
           padding: 24px 32px; max-width: 1600px; margin: 0 auto; }}
  @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  .card {{
    background: white; border-radius: 10px; padding: 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 1px solid #e5e7eb;
  }}
  .card-title {{ font-size: 0.95rem; font-weight: 600; color: {C_PRIMARY};
                 margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #f3f4f6; }}
  .fig-img {{ width: 100%; height: auto; border-radius: 6px; display: block; }}
  .dl-btn {{
    display: inline-block; margin-top: 10px; padding: 6px 14px;
    background: {C_ACCENT}; color: white; border-radius: 5px;
    text-decoration: none; font-size: 0.8rem; font-weight: 500;
  }}
  .dl-btn:hover {{ background: {C_PRIMARY}; }}
  .footer {{ text-align: center; padding: 24px; font-size: 0.8rem; color: #9ca3af; }}
</style>
</head>
<body>
<div class="header">
  <h1>Ageas Direct — Benchmark Net Premium Gamma GLM</h1>
  <p>Internal diagnostic dashboard &bull; Model output &bull; {n_factors} factors, {n_params} parameters</p>
</div>
<div class="stats-banner">
{stats_banner_html}
</div>
<div class="grid">
{_cards_html(include_download=True, lightbox=False)}
</div>
<div class="footer">Generated by build_net_premium_glm.py &bull; Ageas Direct UK Motor Insurance</div>
</body>
</html>"""

    internal_path = figures_dir / "glm_dashboard.html"
    internal_path.write_text(internal_html, encoding="utf-8")
    log.info("  Saved glm_dashboard.html")

    # ------------------------------------------------------------------
    # Stakeholder / publish dashboard HTML
    # ------------------------------------------------------------------
    # SHA-256 of "ageas2026"
    PASS_HASH = "c8d00b37a88dc019f68f637bcf9e5ffa7ccb03b81e4f235a26b391bd4e3e3517"

    publish_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ageas Direct — GLM Results</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: "Inter", -apple-system, sans-serif; background: #f0f2f5; color: #1a1a2e; }}

  /* --- Gate --- */
  #gate {{
    display: flex; align-items: center; justify-content: center;
    min-height: 100vh; background: linear-gradient(135deg, {C_PRIMARY} 0%, {C_ACCENT} 100%);
  }}
  .gate-box {{
    background: white; border-radius: 14px; padding: 48px 40px;
    text-align: center; max-width: 400px; width: 90%;
    box-shadow: 0 20px 60px rgba(0,0,0,0.25);
  }}
  .gate-box h2 {{ color: {C_PRIMARY}; font-size: 1.4rem; margin-bottom: 8px; }}
  .gate-box p {{ color: #6b7280; font-size: 0.88rem; margin-bottom: 28px; }}
  .gate-input {{
    width: 100%; padding: 12px 16px; border: 2px solid #e5e7eb;
    border-radius: 8px; font-size: 1rem; font-family: inherit; outline: none;
    transition: border-color 0.2s;
  }}
  .gate-input:focus {{ border-color: {C_ACCENT}; }}
  .gate-btn {{
    width: 100%; margin-top: 16px; padding: 13px;
    background: {C_PRIMARY}; color: white; border: none; border-radius: 8px;
    font-size: 1rem; font-weight: 600; cursor: pointer; font-family: inherit;
    transition: background 0.2s;
  }}
  .gate-btn:hover {{ background: {C_ACCENT}; }}
  .gate-error {{ color: {C_RED}; font-size: 0.85rem; margin-top: 12px; display: none; }}

  /* --- Dashboard --- */
  #dashboard {{ display: none; }}
  .header {{
    background: linear-gradient(135deg, {C_PRIMARY} 0%, {C_ACCENT} 100%);
    color: white; padding: 28px 40px;
  }}
  .header h1 {{ font-size: 1.7rem; font-weight: 700; margin-bottom: 6px; }}
  .header p {{ font-size: 0.9rem; opacity: 0.85; font-weight: 300; }}
  .stats-banner {{
    display: flex; flex-wrap: wrap; gap: 14px;
    background: white; padding: 20px 40px;
    border-bottom: 2px solid #e5e7eb;
  }}
  .stat-item {{
    background: #f9fafb; border: 1px solid #e5e7eb;
    border-radius: 10px; padding: 12px 20px; min-width: 130px; text-align: center;
  }}
  .stat-label {{ font-size: 0.7rem; color: #9ca3af; text-transform: uppercase;
                 letter-spacing: 0.08em; font-weight: 500; }}
  .stat-value {{ font-size: 1.2rem; font-weight: 700; color: {C_PRIMARY}; margin-top: 4px; }}
  .grid {{
    display: grid; grid-template-columns: repeat(2, 1fr);
    gap: 22px; padding: 28px 40px; max-width: 1600px; margin: 0 auto;
  }}
  @media (max-width: 1200px) {{ .grid {{ padding: 20px 24px; }} }}
  @media (max-width: 768px) {{ .grid {{ grid-template-columns: 1fr; padding: 16px; }} }}
  @media (max-width: 480px) {{ .stats-banner {{ padding: 14px 16px; }} .header {{ padding: 20px 16px; }} }}
  @media (max-width: 400px) {{ .gate-box {{ padding: 32px 20px; }} }}
  .card {{
    background: white; border-radius: 12px; padding: 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07); border: 1px solid #e9ecef;
    transition: box-shadow 0.2s;
  }}
  .card:hover {{ box-shadow: 0 6px 24px rgba(0,0,0,0.12); }}
  .card-title {{ font-size: 0.92rem; font-weight: 600; color: {C_PRIMARY};
                 margin-bottom: 14px; padding-bottom: 10px;
                 border-bottom: 1px solid #f3f4f6; }}
  .fig-img {{
    width: 100%; height: auto; border-radius: 8px; display: block;
    cursor: zoom-in; transition: opacity 0.15s;
  }}
  .fig-img:hover {{ opacity: 0.92; }}
  .footer {{ text-align: center; padding: 28px; font-size: 0.78rem; color: #9ca3af; }}

  /* --- Lightbox --- */
  #lightbox {{
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,0.88); z-index: 9999;
    align-items: center; justify-content: center; flex-direction: column;
  }}
  #lightbox.active {{ display: flex; }}
  #lightbox img {{ max-width: 92vw; max-height: 88vh; border-radius: 8px; box-shadow: 0 8px 40px rgba(0,0,0,0.5); }}
  #lightbox-title {{ color: white; font-size: 1rem; margin-top: 14px; font-weight: 500; }}
  #lightbox-close {{
    position: absolute; top: 18px; right: 24px;
    color: white; font-size: 2rem; cursor: pointer; font-weight: 300; line-height: 1;
  }}
</style>
</head>
<body>

<!-- Password gate -->
<div id="gate">
  <div class="gate-box">
    <h2>Ageas Direct</h2>
    <p>Net Premium GLM Results &mdash; Restricted Access</p>
    <input type="password" id="pwd-input" class="gate-input" placeholder="Enter password" />
    <button class="gate-btn" onclick="checkPassword()">View Dashboard</button>
    <div id="gate-error" class="gate-error">Incorrect password. Please try again.</div>
  </div>
</div>

<!-- Lightbox -->
<div id="lightbox" onclick="closeLightbox()">
  <span id="lightbox-close" onclick="closeLightbox()">&times;</span>
  <img id="lightbox-img" src="" alt="">
  <div id="lightbox-title"></div>
</div>

<!-- Main dashboard (hidden until password correct) -->
<div id="dashboard">
  <div class="header">
    <h1>Ageas Direct &mdash; Net Premium Gamma GLM</h1>
    <p>Stakeholder diagnostic dashboard &bull; {n_factors} factors &bull; {n_params} parameters</p>
  </div>
  <div class="stats-banner">
{stats_banner_html}
  </div>
  <div class="grid">
{_cards_html(include_download=False, lightbox=True)}
  </div>
  <div class="footer">Ageas Direct UK Motor Insurance &bull; Benchmark Net Premium GLM &bull; Confidential</div>
</div>

<script>
const PASS_HASH = "{PASS_HASH}";
const FIGURES = {_lightbox_js_data()};

async function sha256(message) {{
  const msgBuffer = new TextEncoder().encode(message);
  const hashBuffer = await crypto.subtle.digest("SHA-256", msgBuffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, "0")).join("");
}}

async function checkPassword() {{
  const pwd = document.getElementById("pwd-input").value;
  const hash = await sha256(pwd);
  if (hash === PASS_HASH) {{
    document.getElementById("gate").style.display = "none";
    document.getElementById("dashboard").style.display = "block";
    document.getElementById("gate-error").style.display = "none";
  }} else {{
    document.getElementById("gate-error").style.display = "block";
  }}
}}

document.getElementById("pwd-input").addEventListener("keydown", function(e) {{
  if (e.key === "Enter") checkPassword();
}});

function openLightbox(fname) {{
  const data = FIGURES[fname];
  if (!data) return;
  document.getElementById("lightbox-img").src = data.src;
  document.getElementById("lightbox-title").textContent = data.title;
  document.getElementById("lightbox").classList.add("active");
}}

function closeLightbox() {{
  document.getElementById("lightbox").classList.remove("active");
  document.getElementById("lightbox-img").src = "";
}}

document.addEventListener("keydown", function(e) {{
  if (e.key === "Escape") closeLightbox();
}});
</script>
</body>
</html>"""

    publish_path = figures_dir / "glm_dashboard_publish.html"
    publish_path.write_text(publish_html, encoding="utf-8")
    log.info("  Saved glm_dashboard_publish.html")
    log.info("  Dashboards written to %s/", figures_dir)


# =============================================================================
# Section 12: Sensitivity Analysis
# =============================================================================


def run_sensitivity_analysis(results: Dict[str, Any]) -> None:
    """Run three sensitivity experiments and save a comparison CSV.

    Only executes when ``config.run_sensitivity`` is True.  Three experiments:

    1. **Premium cap sensitivity** — re-fit at P99 and hard cap £10K, compare
       Gini/AIC/base premium against the default P99.5 model.
    2. **Imputation sensitivity** — compare full-data model vs. model fitted
       after excluding rows where ``AGE_IMPUTED=1`` or
       ``GROSSVEHICLEWEIGHT_K_IMPUTED=1``.
    3. **Factor dropping** — compare full model vs. top-6-factor model vs.
       model without DTI (Equality Act concern).

    Results are saved to ``sensitivity_analysis.csv``.

    Args:
        results: Full results dict returned by ``run_glm()``.
    """
    config: GLMConfig = results["config"]

    if not config.run_sensitivity:
        log.info("  Sensitivity analysis skipped (--sensitivity flag not set).")
        return

    log.info("\n--- Section 12: Sensitivity Analysis ---")

    output_dir: Path = results["output_dir"]
    train_df: pd.DataFrame = results["train_df"]
    test_df: pd.DataFrame = results["test_df"]
    y_test: pd.Series = results["y_test"]
    selected_factors: List[str] = results["selected_factors"]
    step_log_df: pd.DataFrame = results["step_log_df"]
    baseline_train_diag: Dict[str, Any] = results["train_diag"]
    baseline_test_diag: Dict[str, Any] = results["test_diag"]
    baseline_base_premium: float = results["base_premium"]

    rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Helper — fit model and compute metrics for a given train set / cap
    # ------------------------------------------------------------------
    def _run_variant(
        tr_df: pd.DataFrame,
        te_df: pd.DataFrame,
        y_te: pd.Series,
        factors: List[str],
        y_tr: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """Fit a Gamma GLM and return key diagnostics.

        Args:
            tr_df: Training DataFrame (already consolidated).
            te_df: Test DataFrame (already consolidated).
            y_te: Test response vector.
            factors: Factor list to use.
            y_tr: Training response vector; if None, uses ``AD_POLPREMIUM_CAPPED``
                  from ``tr_df``.

        Returns:
            Dict with keys: gini_train, gini_test, aic, base_premium, n_factors, n_params.
        """
        if y_tr is None:
            y_tr_use = tr_df["AD_POLPREMIUM_CAPPED"] if "AD_POLPREMIUM_CAPPED" in tr_df.columns else tr_df["AD_POLPREMIUM"]
        else:
            y_tr_use = y_tr

        X_tr = prepare_design_matrix(tr_df, factors)
        X_te_raw = prepare_design_matrix(te_df, factors)
        X_te = align_test_matrix(X_tr, X_te_raw)

        try:
            res = fit_gamma_glm(X_tr, y_tr_use)
        except Exception as exc:
            log.warning("  Variant fit failed: %s", exc)
            return {
                "gini_train": float("nan"), "gini_test": float("nan"),
                "aic": float("nan"), "base_premium": float("nan"),
                "n_factors": len(factors), "n_params": 0,
            }

        tr_diag = compute_diagnostics(res, X_tr, y_tr_use, "TRAIN")
        te_diag = compute_diagnostics(res, X_te, y_te, "TEST")
        bp, _ = extract_relativities(res, tr_df, factors)

        return {
            "gini_train": tr_diag["gini"],
            "gini_test": te_diag["gini"],
            "aic": tr_diag.get("aic", float("nan")),
            "base_premium": bp,
            "n_factors": len(factors),
            "n_params": res.n_params,
        }

    # ------------------------------------------------------------------
    # Baseline row
    # ------------------------------------------------------------------
    rows.append({
        "experiment": "baseline",
        "variant": f"P{config.cap_percentile:.0f}",
        "gini_train": baseline_train_diag["gini"],
        "gini_test": baseline_test_diag["gini"],
        "aic": baseline_train_diag.get("aic", float("nan")),
        "base_premium": baseline_base_premium,
        "n_factors": len(selected_factors),
        "n_params": results["final_result"].n_params,
    })

    # ------------------------------------------------------------------
    # Experiment 1 — Premium cap sensitivity
    # ------------------------------------------------------------------
    log.info("  Experiment 1: Premium cap sensitivity")

    for cap_label, cap_fn in [
        ("P99", lambda y: float(np.percentile(y.dropna(), 99))),
        ("hard_10K", lambda y: 10_000.0),
    ]:
        try:
            raw_y = train_df["AD_POLPREMIUM"] if "AD_POLPREMIUM" in train_df.columns else results["y_train"]
            cap = cap_fn(raw_y)
            tr_df_cap = train_df.copy()
            tr_df_cap["AD_POLPREMIUM_CAPPED"] = raw_y.clip(upper=cap)
            y_tr_cap = tr_df_cap["AD_POLPREMIUM_CAPPED"]

            variant_metrics = _run_variant(tr_df_cap, test_df, y_test, selected_factors, y_tr=y_tr_cap)
            rows.append({
                "experiment": "cap_sensitivity",
                "variant": cap_label,
                **variant_metrics,
            })
            log.info(
                "    %s: Gini test=%.4f, AIC=%.1f, base=£%.2f",
                cap_label,
                variant_metrics["gini_test"],
                variant_metrics["aic"] if not np.isnan(variant_metrics["aic"]) else -1,
                variant_metrics["base_premium"],
            )
        except Exception as exc:
            log.warning("  Cap sensitivity variant %s failed: %s", cap_label, exc)
            rows.append({
                "experiment": "cap_sensitivity", "variant": cap_label,
                "gini_train": float("nan"), "gini_test": float("nan"),
                "aic": float("nan"), "base_premium": float("nan"),
                "n_factors": len(selected_factors), "n_params": 0,
            })

    # ------------------------------------------------------------------
    # Experiment 2 — Imputation sensitivity
    # ------------------------------------------------------------------
    log.info("  Experiment 2: Imputation sensitivity")

    try:
        full_csv = pd.read_csv(config.input_path, low_memory=False)
        imputed_cols_all = [c for c in full_csv.columns if c.endswith("_IMPUTED")]

        train_full_raw = full_csv[full_csv["SPLIT"] == "TRAIN"].copy()
        test_full_raw = full_csv[full_csv["SPLIT"] == "TEST"].copy()

        # Variant: exclude rows with key imputation flags
        impute_filter_cols = [c for c in ["AGE_IMPUTED", "GROSSVEHICLEWEIGHT_K_IMPUTED"]
                              if c in train_full_raw.columns]

        if impute_filter_cols:
            mask_imputed = (train_full_raw[impute_filter_cols] == 1).any(axis=1)
            train_no_impute = train_full_raw[~mask_imputed].copy()

            # Exclude leakage columns (mirror load_data logic)
            all_excl = set(EXCLUDE_COLS) | set(imputed_cols_all)
            drop_cols = [c for c in all_excl if c in train_no_impute.columns and c != "AD_POLPREMIUM"]
            train_no_impute = train_no_impute.drop(columns=drop_cols, errors="ignore")
            drop_cols_te = [c for c in all_excl if c in test_full_raw.columns and c != "AD_POLPREMIUM"]
            test_no_impute = test_full_raw.drop(columns=drop_cols_te, errors="ignore")

            train_no_impute = consolidate_categoricals(train_no_impute)
            test_no_impute = consolidate_categoricals(test_no_impute)

            # Cap at P99.5 of the filtered subset
            cap_ni = float(np.percentile(
                train_no_impute["AD_POLPREMIUM"].dropna(), config.cap_percentile
            ))
            train_no_impute["AD_POLPREMIUM_CAPPED"] = train_no_impute["AD_POLPREMIUM"].clip(upper=cap_ni)
            y_ni = train_no_impute["AD_POLPREMIUM_CAPPED"]

            # Test target
            y_te_ni = test_no_impute["AD_POLPREMIUM"].copy()

            # Factors available in the subset
            avail_factors = [f for f in selected_factors if f in train_no_impute.columns]

            metrics_full = _run_variant(train_df, test_df, y_test, selected_factors)
            metrics_no_impute = _run_variant(train_no_impute, test_no_impute, y_te_ni, avail_factors, y_tr=y_ni)

            rows.append({
                "experiment": "imputation_sensitivity",
                "variant": "full_data",
                **metrics_full,
            })
            rows.append({
                "experiment": "imputation_sensitivity",
                "variant": f"exclude_imputed_n{int((~mask_imputed).sum())}",
                **metrics_no_impute,
            })

            log.info(
                "    full_data: Gini test=%.4f  |  exclude_imputed: Gini test=%.4f",
                metrics_full["gini_test"],
                metrics_no_impute["gini_test"],
            )
        else:
            log.warning("  Imputation columns not found — skipping Experiment 2.")
            rows.append({
                "experiment": "imputation_sensitivity", "variant": "skipped_no_imputation_cols",
                "gini_train": float("nan"), "gini_test": float("nan"),
                "aic": float("nan"), "base_premium": float("nan"),
                "n_factors": len(selected_factors), "n_params": 0,
            })
    except Exception as exc:
        log.warning("  Imputation sensitivity failed: %s", exc)
        rows.append({
            "experiment": "imputation_sensitivity", "variant": "error",
            "gini_train": float("nan"), "gini_test": float("nan"),
            "aic": float("nan"), "base_premium": float("nan"),
            "n_factors": len(selected_factors), "n_params": 0,
        })

    # ------------------------------------------------------------------
    # Experiment 3 — Factor dropping
    # ------------------------------------------------------------------
    log.info("  Experiment 3: Factor dropping")

    # 3a. Top 6 factors (parsimonious)
    top6 = (
        step_log_df.sort_values("delta_deviance", ascending=False)["factor"].head(6).tolist()
        if len(step_log_df) > 0
        else selected_factors[:6]
    )
    try:
        metrics_top6 = _run_variant(train_df, test_df, y_test, top6, y_tr=results["y_train"])
        rows.append({
            "experiment": "factor_dropping",
            "variant": "top6_parsimonious",
            **metrics_top6,
        })
        log.info("    top6: Gini test=%.4f, AIC=%.1f", metrics_top6["gini_test"],
                 metrics_top6["aic"] if not np.isnan(metrics_top6["aic"]) else -1)
    except Exception as exc:
        log.warning("  Factor dropping (top6) failed: %s", exc)
        rows.append({
            "experiment": "factor_dropping", "variant": "top6_parsimonious",
            "gini_train": float("nan"), "gini_test": float("nan"),
            "aic": float("nan"), "base_premium": float("nan"),
            "n_factors": 6, "n_params": 0,
        })

    # 3b. Full model (already in baseline, duplicate here for readability)
    rows.append({
        "experiment": "factor_dropping",
        "variant": "full_model",
        "gini_train": baseline_train_diag["gini"],
        "gini_test": baseline_test_diag["gini"],
        "aic": baseline_train_diag.get("aic", float("nan")),
        "base_premium": baseline_base_premium,
        "n_factors": len(selected_factors),
        "n_params": results["final_result"].n_params,
    })

    # 3c. Without DTI (Equality Act concern)
    dti_col = "DTI"
    factors_no_dti = [f for f in selected_factors if f != dti_col]
    if len(factors_no_dti) < len(selected_factors):
        try:
            metrics_no_dti = _run_variant(
                train_df, test_df, y_test, factors_no_dti, y_tr=results["y_train"]
            )
            rows.append({
                "experiment": "factor_dropping",
                "variant": "without_DTI",
                **metrics_no_dti,
            })
            log.info("    without_DTI: Gini test=%.4f", metrics_no_dti["gini_test"])
        except Exception as exc:
            log.warning("  Factor dropping (no DTI) failed: %s", exc)
            rows.append({
                "experiment": "factor_dropping", "variant": "without_DTI",
                "gini_train": float("nan"), "gini_test": float("nan"),
                "aic": float("nan"), "base_premium": float("nan"),
                "n_factors": len(factors_no_dti), "n_params": 0,
            })
    else:
        log.info("  DTI not in selected factors — without_DTI variant identical to full model.")
        rows.append({
            "experiment": "factor_dropping",
            "variant": "without_DTI_not_applicable",
            "gini_train": baseline_train_diag["gini"],
            "gini_test": baseline_test_diag["gini"],
            "aic": baseline_train_diag.get("aic", float("nan")),
            "base_premium": baseline_base_premium,
            "n_factors": len(selected_factors),
            "n_params": results["final_result"].n_params,
        })

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    sens_df = pd.DataFrame(rows)
    out_path = output_dir / "sensitivity_analysis.csv"
    sens_df.to_csv(out_path, index=False)
    log.info("  Saved sensitivity_analysis.csv (%d rows)", len(sens_df))

    # Summary table
    log.info(
        "\n  Sensitivity summary:\n%s",
        sens_df[["experiment", "variant", "gini_test", "aic", "base_premium", "n_factors"]].to_string(index=False),
    )


# =============================================================================
# Section 14: K-Fold Cross-Validation
# =============================================================================


def _generate_cv_visualizations(
    figures_dir: Path,
    cv_results_df: pd.DataFrame,
    factor_stability_df: pd.DataFrame,
    k: int,
) -> None:
    """Generate fig15 (Gini boxplot) and fig16 (factor stability heatmap)."""
    if not HAS_MATPLOTLIB:
        log.warning("  matplotlib not available — skipping CV figures.")
        return

    matplotlib.use("Agg")
    _TITLE_KW = {"color": C_PRIMARY, "fontweight": "bold", "fontsize": 12}
    _SAVE_KW = {"dpi": 150, "bbox_inches": "tight", "facecolor": "white"}

    # Fig 15 — CV Gini Box/Strip Plot
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        gini_long = pd.DataFrame({
            "Fold": list(cv_results_df["fold"]) * 2,
            "Gini": list(cv_results_df["gini_train"]) + list(cv_results_df["gini_val"]),
            "Split": ["Train"] * len(cv_results_df) + ["Validation"] * len(cv_results_df),
        })
        sns.boxplot(data=gini_long, x="Split", y="Gini", hue="Split",
                    palette=[C_PRIMARY, C_ACCENT], width=0.4, legend=False, ax=ax)
        sns.stripplot(data=gini_long, x="Split", y="Gini",
                      color="black", size=6, alpha=0.7, jitter=True, legend=False, ax=ax)
        for i, split in enumerate(["Train", "Validation"]):
            vals = gini_long[gini_long["Split"] == split]["Gini"]
            ax.annotate(f"{vals.mean():.4f} ± {vals.std():.4f}",
                        xy=(i, vals.mean()), xytext=(i + 0.3, vals.mean()),
                        fontsize=9, color=C_PRIMARY, fontweight="bold")
        ax.set_title(f"{k}-Fold Cross-Validation: Gini Coefficient Distribution", **_TITLE_KW)
        ax.set_ylabel("Gini Coefficient")
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / "fig15_cv_gini_boxplot.png", **_SAVE_KW)
        plt.close(fig)
        log.info("  Saved fig15_cv_gini_boxplot.png")
    except Exception as exc:
        log.warning("  Fig 15 failed: %s", exc)
        plt.close("all")

    # Fig 16 — Factor Selection Stability Heatmap
    try:
        fold_cols = [c for c in factor_stability_df.columns if c.startswith("fold_")]
        heatmap_data = factor_stability_df.set_index("factor")[fold_cols].copy()
        heatmap_data["_rate"] = factor_stability_df["selection_rate"].values
        heatmap_data = heatmap_data.sort_values("_rate", ascending=False)
        heatmap_data = heatmap_data.drop(columns=["_rate"])

        fig, ax = plt.subplots(figsize=(max(6, k * 1.2), max(6, len(heatmap_data) * 0.45)))
        sns.heatmap(
            heatmap_data.astype(float), annot=True, fmt=".0f",
            cmap=["#f3f4f6", C_GREEN], cbar=False,
            linewidths=0.5, linecolor="white",
            xticklabels=[f"Fold {i+1}" for i in range(k)],
            yticklabels=heatmap_data.index, ax=ax,
        )
        for i, factor in enumerate(heatmap_data.index):
            rate = factor_stability_df.loc[
                factor_stability_df["factor"] == factor, "selection_rate"
            ].iloc[0]
            colour = C_GREEN if rate >= 0.8 else (C_AMBER if rate >= 0.4 else C_RED)
            ax.text(k + 0.3, i + 0.5, f"{rate:.0%}",
                    va="center", fontsize=9, fontweight="bold", color=colour)
        ax.set_title(f"{k}-Fold CV: Factor Selection Stability", **_TITLE_KW)
        ax.set_xlabel("")
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(figures_dir / "fig16_cv_factor_stability.png", **_SAVE_KW)
        plt.close(fig)
        log.info("  Saved fig16_cv_factor_stability.png")
    except Exception as exc:
        log.warning("  Fig 16 failed: %s", exc)
        plt.close("all")


def run_cross_validation(results: Dict[str, Any]) -> None:
    """Run k-fold cross-validation on training data to assess model stability.

    Each fold runs the full pipeline: stepwise selection, interaction testing,
    final model fitting, and diagnostics.  Only executes when cv_folds >= 2.

    Args:
        results: Full results dict returned by run_glm().
    """
    config: GLMConfig = results["config"]
    if config.cv_folds < 2:
        log.info("  Cross-validation skipped (--cv-folds=%d).", config.cv_folds)
        return

    k = config.cv_folds
    log.info("\n--- Section 14: %d-Fold Cross-Validation ---", k)
    t0_cv = time.time()

    output_dir: Path = results["output_dir"]
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    train_df: pd.DataFrame = results["train_df"]
    y_train: pd.Series = results["y_train"]
    primary_selected: List[str] = results["selected_factors"]
    n = len(train_df)

    # Build fold indices (seeded shuffle, no sklearn dependency)
    rng = np.random.default_rng(config.seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        folds.append((train_idx, val_idx))

    # Per-fold storage
    fold_results: List[Dict[str, Any]] = []
    fold_factors: Dict[str, List[int]] = {f: [] for f in ALL_CANDIDATE_FACTORS}
    fold_relativities: List[pd.DataFrame] = []
    fold_gini_oos: List[float] = []
    fold_gini_train: List[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        log.info("  Fold %d/%d  (train=%d, val=%d)",
                 fold_idx + 1, k, len(train_idx), len(val_idx))

        cv_train = train_df.iloc[train_idx].copy()
        cv_val = train_df.iloc[val_idx].copy()
        cv_y_train = y_train.iloc[train_idx]
        cv_y_val = y_train.iloc[val_idx]

        # 1. Stepwise factor selection
        try:
            fold_selected, _ = stepwise_select(cv_train, cv_y_train, ALL_CANDIDATE_FACTORS)
        except Exception as exc:
            log.warning("  Fold %d stepwise failed: %s", fold_idx + 1, exc)
            for f in ALL_CANDIDATE_FACTORS:
                fold_factors[f].append(0)
            continue

        if not fold_selected:
            fold_selected = ALL_CANDIDATE_FACTORS[:6]

        for f in ALL_CANDIDATE_FACTORS:
            fold_factors[f].append(1 if f in fold_selected else 0)

        # 2. Fit main-effects model
        X_cv_train = prepare_design_matrix(cv_train, fold_selected)
        try:
            fold_main_result = fit_gamma_glm(X_cv_train, cv_y_train)
        except Exception as exc:
            log.warning("  Fold %d main fit failed: %s", fold_idx + 1, exc)
            continue

        # 3. Test interactions
        fold_interactions: List[Tuple[str, str]] = []
        if config.run_interactions and len(fold_selected) >= 2:
            try:
                fold_interactions, _ = test_interactions(
                    cv_train, cv_y_train, fold_selected, fold_main_result
                )
            except Exception:
                pass

        # 4. Final model with interactions
        if fold_interactions:
            X_cv_train_final = _build_interaction_columns(
                cv_train, X_cv_train, fold_interactions
            )
            try:
                fold_final_result = fit_gamma_glm(X_cv_train_final, cv_y_train)
            except Exception:
                X_cv_train_final = X_cv_train
                fold_final_result = fold_main_result
        else:
            X_cv_train_final = X_cv_train
            fold_final_result = fold_main_result

        # 5. Train diagnostics
        train_diag = compute_diagnostics(fold_final_result, X_cv_train_final, cv_y_train, "CV_TRAIN")

        # 6. Validation diagnostics
        X_cv_val = prepare_design_matrix(cv_val, fold_selected)
        if fold_interactions:
            X_cv_val = _build_interaction_columns(cv_val, X_cv_val, fold_interactions)
        X_cv_val = align_test_matrix(X_cv_train_final, X_cv_val)
        val_diag = compute_diagnostics(fold_final_result, X_cv_val, cv_y_val, "CV_VAL")

        # 7. Relativities
        try:
            fold_base, fold_rel = extract_relativities(fold_final_result, cv_train, fold_selected)
        except Exception:
            fold_base = float("nan")
            fold_rel = pd.DataFrame()

        if not fold_rel.empty:
            fold_rel["fold"] = fold_idx + 1
            fold_relativities.append(fold_rel)

        fold_gini_oos.append(val_diag["gini"])
        fold_gini_train.append(train_diag["gini"])

        fold_results.append({
            "fold": fold_idx + 1,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_factors": len(fold_selected),
            "factors": ", ".join(fold_selected),
            "n_interactions": len(fold_interactions),
            "gini_train": round(train_diag["gini"], 6),
            "gini_val": round(val_diag["gini"], 6),
            "mae_train": round(train_diag["mae"], 4),
            "mae_val": round(val_diag["mae"], 4),
            "ae_ratio_val": round(val_diag["ae_ratio"], 6),
            "base_premium": round(fold_base, 2),
            "aic": round(train_diag.get("aic", float("nan")), 2),
            "deviance": round(train_diag.get("deviance", float("nan")), 2),
        })

        log.info("    Gini train=%.4f  val=%.4f  base=£%.2f  factors=%d",
                 train_diag["gini"], val_diag["gini"], fold_base, len(fold_selected))

    if not fold_results:
        log.warning("  No folds completed — CV aborted.")
        return

    # --- Save cv_results.csv ---
    cv_results_df = pd.DataFrame(fold_results)
    cv_results_df.to_csv(output_dir / "cv_results.csv", index=False)
    log.info("  Saved cv_results.csv (%d folds)", len(cv_results_df))

    # --- Save cv_factor_stability.csv ---
    factor_stability_rows = []
    for factor in ALL_CANDIDATE_FACTORS:
        selections = fold_factors.get(factor, [0] * k)
        row: Dict[str, Any] = {"factor": factor}
        for i, sel in enumerate(selections):
            row[f"fold_{i+1}"] = sel
        row["selection_rate"] = round(sum(selections) / len(selections), 4) if selections else 0.0
        row["in_primary_model"] = 1 if factor in primary_selected else 0
        factor_stability_rows.append(row)
    factor_stability_df = pd.DataFrame(factor_stability_rows)
    factor_stability_df.to_csv(output_dir / "cv_factor_stability.csv", index=False)
    log.info("  Saved cv_factor_stability.csv")

    # --- Save cv_coefficient_stability.csv ---
    coeff_stability = pd.DataFrame()
    if fold_relativities:
        all_rels = pd.concat(fold_relativities, ignore_index=True)
        coeff_stability = (
            all_rels.groupby(["factor", "level"])["relativity"]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
        )
        coeff_stability["cv"] = (coeff_stability["std"] / coeff_stability["mean"]).round(6)
        coeff_stability = coeff_stability.rename(columns={
            "mean": "relativity_mean", "std": "relativity_std",
            "min": "relativity_min", "max": "relativity_max",
            "count": "n_folds_present",
        })
        coeff_stability.to_csv(output_dir / "cv_coefficient_stability.csv", index=False)
        log.info("  Saved cv_coefficient_stability.csv")

    # --- Aggregate CV summary ---
    gini_oos_arr = np.array(fold_gini_oos)
    gini_train_arr = np.array(fold_gini_train)
    base_arr = np.array([r["base_premium"] for r in fold_results])

    cv_summary = {
        "cv_folds": k,
        "cv_gini_oos_mean": round(float(gini_oos_arr.mean()), 6),
        "cv_gini_oos_std": round(float(gini_oos_arr.std()), 6),
        "cv_gini_oos_min": round(float(gini_oos_arr.min()), 6),
        "cv_gini_oos_max": round(float(gini_oos_arr.max()), 6),
        "cv_gini_train_mean": round(float(gini_train_arr.mean()), 6),
        "cv_gini_train_std": round(float(gini_train_arr.std()), 6),
        "cv_base_premium_mean": round(float(base_arr.mean()), 2),
        "cv_base_premium_std": round(float(base_arr.std()), 2),
        "cv_base_premium_cv": round(float(base_arr.std() / base_arr.mean()), 6)
        if base_arr.mean() > 0 else None,
        "cv_factor_unanimity": int(
            (factor_stability_df["selection_rate"].isin([0.0, 1.0])).sum()
        ),
        "cv_n_factors_mean": round(float(cv_results_df["n_factors"].mean()), 2),
    }

    # Update model_summary.json
    summary_path = output_dir / "model_summary.json"
    if summary_path.exists():
        with open(summary_path) as fh:
            existing_summary = json.load(fh)
        existing_summary["cross_validation"] = cv_summary
        with open(summary_path, "w") as fh:
            json.dump(existing_summary, fh, indent=2, default=str)
        log.info("  Updated model_summary.json with CV metrics")

    # Store in results dict for downstream hooks
    results["cv_summary"] = cv_summary
    results["cv_results_df"] = cv_results_df
    results["cv_factor_stability_df"] = factor_stability_df
    results["cv_coefficient_stability_df"] = coeff_stability

    # Generate CV visualizations
    _generate_cv_visualizations(figures_dir, cv_results_df, factor_stability_df, k)

    elapsed_cv = time.time() - t0_cv
    log.info("  Cross-validation complete in %.1fs", elapsed_cv)

    print(f"\n  CV Gini (OOS):  {cv_summary['cv_gini_oos_mean']:.4f} "
          f"± {cv_summary['cv_gini_oos_std']:.4f}")
    print(f"  CV Base Prem:   £{cv_summary['cv_base_premium_mean']:,.2f} "
          f"± £{cv_summary['cv_base_premium_std']:,.2f}")
    print(f"  Factor unanimity: "
          f"{cv_summary['cv_factor_unanimity']}/{len(ALL_CANDIDATE_FACTORS)}")


def main() -> None:
    """Parse CLI arguments, build GLMConfig, and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Benchmark Net Premium Gamma GLM — Ageas Direct UK Motor Insurance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="data_to_be_cleaned/net/net_glm_ready.csv",
        help="Path to GLM-ready CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data_to_be_cleaned/net/glm_results",
        help="Directory for output artefacts.",
    )
    parser.add_argument(
        "--cap",
        type=float,
        default=None,
        metavar="VALUE",
        help="Hard premium cap in £ (overrides --cap-percentile).",
    )
    parser.add_argument(
        "--cap-percentile",
        type=float,
        default=99.5,
        metavar="PCT",
        help="Percentile for premium winsorisation (0-100).",
    )
    parser.add_argument(
        "--no-interactions",
        action="store_true",
        help="Skip interaction testing.",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run sensitivity analysis (Section 11).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode: subsample 1000 training rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        metavar="K",
        help="Number of cross-validation folds (0=disabled, 5=standard).",
    )
    args = parser.parse_args()

    config = GLMConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        seed=args.seed,
        cap_percentile=args.cap_percentile,
        cap_value=args.cap,
        run_interactions=not args.no_interactions,
        run_sensitivity=args.sensitivity,
        quick=args.quick,
        cv_folds=args.cv_folds,
    )

    results = run_glm(config)

    # Sections 9-12 hooks — called when appended sections define these functions.
    # Each function receives the full results dict returned by run_glm().
    for fn_name in [
        "run_cross_validation",
        "generate_visualizations",
        "generate_dashboards",
        "run_sensitivity_analysis",
        "fit_parsimonious_model",
    ]:
        fn = globals().get(fn_name)
        if fn is not None and callable(fn):
            log.info("Running %s ...", fn_name)
            try:
                fn(results)
            except Exception as exc:
                log.error("%s failed: %s", fn_name, exc, exc_info=True)

    return None


if __name__ == "__main__":
    main()
