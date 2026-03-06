"""Generic GLM utilities (fitting, design matrices, alignment)."""
from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# statsmodels
try:
    import statsmodels.api as sm
    from statsmodels.genmod.families import Gamma
    from statsmodels.genmod.families.links import Log as LogLink

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


class StatsmodelsResult:
    """Wrapper around a statsmodels GLMResults object."""

    def __init__(self, result: Any, feature_names: List[str]) -> None:
        self._result = result
        self.feature_names = feature_names

    @property
    def params(self) -> pd.Series:
        return self._result.params

    @property
    def bse(self) -> pd.Series:
        return self._result.bse

    @property
    def pvalues(self) -> pd.Series:
        return self._result.pvalues

    @property
    def deviance(self) -> float:
        return float(self._result.deviance)

    @property
    def pearson_chi2(self) -> float:
        return float(self._result.pearson_chi2)

    @property
    def aic(self) -> float:
        return float(self._result.aic)

    @property
    def bic(self) -> float:
        return float(self._result.bic_llf)

    @property
    def df_resid(self) -> float:
        return float(self._result.df_resid)

    @property
    def n_params(self) -> int:
        return int(len(self._result.params))

    @property
    def scale(self) -> float:
        return float(self._result.scale)

    @property
    def resid_deviance(self) -> pd.Series:
        return self._result.resid_deviance

    @property
    def resid_pearson(self) -> pd.Series:
        return self._result.resid_pearson

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_const = sm.add_constant(X, has_constant="add")
        return np.asarray(self._result.predict(X_const))

    def summary(self) -> Any:
        return self._result.summary()

    def get_influence(self) -> Any:
        return self._result.get_influence()


class SklearnResult:
    """Wrapper around sklearn GammaRegressor (fallback)."""

    def __init__(self, model: Any, feature_names: List[str]) -> None:
        self._model = model
        self.feature_names = feature_names
        coef_values = np.concatenate([[model.intercept_], model.coef_])
        coef_names = ["const"] + list(feature_names)
        self._params = pd.Series(coef_values, index=coef_names)

    @property
    def params(self) -> pd.Series:
        return self._params

    @property
    def bse(self) -> pd.Series:
        return pd.Series(np.nan, index=self._params.index)

    @property
    def pvalues(self) -> pd.Series:
        return pd.Series(np.nan, index=self._params.index)

    @property
    def deviance(self) -> float:
        return float("nan")

    @property
    def pearson_chi2(self) -> float:
        return float("nan")

    @property
    def aic(self) -> float:
        return float("nan")

    @property
    def bic(self) -> float:
        return float("nan")

    @property
    def df_resid(self) -> float:
        return float("nan")

    @property
    def n_params(self) -> int:
        return len(self._params)

    @property
    def scale(self) -> float:
        return float("nan")

    @property
    def resid_deviance(self) -> pd.Series:
        return pd.Series(dtype=float)

    @property
    def resid_pearson(self) -> pd.Series:
        return pd.Series(dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._model.predict(X.values.astype(float)))

    def summary(self) -> str:
        return f"SklearnResult (GammaRegressor fallback)\n  n_params: {self.n_params}"

    def get_influence(self) -> None:
        return None


def fit_gamma_glm(
    X: pd.DataFrame,
    y: pd.Series,
    weights: Optional[np.ndarray] = None,
) -> "StatsmodelsResult | SklearnResult":
    """Fit a Gamma GLM with log link.

    Uses statsmodels as primary backend, sklearn as fallback.
    """
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
        feature_names = [c for c in X_fit.columns if c != "_null_"]
        return StatsmodelsResult(result, feature_names)
    else:
        from sklearn.linear_model import GammaRegressor

        if X.shape[1] == 0:
            X_sk = pd.DataFrame({"_null_": np.zeros(len(y))}, index=X.index)
        else:
            X_sk = X.copy()
        model = GammaRegressor(alpha=0.001, max_iter=1000, tol=1e-8)
        model.fit(X_sk.values.astype(float), y.values.astype(float))
        feature_names = [c for c in X_sk.columns if c != "_null_"]
        return SklearnResult(model, feature_names)


def prepare_design_matrix(
    df: pd.DataFrame,
    factors: List[str],
    base_levels: Dict[str, str],
    categorical_factors: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Build design matrix with one-hot encoding for categoricals.

    Args:
        df: Source DataFrame.
        factors: Ordered list of factor names.
        base_levels: Factor -> base level (dropped in one-hot encoding).
        categorical_factors: Set of factor names that are categorical.
            If None, infers from dtype (object/category = categorical).
    """
    parts: List[pd.DataFrame] = []

    for factor in factors:
        if factor not in df.columns:
            log.warning("Factor %s not found in DataFrame — skipped.", factor)
            continue

        is_cat = False
        if categorical_factors is not None:
            is_cat = factor in categorical_factors
        else:
            is_cat = df[factor].dtype == "object" or pd.api.types.is_categorical_dtype(df[factor])

        if is_cat:
            dummies = pd.get_dummies(df[factor], prefix=factor, dtype=float)
            base_col = f"{factor}_{base_levels.get(factor, '')}"
            if base_col in dummies.columns:
                dummies = dummies.drop(columns=[base_col])
            parts.append(dummies)
        else:
            series = df[[factor]].astype(float)
            parts.append(series)

    if not parts:
        return pd.DataFrame(index=df.index)

    return pd.concat(parts, axis=1)


def align_test_matrix(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> pd.DataFrame:
    """Ensure test matrix has same columns as train.

    Missing columns get 0.0, extra columns are dropped, order is matched.
    """
    X_test = X_test.copy()
    missing = set(X_train.columns) - set(X_test.columns)
    for col in missing:
        X_test[col] = 0.0
    return X_test[X_train.columns]
