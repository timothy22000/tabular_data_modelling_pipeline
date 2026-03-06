"""Generic model evaluation metrics (no domain-specific logic)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def clamp_predictions(preds: np.ndarray, floor: float = 1.0) -> np.ndarray:
    """Clamp predictions to a positive floor."""
    return np.maximum(preds, floor)


def compute_gini(y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
    """Compute Gini coefficient via the Lorenz curve.

    Higher Gini = greater discriminatory power.

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

    return float(1 - 2 * cumulative.sum() / (n * total))


def compute_gamma_deviance(
    y_actual: np.ndarray, y_predicted: np.ndarray, floor: float = 1.0
) -> float:
    """Compute Gamma deviance: D = 2 * sum[log(a/p) - (a-p)/p].

    Returns:
        Total Gamma deviance.
    """
    y_actual = np.asarray(y_actual, dtype=float)
    y_predicted = clamp_predictions(np.asarray(y_predicted, dtype=float), floor)

    valid = (y_actual > 0) & (y_predicted > 0)
    ya = y_actual[valid]
    yp = y_predicted[valid]

    if len(ya) == 0:
        return float("nan")

    unit_deviance = 2.0 * (np.log(ya / yp) - (ya - yp) / yp)
    return float(unit_deviance.sum())


def compute_metrics(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    label: str,
    n_params: int = 0,
    floor: float = 1.0,
) -> Dict[str, Any]:
    """Compute standard regression metrics (MAE, RMSE, Gini, A/E, etc.).

    Returns:
        Dictionary of metric name -> value.
    """
    y_actual = np.asarray(y_actual, dtype=float)
    y_predicted = clamp_predictions(np.asarray(y_predicted, dtype=float), floor)

    mae = float(np.mean(np.abs(y_actual - y_predicted)))
    rmse = float(np.sqrt(np.mean((y_actual - y_predicted) ** 2)))
    mean_actual = float(y_actual.mean())
    mean_predicted = float(y_predicted.mean())
    cv_rmse = rmse / mean_actual if mean_actual > 0 else float("nan")
    ae_ratio = (
        float(y_actual.sum() / y_predicted.sum())
        if y_predicted.sum() > 0
        else float("nan")
    )
    gini = compute_gini(y_actual, y_predicted)
    gamma_dev = compute_gamma_deviance(y_actual, y_predicted, floor)

    return {
        "split": label,
        "n": int(len(y_actual)),
        "n_params": n_params,
        "gini": round(gini, 6),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "cv_rmse": round(cv_rmse, 6),
        "ae_ratio": round(ae_ratio, 6),
        "mean_actual": round(mean_actual, 4),
        "mean_predicted": round(mean_predicted, 4),
        "gamma_deviance": round(gamma_dev, 6),
    }


def lorenz_curve(
    y_actual: np.ndarray, y_predicted: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Lorenz curve (cumulative actual vs cumulative proportion).

    Returns:
        Tuple of (x_axis, y_axis) both in [0, 1].
    """
    order = np.argsort(y_predicted)
    y_sorted = np.asarray(y_actual, dtype=float)[order]
    cum = np.cumsum(y_sorted)
    total = cum[-1] if cum[-1] > 0 else 1.0
    n = len(y_sorted)
    x_axis = np.linspace(0, 1, n)
    y_axis = cum / total
    return x_axis, y_axis


def compute_decile_analysis(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    floor: float = 1.0,
) -> pd.DataFrame:
    """Decile lift chart: split by predicted value, summarise per decile.

    Returns:
        DataFrame with one row per decile (1=lowest, 10=highest).
    """
    y_actual = np.asarray(y_actual, dtype=float)
    y_predicted = clamp_predictions(np.asarray(y_predicted, dtype=float), floor)

    n = len(y_actual)
    if n == 0:
        return pd.DataFrame()

    order = np.argsort(y_predicted)
    ya_sorted = y_actual[order]
    yp_sorted = y_predicted[order]

    decile_labels = np.repeat(np.arange(1, 11), n // 10)
    remainder = n - len(decile_labels)
    if remainder > 0:
        decile_labels = np.concatenate([decile_labels, np.full(remainder, 10)])

    rows: List[Dict[str, Any]] = []
    for d in range(1, 11):
        mask = decile_labels == d
        ya_d = ya_sorted[mask]
        yp_d = yp_sorted[mask]

        if len(ya_d) == 0:
            continue

        ae_ratio = (
            float(ya_d.sum() / yp_d.sum()) if yp_d.sum() > 0 else float("nan")
        )

        rows.append(
            {
                "decile": d,
                "n": int(len(ya_d)),
                "mean_actual": round(float(ya_d.mean()), 4),
                "mean_predicted": round(float(yp_d.mean()), 4),
                "ae_ratio": round(ae_ratio, 6),
                "sum_actual": round(float(ya_d.sum()), 2),
                "sum_predicted": round(float(yp_d.sum()), 2),
            }
        )

    return pd.DataFrame(rows)
