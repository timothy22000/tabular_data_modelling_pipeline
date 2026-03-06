"""Generic utilities for the tabular modelling pipeline.

Re-exports commonly used functions from submodules.
"""
from .glm import (
    StatsmodelsResult,
    SklearnResult,
    fit_gamma_glm,
    prepare_design_matrix,
    align_test_matrix,
)
from .metrics import (
    compute_gini,
    compute_gamma_deviance,
    compute_metrics,
    lorenz_curve,
    compute_decile_analysis,
)
from .preprocessing import (
    clamp_predictions,
    cap_target,
    load_csv_with_split,
)

__all__ = [
    "StatsmodelsResult",
    "SklearnResult",
    "fit_gamma_glm",
    "prepare_design_matrix",
    "align_test_matrix",
    "compute_gini",
    "compute_gamma_deviance",
    "compute_metrics",
    "lorenz_curve",
    "compute_decile_analysis",
    "clamp_predictions",
    "cap_target",
    "load_csv_with_split",
]
