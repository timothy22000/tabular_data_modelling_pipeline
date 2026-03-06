"""Dataset configuration for the tabular modelling pipeline.

Users define their dataset schema here (or in a config file under configs/)
to specify target columns, feature lists, GLM factors, constraints, etc.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class DatasetConfig:
    """Configuration describing a tabular dataset for training.

    Attributes:
        target_col: Name of the response variable column.
        weight_col: Optional exposure/weight column name.
        split_col: Column used for train/test split. If None, a random
            80/20 split is used.
        exclude_cols: Columns to drop before modelling.
        continuous_features: List of continuous (numeric) feature column names.
        categorical_features: List of categorical feature column names.
        derived_features: Mapping of derived feature name to a callable
            that takes a DataFrame and returns a Series.  Example:
            {"AGE_SQUARED": lambda df: df["AGE"] ** 2}
        glm_factors: Columns used in the GLM base model design matrix.
            If empty, GLM base predictions default to 1.0.
        base_levels: Reference levels for categorical factors in the GLM
            design matrix (dropped during one-hot encoding).
        monotone_constraints: Mapping of feature name to direction (+1 or -1)
            for monotonicity regularisation during DL training.
        categorical_consolidation: Optional callable that takes a DataFrame
            and returns a cleaned DataFrame (e.g. mapping OCR variants).
            If None, no consolidation is applied.
        family: Distribution family for the GLM/loss function.
            One of "gamma", "gaussian", "tweedie", "poisson".
        link: Link function for the GLM. One of "log", "identity".
        prediction_floor: Minimum prediction value (prevents log(0) in
            Gamma deviance). Default 1.0.
        cap_percentile: Percentile for target capping (winsorisation).
        cap_value: Hard cap override. If set, overrides cap_percentile.
    """

    target_col: str = "target"
    weight_col: Optional[str] = None
    split_col: Optional[str] = None
    exclude_cols: List[str] = field(default_factory=list)
    continuous_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    derived_features: Dict[str, Callable] = field(default_factory=dict)
    glm_factors: List[str] = field(default_factory=list)
    base_levels: Dict[str, str] = field(default_factory=dict)
    monotone_constraints: Dict[str, int] = field(default_factory=dict)
    categorical_consolidation: Optional[Callable] = None
    family: str = "gamma"
    link: str = "log"
    prediction_floor: float = 1.0
    cap_percentile: float = 99.5
    cap_value: Optional[float] = None

    def validate(self) -> None:
        """Check that required fields are set and values are valid."""
        if not self.target_col:
            raise ValueError("target_col must be specified")
        if self.family not in ("gamma", "gaussian", "tweedie", "poisson"):
            raise ValueError(f"Unsupported family: {self.family}")
        if self.link not in ("log", "identity"):
            raise ValueError(f"Unsupported link: {self.link}")
        if self.prediction_floor <= 0:
            raise ValueError("prediction_floor must be positive")
        for feat, direction in self.monotone_constraints.items():
            if direction not in (-1, 1):
                raise ValueError(
                    f"Monotone constraint for '{feat}' must be +1 or -1, got {direction}"
                )


def auto_detect_features(df, target_col: str, exclude_cols: Optional[List[str]] = None) -> DatasetConfig:
    """Auto-detect continuous and categorical features from a DataFrame.

    All numeric columns (except target and exclude) become continuous.
    All object/category columns become categorical.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.
        exclude_cols: Columns to exclude from features.

    Returns:
        DatasetConfig with auto-detected feature lists.
    """
    import pandas as pd

    exclude = set(exclude_cols or [])
    exclude.add(target_col)

    continuous = []
    categorical = []

    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continuous.append(col)
        else:
            categorical.append(col)

    return DatasetConfig(
        target_col=target_col,
        exclude_cols=list(exclude - {target_col}),
        continuous_features=continuous,
        categorical_features=categorical,
    )
