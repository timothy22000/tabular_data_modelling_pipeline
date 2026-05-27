"""Allstate Claims Severity config.

Kaggle competition:
    https://www.kaggle.com/c/allstate-claims-severity

NOTE: Kaggle competition rules permit non-commercial use. You must
accept the competition terms before downloading. Set up Kaggle API
auth (~/.kaggle/kaggle.json) then run:

    python scripts/download_data.py --dataset allstate --kaggle

Predicts ``loss`` (claim severity, USD) from 130 anonymised features
(116 categorical, 14 continuous, plus ``id``). Gamma + log link.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_config import DatasetConfig


# 116 categorical features named cat1..cat116
_CATEGORICAL = [f"cat{i}" for i in range(1, 117)]
# 14 continuous features named cont1..cont14
_CONTINUOUS = [f"cont{i}" for i in range(1, 15)]


config = DatasetConfig(
    target_col="loss",
    weight_col=None,
    split_col=None,
    exclude_cols=["id"],
    continuous_features=_CONTINUOUS,
    categorical_features=_CATEGORICAL,
    derived_features={},
    glm_factors=_CONTINUOUS[:4] + _CATEGORICAL[:4],
    base_levels={},  # All features are anonymised - use mode levels at runtime
    monotone_constraints={},  # No domain knowledge for anonymised features
    family="gamma",
    link="log",
    prediction_floor=1.0,
    cap_percentile=99.5,
)
