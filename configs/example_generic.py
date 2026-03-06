"""Example dataset configuration template.

Copy this file and fill in your dataset's columns.

Usage:
    python train.py --config configs/my_dataset.py --input data/my_data.csv
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_config import DatasetConfig


config = DatasetConfig(
    # Required: name of the column you want to predict
    target_col="price",

    # Optional: column for train/test split (values: "TRAIN"/"TEST")
    # If None, a random 80/20 split is used
    split_col=None,

    # Columns to exclude from features (e.g. IDs, leakage columns)
    exclude_cols=["id", "timestamp"],

    # Numeric features
    continuous_features=[
        "feature_1",
        "feature_2",
        "feature_3",
    ],

    # Categorical features (string/object columns)
    categorical_features=[
        "category_a",
        "category_b",
    ],

    # Derived features: name -> lambda(df) -> Series
    # Leave empty if you don't need derived features
    derived_features={},

    # GLM base model factors (for CANN/LocalGLMnet/DRN architectures)
    # If empty, GLM base predictions default to 1.0
    glm_factors=[],
    base_levels={},

    # Monotonicity constraints: feature -> direction (+1 or -1)
    # +1 = increasing feature should increase prediction
    # -1 = increasing feature should decrease prediction
    monotone_constraints={},

    # Distribution family for GLM loss
    # "gamma" for positive continuous targets (e.g. prices, costs)
    # "gaussian" for general continuous targets
    family="gamma",
    link="log",
    prediction_floor=1.0,
    cap_percentile=99.5,
)
