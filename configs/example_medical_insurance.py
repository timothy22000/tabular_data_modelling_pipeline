"""Medical Insurance Costs config.

Dataset: https://www.kaggle.com/datasets/mirichoi0218/insurance (CC0)
Bundled at: data/medical_insurance.csv (1338 rows, 7 columns)

Predicts ``charges`` (annual medical insurance cost in USD) from demographic
and lifestyle features. Long right-tailed positive target makes gamma + log
link the natural choice.

Usage:
    python train.py \\
        --config configs/example_medical_insurance.py \\
        --input data/medical_insurance.csv \\
        --quick --skip-tuning
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_config import DatasetConfig


config = DatasetConfig(
    target_col="charges",
    weight_col=None,
    split_col=None,  # No pre-defined split - use random 80/20
    exclude_cols=[],
    continuous_features=[
        "age",
        "bmi",
        "children",
    ],
    categorical_features=[
        "sex",
        "smoker",
        "region",
    ],
    derived_features={},
    glm_factors=["age", "bmi", "smoker", "region"],
    base_levels={
        "sex": "female",
        "smoker": "no",
        "region": "northeast",
    },
    monotone_constraints={
        "age": 1,  # Older -> higher charges
        "bmi": 1,  # Higher BMI -> higher charges
    },
    family="gamma",
    link="log",
    prediction_floor=1.0,
    cap_percentile=99.5,
)
