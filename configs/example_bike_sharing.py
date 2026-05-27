"""Bike Sharing Demand config.

UCI ML Repository: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
Fanaee-T & Gama (2014). CC BY 4.0.

Fetch the data with:
    python scripts/download_data.py --dataset bike_sharing

Predicts ``cnt`` (hourly bike rental count) - a non-negative integer count.
Poisson family with log link is the natural choice.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_config import DatasetConfig


config = DatasetConfig(
    target_col="cnt",
    weight_col=None,
    split_col=None,
    # 'casual' and 'registered' are leakage (they sum to cnt).
    # 'dteday' is a date string - drop in favour of yr/mnth/hr columns.
    # 'instant' is a row id.
    exclude_cols=["instant", "dteday", "casual", "registered"],
    continuous_features=[
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "hr",
        "yr",
        "mnth",
    ],
    categorical_features=[
        "season",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
    ],
    derived_features={},
    glm_factors=["temp", "hr", "season", "weathersit"],
    base_levels={
        "season": "1",  # spring
        "weathersit": "1",  # clear
    },
    monotone_constraints={
        "temp": 1,  # Warmer -> more rentals (within reason)
    },
    family="poisson",
    link="log",
    prediction_floor=1.0,
    cap_percentile=99.9,  # Count data - cap less aggressively
)
