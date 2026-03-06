"""Dataset configuration for UK motor net premium (insurance example).

This preserves all the original domain-specific constants from the
build_net_premium_glm.py and build_net_premium_gbm.py scripts.

Usage:
    python train.py --config configs/net_premium.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_config import DatasetConfig


EXCLUDE_COLS = [
    "AD_POLPREMIUM",
    "LOG_AD_POLPREMIUM",
    "AGG_SOURCE",
    "ANNUAL_TOPS_PRICE",
    "HSALE",
    "source_file",
    "SPLIT",
    "CREDIT_SCORE_MISSING",
]

RAW_CONTINUOUS = [
    "AGE", "CREDIT_SCORE", "VEHICLE_VALUE", "ENGINE_SIZE",
    "MILEAGE_K", "NCD_CAPPED", "VEHICLE_AGE", "LICENCEHELD_YEARS",
    "UKRESIDENCY_YEARS", "CLM_NUM_L5Y", "DTI", "NUMBER_OF_DRIVERS",
    "OVERNIGHT_LOCATION", "LICENCE_TYPE", "MONTHOFINCEPTION",
    "GROSSVEHICLEWEIGHT_K",
]

NATIVE_CATEGORICALS = [
    "RISK_AREA", "CONVICTIONS_FLAG", "COVER_TYPE", "DD_DUQ",
    "CLASSOFUSEDESC", "NCDPROTECT", "CLM_GROUP", "FUEL_TYPE_CAT",
]

GLM_HYBRID_FACTORS = [
    "AGE_BAND", "NCD_CAPPED", "CREDIT_SCORE_BAND", "VEHICLE_AGE_BAND",
    "RISK_AREA", "MILEAGE_K_BAND", "VEHICLE_VALUE_BAND",
    "ENGINE_SIZE_BAND", "CLASSOFUSEDESC", "DD_DUQ", "CLM_GROUP",
    "CONVICTIONS_FLAG", "COVER_TYPE",
]

BASE_LEVELS = {
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

MONOTONE_CONSTRAINTS = {
    "NCD_CAPPED": -1,
    "MILEAGE_K": 1,
    "VEHICLE_VALUE": 1,
    "CLM_NUM_L5Y": 1,
    "CREDIT_SCORE": -1,
}

DERIVED_FEATURES = {
    "AGE_SQUARED": lambda df: df["AGE"] ** 2,
    "AGE_X_NCD": lambda df: df["AGE"] * df["NCD_CAPPED"],
    "LOG_VEHICLE_VALUE": lambda df: df["VEHICLE_VALUE"].clip(lower=1).apply(
        lambda x: __import__("numpy").log(x)
    ),
    "EXPERIENCE_RATIO": lambda df: df["LICENCEHELD_YEARS"] / (df["AGE"] - 17).clip(lower=1),
    "NCD_RATE": lambda df: df["NCD_CAPPED"] / (df["AGE"] - 17).clip(lower=1),
}


config = DatasetConfig(
    target_col="AD_POLPREMIUM",
    split_col="SPLIT",
    exclude_cols=EXCLUDE_COLS,
    continuous_features=RAW_CONTINUOUS,
    categorical_features=NATIVE_CATEGORICALS,
    derived_features=DERIVED_FEATURES,
    glm_factors=GLM_HYBRID_FACTORS,
    base_levels=BASE_LEVELS,
    monotone_constraints=MONOTONE_CONSTRAINTS,
    family="gamma",
    link="log",
    prediction_floor=1.0,
    cap_percentile=99.5,
)
