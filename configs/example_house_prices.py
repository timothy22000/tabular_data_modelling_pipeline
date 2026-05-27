"""House Prices: Advanced Regression Techniques config.

Kaggle competition:
    https://www.kaggle.com/c/house-prices-advanced-regression-techniques

Fetch the data with:
    python scripts/download_data.py --dataset house_prices

Predicts ``SalePrice`` (US dollars) from 79 features. Long right-tailed
target -> gamma + log link.

Note: only a representative subset of the 79 features is wired up below.
The pipeline will run if columns are missing; for the full schema see
the Kaggle data dictionary.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_config import DatasetConfig


config = DatasetConfig(
    target_col="SalePrice",
    weight_col=None,
    split_col=None,
    exclude_cols=["Id"],
    continuous_features=[
        "LotArea",
        "YearBuilt",
        "YearRemodAdd",
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF",
        "GrLivArea",
        "FullBath",
        "BedroomAbvGr",
        "TotRmsAbvGrd",
        "GarageCars",
        "GarageArea",
        "OverallQual",
        "OverallCond",
    ],
    categorical_features=[
        "MSZoning",
        "Street",
        "LotShape",
        "Neighborhood",
        "BldgType",
        "HouseStyle",
        "RoofStyle",
        "ExterQual",
        "Foundation",
        "Heating",
        "CentralAir",
        "KitchenQual",
        "SaleType",
        "SaleCondition",
    ],
    derived_features={
        "TotalSF": lambda df: df["TotalBsmtSF"].fillna(0) + df["1stFlrSF"] + df["2ndFlrSF"].fillna(0),
        "HouseAge": lambda df: df["YrSold"] - df["YearBuilt"] if "YrSold" in df.columns else df["YearBuilt"].max() - df["YearBuilt"],
    },
    glm_factors=["GrLivArea", "OverallQual", "Neighborhood", "YearBuilt"],
    base_levels={
        "MSZoning": "RL",
        "Neighborhood": "NAmes",
    },
    monotone_constraints={
        "GrLivArea": 1,
        "OverallQual": 1,
        "TotalBsmtSF": 1,
    },
    family="gamma",
    link="log",
    prediction_floor=1.0,
    cap_percentile=99.5,
)
