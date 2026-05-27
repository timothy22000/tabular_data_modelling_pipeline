# Datasets

This directory holds tabular datasets used by the pipeline. Only **Medical
Insurance Costs** ships in the repo (CC0, ~50 KB) so that
`python train.py --quick` works out of the box on a fresh clone.

The other supported datasets are fetched on demand by
[`scripts/download_data.py`](../scripts/download_data.py):

| File | Dataset | Family | Source | Size | License |
|---|---|---|---|---|---|
| `medical_insurance.csv` | Medical Insurance Costs | gamma | bundled (originally Kaggle) | 50 KB | CC0 |
| `house_prices.csv` | House Prices: Advanced Regression Techniques | gamma | Kaggle / HF mirror | ~500 KB | Kaggle comp - free w/ attribution |
| `bike_sharing.csv` | Bike Sharing Demand | poisson | UCI ML Repository | ~700 KB | CC BY 4.0 |
| `allstate.csv` | Allstate Claims Severity | gamma | Kaggle API (auth required) | ~30 MB | Kaggle comp - non-commercial |

## Fetch a dataset

```bash
# Public datasets - no auth
python scripts/download_data.py --dataset house_prices
python scripts/download_data.py --dataset bike_sharing

# All public datasets (skips Allstate)
python scripts/download_data.py --all

# Allstate requires Kaggle credentials (see README -> Datasets section)
python scripts/download_data.py --dataset allstate --kaggle
```

Datasets are cached here after first download; subsequent calls are no-ops.
