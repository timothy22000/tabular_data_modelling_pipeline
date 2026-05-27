# Tabular Data Modelling Pipeline

[![tests](https://github.com/timothy22000/tabular_data_modelling_pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/timothy22000/tabular_data_modelling_pipeline/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A research-grade pipeline that trains, tunes, ensembles, and explains **eight
tabular architectures** end to end - from gradient-boosted machines to recent
deep tabular models - on any regression dataset described by a single
`DatasetConfig`. Originally built for actuarial pricing (gamma + log link), it
generalises to any positive-target regression problem with the right
distribution family.

```
                        ┌──────────────────────────┐
                        │  Your CSV + DatasetConfig │
                        └────────────┬─────────────┘
                                     │
                  ┌──────────────────▼──────────────────┐
                  │ 1. Load + cap target + split          │
                  │ 2. Encode features (cont + cat)       │
                  │ 3. Fit GLM base model                 │
                  └──────────────────┬──────────────────┘
                                     │
                  ┌──────────────────▼──────────────────┐
                  │       Train all 8 architectures      │
                  │     Optuna tune  →  ensemble seeds   │
                  └──────────────────┬──────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
┌───────────────┐         ┌─────────────────┐          ┌────────────────────┐
│ NNLS stacking │         │ k-fold CV check │          │ Captum + SHAP +     │
│   ensemble    │         │                 │          │ partial-dependence  │
└───────┬───────┘         └────────┬────────┘          └──────────┬─────────┘
        │                          │                              │
        └──────────────────────────┴──────────────────────────────┘
                                     │
                  ┌──────────────────▼──────────────────┐
                  │ evaluation_summary.csv               │
                  │ ensemble_weights.json                │
                  │ feature_importance.csv               │
                  │ dashboard_dl_models.html             │
                  │ dashboard_dl_interpretability.html   │
                  │ figures/*.png                        │
                  └─────────────────────────────────────┘
```

## The eight architectures

| Architecture | Type | Key idea | Best for | Reference |
|---|---|---|---|---|
| **CatBoost** | Gradient boosting | Ordered boosting + categorical handling | Strong baseline on most tabular tasks | [Prokhorenkova et al. 2018](https://arxiv.org/abs/1706.09516) |
| **XGBoost** | Gradient boosting | Regularised boosting, second-order optimisation | The other ubiquitous baseline | [Chen & Guestrin 2016](https://arxiv.org/abs/1603.02754) |
| **CANN** | DL (GLM + NN) | Neural correction on top of a GLM base prediction | When the GLM is already pretty good and you want a residual lift | [Schelldorfer & Wüthrich 2019](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3320525) |
| **CANN-GBM** | Hybrid | GBM produces the base, NN refines it | Larger residual capacity than CANN; faster than full DL | This pipeline (extension of CANN) |
| **FT-Transformer** | DL (Transformer) | Feature-token transformer with attention over columns | Datasets where feature interactions matter and you have rows to burn | [Gorishniy et al. 2021](https://arxiv.org/abs/2106.11189) |
| **TabM** | DL (mixture) | Ensemble-of-MLPs with shared structure | Strong DL baseline, competitive with FT-T on smaller data | [Gorishniy et al. 2024](https://arxiv.org/abs/2410.24210) |
| **LocalGLMnet** | DL (interpretable) | Neural network outputting per-row GLM coefficients | When interpretability of feature effects matters | [Richman & Wüthrich 2023](https://www.cambridge.org/core/journals/astin-bulletin-journal-of-the-iaa/article/abs/localglmnet-interpretable-deep-learning-for-tabular-data/) |
| **DRN** | DL (distributional) | Distributional refinement on top of a base; produces full predictive distributions | Quantile/uncertainty estimation, not just point predictions | [Avanzi, Taylor, Wong 2023](https://arxiv.org/abs/2305.00065) |

All DL architectures are trained as **3-seed ensembles** by default; predictions
are averaged. A final **NNLS-stacked ensemble** is fit over the eight base
predictions for a single combined model.

## Quick start

```bash
# 1. Install (core + all extras)
git clone https://github.com/timothy22000/tabular_data_modelling_pipeline
cd tabular_data_modelling_pipeline
pip install -e ".[all]"

# 2. Train on the bundled Medical Insurance sample (1338 rows, ~60s in --quick mode)
python train.py \
    --config configs/example_medical_insurance.py \
    --input data/medical_insurance.csv \
    --quick --skip-tuning

# 3. Look at the outputs
open results/dl_results/dashboard_dl_models.html
open results/dl_results/dashboard_dl_interpretability.html
```

That's the full pipeline: GLM baseline → 8 architectures (with reduced budget
in `--quick` mode) → ensemble → metrics → figures → interpretability dashboard.

## Datasets

Medical Insurance Costs ships in the repo so the quick-start works offline.
Three more datasets are fetched on demand:

```bash
# List supported datasets
python scripts/download_data.py --list

# Fetch a single dataset
python scripts/download_data.py --dataset house_prices

# All public datasets (skips Allstate, which needs Kaggle auth)
python scripts/download_data.py --all
```

| Dataset | Rows × Cols | Family | License | Source |
|---|---|---|---|---|
| **Medical Insurance Costs** (bundled) | 1,338 × 7 | gamma | CC0 | [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance) (mirror) |
| **House Prices: Advanced Regression** | 1,460 × 81 | gamma | Kaggle comp - free w/ attribution | [OpenML id 42165](https://www.openml.org/d/42165) |
| **Bike Sharing Demand** (hourly) | 17,379 × 17 | poisson | CC BY 4.0 | [UCI ML Repository](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) |
| **Allstate Claims Severity** | 188,318 × 132 | gamma | Kaggle comp - non-commercial | [Kaggle competition](https://www.kaggle.com/c/allstate-claims-severity) |

**Allstate requires Kaggle credentials.** Set up `~/.kaggle/kaggle.json`
([guide](https://github.com/Kaggle/kaggle-api#api-credentials)), accept the
competition rules, then:
```bash
python scripts/download_data.py --dataset allstate --kaggle
```

## Configuring your own dataset

Drop a Python file under `configs/` that defines a module-level `config`
variable of type `DatasetConfig`:

```python
# configs/my_dataset.py
from dataset_config import DatasetConfig

config = DatasetConfig(
    target_col="my_target",
    continuous_features=["x1", "x2", "x3"],
    categorical_features=["cat_a", "cat_b"],
    derived_features={
        "x1_squared": lambda df: df["x1"] ** 2,
    },
    glm_factors=["x1", "cat_a"],          # Used in the GLM base
    base_levels={"cat_a": "reference"},   # Reference levels for one-hot
    monotone_constraints={"x1": +1},      # Prediction must be non-decreasing in x1
    family="gamma",                       # gamma | gaussian | tweedie | poisson
    link="log",
    cap_percentile=99.5,
)
```

Then point the pipeline at it:
```bash
python train.py --config configs/my_dataset.py --input data/my_data.csv
```

Don't want to enumerate the columns? `dataset_config.auto_detect_features(df, target_col)`
will infer continuous/categorical splits from dtypes.

## Outputs

A full run writes the following to `results/dl_results/`:

| File | Contents |
|---|---|
| `evaluation_summary.csv` | Gini, MAE, RMSE, CV-RMSE, A/E ratio, gamma deviance, training time per model |
| `cv_results.csv` / `cv_summary.json` | Per-fold metrics from k-fold CV |
| `ensemble_weights.json` | NNLS-stacked weights over the 8 base predictions |
| `feature_importance.csv` | Permutation importance + CatBoost native importance |
| `dl_metrics_summary.json` | Full structured run record (config, metrics, timing) |
| `drn_distributional_outputs.csv` | DRN's per-row distributional moments |
| `dashboard_dl_models.html` | Interactive comparison: Gini curves, Lorenz plots, calibration deciles, A/P scatter, ensemble weights |
| `dashboard_dl_interpretability.html` | Per-architecture interpretability: attention maps, attributions, partial dependence |
| `figures/fig_dl_*.png` | Standalone publication-quality figures |
| `<arch>.cbm`, `<arch>.json`, `<arch>_memberN.pt` | Trained model artifacts (CatBoost / XGBoost / DL ensemble members) |

Example metrics from a run on UK net-premium data (25k rows):

| Model | Gini (test) | MAE | A/E ratio | Training time |
|---|---|---|---|---|
| **Stacked ensemble** | **0.362** | 435 | 1.09 | - |
| XGBoost | 0.361 | 432 | 1.11 | 1s |
| CatBoost | 0.359 | 446 | 1.07 | 12s |
| CANN-GBM | 0.354 | 462 | 1.04 | 109s |
| LocalGLMnet | 0.329 | 517 | 1.05 | 177s |
| CANN | 0.323 | 535 | 1.03 | 107s |
| DRN | 0.317 | 558 | 0.99 | 196s |
| FT-Transformer | -0.01* | 675 | 1.08 | 7538s |
| TabM | -0.09* | 679 | 1.07 | 3047s |

\* FT-Transformer and TabM under-fit at the default 25k-row scale; they
typically need 100k+ rows and longer training to be competitive.

## Pre-trained models on Hugging Face

Pre-trained baselines are on the Hub:

| Dataset | Family | Dataset card | Model collection | Best test Gini |
|---|---|---|---|---|
| House Prices | gamma | [`house-prices-tabular`](https://huggingface.co/datasets/t22000t/house-prices-tabular) | [`house-prices-tabular-models`](https://huggingface.co/t22000t/house-prices-tabular-models) | 0.2061 (CatBoost) |
| Bike Sharing | poisson | [`bike-sharing-tabular`](https://huggingface.co/datasets/t22000t/bike-sharing-tabular) | [`bike-sharing-tabular-models`](https://huggingface.co/t22000t/bike-sharing-tabular-models) | 0.4975 (XGBoost) |

Quick inference:

```python
from huggingface_hub import hf_hub_download
from catboost import CatBoostRegressor
import pandas as pd

# Load model
path = hf_hub_download("t22000t/house-prices-tabular-models", "catboost.cbm")
model = CatBoostRegressor()
model.load_model(path)

# Load dataset
df = pd.read_csv("hf://datasets/t22000t/house-prices-tabular/train.csv")

# See the model card for the full feature list
preds = model.predict(df[features])
```

The pipeline dispatches XGBoost objectives and CatBoost loss functions
automatically based on `DatasetConfig.family`:

| Family | XGBoost objective | CatBoost loss |
|---|---|---|
| `gaussian` | `reg:squarederror` | `RMSE` |
| `gamma` | `reg:gamma` | `Tweedie:variance_power=2.0` |
| `tweedie` | `reg:tweedie` | `Tweedie:variance_power=1.5` |
| `poisson` | `count:poisson` | `Poisson` |

DL architectures (CANN, FT-Transformer, TabM, ...) will land in a v2 drop
for both dataset families.

## CLI reference

```
python train.py [--config CFG] [--input CSV] [--output-dir DIR]
                [--quick] [--skip-tuning] [--skip-interpretability]
                [--architectures {catboost,xgboost,cann,cann_gbm,ft_transformer,tabm,localglmnet,drn} ...]
                [--n-trials N] [--cv-folds N] [--seed N]
                [--epochs N] [--patience N] [--batch-size N]
                [--device {auto,cpu,cuda,mps}]
                [--val-fraction F] [--n-ensemble N]
                [--catboost-iterations N] [--mono-lambda F]
                [--cap V] [--cap-percentile P]
```

Common patterns:
```bash
# Fast iteration - 1k rows, no tuning, GBMs only
python train.py --config configs/example_medical_insurance.py \
                --input data/medical_insurance.csv \
                --quick --skip-tuning --architectures catboost xgboost

# Full run - all 8 architectures, 50 Optuna trials, 5-fold CV
python train.py --config configs/example_house_prices.py \
                --input data/house_prices.csv \
                --n-trials 50 --cv-folds 5

# DL only on GPU
python train.py --config configs/example_bike_sharing.py \
                --input data/bike_sharing.csv \
                --architectures cann cann_gbm ft_transformer tabm localglmnet drn \
                --device cuda

# Skip interpretability (slowest section) for fast comparison runs
python train.py --skip-interpretability ...
```

## Optional dependencies

Install only what you need:

```bash
pip install -e ".[gbm]"      # CatBoost + XGBoost
pip install -e ".[dl]"       # PyTorch + tabm
pip install -e ".[tuning]"   # Optuna
pip install -e ".[interp]"   # Captum + SHAP
pip install -e ".[viz]"      # matplotlib, seaborn, plotly
pip install -e ".[download]" # OpenML, Kaggle CLI for dataset fetching
pip install -e ".[all]"      # everything above
```

The pipeline degrades gracefully: if `torch` isn't installed, the DL
architectures are skipped with a warning; if `optuna` isn't installed,
default hyperparameters are used.

## Testing

```bash
pip install -e ".[dev]"
pytest                       # 91 tests across 8 files
pytest --cov=modelling       # with coverage
```

## Project structure

```
tabular_data_modelling_pipeline/
├── train.py                       # CLI entry point
├── dataset_config.py              # DatasetConfig dataclass + auto-detect
├── modelling/                     # Pipeline package
│   ├── pipeline.py                # Top-level orchestrator
│   ├── orchestration.py           # Per-architecture training loop
│   ├── config.py                  # DLConfig + argparse + HAS_X flags
│   ├── data.py                    # Load, cap, split, encode features
│   ├── training.py                # DL training loop with early stopping
│   ├── tuning.py                  # Optuna search per architecture
│   ├── ensemble.py                # NNLS stacking
│   ├── evaluation.py              # Metrics + per-model evaluation
│   ├── cv.py                      # k-fold cross-validation
│   ├── interpretability.py        # Captum + SHAP + partial dependence
│   ├── visualization.py           # Static figures
│   ├── output.py                  # Dashboards + summary JSON
│   ├── models/                    # 8 architecture implementations
│   │   ├── catboost_model.py
│   │   ├── xgboost_model.py
│   │   ├── cann.py
│   │   ├── cann_gbm.py
│   │   ├── ft_transformer.py
│   │   ├── tabm.py
│   │   ├── localglmnet.py
│   │   └── drn.py
│   └── utils/                     # Shared helpers
│       ├── glm.py                 # GLM base model fitting
│       ├── metrics.py             # Gini, gamma deviance, Lorenz
│       └── preprocessing.py       # Capping, splitting
├── configs/                       # Example DatasetConfig files
│   ├── example_medical_insurance.py
│   ├── example_house_prices.py
│   ├── example_bike_sharing.py
│   └── example_allstate.py
├── scripts/
│   └── download_data.py           # Multi-source dataset fetcher
├── tests/                         # 91 pytest cases
├── data/                          # Datasets (only Medical Insurance committed)
└── results/                       # Pipeline outputs (gitignored)
```

## Citation

If you use this pipeline in a paper, please cite:

```bibtex
@software{tabular_data_modelling_pipeline,
  author = {Mun, Timothy},
  title  = {tabular-data-modelling-pipeline: 8-architecture tabular
            modelling with Optuna tuning, ensembling, and interpretability},
  url    = {https://github.com/timothy22000/tabular_data_modelling_pipeline},
  year   = {2026}
}
```

Please also cite the individual architecture papers (CatBoost, XGBoost,
CANN, FT-Transformer, TabM, LocalGLMnet, DRN) - see the architecture table
above for references.

## License

MIT - see [LICENSE](LICENSE).

## Related projects

- 📂 [t22000t/house-prices-tabular](https://huggingface.co/datasets/t22000t/house-prices-tabular) - House Prices dataset on HF with baseline metrics
- 🤖 [t22000t/house-prices-tabular-models](https://huggingface.co/t22000t/house-prices-tabular-models) - pre-trained models for the above
- 📂 [t22000t/bike-sharing-tabular](https://huggingface.co/datasets/t22000t/bike-sharing-tabular) - Bike Sharing dataset on HF (Poisson family)
- 🤖 [t22000t/bike-sharing-tabular-models](https://huggingface.co/t22000t/bike-sharing-tabular-models) - pre-trained Poisson baselines for the above
- 🔒 [data-anonymization-toolkit](https://github.com/timothy22000/data-anonymization-toolkit) - config-driven anonymization, synthetic data generation, and red-team validation for the same kind of tabular data this pipeline trains on. Pairs with [Privacy Lab](https://huggingface.co/spaces/t22000t/privacy-lab) and [Synthetic Data Privacy Audit](https://huggingface.co/spaces/t22000t/synthetic-data-privacy-audit) Spaces.
