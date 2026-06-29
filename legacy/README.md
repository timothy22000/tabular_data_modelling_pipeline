# Legacy / pre-refactor scripts

These four monolithic scripts predate the modular `modelling/` package
and the generic `DatasetConfig` abstraction. They were the original
benchmark code, written for one specific proprietary dataset (Ageas
Direct UK Motor Insurance, target column `AD_POLPREMIUM`). They are
**not used by the active pipeline** and exist here purely for
historical reference.

## Provenance

This work began as a one-off actuarial pricing exercise for a UK motor
portfolio. The pipeline was generalised in late 2026 - column names
moved into `dataset_config.py`, training was decomposed into the
`modelling/` package, the CLI moved into `train.py`, and tests followed.
At that point the original monolithic scripts were no longer needed but
were retained here so a reader can compare "before vs after" the
refactor.

| File | What it did | Notes |
|---|---|---:|
| `build_glm.py` | Gamma GLM with stepwise factor selection, interaction testing, actuarial diagnostics | superseded by `modelling/utils/glm.py` |
| `build_gbm.py` | LightGBM + GLM-GBM hybrid + parsimonious GBM, with Optuna and SHAP | superseded by `modelling/models/catboost_model.py` and `xgboost_model.py` (CatBoost replaces LightGBM) |
| `create_diagrams.py` | ReportLab-generated architecture PNGs for CANN, FT-Transformer, TabM | unused; output path is broken |
| `create_presentation.py` | python-pptx slide deck generator for the DL vs GBM comparison | tied to the proprietary dataset's specific results paths |

## Can I run these?

In short: no, not without modification.

- `build_glm.py` and `build_gbm.py` reference `AD_POLPREMIUM` as the
  target column and assume specific factor columns that the public
  datasets in this repo don't have.
- `create_diagrams.py` writes to a hard-coded path
  (`data_to_be_cleaned/net/dl_results/presentation/generated_figures/`)
  that doesn't exist here. The diagram code itself is portable - if you
  want architecture figures, retarget the output dir.
- `create_presentation.py` reads JSON / PNG artefacts from a result
  layout the modern pipeline doesn't produce.

If you want to revive any of them for the public datasets (House Prices,
Bike Sharing, Allstate), the GLM and GBM scripts would need their column
references re-pointed via `DatasetConfig`, and the diagram script just
needs its output dir fixed.

## Why archive instead of delete?

1. **Git history preserves the files** regardless, but `legacy/` makes
   them findable without `git log -- <removed-file>` archaeology.
2. **Comparison value.** Reviewers can see exactly what the modular
   refactor replaced.
3. **The diagram script** in particular is a small (488-line),
   self-contained ReportLab generator that's worth keeping discoverable
   should anyone want to make publication figures.

## Files NOT covered here

Everything else at the repo root - `train.py`, `dataset_config.py`,
`pyproject.toml`, etc. - is part of the active pipeline.

The `modelling/` package, `configs/`, `scripts/`, `tests/`, and `data/`
directories contain only active code.
