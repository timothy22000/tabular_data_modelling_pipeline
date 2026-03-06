#!/usr/bin/env python3
"""Deep Learning & Advanced Modelling Pipeline.

Trains and evaluates 8 model architectures:
  CatBoost, XGBoost, CANN, FT-Transformer, TabM,
  CANN-GBM, LocalGLMnet, DRN.

Usage:
    python train.py
    python train.py --quick --skip-tuning
    python train.py --architectures catboost xgboost
    python train.py --n-trials 50 --epochs 200 --device cpu
    python train.py --skip-tuning --skip-interpretability

Delegates to modelling package — see modelling/ for implementation.
"""
from modelling import main

if __name__ == "__main__":
    main()
