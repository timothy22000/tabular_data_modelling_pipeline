#!/usr/bin/env python3
"""Scan results/ and build dashboard/runs.json - the registry that powers
the static training dashboard.

For every subdirectory of `results/` that contains an `evaluation_summary.csv`,
this script extracts:

  * Per-model metrics (Gini, MAE, RMSE, A/E ratio, training time, n_params)
  * Run config (architectures, n_trials, seed, etc) from model_summary.json
  * Feature importance per model (if feature_importance.csv exists)
  * LocalGLMnet coefficient summary (if localglmnet_coefficients.csv exists)
  * The presence of interpretability artefacts (dashboard, distributional outputs)

The resulting JSON powers a static HTML dashboard at `dashboard/index.html`
that can be served with `python -m http.server` and viewed in a browser.

Usage:
    python scripts/build_dashboard.py
    python scripts/build_dashboard.py --results-dir my_results --output mydash.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Heuristics for inferring run metadata from directory name
# ---------------------------------------------------------------------------

DATASET_PATTERNS = {
    "house_prices": re.compile(r"^house[_-]?prices", re.I),
    "bike_sharing": re.compile(r"^bike[_-]?sharing", re.I),
    "allstate": re.compile(r"^allstate", re.I),
    "net_premium": re.compile(r"^dl_results", re.I),  # legacy proprietary runs
}


def infer_dataset(dir_name: str) -> Optional[str]:
    for name, pat in DATASET_PATTERNS.items():
        if pat.match(dir_name):
            return name
    return None


def infer_label(dir_name: str) -> str:
    """Derive a short human-readable label from the directory name."""
    n = dir_name.lower()
    if n.endswith("_interp"):
        return "v3 (interpretability)"
    if "tuned" in n:
        return "v4 (Optuna tuned)"
    if n.endswith("_8arch"):
        return "v2 (8 architectures)"
    if "baseline" in n:
        return "v1 (GBM baseline)"
    if "8arch" in n:
        return "8 architectures"
    return dir_name


def infer_is_tuned(dir_name: str, model_summary: Dict[str, Any]) -> bool:
    """Decide whether a run used Optuna.

    The pipeline's model_summary.json always logs ``n_tuning_trials`` even when
    ``--skip-tuning`` was passed (default is 30), so we cannot trust that field
    alone. The dir-naming convention used throughout the project is: a
    directory ending in ``_tuned`` (or containing ``tuned`` between
    underscores) is the only one that actually ran Optuna.
    """
    name = dir_name.lower()
    return name.endswith("_tuned") or "_tuned_" in name


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_eval_summary(path: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    out = []
    for _, row in df.iterrows():
        rec = {"model": row["model"]}
        for col in df.columns:
            if col == "model":
                continue
            val = row[col]
            if pd.notna(val):
                rec[col] = float(val)
            else:
                rec[col] = None
        out.append(rec)
    return out


def parse_feature_importance(path: Path) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(path, index_col=0)
    out: Dict[str, Dict[str, float]] = {}
    for col in df.columns:
        # Sort each model's importances descending
        col_data = df[col].dropna().sort_values(ascending=False)
        out[col] = {feat: float(val) for feat, val in col_data.items()}
    return out


def parse_localglmnet_coefs(path: Path) -> Dict[str, Dict[str, float]]:
    """Summarise LocalGLMnet per-row coefficients into mean/std/sign-stability."""
    df = pd.read_csv(path)
    out: Dict[str, Dict[str, float]] = {}
    for col in df.columns:
        vals = df[col].dropna()
        mean = float(vals.mean())
        std = float(vals.std())
        # Sign stability = fraction of rows whose coefficient matches mean's sign
        if mean > 0:
            stability = float((vals > 0).mean())
        else:
            stability = float((vals < 0).mean())
        out[col] = {"mean": mean, "std": std, "sign_stability": stability}
    return out


def parse_model_summary(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------


def scan_results(results_dir: Path) -> List[Dict[str, Any]]:
    runs = []
    for sub in sorted(results_dir.iterdir()):
        if not sub.is_dir():
            continue
        eval_path = sub / "evaluation_summary.csv"
        if not eval_path.exists():
            continue

        dataset = infer_dataset(sub.name)
        if dataset is None:
            continue

        ms_path = sub / "model_summary.json"
        ms = parse_model_summary(ms_path) if ms_path.exists() else {}

        run = {
            "run_id": sub.name,
            "dataset": dataset,
            "label": infer_label(sub.name),
            "tuned": infer_is_tuned(sub.name, ms),
            "models": parse_eval_summary(eval_path),
            "config": ms.get("config", {}),
            "best_model": ms.get("best_model"),
            "glm_gini": ms.get("glm_gini"),
            "timestamp": ms.get("timestamp"),
            "has_interpretability": (sub / "dashboard_dl_interpretability.html").exists(),
            "has_dashboard": (sub / "dashboard_dl_models.html").exists(),
        }

        # Feature importance (if present)
        fi_path = sub / "feature_importance.csv"
        if fi_path.exists():
            run["feature_importance"] = parse_feature_importance(fi_path)

        # LocalGLMnet coefficient summary
        lcoef_path = sub / "localglmnet_coefficients.csv"
        if lcoef_path.exists():
            run["localglmnet_coef_summary"] = parse_localglmnet_coefs(lcoef_path)

        runs.append(run)
    return runs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--output", default="dashboard/runs.json")
    p.add_argument("--kaggle-metrics", default="results/kaggle_metrics.json")
    args = p.parse_args()

    results_dir = ROOT / args.results_dir
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist", file=sys.stderr)
        return 1

    runs = scan_results(results_dir)

    kaggle_ref: Dict[str, Any] = {}
    kpath = ROOT / args.kaggle_metrics
    if kpath.exists():
        try:
            kaggle_ref = json.loads(kpath.read_text()).get("kaggle_reference", {})
        except Exception:
            pass

    out = {
        "schema_version": 1,
        "generated_at": "build-time",  # filled by JS at view time, keeps the JSON deterministic
        "n_runs": len(runs),
        "kaggle_reference": kaggle_ref,
        "runs": runs,
    }

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))

    # Summary to stdout
    print(f"Scanned {results_dir}")
    print(f"  -> {len(runs)} runs found")
    by_ds: Dict[str, int] = {}
    for r in runs:
        by_ds[r["dataset"]] = by_ds.get(r["dataset"], 0) + 1
    for ds, n in sorted(by_ds.items()):
        print(f"  - {ds}: {n} run(s)")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
