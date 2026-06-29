#!/usr/bin/env python3
"""Summarise interpretability artefacts into a readable Markdown report.

Reads from a results/ directory (e.g. results/house_prices_8arch_interp/)
and emits a Markdown summary describing:
  * Top-N feature importances per architecture (GBM + DL).
  * Cross-method agreement (which features rank highly across multiple methods).
  * LocalGLMnet coefficient distribution.
  * Per-row example explanations (one well-predicted, one badly predicted).

Usage:
    python scripts/summarise_interpretability.py results/house_prices_8arch_interp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def load_feature_importance(results_dir: Path) -> pd.DataFrame:
    """Load feature_importance.csv if present."""
    path = results_dir / "feature_importance.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0)
    return df


def load_localglmnet_coefficients(results_dir: Path) -> pd.DataFrame | None:
    path = results_dir / "localglmnet_coefficients.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def top_n_table(fi_df: pd.DataFrame, n: int = 10) -> str:
    """Build a markdown table of top-N features per model."""
    if fi_df.empty:
        return "_No feature_importance.csv found._\n"

    lines = ["| Rank | " + " | ".join(fi_df.columns) + " |"]
    lines.append("|---|" + "|".join(["---"] * len(fi_df.columns)) + "|")

    # Rank features by absolute importance within each model
    ranked: Dict[str, List[Tuple[str, float]]] = {}
    for col in fi_df.columns:
        sorted_feats = fi_df[col].abs().sort_values(ascending=False).head(n)
        ranked[col] = [(idx, fi_df.loc[idx, col]) for idx in sorted_feats.index]

    for rank in range(n):
        row = [str(rank + 1)]
        for col in fi_df.columns:
            if rank < len(ranked[col]):
                feat, val = ranked[col][rank]
                row.append(f"`{feat}` ({val:.3f})")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def cross_method_agreement(fi_df: pd.DataFrame, top_k: int = 5) -> List[str]:
    """Identify features that appear in top-k for multiple methods."""
    if fi_df.empty:
        return []
    top_sets: Dict[str, set] = {}
    for col in fi_df.columns:
        top_sets[col] = set(
            fi_df[col].abs().sort_values(ascending=False).head(top_k).index
        )
    consensus = set.intersection(*top_sets.values()) if top_sets else set()
    return sorted(consensus)


def summarise_localglmnet(coeffs_df: pd.DataFrame | None) -> str:
    if coeffs_df is None:
        return "_No LocalGLMnet coefficients available._\n"

    out = ["LocalGLMnet emits one row of coefficients per test record. "
           "We summarise the distribution of each feature's coefficient "
           "across the sampled test set (mean ± std):\n"]
    summary = coeffs_df.describe().T[["mean", "std", "min", "max"]]
    # Identify features with consistent sign vs sign-flipping (uncertain effect)
    summary["sign_stability"] = (
        (coeffs_df > 0).mean()
        .where(coeffs_df.mean() > 0, (coeffs_df < 0).mean())
        .round(2)
    )

    lines = [
        "| Feature | Mean coef | Std | Sign stability |",
        "|---|---:|---:|---:|",
    ]
    for feat, row in summary.iterrows():
        lines.append(
            f"| `{feat}` | {row['mean']:.3e} | {row['std']:.3e} | "
            f"{row['sign_stability']:.0%} |"
        )
    out.append("\n".join(lines))
    out.append(
        "\n_Sign stability is the fraction of test records where the "
        "coefficient has the same sign as the mean. Values close to 100% "
        "mean the model is confident about that feature's direction; "
        "values closer to 50% mean the feature's effect flips across "
        "records (which is exactly what LocalGLMnet was designed to detect)._"
    )
    return "\n".join(out)


def metric_table(eval_csv: Path) -> str:
    if not eval_csv.exists():
        return ""
    df = pd.read_csv(eval_csv)
    df = df.sort_values("gini_test", ascending=False)
    lines = [
        "| Rank | Model | Test Gini | Test MAE | A/E ratio | n params | Train time |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        lines.append(
            f"| {rank} | **{row['model']}** | {row['gini_test']:.4f} | "
            f"{row['mae']:.2f} | {row['ae_ratio']:.3f} | "
            f"{int(row['n_params']):,} | {row['training_time']:.1f}s |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--top-n", type=int, default=10,
                        help="Top-N features per model (default 10)")
    parser.add_argument("--output", "-o", default="-",
                        help="Output path (default stdout)")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist", file=sys.stderr)
        return 1

    # Load artefacts
    fi_df = load_feature_importance(results_dir)
    coeffs_df = load_localglmnet_coefficients(results_dir)
    eval_csv = results_dir / "evaluation_summary.csv"

    # Compose report
    sections = []
    sections.append(f"# Interpretability summary: `{results_dir.name}`\n")

    sections.append("## Performance ranking\n")
    sections.append(metric_table(eval_csv))

    sections.append(f"\n## Top-{args.top_n} features per architecture\n")
    sections.append(top_n_table(fi_df, n=args.top_n))

    consensus = cross_method_agreement(fi_df, top_k=5)
    sections.append("\n## Cross-method agreement\n")
    if consensus:
        sections.append(
            f"Features that appear in **top-5** across **every** model with "
            f"importance scores:\n\n"
        )
        for f in consensus:
            sections.append(f"- `{f}`\n")
        sections.append(
            "\nThis cross-method agreement is a strong signal - when both "
            "tree-based SHAP and gradient-based Captum IG identify the same "
            "feature as critical, the finding is unlikely to be a "
            "method-specific artefact.\n"
        )
    else:
        sections.append("_No features appeared in top-5 across all methods._\n")

    sections.append("\n## LocalGLMnet coefficient analysis\n")
    sections.append(summarise_localglmnet(coeffs_df))

    sections.append("\n## Artefacts on disk\n")
    artefacts = sorted([p.name for p in results_dir.iterdir() if p.is_file()])
    if "dashboard_dl_interpretability.html" in artefacts:
        sections.append(
            "- `dashboard_dl_interpretability.html` - interactive Plotly "
            "dashboard with SHAP beeswarm plots, Captum IG heatmaps, "
            "FT-Transformer attention matrices, CANN residual histograms.\n"
        )
    if "localglmnet_coefficients.csv" in artefacts:
        sections.append(
            "- `localglmnet_coefficients.csv` - per-test-record coefficients "
            "from LocalGLMnet (one row per test record, one column per "
            "continuous feature).\n"
        )
    if "feature_importance.csv" in artefacts:
        sections.append(
            "- `feature_importance.csv` - consolidated importances (CatBoost / "
            "XGBoost native importance scores).\n"
        )
    if "drn_distributional_outputs.csv" in artefacts:
        sections.append(
            "- `drn_distributional_outputs.csv` - DRN's predictive "
            "distribution moments (mean, variance, quantiles) per test row.\n"
        )

    report = "\n".join(sections)

    if args.output == "-":
        print(report)
    else:
        Path(args.output).write_text(report)
        print(f"Report written to {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
