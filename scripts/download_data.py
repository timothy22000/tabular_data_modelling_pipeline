#!/usr/bin/env python3
"""Download supported public datasets into the ``data/`` directory.

Each dataset has a canonical source:

  * medical_insurance - bundled in the repo (GitHub raw URL fallback)
  * house_prices      - Hugging Face public mirror
  * bike_sharing      - UCI ML Repository (direct zip)
  * allstate          - Kaggle API (requires ~/.kaggle/kaggle.json)

The script caches files to ``data/<name>.csv`` and skips datasets already
present. SHA256 checksums are pinned for the public datasets so users can
verify integrity. Allstate is not pinned because Kaggle re-packages its
competition archives over time.

Usage examples:

    # List supported datasets
    python scripts/download_data.py --list

    # Single dataset
    python scripts/download_data.py --dataset bike_sharing

    # All public datasets (skips Allstate)
    python scripts/download_data.py --all

    # Include Allstate (needs Kaggle credentials)
    python scripts/download_data.py --all --kaggle

    # Re-download even if cached
    python scripts/download_data.py --dataset house_prices --force
"""
from __future__ import annotations

import argparse
import hashlib
import io
import logging
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------


@dataclass
class DatasetSpec:
    """Describes how to fetch one dataset."""

    name: str
    description: str
    license: str
    target_col: str
    size_mb: float
    fetcher: Callable[[Path], None]
    sha256: Optional[str] = None
    auth_required: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# Fetcher implementations
# ---------------------------------------------------------------------------


def _fetch_medical_insurance(dest: Path) -> None:
    """Pull Medical Insurance from a public GitHub mirror."""
    url = (
        "https://raw.githubusercontent.com/stedy/"
        "Machine-Learning-with-R-datasets/master/insurance.csv"
    )
    _http_to_file(url, dest)


def _fetch_house_prices(dest: Path) -> None:
    """Pull House Prices from OpenML (id 42165) - 1460 rows × 81 cols.

    This is the same data as the Kaggle competition training set, hosted
    on OpenML as a stable canonical mirror. The 'test.csv' from Kaggle
    is unlabelled so we don't use it.
    """
    try:
        import openml
    except ImportError:
        raise RuntimeError(
            "openml package not installed. Run:\n"
            "    pip install openml\n"
            "(included when you install with 'pip install -e \".[download]\"')"
        )

    log.info("Fetching House Prices from OpenML (id 42165)...")
    ds = openml.datasets.get_dataset(42165, download_data=True, download_qualities=False, download_features_meta_data=False)
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
    df = X.copy()
    df[ds.default_target_attribute] = y
    df.to_csv(dest, index=False)
    log.info("Wrote %s (%d rows × %d cols)", dest, len(df), len(df.columns))


def _fetch_bike_sharing(dest: Path) -> None:
    """Pull Bike Sharing from UCI; the archive ships day.csv and hour.csv.

    We use ``hour.csv`` (17,379 rows) since hourly granularity exercises
    more of the pipeline (seasonality, hour-of-day patterns).
    """
    url = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
    log.info("Fetching Bike Sharing zip from UCI...")
    blob = _http_to_bytes(url)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        with zf.open("hour.csv") as src, dest.open("wb") as out:
            shutil.copyfileobj(src, out)
    log.info("Extracted hour.csv -> %s", dest)


def _fetch_allstate(dest: Path) -> None:
    """Pull Allstate via the Kaggle API. Requires ~/.kaggle/kaggle.json."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise RuntimeError(
            "kaggle package not installed. Run:\n"
            "    pip install kaggle\n"
            "Then place credentials at ~/.kaggle/kaggle.json - see\n"
            "    https://github.com/Kaggle/kaggle-api#api-credentials"
        )
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        raise RuntimeError(
            "Kaggle authentication failed. Place your token at "
            "~/.kaggle/kaggle.json (chmod 600). "
            f"Underlying error: {exc}"
        )

    log.info("Downloading Allstate Claims Severity from Kaggle (this can take a while)...")
    tmpdir = dest.parent / ".allstate_tmp"
    tmpdir.mkdir(exist_ok=True)
    api.competition_download_files("allstate-claims-severity", path=str(tmpdir))

    # The download is a single zip containing train.csv (+ test.csv)
    zip_path = next(tmpdir.glob("*.zip"))
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("train.csv") as src, dest.open("wb") as out:
            shutil.copyfileobj(src, out)

    shutil.rmtree(tmpdir)
    log.info("Extracted train.csv -> %s", dest)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


REGISTRY: dict[str, DatasetSpec] = {
    "medical_insurance": DatasetSpec(
        name="medical_insurance",
        description="Medical Insurance Costs (1,338 rows, gamma)",
        license="CC0",
        target_col="charges",
        size_mb=0.05,
        fetcher=_fetch_medical_insurance,
        sha256="505c1cbc2e63d0363bac59501563df2530aadf4cdb9cfee226f4ef32f5468281",
    ),
    "house_prices": DatasetSpec(
        name="house_prices",
        description="House Prices: Advanced Regression Techniques (~1.5k rows, gamma)",
        license="Kaggle competition - free use with attribution",
        target_col="SalePrice",
        size_mb=0.5,
        fetcher=_fetch_house_prices,
        sha256=None,  # mirrors vary
    ),
    "bike_sharing": DatasetSpec(
        name="bike_sharing",
        description="Bike Sharing Demand - hourly (17,379 rows, poisson)",
        license="CC BY 4.0 - Fanaee-T & Gama (2014), UCI ML Repository",
        target_col="cnt",
        size_mb=0.7,
        fetcher=_fetch_bike_sharing,
        sha256=None,  # hourly file inside the zip may shift bytes between mirrors
    ),
    "allstate": DatasetSpec(
        name="allstate",
        description="Allstate Claims Severity (~188k rows × 130 features, gamma)",
        license="Kaggle competition - non-commercial use only",
        target_col="loss",
        size_mb=30.0,
        fetcher=_fetch_allstate,
        auth_required=True,
        notes="Requires Kaggle API auth (~/.kaggle/kaggle.json). "
              "Run with --kaggle to acknowledge.",
    ),
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _http_to_bytes(url: str) -> bytes:
    """Fetch a URL to memory. Uses requests if available, falls back to stdlib."""
    try:
        import requests

        r = requests.get(url, timeout=60, stream=True)
        r.raise_for_status()
        return r.content
    except ImportError:
        import urllib.request

        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read()


def _http_to_file(url: str, dest: Path) -> None:
    """Fetch a URL straight to disk."""
    log.info("GET %s", url)
    blob = _http_to_bytes(url)
    dest.write_bytes(blob)
    log.info("Wrote %s (%.1f KB)", dest, len(blob) / 1024)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def fetch(name: str, force: bool = False) -> Path:
    """Fetch one dataset by registry name. Returns the local CSV path."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(REGISTRY)}")
    spec = REGISTRY[name]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / f"{name}.csv"

    if dest.exists() and not force:
        log.info("✓ %s already present at %s (--force to re-download)", name, dest)
        return dest

    log.info(
        "Downloading %s (%s, %.2f MB, license: %s)",
        name, spec.description, spec.size_mb, spec.license,
    )
    spec.fetcher(dest)

    if spec.sha256:
        actual = _sha256(dest)
        if actual != spec.sha256:
            dest.unlink()
            raise RuntimeError(
                f"SHA256 mismatch for {name}:\n"
                f"  expected {spec.sha256}\n"
                f"  actual   {actual}\n"
                f"The download was deleted. The upstream mirror may have changed - "
                f"please open an issue."
            )
        log.info("✓ SHA256 verified")
    else:
        log.info("(no SHA256 pinned - upstream mirrors vary)")

    return dest


def _print_listing() -> None:
    """Print the dataset registry as a table."""
    print(f"\n{'name':<22} {'license':<46} {'size':>8} {'target':<14} auth?")
    print("-" * 110)
    for spec in REGISTRY.values():
        size = f"{spec.size_mb:.2f} MB"
        auth = "yes" if spec.auth_required else "no"
        print(f"{spec.name:<22} {spec.license[:45]:<46} {size:>8} {spec.target_col:<14} {auth}")
    print()
    print("Fetch one:    python scripts/download_data.py --dataset <name>")
    print("Fetch all:    python scripts/download_data.py --all")
    print("With Kaggle:  python scripts/download_data.py --all --kaggle")
    print()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="download_data",
        description="Download tabular datasets used by the modelling pipeline.",
    )
    p.add_argument(
        "--dataset", "-d", choices=sorted(REGISTRY),
        help="Fetch a single dataset by name",
    )
    p.add_argument(
        "--all", "-a", action="store_true",
        help="Fetch all datasets that don't require auth",
    )
    p.add_argument(
        "--kaggle", action="store_true",
        help="Include datasets that require Kaggle API auth (Allstate)",
    )
    p.add_argument(
        "--force", "-f", action="store_true",
        help="Re-download even if the file is already cached",
    )
    p.add_argument(
        "--list", "-l", action="store_true",
        help="List supported datasets and exit",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.list or (not args.dataset and not args.all):
        _print_listing()
        return 0

    if args.dataset:
        spec = REGISTRY[args.dataset]
        if spec.auth_required and not args.kaggle:
            log.error(
                "%s requires Kaggle auth. Re-run with --kaggle once you have "
                "~/.kaggle/kaggle.json in place.", args.dataset,
            )
            return 1
        fetch(args.dataset, force=args.force)
        return 0

    # --all
    failures = []
    for name, spec in REGISTRY.items():
        if spec.auth_required and not args.kaggle:
            log.info("⏭ skipping %s (requires --kaggle)", name)
            continue
        try:
            fetch(name, force=args.force)
        except Exception as exc:
            log.error("✗ %s failed: %s", name, exc)
            failures.append(name)

    if failures:
        log.error("Failed: %s", failures)
        return 2
    log.info("✓ all done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
