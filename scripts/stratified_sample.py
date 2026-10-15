"""Draw a stratified random sample: up to N mentions from each group.

Used to measure something per category (e.g. Hydra-specific sentiment) without
paying to tag every mention — ~200 per group gives a share within about ±7 points.

Usage:
    python scripts/stratified_sample.py data/hydra_relevance_final_final.csv -o data/sentiment_sample.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def stratified_sample(df: pd.DataFrame, by: str, per_group: int, seed: int = 1,
                      skip=("Excluded", "Unresolved")) -> pd.DataFrame:
    pool = df[~df[by].isin(skip)]
    # Shuffle once, then keep the first N of each group: a random N per group.
    return pool.sample(frac=1, random_state=seed).groupby(by).head(per_group).reset_index(drop=True)


def main_cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--by", default="broad_category")
    ap.add_argument("--per-group", type=int, default=200)
    args = ap.parse_args()
    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    out = stratified_sample(df, args.by, args.per_group)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"{len(out):,} rows from {out[args.by].nunique()} groups -> {args.output}")


if __name__ == "__main__":
    main_cli()
