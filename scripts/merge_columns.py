"""Copy columns from a tagged subset back onto the full dataset, matched by an id.

E.g. put the Hydra sentiment fields (tagged on categorised mentions only, or on a
sample) back onto all 71k rows so there is one master file. Rows that weren't in
the subset get the --fill value. Free — no Claude calls.

Usage:
    python scripts/merge_columns.py data/hydra_relevance_final_final.csv data/sentiment_all_in_tagged.csv \
        --cols hydra_sentiment sentiment_basis confidence sentiment_reason -o data/hydra_master.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd


def merge_columns(base: pd.DataFrame, extra: pd.DataFrame, key: str, cols: List[str], fill: str) -> pd.DataFrame:
    missing = [c for c in [key, *cols] if c not in extra.columns]
    if missing:
        raise ValueError(f"tagged file has no column(s): {', '.join(missing)}")
    if key not in base.columns:
        raise ValueError(f"base file has no '{key}' column")
    extra = extra.drop_duplicates(subset=key)
    out = base.drop(columns=[c for c in cols if c in base.columns]).merge(
        extra[[key, *cols]], on=key, how="left", validate="many_to_one")
    out[cols] = out[cols].fillna(fill)
    return out


def main_cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base", type=Path, help="the full dataset")
    ap.add_argument("tagged", type=Path, help="the tagged subset")
    ap.add_argument("--cols", nargs="+", required=True)
    ap.add_argument("--key", default="mention_id")
    ap.add_argument("--fill", default="not rated", help="value for rows not in the tagged subset")
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()
    read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")  # noqa: E731
    base, extra = read(args.base), read(args.tagged)
    try:
        out = merge_columns(base, extra, args.key, args.cols, args.fill)
    except ValueError as e:
        sys.exit(str(e))
    if len(out) != len(base):
        sys.exit(f"Row count changed ({len(base)} -> {len(out)}); '{args.key}' is not unique in the base file.")
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    rated = (out[args.cols[0]] != args.fill).sum()
    print(f"Wrote {args.output}: {len(out):,} rows, {rated:,} with {args.cols[0]} "
          f"({len(out) - rated:,} marked '{args.fill}')")
    print(out[args.cols[0]].value_counts().to_string())


if __name__ == "__main__":
    main_cli()
