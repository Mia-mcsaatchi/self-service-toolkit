"""Compare two tagged outputs of the same input file, e.g. full text vs trimmed.

Reports how often the two runs agree on each output column, split into rows
whose text was shortened and rows sent unchanged. The unchanged rows show how
much the model varies between identical runs anyway, which is the yardstick
for the shortened ones. Disagreements are written to a CSV for reading.

Usage:
    python scripts/compare_runs.py full.csv trimmed.csv --config configs/hydra_mentions.json --trim-chars 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import main  # noqa: E402
from run_tagging import expand_fields  # noqa: E402


def _norm(v: str) -> str:
    return " ".join(str(v).lower().split())


def main_cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--trim-chars", type=int, help="trim size the candidate used (default: the config's)")
    ap.add_argument("-o", "--output", type=Path, help="disagreements CSV (default: <candidate>_diff.csv)")
    args = ap.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    fields = [f["name"] for f in expand_fields(config)]
    trim = config.get("trim")
    if not trim:
        sys.exit("The config has no \"trim\" section, so there is nothing to compare against.")
    chars = args.trim_chars if args.trim_chars is not None else int(trim.get("chars", 500))

    a = pd.read_csv(args.baseline, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    b = pd.read_csv(args.candidate, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if len(a) != len(b):
        sys.exit(f"Row counts differ ({len(a)} vs {len(b)}) — compare outputs of the same input file.")

    trimmed = pd.Series(False, index=a.index)
    for col in trim["columns"]:
        if col in a.columns:
            trimmed |= a[col].map(lambda t: main._trim_text(t, trim["pattern"], chars) != t)

    print(f"{len(a):,} rows: {int(trimmed.sum()):,} shortened at ±{chars} characters, "
          f"{int((~trimmed).sum()):,} sent unchanged\n")
    print(f"{'column':<12} {'shortened rows':>16} {'unchanged rows':>16}")
    for f in fields:
        same = a[f].map(_norm) == b[f].map(_norm)
        def pct(mask):
            return f"{100 * same[mask].mean():.0f}% agree" if mask.any() else "—"
        print(f"{f:<12} {pct(trimmed):>16} {pct(~trimmed):>16}")
    print("\n(Free-text columns like summary rarely match word for word; read the diff file for those.)")

    status_like = [f for f in fields if a[f].nunique() <= 10]
    differ = pd.Series(False, index=a.index)
    for f in status_like or fields:
        differ |= a[f].map(_norm) != b[f].map(_norm)
    keep = [c for c in ("mention_id", "original_title") if c in a.columns]
    diff = a.loc[differ, keep].copy()
    diff["shortened"] = trimmed[differ]
    for f in fields:
        diff[f"{f} (baseline)"] = a.loc[differ, f]
        diff[f"{f} (candidate)"] = b.loc[differ, f]
    out = args.output or args.candidate.with_name(args.candidate.stem + "_diff.csv")
    diff.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n{int(differ.sum())} row(s) where {', '.join(status_like or fields)} differ -> {out}")


if __name__ == "__main__":
    main_cli()
