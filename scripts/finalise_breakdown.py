"""Apply reviewed clean-up rules to a categorised file and write the final breakdown.

map_labels.py proposes categories label by label; a person then reviews the
result and writes rules (configs/<name>_merges.json) to merge duplicates, move
misplaced tags and catch patterns in what is still unresolved. This applies
those rules — no Claude calls, so it is free and can be re-run after every edit
to the rules — then folds new associations below the size threshold into
"Other" within their broad category.

Usage:
    python scripts/finalise_breakdown.py data/hydra_relevance_final_categorised.csv configs/hydra_merges.json
Writes <name>_final.csv and <name>_final_breakdown.xlsx next to the input.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_labels import write_breakdown  # noqa: E402


def apply_rules(cat: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    out = cat.copy()
    out["new_association"] = out["new_association"].astype(str).str.lower().eq("true")
    for rule in cfg.get("rules", []):
        src, dst = rule["from"], rule["to"]
        mask = pd.Series(True, index=out.index)
        if "broad" in src:
            mask &= out["broad_category"] == src["broad"]
        if "specific" in src:
            mask &= out["specific_association"] == src["specific"]
        if "label_regex" in src:
            with warnings.catch_warnings():  # groups in a pattern are fine here
                warnings.simplefilter("ignore", UserWarning)
                mask &= out["context"].str.contains(src["label_regex"], flags=re.IGNORECASE, regex=True)
        out.loc[mask, "broad_category"] = dst["broad"]
        out.loc[mask, "specific_association"] = dst["specific"]
        out.loc[mask, "new_association"] = bool(dst.get("new", False))

    keep = ~out["broad_category"].isin(["Excluded", "Unresolved"])
    sizes = out[keep & out["new_association"]].groupby(["broad_category", "specific_association"]).size()
    small = {k for k, n in sizes.items() if n < cfg.get("min_mentions", 20)}
    is_small = [(b, s) in small for b, s in zip(out["broad_category"], out["specific_association"])]
    out.loc[is_small, "specific_association"] = cfg.get("other_label", "Other (smaller associations)")
    out.loc[is_small, "new_association"] = False
    return out


def main_cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("categorised", type=Path, help="the _categorised.csv written by map_labels.py")
    ap.add_argument("rules", type=Path)
    args = ap.parse_args()

    cat = pd.read_csv(args.categorised, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    final = apply_rules(cat, json.loads(args.rules.read_text(encoding="utf-8")))
    stem = args.categorised.with_suffix("").name.removesuffix("_categorised")
    final.to_csv(args.categorised.with_name(stem + "_final.csv"), index=False, encoding="utf-8-sig")
    s = write_breakdown(final, args.categorised.with_name(stem + "_final_breakdown.xlsx"))
    p = s["population"]
    print(f"Wrote {stem}_final.csv and {stem}_final_breakdown.xlsx")
    print(f"Analytical population {p:,} (excluded {s['excluded']:,}): categorised {p - s['unresolved']:,} "
          f"({(p - s['unresolved']) / p:.1%}), unresolved {s['unresolved']:,} ({s['unresolved'] / p:.1%})")
    print(f"{s['new_tags']} new associations kept (each ≥ threshold), covering {s['new_rows']:,} mentions")


if __name__ == "__main__":
    main_cli()
