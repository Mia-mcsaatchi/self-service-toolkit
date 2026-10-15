"""Build the report tables from a categorised mention file. Free — no Claude calls.

Every cut shows reach next to volume (unique authors, unique texts), because
syndicated news and a few prolific accounts can make an association look bigger
than it is.

Sheets: Summary · By broad category · By association · Connotation · Monthly ·
Countries · Languages · Platforms · Accuracy check (100 random rows to verify by hand).

Usage:
    python scripts/analysis_workbook.py data/hydra_relevance_final_final.csv configs/hydra_connotation.json \
        [--sentiment data/sentiment_sample_tagged.csv] [-o data/hydra_analysis.xlsx]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

SKIP = ("Excluded", "Unresolved")
SENTIMENTS = ["positive", "negative", "mixed", "neutral"]


def add_connotation(df: pd.DataFrame, conn: Dict) -> pd.DataFrame:
    broad, specific = conn.get("broad", {}), conn.get("specific", {})

    def rate(b: str, s: str) -> List[str]:
        return specific.get(s) or broad.get(b) or ["unrated", ""]

    pairs = [rate(b, s) for b, s in zip(df["broad_category"], df["specific_association"])]
    df = df.copy()
    df["connotation"] = [p[0] for p in pairs]
    df["connotation_note"] = [p[1] for p in pairs]
    return df


def metrics(g: pd.DataFrame, population: int) -> Dict:
    authors = g["author"][g["author"].str.strip() != ""]
    texts = g["text_group_id"].where(g["text_group_id"].str.strip() != "", g["cleaned_text"])
    top5 = authors.value_counts().head(5).sum()
    countries = g["country_code"][g["country_code"] != ""].value_counts(normalize=True).head(3)
    return {
        "mentions": len(g),
        "% of analytical population": len(g) / population,
        "unique authors": authors.nunique(),
        "unique texts": texts.nunique(),
        "% substantive": (g["status"] == "substantive").mean(),
        "top-5 authors' share": top5 / len(authors) if len(authors) else None,
        "top countries": ", ".join(f"{c} {v:.0%}" for c, v in countries.items()),
        "top platform": g["platform"].value_counts().index[0] if len(g) else "",
    }


def sentiment_cols(g: Optional[pd.DataFrame]) -> Dict:
    if g is None or not len(g):
        return {"sentiment sample n": 0}
    share = g["hydra_sentiment"].str.lower().value_counts(normalize=True)
    out = {"sentiment sample n": len(g)}
    out.update({f"% {s}": share.get(s, 0.0) for s in SENTIMENTS})
    out["net sentiment (pos − neg)"] = share.get("positive", 0.0) - share.get("negative", 0.0)
    if "sentiment_basis" in g.columns:   # v2 prompt: keep explicit and symbolic sentiment apart
        basis = g["sentiment_basis"].str.lower()
        explicit = g[basis == "explicit"]["hydra_sentiment"].str.lower()
        out["% explicit evaluation"] = (basis == "explicit").mean()
        out["% symbolic"] = (basis == "symbolic").mean()
        out["net sentiment, explicit only (share of sample)"] = (
            ((explicit == "positive").sum() - (explicit == "negative").sum()) / len(g))
    if "confidence" in g.columns:
        out["% low confidence"] = (g["confidence"].str.lower() == "low").mean()
    return out


def table(df: pd.DataFrame, keys: List[str], population: int, sent: Optional[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key))
        row.update({"connotation": g["connotation"].mode().iat[0]} if "connotation" not in keys else {})
        row.update(metrics(g, population))
        if sent is not None:
            sg = sent
            for k, v in zip(keys, key):
                sg = sg[sg[k] == v]
            row.update(sentiment_cols(sg))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mentions", ascending=False)


def mix_table(df: pd.DataFrame, by: str, top: int = 20) -> pd.DataFrame:
    """For each country / language: volume, its three biggest associations, connotation mix."""
    rows = []
    for val, g in df.groupby(by):
        if not str(val).strip():
            continue
        lead = g["broad_category"].value_counts(normalize=True).head(3)
        conn = g["connotation"].value_counts(normalize=True)
        rows.append({by: val, "mentions": len(g), "unique authors": g["author"].nunique(),
                     "leading associations": "; ".join(f"{k} {v:.0%}" for k, v in lead.items()),
                     "% positive connotation": conn.get("positive", 0.0),
                     "% negative connotation": conn.get("negative", 0.0),
                     "% neutral connotation": conn.get("neutral", 0.0)})
    return pd.DataFrame(rows).sort_values("mentions", ascending=False).head(top)


def build(df: pd.DataFrame, conn: Dict, sentiment: Optional[pd.DataFrame], path: Path) -> Dict:
    df = add_connotation(df.fillna(""), conn)
    total, excluded = len(df), int((df["broad_category"] == "Excluded").sum())
    population = total - excluded
    body = df[~df["broad_category"].isin(SKIP)]
    sent = None
    if sentiment is not None:
        sent = add_connotation(sentiment.fillna(""), conn)
        sent = sent[sent["hydra_sentiment"].str.lower().isin(SENTIMENTS)]

    dates = pd.to_datetime(df["publication_date"], errors="coerce", utc=True)
    by_conn = table(body, ["connotation"], population, sent)
    summary = [
        ("Total collected mentions", total), ("Exclusions", excluded), ("Analytical population", population),
        ("Categorised mentions", len(body)), ("Categorised share", len(body) / population),
        ("Unresolved mentions", int((df["broad_category"] == "Unresolved").sum())),
        ("Date range", f"{dates.min():%d %b %Y} – {dates.max():%d %b %Y}" if dates.notna().any() else ""),
        ("Countries", df["country_code"].replace("", pd.NA).nunique()),
        ("Languages", df["language"].replace("", pd.NA).nunique()),
        ("Unique authors (categorised)", body["author"].replace("", pd.NA).nunique()),
        ("Broad categories / named associations", f"{body['broad_category'].nunique()} / {body['specific_association'].nunique()}"),
    ]
    for _, r in by_conn.iterrows():
        summary.append((f"Share of categorised mentions with {r['connotation']} connotation", r["mentions"] / len(body)))
    summary.append(("Hydra-specific sentiment sample size", 0 if sent is None else len(sent)))

    body = body.assign(month=dates[body.index].dt.strftime("%Y-%m"))
    monthly = body.pivot_table(index="month", columns="broad_category", values="status", aggfunc="size", fill_value=0)
    monthly = monthly[monthly.sum().sort_values(ascending=False).index]
    monthly_conn = body.pivot_table(index="month", columns="connotation", values="status", aggfunc="size", fill_value=0)
    platforms = body.pivot_table(index="broad_category", columns="platform", values="status", aggfunc="size", fill_value=0)
    platforms = platforms.loc[platforms.sum(axis=1).sort_values(ascending=False).index]

    check = body.sample(min(100, len(body)), random_state=7)
    check = check[[c for c in ("mention_id", "publication_date", "platform", "original_title", "cleaned_text",
                               "context", "broad_category", "specific_association") if c in check.columns]].copy()
    check["cleaned_text"] = check["cleaned_text"].str.slice(0, 1500)
    check["category correct? (Y/N)"] = ""
    check["note"] = ""

    sheets = {
        "Summary": pd.DataFrame(summary, columns=["Measure", "Value"]),
        "By broad category": table(body, ["broad_category"], population, sent),
        "By association": table(body, ["broad_category", "specific_association"], population, sent),
        "Connotation": by_conn,
        "Monthly": monthly.join(monthly_conn.add_prefix("connotation: ")).reset_index(),
        "Countries": mix_table(body, "country_code"),
        "Languages": mix_table(body, "language"),
        "Platforms": platforms.reset_index(),
        "Accuracy check": check,
    }
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, frame in sheets.items():
            frame.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            for i, col in enumerate(frame.columns, start=1):
                letter = ws.cell(row=1, column=i).column_letter
                ws.column_dimensions[letter].width = min(60, max(12, len(str(col)) + 2))
                if str(col).startswith("%") or "share" in str(col) or "net sentiment" in str(col):
                    for cell in ws[letter][1:]:
                        cell.number_format = "0.0%"
            if name == "Summary":
                for cell in ws["B"][1:]:
                    if isinstance(cell.value, float) and cell.value <= 1:
                        cell.number_format = "0.0%"
    return {"population": population, "categorised": len(body), "sentiment_n": 0 if sent is None else len(sent)}


def main_cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("final", type=Path, help="the _final.csv from finalise_breakdown.py")
    ap.add_argument("connotation", type=Path)
    ap.add_argument("--sentiment", type=Path, help="tagged sentiment sample from run_batch.py")
    ap.add_argument("-o", "--output", type=Path)
    args = ap.parse_args()
    df = pd.read_csv(args.final, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    sent = (pd.read_csv(args.sentiment, dtype=str, keep_default_na=False, encoding="utf-8-sig")
            if args.sentiment else None)
    out = args.output or args.final.with_name("hydra_analysis.xlsx")
    s = build(df, json.loads(args.connotation.read_text(encoding="utf-8")), sent, out)
    print(f"Wrote {out} — {s['categorised']:,} categorised mentions"
          + (f", sentiment from {s['sentiment_n']:,} sampled mentions" if s["sentiment_n"] else
             " (no sentiment yet: add --sentiment once the sample is tagged)"))


if __name__ == "__main__":
    main_cli()
