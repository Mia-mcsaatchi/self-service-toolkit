"""Tests for the stratified sampler and the analysis workbook (offline, free)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import analysis_workbook as aw  # noqa: E402
import stratified_sample as ss  # noqa: E402


def _frame():
    rows = []
    for i in range(30):
        broad, spec = [("Marvel", "Marvel HYDRA organisation"), ("Places and geography", "Hydra / Ydra, Greek island"),
                       ("Unresolved", "Unspecified")][i % 3]
        rows.append({"mention_id": str(i), "publication_date": f"2026-0{1 + i % 3}-10T00:00:00Z",
                     "cleaned_text": f"text {i % 10}", "original_title": "", "platform": "twitter",
                     "country_code": "GBR" if i % 2 else "ITA", "language": "en", "author": f"a{i % 4}",
                     "text_group_id": "", "status": "substantive" if i % 2 else "incidental",
                     "context": "x", "broad_category": broad, "specific_association": spec})
    return pd.DataFrame(rows)


def test_stratified_sample_caps_each_group_and_skips_unresolved():
    s = ss.stratified_sample(_frame(), "broad_category", 4)
    assert s["broad_category"].value_counts().to_dict() == {"Marvel": 4, "Places and geography": 4}


def test_workbook_sheets_connotation_reach_and_sentiment(tmp_path):
    conn = json.loads((ROOT / "configs/hydra_connotation.json").read_text())
    df = _frame()
    sent = df[df["broad_category"] == "Marvel"].head(4).assign(hydra_sentiment=["positive", "negative", "negative", "unsure"])
    out = tmp_path / "a.xlsx"
    res = aw.build(df, conn, sent, out)
    assert res == {"population": 30, "categorised": 20, "sentiment_n": 3}       # "unsure" dropped
    x = pd.ExcelFile(out)
    assert x.sheet_names[0] == "Summary" and "Accuracy check" in x.sheet_names
    broad = pd.read_excel(x, "By broad category").set_index("broad_category")
    assert broad.loc["Marvel", "connotation"] == "negative"
    assert broad.loc["Places and geography", "connotation"] == "positive"      # Greek island override
    assert broad.loc["Marvel", "unique authors"] == 4
    assert broad.loc["Marvel", "unique texts"] == 10
    assert round(broad.loc["Marvel", "net sentiment (pos − neg)"], 3) == round(1 / 3 - 2 / 3, 3)
    assert len(pd.read_excel(x, "Accuracy check")) == 20
