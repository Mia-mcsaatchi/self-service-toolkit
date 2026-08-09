"""End-to-end tests for the Self-Service Toolkit backend.

Runs the whole flow: upload -> configure fields -> process -> export, plus the
auth layer. Uses a small synthetic dataset shaped like a real social-listening
export (a `translated_text` column to tag, an `overall_sentiment` column to
branch on) — no real data is committed to the repo.

The OpenAI call is mocked so the tests are deterministic, offline, and free —
we're verifying the plumbing (parsing, prompt routing, conditional branching,
result writing, export, per-user auth), not the model's answers.

Run:  pytest -q         (from the repo root, inside the venv)
"""
from __future__ import annotations

import io
import json
import os
import re
import time

import pandas as pd
import pytest

# Auth is exercised explicitly below; default the whole module to a signed-in
# local user so the pipeline tests aren't about tokens.
os.environ.setdefault("AUTH_DISABLED", "true")
os.environ.setdefault("OPENAI_API_KEY", "sk-mock")

import jwt  # noqa: E402  (import after env is set)
from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402


# --------------------------------------------------------------------------
# Synthetic sample, shaped like the real export: multilingual verbatims plus a
# pre-existing sentiment label to drive conditional branching. Mix of Negative
# and non-Negative rows so both branches of the conditional field are hit.
# --------------------------------------------------------------------------
SAMPLE_ROWS = [
    ("Replacing the stock tyres on my XC40 — any recommendations?", "Neutral"),
    ("The voice assistant keeps failing on my second profile.", "Negative"),
    ("Someone hit my XC40 and drove off, bumper is damaged.", "Negative"),
    ("Love the range on the EX40, best EV I've owned.", "Positive"),
    ("How do I update the infotainment software?", "Neutral"),
    ("Charging is painfully slow at public stations.", "Negative"),
]
SAMPLE_COLUMNS = ["id", "translated_text", "overall_sentiment"]


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [(i + 1, text, sent) for i, (text, sent) in enumerate(SAMPLE_ROWS)],
        columns=SAMPLE_COLUMNS,
    )


# --------------------------------------------------------------------------
# Mock the OpenAI call: read the requested JSON keys out of the built prompt
# and return a plausible value for each, so the pipeline has real data to write.
# --------------------------------------------------------------------------
def _fake_values_for_prompt(prompt: str) -> str:
    m = re.search(r"EXACTLY these keys:\s*\{([^\n]+)\}", prompt)
    keys = re.findall(r'"([^"]+)"', m.group(1)) if m else []
    out = {}
    for k in keys:
        kl = k.lower()
        if "sentiment" in kl:
            out[k] = "positive"
        elif "topic" in kl:
            out[k] = "tyres"
        else:
            out[k] = f"mock::{k}"
    return json.dumps(out)


@pytest.fixture(autouse=True)
def _mock_openai(monkeypatch):
    async def fake_call(session, semaphore, prompt, **kwargs):
        return _fake_values_for_prompt(prompt)

    monkeypatch.setattr(main, "_call_openai", fake_call)
    main._sessions.clear()      # fresh per-user state each test
    yield
    main._sessions.clear()


@pytest.fixture
def client():
    return TestClient(main.app)


# --------------------------------------------------------------------------
# 1. Upload an .xlsx (exercises the file-parsing path)
# --------------------------------------------------------------------------
def test_upload_excel_parses_shape(client):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        _sample_df().to_excel(w, index=False, sheet_name="Sheet1")
    buf.seek(0)

    r = client.post(
        "/api/upload",
        files={"file": ("sample.xlsx", buf,
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["row_count"] == len(SAMPLE_ROWS)
    assert body["columns"] == SAMPLE_COLUMNS


# --------------------------------------------------------------------------
# 2. Full pipeline: default fields + a conditional field + export
# --------------------------------------------------------------------------
CONFIG = {
    "base_prompt": "You are a data analyst.",
    "fields": [
        {"name": "ai_sentiment", "prompt": "Classify sentiment.",
         "reads_from": ["translated_text"], "mode": "default"},
        {"name": "ai_topic", "prompt": "Give the primary topic.",
         "reads_from": ["translated_text"], "mode": "default"},
        # Only fires on rows already labelled Negative — otherwise writes "n/a"
        {"name": "neg_reason", "prompt": "", "reads_from": ["translated_text"],
         "mode": "conditional",
         "branches": [
             {"conditions": [{"column": "overall_sentiment", "operator": "is",
                              "values": ["Negative"]}],
              "prompt": "Summarise the main complaint."},
         ]},
    ],
}


def _load_sample(client):
    df = _sample_df()
    rows = df.astype(str).values.tolist()
    r = client.post("/api/upload-data",
                    json={"columns": list(df.columns), "rows": rows})
    assert r.status_code == 200, r.text
    return df


def test_full_pipeline_writes_expected_columns(client):
    df = _load_sample(client)

    r = client.post("/api/upload-config", json=CONFIG)
    assert r.status_code == 200, r.text
    assert r.json()["field_count"] == 3

    r = client.post("/api/process", json={"max_rows": 0, "max_concurrent": 8})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["row_count"] == len(SAMPLE_ROWS)
    for col in ("ai_sentiment", "ai_topic", "neg_reason"):
        assert col in body["columns"]

    # Pull the whole result back via CSV export and check the logic held.
    csv = client.get("/api/export/csv")
    assert csv.status_code == 200
    # keep_default_na=False so the literal string "n/a" isn't turned into NaN
    out = pd.read_csv(io.BytesIO(csv.content), keep_default_na=False)

    # default fields filled on every row
    assert (out["ai_sentiment"] == "positive").all()
    assert (out["ai_topic"] == "tyres").all()

    # conditional field: "n/a" unless the row was Negative
    neg_mask = df["overall_sentiment"] == "Negative"
    assert neg_mask.sum() > 0
    assert (out.loc[neg_mask.values, "neg_reason"] == "mock::neg_reason").all()
    assert (out.loc[~neg_mask.values, "neg_reason"] == "n/a").all()


def test_xlsx_export_roundtrips(client):
    _load_sample(client)
    client.post("/api/upload-config", json=CONFIG)
    client.post("/api/process", json={"max_rows": 0, "max_concurrent": 8})
    r = client.get("/api/export/xlsx")
    assert r.status_code == 200
    out = pd.read_excel(io.BytesIO(r.content))
    assert len(out) == len(SAMPLE_ROWS)
    assert "ai_sentiment" in out.columns


# --------------------------------------------------------------------------
# 3. Auth layer (independent of AUTH_DISABLED used above)
# --------------------------------------------------------------------------
def test_auth_rejects_and_isolates(monkeypatch):
    monkeypatch.setattr(main, "AUTH_DISABLED", False)
    monkeypatch.setattr(main, "SUPABASE_JWT_SECRET", "test-secret")
    monkeypatch.setattr(main, "ALLOWED_EMAIL_DOMAIN", "mcsaatchi.com")
    main._sessions.clear()
    c = TestClient(main.app)

    def tok(sub, email, secret="test-secret", aud="authenticated", off=3600):
        return jwt.encode(
            {"sub": sub, "email": email, "aud": aud, "exp": int(time.time()) + off},
            secret, algorithm="HS256",
        )

    assert c.get("/api/health").status_code == 200                     # public
    assert c.get("/api/config").status_code == 401                     # no token
    assert c.get("/api/me", headers={"Authorization": f"Bearer {tok('u1','a@mcsaatchi.com')}"}).status_code == 200
    assert c.get("/api/me", headers={"Authorization": f"Bearer {tok('u1','a@gmail.com')}"}).status_code == 403
    assert c.get("/api/me", headers={"Authorization": f"Bearer {tok('u1','a@mcsaatchi.com','wrong')}"}).status_code == 401
    assert c.get("/api/me", headers={"Authorization": f"Bearer {tok('u1','a@mcsaatchi.com',off=-10)}"}).status_code == 401

    # User A stores a config; user B must not see it
    cfg = {"base_prompt": "x", "fields": [{"name": "t", "prompt": "p", "reads_from": ["translated_text"]}]}
    c.post("/api/upload-config", json=cfg, headers={"Authorization": f"Bearer {tok('userA','a@mcsaatchi.com')}"})
    assert c.get("/api/config", headers={"Authorization": f"Bearer {tok('userA','a@mcsaatchi.com')}"}).status_code == 200
    assert c.get("/api/config", headers={"Authorization": f"Bearer {tok('userB','b@mcsaatchi.com')}"}).status_code == 404
