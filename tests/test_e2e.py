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


def test_result_data_json_survives_messy_text(client):
    # A tag column with commas, quotes and newlines — the case CSV parsing breaks on
    df = _sample_df()
    df["key_quote"] = ['He said "great, but slow"\nreally', 'x', 'y', 'z', 'a', 'b']
    client.post("/api/upload-data", json={"columns": list(df.columns), "rows": df.astype(str).values.tolist()})
    r = client.get("/api/result-data")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["columns"][-1] == "key_quote"
    assert len(body["rows"]) == len(SAMPLE_ROWS)
    # the messy first value comes back intact, and translated_text is untouched
    assert 'great, but slow' in body["rows"][0]["key_quote"]
    assert body["rows"][0]["translated_text"].startswith("Replacing the stock tyres")


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
# 3. AI-suggested prompt
# --------------------------------------------------------------------------
def test_suggest_prompt_returns_text(client, monkeypatch):
    async def fake_call(session, semaphore, prompt, **kwargs):
        # sanity: the field name is passed into the LLM prompt
        assert "urgency" in prompt
        return json.dumps({"prompt": "Rate urgency as high, medium, low, or none."})

    monkeypatch.setattr(main, "_call_openai", fake_call)
    r = client.post("/api/suggest-prompt", json={
        "field_name": "urgency",
        "reads_from": ["translated_text"],
        "is_cluster": False,
        "samples": ["Charging is painfully slow.", "Love the range."],
    })
    assert r.status_code == 200, r.text
    assert "urgency" in r.json()["prompt"].lower()


def test_suggest_prompt_empty_llm_is_502(client, monkeypatch):
    async def empty(session, semaphore, prompt, **kwargs):
        return "{}"

    monkeypatch.setattr(main, "_call_openai", empty)
    r = client.post("/api/suggest-prompt", json={"field_name": "topic"})
    assert r.status_code == 502


# --------------------------------------------------------------------------
# 4. Saved datasets (Phase B) — storage mocked as an in-memory PostgREST
# --------------------------------------------------------------------------
def _make_fake_store():
    store = []

    async def fake(method, path, *, params=None, data=None, prefer=None):
        if method == "POST":
            rec = dict(data)
            rec["id"] = "id" + str(len(store) + 1)
            rec["created_at"] = "2026-08-09T00:00:00Z"
            store.append(rec)
            return [rec]
        if method == "GET":
            uid = params["user_id"][3:]            # strip "eq."
            rows = [r for r in store if r["user_id"] == uid]
            if "id" in params:
                did = params["id"][3:]
                rows = [r for r in rows if r["id"] == did]
            return rows
        if method == "DELETE":
            uid, did = params["user_id"][3:], params["id"][3:]
            store[:] = [r for r in store if not (r["user_id"] == uid and r["id"] == did)]
            return None
        return None

    return fake, store


def test_datasets_crud(client, monkeypatch):
    fake, _ = _make_fake_store()
    monkeypatch.setattr(main, "_sb_request", fake)

    df = _sample_df()
    client.post("/api/upload-data", json={"columns": list(df.columns), "rows": df.astype(str).values.tolist()})

    r = client.post("/api/datasets", json={"name": "My run", "use_result": True})
    assert r.status_code == 200, r.text
    ds_id = r.json()["id"]
    assert r.json()["row_count"] == len(SAMPLE_ROWS)

    r = client.get("/api/datasets")
    lst = r.json()["datasets"]
    assert len(lst) == 1 and lst[0]["name"] == "My run"

    r = client.get("/api/datasets/" + ds_id)
    assert r.status_code == 200
    assert r.json()["columns"] == SAMPLE_COLUMNS
    assert len(r.json()["preview"]) > 0

    assert client.delete("/api/datasets/" + ds_id).status_code == 200
    assert client.get("/api/datasets").json()["datasets"] == []


def test_datasets_isolated_per_user(monkeypatch):
    monkeypatch.setattr(main, "AUTH_DISABLED", False)
    monkeypatch.setattr(main, "SUPABASE_JWT_SECRET", "test-secret")
    monkeypatch.setattr(main, "ALLOWED_EMAIL_DOMAIN", "mcsaatchi.com")
    fake, _ = _make_fake_store()
    monkeypatch.setattr(main, "_sb_request", fake)
    main._sessions.clear()
    c = TestClient(main.app)

    def tok(sub, email):
        return jwt.encode({"sub": sub, "email": email, "aud": "authenticated",
                           "exp": int(time.time()) + 3600}, "test-secret", algorithm="HS256")

    A = {"Authorization": f"Bearer {tok('userA', 'a@mcsaatchi.com')}"}
    B = {"Authorization": f"Bearer {tok('userB', 'b@mcsaatchi.com')}"}

    df = _sample_df()
    c.post("/api/upload-data", json={"columns": list(df.columns), "rows": df.astype(str).values.tolist()}, headers=A)
    ds_id = c.post("/api/datasets", json={"name": "A data"}, headers=A).json()["id"]

    assert c.get("/api/datasets", headers=B).json()["datasets"] == []      # B sees nothing
    assert c.get("/api/datasets/" + ds_id, headers=B).status_code == 404   # B can't load A's
    assert c.get("/api/datasets/" + ds_id, headers=A).status_code == 200   # A can


# --------------------------------------------------------------------------
# 5. Dashboards: save / publish (public link) / share (Phase C)
# --------------------------------------------------------------------------
def _make_fake_db():
    tables, ctr = {"dashboards": [], "dashboard_shares": []}, {"n": 0}

    async def fake(method, path, *, params=None, data=None, prefer=None):
        params = params or {}
        rows = tables.setdefault(path, [])
        if method == "POST":
            rec = dict(data); ctr["n"] += 1
            rec.setdefault("id", f"id{ctr['n']}")
            if path == "dashboards":
                rec.setdefault("share_token", f"tok{ctr['n']}")
                rec.setdefault("is_public", False)
            rec.setdefault("created_at", "2026-08-09T00:00:00Z")
            rows.append(rec)
            return [rec]
        if method == "GET":
            res = rows
            for key in ("owner_id", "user_id", "share_token", "dashboard_id", "shared_with_email", "id"):
                val = params.get(key)
                if val and val.startswith("eq."):
                    res = [r for r in res if str(r.get(key)) == val[3:]]
                elif val and val.startswith("in.("):
                    wanted = set(val[4:-1].split(","))
                    res = [r for r in res if str(r.get("id")) in wanted]
            if params.get("is_public") == "eq.true":
                res = [r for r in res if r.get("is_public") is True]
            return res
        if method == "PATCH":
            res = rows
            for key in ("id", "owner_id"):
                val = params.get(key)
                if val and val.startswith("eq."):
                    res = [r for r in res if str(r.get(key)) == val[3:]]
            for r in res:
                r.update(data)
            return res
        if method == "DELETE":
            oid = (params.get("owner_id") or "eq.")[3:]
            did = (params.get("id") or "eq.")[3:]
            tables[path] = [r for r in rows if not (str(r.get("id")) == did and str(r.get("owner_id")) == oid)]
            return None
        return None

    return fake, tables


def test_dashboard_save_publish_public(client, monkeypatch):
    fake, _ = _make_fake_db()
    monkeypatch.setattr(main, "_sb_request", fake)
    df = _sample_df()
    client.post("/api/upload-data", json={"columns": list(df.columns), "rows": df.astype(str).values.tolist()})

    r = client.post("/api/dashboards", json={"name": "Q3 board",
                    "config": {"charts": [{"type": "bar", "column": "overall_sentiment"}]}})
    assert r.status_code == 200, r.text
    did, token = r.json()["id"], r.json()["share_token"]
    assert r.json()["is_public"] is False

    # public link is dead until published
    assert client.get(f"/api/public/dashboard/{token}").status_code == 404

    r = client.post(f"/api/dashboards/{did}/publish", json={"is_public": True})
    assert r.status_code == 200 and r.json()["is_public"] is True

    # now anyone with the token can read it — no auth
    r = client.get(f"/api/public/dashboard/{token}")
    assert r.status_code == 200
    assert r.json()["config"]["charts"][0]["column"] == "overall_sentiment"
    rows = r.json()["rows"]
    assert len(rows) == len(SAMPLE_ROWS)
    # Rows must be dicts keyed by column name (not positional arrays), so the
    # dashboard chart code that reads row[columnName] renders real data.
    assert isinstance(rows[0], dict)
    assert "overall_sentiment" in rows[0]


def test_dashboard_share_with_colleague(monkeypatch):
    monkeypatch.setattr(main, "AUTH_DISABLED", False)
    monkeypatch.setattr(main, "SUPABASE_JWT_SECRET", "test-secret")
    monkeypatch.setattr(main, "ALLOWED_EMAIL_DOMAIN", "mcsaatchi.com")
    fake, _ = _make_fake_db()
    monkeypatch.setattr(main, "_sb_request", fake)
    main._sessions.clear()
    c = TestClient(main.app)

    def tok(sub, email):
        return jwt.encode({"sub": sub, "email": email, "aud": "authenticated",
                           "exp": int(time.time()) + 3600}, "test-secret", algorithm="HS256")
    A = {"Authorization": f"Bearer {tok('userA', 'a@mcsaatchi.com')}"}
    B = {"Authorization": f"Bearer {tok('userB', 'b@mcsaatchi.com')}"}

    df = _sample_df()
    c.post("/api/upload-data", json={"columns": list(df.columns), "rows": df.astype(str).values.tolist()}, headers=A)
    did = c.post("/api/dashboards", json={"name": "A board", "config": {}}, headers=A).json()["id"]

    assert c.get(f"/api/dashboards/{did}", headers=B).status_code == 403        # not shared yet
    assert c.get("/api/shared-dashboards", headers=B).json()["dashboards"] == []

    assert c.post(f"/api/dashboards/{did}/share", json={"email": "b@mcsaatchi.com"}, headers=A).status_code == 200

    shared = c.get("/api/shared-dashboards", headers=B).json()["dashboards"]
    assert len(shared) == 1 and shared[0]["id"] == did
    assert c.get(f"/api/dashboards/{did}", headers=B).status_code == 200        # now B can load it


# --------------------------------------------------------------------------
# 6. Auth layer (independent of AUTH_DISABLED used above)
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
