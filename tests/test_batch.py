"""Tests for the token-saving changes and the Batch API runner.

- A cluster's shared prompt is written once per request, not once per key.
- Identical rows cost one Claude call, including across /api/process chunks.
- scripts/run_batch.py sends one request per distinct prompt, resubmits
  failures, resumes from its state file, and writes the same columns the
  backend would — all against a fake Batches client (offline, free).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

os.environ.setdefault("AUTH_DISABLED", "true")
os.environ.setdefault("OPENAI_API_KEY", "sk-mock")

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import run_batch  # noqa: E402

CLUSTER_PROMPT = "Analyse how Hydra is used. Give context, summary and status."
CLUSTER_FIELDS = [
    {"name": n, "prompt": CLUSTER_PROMPT, "reads_from": ["text"], "mode": "default"}
    for n in ("context", "summary", "status")
]
ROWS = [
    {"id": "1", "text": "Hydra the torrent client downloads fast."},
    {"id": "2", "text": "Like the Hydra, cut one head and two grow."},
    {"id": "3", "text": "Hydra the torrent client downloads fast."},  # duplicate of row 1
]


def _keys(prompt: str):
    m = re.search(r"EXACTLY these keys:\s*\{([^\n]+)\}", prompt)
    return re.findall(r'"([^"]+)"', m.group(1))


def _answer(prompt: str):
    return {k: f"{k}-for-{'torrent' if 'torrent' in prompt else 'myth'}" for k in _keys(prompt)}


def test_cluster_prompt_written_once():
    row = pd.Series(ROWS[0])
    prompt = main._build_row_prompt("Base.", CLUSTER_FIELDS, row)
    assert prompt.count(CLUSTER_PROMPT) == 1
    assert _keys(prompt) == ["context", "summary", "status"]


def test_independent_fields_keep_per_key_instructions():
    fields = [
        {"name": "a", "prompt": "Prompt A.", "reads_from": ["text"]},
        {"name": "b", "prompt": "Prompt B.", "reads_from": ["text"]},
    ]
    prompt = main._build_row_prompt("Base.", fields, pd.Series(ROWS[0]))
    assert '"a": Prompt A.' in prompt and '"b": Prompt B.' in prompt


def test_duplicate_rows_cost_one_call_across_chunks(monkeypatch):
    calls = []

    async def fake_call(semaphore, prompt, **kwargs):
        calls.append(prompt)
        return _answer(prompt)

    monkeypatch.setattr(main, "_call_claude", fake_call)
    monkeypatch.setattr(main, "ANTHROPIC_API_KEY", "sk-ant-test")
    main._sessions.clear()
    client = TestClient(main.app)

    cols = ["id", "text"]
    client.post("/api/upload-data", json={"columns": cols, "rows": [[r[c] for c in cols] for r in ROWS]})
    client.post("/api/upload-config", json={"fields": CLUSTER_FIELDS})
    for offset in (0, 2):  # rows 1 and 3 land in different chunks
        r = client.post("/api/process", json={"offset": offset, "limit": 2})
        assert r.status_code == 200, r.text

    assert len(calls) == 2  # torrent once, myth once
    out = client.get("/api/result-data").json()["rows"]
    assert [r["context"] for r in out] == ["context-for-torrent", "context-for-myth", "context-for-torrent"]
    main._sessions.clear()


# --------------------------------------------------------------------------
# Fake Message Batches client
# --------------------------------------------------------------------------
class FakeBatches:
    def __init__(self, fail_first=()):
        self.batches = {}
        self.fail_first = set(fail_first)  # custom_ids that error on first submission
        self.submitted = []

    def create(self, requests):
        bid = f"msgbatch_{len(self.batches)}"
        self.batches[bid] = requests
        self.submitted.append([r["custom_id"] for r in requests])
        return SimpleNamespace(id=bid)

    def retrieve(self, bid):
        n = len(self.batches[bid])
        counts = SimpleNamespace(succeeded=n, errored=0, expired=0, canceled=0, processing=0)
        return SimpleNamespace(id=bid, processing_status="ended", request_counts=counts)

    def results(self, bid):
        for req in self.batches[bid]:
            cid = req["custom_id"]
            if cid in self.fail_first:
                self.fail_first.discard(cid)
                yield SimpleNamespace(custom_id=cid, result=SimpleNamespace(type="errored"))
                continue
            prompt = req["params"]["messages"][0]["content"]
            msg = SimpleNamespace(
                content=[SimpleNamespace(type="text", text=json.dumps(_answer(prompt)))],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=100, output_tokens=20),
            )
            yield SimpleNamespace(custom_id=cid, result=SimpleNamespace(type="succeeded", message=msg))


def _fake_client(batches):
    return SimpleNamespace(messages=SimpleNamespace(batches=batches))


def test_batch_runner_dedupes_retries_and_writes_columns(tmp_path):
    df = pd.DataFrame(ROWS)
    _, unique = run_batch.plan_requests(df, CLUSTER_FIELDS, "Base.")
    assert len(unique) == 2
    flaky = sorted(unique)[0]

    batches = FakeBatches(fail_first=[flaky])
    out, usage = run_batch.run(df, CLUSTER_FIELDS, "Base.", _fake_client(batches),
                               "claude-sonnet-5", tmp_path / "state.json", poll_seconds=0)

    assert batches.submitted == [list(unique), [flaky]]
    assert list(out["context"]) == ["context-for-torrent", "context-for-myth", "context-for-torrent"]
    assert list(out["status"]) == ["status-for-torrent", "status-for-myth", "status-for-torrent"]
    assert usage["failed"] == 0
    params = batches.batches["msgbatch_0"][0]["params"]
    assert params["thinking"] == {"type": "disabled"}
    assert params["output_config"]["format"]["schema"]["required"] == ["context", "summary", "status"]


def test_batch_runner_resumes_without_resubmitting(tmp_path):
    df = pd.DataFrame(ROWS)
    state = tmp_path / "state.json"
    batches = FakeBatches()
    run_batch.run(df, CLUSTER_FIELDS, "Base.", _fake_client(batches), "claude-sonnet-5", state, poll_seconds=0)
    assert len(batches.submitted) == 1

    out, _ = run_batch.run(df, CLUSTER_FIELDS, "Base.", _fake_client(batches), "claude-sonnet-5", state, poll_seconds=0)
    assert len(batches.submitted) == 1  # collected from the saved batch, nothing new sent
    assert list(out["summary"]) == ["summary-for-torrent", "summary-for-myth", "summary-for-torrent"]

    with pytest.raises(SystemExit):  # a different model must not reuse that state
        run_batch.run(df, CLUSTER_FIELDS, "Base.", _fake_client(batches), "claude-haiku-5-5", state, poll_seconds=0)
