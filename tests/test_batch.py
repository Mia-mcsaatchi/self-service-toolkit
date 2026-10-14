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
        self.broke = set()                  # batches sent while out of credit
        self.fail_first = set(fail_first)   # custom_ids that error once (server hiccup)
        self.out_of_credit = False
        self.submitted = []

    def create(self, requests):
        bid = f"msgbatch_{len(self.batches)}"
        self.batches[bid] = requests
        if self.out_of_credit:
            self.broke.add(bid)
        self.submitted.append([r["custom_id"] for r in requests])
        return SimpleNamespace(id=bid)

    def retrieve(self, bid):
        n = len(self.batches[bid])
        counts = SimpleNamespace(succeeded=n, errored=0, expired=0, canceled=0, processing=0)
        return SimpleNamespace(id=bid, processing_status="ended", request_counts=counts)

    @staticmethod
    def _error(cid, etype, message):
        err = SimpleNamespace(error=SimpleNamespace(type=etype, message=message))
        return SimpleNamespace(custom_id=cid, result=SimpleNamespace(type="errored", error=err))

    def results(self, bid):
        for req in self.batches[bid]:
            cid = req["custom_id"]
            if bid in self.broke:
                yield self._error(cid, "billing_error", "Your credit balance is too low.")
                continue
            if cid in self.fail_first:
                self.fail_first.discard(cid)
                yield self._error(cid, "api_error", "Internal server error")
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


def test_out_of_credit_stops_then_resumes_after_top_up(tmp_path):
    df = pd.DataFrame(ROWS)
    state = tmp_path / "state.json"
    batches = FakeBatches()
    batches.out_of_credit = True
    out, usage = run_batch.run(df, CLUSTER_FIELDS, "Base.", _fake_client(batches), "claude-sonnet-5",
                               state, poll_seconds=0, parallel=1, batch_size=1)
    assert "credit" in usage["stopped"]
    assert len(batches.submitted) == 1          # stopped after one small batch, not all of them
    assert set(out["status"]) == {"pending"}

    batches.out_of_credit = False               # topped up
    out, usage = run_batch.run(df, CLUSTER_FIELDS, "Base.", _fake_client(batches), "claude-sonnet-5",
                               state, poll_seconds=0, parallel=1, batch_size=1)
    assert usage["stopped"] is None
    assert list(out["status"]) == ["status-for-torrent", "status-for-myth", "status-for-torrent"]


def test_budget_blocks_sending_and_marks_pending(tmp_path):
    df = pd.DataFrame(ROWS)
    batches = FakeBatches()
    out, usage = run_batch.run(df, CLUSTER_FIELDS, "Base.", _fake_client(batches), "claude-sonnet-5",
                               tmp_path / "state.json", poll_seconds=0, budget=0.000001)
    assert batches.submitted == []
    assert "budget" in usage["stopped"]
    assert set(out["context"]) == {"pending"}

    out, usage = run_batch.run(df, CLUSTER_FIELDS, "Base.", _fake_client(batches), "claude-sonnet-5",
                               tmp_path / "state.json", poll_seconds=0, budget=5)
    assert usage["stopped"] is None and "pending" not in set(out["context"])


def test_old_state_file_format_still_resumes(tmp_path):
    df = pd.DataFrame(ROWS)
    state = tmp_path / "state.json"
    batches = FakeBatches()
    run_batch.run(df, CLUSTER_FIELDS, "Base.", _fake_client(batches), "claude-sonnet-5", state, poll_seconds=0)
    saved = json.loads(state.read_text())
    saved["batches"] = [b["id"] for b in saved["batches"]]   # the earlier bare-id format
    state.write_text(json.dumps(saved))
    out, _ = run_batch.run(df, CLUSTER_FIELDS, "Base.", _fake_client(batches), "claude-sonnet-5", state, poll_seconds=0)
    assert len(batches.submitted) == 1
    assert "pending" not in set(out["status"])


# --------------------------------------------------------------------------
# Trimming long posts
# --------------------------------------------------------------------------
LONG_POST = ("Opening about superheroes. " + "filler words here. " * 200
             + "Bucky had a past with HYDRA and nightmares. " + "more filler. " * 200)


def test_trim_keeps_opening_and_hydra_context_with_note():
    out = main._trim_text(LONG_POST, "hydra", 100)
    assert out.startswith("[Shortened:")
    assert "1 match(es)" in out
    assert "Opening about superheroes." in out
    assert "past with HYDRA and nightmares" in out
    assert len(out) < len(LONG_POST) / 3


def test_trim_leaves_short_text_alone():
    short = "Hydra is a torrent client I like."
    assert main._trim_text(short, "hydra", 100) == short
    assert main._trim_text(LONG_POST, "hydra", 0) == LONG_POST


def test_trim_applies_to_prompt_but_not_to_conditions_or_output(tmp_path):
    df = pd.DataFrame([{"id": "1", "text": LONG_POST}])
    trim = {"columns": ["text"], "pattern": "hydra", "chars": 100}
    _, calls = main._plan_row_calls(CLUSTER_FIELDS, "Base.", df.iloc[0], trim)
    assert "[Shortened:" in calls[0][0]

    batches = FakeBatches()
    out, _ = run_batch.run(df, CLUSTER_FIELDS, "Base.", _fake_client(batches), "claude-sonnet-5",
                           tmp_path / "s.json", poll_seconds=0, trim=trim)
    assert "[Shortened:" in batches.batches["msgbatch_0"][0]["params"]["messages"][0]["content"]
    assert out.loc[0, "text"] == LONG_POST          # the output file keeps the full text


def test_estimate_scales_by_length(capsys):
    unique = {"a": ("x" * 100, ["status"]), "b": ("x" * 900, ["status"])}
    counter = SimpleNamespace(count_tokens=lambda model, messages: SimpleNamespace(
        input_tokens=len(messages[0]["content"]) // 4))
    run_batch.estimate(SimpleNamespace(messages=counter), unique, "claude-sonnet-5", 2)
    assert "~125 input tokens each" in capsys.readouterr().out   # (100+900)/4/2


def test_compare_runs_splits_trimmed_and_unchanged(tmp_path, monkeypatch, capsys):
    import compare_runs
    base = pd.DataFrame([
        {"mention_id": "1", "cleaned_text": LONG_POST, "original_title": "", "context": "Marvel HYDRA",
         "summary": "s", "status": "incidental"},
        {"mention_id": "2", "cleaned_text": "short hydra", "original_title": "", "context": "Myth",
         "summary": "s", "status": "incidental"},
    ])
    cand = base.copy()
    cand.loc[0, "status"] = "substantive"
    base.to_csv(tmp_path / "full.csv", index=False)
    cand.to_csv(tmp_path / "trim.csv", index=False)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({
        "fields": [{"names": ["context", "summary", "status"], "prompt": "p",
                    "reads_from": ["cleaned_text", "original_title"]}],
        "trim": {"columns": ["cleaned_text"], "pattern": "hydra", "chars": 100},
    }))
    monkeypatch.setattr(sys, "argv", ["compare_runs.py", str(tmp_path / "full.csv"),
                                      str(tmp_path / "trim.csv"), "--config", str(cfg)])
    compare_runs.main_cli()
    printed = capsys.readouterr().out
    assert "1 shortened" in printed and "status" in printed and "0% agree" in printed
    diff = pd.read_csv(tmp_path / "trim_diff.csv", dtype=str)
    assert list(diff["mention_id"]) == ["1"]
