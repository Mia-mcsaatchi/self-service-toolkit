"""Tag a large CSV with Claude's Message Batches API.

For jobs too big to sit and watch (tens of thousands of rows). Compared with
the live backend (`run_tagging.py` / the web UI):
  • half the price — batch requests are billed at 50% of standard rates;
  • no request timeouts or rate-limit stalls — Anthropic queues the work and
    most batches finish within an hour (24h at most);
  • identical rows are sent once (syndicated articles, retweets);
  • resumable — batch ids are saved next to the output, so if the script is
    stopped, re-running the same command picks up where it left off.

It builds exactly the same per-row prompts as the backend (main._plan_row_calls),
so results match what the UI would produce with the same config. It talks to
Anthropic directly — the backend doesn't need to be running — and reads
ANTHROPIC_API_KEY from .env like the backend does.

Usage:
    python scripts/run_batch.py data.csv configs/hydra_mentions.json --estimate   # cost only, sends nothing
    python scripts/run_batch.py data.csv configs/hydra_mentions.json --max-rows 300
    python scripts/run_batch.py data.csv configs/hydra_mentions.json
    python scripts/run_batch.py data.csv configs/hydra_mentions.json --model claude-haiku-5-5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import anthropic  # noqa: E402
import main  # noqa: E402  (loads .env; provides the prompt builder)
from run_tagging import expand_fields  # noqa: E402

BATCH_SIZE = 10_000      # requests per batch; the API allows 100k / 256 MB
MAX_ROUNDS = 3           # submissions per request before giving up on it
EST_OUTPUT_TOKENS = 80   # a context/summary/status answer is ~50-80 tokens

# Standard $ per 1M tokens (input, output). Batches bill half of this.
PRICES = {
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-opus-5-5": (4.00, 20.00),
}

# Models that accept thinking disabled at low effort. The rest reject
# {"type": "disabled"}, so thinking is left at its default and effort kept low.
THINKING_OFF_OK = {"claude-sonnet-5", "claude-haiku-5-5", "claude-opus-5", "claude-opus-4-8"}

Plan = List[Tuple[Dict[str, str], List[Tuple[str, List[str]]]]]


def plan_requests(df: pd.DataFrame, fields: List[Dict[str, Any]], base_prompt: str
                  ) -> Tuple[Plan, Dict[str, Tuple[str, List[str]]]]:
    """Per-row call plan, plus the distinct requests keyed by prompt hash."""
    rows: Plan = []
    unique: Dict[str, Tuple[str, List[str]]] = {}
    for _, row in df.iterrows():
        known, calls = main._plan_row_calls(fields, base_prompt, row)
        keyed = []
        for prompt, names in calls:
            key = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
            unique.setdefault(key, (prompt, names))
            keyed.append((key, names))
        rows.append((known, keyed))
    return rows, unique


def request_params(prompt: str, field_names: List[str], model: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "model": model,
        "max_tokens": 2048,
        "output_config": {
            "format": {"type": "json_schema", "schema": main._tagging_schema(field_names)},
            "effort": "low",
        },
        "messages": [{"role": "user", "content": prompt}],
    }
    if model in THINKING_OFF_OK:
        params["thinking"] = {"type": "disabled"}
    return params


def estimate(client, unique: Dict[str, Tuple[str, List[str]]], model: str, n_rows: int) -> None:
    sample = random.Random(0).sample(sorted(unique), min(20, len(unique)))
    counts = [
        client.messages.count_tokens(
            model=model, messages=[{"role": "user", "content": unique[k][0]}]
        ).input_tokens
        for k in sample
    ]
    avg_in = sum(counts) / len(counts)
    print(f"{n_rows:,} rows -> {len(unique):,} distinct requests "
          f"(~{avg_in:,.0f} input tokens each, measured on {len(counts)})")
    total_in = avg_in * len(unique)
    total_out = EST_OUTPUT_TOKENS * len(unique)
    print(f"Estimated tokens: {total_in / 1e6:,.1f}M in, {total_out / 1e6:,.1f}M out")
    for m, (pin, pout) in PRICES.items():
        cost = (total_in * pin + total_out * pout) / 1e6 / 2
        marker = "  <- selected" if m == model else ""
        print(f"  {m:<20} ~${cost:,.2f} (batch price){marker}")
    if model not in PRICES:
        print(f"  (no price on file for {model})")


def fingerprint(unique: Dict[str, Any], model: str) -> str:
    h = hashlib.sha1(model.encode())
    for k in sorted(unique):
        h.update(k.encode())
    return h.hexdigest()


def run(df: pd.DataFrame, fields: List[Dict[str, Any]], base_prompt: str, client,
        model: str, state_path: Path, poll_seconds: float = 60, fresh: bool = False
        ) -> Tuple[pd.DataFrame, Dict[str, int]]:
    rows, unique = plan_requests(df, fields, base_prompt)
    fp = fingerprint(unique, model)

    state: Dict[str, Any] = {}
    if state_path.exists() and not fresh:
        state = json.loads(state_path.read_text())
        if state.get("fingerprint") != fp:
            sys.exit(f"{state_path} belongs to a different run (data, config or model changed). "
                     "Pass --fresh to start over, or -o to write somewhere else.")
        print(f"Resuming: {len(state['batches'])} batch(es) already submitted")
    if not state:
        state = {"fingerprint": fp, "model": model, "batches": [], "rounds": {}}

    def save() -> None:
        state_path.write_text(json.dumps(state, indent=2))

    answers: Dict[str, Dict[str, Any]] = {}
    collected: set = set()
    usage = {"input_tokens": 0, "output_tokens": 0, "refused": 0, "failed": 0}

    while True:
        statuses = [client.messages.batches.retrieve(b) for b in state["batches"]]
        running = [s for s in statuses if s.processing_status != "ended"]
        if running:
            done = sum(s.request_counts.succeeded + s.request_counts.errored
                       + s.request_counts.expired + s.request_counts.canceled for s in statuses)
            total = done + sum(s.request_counts.processing for s in statuses)
            print(f"  {time.strftime('%H:%M')}  {done:,}/{total:,} requests finished — waiting")
            time.sleep(poll_seconds)
            continue

        for s in statuses:
            if s.id in collected:
                continue
            for r in client.messages.batches.results(s.id):
                if r.result.type != "succeeded":
                    continue  # errored / expired / canceled: resubmitted below
                msg = r.result.message
                usage["input_tokens"] += msg.usage.input_tokens
                usage["output_tokens"] += msg.usage.output_tokens
                if msg.stop_reason == "refusal":
                    answers[r.custom_id] = {}  # retrying won't change a refusal
                    continue
                text = next((b.text for b in msg.content if b.type == "text"), "")
                try:
                    answers[r.custom_id] = json.loads(text)
                except ValueError:
                    pass  # truncated / malformed: resubmit
            collected.add(s.id)

        missing = [k for k in unique if k not in answers
                   and state["rounds"].get(k, 0) < MAX_ROUNDS]
        if not missing:
            break
        print(f"Submitting {len(missing):,} request(s)")
        for i in range(0, len(missing), BATCH_SIZE):
            chunk = missing[i:i + BATCH_SIZE]
            batch = client.messages.batches.create(requests=[
                {"custom_id": k, "params": request_params(*unique[k], model)} for k in chunk
            ])
            state["batches"].append(batch.id)
            for k in chunk:
                state["rounds"][k] = state["rounds"].get(k, 0) + 1
            save()  # after every batch, so a stop here never loses a paid-for batch
            print(f"  batch {batch.id}: {len(chunk):,} requests")

    out = df.copy()
    for fn in (f["name"] for f in fields):
        if fn not in out.columns:
            out[fn] = ""
    for i, (known, keyed) in enumerate(rows):
        values = dict(known)
        for key, names in keyed:
            parsed = answers.get(key)
            if parsed is None:
                usage["failed"] += 1
                parsed = {}
            elif not parsed:
                usage["refused"] += 1
            for fn in names:
                values[fn] = str(parsed.get(fn, "unsure"))
        for fn, v in values.items():
            out.iat[i, out.columns.get_loc(fn)] = v
    return out, usage


def main_cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", type=Path)
    ap.add_argument("config", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="default: <input>_tagged.csv")
    ap.add_argument("--model", default=main.TAGGING_MODEL,
                    help=f"Claude model (default {main.TAGGING_MODEL}, same as the backend)")
    ap.add_argument("--max-rows", type=int, default=0, help="only the first N rows (0 = all)")
    ap.add_argument("--estimate", action="store_true", help="print the expected cost and stop")
    ap.add_argument("--fresh", action="store_true", help="ignore a saved run and start over")
    args = ap.parse_args()

    if not main.ANTHROPIC_API_KEY:
        sys.exit("ANTHROPIC_API_KEY is not set — add it to .env")

    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if args.max_rows:
        df = df.head(args.max_rows)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fields = expand_fields(config)
    missing = {c for f in fields for c in f["reads_from"]} - set(df.columns) - {f["name"] for f in fields}
    if missing:
        sys.exit(f"Config reads from columns not in the CSV: {', '.join(sorted(missing))}")
    base_prompt = config.get("base_prompt") or main.BASE_PROMPT

    client = anthropic.Anthropic(api_key=main.ANTHROPIC_API_KEY, max_retries=5)

    if args.estimate:
        _, unique = plan_requests(df, fields, base_prompt)
        estimate(client, unique, args.model, len(df))
        return

    out_path = args.output or args.csv.with_name(args.csv.stem + "_tagged.csv")
    state_path = out_path.with_name(out_path.name + ".batch.json")
    try:
        out, usage = run(df, fields, base_prompt, client, args.model, state_path, fresh=args.fresh)
    except KeyboardInterrupt:
        sys.exit(f"\nStopped. Submitted batches keep running at Anthropic; "
                 f"re-run the same command to collect them (state in {state_path.name}).")

    out.to_csv(out_path, index=False, encoding="utf-8-sig")  # BOM so Excel reads accents
    print(f"Wrote {out_path} ({len(out):,} rows)")
    pin, pout = PRICES.get(args.model, (0, 0))
    cost = (usage["input_tokens"] * pin + usage["output_tokens"] * pout) / 1e6 / 2
    print(f"Tokens: {usage['input_tokens']:,} in / {usage['output_tokens']:,} out"
          + (f" (~${cost:,.2f} at batch price)" if pin else ""))
    if usage["failed"] or usage["refused"]:
        print(f"Unanswered calls: {usage['failed']} failed after {MAX_ROUNDS} tries, "
              f"{usage['refused']} refused — those cells say 'unsure'")
    for f in fields:
        counts = out[f["name"]].value_counts()
        if len(counts) <= 10:
            print(f"{f['name']}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main_cli()
