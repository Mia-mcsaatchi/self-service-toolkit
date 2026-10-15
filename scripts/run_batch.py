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
    python scripts/run_batch.py data.csv configs/hydra_mentions.json --budget 120
    python scripts/run_batch.py data.csv configs/hydra_mentions.json --model claude-haiku-5-5

Big runs are sent a few batches at a time (--parallel, --batch-size), so a
credit or limit problem stops the run after one batch's worth of failures
rather than all of it. --budget caps spend: no new batch is sent if it could
take the total past the cap. Whenever the run stops early (budget, credit,
permissions, Ctrl+C) the finished rows are already written — unfinished cells
say "pending" — and re-running the same command carries on from there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import anthropic  # noqa: E402
import main  # noqa: E402  (loads .env; provides the prompt builder)
from run_tagging import expand_fields  # noqa: E402

BATCH_SIZE = 5_000       # requests per batch; the API allows 100k / 256 MB
PARALLEL = 4             # batches in flight at once
MAX_ROUNDS = 3           # submissions per request per run before giving up on it
EST_OUTPUT_TOKENS = 80   # a context/summary/status answer is ~50-80 tokens
CHARS_PER_TOKEN = 3.5    # rough, for budgeting until real usage comes back

# Errors that will hit every request until a person fixes the account, so
# resubmitting is pointless: stop and say what to fix.
ACCOUNT_ERRORS = {"billing_error", "authentication_error", "permission_error"}

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


def plan_requests(df: pd.DataFrame, fields: List[Dict[str, Any]], base_prompt: str,
                  trim: Optional[Dict[str, Any]] = None,
                  ) -> Tuple[Plan, Dict[str, Tuple[str, List[str]]]]:
    """Per-row call plan, plus the distinct requests keyed by prompt hash."""
    rows: Plan = []
    unique: Dict[str, Tuple[str, List[str]]] = {}
    for _, row in df.iterrows():
        known, calls = main._plan_row_calls(fields, base_prompt, row, trim)
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
    """Token count of the whole job, from exact counts on a sample scaled by
    character length (post lengths vary too much to average a small sample)."""
    sample = random.Random(0).sample(sorted(unique), min(50, len(unique)))
    def count(k: str) -> int:
        # Count the request exactly as it will be sent: the JSON output format
        # adds ~250 input tokens per request that a bare prompt count misses.
        params = request_params(*unique[k], model)
        params.pop("max_tokens")
        return client.messages.count_tokens(**params).input_tokens

    counted = sum(count(k) for k in sample)
    tokens_per_char = counted / sum(len(unique[k][0]) for k in sample)
    total_in = tokens_per_char * sum(len(p) for p, _ in unique.values())
    total_out = EST_OUTPUT_TOKENS * len(unique)
    print(f"{n_rows:,} rows -> {len(unique):,} distinct requests "
          f"(~{total_in / len(unique):,.0f} input tokens each)")
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


def batch_cost(model: str, input_tokens: float, output_tokens: float) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return (input_tokens * pin + output_tokens * pout) / 1e6 / 2


def _account_problem(message: str, error_type: str = "") -> bool:
    return error_type in ACCOUNT_ERRORS or "credit balance" in message.lower()


def run(df: pd.DataFrame, fields: List[Dict[str, Any]], base_prompt: str, client,
        model: str, state_path: Path, poll_seconds: float = 60, fresh: bool = False,
        budget: Optional[float] = None, parallel: int = PARALLEL, batch_size: int = BATCH_SIZE,
        trim: Optional[Dict[str, Any]] = None,
        ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run (or resume) the job. Returns the tagged frame and a usage summary;
    usage["stopped"] holds the reason when the run ended before finishing."""
    rows, unique = plan_requests(df, fields, base_prompt, trim)
    fp = fingerprint(unique, model)
    if budget is not None and model not in PRICES:
        sys.exit(f"No price on file for {model}, so --budget can't be enforced.")

    state: Dict[str, Any] = {}
    if state_path.exists() and not fresh:
        state = json.loads(state_path.read_text())
        if state.get("fingerprint") != fp:
            sys.exit(f"{state_path} belongs to a different run (data, config or model changed). "
                     "Pass --fresh to start over, or -o to write somewhere else.")
        print(f"Resuming: {len(state['batches'])} batch(es) already submitted")
    if not state:
        state = {"fingerprint": fp, "model": model, "batches": []}
    # Older state files listed bare batch ids; their keys are unknown.
    state["batches"] = [b if isinstance(b, dict) else {"id": b, "keys": None}
                        for b in state["batches"]]

    def save() -> None:
        state_path.write_text(json.dumps(state))

    answers: Dict[str, Dict[str, Any]] = {}
    collected: set = set()
    sent_this_run: set = set()  # account errors from earlier runs are assumed fixed
    tries: Dict[str, int] = {}
    usage: Dict[str, Any] = {"input_tokens": 0, "output_tokens": 0, "succeeded": 0,
                             "refused": 0, "failed": 0, "pending": 0, "stopped": None}
    est_per_request = batch_cost(
        model,
        sum(len(p) for p, _ in unique.values()) / max(len(unique), 1) / CHARS_PER_TOKEN,
        EST_OUTPUT_TOKENS,
    )

    def per_request() -> float:  # real average once results are in
        if usage["succeeded"]:
            return batch_cost(model, usage["input_tokens"], usage["output_tokens"]) / usage["succeeded"]
        return est_per_request

    while True:
        statuses = {b["id"]: client.messages.batches.retrieve(b["id"]) for b in state["batches"]}

        for b in state["batches"]:
            s = statuses[b["id"]]
            if s.processing_status != "ended" or b["id"] in collected:
                continue
            for r in client.messages.batches.results(b["id"]):
                if r.result.type == "errored":
                    err = r.result.error.error
                    if b["id"] in sent_this_run and _account_problem(err.message, err.type):
                        usage["stopped"] = f"{err.type}: {err.message}"
                    continue  # retried below (or after the account is fixed)
                if r.result.type != "succeeded":
                    continue  # expired / canceled: resubmitted below
                msg = r.result.message
                usage["input_tokens"] += msg.usage.input_tokens
                usage["output_tokens"] += msg.usage.output_tokens
                usage["succeeded"] += 1
                if msg.stop_reason == "refusal":
                    answers[r.custom_id] = {}  # retrying won't change a refusal
                    continue
                text = next((blk.text for blk in msg.content if blk.type == "text"), "")
                try:
                    answers[r.custom_id] = json.loads(text)
                except ValueError:
                    pass  # truncated / malformed: resubmit
            collected.add(b["id"])

        running = [b for b in state["batches"] if statuses[b["id"]].processing_status != "ended"]
        if usage["stopped"]:
            break  # running batches carry on at Anthropic; the next run collects them
        if any(b["keys"] is None for b in running):
            in_flight = set(unique)  # unknown contents: send nothing new until it ends
        else:
            in_flight = {k for b in running for k in b["keys"]}
        missing = [k for k in unique if k not in answers and k not in in_flight
                   and tries.get(k, 0) < MAX_ROUNDS]
        if not missing and not running:
            break

        spent = batch_cost(model, usage["input_tokens"], usage["output_tokens"])
        committed = spent + len(in_flight) * per_request()
        for _ in range(max(parallel - len(running), 0)):
            if not missing:
                break
            chunk = missing[:batch_size]
            if budget is not None and committed + len(chunk) * per_request() > budget:
                if not running:
                    usage["stopped"] = (f"budget of ${budget:,.2f} reached "
                                        f"(${spent:,.2f} spent, {len(missing):,} requests left)")
                break
            try:
                batch = client.messages.batches.create(requests=[
                    {"custom_id": k, "params": request_params(*unique[k], model)} for k in chunk
                ])
            except anthropic.RateLimitError:
                print("  Batch queue is full — will try again shortly")
                break
            except anthropic.APIStatusError as e:
                if e.status_code in (401, 402, 403) or _account_problem(str(e.message)):
                    usage["stopped"] = str(e.message)
                    break
                raise
            state["batches"].append({"id": batch.id, "keys": chunk})
            sent_this_run.add(batch.id)
            save()  # straight away, so a stop here never loses a paid-for batch
            for k in chunk:
                tries[k] = tries.get(k, 0) + 1
            missing = missing[len(chunk):]
            committed += len(chunk) * per_request()
            running.append(state["batches"][-1])
            print(f"  sent batch {batch.id}: {len(chunk):,} requests")
        if usage["stopped"]:
            break

        done = len([k for k in unique if k in answers])
        print(f"  {time.strftime('%H:%M')}  {done:,}/{len(unique):,} answered, "
              f"{len(running)} batch(es) running, ~${batch_cost(model, usage['input_tokens'], usage['output_tokens']):,.2f} spent")
        time.sleep(poll_seconds)

    out = df.copy()
    for fn in (f["name"] for f in fields):
        if fn not in out.columns:
            out[fn] = ""
    for i, (known, keyed) in enumerate(rows):
        values = dict(known)
        for key, names in keyed:
            parsed = answers.get(key)
            if parsed is None:
                if usage["stopped"]:
                    usage["pending"] += 1
                    parsed = {fn: "pending" for fn in names}
                else:
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
    ap.add_argument("--budget", type=float, help="stop before total spend could pass this many US dollars")
    ap.add_argument("--parallel", type=int, default=PARALLEL, help=f"batches in flight at once (default {PARALLEL})")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"requests per batch (default {BATCH_SIZE:,})")
    ap.add_argument("--trim-chars", type=int,
                    help="override the config's trim size (characters kept either side of each match); 0 = no trimming")
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
    trim = config.get("trim")
    if args.trim_chars is not None:
        trim = dict(trim, chars=args.trim_chars) if trim and args.trim_chars > 0 else None
        if args.trim_chars > 0 and not config.get("trim"):
            sys.exit("--trim-chars needs a \"trim\" section in the config (columns + pattern).")
    if trim and int(trim.get("chars", 500)) <= 0:
        trim = None
    if trim:
        trimmed = sum(1 for col in trim["columns"] if col in df.columns
                      for t in df[col] if main._trim_text(t, trim["pattern"], int(trim.get("chars", 500))) != t)
        print(f"Trimming long texts to ±{trim.get('chars', 500)} characters around "
              f"\"{trim['pattern']}\": {trimmed:,} of {len(df):,} rows shortened")

    client = anthropic.Anthropic(api_key=main.ANTHROPIC_API_KEY, max_retries=5)

    if args.estimate:
        _, unique = plan_requests(df, fields, base_prompt, trim)
        estimate(client, unique, args.model, len(df))
        return

    out_path = args.output or args.csv.with_name(args.csv.stem + "_tagged.csv")
    state_path = out_path.with_name(out_path.name + ".batch.json")
    try:
        out, usage = run(df, fields, base_prompt, client, args.model, state_path, fresh=args.fresh,
                         budget=args.budget, parallel=args.parallel, batch_size=args.batch_size,
                         trim=trim)
    except KeyboardInterrupt:
        sys.exit(f"\nStopped. Sent batches keep running at Anthropic; re-run the same command "
                 f"to collect them and send the rest (progress is in {state_path.name}).")

    out.to_csv(out_path, index=False, encoding="utf-8-sig")  # BOM so Excel reads accents
    print(f"Wrote {out_path} ({len(out):,} rows)")
    cost = batch_cost(args.model, usage["input_tokens"], usage["output_tokens"])
    print(f"Tokens: {usage['input_tokens']:,} in / {usage['output_tokens']:,} out"
          + (f" (~${cost:,.2f} at batch price)" if args.model in PRICES else ""))
    if usage["failed"] or usage["refused"]:
        print(f"Unanswered calls: {usage['failed']} failed after {MAX_ROUNDS} tries, "
              f"{usage['refused']} refused — those cells say 'unsure'. "
              "Re-run the same command to retry the failed ones.")
    for f in fields:
        counts = out[f["name"]].value_counts()
        if len(counts) <= 10:
            print(f"{f['name']}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    if usage["stopped"]:
        sys.exit(f"\nSTOPPED EARLY: {usage['stopped']}\n"
                 f"{usage['pending']:,} cells say 'pending'. Nothing finished is lost. Sort out the "
                 f"cause (top up credit / raise --budget), then re-run the same command to carry on.")


if __name__ == "__main__":
    main_cli()
