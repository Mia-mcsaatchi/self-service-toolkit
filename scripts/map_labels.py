"""Map free-text `context` labels onto a fixed category list, then rebuild the breakdown.

A tagging run leaves one short free-text label per mention ("Marvel's Hydra
organization", "HYDRA (Marvel organization)", ...). Tens of thousands of
wordings can't be matched by lookup, so this sends each *distinct* label —
100 per request through the Batch API — and asks Claude to place it in the
taxonomy: a broad category plus one of that category's specific associations,
"Unspecified", or a proposed new association ("NEW: …") for review. Labels are
short, so the whole job costs a few dollars however many mentions share them.

Outputs, next to the tagged file:
  <name>_categorised.csv   the tagged rows plus broad_category / specific_association
  <name>_breakdown.xlsx    the breakdown table, proposed new associations, label map

Usage:
    python scripts/map_labels.py data/hydra_relevance_final_tagged.csv configs/hydra_taxonomy.json --estimate
    python scripts/map_labels.py data/hydra_relevance_final_tagged.csv configs/hydra_taxonomy.json --budget 20

Resumable like run_batch.py: re-run the same command after any stop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import anthropic  # noqa: E402
import main  # noqa: E402  (loads .env)
from run_batch import (  # noqa: E402
    PRICES, THINKING_OFF_OK, _account_problem, batch_cost,
)

LABELS_PER_REQUEST = 100
MAX_ROUNDS = 3
UNCLEAR = "Unclear"
NOT_HYDRA = "Not a Hydra reference"
UNSPECIFIED = "Unspecified"
EXCLUDED_STATUSES = {"irrelevant", "unsure"}


def system_prompt(categories: Dict[str, List[str]], term: str) -> str:
    lines = [f"- {broad}: " + ("; ".join(specs) if specs else "(no specific associations yet)")
             for broad, specs in categories.items()]
    return (
        f"You assign short labels to a fixed taxonomy. Each label was written by an analyst to "
        f"describe what \"{term}\" refers to in one social-media or news mention.\n\n"
        "Taxonomy (broad category: specific associations):\n" + "\n".join(lines) + "\n\n"
        "For each label return:\n"
        "- broad: the broad category it belongs to. Use \"" + UNCLEAR + "\" when the label itself says the "
        "reference is unclear, garbled or unknown, or is too vague to place. Use \"" + NOT_HYDRA + "\" when "
        f"it shows there is no genuine {term} reference (e.g. ordinary words like hydration, text artefacts).\n"
        "- specific: copy one specific association from that broad category exactly as written above when "
        "one fits. Use \"" + UNSPECIFIED + "\" when the category is clear but no listed association fits and "
        "the label names no particular entity. Use \"NEW: <name>\" only when the label clearly names a "
        "distinct entity, product, work or usage missing from the list; give its established name in 2–6 "
        "words and name the same thing the same way every time. For Unclear and Not a Hydra reference, "
        "use \"" + UNSPECIFIED + "\".\n"
        "Judge from the label's meaning, not its wording; different wordings of the same thing get the same answer."
    )


def output_schema(categories: Dict[str, List[str]]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "i": {"type": "integer"},
                "broad": {"type": "string", "enum": list(categories) + [UNCLEAR, NOT_HYDRA]},
                "specific": {"type": "string"},
            },
            "required": ["i", "broad", "specific"],
            "additionalProperties": False,
        }}},
        "required": ["items"],
        "additionalProperties": False,
    }


def build_requests(labels: List[str], categories: Dict[str, List[str]], term: str, model: str
                   ) -> Dict[str, Tuple[Dict[str, Any], List[str]]]:
    system = [{"type": "text", "text": system_prompt(categories, term),
               "cache_control": {"type": "ephemeral"}}]  # shared by every request
    schema = output_schema(categories)
    requests = {}
    for start in range(0, len(labels), LABELS_PER_REQUEST):
        chunk = labels[start:start + LABELS_PER_REQUEST]
        body = "\n".join(f"{i}. {lab}" for i, lab in enumerate(chunk))
        params: Dict[str, Any] = {
            "model": model,
            "max_tokens": 8192,
            "system": system,
            "output_config": {"format": {"type": "json_schema", "schema": schema}, "effort": "low"},
            "messages": [{"role": "user", "content":
                          f"Labels (return one item per label, with its number as i):\n{body}"}],
        }
        if model in THINKING_OFF_OK:
            params["thinking"] = {"type": "disabled"}
        key = hashlib.sha1(("\n".join(chunk) + model).encode("utf-8")).hexdigest()
        requests[key] = (params, chunk)
    return requests


def normalise(broad: str, specific: str, categories: Dict[str, List[str]]) -> Tuple[str, str]:
    """Snap the answer onto the taxonomy's exact wording; anything else becomes NEW."""
    specific = " ".join((specific or "").split())
    if broad in (UNCLEAR, NOT_HYDRA) or not specific or specific.lower() == UNSPECIFIED.lower():
        return broad, UNSPECIFIED
    if specific.upper().startswith("NEW:"):
        return broad, "NEW: " + specific[4:].strip()
    for s in categories.get(broad, []):
        if s.lower() == specific.lower():
            return broad, s
    return broad, "NEW: " + specific


def run(labels: List[str], categories: Dict[str, List[str]], term: str, client, model: str,
        state_path: Path, poll_seconds: float = 60, budget: Optional[float] = None,
        ) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, Any]]:
    requests = build_requests(labels, categories, term, model)
    state: Dict[str, Any] = {"batches": []}
    if state_path.exists():
        saved = json.loads(state_path.read_text())
        if set(saved.get("keys", [])) == set(requests):
            state = saved
            print(f"Resuming: {len(state['batches'])} batch(es) already submitted")
    state["keys"] = sorted(requests)

    def save() -> None:
        state_path.write_text(json.dumps(state))

    mapping: Dict[str, Tuple[str, str]] = {}
    done: set = set()
    collected: set = set()
    sent_this_run: set = set()
    tries: Dict[str, int] = {}
    usage: Dict[str, Any] = {"input_tokens": 0, "output_tokens": 0, "stopped": None}

    while True:
        statuses = {b["id"]: client.messages.batches.retrieve(b["id"]) for b in state["batches"]}
        for b in state["batches"]:
            if statuses[b["id"]].processing_status != "ended" or b["id"] in collected:
                continue
            for r in client.messages.batches.results(b["id"]):
                if r.result.type == "errored":
                    err = r.result.error.error
                    if b["id"] in sent_this_run and _account_problem(err.message, err.type):
                        usage["stopped"] = f"{err.type}: {err.message}"
                    continue
                if r.result.type != "succeeded" or r.custom_id not in requests:
                    continue
                msg = r.result.message
                usage["input_tokens"] += msg.usage.input_tokens
                usage["output_tokens"] += msg.usage.output_tokens
                text = next((blk.text for blk in msg.content if blk.type == "text"), "")
                try:
                    items = json.loads(text)["items"]
                except (ValueError, KeyError, TypeError):
                    continue
                chunk = requests[r.custom_id][1]
                got = {}
                for it in items:
                    if isinstance(it.get("i"), int) and 0 <= it["i"] < len(chunk):
                        got[chunk[it["i"]]] = normalise(it.get("broad", UNCLEAR), it.get("specific", ""), categories)
                if len(got) == len(chunk):     # partial answers are resent whole
                    mapping.update(got)
                    done.add(r.custom_id)
            collected.add(b["id"])

        running = [b for b in state["batches"] if statuses[b["id"]].processing_status != "ended"]
        if usage["stopped"]:
            break
        in_flight = {k for b in running for k in b["keys"]}
        missing = [k for k in requests if k not in done and k not in in_flight and tries.get(k, 0) < MAX_ROUNDS]
        if not missing and not running:
            break
        if missing:
            spent = batch_cost(model, usage["input_tokens"], usage["output_tokens"])
            if budget is not None and spent >= budget:
                usage["stopped"] = f"budget of ${budget:,.2f} reached (${spent:,.2f} spent)"
                break
            try:
                batch = client.messages.batches.create(requests=[
                    {"custom_id": k, "params": requests[k][0]} for k in missing])
            except anthropic.RateLimitError:
                print("  Batch queue is full — will try again shortly")
            except anthropic.APIStatusError as e:
                if e.status_code in (401, 402, 403) or _account_problem(str(e.message)):
                    usage["stopped"] = str(e.message)
                    break
                raise
            else:
                state["batches"].append({"id": batch.id, "keys": missing})
                sent_this_run.add(batch.id)
                save()
                for k in missing:
                    tries[k] = tries.get(k, 0) + 1
                print(f"  sent batch {batch.id}: {len(missing):,} requests "
                      f"({sum(len(requests[k][1]) for k in missing):,} labels)")
        print(f"  {time.strftime('%H:%M')}  {len(mapping):,}/{len(labels):,} labels mapped, "
              f"~${batch_cost(model, usage['input_tokens'], usage['output_tokens']):,.2f} spent")
        time.sleep(poll_seconds)
    return mapping, usage


def apply_mapping(df: pd.DataFrame, mapping: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
    out = df.copy()
    pairs = out["context"].map(lambda c: mapping.get(c.strip(), ("Not mapped", UNSPECIFIED)))
    out["broad_category"] = [p[0] for p in pairs]
    out["specific_association"] = [p[1].removeprefix("NEW: ") for p in pairs]
    out["new_association"] = [p[1].startswith("NEW: ") for p in pairs]
    excluded = out["status"].isin(EXCLUDED_STATUSES) | (out["broad_category"] == NOT_HYDRA)
    out.loc[excluded, ["broad_category", "specific_association"]] = ["Excluded", UNSPECIFIED]
    unresolved = ~excluded & out["broad_category"].isin([UNCLEAR, "Not mapped"])
    out.loc[unresolved, "broad_category"] = "Unresolved"
    return out


def write_breakdown(cat: pd.DataFrame, path: Path) -> Dict[str, int]:
    total = len(cat)
    excluded = int((cat["broad_category"] == "Excluded").sum())
    population = total - excluded
    unresolved = int((cat["broad_category"] == "Unresolved").sum())
    rows: List[List[Any]] = [
        ["HYDRA association breakdown", None, None, None],
        ["Total collected mentions", None, total, None],
        ["Exclusions", None, excluded, None],
        ["Analytical population", None, population, None],
        ["Categorised mentions", None, population - unresolved, None],
        ["Unresolved mentions", None, unresolved, None],
        ["Broad category", "Specific association", "Mention count", "% of total analytical population"],
    ]
    body = cat[~cat["broad_category"].isin(["Excluded", "Unresolved"])]
    for broad, grp in sorted(body.groupby("broad_category"), key=lambda g: -len(g[1])):
        rows.append([broad, "Subtotal", len(grp), len(grp) / population])
        for spec, n in grp["specific_association"].value_counts().items():
            label = f"{spec} (new)" if grp.loc[grp["specific_association"] == spec, "new_association"].any() else spec
            rows.append([None, label, int(n), n / population])
    rows.append(["Unresolved", "All unresolved mentions", unresolved, unresolved / population])

    new = (cat[cat["new_association"] & ~cat["broad_category"].isin(["Excluded", "Unresolved"])]
           .groupby(["broad_category", "specific_association"])
           .agg(mentions=("context", "size"), example_labels=("context", lambda s: "; ".join(s.value_counts().index[:5])))
           .reset_index().sort_values("mentions", ascending=False))
    label_map = (cat.groupby(["context", "broad_category", "specific_association"]).size()
                 .rename("mentions").reset_index().sort_values("mentions", ascending=False))

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame(rows).to_excel(xw, sheet_name="HYDRA Associations", header=False, index=False)
        new.to_excel(xw, sheet_name="Proposed new associations", index=False)
        label_map.to_excel(xw, sheet_name="Label map", index=False)
        ws = xw.sheets["HYDRA Associations"]
        for row in ws.iter_rows(min_row=8, min_col=4, max_col=4):
            for cell in row:
                cell.number_format = "0.0%"
        for col, width in zip("ABCD", (34, 52, 15, 18)):
            ws.column_dimensions[col].width = width
    return {"total": total, "excluded": excluded, "population": population,
            "unresolved": unresolved, "new_rows": int(new["mentions"].sum()), "new_tags": len(new)}


def main_cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tagged", type=Path, help="output of run_batch.py (needs context and status columns)")
    ap.add_argument("taxonomy", type=Path)
    ap.add_argument("--term", default="Hydra")
    ap.add_argument("--model", default=main.TAGGING_MODEL)
    ap.add_argument("--estimate", action="store_true", help="print the expected cost and stop")
    ap.add_argument("--budget", type=float, help="stop sending once spend reaches this many US dollars")
    args = ap.parse_args()

    if not main.ANTHROPIC_API_KEY:
        sys.exit("ANTHROPIC_API_KEY is not set — add it to .env")
    df = pd.read_csv(args.tagged, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    for col in ("context", "status"):
        if col not in df.columns:
            sys.exit(f"{args.tagged} has no '{col}' column — pass the file run_batch.py wrote.")
    categories = json.loads(args.taxonomy.read_text(encoding="utf-8"))["categories"]
    labels = sorted({c.strip() for c in df["context"] if c.strip()})
    client = anthropic.Anthropic(api_key=main.ANTHROPIC_API_KEY, max_retries=5)
    requests = build_requests(labels, categories, args.term, args.model)
    print(f"{len(df):,} mentions, {len(labels):,} distinct labels -> {len(requests):,} requests")

    if args.estimate:
        params = dict(next(iter(requests.values()))[0])
        params.pop("max_tokens")
        per_request_in = client.messages.count_tokens(**params).input_tokens
        out_tokens = 25 * LABELS_PER_REQUEST  # ~25 tokens per mapped label
        cost = batch_cost(args.model, per_request_in * len(requests), out_tokens * len(requests))
        print(f"~{per_request_in:,} input tokens per request -> ~${cost:,.2f} at batch price "
              f"({args.model}; shared instructions may also be cached, which only lowers this)")
        return

    stem = args.tagged.with_suffix("").name.removesuffix("_tagged")
    state_path = args.tagged.with_name(stem + "_labelmap.batch.json")
    try:
        mapping, usage = run(labels, categories, args.term, client, args.model, state_path, budget=args.budget)
    except KeyboardInterrupt:
        sys.exit("\nStopped. Re-run the same command to collect sent batches and continue.")

    cat = apply_mapping(df, mapping)
    cat_path = args.tagged.with_name(stem + "_categorised.csv")
    cat.to_csv(cat_path, index=False, encoding="utf-8-sig")
    summary = write_breakdown(cat, args.tagged.with_name(stem + "_breakdown.xlsx"))
    print(f"Wrote {cat_path.name} and {stem}_breakdown.xlsx")
    print(f"Tokens: {usage['input_tokens']:,} in / {usage['output_tokens']:,} out "
          f"(~${batch_cost(args.model, usage['input_tokens'], usage['output_tokens']):,.2f} at batch price)")
    p = summary["population"]
    print(f"Analytical population {p:,} (excluded {summary['excluded']:,}): "
          f"categorised {p - summary['unresolved']:,} ({(p - summary['unresolved']) / p:.1%}), "
          f"unresolved {summary['unresolved']:,} ({summary['unresolved'] / p:.1%})")
    print(f"{summary['new_tags']} proposed new associations covering {summary['new_rows']:,} mentions "
          "— review them on the 'Proposed new associations' sheet")
    if usage["stopped"]:
        sys.exit(f"\nSTOPPED EARLY: {usage['stopped']}\nLabels not mapped yet count as unresolved. "
                 "Fix the cause, then re-run the same command to carry on.")


if __name__ == "__main__":
    main_cli()
