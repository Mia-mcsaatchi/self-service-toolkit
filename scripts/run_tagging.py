"""Tag a CSV through the toolkit backend from the command line.

Does what the web UI does in Steps 1–3 — upload rows, save the field config,
process in chunks, export — so a saved config can be re-run without clicking
through the page. Works against a local backend or the hosted one.

Usage:
    python scripts/run_tagging.py data.csv configs/hydra_mentions.json
    python scripts/run_tagging.py data.csv config.json --max-rows 20     # trial run
    python scripts/run_tagging.py data.csv config.json \\
        --backend https://self-service-toolkit.onrender.com --token "$TOOLKIT_TOKEN"

Local backend: start it with AUTH_DISABLED=true (see LOCAL_DEV.md) and no token
is needed. Hosted backend: pass your Supabase access token via --token or the
TOOLKIT_TOKEN env var.

Config file: {"base_prompt": optional, "fields": [...]} where each field is a
UI card — "name" for a single field, or "names" (list) for a cluster, which is
expanded into one field per name sharing the same prompt, exactly as the UI does.

Standard library only, so it runs from any Python 3.9+ without the venv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

PROCESS_CHUNK = 100  # rows per /api/process request, same as the UI


class Backend:
    def __init__(self, base_url: str, token: Optional[str]):
        self.base_url = base_url.rstrip("/")
        self.token = token

    def request(self, method: str, path: str, body: Any = None, timeout: float = 900) -> bytes:
        headers = {}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            try:
                detail = json.loads(detail).get("detail", detail)
            except Exception:
                pass
            sys.exit(f"{method} {path} failed ({e.code}): {detail}")
        except urllib.error.URLError as e:
            sys.exit(f"Can't reach the backend at {self.base_url}: {e.reason}")

    def json(self, method: str, path: str, body: Any = None) -> Dict[str, Any]:
        return json.loads(self.request(method, path, body))


def expand_fields(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Turn config cards into the API's flat field list (one entry per output column)."""
    fields: List[Dict[str, Any]] = []
    for card in config["fields"]:
        names = card.get("names") or [card["name"]]
        for name in names:
            fields.append({
                "name": name,
                "prompt": card.get("prompt", ""),
                "reads_from": card.get("reads_from", []),
                "field_type": "independent",
                "is_cluster": card.get("is_cluster", len(names) > 1),
                "mode": card.get("mode", "default"),
                "branches": card.get("branches", []),
            })
    return fields


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", type=Path, help="input CSV")
    ap.add_argument("config", type=Path, help="field config JSON")
    ap.add_argument("-o", "--output", type=Path, help="output CSV (default: <input>_tagged.csv)")
    ap.add_argument("--backend", default=os.environ.get("TOOLKIT_BACKEND", "http://127.0.0.1:8000"))
    ap.add_argument("--token", default=os.environ.get("TOOLKIT_TOKEN"), help="Supabase access token")
    ap.add_argument("--max-rows", type=int, default=0, help="only tag the first N rows (0 = all)")
    ap.add_argument("--max-concurrent", type=int, default=10)
    args = ap.parse_args()

    with args.csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        columns = next(reader)
        rows = [r for r in reader if any(cell.strip() for cell in r)]

    config = json.loads(args.config.read_text(encoding="utf-8"))
    fields = expand_fields(config)
    missing = {c for f in fields for c in f["reads_from"]} - set(columns) - {f["name"] for f in fields}
    if missing:
        sys.exit(f"Config reads from columns not in the CSV: {', '.join(sorted(missing))}")

    api = Backend(args.backend, args.token)
    api.json("GET", "/api/health")
    print(f"Backend {api.base_url} is up")

    up = api.json("POST", "/api/upload-data", {"columns": columns, "rows": rows})
    print(f"Uploaded {up.get('row_count', len(rows))} rows")

    cfg_body: Dict[str, Any] = {"fields": fields}
    if config.get("base_prompt"):
        cfg_body["base_prompt"] = config["base_prompt"]
    if config.get("trim"):
        cfg_body["trim"] = config["trim"]
    api.json("POST", "/api/upload-config", cfg_body)
    print(f"Config saved: {', '.join(f['name'] for f in fields)}")

    offset = 0
    while True:
        d = api.json("POST", "/api/process", {
            "max_rows": args.max_rows,
            "max_concurrent": args.max_concurrent,
            "offset": offset,
            "limit": PROCESS_CHUNK,
        })
        offset = d["processed"]
        print(f"  tagged {offset}/{d['total']}")
        if d["done"]:
            break

    out = args.output or args.csv.with_name(args.csv.stem + "_tagged.csv")
    out.write_bytes(api.request("GET", "/api/export/csv"))
    print(f"Wrote {out}")

    # Quick look at the categorical outputs so obvious problems show up immediately.
    with out.open(newline="", encoding="utf-8-sig") as f:
        tagged = list(csv.DictReader(f))
    for name in (f["name"] for f in fields):
        counts = Counter(r.get(name, "") for r in tagged)
        if len(counts) <= 10:
            print(f"{name}: " + ", ".join(f"{k or '(blank)'}={v}" for k, v in counts.most_common()))


if __name__ == "__main__":
    main()
