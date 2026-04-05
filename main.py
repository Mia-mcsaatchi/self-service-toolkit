from __future__ import annotations

import asyncio
import io
import json
import os
import re
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

app = FastAPI(title="Self-Service Toolkit API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://mia-mcsaatchi.github.io",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory state (single-session MVP)
# ---------------------------------------------------------------------------
_state: Dict[str, Any] = {
    "df": None,
    "config": None,
    "result_df": None,
}

BASE_PROMPT = (
    "You are a top-performing data analyst/consultant. "
    "Write clear, concise outputs optimized for analytics: each field should be a single cell-friendly string. "
    "If the source text is ambiguous or lacks evidence, reply with the token 'unsure'. "
    "Do not add extra commentary or headings beyond the requested fields."
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Condition(BaseModel):
    column: str
    operator: str = "="       # "=" or "!="
    values: List[str] = []    # match if row value is in this list
    prompt: str = ""

class Field(BaseModel):
    name: str
    prompt: str               # default prompt (used when no condition matches)
    reads_from: List[str] = []
    field_type: str = "independent"   # "independent" (dependent coming in phase 2)
    is_cluster: bool = False
    conditions: List[Condition] = []  # empty = run on all rows with default prompt

class FieldConfig(BaseModel):
    base_prompt: str = BASE_PROMPT
    fields: List[Field]

class ProcessRequest(BaseModel):
    max_rows: int = 0
    max_concurrent: int = 10

class RowData(BaseModel):
    columns: List[str]
    rows: List[List[Any]]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Data upload
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), sheet: Optional[str] = None):
    content = await file.read()
    name = (file.filename or "").lower()

    try:
        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(io.BytesIO(content), sheet_name=sheet or 0)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Upload .csv or .xlsx")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    _state["df"] = df
    _state["result_df"] = None

    return {
        "message": "File loaded",
        "columns": df.columns.tolist(),
        "row_count": len(df),
        "preview": df.head(5).fillna("").to_dict("records"),
    }


@app.post("/api/upload-data")
def upload_parsed_data(payload: RowData):
    df = pd.DataFrame(payload.rows, columns=payload.columns)
    _state["df"] = df
    _state["result_df"] = None
    return {
        "message": "Data loaded",
        "columns": df.columns.tolist(),
        "row_count": len(df),
        "preview": df.head(5).fillna("").to_dict("records"),
    }


# ---------------------------------------------------------------------------
# Field config
# ---------------------------------------------------------------------------

@app.post("/api/upload-config")
def upload_config(config: FieldConfig):
    _state["config"] = config.model_dump()
    return {"message": "Config saved", "field_count": len(config.fields)}


@app.get("/api/config")
def get_config():
    if not _state["config"]:
        raise HTTPException(status_code=404, detail="No config loaded yet")
    return _state["config"]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_row_prompt(base_prompt: str, fields: List[Dict], row: pd.Series) -> str:
    field_names = {(f.get("name") or "").strip() for f in fields}

    all_cols: List[str] = []
    for f in fields:
        for c in (f.get("reads_from") or []):
            if c not in all_cols and c not in field_names:
                all_cols.append(c)

    context = "\n".join(
        f"- {col}: {'' if pd.isna(row.get(col)) else str(row.get(col, ''))}"
        for col in all_cols
        if col in row.index
    ) or "(no source columns)"

    instruction_lines = []
    for f in fields:
        name = (f.get("name") or "").strip()
        if not name:
            continue
        prompt_text = (f.get("prompt") or "").strip()
        deps = [c for c in (f.get("reads_from") or []) if c in field_names and c != name]
        dep_note = f" (based on {', '.join(deps)})" if deps else ""
        instruction_lines.append(f'  "{name}": {prompt_text}{dep_note}')

    keys = ", ".join(f'"{f["name"]}"' for f in fields if (f.get("name") or "").strip())

    return (
        f"{base_prompt}\n\n"
        f"Row data:\n{context}\n\n"
        f"Return a JSON object with EXACTLY these keys: {{{keys}}}\n"
        f"Fill the fields IN ORDER — later fields may reference earlier ones.\n"
        f"Field instructions:\n" + "\n".join(instruction_lines) + "\n\n"
        "Rules:\n"
        "  • Each value must be plain text (no nested objects or markdown).\n"
        "  • If a field is ambiguous or missing, use 'unsure'.\n"
        "  • Return ONLY the JSON object, nothing else.\n"
    )


# ---------------------------------------------------------------------------
# Async OpenAI caller
# ---------------------------------------------------------------------------

async def _call_openai(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    prompt: str,
    retries: int = 2,
) -> str:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
    }
    backoff = 1.0
    for attempt in range(retries + 1):
        try:
            async with semaphore:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return (data["choices"][0]["message"]["content"] or "{}").strip()
                    if attempt < retries:
                        await asyncio.sleep(backoff)
                        backoff *= 2
                    else:
                        return "{}"
        except Exception:
            if attempt < retries:
                await asyncio.sleep(backoff)
                backoff *= 2
            else:
                return "{}"
    return "{}"


# ---------------------------------------------------------------------------
# Condition resolver
# ---------------------------------------------------------------------------

def _resolve_prompt(field: Dict[str, Any], row: pd.Series) -> Optional[str]:
    """
    Return the prompt to use for this field+row, or None if all conditions
    are defined but none match (caller writes 'n/a' and skips the API call).

    Rules:
    - No conditions defined → always use field default prompt
    - Conditions defined → evaluate top-to-bottom, first match wins
      - operator "=" : row value must be in condition values list
      - operator "!=" : row value must NOT be in condition values list
    - No match → return None  (caller writes 'n/a')
    """
    conditions = field.get("conditions") or []
    default_prompt = (field.get("prompt") or "").strip()

    if not conditions:
        return default_prompt if default_prompt else None

    for cond in conditions:
        col = cond.get("column", "")
        op = cond.get("operator", "=")
        values = [str(v).strip() for v in (cond.get("values") or [])]
        cond_prompt = (cond.get("prompt") or "").strip()

        if col not in row.index:
            continue

        row_val = str(row[col]).strip() if not pd.isna(row.get(col)) else ""

        matched = (row_val in values) if op == "=" else (row_val not in values)
        if matched and cond_prompt:
            return cond_prompt

    # No condition matched — fall back to default prompt if set
    return default_prompt if default_prompt else None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def _run_pipeline(
    df: pd.DataFrame,
    config: Dict[str, Any],
    max_rows: int,
    max_concurrent: int,
) -> pd.DataFrame:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set on the server.")

    fields = [f for f in config.get("fields", []) if (f.get("name") or "").strip()]
    if not fields:
        raise ValueError("No fields defined in config")

    df = df.copy()
    for f in fields:
        if f["name"] not in df.columns:
            df[f["name"]] = ""

    n = min(max_rows, len(df)) if max_rows > 0 else len(df)
    base_prompt = config.get("base_prompt", BASE_PROMPT)

    semaphore = asyncio.Semaphore(max_concurrent)
    timeout = aiohttp.ClientTimeout(total=45)

    async with aiohttp.ClientSession(timeout=timeout) as session:

        # Build one task per row — each task resolves conditions and calls API
        async def process_row(i: int) -> Dict[str, str]:
            row = df.iloc[i]
            results: Dict[str, str] = {}

            # Group fields that share the same resolved prompt and reads_from
            # into a single API call (cluster behaviour).
            # Fields with no matching condition write "n/a" immediately.
            call_groups: Dict[str, Dict[str, Any]] = {}

            for f in fields:
                fname = (f.get("name") or "").strip()
                if not fname:
                    continue

                resolved_prompt = _resolve_prompt(f, row)

                if resolved_prompt is None:
                    # No condition matched and no default → n/a, no API call
                    results[fname] = "n/a"
                    continue

                # Group key: same prompt + same reads_from = same API call
                group_key = resolved_prompt + "|" + ",".join(sorted(f.get("reads_from") or []))
                if group_key not in call_groups:
                    call_groups[group_key] = {
                        "prompt": resolved_prompt,
                        "reads_from": f.get("reads_from") or [],
                        "field_names": [],
                    }
                call_groups[group_key]["field_names"].append(fname)

            # Fire one API call per group
            for group in call_groups.values():
                group_fields = [
                    {"name": fn, "prompt": group["prompt"], "reads_from": group["reads_from"]}
                    for fn in group["field_names"]
                ]
                built_prompt = _build_row_prompt(base_prompt, group_fields, row)
                raw = await _call_openai(session, semaphore, built_prompt)
                try:
                    parsed = json.loads(raw)
                except Exception:
                    parsed = {}
                for fn in group["field_names"]:
                    results[fn] = parsed.get(fn, "unsure")

            return results

        tasks = [process_row(i) for i in range(n)]
        all_results = await asyncio.gather(*tasks)

    # Write results back to dataframe
    for i, row_results in enumerate(all_results):
        for fname, value in row_results.items():
            if fname in df.columns:
                df.iat[i, df.columns.get_loc(fname)] = value

    return df


@app.post("/api/process")
async def process(body: ProcessRequest):
    df = _state.get("df")
    config = _state.get("config")

    if df is None:
        raise HTTPException(status_code=400, detail="No data loaded — upload a file first")
    if not config:
        raise HTTPException(status_code=400, detail="No config loaded — configure fields first")

    try:
        result = await _run_pipeline(df, config, body.max_rows, body.max_concurrent)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    _state["result_df"] = result

    return {
        "message": "Processing complete",
        "row_count": len(result),
        "columns": result.columns.tolist(),
        "preview": result.head(10).fillna("").to_dict("records"),
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

@app.get("/api/export/csv")
def export_csv():
    df = _state.get("result_df") or _state.get("df")
    if df is None:
        raise HTTPException(status_code=400, detail="No data to export")

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=results.csv"},
    )


@app.get("/api/export/xlsx")
def export_xlsx():
    df = _state.get("result_df") or _state.get("df")
    if df is None:
        raise HTTPException(status_code=400, detail="No data to export")

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=results.xlsx"},
    )
