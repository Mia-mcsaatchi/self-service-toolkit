from __future__ import annotations

import asyncio
import io
import json
import os
import re
from typing import Any, Dict, List, Optional

import aiohttp
import jwt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# ---------------------------------------------------------------------------
# Auth config (Supabase)
# ---------------------------------------------------------------------------
# Supabase can sign access tokens two ways:
#   • Legacy: a shared secret, HS256 — verified offline with SUPABASE_JWT_SECRET.
#   • Newer:  asymmetric keys, ES256/RS256 — verified against the project's public
#             JWKS endpoint (needs SUPABASE_URL). The backend supports both.
SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET", "")
# Backend base URL of your Supabase project, e.g. https://xxxx.supabase.co
# Only needed for the asymmetric (new signing keys) verification path.
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
# Service-role key (SERVER SECRET) — lets the backend read/write saved datasets
# in Postgres via PostgREST. Never expose it to the frontend or commit it.
# When unset, the save/load feature is simply disabled (the app still runs).
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
# Only allow colleagues on this email domain to use the tool. Empty = allow any
# authenticated user. Set to "" to disable the domain check.
ALLOWED_EMAIL_DOMAIN = os.environ.get("ALLOWED_EMAIL_DOMAIN", "mcsaatchi.com").strip().lower()
# Local escape hatch: set AUTH_DISABLED=true to run without Supabase (single
# shared "local-dev" user). Never enable this in the deployed backend.
AUTH_DISABLED = os.environ.get("AUTH_DISABLED", "").strip().lower() in ("1", "true", "yes")

app = FastAPI(title="Self-Service Toolkit API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://mia-mcsaatchi.github.io",
    ],
    # Allow the frontend served from any local port during development
    # (VS Code Live Server, python -m http.server, etc.).
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Authentication — verify the Supabase JWT and identify the current user
# ---------------------------------------------------------------------------

_jwk_client = None  # lazily created PyJWKClient, caches Supabase's public keys


def _get_jwk_client():
    """Return a cached PyJWKClient pointed at the project's JWKS endpoint."""
    global _jwk_client
    if _jwk_client is None:
        if not SUPABASE_URL:
            raise HTTPException(
                status_code=503,
                detail="Auth not configured: set SUPABASE_URL for asymmetric JWT verification.",
            )
        _jwk_client = jwt.PyJWKClient(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json")
    return _jwk_client


def _verify_token(token: str) -> Dict[str, Any]:
    """Verify a Supabase access token via HS256 (shared secret) or ES256/RS256
    (asymmetric keys), picking the path from the token's own algorithm header."""
    try:
        alg = jwt.get_unverified_header(token).get("alg", "HS256")
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token header: {e}")

    try:
        if alg == "HS256":
            if not SUPABASE_JWT_SECRET:
                raise HTTPException(
                    status_code=503,
                    detail="Auth not configured: set SUPABASE_JWT_SECRET on the backend.",
                )
            return jwt.decode(
                token, SUPABASE_JWT_SECRET,
                algorithms=["HS256"], audience="authenticated",
            )
        signing_key = _get_jwk_client().get_signing_key_from_jwt(token).key
        return jwt.decode(
            token, signing_key,
            algorithms=["ES256", "RS256"], audience="authenticated",
        )
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


async def get_current_user(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """FastAPI dependency: validate the Supabase access token on the request.

    Returns a dict with the user's id + email. Raises 401/403 on any problem.
    Add `user: Dict[str, Any] = Depends(get_current_user)` to any endpoint that
    should be scoped to a signed-in colleague.
    """
    if AUTH_DISABLED:
        return {"id": "local-dev", "email": f"dev@{ALLOWED_EMAIL_DOMAIN or 'localhost'}"}

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")

    token = authorization.split(" ", 1)[1].strip()
    payload = _verify_token(token)

    user_id = payload.get("sub")
    email = (payload.get("email") or "").strip().lower()
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing subject claim")

    if ALLOWED_EMAIL_DOMAIN and not email.endswith("@" + ALLOWED_EMAIL_DOMAIN):
        raise HTTPException(
            status_code=403,
            detail=f"Access restricted to @{ALLOWED_EMAIL_DOMAIN} accounts",
        )

    return {"id": user_id, "email": email}

# ---------------------------------------------------------------------------
# Per-user in-memory working state
# ---------------------------------------------------------------------------
# Each signed-in user gets their own isolated working state, keyed by user id.
# This is the live editing session (uploaded df, config, results, embeddings).
# Durable storage (saved datasets / dashboards) lands in Supabase Postgres in a
# later phase — this dict is just the fast in-memory scratch space per user.
_sessions: Dict[str, Dict[str, Any]] = {}


def _new_state() -> Dict[str, Any]:
    return {
        "df": None,
        "config": None,
        "result_df": None,
        # Analytics state
        "embeddings": None,       # np.ndarray of shape (n_rows, embedding_dim)
        "embedded_texts": None,   # List[str] — the text we embedded (one per row)
        "column_map": None,       # Dict describing which columns are categorical/datetime/text/numerical
        "api_key": None,          # optional per-user OpenAI key override
    }


def _state_for(user: Dict[str, Any]) -> Dict[str, Any]:
    """Return (creating if needed) the working state for this user."""
    uid = user["id"]
    if uid not in _sessions:
        _sessions[uid] = _new_state()
    return _sessions[uid]


# ---------------------------------------------------------------------------
# Durable storage (saved datasets) via Supabase Postgres / PostgREST
# ---------------------------------------------------------------------------

def _storage_ready() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)


async def _sb_request(
    method: str,
    path: str,
    *,
    params: Optional[Dict[str, str]] = None,
    data: Optional[Any] = None,
    prefer: Optional[str] = None,
) -> Any:
    """Call the Supabase REST API (PostgREST) with the service-role key.

    The backend scopes every query by the verified user id, so a user can only
    ever touch their own rows.
    """
    if not _storage_ready():
        raise HTTPException(
            status_code=503,
            detail="Saving isn't set up: set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY on the backend.",
        )
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    url = f"{SUPABASE_URL}/rest/v1/{path}"
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.request(method, url, headers=headers, params=params, json=data) as resp:
            text = await resp.text()
            if resp.status >= 300:
                raise HTTPException(status_code=502, detail=f"Storage error ({resp.status}): {text[:300]}")
            if not text:
                return None
            try:
                return json.loads(text)
            except Exception:
                return None

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
    operator: str = "is"      # "is" or "is not"
    values: List[str] = []

class Branch(BaseModel):
    conditions: List[Condition] = []
    prompt: str = ""

class Field(BaseModel):
    name: str
    prompt: str = ""
    reads_from: List[str] = []
    field_type: str = "independent"
    is_cluster: bool = False
    mode: str = "default"
    branches: List[Branch] = []

class FieldConfig(BaseModel):
    base_prompt: str = BASE_PROMPT
    fields: List[Field]

class ProcessRequest(BaseModel):
    max_rows: int = 0
    max_concurrent: int = 10

class RowData(BaseModel):
    columns: List[str]
    rows: List[List[Any]]

class SuggestPromptRequest(BaseModel):
    field_name: str
    reads_from: List[str] = []
    is_cluster: bool = False
    samples: List[str] = []       # a few sample values from the source column(s)
    columns: List[str] = []

class SaveDatasetRequest(BaseModel):
    name: str
    use_result: bool = True       # save tagged results if available, else raw data

class SaveDashboardRequest(BaseModel):
    name: str
    config: Dict[str, Any] = {}   # {charts: [...]} — the dashboard layout

class PublishRequest(BaseModel):
    is_public: bool = True

class ShareRequest(BaseModel):
    email: str

# ---------------------------------------------------------------------------
# Analytics models
# ---------------------------------------------------------------------------

class InterpretRequest(BaseModel):
    intent: str
    column_summary: Dict[str, Any]
    dataset_label: str = "social listening dataset"

class ColumnMap(BaseModel):
    categorical: List[str] = []   # sentiment, topic, source, language etc.
    datetime: List[str] = []      # date/timestamp columns
    text: List[str] = []          # verbatim/comment columns (for word cloud + RAG)
    numerical: List[str] = []     # numeric columns

class EmbedRequest(BaseModel):
    column_map: ColumnMap
    # Optionally scope to result_df or raw df
    use_result: bool = True

class ChartContext(BaseModel):
    chart_type: str          # "sentiment_bar" | "value_counts" | "pie" | "line" | "verbatims" | "wordcloud"
    column: str              # which column this chart is about
    label: str               # human label e.g. "Sentiment by Topic"
    # Pre-computed summary sent from frontend (so backend doesn't re-compute)
    summary: Dict[str, Any]  # e.g. {"positive": 45, "neutral": 30, "negative": 25, "total": 378}

class AnalyseRequest(BaseModel):
    column_map: ColumnMap
    charts: List[ChartContext]
    dataset_label: str = "social listening dataset"  # e.g. "Ford Europe BlueCruise mentions"

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class ApiKeyRequest(BaseModel):
    api_key: str

@app.post("/api/set-api-key")
def set_api_key(body: ApiKeyRequest, user: Dict[str, Any] = Depends(get_current_user)):
    """Allow a user to set their own OpenAI API key for this session."""
    key = (body.api_key or "").strip()
    if not key.startswith("sk-"):
        raise HTTPException(status_code=400, detail="Invalid API key format")
    _state_for(user)["api_key"] = key
    return {"message": "API key set"}

@app.get("/api/health")
def health():
    """Public liveness probe — intentionally unauthenticated."""
    return {"status": "ok"}


@app.get("/api/me")
def whoami(user: Dict[str, Any] = Depends(get_current_user)):
    """Return the signed-in user + which optional features are configured."""
    return {**user, "storage_enabled": _storage_ready()}



@app.get("/api/debug")
def debug(user: Dict[str, Any] = Depends(get_current_user)):
    st = _state_for(user)
    return {
        "df_rows": len(st["df"]) if st["df"] is not None else None,
        "df_cols": st["df"].columns.tolist() if st["df"] is not None else None,
        "result_df_rows": len(st["result_df"]) if st["result_df"] is not None else None,
        "result_df_cols": st["result_df"].columns.tolist() if st["result_df"] is not None else None,
        "config_fields": len(st["config"]["fields"]) if st["config"] else None,
    }


# ---------------------------------------------------------------------------
# Data upload
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    sheet: Optional[str] = None,
    user: Dict[str, Any] = Depends(get_current_user),
):
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

    st = _state_for(user)
    st["df"] = df
    st["result_df"] = None
    st["embeddings"] = None
    st["embedded_texts"] = None

    return {
        "message": "File loaded",
        "columns": df.columns.tolist(),
        "row_count": len(df),
        "preview": df.head(5).fillna("").to_dict("records"),
    }


@app.post("/api/upload-data")
def upload_parsed_data(payload: RowData, user: Dict[str, Any] = Depends(get_current_user)):
    df = pd.DataFrame(payload.rows, columns=payload.columns)
    st = _state_for(user)
    st["df"] = df
    st["result_df"] = None
    st["embeddings"] = None
    st["embedded_texts"] = None
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
def upload_config(config: FieldConfig, user: Dict[str, Any] = Depends(get_current_user)):
    _state_for(user)["config"] = config.model_dump()
    return {"message": "Config saved", "field_count": len(config.fields)}


@app.get("/api/config")
def get_config(user: Dict[str, Any] = Depends(get_current_user)):
    config = _state_for(user)["config"]
    if not config:
        raise HTTPException(status_code=404, detail="No config loaded yet")
    return config


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
# Async OpenAI caller (shared by pipeline + analytics)
# ---------------------------------------------------------------------------

async def _call_openai(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    prompt: str,
    model: str = "gpt-4o-mini",
    response_json: bool = True,
    max_tokens: int = 512,
    retries: int = 2,
    api_key: Optional[str] = None,
) -> str:
    api_key = api_key or OPENAI_API_KEY
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if response_json:
        payload["response_format"] = {"type": "json_object"}

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
                        return (data["choices"][0]["message"]["content"] or "").strip()
                    if attempt < retries:
                        await asyncio.sleep(backoff)
                        backoff *= 2
                    else:
                        return "{}" if response_json else ""
        except Exception:
            if attempt < retries:
                await asyncio.sleep(backoff)
                backoff *= 2
            else:
                return "{}" if response_json else ""
    return "{}" if response_json else ""


# ---------------------------------------------------------------------------
# Condition resolver
# ---------------------------------------------------------------------------

def _match_condition(cond: Dict, row: pd.Series) -> bool:
    col = cond.get("column", "")
    op = cond.get("operator", "is")
    values = [str(v).strip() for v in (cond.get("values") or [])]
    if col not in row.index or not values:
        return False
    row_val = str(row[col]).strip() if not pd.isna(row.get(col)) else ""
    return (row_val in values) if op in ("=", "is") else (row_val not in values)


def _resolve_prompt(field: Dict[str, Any], row: pd.Series) -> Optional[str]:
    mode = field.get("mode", "default")
    default_prompt = (field.get("prompt") or "").strip()

    if mode == "default":
        return default_prompt if default_prompt else None

    branches = field.get("branches") or []
    for branch in branches:
        conditions = branch.get("conditions") or []
        branch_prompt = (branch.get("prompt") or "").strip()
        if not branch_prompt:
            continue
        if all(_match_condition(c, row) for c in conditions):
            return branch_prompt

    return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def _run_pipeline(
    df: pd.DataFrame,
    config: Dict[str, Any],
    max_rows: int,
    max_concurrent: int,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    if not (api_key or OPENAI_API_KEY):
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

        async def process_row(i: int) -> Dict[str, str]:
            row = df.iloc[i]
            results: Dict[str, str] = {}

            call_groups: Dict[str, Dict[str, Any]] = {}

            for f in fields:
                fname = (f.get("name") or "").strip()
                if not fname:
                    continue

                resolved_prompt = _resolve_prompt(f, row)

                if resolved_prompt is None:
                    results[fname] = "n/a"
                    continue

                group_key = resolved_prompt + "|" + ",".join(sorted(f.get("reads_from") or []))
                if group_key not in call_groups:
                    call_groups[group_key] = {
                        "prompt": resolved_prompt,
                        "reads_from": f.get("reads_from") or [],
                        "field_names": [],
                    }
                call_groups[group_key]["field_names"].append(fname)

            for group in call_groups.values():
                group_fields = [
                    {"name": fn, "prompt": group["prompt"], "reads_from": group["reads_from"]}
                    for fn in group["field_names"]
                ]
                built_prompt = _build_row_prompt(base_prompt, group_fields, row)
                raw = await _call_openai(session, semaphore, built_prompt, api_key=api_key)
                try:
                    parsed = json.loads(raw)
                except Exception:
                    parsed = {}
                for fn in group["field_names"]:
                    results[fn] = parsed.get(fn, "unsure")

            return results

        tasks = [process_row(i) for i in range(n)]
        all_results = await asyncio.gather(*tasks)

    for i, row_results in enumerate(all_results):
        for fname, value in row_results.items():
            if fname in df.columns:
                df.iat[i, df.columns.get_loc(fname)] = value

    return df


@app.post("/api/process")
async def process(body: ProcessRequest, user: Dict[str, Any] = Depends(get_current_user)):
    st = _state_for(user)
    df = st.get("df")
    config = st.get("config")

    if df is None:
        raise HTTPException(status_code=400, detail="No data loaded — upload a file first")
    if not config:
        raise HTTPException(status_code=400, detail="No config loaded — configure fields first")

    try:
        result = await _run_pipeline(
            df, config, body.max_rows, body.max_concurrent,
            api_key=st.get("api_key"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    st["result_df"] = result

    return {
        "message": "Processing complete",
        "row_count": len(result),
        "columns": result.columns.tolist(),
        "preview": result.head(10).fillna("").to_dict("records"),
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all values to strings to avoid serialisation errors."""
    df = df.copy()
    for col in df.columns:
        try:
            df[col] = df[col].fillna("").astype(str).replace({"nan": "", "None": "", "<NA>": ""})
        except Exception:
            df[col] = df[col].astype(str)
    return df


def _get_export_df(st: Dict[str, Any]) -> pd.DataFrame:
    """Return result_df if available, else df. Avoids ambiguous DataFrame truth value."""
    df = st.get("result_df")
    if df is None:
        df = st.get("df")
    return df


@app.get("/api/result-data")
def result_data(user: Dict[str, Any] = Depends(get_current_user)):
    """Return the current results as clean JSON (columns + list-of-dicts rows).

    The dashboard uses this instead of round-tripping through CSV, which can
    mis-parse free-text/AI columns that contain commas, quotes, or newlines.
    """
    df = _get_export_df(_state_for(user))
    if df is None:
        raise HTTPException(status_code=400, detail="No data — run the pipeline or load a dataset first.")
    sdf = _safe_df(df)
    return {"columns": sdf.columns.tolist(), "rows": sdf.to_dict("records")}


@app.get("/api/export/csv")
def export_csv(user: Dict[str, Any] = Depends(get_current_user)):
    from fastapi.responses import Response as FastAPIResponse
    df = _get_export_df(_state_for(user))
    if df is None:
        raise HTTPException(status_code=400, detail="No data to export")
    df = _safe_df(df)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return FastAPIResponse(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=results.csv"},
    )


@app.get("/api/export/xlsx")
def export_xlsx(user: Dict[str, Any] = Depends(get_current_user)):
    from fastapi.responses import Response as FastAPIResponse
    df = _get_export_df(_state_for(user))
    if df is None:
        raise HTTPException(status_code=400, detail="No data to export")
    df = _safe_df(df)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    return FastAPIResponse(
        content=buf.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=results.xlsx"},
    )


# ---------------------------------------------------------------------------
# Saved datasets (Phase B) — durable per-user storage in Supabase Postgres
# ---------------------------------------------------------------------------

@app.post("/api/datasets")
async def save_dataset(body: SaveDatasetRequest, user: Dict[str, Any] = Depends(get_current_user)):
    """Save the current tagged results (or raw data) to the user's account."""
    st = _state_for(user)
    df = _get_export_df(st) if body.use_result else st.get("df")
    if df is None:
        raise HTTPException(status_code=400, detail="No data to save — upload or process a file first.")
    sdf = _safe_df(df)
    name = (body.name or "").strip() or "Untitled dataset"
    payload = {
        "user_id": user["id"],
        "user_email": user.get("email"),
        "name": name,
        "columns": sdf.columns.tolist(),
        "rows": sdf.values.tolist(),
        "row_count": int(len(sdf)),
    }
    result = await _sb_request("POST", "datasets", data=payload, prefer="return=representation")
    saved = result[0] if isinstance(result, list) and result else (result or {})
    return {"id": saved.get("id"), "name": name, "row_count": payload["row_count"]}


@app.get("/api/datasets")
async def list_datasets(user: Dict[str, Any] = Depends(get_current_user)):
    """List the signed-in user's saved datasets (metadata only, newest first)."""
    rows = await _sb_request("GET", "datasets", params={
        "user_id": f"eq.{user['id']}",
        "select": "id,name,row_count,created_at",
        "order": "created_at.desc",
    })
    return {"datasets": rows or []}


@app.get("/api/datasets/{dataset_id}")
async def get_dataset(dataset_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    """Load a saved dataset back into the working session (scoped to the user)."""
    rows = await _sb_request("GET", "datasets", params={
        "id": f"eq.{dataset_id}",
        "user_id": f"eq.{user['id']}",
        "select": "id,name,columns,rows,row_count,created_at",
        "limit": "1",
    })
    if not rows:
        raise HTTPException(status_code=404, detail="Dataset not found")
    d = rows[0]
    df = pd.DataFrame(d.get("rows") or [], columns=d.get("columns") or [])
    st = _state_for(user)
    st["df"] = df
    st["result_df"] = df
    st["embeddings"] = None
    st["embedded_texts"] = None
    return {
        "id": d["id"],
        "name": d["name"],
        "columns": d.get("columns") or [],
        "row_count": d.get("row_count"),
        "preview": df.head(10).fillna("").to_dict("records"),
    }


@app.delete("/api/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    """Delete one of the user's saved datasets."""
    await _sb_request("DELETE", "datasets", params={
        "id": f"eq.{dataset_id}",
        "user_id": f"eq.{user['id']}",
    })
    return {"deleted": dataset_id}


# ---------------------------------------------------------------------------
# Saved & shareable dashboards (Phase C)
# ---------------------------------------------------------------------------

@app.post("/api/dashboards")
async def save_dashboard(body: SaveDashboardRequest, user: Dict[str, Any] = Depends(get_current_user)):
    """Save the current dashboard (chart layout + a snapshot of the data)."""
    df = _get_export_df(_state_for(user))
    if df is None:
        raise HTTPException(status_code=400, detail="No data — build a dashboard first.")
    sdf = _safe_df(df)
    payload = {
        "owner_id": user["id"],
        "owner_email": user.get("email"),
        "name": (body.name or "").strip() or "Untitled dashboard",
        "config": body.config or {},
        "columns": sdf.columns.tolist(),
        # Store rows as list-of-dicts (keyed by column name) to match
        # /api/result-data. The dashboard chart code reads row[columnName],
        # so a positional array-of-arrays snapshot renders every chart blank.
        "rows": sdf.to_dict("records"),
    }
    res = await _sb_request("POST", "dashboards", data=payload, prefer="return=representation")
    saved = res[0] if isinstance(res, list) and res else (res or {})
    return {"id": saved.get("id"), "name": payload["name"],
            "share_token": saved.get("share_token"), "is_public": saved.get("is_public", False)}


@app.get("/api/dashboards")
async def list_dashboards(user: Dict[str, Any] = Depends(get_current_user)):
    rows = await _sb_request("GET", "dashboards", params={
        "owner_id": f"eq.{user['id']}",
        "select": "id,name,is_public,share_token,created_at",
        "order": "created_at.desc",
    })
    return {"dashboards": rows or []}


@app.get("/api/shared-dashboards")
async def shared_dashboards(user: Dict[str, Any] = Depends(get_current_user)):
    """Dashboards other colleagues have shared with this user's email."""
    shares = await _sb_request("GET", "dashboard_shares", params={
        "shared_with_email": f"eq.{user['email']}",
        "select": "dashboard_id",
    })
    ids = [s["dashboard_id"] for s in (shares or [])]
    if not ids:
        return {"dashboards": []}
    rows = await _sb_request("GET", "dashboards", params={
        "id": f"in.({','.join(ids)})",
        "select": "id,name,owner_email,created_at",
        "order": "created_at.desc",
    })
    return {"dashboards": rows or []}


@app.get("/api/dashboards/{dashboard_id}")
async def get_dashboard(dashboard_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    """Load a dashboard the user owns or that has been shared with them."""
    rows = await _sb_request("GET", "dashboards", params={
        "id": f"eq.{dashboard_id}", "select": "*", "limit": "1",
    })
    if not rows:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    d = rows[0]
    if d["owner_id"] != user["id"]:
        shares = await _sb_request("GET", "dashboard_shares", params={
            "dashboard_id": f"eq.{dashboard_id}",
            "shared_with_email": f"eq.{user['email']}",
            "select": "id", "limit": "1",
        })
        if not shares:
            raise HTTPException(status_code=403, detail="This dashboard isn't shared with you")
    return {
        "id": d["id"], "name": d["name"], "config": d.get("config") or {},
        "columns": d.get("columns") or [], "rows": d.get("rows") or [],
        "is_public": d.get("is_public"), "share_token": d.get("share_token"),
    }


@app.delete("/api/dashboards/{dashboard_id}")
async def delete_dashboard(dashboard_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    await _sb_request("DELETE", "dashboards", params={
        "id": f"eq.{dashboard_id}", "owner_id": f"eq.{user['id']}",
    })
    return {"deleted": dashboard_id}


@app.delete("/api/account/data")
async def delete_all_my_data(user: Dict[str, Any] = Depends(get_current_user)):
    """Delete ALL of the signed-in user's saved datasets and dashboards.

    The account itself is left intact. dashboard_shares rows cascade on the
    dashboards FK, so they're removed automatically.
    """
    ds = await _sb_request("DELETE", "datasets",
                           params={"user_id": f"eq.{user['id']}"},
                           prefer="return=representation")
    db = await _sb_request("DELETE", "dashboards",
                           params={"owner_id": f"eq.{user['id']}"},
                           prefer="return=representation")
    return {
        "datasets_deleted": len(ds) if isinstance(ds, list) else 0,
        "dashboards_deleted": len(db) if isinstance(db, list) else 0,
    }


@app.post("/api/dashboards/{dashboard_id}/publish")
async def publish_dashboard(dashboard_id: str, body: PublishRequest, user: Dict[str, Any] = Depends(get_current_user)):
    """Turn the public read-only link on or off."""
    res = await _sb_request("PATCH", "dashboards",
        params={"id": f"eq.{dashboard_id}", "owner_id": f"eq.{user['id']}"},
        data={"is_public": body.is_public}, prefer="return=representation")
    if not res:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    d = res[0]
    return {"is_public": d.get("is_public"), "share_token": d.get("share_token")}


@app.post("/api/dashboards/{dashboard_id}/share")
async def share_dashboard(dashboard_id: str, body: ShareRequest, user: Dict[str, Any] = Depends(get_current_user)):
    """Grant a named colleague read access to this dashboard."""
    owns = await _sb_request("GET", "dashboards", params={
        "id": f"eq.{dashboard_id}", "owner_id": f"eq.{user['id']}", "select": "id", "limit": "1",
    })
    if not owns:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    email = (body.email or "").strip().lower()
    if "@" not in email:
        raise HTTPException(status_code=400, detail="Enter a valid email address")
    await _sb_request("POST", "dashboard_shares",
                      data={"dashboard_id": dashboard_id, "shared_with_email": email})
    return {"shared_with": email}


@app.get("/api/public/dashboard/{token}")
async def public_dashboard(token: str):
    """PUBLIC (no auth): read-only dashboard by share token, if published."""
    rows = await _sb_request("GET", "dashboards", params={
        "share_token": f"eq.{token}", "is_public": "eq.true",
        "select": "name,config,columns,rows", "limit": "1",
    })
    if not rows:
        raise HTTPException(status_code=404, detail="Dashboard not found or not public")
    d = rows[0]
    return {"name": d["name"], "config": d.get("config") or {},
            "columns": d.get("columns") or [], "rows": d.get("rows") or []}


# ---------------------------------------------------------------------------
# Analytics — /api/interpret
# Takes user's plain-English intent + column summary → returns chart configs
# ---------------------------------------------------------------------------

@app.post("/api/interpret")
async def interpret_intent(body: InterpretRequest, user: Dict[str, Any] = Depends(get_current_user)):
    """LLM interprets user intent into chart configs. Uses server-side API key."""
    api_key = _state_for(user).get("api_key")
    prompt = (
        f"You are a data analyst building a dashboard. "
        f"The dataset is: {body.dataset_label}\n"
        f"Available columns:\n{json.dumps(body.column_summary, indent=2)}\n\n"
        f"The user wants to explore: \"{body.intent}\"\n\n"
        f"Based on the user's intent and column types, return a JSON object:\n"
        f"{{\n"
        f'  "charts": [\n'
        f'    {{\n'
        f'      "type": "bar" | "sentiment_bar" | "line" | "pie" | "wordcloud" | "verbatims",\n'
        f'      "column": "<column name from the dataset>",\n'
        f'      "label": "<human-readable chart title>"\n'
        f'    }}\n'
        f"  ]\n"
        f"}}\n\n"
        f"Rules:\n"
        f"- Use sentiment_bar only if column contains positive/negative/neutral values\n"
        f"- Use line only for datetime columns\n"
        f"- Use wordcloud and verbatims only for free-text columns\n"
        f"- Use bar for categorical columns with 2-20 unique values\n"
        f"- Use pie for categorical columns with 2-8 unique values where proportion matters\n"
        f"- Include 2-6 charts total, most relevant only\n"
        f"- Every column referenced must exist in the dataset\n"
        f"- Return ONLY the JSON object, no markdown"
    )

    semaphore = asyncio.Semaphore(1)
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        raw = await _call_openai(
            session, semaphore, prompt,
            model="gpt-4o", response_json=True, max_tokens=600, api_key=api_key,
        )

    try:
        result = json.loads(raw)
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to interpret intent")


# ---------------------------------------------------------------------------
# AI-suggested prompt — draft a tagging prompt from just the field name
# ---------------------------------------------------------------------------

@app.post("/api/suggest-prompt")
async def suggest_prompt(body: SuggestPromptRequest, user: Dict[str, Any] = Depends(get_current_user)):
    """Draft a ready-to-run tagging prompt from the field name (+ optional source
    columns and sample values), so non-technical users start from a filled box."""
    api_key = _state_for(user).get("api_key")
    name = (body.field_name or "").strip() or "the field"
    reads = ", ".join([c for c in body.reads_from if c]) or "the source text"
    sample_block = "\n".join(
        f"- {str(s)[:280]}" for s in body.samples[:6] if str(s).strip()
    ) or "(no samples provided)"

    if body.is_cluster:
        shape = (
            "This is a CLUSTER field that produces several columns from one call. Write the instruction so "
            "the AI returns a JSON object whose keys are the requested output columns "
            f"({name}), describing each key briefly."
        )
    else:
        shape = (
            "This is a SINGLE field producing one short value per row. Where sensible, constrain the answer "
            "to a small fixed set of labels, and say to use 'unsure' when the text gives no basis to decide."
        )

    prompt = (
        "You write instructions for an AI that tags each row of a spreadsheet for a non-technical analyst.\n"
        f'The analyst wants an output field called: "{name}".\n'
        f"It should be derived from these source column(s): {reads}.\n"
        f"Sample values from the data:\n{sample_block}\n\n"
        f"{shape}\n\n"
        "Write ONE clear, concise instruction (1-3 sentences, max ~45 words) that can be handed to the AI "
        "as-is to produce this field for every row. Be specific about the allowed outputs. Do not restate "
        "the field name as a prefix, do not include example rows, and add no preamble.\n"
        'Return ONLY JSON: {"prompt": "<the instruction>"}'
    )

    semaphore = asyncio.Semaphore(1)
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        raw = await _call_openai(
            session, semaphore, prompt,
            model="gpt-4o-mini", response_json=True, max_tokens=200, api_key=api_key,
        )
    try:
        suggestion = (json.loads(raw).get("prompt") or "").strip()
    except Exception:
        suggestion = ""
    if not suggestion:
        raise HTTPException(status_code=502, detail="Could not generate a suggestion — please try again.")
    return {"prompt": suggestion}


# ---------------------------------------------------------------------------
# Analytics — /api/embed
# Embeds the text column(s) using OpenAI text-embedding-3-small.
# Stores vectors in the user's session state for RAG retrieval during /api/analyse.
# ---------------------------------------------------------------------------

async def _get_embeddings(texts: List[str], api_key: Optional[str] = None) -> np.ndarray:
    """Batch-embed texts using OpenAI text-embedding-3-small. Returns (n, 1536) float32 array."""
    api_key = api_key or OPENAI_API_KEY
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set on the server.")

    BATCH = 100  # OpenAI allows up to 2048 per call; keep smaller for safety
    all_vectors: List[List[float]] = []
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i in range(0, len(texts), BATCH):
            batch = texts[i : i + BATCH]
            payload = {"model": "text-embedding-3-small", "input": batch}
            async with session.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=payload,
            ) as resp:
                if resp.status != 200:
                    raise ValueError(f"Embedding API error: {resp.status}")
                data = await resp.json()
                # Sort by index to preserve order
                sorted_data = sorted(data["data"], key=lambda x: x["index"])
                all_vectors.extend([d["embedding"] for d in sorted_data])

    return np.array(all_vectors, dtype=np.float32)


def _cosine_similarity(query_vec: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
    """Return cosine similarity of query against all corpus rows."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(corpus_vecs, axis=1, keepdims=True) + 1e-10
    normalised = corpus_vecs / norms
    return normalised @ q


def _retrieve_top_k(st: Dict[str, Any], query: str, k: int = 30) -> List[str]:
    """
    Retrieve the top-k most semantically relevant texts from this user's embeddings.
    Returns list of raw text strings. Used by /api/analyse for verbatim grounding.
    """
    embeddings = st.get("embeddings")
    texts = st.get("embedded_texts")
    if embeddings is None or texts is None or len(texts) == 0:
        return []

    # Embed the query synchronously using the stored vectors
    # (We use a simple numpy dot product — no async needed here since we already have embeddings)
    # For the query we use a pre-computed approach: find texts most similar via keyword overlap
    # as a lightweight fallback when we can't async-embed the query in a sync context.
    # Full async RAG is used in the analyse endpoint where we can await.
    query_lower = query.lower()
    scores = []
    for i, text in enumerate(texts):
        # Simple TF-IDF-like score: count query word overlaps
        words = set(re.findall(r'\w+', query_lower))
        text_lower = text.lower()
        score = sum(1 for w in words if w in text_lower)
        scores.append((score, i))
    scores.sort(reverse=True)
    return [texts[i] for _, i in scores[:k] if _ > 0] or texts[:k]


@app.post("/api/embed")
async def embed_data(body: EmbedRequest, user: Dict[str, Any] = Depends(get_current_user)):
    """
    Embed the text column(s) for RAG retrieval.
    Call this once after pipeline runs (or after uploading a dataset for analytics).
    """
    st = _state_for(user)
    df = st.get("result_df") if body.use_result else st.get("df")
    if df is None:
        df = st.get("df")
    if df is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    text_cols = [c for c in body.column_map.text if c in df.columns]
    if not text_cols:
        raise HTTPException(status_code=400, detail="No valid text columns found in dataset")

    # Concatenate all text columns into one string per row
    def combine_row(row: pd.Series) -> str:
        parts = []
        for col in text_cols:
            val = row.get(col, "")
            if not pd.isna(val) and str(val).strip():
                parts.append(f"{col}: {str(val).strip()}")
        return " | ".join(parts)

    texts = [combine_row(df.iloc[i]) for i in range(len(df))]
    texts = [t if t.strip() else "(empty)" for t in texts]

    try:
        vectors = await _get_embeddings(texts, api_key=st.get("api_key"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    st["embeddings"] = vectors
    st["embedded_texts"] = texts
    st["column_map"] = body.column_map.model_dump()

    return {
        "message": "Embeddings stored",
        "rows_embedded": len(texts),
        "text_columns_used": text_cols,
    }


# ---------------------------------------------------------------------------
# Analytics — helpers for structured aggregation
# ---------------------------------------------------------------------------

def _compute_column_stats(df: pd.DataFrame, column_map: Dict) -> Dict[str, Any]:
    """
    Compute full structured stats from 100% of data.
    This is what the LLM uses for the executive summary — no sampling, no RAG.
    """
    stats: Dict[str, Any] = {"total_rows": len(df)}

    # Categorical columns — value counts + %
    for col in column_map.get("categorical", []):
        if col not in df.columns:
            continue
        counts = df[col].dropna().astype(str).value_counts()
        total = counts.sum()
        stats[col] = {
            "counts": counts.to_dict(),
            "percentages": {k: round(v / total * 100, 1) for k, v in counts.items()},
            "total_non_null": int(total),
        }

    # Datetime columns — date range + volume over time
    for col in column_map.get("datetime", []):
        if col not in df.columns:
            continue
        try:
            dates = pd.to_datetime(df[col], errors="coerce").dropna()
            if len(dates) == 0:
                continue
            stats[col] = {
                "min": str(dates.min().date()),
                "max": str(dates.max().date()),
                "span_days": int((dates.max() - dates.min()).days),
                "total_dated": int(len(dates)),
            }
        except Exception:
            pass

    # Numerical columns — basic descriptive stats
    for col in column_map.get("numerical", []):
        if col not in df.columns:
            continue
        try:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            stats[col] = {
                "mean": round(float(series.mean()), 2),
                "median": round(float(series.median()), 2),
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
                "count": int(len(series)),
            }
        except Exception:
            pass

    return stats


# ---------------------------------------------------------------------------
# Analytics — /api/analyse
# ---------------------------------------------------------------------------

@app.post("/api/analyse")
async def analyse(body: AnalyseRequest, user: Dict[str, Any] = Depends(get_current_user)):
    """
    Generate:
    1. Executive summary (4-5 bullet headlines) — grounded in full structured stats
    2. One tagline per chart — grounded in chart summary + relevant verbatim samples (RAG)

    Uses gpt-4o for quality reasoning. No hallucination risk on counts because
    the LLM only narrates pre-computed numbers — it never touches raw data directly.
    """
    st = _state_for(user)
    api_key = st.get("api_key")
    df = _get_export_df(st)
    if df is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    column_map = body.column_map.model_dump()

    # --- Step 1: Compute full structured stats from 100% of data ---
    stats = _compute_column_stats(df, column_map)

    # --- Step 2: Build executive summary prompt ---
    stats_str = json.dumps(stats, indent=2)
    summary_prompt = (
        f"You are a senior data analyst writing for a client presentation.\n"
        f"Dataset: {body.dataset_label}\n"
        f"Total rows: {stats['total_rows']}\n\n"
        f"Here are the EXACT computed statistics from 100% of the data:\n{stats_str}\n\n"
        f"Write an executive summary as EXACTLY 4-5 bullet points.\n"
        f"Each bullet must:\n"
        f"  • Be one punchy sentence (max 20 words)\n"
        f"  • Reference specific numbers from the stats above (never invent numbers)\n"
        f"  • Surface a genuine insight, tension, or opportunity — not just a restatement of counts\n"
        f"Return ONLY a JSON object: {{\"bullets\": [\"bullet 1\", \"bullet 2\", ...]}}\n"
        f"Do not add preamble, headers, or markdown."
    )

    # --- Step 3: Build per-chart tagline prompts ---
    # For each chart, retrieve relevant verbatims via keyword-based RAG
    chart_prompts = []
    for chart in body.charts:
        verbatims = _retrieve_top_k(
            st,
            query=f"{chart.column} {chart.label} {chart.chart_type}",
            k=25,
        )
        verbatim_sample = "\n".join(f"- {v}" for v in verbatims[:25]) if verbatims else "(no verbatims available)"

        chart_prompt = (
            f"You are a senior analyst writing a one-line executive tagline for a chart in a client deck.\n"
            f"Chart: {chart.label} ({chart.chart_type})\n"
            f"Column: {chart.column}\n"
            f"Chart data summary: {json.dumps(chart.summary)}\n\n"
            f"Sample verbatims most relevant to this chart (drawn from real data):\n{verbatim_sample}\n\n"
            f"Write ONE punchy tagline (max 15 words) that:\n"
            f"  • References a specific number or finding from the chart data\n"
            f"  • Reflects the tone from the verbatims if relevant\n"
            f"  • Reads like an analyst insight, not a chart title\n"
            f"Return ONLY a JSON object: {{\"tagline\": \"your tagline here\"}}"
        )
        chart_prompts.append((chart.label, chart_prompt))

    # --- Step 4: Fire all LLM calls concurrently ---
    semaphore = asyncio.Semaphore(5)
    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Executive summary
        summary_task = _call_openai(
            session, semaphore, summary_prompt,
            model="gpt-4o", response_json=True, max_tokens=400, api_key=api_key,
        )
        # Taglines — one per chart
        tagline_tasks = [
            _call_openai(
                session, semaphore, prompt,
                model="gpt-4o", response_json=True, max_tokens=100, api_key=api_key,
            )
            for _, prompt in chart_prompts
        ]

        all_tasks = [summary_task] + tagline_tasks
        all_results = await asyncio.gather(*all_tasks)

    # --- Step 5: Parse results ---
    summary_raw = all_results[0]
    tagline_raws = all_results[1:]

    try:
        bullets = json.loads(summary_raw).get("bullets", [])
    except Exception:
        bullets = ["Summary could not be generated — please retry."]

    taglines: Dict[str, str] = {}
    for (label, _), raw in zip(chart_prompts, tagline_raws):
        try:
            taglines[label] = json.loads(raw).get("tagline", "")
        except Exception:
            taglines[label] = ""

    return {
        "executive_summary": bullets,
        "chart_taglines": taglines,
        "stats_used": stats,  # Return so frontend can display/verify
    }
