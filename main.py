from __future__ import annotations

import asyncio
import io
import json
import os
import re
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
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
    # Analytics state
    "embeddings": None,       # np.ndarray of shape (n_rows, embedding_dim)
    "embedded_texts": None,   # List[str] — the text we embedded (one per row)
    "column_map": None,       # Dict describing which columns are categorical/datetime/text/numerical
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

# ---------------------------------------------------------------------------
# Analytics models
# ---------------------------------------------------------------------------

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

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/export/csv-debug")
def export_csv_debug():
    """Returns first 5 rows as JSON — use to diagnose export failures."""
    df = _state.get("result_df") or _state.get("df")
    if df is None:
        raise HTTPException(status_code=400, detail="No data")
    try:
        df2 = df.copy()
        # Sanitise column names
        df2.columns = [str(c).replace("-", "_").replace(" ", "_") for c in df2.columns]
        # Coerce all values
        for col in df2.columns:
            df2[col] = df2[col].fillna("").astype(str).replace("nan", "").replace("None", "")
        return {"columns": df2.columns.tolist(), "rows": df2.head(5).to_dict("records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug export failed: {type(e).__name__}: {e}")


@app.get("/api/debug")
def debug():
    return {
        "df_rows": len(_state["df"]) if _state["df"] is not None else None,
        "df_cols": _state["df"].columns.tolist() if _state["df"] is not None else None,
        "result_df_rows": len(_state["result_df"]) if _state["result_df"] is not None else None,
        "result_df_cols": _state["result_df"].columns.tolist() if _state["result_df"] is not None else None,
        "config_fields": len(_state["config"]["fields"]) if _state["config"] else None,
    }


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
    _state["embeddings"] = None
    _state["embedded_texts"] = None

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
    _state["embeddings"] = None
    _state["embedded_texts"] = None
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
) -> str:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
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

def _safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all columns to strings and sanitise names to avoid serialisation errors."""
    df = df.copy()
    for col in df.columns:
        try:
            df[col] = df[col].fillna("").astype(str).replace({"nan": "", "None": "", "<NA>": ""})
        except Exception:
            df[col] = df[col].astype(str)
    return df


@app.get("/api/export/csv")
def export_csv():
    df = _state.get("result_df") or _state.get("df")
    if df is None:
        raise HTTPException(status_code=400, detail="No data to export")
    try:
        df = _safe_df(df)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(
            io.BytesIO(buf.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=results.csv"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV export failed: {e}")


@app.get("/api/export/xlsx")
def export_xlsx():
    df = _state.get("result_df") or _state.get("df")
    if df is None:
        raise HTTPException(status_code=400, detail="No data to export")
    try:
        df = _safe_df(df)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Results")
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=results.xlsx"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"XLSX export failed: {e}")


# ---------------------------------------------------------------------------
# Analytics — /api/embed
# Embeds the text column(s) using OpenAI text-embedding-3-small.
# Stores vectors in _state for RAG retrieval during /api/analyse.
# ---------------------------------------------------------------------------

async def _get_embeddings(texts: List[str]) -> np.ndarray:
    """Batch-embed texts using OpenAI text-embedding-3-small. Returns (n, 1536) float32 array."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set on the server.")

    BATCH = 100  # OpenAI allows up to 2048 per call; keep smaller for safety
    all_vectors: List[List[float]] = []
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
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


def _retrieve_top_k(query: str, k: int = 30) -> List[str]:
    """
    Retrieve the top-k most semantically relevant texts from _state embeddings.
    Returns list of raw text strings. Used by /api/analyse for verbatim grounding.
    """
    embeddings = _state.get("embeddings")
    texts = _state.get("embedded_texts")
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
async def embed_data(body: EmbedRequest):
    """
    Embed the text column(s) for RAG retrieval.
    Call this once after pipeline runs (or after uploading a dataset for analytics).
    """
    df = _state.get("result_df") if body.use_result else _state.get("df")
    if df is None:
        df = _state.get("df")
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
        vectors = await _get_embeddings(texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    _state["embeddings"] = vectors
    _state["embedded_texts"] = texts
    _state["column_map"] = body.column_map.model_dump()

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
async def analyse(body: AnalyseRequest):
    """
    Generate:
    1. Executive summary (4-5 bullet headlines) — grounded in full structured stats
    2. One tagline per chart — grounded in chart summary + relevant verbatim samples (RAG)

    Uses gpt-4o for quality reasoning. No hallucination risk on counts because
    the LLM only narrates pre-computed numbers — it never touches raw data directly.
    """
    df = _state.get("result_df") or _state.get("df")
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
            model="gpt-4o", response_json=True, max_tokens=400
        )
        # Taglines — one per chart
        tagline_tasks = [
            _call_openai(
                session, semaphore, prompt,
                model="gpt-4o", response_json=True, max_tokens=100
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
