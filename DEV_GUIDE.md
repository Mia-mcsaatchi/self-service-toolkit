# Dev Guide — Self-Service Toolkit

Full technical reference for the M&C Saatchi Data Team.

---

## Contents

1. [Architecture](#1-architecture)
2. [Backend — main.py](#2-backend)
3. [Field model reference](#3-field-model-reference)
4. [API reference](#4-api-reference)
5. [Frontend — index.html](#5-frontend)
6. [Deployment](#6-deployment)
7. [Adding a new feature](#7-adding-a-new-feature)
8. [Known limitations](#8-known-limitations)
9. [Phase 3 backlog](#9-phase-3-backlog)

---

## 1. Architecture

```
┌─────────────────────────────────────┐
│  index.html                         │
│  GitHub Pages — static, always on   │
│  Parses files in-browser (PapaParse,│
│  SheetJS), sends rows to backend    │
└─────────────┬───────────────────────┘
              │ fetch() — HTTPS
              ▼
┌─────────────────────────────────────┐
│  main.py (FastAPI + uvicorn)        │
│  Railway — persistent, always on    │
│  Holds session state in memory      │
│  Builds prompts, calls OpenAI       │
│  Returns results + exports          │
└─────────────┬───────────────────────┘
              │ HTTPS — Bearer token
              ▼
┌─────────────────────────────────────┐
│  OpenAI API — gpt-4o-mini           │
│  Structured JSON output             │
│  Temperature 0, max_tokens 512      │
└─────────────────────────────────────┘
```

**Key design decisions:**

- File parsing happens in the browser — no file upload to the backend, just row arrays sent as JSON. Keeps the backend stateless with respect to raw files.
- API key lives server-side only — set as a Railway environment variable, never sent by the client.
- In-memory state — `_state` dict holds df, config, result_df per server process. Single-user for MVP. Multi-user requires session tokens (Phase 3).
- CORS locked to GitHub Pages URL + localhost for local dev.

---

## 2. Backend

**Stack:** Python 3.11, FastAPI, uvicorn, pandas, aiohttp, pydantic, python-dotenv

**Entry point:** `main.py`

**Key globals:**

```python
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # read at startup

_state = {
    "df": None,          # pd.DataFrame — uploaded data
    "config": None,      # dict — field config from /api/upload-config
    "result_df": None,   # pd.DataFrame — results after /api/process
}
```

**Request flow for a full pipeline run:**

```
POST /api/upload-data   → stores df in _state
POST /api/upload-config → stores config in _state
POST /api/process       → runs _run_pipeline() → stores result_df
GET  /api/export/csv    → streams result_df as CSV
```

**Pipeline execution — `_run_pipeline()`:**

For each row (up to `max_rows`, default = all):
1. For each field, call `_resolve_prompt(field, row)` to determine which prompt to use
2. Group fields that share the same resolved prompt + reads_from into one API call (cluster behaviour — one call, multiple output keys)
3. Fire all groups for this row concurrently via `asyncio.gather`
4. Parse JSON response, write each key to the appropriate column
5. Fields where `_resolve_prompt` returns `None` (condition not met) write `"n/a"` — no API call made

**Concurrency:** controlled by `asyncio.Semaphore(max_concurrent)`. Default 10 parallel calls. User-configurable in the frontend.

**Retry logic:** 2 retries with exponential backoff (1s, 2s). On final failure returns `"{}"` — field writes `"unsure"`.

---

## 3. Field model reference

### Pydantic models

```python
class Condition(BaseModel):
    column: str           # column to check (source column for independent fields)
    operator: str         # "is" or "is not"
    values: List[str]     # match if row[column] is in this list (OR within values)

class Branch(BaseModel):
    conditions: List[Condition]   # ALL must match — AND logic within a branch
    prompt: str                   # prompt to fire if all conditions match

class Field(BaseModel):
    name: str             # output column name
    prompt: str           # default prompt (mode=default) — empty if mode=conditional
    reads_from: List[str] # source columns to include as context in the prompt
    field_type: str       # "independent" | "dependent" (dependent = Phase 3)
    is_cluster: bool      # True if multiple fields share this config (cluster mode)
    mode: str             # "default" | "conditional"
    branches: List[Branch]# evaluated top-to-bottom; first full match wins
```

### Prompt resolution logic

```
mode = "default"
  → always use field.prompt (runs on every row)

mode = "conditional"
  → for each branch (top to bottom):
      if ALL branch.conditions match the row → use branch.prompt, stop
  → no branch matched → return None → write "n/a", skip API call
```

### Condition matching

```python
# operator "is":    row[column] in condition.values
# operator "is not": row[column] not in condition.values
# Multiple values within one condition = OR (any match fires)
# Multiple conditions within one branch = AND (all must match)
# Multiple branches = OR (first full match wins)
```

### Example — full config payload

```json
{
  "base_prompt": "You are a data analyst...",
  "fields": [
    {
      "name": "translation",
      "prompt": "Translate this text to English. If already English, return unchanged.",
      "reads_from": ["post_text"],
      "field_type": "independent",
      "is_cluster": false,
      "mode": "default",
      "branches": []
    },
    {
      "name": "follow_up",
      "prompt": "",
      "reads_from": ["post_text", "overall_sentiment"],
      "field_type": "independent",
      "is_cluster": false,
      "mode": "conditional",
      "branches": [
        {
          "conditions": [
            { "column": "overall_sentiment", "operator": "is", "values": ["Negative"] },
            { "column": "language", "operator": "is", "values": ["de", "fr", "es"] }
          ],
          "prompt": "What is the main complaint? Under 15 words."
        },
        {
          "conditions": [
            { "column": "overall_sentiment", "operator": "is", "values": ["Positive"] }
          ],
          "prompt": "What is the main positive signal? Under 15 words."
        }
      ]
    }
  ]
}
```

---

## 4. API reference

Base URL: `https://self-service-toolkit-production.up.railway.app`

### GET /api/health
Returns `{"status": "ok"}`. Use to check backend is alive.

---

### POST /api/upload-data
Load pre-parsed rows from the frontend.

**Request:**
```json
{
  "columns": ["col1", "col2"],
  "rows": [["val1", "val2"], ["val3", "val4"]]
}
```

**Response:**
```json
{
  "message": "Data loaded",
  "columns": ["col1", "col2"],
  "row_count": 2,
  "preview": [{"col1": "val1", "col2": "val2"}]
}
```

---

### POST /api/upload
Load a CSV or XLSX file directly (multipart/form-data). Optional `?sheet=SheetName` param for Excel.

---

### POST /api/upload-config
Save field configuration. See Field model reference above for full schema.

**Response:**
```json
{ "message": "Config saved", "field_count": 3 }
```

---

### GET /api/config
Returns the currently stored field config.

---

### POST /api/process
Run the pipeline against the loaded data.

**Request:**
```json
{ "max_rows": 0, "max_concurrent": 10 }
```
`max_rows: 0` = process all rows.

**Response:**
```json
{
  "message": "Processing complete",
  "row_count": 367,
  "columns": ["col1", "col2", "output_field"],
  "preview": [{ "col1": "...", "output_field": "..." }]
}
```

---

### GET /api/export/csv
Stream the result DataFrame as CSV. Falls back to uploaded data if pipeline hasn't run.

### GET /api/export/xlsx
Same as above, Excel format.

---

## 5. Frontend

**Stack:** Vanilla HTML/CSS/JS — no framework, no build step. Single file `index.html`.

**Libraries loaded from CDN:**
- PapaParse 5.3.0 — CSV parsing
- SheetJS 0.18.5 — Excel parsing

**Key globals:**

```javascript
const BACKEND = 'https://self-service-toolkit-production.up.railway.app';

const state = {
  columns: [],      // column names from uploaded file
  parsedData: null, // { columns, rows } — parsed file data
  workbook: null,   // SheetJS workbook (Excel only)
};

const PROMPT_BANK = [...]; // 13 pre-built prompt templates
```

**Field card state** is stored as `card.dataset.mode` (`"default"` or `"conditional"`). Output type (single/cluster) is inferred from the badge class on the card. This means field state is DOM-bound — refreshing the page clears all configured fields.

**Categorical column detection** (`getCategoricalColumns()`): scans `state.parsedData` for columns with 2–30 unique non-null values. Used to populate the conditional branch value checklists and to grey out non-categorical columns.

**collectFields()** — assembles the config payload from the DOM before sending to `/api/upload-config`. Expands cluster field names (comma-separated) into individual field objects sharing the same prompt.

---

## 6. Deployment

### Backend — Railway

- Railway detects `Dockerfile` automatically on push
- Build: `pip install -r requirements.txt`
- Run: `uvicorn main:app --host 0.0.0.0 --port 8000`
- Environment variable: `OPENAI_API_KEY` — set in Railway dashboard → Variables tab
- Auto-deploys on every push to `main` branch
- Free tier: $5/month credit — sufficient for dev/demo usage

### Frontend — GitHub Pages

- Enabled in repo Settings → Pages → Source: main branch, / (root)
- Serves `index.html` at `https://mia-mcsaatchi.github.io/self-service-toolkit`
- Serves `docs/index.html` at `.../self-service-toolkit/docs/`
- Auto-deploys on every push to `main` — update takes ~30 seconds

### CORS

Allowed origins in `main.py`:
```python
allow_origins=[
    "https://mia-mcsaatchi.github.io",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
```
Update this list if the frontend moves to a custom domain.

---

## 7. Adding a new feature

### Backend only (new endpoint or logic change)

```bash
# 1. Edit main.py locally
# 2. Test at http://127.0.0.1:8000/docs
uvicorn main:app --reload

# 3. Push
git add main.py
git commit -m "Add: description"
git push
# Railway redeploys in ~2 min
```

### Frontend only (UI change)

```bash
# 1. Edit index.html
# 2. Open index.html directly in browser to test
#    (hardcoded BACKEND URL points to Railway — needs internet)
#    Or temporarily change BACKEND to http://127.0.0.1:8000 for offline test

# 3. Push
git add index.html
git commit -m "UI: description"
git push
# GitHub Pages updates in ~30 seconds
```

### Both together

```bash
git add main.py index.html
git commit -m "Feature: description"
git push
```

### Adding a new prompt to the prompt bank

In `index.html`, find `const PROMPT_BANK = [...]` and add an entry:

```javascript
{
  label: 'Your prompt name',      // shown in dropdown
  value: 'unique_key',            // internal identifier
  prompt: 'Your prompt text...',  // pre-fills the textarea
  // Optional for cluster presets:
  isCluster: true,
  clusterCols: 'col1, col2, col3'
}
```

No backend change needed.

### Adding a new export format

In `main.py`, add a new route following the pattern of `export_csv` / `export_xlsx`. In `index.html`, add a button calling `exportFile('yourformat')` which opens `BACKEND + '/api/export/yourformat'`.

---

## 8. Known limitations

| Limitation | Impact | Workaround / Plan |
|---|---|---|
| In-memory state — single session | Two users running simultaneously overwrite each other's data | Acceptable for MVP internal use. Phase 3: session tokens. |
| No persistence — server restart clears state | Data and config lost on Railway redeploy | Railway rarely restarts mid-run. Phase 3: file/DB storage. |
| Conditional branching — independent fields only | Dependent fields (chaining on prior output) not yet built | Phase 3 |
| Google Sheet export | Writing back to a Sheet requires OAuth | Phase 3 |
| No progress streaming | Long runs (100+ rows) show no per-row progress | Phase 3: SSE streaming |
| Categorical detection heuristic | Columns with 2–30 unique values are assumed categorical | Works well in practice. Edge cases: columns with exactly 30 unique values but are free text. |
| CORS locked to GitHub Pages | Cannot use frontend from other domains without updating `allow_origins` | Update `main.py` CORS list and redeploy |

---

## 9. Phase 3 backlog

Prioritised by expected value:

1. **Dependent field type** — fields that read from independent output columns (e.g. run a follow-up prompt on the sentiment output)
2. **Session management** — UUID session tokens, per-session state — enables concurrent users
3. **SSE progress streaming** — row-by-row progress bar during long runs
4. **Google Sheet export** — write results back to a Sheet via service account (no user OAuth)
5. **Save custom prompts** — per-user prompt library (requires a database)
6. **Data persistence** — store uploads and results to S3/R2 so server restarts don't wipe state
7. **Tests and CI** — pytest for API routes, GitHub Actions on push
8. **Cost controls** — dry-run mode (token estimate before spending), per-run spend cap
