# Self-Service Toolkit

An AI data-enrichment and analytics tool for non-technical teams. Upload a
spreadsheet, describe the columns you want in plain English (or let the AI draft
the prompt for you), and every row is tagged by GPT — then explore the results
in an auto-built dashboard and download them as CSV or Excel.

Built by the M&C Saatchi Data Team.

---

## What it does

- **Tag any spreadsheet with AI** — add output fields like `sentiment`, `topic`,
  or `urgency`; each row is processed and filled in.
- **✨ AI-suggested prompts** — name a field and the tool drafts a ready-to-run
  prompt, so you never start from a blank box.
- **Prompt bank** — one-click starting points (sentiment, topic, translation,
  complaint detection, purchase intent, and more).
- **Cluster fields** — produce several columns from a single AI call
  (e.g. topic + sentiment + key quote together).
- **Conditional fields** — only run a prompt when a row matches a rule
  (e.g. summarise the complaint *only* for negative rows).
- **Analytics dashboard** — describe what you want to see and the AI builds the
  charts, with an executive summary and per-chart insight lines.
- **Export** — download the enriched data as CSV or XLSX.
- **Sign-in & isolation** — colleagues sign in with their `@mcsaatchi.com` email
  (magic link); each person only sees their own data.

---

## Architecture

```
index.html          → GitHub Pages (static frontend)
     │  sign in (magic link) → Supabase Auth → JWT
     │  fetch() with the JWT
     ▼
main.py (FastAPI)    → any Docker host
     │  verifies the JWT, isolates each user's session
     ▼
Anthropic API        → Claude Sonnet 5 (tagging) / Claude Opus 4.8 (dashboards)
OpenAI API           → text-embedding-3-small (embeddings only)
```

The frontend parses files in-browser and sends rows to the backend. The backend
verifies the caller's Supabase token, keeps each user's working data separate,
calls Claude, and returns results. Both API keys live on the server — never
sent by the client.

Two providers on purpose: Claude handles all reasoning; OpenAI is used only for
embeddings, which Anthropic doesn't offer and the analytics retrieval needs.

---

## Repo structure

```
self-service-toolkit/
├── index.html            # Frontend — single-page app
├── main.py               # Backend — FastAPI application
├── requirements.txt      # Runtime dependencies (pinned)
├── requirements-dev.txt  # Test dependencies (pytest, httpx)
├── Dockerfile            # Builds/runs the backend on any Docker host
├── .env.example          # Environment variable template
├── scripts/run_tagging.py # Run a saved field config over a CSV via the backend
├── scripts/run_batch.py  # Same, via Claude's Batch API — for very large files
├── scripts/compare_runs.py # Compare two tagged outputs (e.g. full vs trimmed text)
├── scripts/map_labels.py  # Map free-text labels onto a category list; rebuild the breakdown
├── scripts/finalise_breakdown.py # Apply reviewed merge rules; write the final breakdown
├── configs/              # Saved field configs (e.g. hydra_mentions.json)
├── tests/                # End-to-end tests (upload → tag → export → auth)
├── docs/                 # User guide (served by GitHub Pages)
├── README.md             # This file
├── DEV_GUIDE.md          # Full technical reference
├── LOCAL_DEV.md          # Run locally with real login (step by step)
└── SUPABASE_SETUP.md     # One-time Supabase project setup
```

---

## Running locally

**Requirements:** Python 3.9+, and a Supabase project (see `SUPABASE_SETUP.md`)
if you want real login. To skip login entirely, set `AUTH_DISABLED=true`.

You run **two servers** — the backend API and a static server for the page —
each in its own terminal. Full walkthrough with troubleshooting is in
**`LOCAL_DEV.md`**; the short version:

```bash
git clone https://github.com/Mia-mcsaatchi/self-service-toolkit.git
cd self-service-toolkit

# One-time
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then set ANTHROPIC_API_KEY, OPENAI_API_KEY, SUPABASE_URL

# Terminal 1 — backend
uvicorn main:app --reload --port 8000

# Terminal 2 — frontend
python3 -m http.server 5500 --bind 127.0.0.1
```

Open <http://127.0.0.1:5500/index.html>. The frontend auto-detects it's local
and talks to the backend on port 8000.

### Sign-in / multi-user

Colleagues sign in with their `@mcsaatchi.com` email via a Supabase magic link,
and each person's uploads and results are isolated. One-time setup is in
**`SUPABASE_SETUP.md`**. The backend verifies both legacy (HS256) and new
asymmetric (ES256/RS256) Supabase tokens.

### Tagging a CSV from the command line

`scripts/run_tagging.py` does what Steps 1–3 of the page do — upload, save the
field config, tag in chunks, export — so a saved config in `configs/` can be
re-run without clicking through the UI. With the backend running locally
(`AUTH_DISABLED=true` in `.env`):

```bash
python scripts/run_tagging.py hydra_sample_300.csv configs/hydra_mentions.json --max-rows 20  # trial
python scripts/run_tagging.py hydra_sample_300.csv configs/hydra_mentions.json                # full run
```

The result is written next to the input as `<name>_tagged.csv`. To use the
hosted backend instead, add `--backend https://self-service-toolkit.onrender.com
--token <your Supabase access token>`.

### Large files: the Batch API runner

For tens of thousands of rows, use `scripts/run_batch.py` instead. It sends the
same prompts straight to Claude's Message Batches API: half the price, no
timeouts, identical rows sent once, and resumable (re-run the same command if
it's stopped). It needs `ANTHROPIC_API_KEY` in `.env`; the backend doesn't have
to be running.

```bash
python scripts/run_batch.py big.csv configs/hydra_mentions.json --estimate         # cost only, sends nothing
python scripts/run_batch.py big.csv configs/hydra_mentions.json --max-rows 300     # trial
python scripts/run_batch.py big.csv configs/hydra_mentions.json                    # full run
```

Add `--model claude-haiku-5-5` to try a cheaper model (check its answers on a
sample first).

Long posts can be trimmed to cut cost: a `"trim"` section in the config keeps
each post's opening plus N characters either side of every match of a term
(with a note telling the model how much was cut). Try sizes on a sample with
`--trim-chars`, then compare runs with `scripts/compare_runs.py`.

To group the free-text `context` labels into a fixed category list, run
`scripts/map_labels.py <tagged.csv> configs/hydra_taxonomy.json`. It maps each
distinct label (100 per request, ~$5 for 30k labels), proposes new associations
for review, and writes a categorised CSV plus a breakdown workbook. After
reviewing it, put merges and fixes in a rules file (see
`configs/hydra_merges.json`) and run `scripts/finalise_breakdown.py` — free,
re-runnable, no Claude calls.

For long runs: keep data in `data/` (git-ignored), set a spending cap with
`--budget <dollars>`, and on a Mac prefix the command with `caffeinate -i` so
the laptop doesn't sleep. Work is sent a few batches at a time, so a credit or
permission problem stops the run after one small batch; finished rows are
written straight away (unfinished cells say `pending`) and re-running the same
command continues from there.

---

## Testing

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

The suite runs the whole flow with the Claude call mocked (deterministic,
offline, free): `.xlsx` upload, the tagging pipeline (including conditional
branching), CSV/XLSX export, AI-prompt suggestion, and the auth layer
(rejection + per-user isolation).

---

## Deployment

**Backend → any Docker host** (Render, Fly.io, Cloud Run, Railway, …). The
`Dockerfile` builds and runs it. Set these environment variables on the host:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude access — tagging, dashboards, analysis (required) |
| `OPENAI_API_KEY` | OpenAI access — embeddings only (required for analytics) |
| `SUPABASE_URL` | your project URL, for verifying new-style tokens |
| `SUPABASE_JWT_SECRET` | only if your project uses the legacy HS256 secret |
| `ALLOWED_EMAIL_DOMAIN` | restrict sign-in (default `mcsaatchi.com`) |

Then point the frontend's production `BACKEND` constant (top of `index.html`)
at the deployed URL, and add that URL to the backend CORS allow-list.

**Frontend → GitHub Pages** — Settings → Pages → source branch `/` (root).

---

See `DEV_GUIDE.md` for the full technical reference.
