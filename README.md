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
OpenAI API           → GPT-4o-mini (tagging) / GPT-4o (analytics)
```

The frontend parses files in-browser and sends rows to the backend. The backend
verifies the caller's Supabase token, keeps each user's working data separate,
calls OpenAI, and returns results. The OpenAI key lives on the server — never
sent by the client.

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
cp .env.example .env          # then set OPENAI_API_KEY (and SUPABASE_URL)

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

---

## Testing

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

The suite runs the whole flow with the OpenAI call mocked (deterministic,
offline, free): `.xlsx` upload, the tagging pipeline (including conditional
branching), CSV/XLSX export, AI-prompt suggestion, and the auth layer
(rejection + per-user isolation).

---

## Deployment

**Backend → any Docker host** (Render, Fly.io, Cloud Run, Railway, …). The
`Dockerfile` builds and runs it. Set these environment variables on the host:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | OpenAI access (required) |
| `SUPABASE_URL` | your project URL, for verifying new-style tokens |
| `SUPABASE_JWT_SECRET` | only if your project uses the legacy HS256 secret |
| `ALLOWED_EMAIL_DOMAIN` | restrict sign-in (default `mcsaatchi.com`) |

Then point the frontend's production `BACKEND` constant (top of `index.html`)
at the deployed URL, and add that URL to the backend CORS allow-list.

**Frontend → GitHub Pages** — Settings → Pages → source branch `/` (root).

---

See `DEV_GUIDE.md` for the full technical reference.
