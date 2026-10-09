# CLAUDE.md — working on the Self-Service Toolkit

An internal M&C Saatchi Data Team tool: upload a spreadsheet, tag every row with
Claude, explore the results in a dashboard, save and share it. Users are
non-technical analysts, so plain wording and no dead ends matter more than flash.

## Layout

| Path | What it is |
|---|---|
| `main.py` | The whole backend (FastAPI): auth, tagging pipeline, analytics, storage |
| `index.html` | The whole frontend (single file, vanilla JS + Chart.js), served by GitHub Pages |
| `tests/test_e2e.py` | End-to-end tests; Claude, OpenAI and Supabase are mocked |
| `supabase/schema.sql` | Database tables (`datasets`, `dashboards`, `dashboard_shares`) |
| `DEPLOY.md`, `LOCAL_DEV.md`, `SUPABASE_SETUP.md` | How to deploy, run locally, set up Supabase |
| `ROADMAP.md`, `HANDOVER.md` | Product backlog and project handover |

## Run and test

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest -q                                   # must pass before any PR
AUTH_DISABLED=true uvicorn main:app --reload --port 8000    # local backend, no login
python3 -m http.server 5500 --bind 127.0.0.1                # local frontend
```

Open <http://127.0.0.1:5500/index.html>; the page talks to the local backend
automatically. Environment variables are listed in `.env.example`.

## Deploy

Merging to `main` deploys both halves: GitHub Pages serves `index.html`, and
Render rebuilds the backend from the `Dockerfile`. Render's free tier restarts
the backend on every deploy and after ~15 min idle, which wipes in-memory
sessions; the page keeps a copy of tagged results and restores them.

## House rules

- **Never push to `main`.** Work on a branch and open a pull request; the
  `Tests` workflow must be green.
- **Add or update a test** for every behaviour change in `main.py`. Mock the
  model and storage calls the way `tests/test_e2e.py` does; tests never call
  real APIs.
- **No real data in the repo.** Uploaded files contain client posts and author
  names. Use the synthetic sample in the tests.
- **Secrets stay out of code**: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
  `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_JWT_SECRET` live in `.env` locally and
  in Render's environment. The Supabase URL and publishable key in
  `index.html` are public by design.
- **Claude calls go through `_call_claude`** with a JSON schema
  (`output_config.format`). Models are the constants near the top of `main.py`
  (`TAGGING_MODEL`, `DASHBOARD_MODEL`); check thinking/effort rules before
  changing them — some newer models reject `thinking: {"type": "disabled"}`.
- **Failures must be visible.** Log errors (`logger`) rather than swallowing
  them, and give users a message that says what happened and what to do next.
- Keep the UI copy plain and specific; the users are analysts, not engineers.
