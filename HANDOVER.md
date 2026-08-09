# Handover — Self-Service Toolkit

**Purpose of this doc:** everything a new owner needs to take this project over,
run it, and decide what to harden for an enterprise / production rollout. Written
at the end of the POC phase.

- **Status:** working proof-of-concept. Multi-user login, AI tagging pipeline,
  interactive dashboards, saved datasets, and saveable + shareable dashboards
  all function and are covered by tests.
- **Audience:** the receiving data lead / engineering team.
- **Companion docs:** `README.md` (overview), `DEV_GUIDE.md` (technical
  reference), `LOCAL_DEV.md` (run locally), `SUPABASE_SETUP.md` (Supabase setup).

---

## 1. What it does

Upload a spreadsheet → describe output columns in plain English (or let the AI
draft the prompt) → every row is tagged by GPT → explore results in an
interactive dashboard → save and share it. Built for non-technical users.

## 2. Architecture

```
index.html  ── GitHub Pages (static SPA; parses files in-browser)
   │  sign in (Supabase magic link) → JWT
   │  fetch() with Authorization: Bearer <jwt>
   ▼
main.py  ── FastAPI on any Docker host  (currently run locally in dev)
   │  verifies the JWT, isolates each user's session, calls OpenAI
   ├── OpenAI API   (gpt-4o-mini tagging, gpt-4o analytics, embeddings)
   └── Supabase     (Auth + Postgres via PostgREST, service-role key)
```

Two independently deployed halves: a static frontend and a stateless-ish
backend. The OpenAI key and the Supabase service-role key live **only** on the
backend.

## 3. Repo map

| Path | What it is |
|---|---|
| `main.py` | FastAPI backend — all API endpoints, auth, pipeline, storage |
| `index.html` | Entire frontend (single-file SPA, vanilla JS + Chart.js) |
| `requirements.txt` / `requirements-dev.txt` | Runtime / test deps |
| `Dockerfile` | Builds & runs the backend on any Docker host |
| `.env.example` | All backend env vars (copy to `.env`) |
| `supabase/schema.sql` | DB migration — `datasets`, `dashboards`, `dashboard_shares` |
| `tests/test_e2e.py` | End-to-end tests (OpenAI + storage mocked) |
| `.github/workflows/keep-supabase-awake.yml` | Daily ping so the free Supabase project doesn't pause |
| `README.md`, `DEV_GUIDE.md`, `LOCAL_DEV.md`, `SUPABASE_SETUP.md` | Docs |

## 4. Environment & secrets

Backend env vars (see `.env.example`):

| Var | Secret? | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | **secret** | OpenAI access (tagging + analytics) |
| `SUPABASE_URL` | public | project URL, for JWT (JWKS) verification + PostgREST |
| `SUPABASE_JWT_SECRET` | **secret** | only if the project uses legacy HS256 tokens (current project uses new ES256 keys, so this is blank) |
| `SUPABASE_SERVICE_ROLE_KEY` | **secret** | backend read/write to Postgres (saved datasets/dashboards) |
| `ALLOWED_EMAIL_DOMAIN` | public | restricts sign-in (default `mcsaatchi.com`) |
| `AUTH_DISABLED` | — | local-only bypass; **never** true in a deployment |

Frontend constants (top of `index.html`, both **public / safe to commit**):
`SUPABASE_URL`, `SUPABASE_ANON_KEY` (the publishable key).

> **Accounts to transfer to the new owner:** the Supabase project, the OpenAI
> account/billing, the GitHub repo, and whatever host runs the backend.

## 5. Running & deploying

- **Local:** see `LOCAL_DEV.md` (two servers: `uvicorn` on :8000 + a static
  server on :5500). `AUTH_DISABLED=true` skips Supabase for quick local runs.
- **Backend deploy:** any Docker host (Render / Fly.io / Cloud Run / ECS). Set
  the env vars above; then point the frontend's production `BACKEND` constant at
  the deployed URL and add that URL to CORS (`allow_origins` in `main.py`).
- **Frontend deploy:** GitHub Pages (Settings → Pages → branch `/` root).
- **Tests:** `pip install -r requirements.txt -r requirements-dev.txt && pytest -q`.

## 6. Data model (Supabase Postgres)

- **`datasets`** — a saved tagged dataset per user (`user_id`, `columns`,
  `rows` JSONB, `row_count`).
- **`dashboards`** — a saved dashboard: `owner_id`, `config` (chart layout),
  a **data snapshot** (`columns`/`rows`), `is_public`, `share_token`.
- **`dashboard_shares`** — `dashboard_id` → `shared_with_email` grants.

All three have **RLS enabled with no policies**: the anon/publishable key cannot
touch them directly; only the backend (service-role key) can, and the backend
scopes every query by the verified user id.

## 7. API surface (all `/api/*`, JWT-required unless noted)

- **Public:** `GET /health`, `GET /public/dashboard/{token}` (read-only if published).
- **Auth/identity:** `GET /me` (+ `storage_enabled`).
- **Pipeline:** `POST /upload`, `POST /upload-data`, `POST /upload-config`,
  `GET /config`, `POST /process`, `GET /result-data`, `GET /export/{csv,xlsx}`.
- **AI helpers:** `POST /suggest-prompt`, `POST /interpret`, `POST /embed`, `POST /analyse`.
- **Datasets:** `POST/GET /datasets`, `GET/DELETE /datasets/{id}`.
- **Dashboards:** `POST/GET /dashboards`, `GET/DELETE /dashboards/{id}`,
  `POST /dashboards/{id}/publish`, `POST /dashboards/{id}/share`, `GET /shared-dashboards`.

## 8. Known limitations / tech debt

These are deliberate POC trade-offs, listed so the next team isn't surprised:

1. **In-memory per-user working state.** `main.py` keeps each user's active
   df/config/results in a process-local dict (`_sessions`). This means the
   backend **cannot be horizontally scaled** as-is (a second instance won't share
   state) and a restart drops in-progress work. Saved datasets/dashboards *are*
   durable in Postgres; the live editing buffer is not.
2. **Full-data JSONB snapshots.** Datasets and dashboards store all rows as
   JSONB. Fine for hundreds/low-thousands of rows; won't scale to large data.
   Dashboards duplicate the dataset snapshot rather than referencing it.
3. **Public link = anyone-with-link.** `?view=<token>` needs no login by design.
   The named-colleague share is the access-controlled option.
4. **Dashboard filters aren't persisted** — saved dashboards keep charts, not the
   active filter state.
5. **Single shared OpenAI key**, no per-user cost controls or rate limiting.
6. **No CI, monitoring, or error tracking** wired up (tests exist but run manually).
7. **Free-tier Supabase** pauses on inactivity (mitigated by the keep-alive
   workflow, which only runs once the branch is on the default branch).

## 9. Enterprise / production readiness roadmap

What to harden to take this from POC to enterprise, roughly in priority order:

**Infrastructure & scale**
- Externalize the in-memory `_sessions` state (e.g. Redis) or make requests
  stateless, so the backend can autoscale behind a load balancer.
- Move the data snapshots out of JSONB for large data (object storage / a
  normalized schema); reference datasets from dashboards instead of copying.
- Managed backend host with autoscaling + health checks (Cloud Run / ECS / Fly).
- Supabase **Pro** (no auto-pause, backups, higher limits) or self-managed Postgres.

**Auth & access**
- Replace domain-restricted magic link with **org SSO** (Google/Microsoft) via
  Supabase, plus roles (viewer/editor/admin) → this is the "user/session
  management" phase.
- Consider real **RLS policies** (client talks to Postgres directly under the
  user's JWT) instead of the trusted-backend + service-role model, to reduce
  blast radius.
- Secrets in a manager (not `.env`): OpenAI key, service-role key; add rotation.

**Security & governance**
- Security review of the **unauthenticated** `/public/dashboard/{token}` path,
  CORS, and input validation. Consider expiring/revocable share tokens.
- **Data governance:** the sample data is social-listening text with author
  names (PII). Define retention, deletion, and access policies before scaling.
- Rate limiting + per-user OpenAI budgets/quotas; usage logging.

**Delivery & ops**
- CI running `pytest` on PRs; automated deploys for frontend + backend.
- Observability: structured logging, error tracking (e.g. Sentry), uptime + cost
  dashboards.

## 10. Handover checklist for the receiver

- [ ] GitHub repo access (`mia-mcsaatchi/self-service-toolkit`) + this branch.
- [ ] Supabase project ownership/access (project `wyhtzqurehmgejpsdlpf`).
- [ ] OpenAI account + billing.
- [ ] Backend host account (once deployed off local).
- [ ] Read `DEV_GUIDE.md` and this file; run the app locally via `LOCAL_DEV.md`.
- [ ] Run `pytest -q` to confirm a green baseline.
- [ ] Rotate all secrets after transfer (OpenAI key, service-role key).
