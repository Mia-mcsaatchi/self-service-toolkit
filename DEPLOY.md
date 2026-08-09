# Deploying the Self-Service Toolkit (free-tier PoC)

This is a two-part app:

- **Frontend** — a single static `index.html` (+ `docs/`). Hosted on **GitHub Pages** (free).
- **Backend** — a FastAPI app (`main.py`) in a Docker container. Hosted on **Render** free tier (free; sleeps after 15 min idle, ~50s cold start on the next hit).

You also need a **Supabase** project (free tier) for auth + storage — see [`SUPABASE_SETUP.md`](SUPABASE_SETUP.md).

> **Why Render + Pages?** It's the fastest path that stays free, and it reuses the
> `Dockerfile` already in this repo. When you move to AWS later, the same image lifts
> straight over (ECS/App Runner) — only the URL in the frontend changes.

---

## Prerequisites (one-time)

1. A **Supabase** project. Run [`supabase/schema.sql`](supabase/schema.sql) in the SQL Editor to create the `datasets`, `dashboards`, and `dashboard_shares` tables.
2. In Supabase → **Authentication → Providers → Email**: make sure **Email** is enabled (it is by default). This is what powers both magic links and email+password.
3. An **OpenAI API key** (for the enrichment + analysis calls).
4. From Supabase → **Project Settings → API**, note: `Project URL`, `anon`/publishable key, `service_role` key, and the **JWT secret** (Settings → API → JWT Settings).

---

## Part 1 — Backend on Render

1. Push this repo to GitHub (already done).
2. Go to [render.com](https://render.com) → **New → Web Service** → connect this GitHub repo.
3. Render auto-detects the `Dockerfile`. Settings:
   - **Runtime:** Docker
   - **Instance type:** Free
   - **Health check path:** `/api/health`
4. Add **Environment variables** (Render → the service → Environment):

   | Key | Value |
   |-----|-------|
   | `SUPABASE_URL` | your Supabase Project URL |
   | `SUPABASE_SERVICE_ROLE_KEY` | your `service_role` key (secret — backend only) |
   | `SUPABASE_JWT_SECRET` | your Supabase JWT secret |
   | `OPENAI_API_KEY` | your OpenAI key |
   | `ALLOWED_EMAIL_DOMAIN` | `mcsaatchi.com` |

   Do **not** set `AUTH_DISABLED` in production.
5. Deploy. When it's live, note the URL, e.g. `https://self-service-toolkit.onrender.com`.
6. Verify: open `https://<your-render-url>/api/health` — you should get a JSON `{"status": "ok", ...}`.

### Keep it warm (optional)
Render free sleeps after 15 min idle. This repo has a daily keep-alive GitHub Action
(originally for Supabase). You can add a step that pings `https://<your-render-url>/api/health`
every ~10 min to reduce cold starts. Note: Render free may still nap; for a demo that must
be instant, upgrade to the $7/mo instance or switch to Cloud Run.

---

## Part 2 — Frontend on GitHub Pages

1. Edit **`index.html`**:
   - Set `PROD_BACKEND` (near the top `<script>`) to your Render URL, **no trailing slash**.
   - Confirm `SUPABASE_URL` and `SUPABASE_ANON_KEY` are your project's values (these are public/safe in the browser).
2. Edit **`main.py`** CORS (`allow_origins`) to include your Pages origin if it differs from `https://mia-mcsaatchi.github.io`. (CORS matches the origin only — the `/self-service-toolkit/` path doesn't matter.)
3. Commit + push.
4. GitHub repo → **Settings → Pages** → Source: **Deploy from a branch** → Branch: `main`, folder: `/ (root)` → Save.
5. Wait ~1 min. Your app is at **`https://mia-mcsaatchi.github.io/self-service-toolkit/`** and the guide at `…/self-service-toolkit/docs/`.

---

## The link to send colleagues

**`https://mia-mcsaatchi.github.io/self-service-toolkit/`**

They sign in with their `@mcsaatchi.com` email — magic link or email+password (self sign-up). First password sign-up sends a one-off confirmation email.

---

## Checklist

- [ ] `supabase/schema.sql` run in Supabase
- [ ] Render service live; `/api/health` returns ok
- [ ] All 5 backend env vars set on Render
- [ ] `PROD_BACKEND` in `index.html` = Render URL
- [ ] CORS `allow_origins` includes the Pages origin
- [ ] GitHub Pages enabled on `main` / root
- [ ] Signed in from the Pages URL end-to-end (upload → enrich → dashboard → publish link)

---

## Moving to AWS later

The `Dockerfile` is host-agnostic and binds to `$PORT`. On AWS, **App Runner** (or ECS
Fargate) can build from this repo/image directly; set the same 5 env vars, point
`PROD_BACKEND` at the new URL, and add the Pages origin to CORS. Nothing else changes.
