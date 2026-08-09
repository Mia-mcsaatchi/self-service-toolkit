# Supabase setup — Phase A (auth + per-user isolation)

This is the one-time setup to turn the toolkit from a single shared session into
a proper multi-user tool where each colleague signs in and only sees their own
data. You can do all of this yourself — no data lead needed for the POC.

**Time: ~20 minutes. You will not need to paste any secret into a chat.**

---

## What you're setting up

```
Browser (index.html)  ──sign in──▶  Supabase Auth  ──sends a magic link──▶  your inbox
        │                                                                        │
        │  gets a JWT after you click the link  ◀────────────────────────────────┘
        ▼
Backend (main.py)  ──verifies the JWT──▶  knows which colleague is calling,
                                          keeps each person's data separate
```

Login is **magic link**: a colleague types their `@mcsaatchi.com` email, gets a
one-click link by email, and they're in. No passwords.

---

## Step 1 — Create the Supabase project

1. Go to <https://supabase.com> → sign in → **New project**.
2. Name it (e.g. `self-service-toolkit`), pick a region close to the UK
   (e.g. London / `eu-west-2`), set a database password (save it in your
   password manager — you won't need it for this POC).
3. Wait ~2 min for it to provision.

## Step 2 — Collect the three values you need

In the Supabase dashboard → **Project Settings → API**:

| Value | Where | Used by | Secret? |
|---|---|---|---|
| **Project URL** | "Project URL" | frontend | No — public |
| **anon public key** | "Project API keys → anon / public" | frontend | No — public |
| **JWT Secret** | "JWT Settings → JWT Secret" | backend | **YES — keep secret** |

> If your project only shows the new "JWT signing keys" (asymmetric) and no
> "JWT Secret", switch on the **Legacy JWT secret (HS256)** option, or tell me
> and I'll switch the backend to the JWKS verification method instead.

## Step 3 — Restrict sign-ups to M&C Saatchi

Dashboard → **Authentication → Providers → Email**: make sure **Email** is
enabled (magic link works out of the box).

Then lock it to colleagues only. Two layers, both already handled:

- **Backend** already rejects any non-`@mcsaatchi.com` token (`ALLOWED_EMAIL_DOMAIN`).
- **Frontend** already refuses to send a link to a non-`@mcsaatchi.com` address.

(Optional, strongest) Dashboard → **Authentication → URL Configuration** and
**Auth settings** — if your plan supports it, add `mcsaatchi.com` to allowed
email domains so Supabase itself blocks other domains at the source.

## Step 4 — Allow the login link to redirect back to the app

Dashboard → **Authentication → URL Configuration**:

- **Site URL:** `https://mia-mcsaatchi.github.io/self-service-toolkit/`
- **Redirect URLs:** add both
  - `https://mia-mcsaatchi.github.io/self-service-toolkit/`
  - `http://localhost:8000/` (for local testing, if you open index.html locally)

The magic link won't complete sign-in unless the return URL is on this list.

## Step 5 — Put the two public values into the frontend

Edit `index.html`, near the top of the `<script>` block:

```js
const SUPABASE_URL      = 'https://YOUR-PROJECT.supabase.co';   // ← your Project URL
const SUPABASE_ANON_KEY = 'YOUR-SUPABASE-ANON-KEY';             // ← your anon public key
```

These two are safe to commit — the anon key is designed to live in the browser.

## Step 6 — Put the JWT secret into the backend (never the frontend)

Wherever the backend runs (Render/Fly/etc.), set these environment variables:

| Variable | Value |
|---|---|
| `OPENAI_API_KEY` | your OpenAI key (unchanged) |
| `SUPABASE_JWT_SECRET` | the **JWT Secret** from Step 2 |
| `ALLOWED_EMAIL_DOMAIN` | `mcsaatchi.com` (default; leave as-is) |

Do **not** set `AUTH_DISABLED` in production. See `.env.example` for the full list.

## Step 7 — Test it

1. Redeploy the backend (so it picks up `SUPABASE_JWT_SECRET`).
2. Open the frontend, enter your `@mcsaatchi.com` email, click **Send login link**.
3. Open the email, click the link → you land back in the app, signed in
   (your email shows in the top-right pill; click it to sign out).
4. Ask a colleague to do the same — you should each only see your own uploads.

---

## Local development without Supabase

To run the backend locally without wiring up Supabase, set `AUTH_DISABLED=true`
in your `.env`. Every request is then treated as a single shared `local-dev`
user. **Never** enable this on the deployed backend.

---

## Phase B — saving datasets (optional)

Lets each user save tagged results to their account and reload them later, so a
restart (or a Supabase unpause) no longer wipes their work.

1. **Create the table.** Supabase dashboard → **SQL Editor → New query**, paste
   the contents of `supabase/schema.sql`, and **Run**.
2. **Add the service-role key to the backend.** Supabase → **Project Settings →
   API → `service_role`** (secret). Put it in your backend `.env`:
   ```
   SUPABASE_SERVICE_ROLE_KEY=eyJ... (the service_role secret)
   ```
   This is a **server secret** — never commit it, never put it in the frontend.
   The backend uses it to read/write the `datasets` table and scopes every query
   by the signed-in user's id, so users only ever see their own data.
3. **Restart the backend.** A "📁 Your saved datasets" panel appears on Step 1,
   and a "💾 Save to my datasets" button appears after you run the pipeline.

If you skip this, the save/load feature is simply hidden and everything else
works exactly as before.

## What's next

- **Phase C** — save a tagged dataset + its chosen dimension as a reusable,
  revisitable dashboard.
