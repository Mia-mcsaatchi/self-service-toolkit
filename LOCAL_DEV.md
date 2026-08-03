# Running locally with real login (VS Code)

This is the "develop on my machine, but test the real Supabase magic-link login"
setup. Everything runs on your laptop — the backend in VS Code, the frontend
served locally, and Supabase handling sign-in.

> Colleagues can't reach a backend running on your laptop — this mode is for
> **you** to develop and test. Hosting it for the team is a later step.

---

## One-time setup

### 1. Frontend: anon key
`index.html` already has your Supabase **Project URL**. It still needs the
**anon / public key** pasted in next to it (ask Claude to add it, or edit the
`SUPABASE_ANON_KEY` line near the top of the `<script>`). Login stays disabled
until it's filled in.

### 2. Backend: `.env`
In the project folder, copy the template and fill it in:

```bash
cp .env.example .env
```

Set these in `.env` (this file is gitignored — the secret never leaves your machine):

```
OPENAI_API_KEY=sk-...                      # your OpenAI key
SUPABASE_JWT_SECRET=...                     # Supabase → Settings → API → JWT Secret
ALLOWED_EMAIL_DOMAIN=mcsaatchi.com
# leave AUTH_DISABLED unset (or false) so real login is used
```

### 3. Install dependencies (first time only)

```bash
pip install -r requirements.txt
```

### 4. Supabase: allow your local URL to receive the login link
Supabase dashboard → **Authentication → URL Configuration**:
- **Site URL:** `http://127.0.0.1:5500`  *(use whatever port you serve the frontend on — see step B below)*
- **Redirect URLs:** add `http://127.0.0.1:5500/**` and `http://localhost:5500/**`

The magic link won't sign you in unless the page's URL is on this list.

---

## Every time you run it

**A. Start the backend** (VS Code terminal):

```bash
uvicorn main:app --reload --port 8000
```

Leave it running. It's live at `http://127.0.0.1:8000` (check `/api/health`).

**B. Serve the frontend** — it must be served over http, not opened as a
`file://` path, or login and CORS won't work. Two easy options:

- **VS Code Live Server extension:** right-click `index.html` → *Open with Live
  Server* (usually `http://127.0.0.1:5500`).
- **Or a one-line static server** in a second terminal:
  ```bash
  python -m http.server 5500
  ```
  then open `http://127.0.0.1:5500/index.html`.

The frontend auto-detects it's running locally and talks to your local backend
on port 8000 — no code change needed to switch between local and hosted.

**C. Sign in:** enter your `@mcsaatchi.com` email → **Send login link** → open
the email on the same machine → you land back in the app, signed in.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Login button says "not configured" | The anon key isn't set in `index.html` yet. |
| "Auth not configured: set SUPABASE_JWT_SECRET" | `.env` missing the JWT secret, or backend not restarted after editing `.env`. |
| Clicking the email link doesn't sign you in | The page URL/port isn't in Supabase → URL Configuration → Redirect URLs. |
| Browser console shows a CORS error | Serve the frontend over http (Live Server / http.server), not `file://`. |
| "backend offline" pill | The `uvicorn` backend isn't running, or is on a different port than 8000. |

---

## Just want to see it run, skipping login?

Set `AUTH_DISABLED=true` in `.env` and restart the backend. Every request is
then a single shared `local-dev` user and you don't need Supabase at all. Never
use this once the backend is hosted.
