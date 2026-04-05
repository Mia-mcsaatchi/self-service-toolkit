# Self-Service Toolkit

AI-powered data enrichment pipeline. Upload a spreadsheet, define output fields with plain-English prompts, and the tool processes each row through GPT-4o-mini — returning structured results you can download as CSV or Excel.

Built by M&C Saatchi Data Team.

---

## Live URLs

| | URL |
|---|---|
| **Frontend** | https://mia-mcsaatchi.github.io/self-service-toolkit |
| **Backend API** | https://self-service-toolkit-production.up.railway.app |
| **API docs** | https://self-service-toolkit-production.up.railway.app/docs |
| **User guide** | https://mia-mcsaatchi.github.io/self-service-toolkit/docs/ |

---

## Architecture

```
index.html          → GitHub Pages (static frontend)
     ↓ fetch()
main.py (FastAPI)   → Railway (persistent backend)
     ↓ POST
OpenAI API          → GPT-4o-mini
```

The frontend parses files in-browser and sends data to the backend. The backend holds state in memory for the session, calls OpenAI, and returns results. The OpenAI API key lives on the server — never sent by the client.

---

## Repo structure

```
self-service-toolkit/
├── index.html          # Frontend — single-page app
├── main.py             # Backend — FastAPI application
├── requirements.txt    # Python dependencies (pinned)
├── Dockerfile          # Used by Railway to build and run
├── .env.example        # Environment variable template
├── .gitignore          # Excludes .env and __pycache__
├── docs/
│   └── index.html      # User guide (served by GitHub Pages)
├── README.md           # This file
└── DEV_GUIDE.md        # Full technical reference for developers
```

---

## Running locally

**Requirements:** Python 3.11+, pip

```bash
# Clone
git clone https://github.com/Mia-mcsaatchi/self-service-toolkit.git
cd self-service-toolkit

# Install dependencies
pip install -r requirements.txt

# Create .env with your OpenAI key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# Start the server
uvicorn main:app --reload
```

Backend runs at `http://127.0.0.1:8000`. Open `index.html` in a browser or visit `/docs` for the interactive API explorer.

---

## Deployment

**Backend → Railway**
- Connect repo at railway.app → auto-detects Dockerfile → deploys on every push to `main`
- Set `OPENAI_API_KEY` in Railway → Variables tab

**Frontend → GitHub Pages**
- Repo Settings → Pages → Source: main branch, / (root)
- Auto-deploys on every push to `main`

---

## Development workflow

```bash
# Make changes locally
# Test at http://127.0.0.1:8000

git add .
git commit -m "description of change"
git push
# Railway and Pages update automatically (~2 min)
```

See `DEV_GUIDE.md` for full technical reference.
