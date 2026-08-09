# Roadmap / Backlog

**Status:** Internal tool, in PoC. Live for the Data team to trial.
**Next gate:** Gather feedback from colleagues before committing to any of the below.
**Longer-term opportunity:** Cross-sell to other teams once it earns trust internally.

> Nothing here is scheduled yet — this is a parked backlog from a product review.
> Re-prioritise after colleague feedback comes in.

---

## The honest read (product review)

A polished PoC with a real end-to-end loop (upload → AI enrich → dashboard → share → deck).
To become a *product* (not just a feature), the gaps are in **trust, reliability, and cost** —
not in adding more flash. Do the P0s first so later "wow" features land on solid ground.

### Top 3 blockers to "productable"
1. **Data governance** — client data is sent to OpenAI via a single shared backend key.
   Needs a data-handling stance (DPA? is client data allowed to leave the tenant?),
   PII detection/redaction, and per-workspace isolation. Biggest blocker for agency/client data.
2. **AI accuracy is unverifiable** — no way to review, correct, or measure tagging quality.
   Output can't be fully trusted in a client deck without a human spot-check.
3. **Reliability & cost on free-tier defaults** — Render cold starts (~50s) read as "broken";
   no cost ceiling, per-user quota, spend visibility, or progress/resumability on big runs.

---

## Backlog (prioritised)

### P0 — trust & reliability (do before going wide)
- [ ] PII / confidentiality guardrails + written data-handling policy
- [ ] Tag review & correct — spot-check a sample, edit tags, re-run (credibility unlock)
- [ ] Cost estimate before a run + per-user quota + enrichment progress bar
- [ ] Move off free-tier defaults for reliability (warm backend / paid instance or Cloud Run)

### P1 — depth of value
- [ ] Analysis templates — one-click Brand Health / Campaign Tracker / Crisis (use-cases already in docs)
- [ ] Drill-down — click a chart segment → see the underlying rows / verbatims
- [ ] Scheduled / recurring reports emailed as a deck

### P2 — wow / differentiation
- [ ] Ask-your-data — conversational analytics ("what changed vs last month?") → chart (builds on /api/interpret)
- [ ] Auto-insights — proactively surface spikes / shifts / outliers
- [ ] Branded, client-ready PPTX using an M&C Saatchi master template

### Productization enablers (needed for cross-team / cross-sell)
- [ ] Workspaces + roles (admin / editor / viewer); today it's single-domain, no teams
- [ ] Bring-your-own OpenAI API key per user/team (removes shared-key cost & isolation risk)
- [ ] Reliable transactional email (custom SMTP e.g. Resend/SendGrid) — free-tier Supabase email is rate-limited
- [ ] Observability — error tracking + basic usage analytics
- [ ] External collaborators via invite tokens (currently @mcsaatchi.com only)

---

## If only three things
1. **Tag review + correction** — turns "AI guessed" into "analyst-verified"; the trust unlock. Self-contained.
2. **Analysis templates** — collapses best use-cases into one click; fastest "neat → weekly habit".
3. **Ask-your-data chat** — the genuine wow; the demo that wins budget.
