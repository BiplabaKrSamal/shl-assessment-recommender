# Deployment Guide

Complete guide for getting the SHL Assessment Recommender live in under 20 minutes.

---

## Prerequisites (get these first, all free)

| What | Where | Takes |
|------|-------|-------|
| GitHub account | github.com | Already have |
| Gemini API key | aistudio.google.com/app/apikey | 2 min |
| Fly.io account | fly.io/app/sign-up | 2 min |
| Fly.io API token | fly.io/user/personal_access_tokens | 1 min |

---

## Option A — Automated (Recommended, 5 minutes)

If you can run Python locally:

```bash
# 1. Enter the project directory
cd shl-recommender/

# 2. Run the one-shot deploy script
python deploy.py \
  --github-token   ghp_XXXXXXXXXXXXXXXXXXXX \
  --gemini-key     AIzaSyXXXXXXXXXXXXXXXXXX \
  --fly-token      fo1_XXXXXXXXXXXXXXXXXXXX \
  --github-username YOUR_GITHUB_USERNAME \
  --repo-name      shl-assessment-recommender \
  --fly-app        shl-assessment-recommender
```

This script:
1. Creates the GitHub repo via API
2. Commits all files and pushes
3. Sets `GEMINI_API_KEY` and `FLY_API_TOKEN` as GitHub Actions secrets
4. Creates the Fly.io app
5. Sets `GEMINI_API_KEY` on Fly.io
6. Deploys (Docker build + model pre-cache ≈ 4 min)
7. Runs health check
8. Prints your live URL

**Done.** Submit `https://shl-assessment-recommender.fly.dev` to SHL.

---

## Option B — Manual GitHub Web Upload (10 minutes)

Use this if you can't run Python locally.

### B1. Create GitHub repo

1. Go to **github.com → "+" → New repository**
2. Name: `shl-assessment-recommender`
3. Description: `Conversational AI agent recommending SHL assessments. FastAPI + BM25/FAISS + Gemini.`
4. **Public** ✓ — check **"Add a README file"** → **Create repository**

---

### B2. Upload files — 4 batches

> **Trick for creating folders in GitHub web:** click **"Add file → Create new file"**, type `folder/filename` in the name box. GitHub creates the folder automatically.

---

**Batch 1 — Root files** (click **Add file → Upload files**, drag all at once):

```
README.md           ARCHITECTURE.md     approach_document.md
Dockerfile          fly.toml            render.yaml
requirements.txt    pytest.ini          conftest.py
deploy.py           Makefile            LICENSE
.env.example        .gitignore
```
Commit message: `feat: SHL Assessment Recommender — initial commit`

---

**Batch 2 — app/ folder**

First: **Add file → Create new file** → type `app/__init__.py` → content: `# SHL Assessment Recommender` → commit: `feat: add app package`

Then, navigate into the `app/` folder → **Add file → Upload files** → drag:
```
models.py    prompts.py    retriever.py
llm_client.py    agent.py    main.py
```
Commit: `feat: core modules — retriever, agent, LLM client, API`

---

**Batch 3 — data/ folder**

**Add file → Create new file** → type `data/catalog.json` → paste entire catalog.json contents → commit: `data: 57 SHL Individual Test Solutions`

---

**Batch 4 — tests/ and scripts/ folders**

```
# Create tests/
Add file → Create new file → tests/__init__.py → commit
Navigate into tests/ → Upload → tests/test_full.py
Commit: "test: 35 tests — schema, retriever, behavior probes, Recall@10"

# Create scripts/
Add file → Create new file → scripts/__init__.py → commit
Navigate into scripts/ → Upload → scrape_catalog.py + evaluate.py + deploy.py
Commit: "scripts: catalog scraper + local evaluator + deploy automation"
```

---

**Batch 5 — GitHub Actions workflow**

**Add file → Create new file** → type `.github/workflows/deploy.yml` → paste full content of deploy.yml → commit: `ci: GitHub Actions — test + Fly.io auto-deploy`

---

### B3. Add GitHub Actions secrets

Go to your repo → **Settings → Secrets and variables → Actions → New repository secret**

| Name | Value |
|------|-------|
| `GEMINI_API_KEY` | your key from aistudio.google.com |
| `FLY_API_TOKEN` | your token from fly.io/user/personal_access_tokens |

---

### B4. Set repo topics (30 seconds)

Repo main page → gear icon next to "About" → Topics:
```
fastapi  python  rag  faiss  bm25  conversational-ai  gemini  nlp  hr-tech  assessment
```

---

## Option C — Deploy to Fly.io (from terminal, after GitHub push)

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Authenticate
flyctl auth login

# Create app (one time)
flyctl apps create shl-assessment-recommender

# Set secret
flyctl secrets set GEMINI_API_KEY=your_key --app shl-assessment-recommender

# Deploy (≈4 minutes — downloads model at build time)
flyctl deploy --remote-only --app shl-assessment-recommender

# Check it's alive
curl https://shl-assessment-recommender.fly.dev/health
# → {"status": "ok"}
```

---

## Option D — Deploy to Render.com (zero CLI, all web)

1. Push to GitHub (via Option B above)
2. Go to **render.com → New → Web Service**
3. Connect your `shl-assessment-recommender` repo
4. Render reads `render.yaml` automatically
5. Add environment variable: `GEMINI_API_KEY = your_key`
6. Click **Deploy**

> ⚠️ Render free tier sleeps after 15 min. The Dockerfile pre-caches the model so cold
> start is ~5s (within the 2-minute `/health` allowance). Fly.io is preferred.

---

## Verify deployment

```bash
# Health check
curl https://YOUR-APP.fly.dev/health
# → {"status": "ok"}

# Test vague query (should clarify, not recommend)
curl -X POST https://YOUR-APP.fly.dev/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I need an assessment"}]}'
# → {"reply": "...(clarifying question)?", "recommendations": [], "end_of_conversation": false}

# Test full conversation
curl -X POST https://YOUR-APP.fly.dev/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
      {"role": "assistant", "content": "{\"reply\":\"What seniority level?\",\"recommendations\":[],\"end_of_conversation\":false}"},
      {"role": "user", "content": "Mid-level, around 4 years"}
    ]
  }'
# → {"reply": "...", "recommendations": [{...}, {...}], "end_of_conversation": false}

# Browse API docs
open https://YOUR-APP.fly.dev/docs
```

---

## CI/CD: What happens on every push

```
git push origin main
        │
        ▼
GitHub Actions (.github/workflows/deploy.yml)
        │
        ├── Job 1: test (ubuntu-latest)
        │     ├── pip install
        │     ├── Cache sentence-transformer model (speeds up future runs)
        │     ├── pytest tests/ -v   ← 35 tests, all mocked
        │     ├── Catalog schema validation
        │     └── Pydantic schema smoke test
        │
        └── Job 2: deploy (only if test passes, only on main)
              ├── flyctl deploy --remote-only
              ├── Health check (up to 5 minutes)
              └── Post live URL to GitHub Actions summary
```

Every merge to main automatically redeploys. Tests protect against regressions.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `/health` returns 503 | Agent still initializing — wait 30s, retry |
| `/chat` timeout | Model loading on first request — Dockerfile cold-start fix should prevent this |
| `GEMINI_API_KEY not set` | Add to Fly secrets: `flyctl secrets set GEMINI_API_KEY=xxx` |
| JSON parse error from LLM | Retry — transient; Gemini JSON mode is reliable but not 100% |
| Fly deploy fails: "app name taken" | Choose a different `--fly-app` name (must be globally unique) |
| GitHub Actions test failing | Check that `data/catalog.json` is committed (not gitignored) |
