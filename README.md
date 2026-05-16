# SHL Assessment Recommender

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%201.5%20Flash-4285F4?logo=google&logoColor=white)](https://ai.google.dev)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-blue)](https://github.com/facebookresearch/faiss)
[![Tests](https://img.shields.io/badge/Tests-35%20cases-brightgreen)](tests/test_full.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **SHL Labs AI Intern Take-Home Assignment**
>
> A conversational agent that takes a hiring manager from *"I need to hire a Java developer"* to a grounded shortlist of SHL Individual Test Solutions — through multi-turn dialogue, never recommending anything outside the scraped product catalog.

## Demo

```
User:  "I am hiring a Java developer who works with stakeholders"
Agent: "What seniority level are you hiring for?"
User:  "Mid-level, around 4 years of experience"
Agent: "Here are 4 assessments for a mid-level Java developer:
        1. Java 8 (New) [K]  2. OPQ32 [P]  3. Verify Verbal [A]  4. Core Java [K]"
User:  "Actually, also add a coding simulation"
Agent: "Updated shortlist: ... + Coding Simulation – Backend [S]"
```

## Architecture

```
POST /chat (full stateless history)
      │
      ├─ Guard layer (regex, no LLM): injection / off-topic / turn-cap
      ├─ Hybrid Retriever: BM25 (lexical) + FAISS (semantic) → RRF fusion
      ├─ Prompt Builder: full 57-item catalog in system + top-10 hint in user msg
      └─ Gemini 1.5 Flash (temp=0.1, JSON mode) → Parser → validated ChatResponse
```

Full diagram → [ARCHITECTURE.md](ARCHITECTURE.md)

## Scoring Rubric Coverage

| Check | Implementation |
|-------|---------------|
| **Hard eval:** Schema compliance | Pydantic v2 models + strict JSON parser |
| **Hard eval:** Catalog URLs only | URL allowlist (`https://www.shl.com` prefix) |
| **Hard eval:** Turn cap ≤ 8 | Force-recommend injection at turn 8 |
| **Recall@10** | Hybrid BM25 + FAISS with RRF fusion |
| **Probe:** Refuses off-topic | Guard layer fires before LLM call |
| **Probe:** No recs on vague turn-1 | Vague-query heuristic returns clarification |
| **Probe:** Honors mid-conv edits | Full history concatenation for retrieval query |
| **Probe:** Zero hallucination | Full catalog in system prompt prevents invented URLs |

## Project Structure

```
shl-recommender/
├── app/
│   ├── main.py          # FastAPI — GET /health, POST /chat
│   ├── agent.py         # Orchestrator: guards, intent routing, turn management
│   ├── retriever.py     # Hybrid BM25 + FAISS + RRF fusion
│   ├── prompts.py       # System prompt, catalog injection, retrieval hints
│   ├── llm_client.py    # Gemini client, retry, response parser/validator
│   └── models.py        # Pydantic schema (non-negotiable per spec)
├── data/
│   └── catalog.json     # 57 SHL Individual Test Solutions (scraped + curated)
├── scripts/
│   ├── scrape_catalog.py  # Scrapes shl.com product catalog
│   └── evaluate.py        # Local Recall@10 harness (mirrors SHL evaluator)
├── tests/
│   └── test_full.py     # 35 tests: schema, retriever, behavior probes, integration
├── ARCHITECTURE.md      # System design with ASCII flow diagrams
├── approach_document.md # 2-page design writeup (submission document)
├── Dockerfile           # Multi-stage; model pre-downloaded at build time
├── fly.toml             # Fly.io — no sleep, zero cold-start risk (recommended)
└── render.yaml          # Render.com — free tier fallback
```

## Quick Start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add GEMINI_API_KEY=your_key
uvicorn app.main:app --reload --port 8000
```

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I need to hire a data scientist"}]}'
```

## Tests

```bash
pytest tests/ -v                     # 35 tests, mocked LLM, ~10s
pytest tests/ -v -k "TestRecallAtK"  # Recall@10 self-evaluation
pytest tests/ -v --live              # live LLM tests (needs GEMINI_API_KEY)
```

## Local Evaluation

```bash
python scripts/evaluate.py --local              # direct agent, no HTTP
python scripts/evaluate.py --endpoint http://localhost:8000  # against server
```

## Deploy

**Fly.io** (recommended — no cold-start):
```bash
fly launch --name shl-recommender --no-deploy
fly secrets set GEMINI_API_KEY=your_key
fly deploy
```

**Render.com:** connect repo → Docker → set `GEMINI_API_KEY`. Uses `render.yaml`.

## Response Schema

```json
{
  "reply": "Here are 4 assessments for a mid-level Java developer...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

Full design writeup → [approach_document.md](approach_document.md)
