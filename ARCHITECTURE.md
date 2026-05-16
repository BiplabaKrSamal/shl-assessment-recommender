# System Architecture

## Request Flow

```
POST /chat  ←  { "messages": [...full history...] }
      │
      ▼
┌─────────────────────────────────────────────────────┐
│              Agent Orchestrator                      │
│                  app/agent.py                        │
│                                                      │
│  1. Guard Layer (regex, microseconds, no LLM call)   │
│     ├─ Prompt injection detected?  → refuse          │
│     ├─ Off-topic query?            → refuse          │
│     └─ Turn ≥ 8?                   → force-recommend │
│                                                      │
│  2. Intent Router                                    │
│     ├─ Vague turn-1?    → clarify (no LLM call)      │
│     ├─ Comparison?      → named lookup + semantic    │
│     └─ Normal / refine  → full retrieval pipeline    │
│                                                      │
│  3. Retrieval Query Builder                          │
│     └─ ALL user turns concatenated (recent ×2)       │
│        so "add personality" still carries            │
│        context from "Java developer" 3 turns ago     │
└─────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────┐
│              Hybrid Retriever                        │
│               app/retriever.py                       │
│                                                      │
│  BM25 (lexical, pure Python)                         │
│  ├─ Exact product name recall ("OPQ32", "Java 8")    │
│  └─ Weight: 0.4                                      │
│                                                      │
│  FAISS IndexFlatIP (semantic)                        │
│  ├─ Model: all-MiniLM-L6-v2 (22MB, CPU, ~80ms)       │
│  ├─ Intent queries ("handle workplace pressure")     │
│  └─ Weight: 0.6                                      │
│                                                      │
│  RRF Fusion (k=60)                                   │
│  └─ score = 0.4/(60+rank_bm25) + 0.6/(60+rank_sem)  │
│     No score normalization needed — rank-based only  │
└─────────────────────────────────────────────────────┘
      │ top-10 results
      ▼
┌─────────────────────────────────────────────────────┐
│              Prompt Builder                          │
│               app/prompts.py                         │
│                                                      │
│  System prompt contains:                             │
│  ├─ Behavioral contract (4 rules, 4 behaviors)       │
│  ├─ Output schema (JSON, non-negotiable)             │
│  └─ Full 57-item catalog ← URL ground truth          │
│     (prevents hallucinated URLs)                     │
│                                                      │
│  User message augmented with:                        │
│  └─ [RETRIEVAL HINT: top-10 items, detailed]         │
│     (focuses attention without reducing ground truth)│
└─────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────┐
│              LLM Client                              │
│             app/llm_client.py                        │
│                                                      │
│  Model:  Gemini 1.5 Flash                            │
│  Temp:   0.1  (near-deterministic JSON output)       │
│  Mode:   response_mime_type="application/json"       │
│  Retry:  3× exponential backoff on quota/rate limit  │
│  Fallback: gemini-1.5-flash-8b on quota exhaustion   │
└─────────────────────────────────────────────────────┘
      │ raw JSON string
      ▼
┌─────────────────────────────────────────────────────┐
│              Response Parser & Validator             │
│             app/llm_client.py                        │
│                                                      │
│  ✓ Strip markdown fences (```json ... ```)           │
│  ✓ Parse JSON with regex fallback                    │
│  ✓ Filter: only https://www.shl.com URLs             │
│  ✓ Validate test_type ∈ {A,B,C,D,E,K,M,P,S}         │
│  ✓ Deduplicate by name                               │
│  ✓ Cap recommendations at 10                         │
│  ✓ Safe fallback on any parse failure                │
└─────────────────────────────────────────────────────┘
      │
      ▼
POST /chat response  ←  { reply, recommendations[], end_of_conversation }
```

---

## Stateless Design

Every `POST /chat` carries the **full conversation history**.
No database. No session store. No sticky routing needed.
The service can restart mid-conversation without data loss — the
next request brings everything back.

```
Client                          Service
  │                               │
  │── POST /chat {turn 1} ───────►│ processes, returns reply
  │◄─ {reply, recs:[], eoc:false}─│
  │                               │  ← service can restart here
  │── POST /chat {turn 1+2} ─────►│ full history re-sent
  │◄─ {reply, recs:[...]} ────────│
```

---

## Cold-Start Strategy

**Problem:** Free-tier hosts (Render) sleep after 15min inactivity.
The evaluator allows 2 minutes for `/health` to respond, then 30s per `/chat` call.
Downloading `all-MiniLM-L6-v2` (~22MB) at runtime takes 30–60s — fatal.

**Solution:** Model downloaded at **Docker build time** into `/app/model_cache`.
Cold start only needs to: load model from disk (~1s) + rebuild FAISS index (~3s) = **~5s total**.

```dockerfile
RUN python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', 
                    cache_folder='/app/model_cache')"
```

**Fly.io alternative:** `auto_stop_machines = false` — machine never sleeps.
Zero cold-start risk. Preferred for submission.

---

## Scoring Rubric Alignment

| Evaluator Check | Where it's handled |
|----------------|-------------------|
| Schema compliance (every response) | `app/models.py` Pydantic + `app/llm_client.py` parser |
| Catalog URLs only | URL prefix filter in parser + full catalog in system prompt |
| Turn cap ≤ 8 honored | `app/agent.py` turn-8 force-recommend path |
| Recall@10 on recommendations | `app/retriever.py` hybrid BM25+FAISS |
| Refuses off-topic | `app/agent.py` guard layer (before LLM call) |
| No recs on vague turn-1 | `app/agent.py` vague-query heuristic |
| Honors mid-conversation edits | `_build_retrieval_query()` full-history concatenation |
| Zero hallucination rate | Full catalog in system prompt, URL allowlist enforcement |
