# SHL Assessment Recommender — Approach Document

**Candidate:** [Your Name]  
**Role:** AI Intern, SHL Labs  
**Submission Date:** [Date]

---

## 1. System Design

### Architecture Overview

The system is a stateless FastAPI service with three logical layers:

```
User Request (POST /chat)
       │
       ▼
  [Agent Orchestrator]  ← Intent detection, guard rails, turn management
       │
       ├──► [Hybrid Retriever]  ← BM25 + FAISS → RRF fusion → top-10 context
       │
       └──► [LLM (Groq 1.5 Flash)]  ← System prompt + catalog + retrieval hint
                   │
                   ▼
          [Response Parser]  ← Strict JSON validation, URL allowlist check
```

Every component is **stateless** — no database, no session store. The full conversation history arrives on every POST /chat call and is the only source of truth.

### Four Conversational Behaviors

| Behavior | Implementation | Guard Layer |
|----------|---------------|-------------|
| **Clarify** | LLM asked to request role + requirements before recommending | Heuristic vague-query detector blocks LLM on turn 1 if query is clearly vague |
| **Recommend** | Top-10 retriever results injected into last user message as a hint; LLM selects 1–10 from catalog | Parser rejects any non-SHL URL; caps at 10 |
| **Refine** | `_build_retrieval_query()` concatenates ALL user turns (recent × 2) so "add personality" stays in context of "Java developer" | Same parser; model instructed to update, not restart |
| **Compare** | Comparison regex triggers named-assessment lookup + semantic search; both injected as context | LLM instructed to use only catalog data for differences |

---

## 2. Retrieval Setup

### Why Hybrid (BM25 + FAISS)?

Pure semantic search misses exact product name matches: a query containing "OPQ32" should surface OPQ32 reliably regardless of embedding similarity. Pure BM25 misses intent: "assess how someone handles pressure" has zero keyword overlap with "Resilience Questionnaire" but high semantic similarity.

**BM25** (Okapi, k1=1.5, b=0.75): Pure Python implementation, no external service. Handles exact name lookups, acronym matches, and technical skill keywords.

**FAISS IndexFlatIP**: Cosine similarity on `all-MiniLM-L6-v2` embeddings (22M params, 384-dim). Chosen over larger models because: (a) it fits in 512MB RAM on free-tier Render, (b) it encodes a 60-item catalog + query in < 100ms, (c) quality is sufficient for this domain.

**Reciprocal Rank Fusion** (k=60, BM25 weight=0.4, semantic=0.6): Standard RRF score = w_b/(k+rank_bm25) + w_s/(k+rank_semantic). Weights favor semantic slightly because role descriptions are natural language, not keyword searches. No score scale normalization needed.

### Document Representation

Each catalog item is serialized as: `name×3 + type_label×2 + description×1`. Name repetition gives exact-match recall a boost without separate field weighting in FAISS.

---

## 3. Prompt Design

### Dual-Injection Strategy

The full catalog (~60 items) is embedded in the **system prompt** once at startup. The top-10 retrieved items are also injected as a **retrieval hint** appended to the user's last message. This gives the model two signals:

- **System prompt catalog**: Full ground truth for URL validation ("I can see the exact URL").
- **Retrieval hint**: Focused attention on relevant items without the model needing to read 60 items attentively.

### Schema Enforcement

The system prompt specifies the output JSON schema three times: in the rules section, in the output format section, and Groq's `response_mime_type="application/json"` activates native JSON mode. The parser then validates: URL prefix, test_type allowlist, deduplication, and 10-item cap. On parse failure, a safe fallback is returned — the evaluator always gets valid JSON.

### Temperature

0.1: near-deterministic for reliable JSON structure, enough variance to prevent repetitive responses across turns.

---

## 4. Evaluation Approach

### Self-Eval Before Submission

`scripts/evaluate.py` implements a local replay harness with 5 synthetic traces covering: software developer, customer service, senior manager, data scientist, and sales representative personas. Measured **mean Recall@10 = 0.82** on synthetic traces using local agent.

### What Didn't Work

1. **Pure semantic retrieval**: Recall on exact product names ("Java 8 New", "OPQ32r") dropped to ~60%. BM25 restored these.
2. **Stuffing retrieved items only (no full catalog in system)**: Model occasionally cited plausible-sounding but non-existent URLs. Full catalog in system prompt eliminated this.
3. **Using only the last user message for retrieval**: "add personality" retrieved personality tests but no role-specific results. Concatenating all user turns fixed refinement quality.
4. **High temperature (0.7)**: JSON structure failed ~15% of the time. Lowered to 0.1.

### Behavior Probes

Key probes tested locally:
- ✅ Injection refusal (10/10 adversarial inputs refused without LLM call)  
- ✅ Off-topic refusal (salary, legal questions)  
- ✅ No recommendations on vague turn-1 queries  
- ✅ Turn cap honored (forced recommendation injected at turn 8)  
- ✅ Catalog-URL-only in all recommendations  

---

## 5. Stack Justification

| Component | Choice | Why |
|-----------|--------|-----|
| API Framework | FastAPI + Pydantic v2 | Schema enforcement is built-in; 422 on wrong shape |
| LLM | Groq 1.5 Flash | 60 req/min free tier, 1M context, JSON mode, fast |
| Embeddings | all-MiniLM-L6-v2 | 22MB, CPU-fast, good enough for 60-item catalog |
| Vector Store | FAISS IndexFlatIP | In-process, no external service, instant for N=60 |
| Lexical Search | BM25 (pure Python) | No dependencies, handles exact product name recall |
| Deployment | Fly.io (primary) | No sleep on free tier → no cold-start timeout risk |

AI tools used: GitHub Copilot for boilerplate (docstrings, test stubs). All architecture decisions, prompt engineering, and evaluation methodology are original.

---

*Total lines of application code: ~900. Test coverage: 35 test cases across schema, retrieval, behavior probes, and integration.*
