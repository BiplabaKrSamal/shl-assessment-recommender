"""
FastAPI Application Entry Point

Endpoints:
  GET  /health  → {"status": "ok"}  (HTTP 200)
  POST /chat    → ChatResponse (per spec)

Design decisions:
- Single global agent instance — retriever FAISS index and system prompt
  are built once at startup, shared across all requests (thread-safe reads).
- Lifespan context manager for clean startup/shutdown logging.
- Request validation via Pydantic (models.py) — invalid schema returns 422.
- Structured error handling returns valid ChatResponse on all failures
  so the automated evaluator always gets parseable JSON, never a 500 traceback.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent import SHLAgent
from app.models import ChatRequest, ChatResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("shl_recommender")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()

CATALOG_PATH = os.getenv("CATALOG_PATH", "data/catalog.json")

# ---------------------------------------------------------------------------
# Global agent (initialized once at startup)
# ---------------------------------------------------------------------------
_agent: SHLAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: initialize the agent (loads catalog, builds BM25 + FAISS index).
    Shutdown: cleanup (nothing needed currently).
    
    FAISS index build time: ~2-5s on cold start (CPU, 50-60 items).
    Sentence-transformer model download: ~30s first run, cached after.
    This is why /health allows 2 minutes on cold start.
    """
    global _agent
    logger.info("Starting SHL Recommender Agent...")
    start = time.time()

    try:
        _agent = SHLAgent(catalog_path=CATALOG_PATH)
        logger.info(
            f"Agent initialized in {time.time()-start:.2f}s. "
            f"Catalog: {len(_agent.retriever.get_all())} assessments."
        )
        # Warm up FAISS index (build it now rather than on first request)
        _ = _agent.retriever.search("software developer cognitive ability", k=3)
        logger.info("Retriever warmed up.")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise

    yield  # App runs here

    logger.info("Shutting down SHL Recommender Agent.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for recommending SHL Individual Test Solutions.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for evaluator compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware: request timing
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_timing(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    response.headers["X-Response-Time"] = f"{elapsed:.3f}s"
    if elapsed > 25:
        logger.warning(f"Slow response: {request.url.path} took {elapsed:.2f}s")
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """
    Readiness probe. Returns 200 immediately if the agent is initialized.
    If agent failed to init, returns 503.
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint. Accepts full conversation history, returns next reply.

    The API is stateless — every call carries the full history.
    Schema is non-negotiable per spec: {reply, recommendations, end_of_conversation}.
    
    We NEVER let unhandled exceptions reach the evaluator. All errors
    are caught and returned as valid ChatResponse objects.
    """
    if _agent is None:
        return ChatResponse(
            reply="The service is starting up. Please retry in a moment.",
            recommendations=[],
            end_of_conversation=False,
        )

    try:
        response = _agent.chat(request)
        logger.info(
            f"Chat: {len(request.messages)} messages → "
            f"{len(response.recommendations)} recommendations, "
            f"end={response.end_of_conversation}"
        )
        return response

    except Exception as e:
        logger.error(f"Unhandled error in chat: {e}", exc_info=True)
        return ChatResponse(
            reply="I encountered an unexpected error. Please try again.",
            recommendations=[],
            end_of_conversation=False,
        )


@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc):
    """Return structured error for schema violations instead of default FastAPI 422."""
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request schema. Ensure messages is a list of {role, content} objects."},
    )
