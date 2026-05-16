"""
LLM Client — Groq API (OpenAI-compatible).

WHY GROQ:
- Explicitly listed in SHL assignment as approved free resource
- Free tier: 30 req/min, 6000 req/day — far exceeds evaluator needs
- LPU inference: ~300 tokens/sec (10x faster than Gemini)
- OpenAI-compatible API: simple, stable, no SDK version drama
- Model: llama-3.3-70b-versatile — excellent JSON structured output

No SDK needed — plain httpx POST to api.groq.com/openai/v1/chat/completions
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.models import ChatResponse, Recommendation

logger = logging.getLogger("shl_recommender")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"

_SHL = "https://www.shl.com"
_VALID = set("ABCDEKPMS")


def _get_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY not set in environment variables.")
    return key


# ── API call ──────────────────────────────────────────────────────────────
@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=6),
    reraise=True,
)
def _call(system_prompt: str, messages: list[dict], model: str) -> str:
    payload = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 1200,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            *messages,
        ],
    }
    with httpx.Client(timeout=25.0) as client:
        resp = client.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {_get_key()}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    if resp.status_code == 429:
        raise Exception(f"429 rate limit: {resp.text[:200]}")
    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ── Parser ────────────────────────────────────────────────────────────────
def _parse(raw: str, valid_urls: set[str]) -> ChatResponse:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned.strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        try:
            data = json.loads(m.group()) if m else {}
        except Exception:
            return _fallback("I had trouble formatting my response. Please try again.")

    reply = str(data.get("reply", "")).strip() or "Could you tell me more about the role?"
    raw_recs = data.get("recommendations", [])
    if not isinstance(raw_recs, list):
        raw_recs = []

    validated, seen = [], set()
    for rec in raw_recs:
        if not isinstance(rec, dict):
            continue
        name  = str(rec.get("name", "")).strip()
        url   = str(rec.get("url", "")).strip()
        ttype = str(rec.get("test_type", "A")).strip().upper()
        if not name or name.lower() in seen:
            continue
        if not url.startswith(_SHL):
            continue
        if ttype not in _VALID:
            ttype = "A"
        seen.add(name.lower())
        validated.append(Recommendation(name=name, url=url, test_type=ttype))

    return ChatResponse(
        reply=reply,
        recommendations=validated[:10],
        end_of_conversation=bool(data.get("end_of_conversation", False)),
    )


def _fallback(msg: str) -> ChatResponse:
    return ChatResponse(reply=msg, recommendations=[], end_of_conversation=False)


# ── Public ────────────────────────────────────────────────────────────────
def generate_response(
    system_prompt: str,
    messages: list[dict],
    valid_catalog_urls: set[str],
    timeout_seconds: int = 25,
) -> ChatResponse:
    start = time.time()
    try:
        raw = _call(system_prompt, messages, PRIMARY_MODEL)
        logger.info(f"Groq call OK ({PRIMARY_MODEL}) in {time.time()-start:.1f}s")
    except Exception as e:
        logger.warning(f"Groq primary failed ({PRIMARY_MODEL}): {e}")
        try:
            raw = _call(system_prompt, messages, FALLBACK_MODEL)
            logger.info(f"Groq fallback OK ({FALLBACK_MODEL})")
        except Exception as e2:
            logger.error(f"Groq fallback failed: {e2}")
            return _fallback("I'm temporarily unavailable. Please retry in a moment.")

    if time.time() - start > timeout_seconds:
        return _fallback("Response took too long. Please try again.")

    return _parse(raw, valid_catalog_urls)
