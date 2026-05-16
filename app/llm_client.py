"""
LLM Client — google-generativeai SDK.
Model: gemini-1.5-flash (free tier, 60 req/min, JSON mode, 1M context).
Temperature: 0.1 — near-deterministic for reliable JSON output.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.models import ChatResponse, Recommendation

logger = logging.getLogger("shl_recommender")

# ── Init ──────────────────────────────────────────────────────────────────
_initialized = False

def _ensure_init():
    global _initialized
    if not _initialized:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Add it in Render → Environment Variables."
            )
        genai.configure(api_key=api_key)
        logger.info(f"Gemini configured. Key prefix: {api_key[:8]}...")
        _initialized = True


# ── LLM call ─────────────────────────────────────────────────────────────
@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=6),
    reraise=True,
)
def _call(system_prompt: str, messages: list[dict], model_name: str) -> str:
    _ensure_init()

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=1200,
            response_mime_type="application/json",
        ),
    )

    # Convert to Gemini format
    gemini_messages = [
        {"role": "user" if m["role"] == "user" else "model",
         "parts": [{"text": m["content"]}]}
        for m in messages
    ]

    response = model.generate_content(gemini_messages)
    return response.text


# ── Parser ────────────────────────────────────────────────────────────────
_SHL = "https://www.shl.com"
_VALID = set("ABCDEKPMS")


def _parse(raw: str, valid_urls: set[str]) -> ChatResponse:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned.strip())

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
            except Exception:
                return _fallback("I had trouble formatting my response. Please try again.")
        else:
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
    primary = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    try:
        raw = _call(system_prompt, messages, primary)
    except Exception as e:
        logger.error(f"Gemini call failed ({primary}): {e}")
        # Fallback to smaller model
        try:
            raw = _call(system_prompt, messages, "gemini-1.5-flash-8b")
        except Exception as e2:
            logger.error(f"Gemini fallback failed: {e2}")
            return _fallback("I'm temporarily unavailable. Please retry in a moment.")

    if time.time() - start > timeout_seconds:
        return _fallback("Response took too long. Please try again.")

    return _parse(raw, valid_catalog_urls)
