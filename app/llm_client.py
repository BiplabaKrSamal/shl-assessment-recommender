"""
LLM Client — google-genai SDK (v1 API, current).
Model: gemini-2.0-flash (free tier, fast, supports JSON mode).
Root cause of previous failure: google-generativeai==0.8.3 uses v1beta
which no longer supports gemini-1.5-flash model name.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.models import ChatResponse, Recommendation

logger = logging.getLogger("shl_recommender")

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        _client = genai.Client(api_key=api_key)
        logger.info(f"Gemini client ready. Key: {api_key[:8]}...")
    return _client


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=6),
    reraise=True,
)
def _call(system_prompt: str, messages: list[dict], model: str = "gemini-2.0-flash") -> str:
    client = _get_client()

    contents = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part(text=m["content"])]
        )
        for m in messages
    ]

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.1,
            max_output_tokens=1200,
            response_mime_type="application/json",
        ),
    )
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
        data = json.loads(m.group()) if m else {}

    reply = str(data.get("reply", "")).strip() or "Could you tell me more about the role?"
    raw_recs = data.get("recommendations", []) if isinstance(data.get("recommendations"), list) else []

    validated, seen = [], set()
    for rec in raw_recs:
        if not isinstance(rec, dict): continue
        name  = str(rec.get("name", "")).strip()
        url   = str(rec.get("url", "")).strip()
        ttype = str(rec.get("test_type", "A")).strip().upper()
        if not name or name.lower() in seen or not url.startswith(_SHL): continue
        if ttype not in _VALID: ttype = "A"
        seen.add(name.lower())
        validated.append(Recommendation(name=name, url=url, test_type=ttype))

    return ChatResponse(
        reply=reply,
        recommendations=validated[:10],
        end_of_conversation=bool(data.get("end_of_conversation", False)),
    )


def _fallback(msg: str) -> ChatResponse:
    return ChatResponse(reply=msg, recommendations=[], end_of_conversation=False)


def generate_response(
    system_prompt: str,
    messages: list[dict],
    valid_catalog_urls: set[str],
    timeout_seconds: int = 25,
) -> ChatResponse:
    start = time.time()
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    try:
        raw = _call(system_prompt, messages, model)
    except Exception as e:
        logger.error(f"Gemini failed ({model}): {e}")
        try:
            raw = _call(system_prompt, messages, "gemini-2.0-flash-lite")
        except Exception as e2:
            logger.error(f"Gemini fallback failed: {e2}")
            return _fallback("I'm temporarily unavailable. Please retry in a moment.")
    if time.time() - start > timeout_seconds:
        return _fallback("Response took too long. Please try again.")
    return _parse(raw, valid_catalog_urls)
