"""
LLM Client — google-genai SDK (new, stable API).

Model: gemini-1.5-flash (free tier, 60 req/min, JSON mode, 1M context).
Temperature: 0.1 — near-deterministic for reliable JSON output.
Retry: 3x exponential backoff on quota/rate errors.
Parser: strict validation — URL allowlist, type codes, dedup, cap@10.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.models import ChatResponse, Recommendation


# ── Init ──────────────────────────────────────────────────────────────────
_client: genai.Client | None = None

def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set.")
        _client = genai.Client(api_key=api_key)
    return _client


# ── LLM call with retry ───────────────────────────────────────────────────
@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _call_gemini(system_prompt: str, messages: list[dict], model: str = "gemini-1.5-flash") -> str:
    client = _get_client()

    # Convert to google-genai Contents format
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

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


# ── Response parser ───────────────────────────────────────────────────────
_SHL_URL_PREFIX = "https://www.shl.com"
_VALID_TYPES = set("ABCDEKPMS")


def _parse(raw: str, valid_urls: set[str]) -> ChatResponse:
    # Strip markdown fences
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

    validated = []
    seen = set()
    for rec in raw_recs:
        if not isinstance(rec, dict):
            continue
        name = str(rec.get("name", "")).strip()
        url  = str(rec.get("url", "")).strip()
        ttype = str(rec.get("test_type", "A")).strip().upper()
        if not name or name.lower() in seen:
            continue
        if not url.startswith(_SHL_URL_PREFIX):
            continue
        if ttype not in _VALID_TYPES:
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


# ── Public API ────────────────────────────────────────────────────────────
def generate_response(
    system_prompt: str,
    messages: list[dict],
    valid_catalog_urls: set[str],
    timeout_seconds: int = 25,
) -> ChatResponse:
    start = time.time()
    try:
        raw = _call_gemini(system_prompt, messages, model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
    except Exception as e:
        err = str(e).lower()
        if "quota" in err or "rate" in err:
            try:
                raw = _call_gemini(system_prompt, messages, model="gemini-1.5-flash-8b")
            except Exception:
                return _fallback("I'm temporarily over capacity. Please retry in a moment.")
        else:
            return _fallback("I encountered an error. Please try again.")

    if time.time() - start > timeout_seconds:
        return _fallback("Response took too long. Please try a shorter message.")

    return _parse(raw, valid_catalog_urls)
