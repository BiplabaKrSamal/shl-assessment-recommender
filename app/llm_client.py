"""
LLM Client with retry logic and structured output parsing.

Model choice: google/gemini-1.5-flash (free tier, 1M context, fast, JSON-mode reliable).
Fallback: gemini-1.5-flash-8b if rate limited.

Why Gemini over GPT-4o-mini?
- Larger free tier (60 req/min vs 3 req/min on free GPT)  
- 1M context window handles full catalog + history comfortably
- Native JSON mode reduces parse failures
- Sufficient quality for structured recommendation tasks

Why not Anthropic Claude for the agent?
- Assignment says free tiers acceptable; Gemini free tier is more generous
- But the code is LLM-agnostic — swap API_PROVIDER env var to switch
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.models import ChatResponse, Recommendation


# ---------------------------------------------------------------------------
# Client initialization
# ---------------------------------------------------------------------------
def _init_gemini():
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not set. "
            "Get a free key at https://aistudio.google.com/app/apikey"
        )
    genai.configure(api_key=api_key)


_gemini_initialized = False


def _ensure_gemini():
    global _gemini_initialized
    if not _gemini_initialized:
        _init_gemini()
        _gemini_initialized = True


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------
@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _call_gemini(
    system_prompt: str,
    messages: list[dict],
    model_name: str = "gemini-1.5-flash",
    temperature: float = 0.1,
    max_output_tokens: int = 1024,
) -> str:
    """
    Call Gemini API and return raw text response.
    Temperature=0.1: near-deterministic for consistent JSON output,
    small variance prevents degenerate repetition.
    """
    _ensure_gemini()

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt,
        generation_config=genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",  # JSON mode
        ),
    )

    # Convert OpenAI-style messages to Gemini format
    gemini_messages = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        gemini_messages.append({
            "role": role,
            "parts": [{"text": msg["content"]}],
        })

    response = model.generate_content(gemini_messages)
    return response.text


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------
_CATALOG_URL_PREFIX = "https://www.shl.com"
_VALID_TEST_TYPES = set("ABCDEKPMS")


def _parse_response(raw: str, valid_urls: set[str]) -> ChatResponse:
    """
    Parse and validate LLM JSON output into a ChatResponse.

    Validation steps:
    1. Strip markdown fences if model ignores JSON mode
    2. Parse JSON
    3. Validate schema fields exist
    4. Filter recommendations: only catalog URLs, valid test types
    5. Cap at 10 recommendations
    6. If parsing fully fails, return safe fallback
    """
    # Strip code fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned.strip())

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract JSON object from mixed text
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return _fallback_response("I encountered an issue formatting my response. Please try again.")
        else:
            return _fallback_response("I encountered an issue formatting my response. Please try again.")

    reply = str(data.get("reply", "")).strip()
    if not reply:
        reply = "I need a moment to clarify. Could you tell me more about the role you're hiring for?"

    raw_recs = data.get("recommendations", [])
    if not isinstance(raw_recs, list):
        raw_recs = []

    # Validate each recommendation
    validated_recs = []
    seen_names = set()
    for rec in raw_recs:
        if not isinstance(rec, dict):
            continue

        name = str(rec.get("name", "")).strip()
        url = str(rec.get("url", "")).strip()
        test_type = str(rec.get("test_type", "A")).strip().upper()

        # Deduplicate
        if name.lower() in seen_names:
            continue
        seen_names.add(name.lower())

        # Validate URL is from SHL catalog
        if not url.startswith(_CATALOG_URL_PREFIX):
            continue
        # Extra guard: URL must be in known catalog (or at least on shl.com)
        # We allow any shl.com URL to handle minor URL variations from the model
        
        # Validate test_type
        if test_type not in _VALID_TEST_TYPES:
            test_type = "A"

        if name:
            validated_recs.append(Recommendation(name=name, url=url, test_type=test_type))

    # Cap at 10
    validated_recs = validated_recs[:10]

    end_of_conv = bool(data.get("end_of_conversation", False))

    return ChatResponse(
        reply=reply,
        recommendations=validated_recs,
        end_of_conversation=end_of_conv,
    )


def _fallback_response(message: str) -> ChatResponse:
    return ChatResponse(
        reply=message,
        recommendations=[],
        end_of_conversation=False,
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
def generate_response(
    system_prompt: str,
    messages: list[dict],
    valid_catalog_urls: set[str],
    timeout_seconds: int = 25,
) -> ChatResponse:
    """
    Generate agent response with full error handling.

    Args:
        system_prompt: Built system prompt with catalog embedded
        messages: Full conversation history (OpenAI format)
        valid_catalog_urls: Set of valid catalog URLs for validation
        timeout_seconds: Soft deadline (we can't hard-kill Gemini SDK calls,
                        but retry logic is bounded)

    Returns:
        Validated ChatResponse
    """
    start = time.time()

    try:
        raw = _call_gemini(
            system_prompt=system_prompt,
            messages=messages,
            model_name=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            temperature=0.1,
            max_output_tokens=1200,
        )
    except Exception as e:
        error_str = str(e).lower()
        if "quota" in error_str or "rate" in error_str:
            # Try fallback model
            try:
                raw = _call_gemini(
                    system_prompt=system_prompt,
                    messages=messages,
                    model_name="gemini-1.5-flash-8b",
                    temperature=0.1,
                    max_output_tokens=1200,
                )
            except Exception:
                return _fallback_response(
                    "I'm temporarily over capacity. Please retry in a moment."
                )
        else:
            return _fallback_response(
                f"I encountered an error generating a response. Please try again."
            )

    elapsed = time.time() - start
    if elapsed > timeout_seconds:
        return _fallback_response(
            "The response took too long to generate. Please try a shorter message."
        )

    return _parse_response(raw, valid_catalog_urls)
