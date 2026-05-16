"""
Agent Orchestrator

This is the brain of the system. It decides:
  1. Is this query vague? → extract minimal context for clarification
  2. Is this a comparison query? → fetch named assessments specifically
  3. Is this a refinement? → rebuild query from full history + delta
  4. Is this an off-topic/injection attempt? → refuse before LLM call
  5. Build retrieval query → fetch top-k → inject into prompt → call LLM

Design choices:
- We do intent detection with regex + heuristics BEFORE the LLM call.
  This saves a round-trip on obvious refusals and vague queries.
- We extract the "effective retrieval query" from the FULL conversation history,
  not just the last message. This is critical for refinement: "add personality"
  only makes sense in context of the prior role/seniority.
- We pass the ENTIRE catalog in the system prompt AND top-k retrieved items
  appended to the user message. This dual-injection strategy ensures:
  (a) The model can validate recommendations against the full catalog.
  (b) The retrieval hint guides attention to the most relevant items.
"""

from __future__ import annotations

import re
from typing import Any

from app.models import ChatRequest, ChatResponse, Message
from app.retriever import CatalogRetriever
from app.prompts import build_system_prompt, build_messages_for_llm
from app.llm_client import generate_response


# ---------------------------------------------------------------------------
# Off-topic / injection guard patterns
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS = [
    r"ignore (previous|all|your) (instructions?|prompt|rules?)",
    r"you are now",
    r"pretend (to be|you are|you're)",
    r"act as (a )?(?!shl)",
    r"disregard (your|the) (instructions?|prompt|rules?)",
    r"new (instructions?|prompt|persona)",
    r"(reveal|show|print|output) (your |the )?(system |)prompt",
    r"jailbreak",
    r"dan mode",
]

_OFF_TOPIC_PATTERNS = [
    r"\b(salary|compensation|pay|wage)\b",
    r"\b(visa|immigration|work permit)\b",
    r"\b(interview (questions?|tips?|advice))\b",
    r"\b(resume|cv) (review|help|advice)\b",
    r"\b(legal (advice|compliance|question))\b",
    r"\b(competitor|rival|other (company|provider|vendor))\b",
]

_VAGUE_PATTERNS = [
    r"^i need (an? )?assessment\.?$",
    r"^help( me)?$",
    r"^i want (an? )?test\.?$",
    r"^(recommend|suggest|give me) (something|an? test|an? assessment)\.?$",
    r"^what (should|do) i use\??$",
]

_COMPARISON_PATTERNS = [
    r"\b(compare|difference between|vs\.?|versus|which is better)\b",
    r"\bhow (does|do) .+ (differ|compare|relate)\b",
]


def _is_injection(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in _INJECTION_PATTERNS)


def _is_off_topic(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in _OFF_TOPIC_PATTERNS)


def _is_vague(text: str) -> bool:
    t = text.strip().lower()
    return any(re.match(p, t) for p in _VAGUE_PATTERNS)


def _is_comparison(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in _COMPARISON_PATTERNS)


def _extract_named_assessments(text: str, retriever: CatalogRetriever) -> list[dict]:
    """
    Find assessment names mentioned in a comparison query and return their catalog entries.
    Tries to match known assessment names within the user text.
    """
    results = []
    for assessment in retriever.get_all():
        name = assessment["name"]
        # Check if name (or abbreviation) appears in the text
        if name.lower() in text.lower():
            results.append(assessment)
        # Check common abbreviations
        abbreviation = re.sub(r"[^A-Z0-9]", "", name)
        if len(abbreviation) >= 2 and abbreviation.lower() in text.lower():
            if assessment not in results:
                results.append(assessment)
    return results[:5]  # Cap at 5 for comparison context


# ---------------------------------------------------------------------------
# Conversation summarizer for retrieval query
# ---------------------------------------------------------------------------
def _build_retrieval_query(messages: list[dict]) -> str:
    """
    Extract a rich retrieval query from the full conversation history.

    Strategy: concatenate all user messages into a single query string.
    This is better than just the last message because:
    - "add personality tests" (turn 3) needs context from "Java developer" (turn 1)
    - "actually make it senior level" needs prior role context to be useful
    
    We give more weight to recent messages by repeating them.
    """
    user_messages = [m["content"] for m in messages if m["role"] == "user"]

    if not user_messages:
        return ""

    # Most recent message gets 2x weight (repeated)
    recent = user_messages[-1]
    earlier = " ".join(user_messages[:-1])

    # Extract structured signals from all user messages
    full_text = " ".join(user_messages)

    return f"{earlier} {recent} {recent}".strip()


def _turn_count(messages: list[dict]) -> int:
    """Count total turns (user + assistant) in the conversation."""
    return len(messages)


def _has_prior_recommendations(messages: list[dict]) -> bool:
    """Check if the assistant has already given recommendations."""
    for m in messages:
        if m["role"] == "assistant":
            try:
                import json
                data = json.loads(m["content"])
                if data.get("recommendations"):
                    return True
            except Exception:
                # assistant content might be plain text in edge cases
                if "url" in m["content"].lower() and "shl.com" in m["content"].lower():
                    return True
    return False


# ---------------------------------------------------------------------------
# Main agent function
# ---------------------------------------------------------------------------
class SHLAgent:
    """
    Stateless agent — all state is passed in via the request's message history.
    
    __init__ is called once at startup and builds the retriever index.
    chat() is called per request — it must be fast (< 30s cold, < 5s warm).
    """

    def __init__(self, catalog_path: str = "data/catalog.json"):
        self.retriever = CatalogRetriever(catalog_path=catalog_path)
        self._system_prompt = build_system_prompt(self.retriever.get_all())
        self._valid_urls: set[str] = {
            a["url"] for a in self.retriever.get_all()
        }

    def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Process a full conversation history and return the next agent reply.

        Args:
            request: ChatRequest with full message history

        Returns:
            ChatResponse with reply, recommendations, end_of_conversation
        """
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        if not messages:
            return ChatResponse(
                reply="Hello! I'm your SHL Assessment Recommender. Tell me about the role you're hiring for and I'll suggest the most relevant assessments.",
                recommendations=[],
                end_of_conversation=False,
            )

        last_user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            ""
        )

        # --- Guard: Turn limit (evaluator caps at 8) ---
        if _turn_count(messages) >= 8:
            # Force a recommendation on the last allowed turn
            retrieval_query = _build_retrieval_query(messages)
            retrieved = self.retriever.search(retrieval_query, k=10)
            system, llm_messages = build_messages_for_llm(
                messages, self._system_prompt, retrieved
            )
            # Override with instruction to finalize
            llm_messages = llm_messages[:-1] + [{
                "role": "user",
                "content": last_user_msg + "\n\n[SYSTEM: This is the final turn. You must provide your best recommendation shortlist now, even if you need more information. Do not ask further questions.]"
            }]
            return generate_response(self._system_prompt, llm_messages, self._valid_urls)

        # --- Guard: Prompt injection ---
        if _is_injection(last_user_msg):
            return ChatResponse(
                reply="I can only help with SHL assessment selection. How can I assist you with finding the right SHL assessments?",
                recommendations=[],
                end_of_conversation=False,
            )

        # --- Guard: Off-topic ---
        if _is_off_topic(last_user_msg):
            return ChatResponse(
                reply="I can only help with SHL assessment selection. I'm not able to advise on that topic. Is there a specific role or hiring need I can help you find assessments for?",
                recommendations=[],
                end_of_conversation=False,
            )

        # --- Path: Comparison query ---
        if _is_comparison(last_user_msg):
            named = _extract_named_assessments(last_user_msg, self.retriever)
            # Also do a semantic search on the comparison query
            semantic_hits = self.retriever.search(last_user_msg, k=6)
            # Merge, deduplicate
            all_context = named + [h for h in semantic_hits if h not in named]
            system, llm_messages = build_messages_for_llm(
                messages, self._system_prompt, all_context[:8]
            )
            return generate_response(system, llm_messages, self._valid_urls)

        # --- Path: Vague query (no prior context) ---
        is_first_turn = _turn_count(messages) <= 1
        if is_first_turn and _is_vague(last_user_msg):
            return ChatResponse(
                reply="I'd be happy to help! To recommend the right SHL assessments, could you tell me: (1) What role or job family are you hiring for? (2) What are the key skills or competencies most important for this role?",
                recommendations=[],
                end_of_conversation=False,
            )

        # --- Normal path: Build retrieval query from full history ---
        retrieval_query = _build_retrieval_query(messages)

        # Retrieve top-10 relevant assessments
        retrieved = self.retriever.search(retrieval_query, k=10)

        # Build LLM context with retrieval injection
        system, llm_messages = build_messages_for_llm(
            messages, self._system_prompt, retrieved
        )

        # Call LLM
        response = generate_response(system, llm_messages, self._valid_urls)

        return response
