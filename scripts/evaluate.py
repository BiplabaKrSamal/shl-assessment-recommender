"""
Local Evaluation Harness
========================
Simulates the SHL automated evaluator locally against public conversation traces.

Usage:
    python scripts/evaluate.py --traces data/traces/ --endpoint http://localhost:8000
    python scripts/evaluate.py --traces data/traces/ --local   # uses agent directly

Each trace JSON file has the format:
{
  "persona": "...",
  "facts": {...},
  "expected_assessments": ["OPQ32", "Verify Numerical Reasoning", ...],
  "conversation": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}

This harness:
1. Replays each trace against your /chat endpoint (or local agent)
2. Collects the final recommendations
3. Computes Recall@10 per trace
4. Reports mean Recall@10 and behavior probe pass-rates
5. Flags hallucinations (non-catalog URLs in recommendations)

The harness uses a SIMULATED USER that reads from the trace's conversation
history, mirroring how the real evaluator works.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Recall@K computation
# ---------------------------------------------------------------------------
def recall_at_k(recommended: list[str], expected: list[str], k: int = 10) -> float:
    """
    Recall@K = (relevant in top-K) / total relevant

    Args:
        recommended: List of recommended assessment names (up to K)
        expected: List of expected/relevant assessment names
        k: Cutoff

    Returns:
        Float in [0, 1]
    """
    if not expected:
        return 1.0  # vacuously true

    recommended_lower = set(r.lower() for r in recommended[:k])
    hits = sum(
        1 for e in expected
        if any(
            e.lower() in r or r in e.lower()
            for r in recommended_lower
        )
    )
    return hits / len(expected)


# ---------------------------------------------------------------------------
# Simulated user (reads from trace)
# ---------------------------------------------------------------------------
class SimulatedUser:
    """
    Replays a conversation trace to simulate a real user.
    If the trace runs out, responds with "I have no preference" to any question.
    """

    def __init__(self, trace: dict):
        self.facts = trace.get("facts", {})
        self.persona = trace.get("persona", "")
        self.expected = trace.get("expected_assessments", [])
        # Pre-built user turns from the trace
        self.user_turns = [
            m["content"] for m in trace.get("conversation", [])
            if m["role"] == "user"
        ]
        self._turn_idx = 0

    def next_message(self, agent_reply: str) -> str | None:
        """
        Return the next user message. Returns None if conversation should end.
        """
        if self._turn_idx < len(self.user_turns):
            msg = self.user_turns[self._turn_idx]
            self._turn_idx += 1
            return msg

        # Out of scripted turns — respond generically if agent is still asking
        if "?" in agent_reply:
            return "I have no preference on that."

        # Agent gave recommendations or said it's done — end conversation
        return None


# ---------------------------------------------------------------------------
# Endpoint runner
# ---------------------------------------------------------------------------
def run_trace_against_endpoint(
    trace: dict,
    endpoint: str,
    max_turns: int = 8,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """
    Run a single trace against a live /chat endpoint.

    Returns dict with:
        - final_recommendations: list of recommended names
        - recall: Recall@10 score
        - turns_used: number of turns taken
        - timing: list of per-turn response times
        - hard_eval_pass: bool (schema, catalog URLs, turn cap)
        - behavior_probes: dict of probe results
        - errors: list of errors encountered
    """
    user = SimulatedUser(trace)
    messages = []
    final_recs = []
    timing = []
    errors = []
    behavior_probes = {
        "no_recs_on_turn_1_vague": None,
        "honored_turn_cap": True,
        "schema_compliant": True,
        "catalog_urls_only": True,
        "ended_after_shortlist": None,
    }

    client = httpx.Client(timeout=timeout)
    first_user_msg = user.user_turns[0] if user.user_turns else "I need an assessment"

    # Seed first user message
    current_user_msg = first_user_msg
    turn = 0

    while turn < max_turns and current_user_msg is not None:
        turn += 1
        messages.append({"role": "user", "content": current_user_msg})

        start = time.time()
        try:
            resp = client.post(
                f"{endpoint}/chat",
                json={"messages": messages},
            )
            elapsed = time.time() - start
            timing.append(elapsed)

            if resp.status_code != 200:
                errors.append(f"Turn {turn}: HTTP {resp.status_code}: {resp.text[:200]}")
                behavior_probes["schema_compliant"] = False
                break

            data = resp.json()

            # Schema validation
            required_keys = {"reply", "recommendations", "end_of_conversation"}
            if not required_keys.issubset(data.keys()):
                errors.append(f"Turn {turn}: Missing keys: {required_keys - set(data.keys())}")
                behavior_probes["schema_compliant"] = False
                break

            reply = data["reply"]
            recs = data.get("recommendations", [])
            end = data.get("end_of_conversation", False)

            # Check behavior probe: turn 1 vague → no recommendations
            if turn == 1 and _is_vague_query(current_user_msg):
                behavior_probes["no_recs_on_turn_1_vague"] = len(recs) == 0

            # Check catalog URL validity
            for rec in recs:
                url = rec.get("url", "")
                if not url.startswith("https://www.shl.com"):
                    errors.append(f"Turn {turn}: Non-SHL URL: {url}")
                    behavior_probes["catalog_urls_only"] = False

            # Collect final recommendations
            if recs:
                final_recs = [r["name"] for r in recs]

            # Add assistant message to history
            messages.append({"role": "assistant", "content": json.dumps(data)})

            if end:
                behavior_probes["ended_after_shortlist"] = bool(recs)
                break

            if elapsed > timeout:
                errors.append(f"Turn {turn}: Timeout ({elapsed:.1f}s)")
                break

        except Exception as e:
            errors.append(f"Turn {turn}: Exception: {e}")
            break

        current_user_msg = user.next_message(reply if "reply" in locals() else "")

    behavior_probes["honored_turn_cap"] = turn <= max_turns

    recall = recall_at_k(final_recs, trace.get("expected_assessments", []))

    return {
        "trace_id": trace.get("id", "unknown"),
        "persona": trace.get("persona", ""),
        "expected": trace.get("expected_assessments", []),
        "final_recommendations": final_recs,
        "recall_at_10": recall,
        "turns_used": turn,
        "timing": timing,
        "avg_latency": sum(timing) / len(timing) if timing else 0,
        "hard_eval_pass": (
            behavior_probes["schema_compliant"]
            and behavior_probes["catalog_urls_only"]
            and behavior_probes["honored_turn_cap"]
        ),
        "behavior_probes": behavior_probes,
        "errors": errors,
    }


def _is_vague_query(text: str) -> bool:
    import re
    patterns = [
        r"^i need (an? )?assessment",
        r"^help( me)?$",
        r"^i want (an? )?test",
    ]
    t = text.strip().lower()
    return any(re.match(p, t) for p in patterns)


# ---------------------------------------------------------------------------
# Local agent runner (no HTTP)
# ---------------------------------------------------------------------------
def run_trace_locally(trace: dict, max_turns: int = 8) -> dict[str, Any]:
    """
    Run a trace against the local agent (no HTTP overhead).
    Useful for fast iteration during development.
    """
    from app.agent import SHLAgent
    from app.models import ChatRequest, Message

    agent = SHLAgent(catalog_path="data/catalog.json")
    user = SimulatedUser(trace)
    messages = []
    final_recs = []
    timing = []
    errors = []

    current_user_msg = user.user_turns[0] if user.user_turns else "I need an assessment"
    turn = 0
    last_reply = ""

    while turn < max_turns and current_user_msg is not None:
        turn += 1
        messages.append(Message(role="user", content=current_user_msg))

        start = time.time()
        try:
            request = ChatRequest(messages=messages)
            response = agent.chat(request)
            elapsed = time.time() - start
            timing.append(elapsed)

            if response.recommendations:
                final_recs = [r.name for r in response.recommendations]

            messages.append(Message(
                role="assistant",
                content=json.dumps({
                    "reply": response.reply,
                    "recommendations": [r.model_dump() for r in response.recommendations],
                    "end_of_conversation": response.end_of_conversation,
                })
            ))

            last_reply = response.reply
            if response.end_of_conversation:
                break

        except Exception as e:
            errors.append(f"Turn {turn}: {e}")
            break

        current_user_msg = user.next_message(last_reply)

    recall = recall_at_k(final_recs, trace.get("expected_assessments", []))

    return {
        "trace_id": trace.get("id", "unknown"),
        "persona": trace.get("persona", "")[:60],
        "expected": trace.get("expected_assessments", []),
        "final_recommendations": final_recs,
        "recall_at_10": recall,
        "turns_used": turn,
        "avg_latency": sum(timing) / len(timing) if timing else 0,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def print_report(results: list[dict]) -> None:
    print("\n" + "="*70)
    print("SHL ASSESSMENT RECOMMENDER — EVALUATION REPORT")
    print("="*70)

    recalls = [r["recall_at_10"] for r in results]
    mean_recall = sum(recalls) / len(recalls) if recalls else 0
    hard_pass = sum(1 for r in results if r.get("hard_eval_pass", True))

    print(f"\n📊 SUMMARY")
    print(f"   Traces evaluated:     {len(results)}")
    print(f"   Mean Recall@10:       {mean_recall:.3f} ({mean_recall*100:.1f}%)")
    print(f"   Hard eval pass rate:  {hard_pass}/{len(results)}")
    print(f"   Avg latency/turn:     {sum(r['avg_latency'] for r in results)/len(results):.2f}s")

    print(f"\n📋 PER-TRACE RESULTS")
    print(f"{'ID':<20} {'Recall@10':>10} {'Turns':>6} {'Latency':>8} {'Errors':>6}")
    print("-"*60)

    for r in results:
        err_flag = "❌" if r["errors"] else "✓"
        print(
            f"{str(r['trace_id']):<20} "
            f"{r['recall_at_10']:>10.3f} "
            f"{r['turns_used']:>6} "
            f"{r['avg_latency']:>7.2f}s "
            f"{err_flag:>6}"
        )

    # Show misses
    low_recall = [r for r in results if r["recall_at_10"] < 0.5]
    if low_recall:
        print(f"\n⚠️  LOW RECALL TRACES (<0.5)")
        for r in low_recall:
            print(f"\n  Trace: {r['trace_id']}")
            print(f"  Expected:    {r['expected']}")
            print(f"  Got:         {r['final_recommendations']}")
            if r["errors"]:
                print(f"  Errors:      {r['errors']}")

    print("\n" + "="*70)
    grade = "PASS ✅" if mean_recall >= 0.5 and hard_pass == len(results) else "NEEDS IMPROVEMENT ⚠️"
    print(f"OVERALL: {grade}")
    print("="*70 + "\n")


# ---------------------------------------------------------------------------
# Synthetic traces for local testing (when you don't have the official traces)
# ---------------------------------------------------------------------------
SYNTHETIC_TRACES = [
    {
        "id": "java_developer",
        "persona": "HR manager at a software company hiring a Java developer",
        "facts": {"role": "Java developer", "level": "mid-level", "years": 4},
        "expected_assessments": ["Java 8 (New)", "Core Java", "OPQ32 (Occupational Personality Questionnaire)"],
        "conversation": [
            {"role": "user", "content": "I am hiring a Java developer who works closely with stakeholders"},
            {"role": "user", "content": "Mid-level, around 4 years of experience"},
            {"role": "user", "content": "Yes, also interested in personality assessment"},
        ]
    },
    {
        "id": "customer_service",
        "persona": "Recruiter at a call center company",
        "facts": {"role": "customer service representative", "level": "entry", "volume": "high"},
        "expected_assessments": ["Customer Service Aptitude (CSA)", "Contact Centre Screener", "Verify Verbal Reasoning"],
        "conversation": [
            {"role": "user", "content": "We need assessments for customer service roles at a call center"},
            {"role": "user", "content": "Entry level, we hire about 200 people a year"},
        ]
    },
    {
        "id": "senior_manager",
        "persona": "CHRO looking for leadership assessment",
        "facts": {"role": "senior manager", "level": "senior", "focus": "leadership personality"},
        "expected_assessments": ["OPQ32 (Occupational Personality Questionnaire)", "Leadership Report (OPQ32)", "Motivation Questionnaire (MQ)"],
        "conversation": [
            {"role": "user", "content": "I need to assess senior managers for a leadership development program"},
            {"role": "user", "content": "Personality and leadership style are most important"},
        ]
    },
    {
        "id": "data_scientist",
        "persona": "Tech company hiring data scientists",
        "facts": {"role": "data scientist", "skills": ["Python", "ML", "SQL"]},
        "expected_assessments": ["Python (New)", "Data Science with Python", "Machine Learning Concepts", "SQL (New)"],
        "conversation": [
            {"role": "user", "content": "Hiring data scientists with Python and machine learning skills"},
            {"role": "user", "content": "Mid to senior level, SQL knowledge important too"},
        ]
    },
    {
        "id": "sales_rep",
        "persona": "Sales director at B2B company",
        "facts": {"role": "sales representative", "focus": "personality motivation"},
        "expected_assessments": ["Sales Aptitude (SA)", "Sales Preference Questionnaire (SPQ)", "Motivation Questionnaire (MQ)"],
        "conversation": [
            {"role": "user", "content": "Looking for assessments for B2B sales representatives"},
            {"role": "user", "content": "Personality and motivation are key for us"},
        ]
    },
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SHL Recommender Evaluation Harness")
    parser.add_argument("--endpoint", default="http://localhost:8000", help="API endpoint")
    parser.add_argument("--traces", default=None, help="Path to traces directory (JSON files)")
    parser.add_argument("--local", action="store_true", help="Run against local agent (no HTTP)")
    parser.add_argument("--max-turns", type=int, default=8, help="Max conversation turns")
    args = parser.parse_args()

    # Load traces
    if args.traces and Path(args.traces).exists():
        trace_dir = Path(args.traces)
        traces = []
        for f in sorted(trace_dir.glob("*.json")):
            traces.append(json.loads(f.read_text()))
        print(f"Loaded {len(traces)} traces from {args.traces}")
    else:
        print(f"No traces directory found. Using {len(SYNTHETIC_TRACES)} synthetic traces.")
        traces = SYNTHETIC_TRACES

    results = []
    for i, trace in enumerate(traces):
        print(f"\n[{i+1}/{len(traces)}] Running trace: {trace.get('id', 'unknown')}")
        print(f"  Persona: {trace.get('persona', '')[:60]}")

        if args.local:
            result = run_trace_locally(trace, max_turns=args.max_turns)
        else:
            result = run_trace_against_endpoint(
                trace,
                endpoint=args.endpoint,
                max_turns=args.max_turns,
            )

        print(f"  Recall@10: {result['recall_at_10']:.3f} | Turns: {result['turns_used']} | Avg latency: {result['avg_latency']:.2f}s")
        if result.get("errors"):
            print(f"  Errors: {result['errors']}")
        results.append(result)

    print_report(results)

    # Save results
    output_path = Path("data/eval_results.json")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
