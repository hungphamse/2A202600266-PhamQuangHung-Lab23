"""Node skeletons for the LangGraph workflow.

Each function should be small, testable, and return a partial state update. Avoid mutating the
input state in place.
"""

from __future__ import annotations

import re

from .state import AgentState, ApprovalDecision, Route, make_event

_PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
_EMAIL_PATTERN = re.compile(r"(?<!\w)[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?!\w)")
_ORDER_ID_PATTERN = re.compile(
    r"(?i)\b(?:order\s*(?:id)?|ticket\s*(?:id)?|case\s*(?:id)?)\s*[:#-]?\s*([A-Za-z0-9][A-Za-z0-9-]{2,})"
)


def _redact_sensitive_text(value: str) -> tuple[str, dict[str, str]]:
    """Replace common sensitive tokens with stable placeholders."""
    metadata: dict[str, str] = {}

    def replace_phone(match: re.Match[str]) -> str:
        metadata.setdefault("phone_number", match.group(0))
        return "PHONE_NUMBER"

    def replace_email(match: re.Match[str]) -> str:
        metadata.setdefault("email_address", match.group(0))
        return "EMAIL_ADDRESS"

    def replace_order_id(match: re.Match[str]) -> str:
        metadata.setdefault("order_id", match.group(1))
        prefix = match.group(0)[: match.group(0).lower().find(match.group(1).lower())]
        return f"{prefix}ORDER_ID"

    redacted = _PHONE_PATTERN.sub(replace_phone, value)
    redacted = _EMAIL_PATTERN.sub(replace_email, redacted)
    redacted = _ORDER_ID_PATTERN.sub(replace_order_id, redacted)
    return redacted, metadata


def intake_node(state: AgentState) -> dict:
    """Normalize raw query into state fields.

    TODO(student): add normalization, PII checks, and metadata extraction.
    """
    raw_query = state.get("query", "")
    query = " ".join(raw_query.strip().split())
    redacted_query, pii_metadata = _redact_sensitive_text(query)
    metadata_bits = []
    if pii_metadata:
        metadata_bits.append(f"pii={','.join(sorted(pii_metadata))}")
    if redacted_query != query:
        metadata_bits.append("redacted")
    event_message = "query normalized"
    if metadata_bits:
        event_message = f"{event_message}; {'; '.join(metadata_bits)}"
    return {
        "query": redacted_query,
        "messages": [f"intake:{redacted_query[:40]}"],
        "events": [
            make_event(
                "intake", "completed", event_message, original_length=len(query), **pii_metadata
            )
        ],
    }


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route.

    TODO(student): replace keyword heuristics with a clear routing policy.
    Required routes: simple, tool, missing_info, risky, error.
    """
    query = state.get("query", "").lower()
    words = query.split()
    clean_words = [w.strip("?!.,;:") for w in words]
    route = Route.SIMPLE
    risk_level = "low"
    if any(token in query for token in ("refund", "delete", "send", "cancel", "transfer")):
        route = Route.RISKY
        risk_level = "high"
    elif any(
        token in query for token in ("status", "order", "lookup", "track", "where is", "find")
    ):
        route = Route.TOOL
    elif len(clean_words) < 5 and any(
        token in clean_words for token in ("it", "this", "that", "they")
    ):
        route = Route.MISSING_INFO
    elif any(token in query for token in ("timeout", "fail", "error", "exception")):
        route = Route.ERROR
    return {
        "route": route.value,
        "risk_level": risk_level,
        "events": [make_event("classify", "completed", f"route={route.value}")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    TODO(student): generate a specific clarification question from state.
    """
    query = state.get("query", "")
    if "order" in query.lower():
        question = "Can you provide the order ID so I can look it up?"
    else:
        question = "Can you share the missing details so I can continue?"
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "missing information requested")],
    }


def tool_node(state: AgentState) -> dict:
    """Call a mock tool.

    Simulates transient failures for error-route scenarios to demonstrate retry loops.
    TODO(student): implement idempotent tool execution and structured tool results.
    """
    attempt = int(state.get("attempt", 0))
    if state.get("route") == Route.ERROR.value and attempt < 2:
        result = f"ERROR: transient failure attempt={attempt} scenario={state.get('scenario_id', 'unknown')}"
    else:
        query = state.get("query", "")
        if "order" in query.lower():
            result = f"Order lookup completed for scenario={state.get('scenario_id', 'unknown')}"
        else:
            result = f"mock-tool-result for scenario={state.get('scenario_id', 'unknown')}"
    return {
        "tool_results": [result],
        "events": [make_event("tool", "completed", f"tool executed attempt={attempt}")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for approval.

    TODO(student): create a proposed action with evidence and risk justification.
    """
    query = state.get("query", "")
    reason = "high-risk action requested"
    if "refund" in query.lower():
        reason = "refund request may affect customer balance"
    elif any(token in query.lower() for token in ("delete", "send", "transfer", "cancel")):
        reason = "external or destructive action requested"
    return {
        "proposed_action": f"{reason}; approval required",
        "events": [make_event("risky_action", "pending_approval", reason)],
    }


def approval_node(state: AgentState) -> dict:
    """Human approval step with optional LangGraph interrupt().

    Set LANGGRAPH_INTERRUPT=true to use real interrupt() for HITL demos.
    Default uses mock decision so tests and CI run offline.

    TODO(student): implement reject/edit decisions and timeout escalation.
    """
    import os

    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        value = interrupt(
            {
                "proposed_action": state.get("proposed_action"),
                "risk_level": state.get("risk_level"),
            }
        )
        if isinstance(value, dict):
            decision = ApprovalDecision(**value)
        else:
            decision = ApprovalDecision(approved=bool(value))
    else:
        decision = ApprovalDecision(approved=True, comment="mock approval for lab")
    return {
        "approval": decision.model_dump(),
        "events": [make_event("approval", "completed", f"approved={decision.approved}")],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt or fallback decision.

    TODO(student): implement bounded retry, exponential backoff metadata, and fallback route.
    """
    attempt = int(state.get("attempt", 0)) + 1
    previous_errors = list(state.get("errors", []))
    errors = previous_errors + [f"transient failure attempt={attempt}"]
    route = (
        Route.DEAD_LETTER.value
        if attempt >= int(state.get("max_attempts", 3))
        else Route.TOOL.value
    )
    return {
        "attempt": attempt,
        "route": route,
        "errors": errors,
        "events": [make_event("retry", "completed", "retry attempt recorded", attempt=attempt)],
    }


def answer_node(state: AgentState) -> dict:
    """Produce a final response.

    TODO(student): ground the answer in tool_results and approval where relevant.
    """
    approval = state.get("approval") or {}
    if state.get("route") == Route.RISKY.value and not approval.get("approved"):
        answer = "I could not proceed because the action was not approved."
    elif state.get("tool_results"):
        answer = f"I found: {state['tool_results'][-1]}"
    else:
        answer = "This is a safe mock answer. Replace with your agent response."
    return {
        "final_answer": answer,
        "events": [make_event("answer", "completed", "answer generated")],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the 'done?' check that enables retry loops.

    TODO(student): replace heuristic with LLM-as-judge or structured validation.
    """
    tool_results = state.get("tool_results", [])
    latest = tool_results[-1] if tool_results else ""
    if "ERROR" in latest:
        return {
            "evaluation_result": "needs_retry",
            "events": [
                make_event("evaluate", "completed", "tool result indicates failure, retry needed")
            ],
        }
    if not latest:
        return {
            "evaluation_result": "needs_retry",
            "events": [make_event("evaluate", "completed", "missing tool result, retry needed")],
        }
    return {
        "evaluation_result": "success",
        "events": [make_event("evaluate", "completed", "tool result satisfactory")],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Log unresolvable failures for manual review.

    Third layer of error strategy: retry -> fallback -> dead letter.
    TODO(student): persist to dead-letter queue, alert on-call, or create support ticket.
    """
    return {
        "final_answer": "Request could not be completed after maximum retry attempts. Logged for manual review.",
        "events": [
            make_event(
                "dead_letter",
                "completed",
                f"max retries exceeded, attempt={state.get('attempt', 0)}",
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Finalize the run and emit a final audit event."""
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
