"""Graph construction.

This module is intentionally import-safe. It imports LangGraph only inside the builder so unit tests
that check schema/metrics can run even if students are still debugging graph wiring.
"""

from __future__ import annotations

from typing import Any

from .nodes import (
    answer_node,
    approval_node,
    ask_clarification_node,
    classify_node,
    dead_letter_node,
    evaluate_node,
    finalize_node,
    intake_node,
    retry_or_fallback_node,
    risky_action_node,
    tool_node,
)
from .routing import (
    route_after_approval,
    route_after_classify,
    route_after_evaluate,
    route_after_retry,
)
from .state import AgentState


def build_graph(checkpointer: Any | None = None):
    """Build and compile the LangGraph workflow.

    TODO(student): review the architecture and modify nodes/edges only with a clear reason.
    Required behaviors:
    - intake -> classify (normalization + routing)
    - classify routes to answer/tool/clarify/risky/retry
    - tool -> evaluate creates the retry loop (slide: "done?" check)
    - risky path requires approval before tool/action
    - retry loop bounded by max_attempts -> dead_letter on exhaustion
    - all paths eventually reach finalize -> END
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception as exc:  # pragma: no cover - helpful install error
        raise RuntimeError(
            "LangGraph is required. Run: pip install -e '.[dev]' or pip install langgraph"
        ) from exc

    graph = StateGraph(AgentState)
    graph.add_node("intake", intake_node)
    graph.add_node("classify", classify_node)
    graph.add_node("answer", answer_node)
    graph.add_node("tool", tool_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("clarify", ask_clarification_node)
    graph.add_node("risky_action", risky_action_node)
    graph.add_node("approval", approval_node)
    graph.add_node("retry", retry_or_fallback_node)
    graph.add_node("dead_letter", dead_letter_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "classify")
    graph.add_conditional_edges("classify", route_after_classify)
    graph.add_edge("tool", "evaluate")
    graph.add_conditional_edges("evaluate", route_after_evaluate)
    graph.add_edge("clarify", "finalize")
    graph.add_edge("risky_action", "approval")
    graph.add_conditional_edges("approval", route_after_approval)
    graph.add_conditional_edges("retry", route_after_retry)
    graph.add_edge("answer", "finalize")
    graph.add_edge("dead_letter", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)
