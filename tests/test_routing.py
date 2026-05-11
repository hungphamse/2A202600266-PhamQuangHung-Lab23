from langgraph_agent_lab.routing import (
    route_after_approval,
    route_after_classify,
    route_after_evaluate,
    route_after_retry,
)
from langgraph_agent_lab.state import Route


def test_route_after_classify():
    assert route_after_classify({"route": Route.SIMPLE.value}) == "answer"
    assert route_after_classify({"route": Route.TOOL.value}) == "tool"
    assert route_after_classify({"route": Route.RISKY.value}) == "risky_action"


def test_route_after_approval():
    assert route_after_approval({"approval": {"approved": True}}) == "tool"
    assert route_after_approval({"approval": {"approved": False}}) == "clarify"


def test_route_after_retry_bound():
    assert route_after_retry({"attempt": 0, "max_attempts": 3}) == "tool"
    assert route_after_retry({"attempt": 3, "max_attempts": 3}) == "dead_letter"


def test_route_after_evaluate():
    assert route_after_evaluate({"evaluation_result": "success"}) == "answer"
    assert route_after_evaluate({"evaluation_result": "needs_retry"}) == "retry"
