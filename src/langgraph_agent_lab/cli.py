"""CLI for the lab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from .graph import build_graph
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer
from .report import write_report
from .scenarios import load_scenarios
from .state import initial_state

app = typer.Typer(no_args_is_help=True)


def _select_scenario(scenarios: list, scenario_id: str | None):
    if scenario_id is None:
        return scenarios[0]
    for scenario in scenarios:
        if scenario.id == scenario_id:
            return scenario
    available = ", ".join(item.id for item in scenarios)
    raise typer.BadParameter(f"Unknown scenario_id '{scenario_id}'. Available: {available}")


def _snapshot_values(snapshot: Any) -> dict[str, Any]:
    if hasattr(snapshot, "values"):
        values = snapshot.values
    elif isinstance(snapshot, dict):
        values = snapshot.get("values", {})
    else:
        values = {}
    return dict(values) if isinstance(values, dict) else {}


def _snapshot_metadata(snapshot: Any) -> dict[str, Any]:
    if hasattr(snapshot, "metadata"):
        metadata = snapshot.metadata
    elif isinstance(snapshot, dict):
        metadata = snapshot.get("metadata", {})
    else:
        metadata = {}
    return dict(metadata) if isinstance(metadata, dict) else {}


def _snapshot_next(snapshot: Any) -> list[str]:
    if hasattr(snapshot, "next"):
        next_nodes = snapshot.next
    elif isinstance(snapshot, dict):
        next_nodes = snapshot.get("next", [])
    else:
        next_nodes = []
    return list(next_nodes) if isinstance(next_nodes, (list, tuple)) else []


def _snapshot_checkpoint_id(snapshot: Any) -> str | None:
    metadata = _snapshot_metadata(snapshot)
    return metadata.get("checkpoint_id") or metadata.get("id")


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run all grading scenarios and write metrics JSON."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer = build_checkpointer(cfg.get("checkpointer", "memory"), cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)
    metrics = []
    for scenario in scenarios:
        state = initial_state(scenario)
        run_config = {"configurable": {"thread_id": state["thread_id"]}}
        final_state = graph.invoke(state, config=run_config)
        metrics.append(
            metric_from_state(
                final_state, scenario.expected_route.value, scenario.requires_approval
            )
        )
    report = summarize_metrics(metrics)
    write_metrics(report, output)
    if cfg.get("report_path"):
        write_report(report, cfg["report_path"])
    typer.echo(f"Wrote metrics to {output}")


@app.command("time-travel")
def time_travel(
    config: Annotated[Path, typer.Option("--config")],
    scenario_id: Annotated[str | None, typer.Option("--scenario-id")] = None,
    checkpoint_index: Annotated[int, typer.Option("--checkpoint-index")] = -2,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    replay: Annotated[bool, typer.Option("--replay/--no-replay")] = True,
) -> None:
    """Replay from an earlier checkpoint using state history."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    scenario = _select_scenario(scenarios, scenario_id)
    checkpointer = build_checkpointer(cfg.get("checkpointer", "memory"), cfg.get("database_url"))
    if checkpointer is None:
        raise typer.BadParameter(
            "Time travel requires a checkpointer. Set checkpointer=memory/sqlite/postgres."
        )
    graph = build_graph(checkpointer=checkpointer)
    state = initial_state(scenario)
    run_config = {"configurable": {"thread_id": state["thread_id"]}}
    graph.invoke(state, config=run_config)
    history = list(graph.get_state_history(run_config))
    if not history:
        raise typer.BadParameter("No state history found. Ensure a checkpointer is enabled.")
    index = checkpoint_index if checkpoint_index >= 0 else len(history) + checkpoint_index
    if index < 0 or index >= len(history):
        raise typer.BadParameter(f"checkpoint_index out of range: {checkpoint_index}")
    snapshot = history[index]
    summary = {
        "scenario_id": scenario.id,
        "thread_id": state["thread_id"],
        "selected_index": index,
        "history": [
            {
                "index": idx,
                "next": _snapshot_next(item),
                "route": _snapshot_values(item).get("route"),
                "attempt": _snapshot_values(item).get("attempt"),
            }
            for idx, item in enumerate(history)
        ],
    }
    if output:
        output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        typer.echo(f"Wrote time travel history to {output}")
    else:
        typer.echo(f"Loaded {len(history)} checkpoints; selected index {index}")
    if replay:
        replay_thread_id = f"{state['thread_id']}-replay-{index}"
        replay_config = {"configurable": {"thread_id": replay_thread_id}}
        checkpoint_id = _snapshot_checkpoint_id(snapshot)
        replay_state: dict[str, Any] | None = _snapshot_values(snapshot)
        if checkpoint_id:
            replay_config["configurable"]["checkpoint_id"] = checkpoint_id
            replay_state = None
        try:
            replay_final = graph.invoke(replay_state, config=replay_config)
        except TypeError:
            replay_final = graph.invoke(
                _snapshot_values(snapshot), config={"configurable": {"thread_id": replay_thread_id}}
            )
        typer.echo(f"Replay completed. route={replay_final.get('route')}")


@app.command("validate-metrics")
def validate_metrics(metrics: Annotated[Path, typer.Option("--metrics")]) -> None:
    """Validate metrics JSON schema for grading."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    typer.echo(f"Metrics valid. success_rate={report.success_rate:.2%}")


if __name__ == "__main__":
    app()
