# src/graph/graph_builder.py

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from graph.state import AgentState
from graph.nodes import (
    router_node,
    single_analysis_node,
    batch_analysis_node,
    stats_node,
    final_output_node,
)


def _route_selector(state: AgentState) -> Literal["single", "batch"]:
    route = state.get("route", "single")
    if route not in ("single", "batch"):
        return "single"
    return route  # type: ignore


def build_agent_graph():
    """
    Construye y compila el LangGraph del agente:

    Start -> router -> (single_analysis | batch_analysis) -> stats -> final -> END

    Usa MemorySaver como checkpointer, lo que da memoria por thread_id.
    """

    workflow = StateGraph(AgentState)

    # Nodos
    workflow.add_node("router", router_node)
    workflow.add_node("single_analysis", single_analysis_node)
    workflow.add_node("batch_analysis", batch_analysis_node)
    workflow.add_node("stats", stats_node)
    workflow.add_node("final", final_output_node)

    # Entry point
    workflow.set_entry_point("router")

    # Routing condicional desde 'router'
    workflow.add_conditional_edges(
        "router",
        _route_selector,
        {
            "single": "single_analysis",
            "batch": "batch_analysis",
        },
    )

    # Ambos caminos pasan luego por stats -> final -> END
    workflow.add_edge("single_analysis", "stats")
    workflow.add_edge("batch_analysis", "stats")
    workflow.add_edge("stats", "final")
    workflow.add_edge("final", END)

    # Memoria (LangGraph checkpoint)
    checkpointer = MemorySaver()

    app = workflow.compile(checkpointer=checkpointer)

    return app
