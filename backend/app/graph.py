from __future__ import annotations

import os

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from backend.app.nodes.diagnostic_agent import diagnostic_agent
from backend.app.nodes.physician_review import physician_review
from backend.app.nodes.report_agent import report_agent
from backend.app.nodes.supervisor import route_after_supervisor, supervisor
from backend.app.state import MedicalState


load_dotenv()


def _running_under_langgraph_api() -> bool:
    return bool(os.getenv("LANGSMITH_LANGGRAPH_API_VARIANT"))


def build_graph():
    """
    Build and compile the medical multi-agent LangGraph workflow.
    """
    builder = StateGraph(MedicalState)

    builder.add_node("supervisor", supervisor)
    builder.add_node("diagnostic_agent", diagnostic_agent)
    builder.add_node("physician_review", physician_review)
    builder.add_node("report_agent", report_agent)

    builder.set_entry_point("supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "diagnostic_agent": "diagnostic_agent",
            "physician_review": "physician_review",
            "report_agent": "report_agent",
            "FINISH": END,
        },
    )

    # Every agent returns to the supervisor, which reads state["next"] and routes.
    builder.add_edge("diagnostic_agent", "supervisor")
    builder.add_edge("physician_review", "supervisor")
    builder.add_edge("report_agent", "supervisor")

    if _running_under_langgraph_api():
        return builder.compile()

    return builder.compile(checkpointer=MemorySaver())


# Exported graph used by FastAPI, LangGraph CLI and LangGraph Studio.
graph = build_graph()
