import os

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from backend.app.state import MedicalState


def physician_review(state: MedicalState) -> MedicalState:
    """
    Human-in-the-loop supervision step.

    Attend une decision reelle via Command(resume={...}) :
    {"decision": "approve|modify|reject", "treatment": "...", "notes": "..."}
    """
    decision_payload = state.get("physician_decision")

    if not decision_payload and os.getenv("MEDICAL_SIMULATION_MODE", "false").lower() == "true":
        decision_payload = {
            "decision": "approve",
            "treatment": "Validation humaine simulee pour demonstration academique.",
            "notes": "Mode simulation actif.",
        }

    if not decision_payload:
        decision_payload = interrupt({
            "type": "physician_review",
            "diagnostic_summary": state.get("diagnostic_summary", ""),
            "interim_care": state.get("interim_care", ""),
            "clinical_score": state.get("clinical_score", 0),
            "severity_level": state.get("severity_level", "faible"),
            "allowed_decisions": ["approve", "modify", "reject"],
        })

    if isinstance(decision_payload, dict):
        decision = decision_payload.get("decision", "approve")
        treatment = decision_payload.get("treatment", "")
        notes = decision_payload.get("notes", "")
    else:
        decision = str(decision_payload)
        treatment = ""
        notes = ""

    if decision not in {"approve", "modify", "reject"}:
        decision = "modify"

    if not treatment:
        treatment = "Supervision humaine simulee non encore renseignee."

    return {
        "messages": [
            AIMessage(
                content=f"[Human Supervision] Decision HITL : {decision}",
                name="physician_review",
            )
        ],
        "physician_decision": decision,
        "physician_approval": decision == "approve",
        "physician_treatment": treatment,
        "physician_notes": notes,
        "report_status": decision,
        "next": "report_agent",
    }
