from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command
from pydantic import BaseModel, Field

from backend.app.graph import graph
from backend.app.schemas import ChatRequest, DiagnosisRequest, ReportRequest, ResumeRequest
from backend.app.services.patient_memory import patient_memory
from backend.app.services.performance import perf_timer
from backend.app.services.pdf_export import generate_report_pdf
from backend.app.services.safety import validate_patient_input


API_VERSION = "1.0.0"
GRAPH_ID = "medical_workflow"

logger = logging.getLogger("medical_multiagents.api")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


class SessionStartRequest(BaseModel):
    """Payload used to create a frontend-ready consultation session."""

    patient_id: str | None = None
    patient_name: str | None = None
    session_id: str | None = None


class ConsultationStartRequest(BaseModel):
    """Payload used to start a LangGraph medical consultation workflow."""

    patient_case: str = Field(..., min_length=1)
    patient_id: str | None = None
    patient_name: str | None = None
    session_id: str | None = None
    symptoms: list[str] = Field(default_factory=list)
    medical_history: list[str] = Field(default_factory=list)


class ConsultationResumeRequest(BaseModel):
    """Payload used to resume a workflow paused by LangGraph interrupt()."""

    thread_id: str = Field(..., min_length=1)
    resume: Any


app = FastAPI(
    title="Medical Multi-Agents API",
    version=API_VERSION,
    description="LangGraph medical multi-agent workflow with HITL, reports and export.",
)


# CORS is intentionally permissive for local academic development.
# In production, replace these origins with the deployed React/Next.js domains.
frontend_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _new_thread_id(session_id: str | None = None) -> str:
    """Return a stable LangGraph thread id for one medical consultation."""
    return session_id or str(uuid.uuid4())


def _config(thread_id: str) -> dict[str, dict[str, str]]:
    """Build the LangGraph config used by invoke, resume and get_state."""
    return {"configurable": {"thread_id": thread_id}}


def _initial_medical_state(
    *,
    patient_case: str,
    thread_id: str,
    patient_id: str | None = None,
    patient_name: str | None = None,
    symptoms: list[str] | None = None,
    medical_history: list[str] | None = None,
) -> dict[str, Any]:
    """
    Create a complete MedicalState-compatible initial payload.

    The defaults keep the graph deterministic and prepare the API for future
    MongoDB/PostgreSQL persistence without changing the public contract.
    """
    return {
        "messages": [],
        "patient_case": patient_case,
        "patient_id": patient_id,
        "patient_name": patient_name,
        "session_id": thread_id,
        "symptoms": symptoms or [],
        "medical_history": medical_history or [],
        "question_count": 0,
        "patient_responses": [],
        "asked_questions": [],
        "contextual_symptoms": [],
        "clinical_memory": {},
        "current_question": None,
        "patient_answer": None,
        "diagnostic_summary": None,
        "interim_care": None,
        "mcp_context": None,
        "rag_context": None,
        "clinical_score": 0,
        "severity_level": None,
        "clinical_category": None,
        "physician_treatment": None,
        "final_report": None,
        "consultation_finished": False,
        "next": None,
    }


def _encode(data: Any) -> Any:
    """Convert LangChain/LangGraph objects into JSON-safe data."""
    return jsonable_encoder(data)


def _extract_interrupt_payload(data: Any) -> dict[str, Any] | None:
    """Return the first LangGraph interrupt payload, if present."""
    if not isinstance(data, dict):
        return None

    interrupt_data = data.get("__interrupt__")
    if not interrupt_data:
        return None

    first = interrupt_data[0] if isinstance(interrupt_data, list) else interrupt_data
    if isinstance(first, dict):
        value = first.get("value", first)
        return value if isinstance(value, dict) else {"type": "unknown", "message": str(value)}

    value = getattr(first, "value", first)
    return value if isinstance(value, dict) else {"type": "unknown", "message": str(value)}


def _response_status(data: Any) -> str:
    """Map a graph result to a frontend-friendly consultation status."""
    interrupt_payload = _extract_interrupt_payload(data)
    if interrupt_payload:
        interrupt_type = interrupt_payload.get("type")
        if interrupt_type == "patient_question":
            return "waiting_patient_response"
        if interrupt_type == "physician_review":
            return "waiting_physician_review"
        return "waiting_human_input"

    if isinstance(data, dict) and data.get("final_report"):
        return "finished"
    if isinstance(data, dict) and data.get("consultation_finished"):
        return "finished"
    return "running"


def _compact_workflow_data(data: Any) -> Any:
    """Keep resume responses lightweight while preserving clinical state fields."""
    if not isinstance(data, dict):
        return data
    compacted = dict(data)
    messages = compacted.get("messages")
    if isinstance(messages, list) and len(messages) > 3:
        compacted["messages"] = messages[-3:]
        compacted["message_count"] = len(messages)
    return compacted


def _workflow_response(
    thread_id: str,
    data: Any,
    *,
    compact: bool = False,
    performance_ms: float | None = None,
) -> dict[str, Any]:
    """Build a stable API response for HITL frontend clients."""
    encoded = _encode(_compact_workflow_data(data) if compact else data)
    status = _response_status(encoded)
    response = {
        "status": status,
        "thread_id": thread_id,
        "data": encoded,
        "interrupt": _extract_interrupt_payload(encoded),
    }
    if performance_ms is not None:
        response["performance_ms"] = performance_ms
    return response


def _workflow_has_started(values: dict[str, Any]) -> bool:
    """Return True when a thread has at least entered the graph."""
    return bool(values.get("messages") or values.get("patient_case") or values.get("next"))


def _get_workflow_state(thread_id: str) -> dict[str, Any]:
    """
    Fetch the latest workflow state for a LangGraph thread.

    Requires graph.py to compile the graph with a checkpointer.
    """
    if not thread_id.strip():
        raise HTTPException(status_code=400, detail="thread_id is required.")

    try:
        snapshot = graph.get_state(_config(thread_id))
    except Exception as exc:
        logger.exception("workflow_state_load_failed", extra={"thread_id": thread_id})
        raise HTTPException(
            status_code=500,
            detail="Unable to load workflow state. Check that LangGraph checkpointer is enabled.",
        ) from exc

    values = dict(getattr(snapshot, "values", {}) or {})
    if not values:
        logger.warning("thread_not_found", extra={"thread_id": thread_id})
        raise HTTPException(
            status_code=404,
            detail={
                "status": "error",
                "error": "thread_not_found",
                "thread_id": thread_id,
                "message": (
                    "Consultation thread not found. Use the thread_id returned by "
                    "POST /sessions/start or POST /consultation/start. "
                    "If you entered a patient id such as P001, use GET /patient/P001 instead."
                ),
                "next_steps": [
                    "POST /sessions/start to create a thread_id",
                    "POST /consultation/start with session_id equal to that thread_id",
                    "GET /patient/{patient_id} to load patient history",
                ],
            },
        )

    if not _workflow_has_started(values):
        logger.warning("workflow_not_started", extra={"thread_id": thread_id})
        raise HTTPException(
            status_code=409,
            detail={
                "status": "error",
                "error": "workflow_not_started",
                "thread_id": thread_id,
                "message": "Workflow exists but has not started. Call POST /consultation/start first.",
            },
        )

    logger.info("thread_loaded", extra={"thread_id": thread_id})
    return values


def _invoke_graph(initial_state: dict[str, Any], thread_id: str) -> dict[str, Any]:
    """Run the LangGraph workflow and return a JSON-serializable response."""
    try:
        logger.info("workflow_started", extra={"thread_id": thread_id})
        with perf_timer("workflow_start", thread_id=thread_id) as stats:
            result = graph.invoke(initial_state, config=_config(thread_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("workflow_start_failed", extra={"thread_id": thread_id})
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {exc}") from exc

    if isinstance(result, dict) and result.get("final_report"):
        logger.info("report_generated", extra={"thread_id": thread_id})

    return _workflow_response(thread_id, result, compact=True, performance_ms=stats["elapsed_ms"])


def _resume_graph(thread_id: str, resume: Any) -> dict[str, Any]:
    """Resume a LangGraph workflow paused by interrupt()."""
    if not thread_id.strip():
        raise HTTPException(status_code=400, detail="thread_id is required.")

    try:
        logger.info("workflow_resumed", extra={"thread_id": thread_id})
        with perf_timer("resume_workflow", thread_id=thread_id) as stats:
            result = graph.invoke(Command(resume=resume), config=_config(thread_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("workflow_resume_failed", extra={"thread_id": thread_id})
        raise HTTPException(status_code=500, detail=f"Workflow resume failed: {exc}") from exc

    if isinstance(result, dict) and result.get("final_report"):
        logger.info("report_generated", extra={"thread_id": thread_id})

    return _workflow_response(thread_id, result, compact=True, performance_ms=stats["elapsed_ms"])


@app.get("/health")
def health() -> dict[str, Any]:
    """
    Return service health and integration status.

    Useful for LangGraph Studio, deployment probes, and a future React/Next.js
    frontend.
    """
    try:
        import reportlab  # noqa: F401

        pdf_export_available = True
    except ImportError:
        pdf_export_available = False

    return {
        "status": "healthy",
        "service": "medical-multiagents-api",
        "version": API_VERSION,
        "graph_id": GRAPH_ID,
        "checkpointer": "MemorySaver",
        "langsmith_tracing": os.getenv("LANGSMITH_TRACING", "false"),
        "simulation_mode": os.getenv("MEDICAL_SIMULATION_MODE", "false"),
        "pdf_export_available": pdf_export_available,
        "memory_backend": "in_memory",
        "database_ready": {
            "mongodb": False,
            "postgresql": False,
        },
    }


@app.post("/sessions/start")
def start_session(payload: SessionStartRequest) -> dict[str, Any]:
    """
    Create a consultation session id before the medical workflow starts.

    The returned thread_id must be reused by the frontend for start, resume,
    state loading, report loading and PDF export.
    """
    thread_id = _new_thread_id(payload.session_id)
    logger.info("session_created", extra={"thread_id": thread_id})
    return {
        "status": "ok",
        "thread_id": thread_id,
        "data": {
            "patient_id": payload.patient_id,
            "patient_name": payload.patient_name,
            "session_id": thread_id,
            "next": "consultation/start",
        },
    }


@app.post("/consultation/start")
def consultation_start(payload: ConsultationStartRequest) -> dict[str, Any]:
    """
    Start the full medical multi-agent workflow.

    The graph may finish directly in simulation mode or pause on interrupt()
    when real patient/physician input is required.
    """
    patient_case = validate_patient_input(payload.patient_case)
    thread_id = _new_thread_id(payload.session_id)
    initial_state = _initial_medical_state(
        patient_case=patient_case,
        patient_id=payload.patient_id,
        patient_name=payload.patient_name,
        thread_id=thread_id,
        symptoms=payload.symptoms,
        medical_history=payload.medical_history,
    )
    return _invoke_graph(initial_state, thread_id)


@app.post("/consultation/resume")
def consultation_resume(payload: ConsultationResumeRequest) -> dict[str, Any]:
    """
    Resume a paused consultation using LangGraph Command(resume=...).

    Patient answers and physician decisions should both be sent here by the
    frontend using the same thread_id returned by /consultation/start.
    """
    return _resume_graph(payload.thread_id, payload.resume)


@app.get("/consultation/{thread_id}")
def consultation_state(thread_id: str) -> dict[str, Any]:
    """Return the latest workflow state for one consultation thread."""
    values = _get_workflow_state(thread_id)
    response = _workflow_response(thread_id, values)
    response["workflow_state"] = response["data"]
    return response


@app.get("/consultation/{thread_id}/report")
def consultation_report(thread_id: str) -> dict[str, Any]:
    """Return the final report for a consultation, if already generated."""
    values = _get_workflow_state(thread_id)
    final_report = values.get("final_report")

    if not final_report:
        raise HTTPException(
            status_code=404,
            detail={
                "status": "error",
                "error": "report_not_ready",
                "thread_id": thread_id,
                "message": "The final report is not available yet. Complete the workflow first.",
                "next_steps": [
                    "Continue the workflow with POST /consultation/resume if it is interrupted",
                    "Use GET /consultation/{thread_id} to inspect the current workflow_state",
                ],
            },
        )

    logger.info("report_loaded", extra={"thread_id": thread_id})
    return {
        "status": "ok",
        "thread_id": thread_id,
        "final_report": final_report,
    }


@app.post("/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    """
    Backward-compatible chat endpoint.

    New clients should prefer /consultation/start.
    """
    patient_case = validate_patient_input(payload.patient_case)
    thread_id = _new_thread_id(payload.session_id)
    initial_state = _initial_medical_state(
        patient_case=patient_case,
        patient_id=payload.patient_id,
        patient_name=payload.patient_name,
        thread_id=thread_id,
    )
    return _invoke_graph(initial_state, thread_id)


@app.post("/diagnosis")
def diagnosis(payload: DiagnosisRequest) -> dict[str, Any]:
    """Backward-compatible diagnosis endpoint with symptoms and history."""
    patient_case = validate_patient_input(payload.patient_case)
    thread_id = _new_thread_id(payload.session_id)
    initial_state = _initial_medical_state(
        patient_case=patient_case,
        patient_id=payload.patient_id,
        patient_name=payload.patient_name,
        thread_id=thread_id,
        symptoms=payload.symptoms,
        medical_history=payload.medical_history,
    )
    return _invoke_graph(initial_state, thread_id)


@app.post("/resume")
def resume_workflow(payload: ResumeRequest) -> dict[str, Any]:
    """
    Backward-compatible resume endpoint.

    New clients should prefer /consultation/resume.
    """
    return _resume_graph(payload.thread_id, payload.resume)


@app.get("/report/{thread_id}")
def report(thread_id: str) -> dict[str, Any]:
    """Backward-compatible report endpoint."""
    return consultation_report(thread_id)


@app.post("/export/pdf")
def export_pdf(payload: ReportRequest) -> Response:
    """
    Export a generated medical Markdown report as PDF.

    The endpoint remains stateless so a frontend can export reports fetched
    from /consultation/{thread_id}/report.
    """
    if not payload.final_report.strip():
        raise HTTPException(status_code=400, detail="final_report cannot be empty.")

    try:
        pdf = generate_report_pdf(payload.final_report, payload.severity_level or "Non renseigné")
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF export failed: {exc}") from exc

    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=rapport-medical.pdf"},
    )


@app.get("/history")
def history(patient_id: str) -> dict[str, Any]:
    """
    Return patient history from the current repository.

    The repository is in-memory today and intentionally isolated so it can
    later be replaced by MongoDB or PostgreSQL.
    """
    if not patient_id.strip():
        raise HTTPException(status_code=400, detail="patient_id is required.")

    return {
        "status": "ok",
        "patient_id": patient_id,
        "history": patient_memory.get_patient_history(patient_id),
    }


@app.get("/patient/{patient_id}")
def patient(patient_id: str) -> dict[str, Any]:
    """Return a patient profile and medical history summary."""
    if not patient_id.strip():
        raise HTTPException(status_code=400, detail="patient_id is required.")

    return {
        "status": "ok",
        "patient_id": patient_id,
        "history": patient_memory.get_patient_history(patient_id),
    }
