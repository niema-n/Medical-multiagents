from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class PatientSession:
    session_id: str
    patient_id: str
    created_at: str
    symptoms: list[str] = field(default_factory=list)
    treatments: list[str] = field(default_factory=list)
    report: str | None = None
    diagnosis: str | None = None


class PatientMemoryRepository:
    """In-memory repository, swappable later for MongoDB/PostgreSQL."""

    def __init__(self) -> None:
        self._sessions: dict[str, list[PatientSession]] = {}

    def save_session(self, patient_id: str, session_id: str, payload: dict[str, Any]) -> None:
        session = PatientSession(
            patient_id=patient_id,
            session_id=session_id,
            created_at=datetime.utcnow().isoformat(),
            symptoms=list(payload.get("symptoms", [])),
            treatments=list(payload.get("treatments", [])),
            report=payload.get("report"),
            diagnosis=payload.get("diagnosis"),
        )
        self._sessions.setdefault(patient_id, []).append(session)

    def get_patient_history(self, patient_id: str) -> dict[str, Any]:
        sessions = self._sessions.get(patient_id, [])
        return {
            "previous_sessions": [session.__dict__ for session in sessions],
            "previous_reports": [session.report for session in sessions if session.report],
            "previous_diagnoses": [session.diagnosis for session in sessions if session.diagnosis],
            "symptom_history": [symptom for session in sessions for symptom in session.symptoms],
            "treatment_history": [treatment for session in sessions for treatment in session.treatments],
        }


patient_memory = PatientMemoryRepository()
