from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    patient_case: str = Field(..., min_length=1)
    patient_id: str | None = None
    patient_name: str | None = None
    session_id: str | None = None


class ResumeRequest(BaseModel):
    thread_id: str
    resume: Any


class DiagnosisRequest(ChatRequest):
    symptoms: list[str] = Field(default_factory=list)
    medical_history: list[str] = Field(default_factory=list)


class PhysicianDecision(BaseModel):
    decision: Literal["approve", "modify", "reject"]
    treatment: str | None = None
    notes: str | None = None


class ReportRequest(BaseModel):
    final_report: str
    severity_level: str | None = None


class ApiResponse(BaseModel):
    status: str
    data: dict[str, Any]
