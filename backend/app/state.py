from typing_extensions import TypedDict
from typing import Annotated, List, Dict, Any, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class MedicalState(TypedDict, total=False):
    """State for the medical diagnostic workflow"""
    
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Patient information
    patient_case: Optional[str]
    session_id: Optional[str]
    patient_id: Optional[str]
    patient_name: Optional[str]
    age: Optional[int]
    symptoms: List[str]
    medical_history: List[str]
    patient_responses: List[str]
    asked_questions: List[str]
    question_count: int
    current_question: Optional[str]
    patient_answer: Optional[str]
    symptom_history: List[str]
    contextual_symptoms: List[str]
    clinical_memory: Dict[str, Any]
    
    # Diagnostic information
    diagnosis: Optional[str]
    diagnostic_summary: Optional[str]
    previous_diagnoses: List[str]
    confidence: Optional[float]
    differential_diagnosis: List[str]
    interim_care: Optional[str]
    clinical_score: int
    severity_level: Optional[str]
    clinical_category: Optional[str]
    mcp_context: Optional[str]
    rag_context: Optional[str]
    
    # Treatment plan
    treatment_plan: Optional[str]
    medications: List[str]
    follow_up: Optional[str]
    
    # Physician review
    physician_approval: Optional[bool]
    physician_decision: Optional[str]
    physician_notes: Optional[str]
    physician_treatment: Optional[str]
    
    # Report
    final_report: Optional[str]
    previous_reports: List[str]
    previous_sessions: List[Dict[str, Any]]
    report_status: Optional[str]
    pdf_path: Optional[str]
    consultation_finished: bool
    
    # Metadata
    current_agent: Optional[str]
    next_agent: Optional[str]
    next: Optional[str]
