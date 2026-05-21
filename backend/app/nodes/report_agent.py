from datetime import datetime

from langchain_core.messages import AIMessage

from backend.app.services.patient_memory import patient_memory
from backend.app.state import MedicalState


DISCLAIMER = "Ce systeme ne remplace pas une consultation medicale."


def _safe_text(value: object, default: str = "Non renseigne") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _format_optional_line(label: str, value: object) -> str:
    text = _safe_text(value, "")
    return f"- {label}: {text}" if text else ""


def _format_patient_responses(responses: object) -> str:
    if not isinstance(responses, list) or not responses:
        return "- Aucune reponse patient collectee."
    return "\n".join(f"- Q{index}: {_safe_text(response)}" for index, response in enumerate(responses, start=1))


def _format_contextual_symptoms(symptoms: object) -> str:
    if not isinstance(symptoms, list) or not symptoms:
        return "- Aucun symptome implicite significatif extrait."
    return "\n".join(f"- {_safe_text(symptom)}" for symptom in symptoms[:5])


def _format_clinical_memory(memory: object) -> str:
    if not isinstance(memory, dict) or not memory:
        return "- Memoire clinique non disponible."

    lines = []
    confirmed = memory.get("confirmed_symptoms") or []
    denied = memory.get("denied_symptoms") or []
    medications = memory.get("medications") or []
    red_flags = memory.get("red_flags") or []

    if memory.get("dominant_category"):
        lines.append(f"- Categorie dominante retenue: {_safe_text(memory.get('dominant_category'))}")
    if memory.get("temperature"):
        lines.append(f"- Temperature deja indiquee: {_safe_text(memory.get('temperature'))} C")
    if confirmed:
        lines.append(f"- Symptomes confirmes: {', '.join(_safe_text(item) for item in confirmed[:6])}")
    if denied:
        lines.append(f"- Symptomes nies: {', '.join(_safe_text(item) for item in denied[:6])}")
    if medications:
        lines.append(f"- Medicaments mentionnes: {', '.join(_safe_text(item) for item in medications[:5])}")
    if red_flags:
        lines.append(f"- Signes de gravite detectes: {', '.join(_safe_text(item) for item in red_flags[:5])}")
    if memory.get("denied_red_flags"):
        lines.append("- Signes d'alerte explicitement nies dans les reponses.")

    return "\n".join(lines) if lines else "- Aucun element memorise significatif."


def _split_mcp_data(diagnostic_summary: str) -> tuple[str, str]:
    markers = ["--- Données MCP ---", "--- Donnees MCP ---", "--- Donn"]
    for marker in markers:
        if marker in diagnostic_summary:
            summary, mcp_data = diagnostic_summary.split(marker, 1)
            return summary.strip(), mcp_data.strip()
    return diagnostic_summary.strip(), ""


def _clean_summary(diagnostic_summary: str) -> str:
    summary, _ = _split_mcp_data(diagnostic_summary)
    for marker in ["\n\nMCP retenu :", "\n\nRAG retenu :", "\n\nDonnees MCP", "\n\nContexte RAG"]:
        if marker in summary:
            summary = summary.split(marker, 1)[0]
    return summary.strip()


def _priority_from_severity(severity_level: str) -> str:
    mapping = {
        "faible": "Faible",
        "modéré": "Modere",
        "modere": "Modere",
        "élevé": "Eleve",
        "eleve": "Eleve",
        "critique": "Critique",
    }
    return mapping.get(severity_level.lower(), "Faible")


def _compact_context(value: str, limit: int = 950) -> str:
    if not value:
        return ""
    return value if len(value) <= limit else f"{value[:limit].rstrip()}\n\n[Contexte tronque dans le rapport]"


def _compact_evidence(value: str, limit: int = 1100) -> str:
    text = _compact_context(value, limit=limit)
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    useful = [
        line
        for line in lines
        if line.startswith(("-", "[RAG", "Analyse MCP", "Orientations", "Symptomes", "Signes"))
        or "source=" in line
    ]
    return "\n".join(useful[:12]) if useful else text


def report_agent(state: MedicalState) -> MedicalState:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    patient_case = _safe_text(state.get("patient_case"))
    patient_responses = state.get("patient_responses", [])
    contextual_symptoms = state.get("contextual_symptoms", [])
    clinical_memory = state.get("clinical_memory", {})
    raw_diagnostic_summary = _safe_text(state.get("diagnostic_summary"), "Non disponible")
    diagnostic_summary, mcp_data_from_summary = _split_mcp_data(raw_diagnostic_summary)
    diagnostic_summary = _clean_summary(diagnostic_summary)
    mcp_data = _safe_text(state.get("mcp_context"), mcp_data_from_summary)
    rag_context = _safe_text(state.get("rag_context"), "")
    interim_care = _safe_text(state.get("interim_care"), "Non disponible")
    physician_treatment = _safe_text(
        state.get("physician_treatment"),
        "Avis du medecin traitant non encore renseigne.",
    )
    clinical_score = state.get("clinical_score", 0)
    severity_level = _safe_text(state.get("severity_level"), "faible")
    clinical_category = _safe_text(state.get("clinical_category"), "General")
    priority_level = _priority_from_severity(severity_level)

    patient_info_lines = [
        _format_optional_line("Identifiant patient", state.get("patient_id")),
        _format_optional_line("Nom du patient", state.get("patient_name")),
        _format_optional_line("Session", state.get("session_id")),
        f"- Date de generation: {generated_at}",
    ]
    patient_info = "\n".join(line for line in patient_info_lines if line)

    report = f"""# Rapport Clinique Preliminaire

## 1. Synthese
**Patient:** {state.get('patient_id', 'Non renseigne')} | **Session:** {state.get('session_id', 'Non renseigne')} | **Date:** {generated_at}
**Cas initial:** {patient_case}
**Reponses:**
{_format_patient_responses(patient_responses)}

**Resume clinique:**
{diagnostic_summary}

## 2. Evaluation
- **Score clinique:** {clinical_score}
- **Niveau de gravite:** {severity_level} (Priorite: {priority_level})
- **Categorie dominante:** {clinical_category}

## 3. Recommandations & Conduite a tenir
**Recommandation intermediaire:**
{interim_care}

**Avis medical:**
{physician_treatment}

## 4. Contexte MCP / RAG
**MCP:**
{_compact_evidence(mcp_data) if mcp_data else "Aucune donnee MCP specifique."}

**RAG:**
{_compact_evidence(rag_context) if rag_context else "Aucun contexte RAG."}

## Disclaimer
*{DISCLAIMER} Ce rapport est une orientation preliminaire et ne constitue pas un diagnostic definitif.*
""".strip()

    patient_id = state.get("patient_id")
    session_id = state.get("session_id")
    if patient_id and session_id:
        patient_memory.save_session(
            patient_id=patient_id,
            session_id=session_id,
            payload={
                "symptoms": state.get("symptoms", []),
                "treatments": [physician_treatment],
                "report": report,
                "diagnosis": diagnostic_summary,
            },
        )

    return {
        "messages": [AIMessage(content="[Report Agent] Rapport final genere.", name="report_agent")],
        "final_report": report,
        "consultation_finished": True,
        "next": "FINISH",
    }
