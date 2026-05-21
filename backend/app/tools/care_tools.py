from __future__ import annotations

import os

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from backend.app.services.clinical_scoring import (
    calculate_clinical_score,
    classify_clinical_category,
)


DISCLAIMER = "Ce systeme ne remplace pas une consultation medicale."


def _normalise_severity(value: str | None) -> str:
    value = (value or "faible").strip().lower()
    if value in {"eleve", "elevé", "éleve", "élevé"}:
        return "eleve"
    if value in {"modere", "modéré"}:
        return "modere"
    if value == "critique":
        return "critique"
    return "faible"


def _has_category(clinical_category: str | None, category: str) -> bool:
    return category.lower() in (clinical_category or "").lower()


def _category_advice(clinical_category: str | None, severity: str) -> str:
    if _has_category(clinical_category, "Cardiaque"):
        return (
            "- Surveiller douleur thoracique, irradiation, sueurs, malaise et essoufflement.\n"
            "- Eviter l'effort; urgence si douleur persistante ou signes associes."
        )
    if _has_category(clinical_category, "Respiratoire"):
        return (
            "- Surveiller respiration, capacite a parler, fievre et saturation si disponible.\n"
            "- Urgence si gene respiratoire importante, saturation basse, confusion ou aggravation rapide."
        )
    if _has_category(clinical_category, "Neurologique"):
        return (
            "- Surveiller confusion, somnolence, trouble de la parole, faiblesse d'un cote ou convulsions.\n"
            "- Urgence/SAMU si apparition brutale, trouble de conscience ou deficit neurologique."
        )
    if _has_category(clinical_category, "Infectieux") or _has_category(clinical_category, "Febrile"):
        return (
            "- Surveiller temperature mesuree, frissons, hydratation, fatigue, evolution et symptomes localisateurs.\n"
            "- Avis medical rapide si fievre elevee ou persistante, confusion, raideur de nuque, gene respiratoire, malaise, taches cutanees ou terrain fragile."
        )
    if _has_category(clinical_category, "Digestif"):
        return (
            "- Surveiller douleur abdominale, vomissements, diarrhee, hydratation, fievre et sang dans les selles.\n"
            "- Avis rapide si douleur localisee intense, ventre dur, vomissements repetes ou impossibilite de boire."
        )
    if _has_category(clinical_category, "Musculo-articulaire"):
        return (
            "- Surveiller localisation, intensite, gonflement, rougeur, chaleur, deformation et mobilite du membre.\n"
            "- Avis medical rapide si douleur 7/10 ou plus, traumatisme, gonflement important, aggravation ou difficulte a bouger."
        )
    if _has_category(clinical_category, "Urinaire"):
        return (
            "- Surveiller fievre, frissons, douleur lombaire, sang dans les urines et possibilite de boire normalement.\n"
            "- Avis medical rapide si fievre, douleur lombaire, grossesse, malaise ou sang dans les urines."
        )
    if _has_category(clinical_category, "Dermatologique"):
        return (
            "- Surveiller extension, demangeaisons, douleur, chaleur locale, fievre et gonflement du visage ou des levres.\n"
            "- Urgence si gene respiratoire, gonflement du visage, malaise ou eruption tres etendue."
        )
    if _has_category(clinical_category, "ORL"):
        return (
            "- Surveiller fievre, hydratation, douleur, difficulte a avaler et gene respiratoire.\n"
            "- Avis rapide si impossibilite d'avaler, fievre elevee, douleur intense ou aggravation."
        )
    if severity in {"eleve", "critique"}:
        return "- Surveiller douleur, malaise, respiration, conscience et aggravation rapide."
    return "- Hydratation reguliere, repos adapte et reevaluation si apparition de nouveaux signes."


def _score_advice(clinical_score: int) -> str:
    if clinical_score >= 12:
        return "- Le score clinique eleve impose de ne pas banaliser la situation et de privilegier une evaluation urgente."
    if clinical_score >= 7:
        return "- Le score clinique justifie un avis medical rapide, surtout si les symptomes persistent ou progressent."
    if clinical_score >= 3:
        return "- Le score clinique reste intermediaire : reevaluer l'evolution et consulter si absence d'amelioration."
    return "- Le score clinique est faible, mais une aggravation ou un nouveau signe doit faire demander un avis medical."


def _simulate_care(
    diagnostic_summary: str,
    patient_responses: list[str],
    patient_case: str | None = None,
    clinical_score: int | None = None,
    severity_level: str | None = None,
    clinical_category: str | None = None,
) -> str:
    if clinical_score is None or not severity_level:
        clinical_score, severity_level = calculate_clinical_score(patient_case, patient_responses)
    if not clinical_category:
        clinical_category = classify_clinical_category(patient_case, patient_responses)

    severity = _normalise_severity(severity_level)

    if severity == "critique":
        return (
            f"Recommandation intermediaire : niveau critique, categorie(s) clinique(s) "
            f"{clinical_category}, score clinique {clinical_score}.\n"
            "- Orientation urgente recommandee : contacter les urgences ou le SAMU sans delai.\n"
            "- Ne pas conduire soi-meme et rester accompagne si possible.\n"
            f"{_score_advice(clinical_score)}\n"
            f"{_category_advice(clinical_category, severity)}\n"
            "- Cette orientation clinique preliminaire ne constitue pas un diagnostic definitif.\n"
            f"{DISCLAIMER}"
        )

    if severity == "eleve":
        return (
            f"Recommandation intermediaire : niveau eleve, categorie(s) clinique(s) "
            f"{clinical_category}, score clinique {clinical_score}.\n"
            "- Avis medical rapide recommande, surtout en cas d'aggravation ou de symptome persistant.\n"
            f"{_score_advice(clinical_score)}\n"
            f"{_category_advice(clinical_category, severity)}\n"
            "- Eviter l'effort et rester sous surveillance d'un proche si possible.\n"
            "- Cette orientation clinique preliminaire doit etre confirmee par un professionnel de sante.\n"
            f"{DISCLAIMER}"
        )

    if severity == "modere":
        return (
            f"Recommandation intermediaire : niveau modere, categorie(s) clinique(s) "
            f"{clinical_category}, score clinique {clinical_score}.\n"
            "- Programmer un avis medical si les symptomes persistent ou s'aggravent.\n"
            f"{_score_advice(clinical_score)}\n"
            f"{_category_advice(clinical_category, severity)}\n"
            "- Utiliser uniquement les traitements habituels ou deja prescrits, selon les consignes connues.\n"
            "- Consulter rapidement en cas de douleur thoracique, malaise, essoufflement ou fievre elevee.\n"
            f"{DISCLAIMER}"
        )

    return (
        f"Recommandation intermediaire : niveau faible, categorie(s) clinique(s) "
        f"{clinical_category}, score clinique {clinical_score}.\n"
        "- Surveillance simple des symptomes et repos adapte.\n"
        f"{_score_advice(clinical_score)}\n"
        f"{_category_advice(clinical_category, severity)}\n"
        "- Demander un avis medical si les symptomes persistent, s'aggravent ou deviennent inhabituels.\n"
        f"{DISCLAIMER}"
    )


@tool
def recommend_interim_care(
    diagnostic_summary: str,
    patient_responses: list[str],
    patient_case: str | None = None,
    clinical_score: int | None = None,
    severity_level: str | None = None,
    clinical_category: str | None = None,
) -> str:
    """
    Genere une recommandation intermediaire prudente selon la meme evaluation
    clinique que le diagnostic agent et le rapport final.
    """
    if clinical_score is None or not severity_level:
        clinical_score, severity_level = calculate_clinical_score(patient_case, patient_responses)
    if not clinical_category:
        clinical_category = classify_clinical_category(patient_case, patient_responses)

    api_key = os.getenv("OPENAI_API_KEY", "")
    if (
        not api_key
        or os.getenv("MEDICAL_SIMULATION_MODE", "false").lower() == "true"
        or os.getenv("MEDICAL_LLM_CARE", "false").lower() != "true"
    ):
        return _simulate_care(
            diagnostic_summary=diagnostic_summary,
            patient_responses=patient_responses,
            patient_case=patient_case,
            clinical_score=clinical_score,
            severity_level=severity_level,
            clinical_category=clinical_category,
        )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, timeout=20, api_key=api_key)
    prompt = f"""Tu es un assistant medical prudent.
Genere une recommandation intermediaire claire, realiste et non definitive.

Synthese clinique :
{diagnostic_summary}

Reponses du patient :
{chr(10).join(f'- {response}' for response in patient_responses)}

Evaluation algorithmique a respecter strictement :
- Score clinique : {clinical_score}
- Niveau de gravite : {severity_level}
- Categorie(s) clinique(s) : {clinical_category}

Regles :
- Ne formule jamais de diagnostic definitif.
- Utilise les termes "orientation clinique preliminaire" et "recommandation intermediaire".
- N'abaisse jamais le niveau de gravite par rapport a l'evaluation algorithmique.
- Termine toujours par : "{DISCLAIMER}"
- Reste concis, professionnel et academique.
"""
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception:
        return _simulate_care(
            diagnostic_summary=diagnostic_summary,
            patient_responses=patient_responses,
            patient_case=patient_case,
            clinical_score=clinical_score,
            severity_level=severity_level,
            clinical_category=clinical_category,
        )
