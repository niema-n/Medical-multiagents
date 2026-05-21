from __future__ import annotations

import os
import sys

from langchain_core.tools import tool


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from backend.mcp_server.server import get_drug_info, search_symptoms  # noqa: E402


def _format_list(values: list[str]) -> str:
    return ", ".join(values) if values else "Non determine"


def _severity_rank(value: str | None) -> int:
    text = (value or "").lower()
    if "critique" in text:
        return 4
    if "eleve" in text or "urgent" in text:
        return 3
    if "modere" in text:
        return 2
    if "faible" in text:
        return 1
    return 0


def _probability_rank(value: str | None) -> int:
    text = (value or "").lower()
    if "probable" in text:
        return 3
    if "possible" in text:
        return 2
    if "evaluer" in text or "exclure" in text:
        return 1
    return 0


def _rank_pathologies(pathologies: list[dict]) -> list[dict]:
    return sorted(
        pathologies,
        key=lambda item: (
            _severity_rank(item.get("gravite")),
            _probability_rank(item.get("probabilite")),
            len(item.get("symptomes_support", []) or []),
        ),
        reverse=True,
    )


def _severity_from_score(score: int | None, fallback: str = "faible") -> str:
    if score is None:
        return fallback
    if score >= 12:
        return "critique"
    if score >= 7:
        return "eleve"
    if score >= 3:
        return "modere"
    return "faible"


def format_mcp_symptom_result(
    result: dict,
    clinical_score: int | None = None,
    severity_level: str | None = None,
    clinical_category: str | None = None,
) -> str:
    """
    Format MCP knowledge while keeping the main clinical score authoritative.
    MCP remains a knowledge source; final severity is harmonized by diagnostic_agent.
    """
    mcp_score = int(result.get("clinical_score_hint", 0) or 0)
    mcp_severity = result.get("severity_level", "faible")
    final_severity = severity_level or _severity_from_score(clinical_score, mcp_severity)

    lines = [
        "Analyse MCP medicale :",
        f"- Categories cliniques MCP : {_format_list(result.get('categories', []))}",
    ]
    if clinical_category:
        lines.append(f"- Categorie clinique finale : {clinical_category}")
    if clinical_score is not None:
        lines.extend([
            f"- Score clinique final harmonise : {clinical_score}",
            f"- Niveau de gravite final : {final_severity}",
            f"- Score MCP indicatif brut : {mcp_score} ({mcp_severity})",
        ])
        if _severity_rank(final_severity) > _severity_rank(mcp_severity):
            lines.append("- Note : le score final prime car il integre les reponses contextualisees et les facteurs transversaux.")
    else:
        lines.extend([
            f"- Niveau de gravite MCP : {mcp_severity}",
            f"- Score clinique indicatif MCP : {mcp_score}",
        ])
    lines.append(f"- Recommandation prudente : {result.get('recommendation', 'Surveillance prudente.')}")

    normalized_symptoms = result.get("normalized_symptoms", [])
    if normalized_symptoms:
        lines.append("\nSymptomes normalises principaux :")
        for item in normalized_symptoms[:6]:
            aliases = _format_list((item.get("matched_aliases", []) or [])[:3])
            lines.append(
                f"- {item['symptom']} | categorie: {item['category']} | "
                f"gravite: {item['severity']} | reconnu via: {aliases}"
            )
        if len(normalized_symptoms) > 6:
            lines.append(f"- {len(normalized_symptoms) - 6} autre(s) symptome(s) reconnu(s) non affiche(s).")

    pathologies = result.get("pathologies", [])
    if pathologies:
        ranked_pathologies = _rank_pathologies(pathologies)[:5]
        lines.append("\nOrientations MCP prioritaires a discuter :")
        for pathology in ranked_pathologies:
            categories = _format_list(pathology.get("categories", []))
            support = _format_list((pathology.get("symptomes_support", []) or [])[:3])
            lines.append(
                f"- {pathology['pathologie']} "
                f"(probabilite: {pathology['probabilite']}, gravite: {pathology['gravite']}, "
                f"categorie: {categories}, support: {support})"
            )
        if len(pathologies) > len(ranked_pathologies):
            lines.append(f"- {len(pathologies) - len(ranked_pathologies)} orientation(s) moins pertinente(s) masquee(s).")

    red_flags = result.get("red_flags", [])
    if red_flags:
        lines.append(f"\nSignes d'alerte detectes : {_format_list(red_flags)}")

    lines.append(f"\nAttention : {result['disclaimer']}")
    return "\n".join(lines)


@tool
def mcp_search_symptoms(symptom: str) -> str:
    """
    Interroge le serveur MCP pour analyser un symptome ou un texte clinique.
    """
    result = search_symptoms(symptom=symptom)

    if not result["found"]:
        return (
            "Aucune donnee MCP specifique n'a ete trouvee.\n"
            f"- Message : {result.get('message', 'Symptome non reconnu.')}\n"
            "- Recommandation prudente : demander un avis medical en cas de doute, persistance ou aggravation."
        )

    return format_mcp_symptom_result(result)


@tool
def mcp_get_drug_info(drug_name: str) -> str:
    """
    Interroge le serveur MCP pour obtenir des informations prudentes sur un medicament.
    """
    result = get_drug_info(drug_name=drug_name)

    if not result["found"]:
        return result["message"]

    return (
        f"{result['drug'].capitalize()} ({result['classe']})\n"
        f"- Posologie adulte : {result['posologie_adulte']}\n"
        f"- Contre-indications : {result['contre_indications']}\n"
        f"- Interactions : {result['interactions']}\n"
        f"- Remarque : {result['remarque']}\n"
        f"- Attention : {result['disclaimer']}"
    )
