from __future__ import annotations

import re
from typing import Any


DANGEROUS_PROMPT_PATTERNS = [
    r"ignore\s+(les|toutes)\s+instructions",
    r"diagnostic\s+definitif",
    r"prescris\s+moi",
    r"ordonnance",
    r"suicide|automutilation",
]


def validate_patient_input(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Le message patient ne peut pas être vide.")
    if len(cleaned) > 4000:
        raise ValueError("Le message patient est trop long.")

    lowered = cleaned.lower()
    if any(re.search(pattern, lowered) for pattern in DANGEROUS_PROMPT_PATTERNS):
        raise ValueError("Le message contient une demande médicale ou système non autorisée.")

    return cleaned


def mask_sensitive_data(text: Any) -> str:
    masked = str(text or "")
    masked = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[email masqué]", masked)
    masked = re.sub(r"\b(?:\+?\d[\s.-]?){8,}\b", "[téléphone masqué]", masked)
    return masked


def validate_medical_response(text: str) -> str:
    response = text or ""
    forbidden = ["diagnostic définitif", "diagnostic certain", "vous avez forcément"]
    if any(term in response.lower() for term in forbidden):
        response += "\n\nNote de sécurité : cette sortie doit être reformulée comme une orientation clinique préliminaire."
    return response
