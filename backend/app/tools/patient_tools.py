from langchain_core.tools import tool


@tool
def ask_patient(question: str) -> str:
    """
    Pose une question médicale au patient et retourne sa réponse.

    Args:
        question: La question médicale à poser au patient.

    Returns:
        La réponse du patient (saisie via l'API en production,
        simulée ici pour les tests unitaires).
    """
    # En production : cette valeur sera injectée depuis l'API FastAPI.
    # En test : on retourne un marqueur spécial pour indiquer
    # que le graphe doit s'interrompre et attendre la réponse.
    return "__AWAITING_PATIENT_RESPONSE__"