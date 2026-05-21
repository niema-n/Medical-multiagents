from backend.app.state import MedicalState


def supervisor(state: MedicalState) -> MedicalState:
    """
    Orchestre le workflow en lisant state["next"].

    Si aucun "next" n'est defini, on demarre par le diagnostic.
    Le diagnostic_agent gere lui-meme la limite de questions et produit
    la synthese clinique avant de passer au physician_review.
    """
    next_node = state.get("next")

    if next_node is None:
        return {"next": "diagnostic_agent"}

    return {"next": next_node}


def route_after_supervisor(state: MedicalState) -> str:
    return state.get("next", "diagnostic_agent")
