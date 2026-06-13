import os
import re
import logging
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

from backend.app.rag.retriever import retrieve_medical_context
from backend.app.services.clinical_scoring import (
    calculate_clinical_score,
    classify_clinical_categories,
    classify_clinical_category,
    flatten_text,
    has_positive_keyword,
    infer_contextual_symptoms,
)
from backend.app.services.safety import (
    mask_sensitive_data,
    validate_medical_response,
    validate_patient_input,
)
from backend.app.services.hitl_cache import cached_clinical_categories, cached_contextual_symptoms
from backend.app.services.performance import perf_timer
from backend.app.state import MedicalState
from backend.app.tools.care_tools import recommend_interim_care
from backend.app.tools.mcp_client import format_mcp_symptom_result, mcp_get_drug_info, mcp_search_symptoms
from backend.mcp_server.server import search_symptoms
from backend.app.tools.patient_tools import ask_patient


logger = logging.getLogger("medical_multiagents.diagnostic_agent")

MAX_QUESTIONS = 5
PATIENT_CASE_FALLBACK = "Cas non renseigne."

SYSTEM_PROMPT = """Tu es un assistant medical conversationnel prudent.

Regles strictes :
- Tu poses une seule question a la fois, naturelle et contextualisee.
- Tu utilises toujours le tool ask_patient pour poser une question.
- Tu ne poses jamais plus de {max_q} questions au total.
- Tu adaptes la prochaine question au cas, aux reponses deja obtenues et aux questions deja posees.
- Tu evites toute repetition ou reformulation inutile.
- Tu ne donnes jamais de diagnostic definitif.
- MCP et RAG enrichissent l'analyse; ils ne remplacent pas le raisonnement clinique.
"""


@dataclass(frozen=True)
class ClinicalObjective:
    code: str
    categories: tuple[str, ...]
    triggers: tuple[str, ...]
    already_known: tuple[str, ...]
    question: str
    priority: int = 50
    reason: str = ""


OBJECTIVES = [
    ClinicalObjective(
        code="febrile_temperature",
        categories=("Infectieux/Febrile",),
        triggers=("fievre", "temperature", "frissons", "infection", "doliprane", "paracetamol"),
        already_known=("38", "39", "40", "temperature mesuree", "thermometre", "degres"),
        question="Pouvez-vous preciser la temperature mesuree au thermometre, en degres Celsius, et a quel moment elle a ete prise ?",
        priority=3,
    ),
    ClinicalObjective(
        code="febrile_duration",
        categories=("Infectieux/Febrile",),
        triggers=("fievre", "temperature", "frissons", "infection", "fatigue"),
        already_known=("depuis", "heure", "jour", "semaine", "brutal", "progressif"),
        question="Depuis quand la fievre ou les frissons ont-ils commence, et l'evolution est-elle stable, en hausse ou par poussees ?",
        priority=5,
    ),
    ClinicalObjective(
        code="febrile_associated",
        categories=("Infectieux/Febrile",),
        triggers=("fievre", "temperature", "frissons", "courbatures", "infection", "fatigue"),
        already_known=("toux", "gorge", "urinaire", "ventre", "eruption", "douleur localisee"),
        question="Avez-vous un symptome associe qui oriente la fievre : toux, mal de gorge, brulures urinaires, douleur abdominale, eruption cutanee ou douleur localisee ?",
        priority=8,
    ),
    ClinicalObjective(
        code="febrile_red_flags",
        categories=("Infectieux/Febrile",),
        triggers=("fievre", "temperature", "frissons", "infection"),
        already_known=("confusion", "raideur", "respiration", "malaise", "deshydratation", "tache"),
        question="Y a-t-il un signe d'alerte avec cette fievre : confusion, raideur de nuque, gene respiratoire, malaise, taches cutanees ou difficulte a boire ?",
        priority=10,
    ),
    ClinicalObjective(
        code="febrile_medication",
        categories=("Infectieux/Febrile",),
        triggers=("fievre", "temperature", "frissons", "doliprane", "paracetamol"),
        already_known=("doliprane", "paracetamol", "ibuprofene", "dose", "allergie"),
        question="Avez-vous pris du Doliprane/paracetamol ou un autre medicament, a quelle dose, et avec quel effet sur la temperature ?",
        priority=18,
    ),
    ClinicalObjective(
        code="musculo_location",
        categories=("Musculo-articulaire",),
        triggers=(
            "douleur main",
            "douleur aux mains",
            "main douloureuse",
            "mains douloureuses",
            "mal main",
            "mal mains",
            "mal au main",
            "mal aux main",
            "mal aux mains",
            "douleur bras",
            "douleur jambe",
            "douleur articulation",
            "gonflement",
        ),
        already_known=("main droite", "main gauche", "bras", "jambe", "poignet", "doigt", "genou", "cheville", "articulation"),
        question="Ou se situe exactement la douleur : main droite ou gauche, doigts, poignet, bras, jambe ou une articulation precise ?",
        priority=4,
    ),
    ClinicalObjective(
        code="musculo_intensity",
        categories=("Musculo-articulaire",),
        triggers=("douleur", "mal", "main", "mains", "gonflement", "traumatisme"),
        already_known=("sur 10", "/10", "intensite", "leger", "modere", "intense"),
        question="Sur une echelle de 0 a 10, quelle est l'intensite maximale de cette douleur actuellement ?",
        priority=5,
    ),
    ClinicalObjective(
        code="musculo_swelling",
        categories=("Musculo-articulaire",),
        triggers=("douleur", "mal", "articulation", "main", "bras", "jambe"),
        already_known=("gonflement", "gonfle", "rougeur", "chaud", "chaleur", "deformation"),
        question="La zone douloureuse est-elle gonflee, rouge, chaude, deformee ou plus sensible au toucher ?",
        priority=7,
    ),
    ClinicalObjective(
        code="musculo_trauma",
        categories=("Musculo-articulaire",),
        triggers=("douleur", "mal", "main", "mains", "gonflement", "difficulte a bouger"),
        already_known=("chute", "traumatisme", "coup", "accident", "entorse", "fracture", "faux mouvement"),
        question="Y a-t-il eu une chute, un coup, un faux mouvement, un traumatisme ou un effort inhabituel avant le debut de la douleur ?",
        priority=9,
    ),
    ClinicalObjective(
        code="musculo_mobility",
        categories=("Musculo-articulaire",),
        triggers=("douleur", "mal", "main", "bras", "jambe", "articulation", "gonflement"),
        already_known=("bouger", "mobilite", "plier", "ouvrir", "fermer", "marcher", "porter", "impossible"),
        question="Pouvez-vous bouger normalement la main ou le membre concerne, le plier, l'utiliser ou marcher selon la zone atteinte ?",
        priority=11,
    ),
    ClinicalObjective(
        code="ent_swallowing",
        categories=("ORL",),
        triggers=("mal de gorge", "gorge douloureuse", "angine", "difficulte a avaler"),
        already_known=("avaler", "boire", "respirer", "salive", "ganglion"),
        question="Avec ce mal de gorge, arrivez-vous a avaler et boire normalement, et existe-t-il une gene pour respirer ou avaler la salive ?",
        priority=6,
    ),
    ClinicalObjective(
        code="ent_duration",
        categories=("ORL",),
        triggers=("mal de gorge", "gorge douloureuse", "angine", "nez", "oreille", "sinus", "rhinopharyngite", "rhume", "otite"),
        already_known=("depuis", "jour", "semaine", "brutal", "progressif"),
        question="Depuis quand les symptomes ORL ont-ils commence, et l'evolution est-elle stable, en amelioration ou en aggravation ?",
        priority=8,
    ),
    ClinicalObjective(
        code="ent_fever",
        categories=("ORL",),
        triggers=("mal de gorge", "gorge douloureuse", "angine", "rhinopharyngite", "rhume", "otite", "sinusite"),
        already_known=("fievre", "temperature", "frissons", "ganglion"),
        question="Avez-vous de la fievre mesuree, des frissons, des ganglions douloureux ou une fatigue importante avec ces symptomes ORL ?",
        priority=10,
    ),
    ClinicalObjective(
        code="ent_nasal_ear",
        categories=("ORL",),
        triggers=("mal de gorge", "nez", "oreille", "sinus", "rhinopharyngite", "rhume", "otite", "sinusite"),
        already_known=("nez", "oreille", "sinus", "ecoulement", "voix", "toux"),
        question="Y a-t-il un nez qui coule ou bouche, une douleur d'oreille, une douleur des sinus, une toux ou une voix modifiee ?",
        priority=12,
    ),
    ClinicalObjective(
        code="ent_treatments",
        categories=("ORL",),
        triggers=("mal de gorge", "gorge douloureuse", "angine", "rhinopharyngite", "rhume", "otite", "sinusite"),
        already_known=("traitement", "medicament", "paracetamol", "antibiotique", "allergie"),
        question="Avez-vous deja pris un traitement pour ces symptomes ORL, et avez-vous des allergies ou antecedents ORL importants ?",
        priority=18,
    ),
    ClinicalObjective(
        code="urinary_red_flags",
        categories=("Urinaire",),
        triggers=("brulure urinaire", "infection urinaire", "cystite", "douleur en urinant"),
        already_known=("fievre", "lombaire", "sang", "grossesse", "frissons"),
        question="Avez-vous de la fievre, des frissons, une douleur dans le dos ou les lombaires, du sang dans les urines ou une grossesse possible ?",
        priority=6,
    ),
    ClinicalObjective(
        code="urinary_duration",
        categories=("Urinaire",),
        triggers=("brulure urinaire", "brulures urinaires", "infection urinaire", "cystite", "douleur en urinant", "urines"),
        already_known=("depuis", "jour", "heure", "recidive", "recurrent"),
        question="Depuis quand les symptomes urinaires ont-ils commence, et est-ce un premier episode ou une recidive ?",
        priority=8,
    ),
    ClinicalObjective(
        code="urinary_frequency",
        categories=("Urinaire",),
        triggers=("brulure urinaire", "infection urinaire", "cystite", "urines", "pipi"),
        already_known=("frequent", "souvent", "quantite", "urgent", "envie"),
        question="Avez-vous des envies frequentes ou urgentes d'uriner, et les quantites d'urines sont-elles reduites ou normales ?",
        priority=10,
    ),
    ClinicalObjective(
        code="urinary_appearance",
        categories=("Urinaire",),
        triggers=("brulure urinaire", "infection urinaire", "cystite", "urines", "pipi"),
        already_known=("trouble", "odeur", "sang", "couleur"),
        question="Les urines sont-elles troubles, malodorantes, colorees differemment, ou avez-vous vu du sang ?",
        priority=12,
    ),
    ClinicalObjective(
        code="urinary_risk_factors",
        categories=("Urinaire",),
        triggers=("brulure urinaire", "infection urinaire", "cystite", "urines", "pyelonephrite"),
        already_known=("grossesse", "diabete", "immunodepression", "homme", "antecedent", "traitement"),
        question="Y a-t-il un facteur de risque : grossesse possible, diabete, immunodepression, antecedents urinaires, ou traitement en cours ?",
        priority=18,
    ),
    ClinicalObjective(
        code="dermato_allergy",
        categories=("Dermatologique",),
        triggers=("eruption", "boutons", "urticaire", "demangeaisons", "plaques rouges"),
        already_known=("respirer", "visage", "levres", "allergie", "medicament", "aliment", "extension"),
        question="L'eruption s'etend-elle rapidement, et avez-vous un gonflement du visage ou des levres, une gene respiratoire, ou une exposition recente a un medicament/aliment ?",
        priority=6,
    ),
    ClinicalObjective(
        code="dermato_location",
        categories=("Dermatologique",),
        triggers=("eruption", "boutons", "urticaire", "demangeaisons", "plaques rouges", "eczema", "rash", "piqure"),
        already_known=("visage", "bras", "jambe", "tronc", "localise", "diffus", "partout"),
        question="Ou sont situees les lesions cutanees : visage, tronc, bras, jambes, une zone precise ou plusieurs zones du corps ?",
        priority=8,
    ),
    ClinicalObjective(
        code="dermato_symptoms",
        categories=("Dermatologique",),
        triggers=("eruption", "boutons", "urticaire", "demangeaisons", "plaques rouges", "eczema", "rash", "rougeur"),
        already_known=("demangeaison", "douleur", "chaleur", "suintement", "gonflement"),
        question="Les lesions demangent-elles, sont-elles douloureuses, chaudes, gonflees ou avec un suintement ?",
        priority=10,
    ),
    ClinicalObjective(
        code="dermato_onset_exposure",
        categories=("Dermatologique",),
        triggers=("eruption", "boutons", "urticaire", "demangeaisons", "plaques rouges", "eczema", "rash", "piqure"),
        already_known=("depuis", "medicament", "aliment", "piqure", "produit", "contact"),
        question="Depuis quand l'eruption est-elle apparue, et y a-t-il eu un nouveau medicament, aliment, produit, piqure ou contact suspect ?",
        priority=12,
    ),
    ClinicalObjective(
        code="dermato_systemic",
        categories=("Dermatologique",),
        triggers=("eruption", "boutons", "urticaire", "demangeaisons", "plaques rouges", "eczema", "rash"),
        already_known=("fievre", "malaise", "respirer", "levres", "visage", "extension"),
        question="Avez-vous de la fievre, un malaise, une extension rapide des lesions, ou une gene respiratoire avec gonflement du visage ou des levres ?",
        priority=14,
    ),
    ClinicalObjective(
        code="cardiac_character",
        categories=("Cardiaque",),
        triggers=(
            "douleur thoracique",
            "douleurs thoraciques",
            "douleur poitrine",
            "douleurs poitrine",
            "douleur au coeur",
            "douleurs au coeur",
            "mal au coeur",
            "douleur cardiaque",
            "oppression",
            "poitrine serree",
            "coeur serre",
            "probleme cardiaque",
            "problemes cardiaques",
            "probleme cadiaque",
            "problemes cadiaques",
            "souci cardiaque",
            "souci cadiaque",
            "cardiaque",
            "cadiaque",
        ),
        already_known=("oppression", "brulure", "serrement", "aigue", "persistante"),
        question="Quand vous parlez de cette gene au niveau du coeur ou de la poitrine, est-ce plutot une oppression, une brulure, un serrement ou une douleur aigue, et est-elle encore presente maintenant ?",
        priority=5,
    ),
    ClinicalObjective(
        code="cardiac_associated",
        categories=("Cardiaque",),
        triggers=(
            "douleur thoracique",
            "douleurs thoraciques",
            "douleur poitrine",
            "douleur au coeur",
            "douleurs au coeur",
            "mal au coeur",
            "douleur cardiaque",
            "oppression",
            "poitrine serree",
            "coeur serre",
            "probleme cardiaque",
            "problemes cardiaques",
            "probleme cadiaque",
            "problemes cadiaques",
            "souci cardiaque",
            "souci cadiaque",
            "cardiaque",
            "cadiaque",
        ),
        already_known=("essoufflement", "malaise", "sueurs", "palpitations", "nausee"),
        question="Cette douleur s'accompagne-t-elle d'essoufflement, de malaise, de sueurs froides, de palpitations ou de nausees ?",
        priority=6,
    ),
    ClinicalObjective(
        code="cardiac_radiation",
        categories=("Cardiaque",),
        triggers=("douleur thoracique", "douleurs thoraciques", "douleur au coeur", "douleurs au coeur", "mal au coeur", "douleur cardiaque", "oppression", "sueurs", "malaise", "probleme cardiaque", "probleme cadiaque", "cardiaque", "cadiaque"),
        already_known=("bras", "machoire", "dos", "irradi"),
        question="Cette douleur se propage-t-elle vers le bras, la machoire ou le dos, ou s'accompagne-t-elle de sueurs froides ?",
        priority=8,
    ),
    ClinicalObjective(
        code="cardiac_context",
        categories=("Cardiaque",),
        triggers=("douleur thoracique", "douleurs thoraciques", "douleur au coeur", "douleurs au coeur", "mal au coeur", "douleur cardiaque", "palpitations", "malaise", "oppression", "probleme cardiaque", "probleme cadiaque", "cardiaque", "cadiaque"),
        already_known=("effort", "repos", "stress", "apres repas"),
        question="Est-ce que la gene apparait plutot a l'effort, au repos, apres un stress ou sans facteur declenchant evident ?",
        priority=20,
    ),
    ClinicalObjective(
        code="cardiac_history",
        categories=("Cardiaque",),
        triggers=("douleur thoracique", "douleurs thoraciques", "douleur au coeur", "douleurs au coeur", "mal au coeur", "douleur cardiaque", "oppression", "palpitations", "probleme cardiaque", "probleme cadiaque", "cardiaque", "cadiaque"),
        already_known=("infarctus", "angine", "stent", "hypertension", "diabete", "cholesterol", "traitement cardiaque"),
        question="Avez-vous des antecedents cardiaques, hypertension, diabete, cholesterol, tabac, ou un traitement cardiaque habituel ?",
        priority=18,
    ),
    ClinicalObjective(
        code="respiratory_severity",
        categories=("Respiratoire", "Cardiaque"),
        triggers=("essoufflement", "respiration genee", "souffle court", "saturation basse", "toux"),
        already_known=("saturation", "spo2", "parler", "phrases", "bleu", "cyanose"),
        question="Votre respiration vous permet-elle de parler en phrases completes, et avez-vous une saturation mesuree si un oxymetre est disponible ?",
        priority=6,
    ),
    ClinicalObjective(
        code="respiratory_wheezing",
        categories=("Respiratoire",),
        triggers=("asthme", "sifflement", "respiration genee", "essoufflement"),
        already_known=("sifflement", "ventoline", "salbutamol", "inhalateur", "crise"),
        question="Entendez-vous des sifflements en respirant, et avez-vous utilise un inhalateur comme la Ventoline ou un traitement respiratoire habituel ?",
        priority=10,
    ),
    ClinicalObjective(
        code="respiratory_infection",
        categories=("Respiratoire", "General"),
        triggers=("toux", "fievre", "frissons", "expectorations", "respiration"),
        already_known=("expectorations", "crachats", "fievre", "temperature", "douleur a la respiration"),
        question="Avez-vous de la fievre mesuree, des frissons, des crachats ou une douleur quand vous respirez profondement ?",
        priority=18,
    ),
    ClinicalObjective(
        code="respiratory_duration",
        categories=("Respiratoire",),
        triggers=("toux", "respiration", "essoufflement", "dyspnee", "pneumonie", "bronchite", "bpco"),
        already_known=("depuis", "heure", "jour", "semaine", "brutal", "progressif"),
        question="Depuis quand la toux ou la gene respiratoire a-t-elle commence, et l'evolution est-elle stable, progressive ou rapidement aggravee ?",
        priority=8,
    ),
    ClinicalObjective(
        code="respiratory_cough",
        categories=("Respiratoire",),
        triggers=("toux", "respiration", "essoufflement", "pneumonie", "bronchite"),
        already_known=("toux seche", "toux grasse", "crachats", "sang", "expectorations"),
        question="La toux est-elle seche ou grasse, avec des crachats, du sang, ou une douleur quand vous respirez profondement ?",
        priority=12,
    ),
    ClinicalObjective(
        code="respiratory_history",
        categories=("Respiratoire",),
        triggers=("asthme", "sifflement", "respiration", "essoufflement", "bpco", "bronchite"),
        already_known=("asthme", "bpco", "tabac", "inhalateur", "ventoline", "traitement"),
        question="Avez-vous un antecedent respiratoire comme asthme/BPCO, du tabac, ou un traitement inhalateur habituel ?",
        priority=20,
    ),
    ClinicalObjective(
        code="digestive_location",
        categories=("Digestif",),
        triggers=("mal au ventre", "douleur abdominale", "douleur estomac", "vomissements", "diarrhee"),
        already_known=("haut du ventre", "bas du ventre", "droite", "gauche", "diffuse", "localisee"),
        question="Ou se situe surtout la douleur abdominale : haut du ventre, bas du ventre, cote droit, cote gauche ou partout ?",
        priority=10,
    ),
    ClinicalObjective(
        code="digestive_transit",
        categories=("Digestif",),
        triggers=("mal au ventre", "douleur abdominale", "nausee", "vomissements", "diarrhee", "constipation"),
        already_known=("vomissement", "vomissements", "diarrhee", "constipation", "nausee", "selles"),
        question="Depuis le debut, avez-vous surtout des nausees, des vomissements, une diarrhee, une constipation ou un changement des selles ?",
        priority=12,
    ),
    ClinicalObjective(
        code="digestive_red_flags",
        categories=("Digestif",),
        triggers=("douleur abdominale", "mal au ventre", "vomissements", "diarrhee"),
        already_known=("sang", "ventre dur", "boire", "deshydratation", "malaise", "fievre"),
        question="Y a-t-il du sang dans les selles ou les vomissements, un ventre dur, une impossibilite de boire, de la fievre ou un malaise ?",
        priority=7,
    ),
    ClinicalObjective(
        code="digestive_hydration",
        categories=("Digestif",),
        triggers=("vomissements", "diarrhee", "nausee", "mal au ventre"),
        already_known=("boire", "urines", "deshydratation", "bouche seche"),
        question="Arrivez-vous a boire et uriner normalement, ou avez-vous des signes de deshydratation comme bouche seche, vertiges ou urines rares ?",
        priority=8,
    ),
    ClinicalObjective(
        code="digestive_context",
        categories=("Digestif",),
        triggers=("mal au ventre", "douleur abdominale", "vomissements", "diarrhee", "nausee", "appendicite", "gastro", "intoxication"),
        already_known=("repas", "aliment", "intoxication", "traitement", "antecedent", "chirurgie"),
        question="Y a-t-il eu un repas suspect, une intoxication alimentaire possible, un traitement recent, ou des antecedents digestifs/chirurgicaux ?",
        priority=18,
    ),
    ClinicalObjective(
        code="neurologic_onset",
        categories=("Neurologique",),
        triggers=("confusion", "mal de tete", "cephalee", "perte de connaissance", "convulsion", "paralysie"),
        already_known=("brutal", "progressif", "depuis", "heure", "au reveil"),
        question="Le symptome neurologique est-il apparu brutalement ou progressivement, et a quelle heure environ a-t-il commence ?",
        priority=4,
    ),
    ClinicalObjective(
        code="neurologic_deficit",
        categories=("Neurologique",),
        triggers=("confusion", "mal de tete", "paralysie", "vertige", "perte de connaissance"),
        already_known=("faiblesse", "parole", "vision", "equilibre", "cote", "visage"),
        question="Avez-vous remarque une faiblesse d'un cote, un trouble de la parole, de la vision, de l'equilibre ou une asymetrie du visage ?",
        priority=6,
    ),
    ClinicalObjective(
        code="neurologic_infection",
        categories=("Neurologique", "General"),
        triggers=("mal de tete", "cephalee", "confusion", "fievre"),
        already_known=("raideur", "nuque", "lumiere", "photophobie", "fievre"),
        question="Avec le mal de tete ou la confusion, avez-vous de la fievre, une raideur de nuque ou une gene importante a la lumiere ?",
        priority=9,
    ),
    ClinicalObjective(
        code="neurologic_consciousness",
        categories=("Neurologique",),
        triggers=("confusion", "malaise", "perte de connaissance", "convulsion", "somnolence", "avc", "meningite"),
        already_known=("confusion", "somnolence", "conscience", "malaise", "convulsion", "chute"),
        question="Y a-t-il eu confusion, somnolence inhabituelle, perte de connaissance, convulsion ou malaise avec chute ?",
        priority=7,
    ),
    ClinicalObjective(
        code="neurologic_history",
        categories=("Neurologique",),
        triggers=("confusion", "mal de tete", "cephalee", "vertige", "avc", "migraine", "convulsion"),
        already_known=("migraine", "avc", "hypertension", "diabete", "anticoagulant", "traitement"),
        question="Avez-vous des antecedents neurologiques, migraine/AVC, hypertension, diabete, anticoagulants ou un traitement en cours ?",
        priority=18,
    ),
    ClinicalObjective(
        code="time_course",
        categories=(
            "Cardiaque",
            "Respiratoire",
            "Digestif",
            "Neurologique",
            "Musculo-articulaire",
            "ORL",
            "Urinaire",
            "Dermatologique",
            "Infectieux/Febrile",
            "General",
        ),
        triggers=(),
        already_known=("depuis", "heure", "jour", "semaine", "brutal", "progressif"),
        question="Depuis quand exactement les symptomes ont-ils commence, et l'evolution est-elle stable, progressive ou rapide ?",
        priority=15,
    ),
    ClinicalObjective(
        code="intensity",
        categories=("Cardiaque", "Digestif", "Neurologique", "Musculo-articulaire", "ORL", "Urinaire", "Dermatologique", "General"),
        triggers=("douleur", "gene", "mal"),
        already_known=("sur 10", "/10", "intensite", "leger", "modere", "intense"),
        question="Sur une echelle de 0 a 10, quelle est l'intensite maximale de la douleur ou de la gene actuellement ?",
        priority=22,
    ),
    ClinicalObjective(
        code="history_treatments",
        categories=(
            "Cardiaque",
            "Respiratoire",
            "Digestif",
            "Neurologique",
            "Musculo-articulaire",
            "ORL",
            "Urinaire",
            "Dermatologique",
            "Infectieux/Febrile",
            "General",
        ),
        triggers=(),
        already_known=("antecedent", "traitement", "medicament", "allergie", "diabete", "asthme", "hypertension"),
        question="Avez-vous des antecedents importants, des traitements habituels, une allergie connue ou un medicament pris depuis le debut des symptomes ?",
        priority=35,
    ),
]


def _build_llm():
    if os.getenv("MEDICAL_SIMULATION_MODE", "false").lower() == "true":
        return None
    if os.getenv("MEDICAL_LLM_QUESTIONS", "false").lower() != "true":
        return None

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None

    # The questioning LLM should only use ask_patient. MCP and RAG search tools
    # are called at the end of the questioning phase for clinical synthesis,
    # preventing premature or out-of-context RAG/MCP calls.
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        timeout=20,
        api_key=api_key,
    ).bind_tools([ask_patient])


def _fast_hitl_enabled() -> bool:
    """
    Keep patient-answer resumes real-time.

    LangGraph re-enters the interrupted node when Command(resume=...) is used,
    so question generation must stay local and deterministic on this path.
    Heavy LLM/RAG/MCP work is reserved for the post-questioning synthesis.
    """
    return os.getenv("MEDICAL_FAST_HITL", "true").lower() == "true"


def _case_has(patient_case: str, keyword: str) -> bool:
    return has_positive_keyword(patient_case, keyword)


def _safe_contextual_symptoms(
    patient_case: str,
    responses: list[str],
    asked_questions: list[str] | None = None,
) -> list[str]:
    try:
        return list(cached_contextual_symptoms(patient_case, tuple(responses), tuple(asked_questions or [])))
    except Exception:
        return []


def _analysis_responses(patient_case: str, responses: list[str], asked_questions: list[str] | None = None) -> list[str]:
    inferred = _safe_contextual_symptoms(patient_case, responses, asked_questions)
    return [*responses, *inferred]


def _response_denies(response: str, keyword: str) -> bool:
    text = flatten_text(response)
    keyword = flatten_text(keyword)
    meaningful_parts = [
        part
        for part in keyword.split()
        if len(part) > 3 and part not in {"avec", "dans", "pour", "sans", "douleur", "gene", "mal", "de", "des", "les"}
    ]
    hit_positions = []
    exact_index = text.find(keyword)
    if exact_index >= 0:
        hit_positions.append(exact_index)
    hit_positions.extend(text.find(part) for part in meaningful_parts if text.find(part) >= 0)
    if not hit_positions:
        return False
    denial_markers = ["pas de", "pas d", "aucun", "aucune", "non", "sans", "ni "]
    return any(
        any(marker in text[max(0, position - 24):position] for marker in denial_markers)
        for position in hit_positions
    )


def _extract_clinical_memory(
    patient_case: str,
    responses: list[str],
    asked_questions: list[str] | None = None,
) -> dict:
    enriched = _analysis_responses(patient_case, responses, asked_questions)
    text = flatten_text(patient_case, enriched)
    confirmed: list[str] = []
    denied: list[str] = []
    medications: list[str] = []
    red_flags: list[str] = []

    symptom_keywords = {
        "fievre": ["fievre", "temperature", "etat febrile"],
        "frissons": ["frissons"],
        "courbatures": ["courbatures"],
        "fatigue": ["fatigue"],
        "toux": ["toux"],
        "gene respiratoire": ["gene respiratoire", "difficulte respiratoire", "essoufflement", "souffle court"],
        "douleur thoracique": [
            "douleur thoracique",
            "douleurs thoraciques",
            "oppression poitrine",
            "oppression thoracique",
            "oppression dans la poitrine",
            "douleur poitrine",
            "douleurs poitrine",
            "douleur au coeur",
            "douleurs au coeur",
            "mal au coeur",
            "douleur cardiaque",
            "poitrine serree",
            "coeur serre",
        ],
        "confusion": ["confusion", "desorientation"],
        "cephalee importante": ["cephalee importante", "mal de tete important", "maux de tete importants", "migraine intense"],
        "trouble equilibre": ["perte d equilibre", "trouble de l equilibre", "vertige important"],
        "raideur de nuque": ["raideur nuque", "raideur de nuque"],
        "malaise": ["malaise", "perte de connaissance"],
        "douleur abdominale": ["douleur abdominale", "mal au ventre"],
        "brulures urinaires": ["brulure urinaire", "brulures urinaires", "douleur en urinant"],
        "eruption cutanee": ["eruption cutanee", "boutons", "plaques rouges"],
    }

    for label, aliases in symptom_keywords.items():
        if any(has_positive_keyword(text, alias) for alias in aliases):
            confirmed.append(label)
        if any(_response_denies(response, alias) for response in responses for alias in aliases):
            denied.append(label)

    for medication in ["doliprane", "paracetamol", "ibuprofene", "advil", "amoxicilline", "salbutamol", "ventoline"]:
        if has_positive_keyword(text, medication):
            medications.append(medication)

    temperatures = []
    for match in re.finditer(r"\b(3[5-9](?:[,.]\d)?|4[0-2](?:[,.]\d)?)\b", text):
        temperatures.append(match.group(1).replace(",", "."))

    red_flag_aliases = {
        "respiration difficile": ["gene respiratoire", "difficulte respiratoire", "essoufflement", "ne peut pas parler"],
        "neurologique": [
            "confusion",
            "perte de connaissance",
            "trouble de la parole",
            "faiblesse d un cote",
            "raideur nuque",
            "cephalee importante",
            "perte d equilibre",
        ],
        "cardiaque": [
            "douleur thoracique",
            "douleurs thoraciques",
            "oppression poitrine",
            "oppression thoracique",
            "douleur au coeur",
            "douleurs au coeur",
            "mal au coeur",
            "douleur cardiaque",
            "poitrine serree",
            "coeur serre",
            "malaise",
            "sueurs froides",
        ],
        "deshydratation": ["deshydratation", "impossibilite de boire", "urines rares"],
    }
    for label, aliases in red_flag_aliases.items():
        if any(has_positive_keyword(text, alias) for alias in aliases):
            red_flags.append(label)

    categories = classify_clinical_categories(patient_case, enriched)
    dominant_category = categories[0] if categories else "General"

    denied_red_flags = False
    for question, response in zip(asked_questions or [], responses):
        question_text = flatten_text(question)
        response_text = flatten_text(response)
        if "signe" in question_text and "alerte" in question_text and any(marker in response_text for marker in ["pas", "non", "aucun", "aucune", "sans"]):
            denied_red_flags = True

    return {
        "confirmed_symptoms": sorted(set(confirmed)),
        "denied_symptoms": sorted(set(denied)),
        "medications": sorted(set(medications)),
        "temperature": temperatures[-1] if temperatures else None,
        "dominant_category": dominant_category,
        "red_flags": sorted(set(red_flags)),
        "denied_red_flags": denied_red_flags,
    }


def _objective_covered_by_memory(objective: ClinicalObjective, memory: dict) -> bool:
    confirmed = set(memory.get("confirmed_symptoms", []))
    denied = set(memory.get("denied_symptoms", []))
    medications = set(memory.get("medications", []))
    if objective.code == "febrile_temperature" and memory.get("temperature"):
        return True
    if objective.code == "febrile_associated" and (confirmed | denied) & {
        "toux", "brulures urinaires", "douleur abdominale", "eruption cutanee", "gene respiratoire"
    }:
        return True
    if objective.code == "febrile_red_flags" and (memory.get("denied_red_flags") or memory.get("red_flags")):
        return True
    if objective.code == "febrile_medication" and medications & {"doliprane", "paracetamol", "ibuprofene", "advil"}:
        return True
    neuro_triggers = {
        "confusion",
        "malaise",
        "cephalee importante",
        "trouble equilibre",
        "raideur de nuque",
    }
    if objective.code.startswith("neurologic") and not (confirmed & neuro_triggers or "neurologique" in memory.get("red_flags", [])):
        return True
    return False


CATEGORY_ORDER = [
    "Cardiaque",
    "Respiratoire",
    "Neurologique",
    "Infectieux/Febrile",
    "Digestif",
    "Musculo-articulaire",
    "Urinaire",
    "Dermatologique",
    "ORL",
    "General",
]

CATEGORY_OBJECTIVE_PREFIXES = {
    "Cardiaque": ("cardiac_",),
    "Respiratoire": ("respiratory_",),
    "Digestif": ("digestive_",),
    "Infectieux/Febrile": ("febrile_",),
    "Neurologique": ("neurologic_",),
    "Musculo-articulaire": ("musculo_",),
    "ORL": ("ent_",),
    "Urinaire": ("urinary_",),
    "Dermatologique": ("dermato_",),
}

CATEGORY_TRIGGER_TERMS = {
    "Cardiaque": (
        "douleur thoracique",
        "douleurs thoraciques",
        "douleur poitrine",
        "oppression poitrine",
        "oppression thoracique",
        "douleur au coeur",
        "douleurs au coeur",
        "mal au coeur",
        "douleur cardiaque",
        "poitrine serree",
        "coeur serre",
        "palpitations",
        "sueurs froides",
        "probleme cardiaque",
        "problemes cardiaques",
        "probleme cadiaque",
        "problemes cadiaques",
        "souci cardiaque",
        "souci cadiaque",
        "cardiaque",
        "cadiaque",
    ),
    "Respiratoire": (
        "essoufflement",
        "gene respiratoire",
        "difficulte respiratoire",
        "respiration genee",
        "souffle court",
        "saturation basse",
        "difficulte a respirer",
        "difficultes a respirer",
        "toux",
        "sifflement",
        "pneumonie",
        "bronchite",
        "bpco",
        "detresse respiratoire",
    ),
    "Neurologique": (
        "confusion",
        "perte de connaissance",
        "trouble de la parole",
        "faiblesse d un cote",
        "paralysie",
        "convulsion",
        "raideur nuque",
        "raideur de nuque",
        "cephalee importante",
    ),
}


def _category_has_real_trigger(category: str, patient_case: str, responses: list[str]) -> bool:
    terms = CATEGORY_TRIGGER_TERMS.get(category)
    if not terms:
        return True
    text = flatten_text(patient_case, responses)
    return any(has_positive_keyword(text, term) for term in terms)


def _objective_matches_category(objective: ClinicalObjective, category: str) -> bool:
    prefixes = CATEGORY_OBJECTIVE_PREFIXES.get(category)
    if not prefixes:
        return category in objective.categories
    return objective.code.startswith(prefixes)


def _answered_by_responses_or_memory(objective: ClinicalObjective, responses: list[str], memory: dict) -> bool:
    if _objective_covered_by_memory(objective, memory):
        return True
    if not objective.already_known or not responses:
        return False
    response_text = flatten_text(responses)
    return _contains_any(response_text, objective.already_known)


def _priority_categories(patient_case: str, responses: list[str], asked_questions: list[str] | None = None) -> list[str]:
    enriched_responses = _analysis_responses(patient_case, responses, asked_questions)
    initial_categories = list(cached_clinical_categories(patient_case, ()))
    categories = list(cached_clinical_categories(patient_case, tuple(enriched_responses)))

    initial_specific = [category for category in initial_categories if category != "General"]
    detected_specific = [category for category in categories if category != "General"]
    dominant = initial_specific[0] if initial_specific else (detected_specific[0] if detected_specific else "General")

    allowed = {dominant}
    for category in detected_specific:
        if category == dominant:
            continue
        if category in CATEGORY_TRIGGER_TERMS and _category_has_real_trigger(category, patient_case, responses):
            allowed.add(category)
        elif dominant == "General":
            allowed.add(category)

    if dominant in CATEGORY_TRIGGER_TERMS and not _category_has_real_trigger(dominant, patient_case, responses):
        allowed.discard(dominant)

    if not allowed:
        allowed.add("General")

    ordered = [category for category in CATEGORY_ORDER if category in allowed]
    return ordered or ["General"]


def _contains_any(text: str, values: tuple[str, ...]) -> bool:
    return any(value and value in text for value in values)


def _asked_similar(question: str, asked_questions: list[str]) -> bool:
    question_terms = {term for term in flatten_text(question).split() if len(term) > 4}
    for asked in asked_questions:
        asked_terms = {term for term in flatten_text(asked).split() if len(term) > 4}
        if question_terms and len(question_terms & asked_terms) / max(len(question_terms), 1) >= 0.75:
            return True
    return False


def _objective_applies(
    objective: ClinicalObjective,
    categories: list[str],
    case_text: str,
    full_text: str,
    response_text: str = "",
    memory: dict | None = None,
) -> bool:
    if not any(category in objective.categories for category in categories):
        return False
    if objective.triggers and not (
        _contains_any(case_text, objective.triggers) or _contains_any(response_text, objective.triggers)
    ):
        return False
    return not _answered_by_responses_or_memory(objective, [response_text] if response_text else [], memory or {})


def _select_objective(
    patient_case: str,
    responses: list[str],
    asked_questions: list[str],
    question_count: int,
) -> ClinicalObjective:
    categories = _priority_categories(patient_case, responses, asked_questions)
    dominant_category = categories[0] if categories else "General"
    case_text = flatten_text(patient_case)
    memory = _extract_clinical_memory(patient_case, responses, asked_questions)
    full_text = flatten_text(patient_case, _analysis_responses(patient_case, responses, asked_questions))
    response_text = flatten_text(responses)

    strict_categories = [category for category in categories if category != "General"]
    if strict_categories:
        primary_candidates = [
            objective
            for objective in OBJECTIVES
            if _objective_matches_category(objective, dominant_category)
            and not _answered_by_responses_or_memory(objective, responses, memory)
            and not _asked_similar(objective.question, asked_questions)
        ]
    else:
        primary_candidates = []

    secondary_candidates = [
        objective
        for objective in OBJECTIVES
        if objective not in primary_candidates
        and not objective.code.startswith(("cardiac_", "respiratory_", "digestive_", "febrile_", "neurologic_"))
        and _objective_applies(objective, categories, case_text, full_text, response_text, memory)
        and not _asked_similar(objective.question, asked_questions)
    ]

    candidates = primary_candidates or secondary_candidates

    if not candidates:
        candidates = [
            objective
            for objective in OBJECTIVES
            if "General" in objective.categories
            and not _answered_by_responses_or_memory(objective, responses, memory)
            and not _asked_similar(objective.question, asked_questions)
        ]

    if not candidates:
        return ClinicalObjective(
            code=f"clarification_{question_count}",
            categories=("General",),
            triggers=(),
            already_known=(),
            question="Quel element nouveau ou important concernant vos symptomes n'a pas encore ete mentionne ?",
            priority=99,
        )

    def route_priority(objective: ClinicalObjective) -> tuple[int, int]:
        if _objective_matches_category(objective, dominant_category):
            penalty = 0
        elif dominant_category in objective.categories:
            penalty = 15
        elif "General" in objective.categories:
            penalty = 80
        else:
            penalty = 100
        return objective.priority + penalty, objective.priority

    return sorted(candidates, key=route_priority)[0]


def _contextualize_question(
    objective: ClinicalObjective,
    patient_case: str,
    responses: list[str],
    asked_questions: list[str] | None = None,
) -> str:
    categories = ", ".join(_priority_categories(patient_case, responses, asked_questions))
    memory = _extract_clinical_memory(patient_case, responses, asked_questions)
    symptom_hint = patient_case.strip().rstrip(".")
    if len(symptom_hint) > 120:
        symptom_hint = f"{symptom_hint[:117].rstrip()}..."

    last_response = responses[-1].strip() if responses else ""
    if len(last_response) > 95:
        last_response = f"{last_response[:92].rstrip()}..."

    if objective.code == "time_course":
        return f"Pour mieux situer le cas ({symptom_hint}), depuis quand les symptomes ont-ils commence et l'evolution est-elle stable, progressive ou rapide ?"
    if objective.code == "history_treatments":
        return f"Comme l'orientation actuelle est {categories}, avez-vous des antecedents importants, traitements habituels, allergies ou medicaments pris depuis le debut ?"
    if not responses:
        return f"Pour commencer et rester centre sur le motif principal, {objective.question[0].lower()}{objective.question[1:]}"

    memory_notes = []
    if memory.get("temperature"):
        memory_notes.append(f"je note une temperature deja indiquee a {memory['temperature']} C")
    if memory.get("confirmed_symptoms"):
        memory_notes.append(f"je garde en tete : {', '.join(memory['confirmed_symptoms'][:3])}")
    if memory.get("denied_symptoms"):
        memory_notes.append(f"et l'absence de {', '.join(memory['denied_symptoms'][:2])}")
    memory_prefix = f"{'; '.join(memory_notes)}. " if memory_notes else ""

    transitions = {
        "Musculo-articulaire": "Merci, cela aide a preciser l'atteinte du membre. Pour evaluer la gravite fonctionnelle,",
        "Cardiaque": "Merci, je verifie maintenant les signes associes importants pour une douleur thoracique,",
        "Respiratoire": "Merci, pour mieux estimer le retentissement respiratoire,",
        "Digestif": "Merci, pour orienter prudemment l'analyse digestive,",
        "Neurologique": "Merci, je recherche surtout des signes neurologiques d'alerte,",
        "ORL": "Merci, pour verifier le retentissement ORL et l'hydratation,",
        "Urinaire": "Merci, je recherche les signes qui peuvent compliquer un trouble urinaire,",
        "Dermatologique": "Merci, pour evaluer l'extension et le risque allergique ou infectieux,",
        "Infectieux/Febrile": "Merci, je reste centre sur l'episode febrile pour preciser son intensite et ses signes associes,",
        "General": "Merci pour cette precision. Pour mieux prioriser la suite,",
    }
    priority_categories = _priority_categories(patient_case, responses, asked_questions)
    category = next(
        (candidate for candidate in priority_categories if candidate in objective.categories),
        objective.categories[0] if objective.categories else "General",
    )
    transition = transitions.get(category, transitions["General"])
    return f"{transition} {memory_prefix}{objective.question[0].lower()}{objective.question[1:]}"


def _build_dynamic_question_without_llm(
    patient_case: str,
    responses: list[str],
    asked_questions: list[str],
    question_count: int,
) -> str:
    objective = _select_objective(patient_case, responses, asked_questions, question_count)
    return _contextualize_question(objective, patient_case, responses, asked_questions)


def _question_stays_in_route(
    question: str,
    categories: list[str],
    objective: ClinicalObjective,
    patient_case: str,
    responses: list[str],
    asked_questions: list[str],
) -> bool:
    question_categories = set(classify_clinical_categories(question))
    allowed_categories = set(categories) | set(objective.categories) | {"General"}
    unexpected = question_categories - allowed_categories
    if not unexpected:
        return True

    context_text = flatten_text(patient_case, _analysis_responses(patient_case, responses, asked_questions))
    alert_terms = [
        "confusion",
        "perte de connaissance",
        "trouble de la parole",
        "faiblesse d un cote",
        "raideur nuque",
        "douleur thoracique",
        "oppression poitrine",
        "difficulte respiratoire",
        "saturation basse",
    ]
    return any(has_positive_keyword(context_text, term) for term in alert_terms)


CATEGORY_STRATEGIES = {
    "Cardiaque": (
        "1. Début et évolution temporelle de la douleur.\n"
        "2. Type exact de la douleur (oppression, brûlure, serrement, etc.).\n"
        "3. Intensité de la douleur (sur 10).\n"
        "4. Irradiations (bras, mâchoire, dos).\n"
        "5. Signes associés (essoufflement, malaise, sueurs) et antécédents cardiaques."
    ),
    "Respiratoire": (
        "1. Début et évolution de la gêne respiratoire.\n"
        "2. Retentissement sur la parole (peut-il faire des phrases ?) et saturation (si oxymètre).\n"
        "3. Présence de toux, crachats ou sifflements.\n"
        "4. Fièvre associée ou douleur à la respiration.\n"
        "5. Antécédents respiratoires (asthme, BPCO) et traitements en cours."
    ),
    "Digestif": (
        "1. Localisation exacte de la douleur (haut, bas, droite, gauche) et évolution.\n"
        "2. Type de troubles du transit (vomissements, diarrhée, constipation).\n"
        "3. Signes de gravité (sang dans les selles/vomissements, ventre dur).\n"
        "4. Capacité à boire et signes de déshydratation.\n"
        "5. Fièvre associée et antécédents médicaux."
    ),
    "Infectieux/Febrile": (
        "1. Température maximale mesurée et date de début.\n"
        "2. Frissons, sueurs ou courbatures associés.\n"
        "3. Symptôme localisateur (toux, brûlures urinaires, maux de ventre, éruption).\n"
        "4. Signes de gravité (confusion, raideur de nuque, difficulté à respirer).\n"
        "5. Traitements déjà pris (paracétamol, ibuprofène) et efficacité."
    ),
    "Neurologique": (
        "1. Heure précise du début et mode d'apparition (brutal ou progressif).\n"
        "2. Présence d'un déficit (faiblesse d'un côté, trouble de la parole ou de la vision).\n"
        "3. Troubles de l'équilibre ou de la conscience.\n"
        "4. Maux de tête intenses ou raideur de nuque.\n"
        "5. Fièvre associée et antécédents (hypertension, AVC, traitements)."
    ),
    "ORL": (
        "1. Début et évolution des symptômes (mal de gorge, oreille, nez).\n"
        "2. Capacité à avaler (liquides et solides).\n"
        "3. Gêne respiratoire associée.\n"
        "4. Fièvre mesurée.\n"
        "5. Antécédents et traitements en cours."
    ),
    "Urinaire": (
        "1. Début et intensité des brûlures ou de la gêne urinaire.\n"
        "2. Présence de sang dans les urines ou d'envies fréquentes.\n"
        "3. Douleur dans le dos ou les lombaires.\n"
        "4. Fièvre et frissons associés.\n"
        "5. Possibilité de grossesse (si femme) ou antécédents urologiques."
    ),
    "Dermatologique": (
        "1. Début, localisation et évolution de l'éruption ou des lésions.\n"
        "2. Présence de démangeaisons ou de douleur locale.\n"
        "3. Signes d'alerte (gonflement du visage/lèvres, gêne respiratoire).\n"
        "4. Fièvre associée.\n"
        "5. Prise d'un nouveau médicament, aliment ou contact suspect."
    ),
    "Musculo-articulaire": (
        "1. Localisation exacte (quelle articulation, quel membre) et évolution.\n"
        "2. Notion de traumatisme (chute, coup, faux mouvement).\n"
        "3. Intensité de la douleur (sur 10) et gonflement/rougeur/chaleur.\n"
        "4. Capacité à bouger le membre ou à marcher.\n"
        "5. Antécédents similaires ou traitements pris."
    ),
    "General": (
        "1. Début et chronologie exacte des symptômes.\n"
        "2. Intensité de la gêne principale (sur 10).\n"
        "3. Présence de fièvre ou d'une fatigue inhabituelle.\n"
        "4. Capacité à s'hydrater et s'alimenter.\n"
        "5. Antécédents médicaux majeurs et traitements habituels."
    )
}

def _build_question(
    patient_case: str,
    responses: list[str],
    asked_questions: list[str],
    question_count: int,
    llm,
) -> str:
    objective = _select_objective(patient_case, responses, asked_questions, question_count)
    fallback_question = _contextualize_question(objective, patient_case, responses, asked_questions)
    categories = _priority_categories(patient_case, responses, asked_questions)
    memory = _extract_clinical_memory(patient_case, responses, asked_questions)

    # Bypass LLM on the HITL path: Command(resume=...) may re-run this node,
    # and a network call here directly blocks the next patient question.
    if llm is None or question_count == 0 or _fast_hitl_enabled():
        with perf_timer("llm_question", question_index=question_count + 1, skipped=True, fast_hitl=_fast_hitl_enabled()):
            pass
        return fallback_question

    dominant_category = categories[0] if categories else "General"
    strategy = CATEGORY_STRATEGIES.get(dominant_category, CATEGORY_STRATEGIES["General"])

    chat_msgs = [
        {"role": "system", "content": SYSTEM_PROMPT.format(max_q=MAX_QUESTIONS)},
        {
            "role": "user",
            "content": (
                f"Cas patient : {patient_case}\n"
                f"Categorie dominante actuelle : {dominant_category}\n"
                f"Strategie de questions a suivre strictement pour cette categorie :\n{strategy}\n\n"
                f"Question numero : {question_count + 1}/{MAX_QUESTIONS}\n"
                f"Questions deja posees : {' | '.join(asked_questions) if asked_questions else 'aucune'}\n"
                f"Reponses deja collectees : {' | '.join(responses) if responses else 'aucune'}\n"
                f"Memoire clinique : {memory}\n"
                f"Objectif prioritaire suggere : {fallback_question}\n\n"
                "Instructions imperatives :\n"
                "1. Pose UNE SEULE question claire, courte et conversationnelle en francais.\n"
                f"2. Adapte-toi au stade actuel ({question_count + 1}/{MAX_QUESTIONS}) en suivant la strategie '{dominant_category}'.\n"
                "3. INTERDICTION FORMELLE : Ne pose AUCUNE question hors de la categorie dominante (ex: pas de question neurologique pour un cas cardiaque), SAUF si une reponse precedente a explicitement evoque un vrai signe d'alerte.\n"
                "4. Ne repete jamais une question et utilise les reponses precedentes pour rebondir naturellement.\n"
                "5. Ne donne aucun diagnostic et ne pose pas plusieurs questions a la fois."
            ),
        },
    ]

    try:
        with perf_timer("llm_question", question_index=question_count + 1):
            resp = llm.invoke(chat_msgs)
        question = resp.tool_calls[0]["args"].get("question", resp.content) if resp.tool_calls else resp.content
        question = str(question).strip()
    except Exception as e:
        print(f"LLM ERROR in _build_question: {e}")
        return fallback_question
    if not question or _asked_similar(question, asked_questions):
        return fallback_question
    return question


def _dynamic_simulated_response(patient_case: str, question: str, question_count: int) -> str:
    text = flatten_text(patient_case, question)
    question_text = flatten_text(question)

    if "signe d alerte" in question_text:
        return "Pas de confusion, pas de raideur de nuque, pas de gene respiratoire importante ni malaise."

    if "symptome associe" in question_text and "oriente la fievre" in question_text:
        return "Pas de toux, pas de douleur abdominale, pas de brulure urinaire ni eruption cutanee."

    if any(term in question_text for term in ["doliprane", "paracetamol", "medicament"]):
        return "J'ai pris du Doliprane 1 g, avec une baisse partielle de la temperature."

    if "depuis" in question_text:
        if "douleur au coeur" in text or "thoracique" in text:
            return "Depuis quelques heures, avec une gene inhabituelle qui persiste par moments."
        if "respiration" in text or "toux" in text:
            return "Depuis environ trois jours, avec une aggravation progressive."
        if "ventre" in text or "abdominal" in text:
            return "Depuis 24 a 48 heures, avec une douleur fluctuante."
        if any(term in text for term in ["main", "bras", "jambe", "articulation", "poignet"]):
            return "Depuis hier, avec une douleur localisee qui augmente quand je bouge le membre."
        return "Depuis aujourd'hui, avec une evolution a surveiller."

    if any(term in question_text for term in ["main droite", "main gauche", "doigts", "poignet", "articulation"]):
        return "La douleur concerne surtout la main droite, au niveau des doigts et du poignet."

    if any(term in question_text for term in ["gonflee", "rouge", "chaude", "deformee"]):
        return "La main semble un peu gonflee, sans deformation evidente."

    if any(term in question_text for term in ["chute", "coup", "traumatisme", "faux mouvement"]):
        return "Il y a eu un faux mouvement, sans chute importante."

    if any(term in question_text for term in ["bouger", "plier", "utiliser", "marcher"]):
        return "Je peux bouger mais difficilement, la douleur augmente quand j'utilise la main."

    if any(term in question_text for term in ["saturation", "parler", "respiration"]):
        if "saturation" in text or "souffle court" in text:
            return "Je suis essouffle, mais je peux parler; aucune saturation fiable n'est disponible."
        return "Pas de difficulte respiratoire importante signalee."

    if any(term in question_text for term in ["vomissements", "diarrhee", "nausees", "selles"]):
        return "Nausees presentes avec quelques vomissements, pas de sang signale."

    if any(term in question_text.split() for term in ["bras", "machoire", "dos", "sueurs"]):
        return "Pas d'irradiation nette, mais des sueurs et un malaise leger sont possibles."

    if any(term in question_text for term in ["faiblesse", "parole", "vision", "equilibre"]):
        return "Pas de faiblesse d'un cote ni trouble de la parole signale."

    if "10" in question or "intensite" in question_text:
        return "L'intensite est autour de 6 sur 10."

    if any(term in question_text for term in ["antecedents", "traitements", "allergie", "medicaments"]):
        return "Pas d'antecedent majeur connu et aucun nouveau traitement en dehors des medicaments habituels."

    if "fievre" in question_text:
        return "Pas de fievre mesuree pour le moment."

    return "Information precisee selon le cas, sans autre signe majeur rapporte."


def _temperature_clarification_question(question: str, answer: str) -> str | None:
    question_text = flatten_text(question)
    answer_text = flatten_text(answer)
    if not any(term in question_text for term in ["fievre", "temperature", "thermometre"]):
        return None

    numeric_values = [float(value.replace(",", ".")) for value in re.findall(r"\b([2-4][0-9](?:[,.][0-9])?)\b", answer_text)]
    suspicious_values = [value for value in numeric_values if value < 35 or value > 42]
    if suspicious_values:
        return (
            "Je veux éviter une erreur d'interprétation : pouvez-vous préciser la température mesurée "
            "au thermomètre en degrés Celsius, par exemple 38,5 °C, ainsi que l'heure de mesure ?"
        )

    if "taux" in answer_text and "fievre" in answer_text and not any(35 <= value <= 42 for value in numeric_values):
        return (
            "Pouvez-vous reformuler la mesure de fièvre ? J'ai besoin d'une température en degrés Celsius "
            "mesurée au thermomètre."
        )

    return None


def _collect_patient_response(
    patient_case: str,
    question: str,
    question_count: int,
    interrupt_context: dict | None = None,
) -> str:
    if os.getenv("MEDICAL_DEMO_AUTORUN", "false").lower() == "true":
        response = _dynamic_simulated_response(patient_case, question, question_count)
        return mask_sensitive_data(validate_patient_input(response))

    validation_error: str | None = None
    while True:
        payload = {
            "type": "patient_question",
            "question_index": question_count + 1,
            "question": question,
            "current_question": question,
            "next_question": question,
            "progress": f"{question_count + 1}/{MAX_QUESTIONS}",
        }
        if interrupt_context:
            payload.update(interrupt_context)
        if validation_error:
            payload["validation_error"] = validation_error
            payload["message"] = validation_error

        response = interrupt(payload)
        response_text = _extract_patient_response_text(response)
        if response_text:
            return mask_sensitive_data(validate_patient_input(response_text))

        validation_error = "La reponse patient ne peut pas etre vide. Merci de renseigner une reponse avant de continuer."
        logger.warning(
            "empty_patient_response_reprompt",
            extra={
                "question_index": question_count + 1,
                "patient_case_preview": mask_sensitive_data(patient_case[:180]),
            },
        )


def _extract_patient_response_text(response: object) -> str:
    if isinstance(response, dict):
        for key in ("patient_answer", "answer", "response", "message", "text"):
            value = response.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    return str(response or "").strip()


@lru_cache(maxsize=128)
def _enrich_with_mcp_cached(
    patient_case: str,
    responses_key: tuple[str, ...],
    asked_questions_key: tuple[str, ...],
    clinical_score: int | None = None,
    severity_level: str | None = None,
    clinical_category: str | None = None,
) -> str:
    mcp_context = []
    responses = list(responses_key)
    asked_questions = list(asked_questions_key)
    all_text = flatten_text(patient_case, _analysis_responses(patient_case, responses, asked_questions))

    try:
        raw_result = search_symptoms(symptom=all_text)
        if raw_result.get("found"):
            result = format_mcp_symptom_result(
                raw_result,
                clinical_score=clinical_score,
                severity_level=severity_level,
                clinical_category=clinical_category,
            )
        else:
            result = mcp_search_symptoms.invoke({"symptom": all_text})
    except Exception:
        result = mcp_search_symptoms.invoke({"symptom": all_text})

    if "Aucune donnee MCP specifique" not in result:
        mcp_context.append(f"[MCP Analyse clinique]\n{result}")

    for drug in ["paracetamol", "doliprane", "ibuprofene", "advil", "amoxicilline", "salbutamol", "ventoline"]:
        if drug in all_text:
            result = mcp_get_drug_info.invoke({"drug_name": drug})
            mcp_context.append(f"[MCP Medicament]\n{result}")
            break

    return "\n\n".join(mcp_context)


def _enrich_with_mcp(
    patient_case: str,
    responses: list[str],
    asked_questions: list[str] | None = None,
    clinical_score: int | None = None,
    severity_level: str | None = None,
    clinical_category: str | None = None,
) -> str:
    with perf_timer("mcp", clinical_category=clinical_category):
        return _enrich_with_mcp_cached(
            patient_case,
            tuple(responses),
            tuple(asked_questions or []),
            clinical_score,
            severity_level,
            clinical_category,
        )


def _generate_summary(
    patient_case: str,
    responses: list[str],
    rag_context: str,
    mcp_context: str,
    llm,
    asked_questions: list[str] | None = None,
) -> str:
    enriched_responses = _analysis_responses(patient_case, responses, asked_questions)
    contextual_symptoms = _safe_contextual_symptoms(patient_case, responses, asked_questions)
    clinical_memory = _extract_clinical_memory(patient_case, responses, asked_questions)
    clinical_score, severity_level = calculate_clinical_score(patient_case, enriched_responses)
    clinical_category = classify_clinical_category(patient_case, enriched_responses)

    if llm is None or _fast_hitl_enabled():
        with perf_timer("llm_summary", skipped=True, fast_hitl=_fast_hitl_enabled()):
            pass
        formatted_responses = "\n- ".join(responses)
        confirmed = ", ".join(clinical_memory.get("confirmed_symptoms", [])[:5]) or "non specifie"
        denied = ", ".join(clinical_memory.get("denied_symptoms", [])[:5]) or "non specifie"
        medication = ", ".join(clinical_memory.get("medications", [])[:4]) or "non specifie"
        return (
            "Synthese clinique preliminaire :\n\n"
            f"Cas initial : {patient_case}\n\n"
            "Reponses patient :\n"
            f"- {formatted_responses}\n\n"
            f"Memoire clinique : symptomes confirmes ({confirmed}); symptomes nies ({denied}); "
            f"medicaments mentionnes ({medication}).\n\n"
            f"Orientation : categorie(s) {clinical_category}, gravite {severity_level}, "
            f"score clinique {clinical_score}. Les donnees MCP/RAG servent uniquement "
            "d'aide contextuelle et ne constituent pas un diagnostic definitif."
        )

    chat_msgs = [
        {"role": "system", "content": SYSTEM_PROMPT.format(max_q=MAX_QUESTIONS)},
        {"role": "user", "content": f"Cas patient : {patient_case}"},
        {"role": "system", "content": f"Memoire clinique consolidee : {clinical_memory}"},
    ]

    if mcp_context:
        chat_msgs.append({"role": "system", "content": f"Contexte MCP structure :\n{mcp_context}"})
    if rag_context:
        chat_msgs.append({"role": "system", "content": f"Contexte documentaire RAG filtre :\n{rag_context}"})

    for index, response in enumerate(responses):
        chat_msgs.append({"role": "assistant", "content": f"Question {index + 1} posee."})
        chat_msgs.append({"role": "user", "content": f"Reponse : {response}"})

    chat_msgs.append({
        "role": "user",
        "content": (
            "Produis une synthese clinique preliminaire courte, structuree et prudente. "
            "Mentionne les categories, le score, la gravite, les signes d'alerte, "
            "les diagnostics differentiels possibles et les incertitudes. "
            "Reste coherent avec MCP et RAG sans diagnostic definitif."
        ),
    })

    with perf_timer("llm_summary"):
        return validate_medical_response(llm.invoke(chat_msgs).content)


def diagnostic_agent(state: MedicalState) -> MedicalState:
    with perf_timer("diagnostic_agent", question_count=state.get("question_count")):
        return _diagnostic_agent_impl(state)


def _coerce_patient_case(state: MedicalState) -> str:
    """
    Read patient_case from LangGraph state without overwriting a provided value.

    LangGraph Studio and FastAPI both send a plain state dict. This helper keeps
    the direct patient_case field as the source of truth, and only falls back
    when it is absent or blank.
    """
    raw_patient_case = state.get("patient_case")
    if raw_patient_case is None:
        logger.warning(
            "patient_case_missing_using_fallback",
            extra={"state_keys": sorted(str(key) for key in state.keys())},
        )
        return PATIENT_CASE_FALLBACK

    patient_case = str(raw_patient_case).strip()
    if not patient_case:
        logger.warning(
            "patient_case_blank_using_fallback",
            extra={"state_keys": sorted(str(key) for key in state.keys())},
        )
        return PATIENT_CASE_FALLBACK

    logger.debug(
        "patient_case_received",
        extra={
            "patient_case_preview": mask_sensitive_data(patient_case[:180]),
            "patient_id": state.get("patient_id"),
            "patient_name": state.get("patient_name"),
        },
    )
    return patient_case


def _diagnostic_agent_impl(state: MedicalState) -> MedicalState:
    patient_case = _coerce_patient_case(state)
    responses = list(state.get("patient_responses", []))
    asked_questions = list(state.get("asked_questions", []))
    question_count = int(state.get("question_count", len(asked_questions)) or 0)
    llm = None if _fast_hitl_enabled() else _build_llm()

    if question_count < MAX_QUESTIONS:
        question = _build_question(patient_case, responses, asked_questions, question_count, llm)
        preview_responses = _analysis_responses(patient_case, responses, asked_questions)
        preview_score, preview_severity = calculate_clinical_score(patient_case, preview_responses)
        preview_category = classify_clinical_category(patient_case, preview_responses)
        interrupt_context = {
            "clinical_score": preview_score,
            "severity_level": preview_severity,
            "clinical_category": preview_category,
        }
        patient_response = _collect_patient_response(patient_case, question, question_count, interrupt_context)
        clarification_question = _temperature_clarification_question(question, patient_response)
        if clarification_question:
            patient_response = _collect_patient_response(patient_case, clarification_question, question_count, interrupt_context)
            question = clarification_question

        responses.append(patient_response)
        asked_questions.append(question)
        contextual_symptoms = _safe_contextual_symptoms(patient_case, responses, asked_questions)
        enriched_responses = [*responses, *contextual_symptoms]
        clinical_score, severity_level = calculate_clinical_score(patient_case, enriched_responses)
        clinical_category = classify_clinical_category(patient_case, enriched_responses)
        clinical_memory = _extract_clinical_memory(patient_case, responses, asked_questions)

        return {
            "messages": [
                AIMessage(
                    content=f"[Diagnostic Agent] Question {question_count + 1}: {question}\n[Reponse patient collectee]",
                    name="diagnostic_agent",
                )
            ],
            "patient_case": patient_case,
            "patient_responses": responses,
            "asked_questions": asked_questions,
            "contextual_symptoms": contextual_symptoms,
            "clinical_memory": clinical_memory,
            "clinical_score": clinical_score,
            "severity_level": severity_level,
            "clinical_category": clinical_category,
            "question_count": question_count + 1,
            "current_question": None,
            "patient_answer": patient_response,
            "consultation_finished": False,
            "next": "diagnostic_agent",
        }

    contextual_symptoms = _safe_contextual_symptoms(patient_case, responses, asked_questions)
    clinical_memory = _extract_clinical_memory(patient_case, responses, asked_questions)
    enriched_responses = [*responses, *contextual_symptoms]
    clinical_score, severity_level = calculate_clinical_score(patient_case, enriched_responses)
    clinical_category = classify_clinical_category(patient_case, enriched_responses)
    rag_context = retrieve_medical_context(
        f"{clinical_category} {patient_case} {' '.join(enriched_responses)}",
        clinical_category=clinical_category,
    )
    mcp_data = _enrich_with_mcp(
        patient_case,
        responses,
        asked_questions,
        clinical_score=clinical_score,
        severity_level=severity_level,
        clinical_category=clinical_category,
    )
    summary = _generate_summary(patient_case, responses, rag_context, mcp_data, llm, asked_questions)

    interim = recommend_interim_care.invoke({
        "diagnostic_summary": summary,
        "patient_responses": responses,
        "patient_case": patient_case,
        "clinical_score": clinical_score,
        "severity_level": severity_level,
        "clinical_category": clinical_category,
    })

    return {
        "messages": [
            AIMessage(
                content="[Diagnostic Agent] Synthese clinique, score, categorie et recommandation intermediaire generes.",
                name="diagnostic_agent",
            )
        ],
        "patient_case": patient_case,
        "patient_responses": responses,
        "asked_questions": asked_questions,
        "contextual_symptoms": contextual_symptoms,
        "clinical_memory": clinical_memory,
        "question_count": MAX_QUESTIONS,
        "diagnostic_summary": summary,
        "interim_care": interim,
        "mcp_context": mcp_data,
        "rag_context": rag_context,
        "clinical_score": clinical_score,
        "severity_level": severity_level,
        "clinical_category": clinical_category,
        "current_question": None,
        "patient_answer": None,
        "consultation_finished": False,
        "next": "physician_review",
    }
