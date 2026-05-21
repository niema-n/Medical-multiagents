"""
Serveur MCP medical.

Expose deux outils accessibles par les agents LangGraph :
1. search_symptoms : analyse un symptome ou un texte clinique et retourne des
   categories, pathologies possibles, gravite et recommandations prudentes.
2. get_drug_info : retourne des informations prudentes sur un medicament courant.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

from fastmcp import FastMCP


mcp = FastMCP(
    name="medical-knowledge-server",
    instructions="Serveur de connaissances medicales prudent pour assister une orientation clinique preliminaire.",
)


SEVERITY_RANK = {"faible": 1, "modere": 2, "eleve": 3, "critique": 4}
SEVERITY_LABELS = {
    "faible": "faible",
    "modere": "modere",
    "eleve": "eleve",
    "critique": "critique",
}


def _normalise(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9<>=.,% ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_negated_context(prefix: str) -> bool:
    words = prefix.split()
    tail = " ".join(words[-10:])
    short_tail = " ".join(words[-6:])
    if " mais " not in f" {short_tail} " and re.search(
        r"\b(?:pas|aucun|aucune|sans|ni|absence|non)\b(?:\s+\w+){0,4}$",
        short_tail,
    ):
        return True
    patterns = [
        r"(?:pas|absence|sans|aucun|aucune|ni|non)(?:\s+(?:de|d|du|des))?\s*$",
        r"ne\s+presente\s+pas(?:\s+(?:de|d|du|des))?\s*$",
        r"n\s+a\s+pas(?:\s+(?:de|d|du|des))?\s*$",
        r"n\s+ai\s+pas(?:\s+(?:de|d|du|des))?\s*$",
        r"jamais(?:\s+(?:de|d|du|des))?\s*$",
    ]
    return any(re.search(pattern, tail) for pattern in patterns)


def _has_positive_phrase(text: str, phrase: str) -> bool:
    phrase = _normalise(phrase)
    if not phrase or phrase not in text:
        return False

    for match in re.finditer(re.escape(phrase), text):
        prefix = text[max(0, match.start() - 100):match.start()]
        if not _is_negated_context(prefix):
            return True
    return False


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


SYMPTOM_KNOWLEDGE: dict[str, dict[str, Any]] = {
    "douleur thoracique": {
        "category": "Cardiaque",
        "severity": "eleve",
        "score": 5,
        "aliases": [
            "douleur thoracique",
            "douleurs thoraciques",
            "douleur poitrine",
            "douleurs poitrine",
            "douleur dans la poitrine",
            "douleurs dans la poitrine",
            "douleur au coeur",
            "douleurs au coeur",
            "mal au coeur",
            "maux au coeur",
            "mal coeur",
            "douleur cardiaque",
            "douleurs cardiaques",
            "oppression poitrine",
            "oppression thoracique",
            "oppression dans la poitrine",
            "serrement poitrine",
            "poitrine serree",
            "poitrine serre",
            "coeur serre",
            "gene poitrine",
            "gene thoracique",
            "brulure poitrine",
            "douleur thoracique respiratoire",
        ],
        "pathologies": [
            {"pathologie": "Angor / angine de poitrine", "probabilite": "possible", "gravite": "eleve"},
            {"pathologie": "Infarctus du myocarde", "probabilite": "a exclure en urgence selon contexte", "gravite": "critique"},
            {"pathologie": "Insuffisance cardiaque", "probabilite": "possible si dyspnee, oedemes ou terrain cardiaque", "gravite": "eleve"},
            {"pathologie": "Embolie pulmonaire", "probabilite": "a exclure si douleur brutale, essoufflement ou facteur de risque", "gravite": "critique"},
            {"pathologie": "Pericardite", "probabilite": "possible", "gravite": "modere a eleve"},
            {"pathologie": "Douleur thoracique atypique", "probabilite": "possible apres exclusion des urgences", "gravite": "modere"},
            {"pathologie": "Douleur musculo-squelettique", "probabilite": "possible si douleur reproductible", "gravite": "faible a modere"},
        ],
        "red_flags": ["oppression", "malaise", "sueurs", "essoufflement", "douleur bras", "douleur machoire", "perte de connaissance"],
        "recommendation": "Urgence si douleur persistante, oppression, malaise, sueurs, essoufflement ou irradiation.",
    },
    "palpitations": {
        "category": "Cardiaque",
        "severity": "modere",
        "score": 3,
        "aliases": ["palpitations", "coeur qui bat vite", "rythme cardiaque rapide", "tachycardie"],
        "pathologies": [
            {"pathologie": "Trouble du rythme", "probabilite": "possible", "gravite": "modere"},
            {"pathologie": "Anxiete ou stress", "probabilite": "possible apres evaluation", "gravite": "faible a modere"},
        ],
        "red_flags": ["malaise", "douleur thoracique", "essoufflement", "perte de connaissance"],
        "recommendation": "Consultation medicale, urgence si malaise, douleur thoracique ou essoufflement associe.",
    },
    "malaise": {
        "category": "General",
        "severity": "modere",
        "score": 3,
        "aliases": ["malaise", "presyncope", "sensation de tomber", "faiblesse brutale", "lipothymie"],
        "pathologies": [
            {"pathologie": "Malaise vagal", "probabilite": "possible", "gravite": "faible a modere"},
            {"pathologie": "Cause cardiaque ou neurologique", "probabilite": "a evaluer si signes associes", "gravite": "eleve"},
        ],
        "red_flags": ["douleur thoracique", "essoufflement", "confusion", "perte de connaissance"],
        "recommendation": "Surveillance rapprochee et avis medical; urgence si douleur thoracique, dyspnee ou perte de connaissance.",
    },
    "sueurs": {
        "category": "General",
        "severity": "modere",
        "score": 2,
        "aliases": ["sueurs", "sueurs froides", "transpiration froide"],
        "pathologies": [
            {"pathologie": "Reaction vegetative a la douleur ou a la fievre", "probabilite": "possible", "gravite": "modere"},
            {"pathologie": "Syndrome coronarien si douleur thoracique associee", "probabilite": "a exclure", "gravite": "critique"},
        ],
        "red_flags": ["douleur thoracique", "malaise", "essoufflement"],
        "recommendation": "Avis medical rapide si symptome intense ou associe a douleur thoracique, malaise ou essoufflement.",
    },
    "toux": {
        "category": "Respiratoire",
        "severity": "faible",
        "score": 1,
        "aliases": ["toux", "tousser", "quinte de toux", "toux seche", "toux grasse"],
        "pathologies": [
            {"pathologie": "Bronchite", "probabilite": "possible", "gravite": "faible a modere"},
            {"pathologie": "Pneumonie", "probabilite": "possible si fievre ou dyspnee", "gravite": "eleve"},
            {"pathologie": "Grippe", "probabilite": "possible si fievre et courbatures", "gravite": "faible a modere"},
        ],
        "red_flags": ["essoufflement", "saturation basse", "douleur thoracique", "fievre elevee"],
        "recommendation": "Surveillance; consultation si fievre persistante, essoufflement, douleur thoracique ou terrain fragile.",
    },
    "fievre": {
        "category": "Infectieux/Febrile",
        "severity": "faible",
        "score": 1,
        "aliases": ["fievre", "temperature", "etat febrile", "frissons", "courbatures", "infection", "fatigue", "doliprane", "paracetamol"],
        "pathologies": [
            {"pathologie": "Infection virale", "probabilite": "frequent", "gravite": "faible"},
            {"pathologie": "Grippe", "probabilite": "possible", "gravite": "faible a modere"},
            {"pathologie": "Infection bacterienne", "probabilite": "possible selon contexte", "gravite": "modere a eleve"},
        ],
        "red_flags": ["confusion", "raideur nuque", "saturation basse", "fievre elevee"],
        "recommendation": "Hydratation et surveillance; consultation si fievre elevee, persistante ou signes de gravite.",
    },
    "essoufflement": {
        "category": "Respiratoire",
        "severity": "eleve",
        "score": 5,
        "aliases": [
            "essoufflement",
            "souffle court",
            "respiration genee",
            "gene respiratoire",
            "difficulte respiratoire",
            "difficultes a respirer",
            "dyspnee",
            "manque d air",
            "respire mal",
            "mal a respirer",
            "je respire mal",
            "probleme de respiration",
            "problemes de respiration",
            "probleme respiratoire",
        ],
        "pathologies": [
            {"pathologie": "Asthme ou exacerbation respiratoire", "probabilite": "possible", "gravite": "modere a eleve"},
            {"pathologie": "Pneumonie", "probabilite": "possible si fievre/toux", "gravite": "eleve"},
            {"pathologie": "Embolie pulmonaire", "probabilite": "a exclure si brutal ou douleur thoracique", "gravite": "critique"},
            {"pathologie": "Insuffisance cardiaque", "probabilite": "possible selon terrain", "gravite": "eleve"},
        ],
        "red_flags": ["douleur thoracique", "malaise", "saturation basse", "cyanose", "difficulte a parler"],
        "recommendation": "Avis medical rapide; urgence/SAMU si gene severe, saturation basse, douleur thoracique ou malaise.",
    },
    "douleur respiratoire": {
        "category": "Respiratoire",
        "severity": "eleve",
        "score": 4,
        "aliases": ["douleur respiratoire", "douleur a la respiration", "douleur quand je respire", "douleur en respirant"],
        "pathologies": [
            {"pathologie": "Pleurite ou irritation pleurale", "probabilite": "possible", "gravite": "modere a eleve"},
            {"pathologie": "Pneumonie", "probabilite": "possible si fievre ou toux", "gravite": "eleve"},
            {"pathologie": "Embolie pulmonaire", "probabilite": "a exclure si brutal, dyspnee ou facteur de risque", "gravite": "critique"},
            {"pathologie": "Douleur parietale thoracique", "probabilite": "possible si douleur reproductible", "gravite": "faible a modere"},
        ],
        "red_flags": ["essoufflement", "saturation basse", "malaise", "douleur thoracique", "brutal"],
        "recommendation": "Avis medical rapide si douleur respiratoire associee a essoufflement, fievre, malaise ou aggravation.",
    },
    "saturation basse": {
        "category": "Respiratoire",
        "severity": "critique",
        "score": 6,
        "aliases": ["saturation basse", "spo2 basse", "oxygene bas", "desaturation", "sat basse"],
        "pathologies": [
            {"pathologie": "Detresse respiratoire", "probabilite": "a evaluer rapidement", "gravite": "critique"},
            {"pathologie": "Pneumonie severe", "probabilite": "possible", "gravite": "critique"},
            {"pathologie": "Embolie pulmonaire", "probabilite": "possible selon contexte", "gravite": "critique"},
        ],
        "red_flags": ["essoufflement", "confusion", "douleur thoracique", "malaise"],
        "recommendation": "Urgence medicale; appeler le SAMU si saturation basse confirmee ou gene respiratoire importante.",
    },
    "douleur abdominale": {
        "category": "Digestif",
        "severity": "modere",
        "score": 2,
        "aliases": [
            "douleur abdominale",
            "douleurs abdominales",
            "mal au ventre",
            "maux de ventre",
            "douleur au ventre",
            "douleur estomac",
            "mal estomac",
            "mal a l estomac",
            "brulure estomac",
            "crampe abdominale",
            "crampes ventre",
        ],
        "pathologies": [
            {"pathologie": "Gastro-enterite", "probabilite": "frequent si diarrhee/vomissements", "gravite": "faible a modere"},
            {"pathologie": "Appendicite", "probabilite": "a evoquer si douleur fosse iliaque droite", "gravite": "eleve"},
            {"pathologie": "Intoxication alimentaire", "probabilite": "possible apres repas suspect", "gravite": "faible a modere"},
            {"pathologie": "Colique hepatique/renale ou autre cause abdominale", "probabilite": "possible selon localisation", "gravite": "modere a eleve"},
        ],
        "red_flags": ["douleur intense", "fievre", "vomissements repetes", "sang selles", "malaise", "ventre dur"],
        "recommendation": "Consultation si douleur intense, localisee, persistante, fievre, vomissements repetes ou aggravation.",
    },
    "vomissements": {
        "category": "Digestif",
        "severity": "modere",
        "score": 2,
        "aliases": ["vomissements", "vomissement", "vomir", "je vomis", "emetique"],
        "pathologies": [
            {"pathologie": "Gastro-enterite", "probabilite": "frequent", "gravite": "faible a modere"},
            {"pathologie": "Intoxication alimentaire", "probabilite": "possible", "gravite": "modere"},
            {"pathologie": "Cause abdominale urgente si douleur importante", "probabilite": "a evaluer", "gravite": "eleve"},
        ],
        "red_flags": ["sang", "deshydratation", "douleur abdominale", "malaise", "impossibilite de boire"],
        "recommendation": "Hydratation fractionnee; consultation si vomissements repetes, sang, douleur intense ou impossibilite de boire.",
    },
    "diarrhee": {
        "category": "Digestif",
        "severity": "faible",
        "score": 1,
        "aliases": ["diarrhee", "selles liquides", "diarrhees", "troubles du transit"],
        "pathologies": [
            {"pathologie": "Gastro-enterite", "probabilite": "frequent", "gravite": "faible a modere"},
            {"pathologie": "Intoxication alimentaire", "probabilite": "possible", "gravite": "modere"},
        ],
        "red_flags": ["sang selles", "deshydratation", "fievre elevee", "terrain fragile"],
        "recommendation": "Hydratation et surveillance; consultation si sang, deshydratation, fievre elevee ou diarrhee severe.",
    },
    "nausee": {
        "category": "Digestif",
        "severity": "faible",
        "score": 1,
        "aliases": ["nausee", "nausees", "envie de vomir", "mal au coeur digestif", "mal digestion"],
        "pathologies": [
            {"pathologie": "Gastro-enterite debutante", "probabilite": "possible", "gravite": "faible"},
            {"pathologie": "Intoxication alimentaire", "probabilite": "possible", "gravite": "faible a modere"},
        ],
        "red_flags": ["douleur thoracique", "douleur abdominale intense", "vomissements repetes"],
        "recommendation": "Surveillance; consultation si douleurs importantes, vomissements repetes ou signes de deshydratation.",
    },
    "constipation": {
        "category": "Digestif",
        "severity": "faible",
        "score": 1,
        "aliases": ["constipation", "constipe", "pas de selles", "transit bloque"],
        "pathologies": [
            {"pathologie": "Constipation fonctionnelle", "probabilite": "frequent", "gravite": "faible"},
            {"pathologie": "Occlusion intestinale", "probabilite": "rare mais a exclure si vomissements/douleur", "gravite": "critique"},
        ],
        "red_flags": ["ventre dur", "vomissements", "douleur intense", "arret gaz"],
        "recommendation": "Mesures hygieno-dietetiques si simple; consultation urgente si douleur intense, vomissements ou arret des gaz.",
    },
    "douleur musculo-articulaire": {
        "category": "Musculo-articulaire",
        "severity": "modere",
        "score": 2,
        "aliases": [
            "douleur main",
            "douleur mains",
            "douleur aux mains",
            "mal a la main",
            "mal aux mains",
            "douleur bras",
            "douleur jambe",
            "douleur poignet",
            "douleur articulation",
            "douleurs articulaires",
            "articulation douloureuse",
            "douleur genou",
            "douleur cheville",
            "douleur pied",
        ],
        "pathologies": [
            {"pathologie": "Entorse ou foulure", "probabilite": "possible selon traumatisme", "gravite": "faible a modere"},
            {"pathologie": "Contusion", "probabilite": "possible apres coup", "gravite": "faible a modere"},
            {"pathologie": "Tendinite ou surmenage", "probabilite": "possible si douleur progressive", "gravite": "faible a modere"},
            {"pathologie": "Fracture ou lesion importante", "probabilite": "a exclure si traumatisme/douleur intense", "gravite": "eleve"},
            {"pathologie": "Inflammation articulaire", "probabilite": "possible si gonflement/rougeur", "gravite": "modere"},
        ],
        "red_flags": ["douleur intense", "gonflement important", "traumatisme", "impossibilite de bouger", "deformation"],
        "recommendation": "Repos relatif et surveillance; consultation si douleur forte, gonflement, traumatisme ou mobilite reduite.",
    },
    "gonflement articulaire": {
        "category": "Musculo-articulaire",
        "severity": "modere",
        "score": 2,
        "aliases": ["main gonflee", "membre gonfle", "articulation gonflee", "gonflement articulaire", "chaleur locale articulaire"],
        "pathologies": [
            {"pathologie": "Inflammation articulaire ou tendineuse", "probabilite": "possible", "gravite": "modere"},
            {"pathologie": "Entorse ou contusion", "probabilite": "possible apres traumatisme", "gravite": "modere"},
            {"pathologie": "Infection locale", "probabilite": "a evoquer si rougeur/chaleur/fievre", "gravite": "eleve"},
        ],
        "red_flags": ["fievre", "douleur intense", "rougeur", "chaleur locale", "traumatisme"],
        "recommendation": "Consultation si gonflement important, rougeur, chaleur, fievre, traumatisme ou aggravation.",
    },
    "traumatisme membre": {
        "category": "Musculo-articulaire",
        "severity": "eleve",
        "score": 4,
        "aliases": ["traumatisme", "chute", "coup", "accident", "faux mouvement", "entorse", "fracture"],
        "pathologies": [
            {"pathologie": "Entorse", "probabilite": "possible", "gravite": "modere"},
            {"pathologie": "Contusion", "probabilite": "possible", "gravite": "faible a modere"},
            {"pathologie": "Fracture", "probabilite": "a exclure si douleur intense/deformation", "gravite": "eleve"},
        ],
        "red_flags": ["deformation", "impossibilite de bouger", "douleur intense", "engourdissement"],
        "recommendation": "Eviter de forcer; avis medical rapide si douleur importante, deformation, gonflement ou mobilite impossible.",
    },
    "difficulte a bouger": {
        "category": "Musculo-articulaire",
        "severity": "eleve",
        "score": 5,
        "aliases": [
            "difficulte a bouger",
            "difficulte de bouger",
            "impossible de bouger",
            "impossibilite de bouger",
            "mobilite reduite",
            "je ne peux pas bouger",
            "je ne peux plus bouger",
        ],
        "pathologies": [
            {"pathologie": "Entorse severe ou lesion ligamentaire", "probabilite": "possible", "gravite": "eleve"},
            {"pathologie": "Fracture ou luxation", "probabilite": "a exclure", "gravite": "eleve"},
            {"pathologie": "Inflammation importante", "probabilite": "possible", "gravite": "modere a eleve"},
        ],
        "red_flags": ["traumatisme", "deformation", "engourdissement", "douleur intense"],
        "recommendation": "Consultation rapide; urgence si impossibilite totale, deformation, traumatisme important ou trouble de sensibilite.",
    },
    "cephalee": {
        "category": "Neurologique",
        "severity": "modere",
        "score": 2,
        "aliases": ["cephalee", "maux de tete", "mal de tete", "migraine", "douleur tete", "trouble neurologique"],
        "pathologies": [
            {"pathologie": "Cephalee de tension", "probabilite": "frequent", "gravite": "faible"},
            {"pathologie": "Migraine", "probabilite": "possible", "gravite": "faible a modere"},
            {"pathologie": "Meningite ou cause neurologique grave", "probabilite": "a exclure si signes d'alerte", "gravite": "critique"},
        ],
        "red_flags": ["confusion", "raideur nuque", "deficit neurologique", "fievre elevee", "brutal"],
        "recommendation": "Consultation si inhabituel; urgence si brutal, confusion, raideur de nuque, deficit ou fievre elevee.",
    },
    "confusion": {
        "category": "Neurologique",
        "severity": "critique",
        "score": 5,
        "aliases": ["confusion", "desorientation", "trouble conscience", "somnolence inhabituelle", "perte d equilibre", "trouble equilibre", "trouble de l equilibre"],
        "pathologies": [
            {"pathologie": "Trouble neurologique aigu", "probabilite": "a evaluer en urgence", "gravite": "critique"},
            {"pathologie": "Infection severe ou trouble metabolique", "probabilite": "possible", "gravite": "critique"},
        ],
        "red_flags": ["fievre", "deficit neurologique", "perte de connaissance", "saturation basse"],
        "recommendation": "Urgence medicale; appeler le SAMU si confusion recente ou trouble de conscience.",
    },
    "mal de gorge": {
        "category": "ORL",
        "severity": "faible",
        "score": 1,
        "aliases": ["mal de gorge", "gorge douloureuse", "angine", "douleur gorge", "difficulte a avaler", "nez bouche", "nez qui coule", "douleur oreille", "mal oreille", "sinus"],
        "pathologies": [
            {"pathologie": "Rhinopharyngite", "probabilite": "frequent", "gravite": "faible"},
            {"pathologie": "Angine virale ou bacterienne", "probabilite": "possible", "gravite": "faible a modere"},
            {"pathologie": "Abces ORL", "probabilite": "rare, a evoquer si signes severes", "gravite": "eleve"},
        ],
        "red_flags": ["difficulte a respirer", "difficulte a avaler", "impossibilite d avaler", "fievre elevee", "raideur nuque"],
        "recommendation": "Surveillance et hydratation; consultation si fievre elevee, gene respiratoire, douleur intense ou impossibilite d'avaler.",
    },
    "brulure urinaire": {
        "category": "Urinaire",
        "severity": "modere",
        "score": 2,
        "aliases": ["brulure urinaire", "brulures urinaires", "douleur en urinant", "infection urinaire", "cystite", "envie frequente d uriner", "envie d uriner souvent", "douleur pipi", "pipi brule"],
        "pathologies": [
            {"pathologie": "Cystite", "probabilite": "possible", "gravite": "modere"},
            {"pathologie": "Pyelonephrite", "probabilite": "a evoquer si fievre/douleur lombaire", "gravite": "eleve"},
            {"pathologie": "Colique nephretique", "probabilite": "possible si douleur lombaire intense", "gravite": "eleve"},
        ],
        "red_flags": ["fievre", "douleur lombaire", "sang urines", "grossesse", "malaise"],
        "recommendation": "Consultation medicale si brulures persistantes; avis rapide si fievre, douleur lombaire, grossesse, sang ou malaise.",
    },
    "eruption cutanee": {
        "category": "Dermatologique",
        "severity": "modere",
        "score": 2,
        "aliases": ["eruption cutanee", "boutons", "plaques rouges", "rougeurs", "urticaire", "demangeaisons", "rash", "eczema", "infection cutanee", "peau rouge"],
        "pathologies": [
            {"pathologie": "Reaction allergique", "probabilite": "possible", "gravite": "modere"},
            {"pathologie": "Infection cutanee", "probabilite": "possible si douleur/chaleur/fievre", "gravite": "modere a eleve"},
            {"pathologie": "Dermatite ou eczema", "probabilite": "possible", "gravite": "faible a modere"},
        ],
        "red_flags": ["difficulte respiratoire", "gonflement visage", "fievre", "douleur intense", "extension rapide"],
        "recommendation": "Avis medical si extension, fievre, douleur ou doute; urgence si gene respiratoire ou gonflement du visage.",
    },
}


def _matched_symptoms(text: str) -> list[dict[str, Any]]:
    matches = []
    for canonical, entry in SYMPTOM_KNOWLEDGE.items():
        aliases = [canonical, *entry.get("aliases", [])]
        matched_aliases = [alias for alias in aliases if _has_positive_phrase(text, alias)]
        if matched_aliases:
            matches.append({"canonical": canonical, "matched_aliases": _dedupe(matched_aliases), **entry})
    return matches


def _temperature_severity(text: str) -> tuple[str | None, int]:
    temperatures = []
    for match in re.finditer(r"\b(3[8-9](?:[,.]\d)?|4[0-1](?:[,.]\d)?)\b", text):
        prefix = text[max(0, match.start() - 100):match.start()]
        if not _is_negated_context(prefix):
            temperatures.append(float(match.group(1).replace(",", ".")))

    if any(temp >= 40 for temp in temperatures):
        return "critique", 5
    if any(temp >= 39.5 for temp in temperatures):
        return "eleve", 4
    if any(temp >= 38 for temp in temperatures):
        return "modere", 1
    return None, 0


def _spo2_severity(text: str) -> tuple[str | None, int]:
    if re.search(r"\b(?:spo2|saturation|sat)\s*(?:<|inferieure a|moins de)?\s*(8[0-9]|9[0-2])\s*%?\b", text):
        return "critique", 6
    if re.search(r"\b(?:spo2|saturation|sat)\s*(?:<|inferieure a|moins de)?\s*(9[3-4])\s*%?\b", text):
        return "eleve", 4
    return None, 0


def _intensity_severity(text: str) -> tuple[str | None, int]:
    values: list[int] = []
    patterns = [
        r"\b(10|[1-9])\s*/\s*10\b",
        r"\b(10|[1-9])\s*sur\s*10\b",
        r"\b(10|[1-9])\s*(?:sur|/)\s*dix\b",
        r"\b(?:intensite|douleur|gene|mal)\s*(?:a|de|:)?\s*(10|[1-9])\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            prefix = text[max(0, match.start() - 100):match.start()]
            if not _is_negated_context(prefix):
                values.append(int(match.group(1)))
    if not values:
        return None, 0

    intensity = max(values)
    if intensity >= 9:
        return "eleve", 7
    if intensity >= 7:
        return "eleve", 5
    if intensity >= 4:
        return "modere", 3
    if intensity >= 1:
        return "faible", 1
    return None, 0


def _highest_severity(levels: list[str]) -> str:
    if not levels:
        return "faible"
    return max(levels, key=lambda level: SEVERITY_RANK.get(level, 0))


def _recommendation_for(severity: str, categories: list[str]) -> str:
    if severity == "critique":
        return "Urgence : appeler le SAMU/les urgences si les symptomes sont presents, severes ou s'aggravent."
    if severity == "eleve":
        if "Cardiaque" in categories:
            return "Avis medical urgent pour douleur thoracique/cardiaque; SAMU si douleur persistante, oppression, malaise, sueurs, essoufflement ou irradiation."
        if "Musculo-articulaire" in categories:
            return "Consultation medicale rapide si douleur forte, traumatisme, gonflement important ou mobilite reduite."
        return "Consultation medicale rapide ou urgence selon l'intensite, l'evolution et le terrain du patient."
    if severity == "modere":
        if "Cardiaque" in categories:
            return "Douleur thoracique/cardiaque a ne pas banaliser : avis medical recommande, urgence si persistance ou signe associe."
        if "Musculo-articulaire" in categories:
            return "Repos relatif, eviter de forcer, surveiller gonflement/rougeur/mobilite et consulter si persistance ou aggravation."
        return "Consultation medicale recommandee si symptomes persistants, inhabituels, associes ou aggravation."
    if categories == ["General"]:
        return "Surveillance prudente, hydratation/repos si adapte, et avis medical en cas de doute ou d'aggravation."
    return "Surveillance prudente avec consultation si persistance, aggravation ou apparition de signes d'alerte."


def _evolution_score(text: str) -> tuple[str | None, int]:
    score = 0
    levels: list[str] = []
    if any(_has_positive_phrase(text, phrase) for phrase in ["aggravation rapide", "s aggrave rapidement", "empire rapidement"]):
        score += 3
        levels.append("eleve")
    if any(_has_positive_phrase(text, phrase) for phrase in ["brutal", "apparition brutale", "soudain"]):
        score += 2
        levels.append("modere")
    if any(_has_positive_phrase(text, phrase) for phrase in ["douleur intense", "douleur insupportable", "impossibilite de boire"]):
        score += 3
        levels.append("eleve")
    return (_highest_severity(levels) if levels else None), score


def analyse_medical_text(text: str) -> dict[str, Any]:
    normalized_text = _normalise(text)
    matches = _matched_symptoms(normalized_text)

    if not matches:
        generic_categories = {
            "cardiaque": {"canonical": "symptomes cardiaques", "category": "Cardiaque", "severity": "eleve", "score": 3, "recommendation": "Avis medical recommande pour tout symptome cardiaque.", "pathologies": [{"pathologie": "Trouble cardiaque a explorer", "probabilite": "possible", "gravite": "eleve"}], "matched_aliases": ["cardiaque", "coeur"]},
            "respiratoire": {"canonical": "symptomes respiratoires", "category": "Respiratoire", "severity": "modere", "score": 2, "recommendation": "Surveiller la respiration.", "pathologies": [{"pathologie": "Atteinte respiratoire", "probabilite": "possible", "gravite": "modere"}], "matched_aliases": ["respiratoire", "poumon"]},
            "digestif": {"canonical": "symptomes digestifs", "category": "Digestif", "severity": "faible", "score": 1, "recommendation": "Surveiller l'hydratation.", "pathologies": [{"pathologie": "Trouble digestif", "probabilite": "possible", "gravite": "faible a modere"}], "matched_aliases": ["digestif", "ventre", "estomac"]},
            "neurologique": {"canonical": "symptomes neurologiques", "category": "Neurologique", "severity": "eleve", "score": 4, "recommendation": "Avis medical urgent pour tout trouble neurologique.", "pathologies": [{"pathologie": "Trouble neurologique a explorer", "probabilite": "possible", "gravite": "eleve"}], "matched_aliases": ["neurologique", "cerveau"]},
            "musculo-articulaire": {"canonical": "symptomes musculo-articulaires", "category": "Musculo-articulaire", "severity": "faible", "score": 1, "recommendation": "Repos et surveillance.", "pathologies": [{"pathologie": "Atteinte musculo-squelettique", "probabilite": "possible", "gravite": "faible a modere"}], "matched_aliases": ["musculo", "articulaire", "muscle", "articulation", "os"]},
            "orl": {"canonical": "symptomes ORL", "category": "ORL", "severity": "faible", "score": 1, "recommendation": "Surveillance.", "pathologies": [{"pathologie": "Infection ORL", "probabilite": "possible", "gravite": "faible"}], "matched_aliases": ["orl", "gorge", "oreille", "nez"]},
            "urinaire": {"canonical": "symptomes urinaires", "category": "Urinaire", "severity": "faible", "score": 1, "recommendation": "Hydratation et surveillance.", "pathologies": [{"pathologie": "Trouble urinaire", "probabilite": "possible", "gravite": "faible"}], "matched_aliases": ["urinaire", "urine", "rein"]},
            "dermatologique": {"canonical": "symptomes dermatologiques", "category": "Dermatologique", "severity": "faible", "score": 1, "recommendation": "Surveillance de l'eruption.", "pathologies": [{"pathologie": "Atteinte dermatologique", "probabilite": "possible", "gravite": "faible"}], "matched_aliases": ["dermatologique", "cutane", "peau"]},
            "infectieux": {"canonical": "symptomes infectieux", "category": "Infectieux/Febrile", "severity": "faible", "score": 1, "recommendation": "Surveiller la temperature.", "pathologies": [{"pathologie": "Syndrome infectieux", "probabilite": "possible", "gravite": "faible"}], "matched_aliases": ["infectieux", "febrile", "infection", "fievre"]},
        }
        for kw, data in generic_categories.items():
            if _has_positive_phrase(normalized_text, kw) or any(_has_positive_phrase(normalized_text, alias) for alias in data["matched_aliases"]):
                matches.append(data)

    if not matches:
        return {
            "query": text,
            "found": False,
            "normalized_symptoms": [],
            "categories": [],
            "clinical_score_hint": 0,
            "severity_level": "faible",
            "recommendation": "Aucune donnee MCP specifique. Reformuler les symptomes ou demander un avis medical en cas de doute.",
            "pathologies": [],
            "red_flags": [],
            "message": "Aucun symptome reconnu dans la base MCP.",
        }

    categories = _dedupe([match["category"] for match in matches])
    score = sum(int(match.get("score", 0)) for match in matches)
    severity_levels = [match["severity"] for match in matches]

    temp_severity, temp_score = _temperature_severity(normalized_text)
    spo2_severity, spo2_score = _spo2_severity(normalized_text)
    intensity_severity, intensity_score = _intensity_severity(normalized_text)
    evolution_severity, evolution_score = _evolution_score(normalized_text)
    score += temp_score + spo2_score + intensity_score + evolution_score
    if temp_severity:
        severity_levels.append(temp_severity)
    if spo2_severity:
        severity_levels.append(spo2_severity)
    if intensity_severity:
        severity_levels.append(intensity_severity)
    if evolution_severity:
        severity_levels.append(evolution_severity)

    canonical_names = [match["canonical"] for match in matches]
    if "douleur thoracique" in canonical_names and ("essoufflement" in canonical_names or "malaise" in canonical_names):
        score += 4
        severity_levels.append("critique")
    if "saturation basse" in canonical_names and "essoufflement" in canonical_names:
        score += 3
        severity_levels.append("critique")
    if "douleur abdominale" in canonical_names and ("vomissements" in canonical_names or "fievre" in canonical_names):
        score += 2
        severity_levels.append("eleve")
    if "douleur musculo-articulaire" in canonical_names and (
        "traumatisme membre" in canonical_names
        or "difficulte a bouger" in canonical_names
        or "gonflement articulaire" in canonical_names
    ):
        score += 3
        severity_levels.append("eleve")

    computed_severity = _highest_severity(severity_levels)
    if score >= 12:
        computed_severity = _highest_severity([computed_severity, "critique"])
    elif score >= 7:
        computed_severity = _highest_severity([computed_severity, "eleve"])
    elif score >= 3:
        computed_severity = _highest_severity([computed_severity, "modere"])

    pathology_map: dict[str, dict[str, Any]] = {}
    for match in matches:
        for pathology in match.get("pathologies", []):
            name = pathology["pathologie"]
            if name not in pathology_map:
                pathology_map[name] = {
                    "pathologie": name,
                    "probabilite": pathology.get("probabilite", "possible"),
                    "gravite": pathology.get("gravite", match["severity"]),
                    "categories": [match["category"]],
                    "symptomes_support": [match["canonical"]],
                }
            else:
                pathology_map[name]["categories"].append(match["category"])
                pathology_map[name]["symptomes_support"].append(match["canonical"])

    red_flags = []
    for match in matches:
        for red_flag in match.get("red_flags", []):
            if _has_positive_phrase(normalized_text, red_flag):
                red_flags.append(red_flag)
    red_flags = _dedupe(red_flags)
    if red_flags:
        score += min(len(red_flags), 3) * 2
        severity_levels.append("modere")
        if any(flag in red_flags for flag in ["difficulte a respirer", "gonflement visage", "perte de connaissance"]):
            severity_levels.append("eleve")

    computed_severity = _highest_severity(severity_levels)
    if score >= 12:
        computed_severity = _highest_severity([computed_severity, "critique"])
    elif score >= 7:
        computed_severity = _highest_severity([computed_severity, "eleve"])
    elif score >= 3:
        computed_severity = _highest_severity([computed_severity, "modere"])

    category_counts = Counter(match["category"] for match in matches)
    dominant_categories = [category for category, _ in category_counts.most_common()]

    return {
        "query": text,
        "found": True,
        "normalized_symptoms": [
            {
                "symptom": match["canonical"],
                "category": match["category"],
                "matched_aliases": match["matched_aliases"],
                "severity": match["severity"],
                "recommendation": match["recommendation"],
            }
            for match in matches
        ],
        "categories": dominant_categories,
        "clinical_score_hint": score,
        "severity_level": SEVERITY_LABELS[computed_severity],
        "recommendation": _recommendation_for(computed_severity, dominant_categories),
        "pathologies": list(pathology_map.values()),
        "red_flags": red_flags,
        "disclaimer": "Informations indicatives pour orientation preliminaire. Seul un professionnel de sante peut poser un diagnostic.",
    }


@mcp.tool()
def search_symptoms(symptom: str) -> dict:
    """Recherche les orientations possibles associees a un symptome ou a un texte clinique."""
    return analyse_medical_text(symptom)


DRUG_DB = {
    "paracetamol": {
        "classe": "Antipyretique / analgesique",
        "posologie_adulte": "500 mg a 1 g toutes les 4 a 6 h. Max 3 g/jour sans avis medical.",
        "contre_indications": "Insuffisance hepatique severe, allergie connue.",
        "interactions": "Anticoagulants oraux : surveillance medicale.",
        "remarque": "Medicament courant pour fievre et douleurs legeres a moderees; attention aux doublons contenant du paracetamol.",
    },
    "ibuprofene": {
        "classe": "Anti-inflammatoire non steroidien",
        "posologie_adulte": "200 a 400 mg toutes les 6 a 8 h. Max 1200 mg/jour sans avis medical.",
        "contre_indications": "Grossesse avancee, ulcere gastrique, insuffisance renale, allergie AINS, varicelle.",
        "interactions": "Anticoagulants, aspirine, corticoides, certains antihypertenseurs.",
        "remarque": "A eviter en cas de douleur abdominale non expliquee, infection severe ou deshydratation sans avis medical.",
    },
    "amoxicilline": {
        "classe": "Antibiotique beta-lactamine",
        "posologie_adulte": "Selon prescription medicale uniquement.",
        "contre_indications": "Allergie aux penicillines ou beta-lactamines.",
        "interactions": "Anticoagulants, methotrexate.",
        "remarque": "Ne pas utiliser sans prescription; inutile sur la plupart des infections virales.",
    },
    "salbutamol": {
        "classe": "Bronchodilatateur beta-2 mimetique",
        "posologie_adulte": "Selon prescription. Urgences si besoin repete ou crise severe.",
        "contre_indications": "Hypersensibilite au principe actif.",
        "interactions": "Betabloquants, certains traitements cardiaques.",
        "remarque": "Traitement de secours respiratoire; avis urgent si soulagement insuffisant ou dyspnee severe.",
    },
}

DRUG_ALIASES = {
    "paracetamol": "paracetamol",
    "doliprane": "paracetamol",
    "efferalgan": "paracetamol",
    "dafalgan": "paracetamol",
    "ibuprofene": "ibuprofene",
    "advil": "ibuprofene",
    "nurofen": "ibuprofene",
    "amoxicilline": "amoxicilline",
    "clamoxyl": "amoxicilline",
    "salbutamol": "salbutamol",
    "ventoline": "salbutamol",
}


@mcp.tool()
def get_drug_info(drug_name: str) -> dict:
    """Retourne des informations prudentes sur un medicament courant."""
    drug_key = _normalise(drug_name)
    drug_key = DRUG_ALIASES.get(drug_key, drug_key)

    info = DRUG_DB.get(drug_key)
    if not info:
        info = next(
            (
                values
                for key, values in DRUG_DB.items()
                if drug_key in _normalise(key) or _normalise(key) in drug_key
            ),
            None,
        )

    if not info:
        return {
            "drug": drug_name,
            "found": False,
            "message": f"Medicament '{drug_name}' non trouve. Consultez un pharmacien ou un medecin.",
        }

    return {
        "drug": drug_key,
        "found": True,
        **info,
        "disclaimer": "Toujours suivre la prescription medicale, lire la notice et demander conseil en cas de doute.",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
