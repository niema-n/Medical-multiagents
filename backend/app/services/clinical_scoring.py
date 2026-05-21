from __future__ import annotations

import re
import unicodedata
from typing import Any


CARDIAC_KEYWORDS = [
    "douleur thoracique",
    "douleurs thoraciques",
    "oppression thoracique",
    "oppression dans la poitrine",
    "oppression poitrine",
    "douleur poitrine",
    "douleurs poitrine",
    "douleur dans la poitrine",
    "douleurs dans la poitrine",
    "douleur au coeur",
    "douleurs au coeur",
    "mal au coeur",
    "maux au coeur",
    "douleur cardiaque",
    "douleurs cardiaques",
    "serrement poitrine",
    "poitrine serree",
    "poitrine serre",
    "coeur serre",
    "coeur serree",
    "gene poitrine",
    "gene thoracique",
    "malaise",
    "palpitations",
    "sueurs",
    "sueurs froides",
    "angor",
    "angine de poitrine",
    "infarctus",
    "infarctus du myocarde",
    "pericardite",
    "insuffisance cardiaque",
    "embolie pulmonaire",
    "douleur bras gauche",
    "douleur au bras gauche",
    "irradiation bras",
    "irradiation machoire",
    "irradiation dos",
    "probleme cardiaque",
    "problemes cardiaques",
    "probleme cadiaque",
    "problemes cadiaques",
    "souci cardiaque",
    "souci cadiaque",
    "cardiaque",
    "cadiaque",
]

RESPIRATORY_KEYWORDS = [
    "toux",
    "respiration",
    "probleme de respiration",
    "problemes de respiration",
    "probleme respiratoire",
    "respiration genee",
    "gene respiratoire",
    "essoufflement",
    "souffle court",
    "difficulte respiratoire",
    "difficulte a respirer",
    "difficultes a respirer",
    "mal a respirer",
    "difficultes a respirer",
    "dyspnee",
    "manque d air",
    "respire mal",
    "mal a respirer",
    "je respire mal",
    "saturation basse",
    "spo2 basse",
    "desaturation",
    "asthme",
    "bpco",
    "pneumonie",
    "bronchite",
    "detresse respiratoire",
    "sifflement",
    "douleur a la respiration",
    "douleur respiratoire",
    "crachats",
    "expectorations",
]

DIGESTIVE_KEYWORDS = [
    "mal au ventre",
    "maux de ventre",
    "douleur abdominale",
    "douleur au ventre",
    "douleur estomac",
    "mal estomac",
    "mal a l estomac",
    "brulure estomac",
    "abdominal",
    "vomissement",
    "vomissements",
    "diarrhee",
    "diarrhee importante",
    "selles liquides",
    "sang dans les selles",
    "sang selles",
    "ventre dur",
    "mal digestion",
    "nausee",
    "nausees",
    "envie de vomir",
    "constipation",
    "constipe",
    "intoxication alimentaire",
    "gastro enterite",
    "gastro-enterite",
    "appendicite",
    "colique abdominale",
    "repas suspect",
    "douleur bas ventre",
    "douleur bas a droite",
]

NEUROLOGIC_KEYWORDS = [
    "confusion",
    "convulsion",
    "paralysie",
    "vertige",
    "perte de connaissance",
    "trouble conscience",
    "desorientation",
    "somnolence inhabituelle",
    "faiblesse d un cote",
    "trouble de la parole",
    "trouble de la vision",
    "trouble de l equilibre",
    "perte d equilibre",
    "trouble equilibre",
    "trouble neurologique",
    "maux de tete",
    "mal de tete",
    "cephalee",
    "migraine",
    "infection neuromeningee",
    "meningite",
    "avc",
    "accident neurologique",
    "accident vasculaire",
    "raideur de nuque",
    "raideur nuque",
    "photophobie",
    "deficit neurologique",
    "deficit focal",
]

MUSCULOSKELETAL_KEYWORDS = [
    "douleur main",
    "douleur mains",
    "douleur aux mains",
    "main douloureuse",
    "mains douloureuses",
    "mal main",
    "mal mains",
    "mal au main",
    "mal aux main",
    "mal a la main",
    "mal aux mains",
    "douleur poignet",
    "douleur bras",
    "douleur avant bras",
    "douleur coude",
    "douleur epaule",
    "douleur jambe",
    "douleur genou",
    "douleur cheville",
    "douleur pied",
    "douleur articulation",
    "douleurs articulaires",
    "articulation douloureuse",
    "tendon douloureux",
    "douleur tendon",
    "douleur musculaire",
    "douleurs musculaires",
    "douleur muscle",
    "douleur muscles",
    "main gonflee",
    "membre gonfle",
    "articulation gonflee",
    "traumatisme",
    "chute",
    "coup",
    "faux mouvement",
    "effort inhabituel",
    "accident",
    "deformation",
    "entorse",
    "foulure",
    "contusion",
    "tendinite",
    "fracture",
    "difficulte a bouger",
    "difficulte de bouger",
    "impossible de bouger",
    "impossibilite de bouger",
    "mobilite reduite",
    "raideur articulaire",
]

ENT_KEYWORDS = [
    "mal de gorge",
    "gorge douloureuse",
    "douleur gorge",
    "angine",
    "difficulte a avaler",
    "difficulte avaler",
    "impossibilite d avaler",
    "nez qui coule",
    "nez bouche",
    "douleur oreille",
    "mal oreille",
    "sinus",
    "rhinopharyngite",
    "rhume",
    "sinusite",
    "otite",
    "ganglion",
    "ganglions",
    "voix modifiee",
    "voix changee",
]

URINARY_KEYWORDS = [
    "brulure urinaire",
    "brulures urinaires",
    "brulures en urinant",
    "brulure en urinant",
    "douleur en urinant",
    "infection urinaire",
    "cystite",
    "pyelonephrite",
    "envie frequente d uriner",
    "envie d uriner souvent",
    "envie frequente",
    "envies frequentes",
    "urines troubles",
    "douleur pipi",
    "pipi brule",
    "sang urines",
    "sang dans les urines",
    "douleur lombaire",
    "colique nephretique",
]

DERMATOLOGIC_KEYWORDS = [
    "eruption cutanee",
    "boutons",
    "plaques rouges",
    "rougeurs",
    "eczema",
    "infection cutanee",
    "peau rouge",
    "douleur locale peau",
    "chaleur locale",
    "suintement",
    "lesion cutanee",
    "lesions cutanees",
    "urticaire",
    "demangeaisons",
    "rash",
    "gonflement visage",
    "reaction allergique",
    "allergie cutanee",
    "dermatite",
    "gonflement des levres",
    "levres gonflees",
    "piqure",
    "rougeur douloureuse",
]

FEBRILE_INFECTIOUS_KEYWORDS = [
    "fievre",
    "temperature",
    "etat febrile",
    "frissons",
    "courbatures",
    "infection",
    "grippe",
    "syndrome infectieux",
    "infection virale",
    "fatigue",
    "fatigue importante",
    "somnolence",
    "doliprane",
    "paracetamol",
    "prise doliprane",
    "prise paracetamol",
    "taches cutanees",
    "purpura",
    "marbrures",
    "fievre elevee",
    "temperature elevee",
]

GENERAL_KEYWORDS = [
    "fievre elevee",
    "fatigue",
    "aggravation",
    "alteration etat general",
    "deshydratation",
    "immunodepression",
    "grossesse",
    "personne agee",
]

SEVERITY_WEIGHTS = {
    "douleur thoracique": 7,
    "douleurs thoraciques": 7,
    "oppression thoracique": 7,
    "oppression dans la poitrine": 7,
    "oppression poitrine": 7,
    "douleur poitrine": 7,
    "douleurs poitrine": 7,
    "douleur dans la poitrine": 7,
    "douleurs dans la poitrine": 7,
    "douleur au coeur": 7,
    "douleurs au coeur": 7,
    "mal au coeur": 7,
    "maux au coeur": 7,
    "douleur cardiaque": 7,
    "douleurs cardiaques": 7,
    "serrement poitrine": 7,
    "poitrine serree": 7,
    "poitrine serre": 7,
    "coeur serre": 7,
    "probleme cardiaque": 7,
    "essoufflement": 5,
    "souffle court": 5,
    "probleme de respiration": 4,
    "problemes de respiration": 4,
    "probleme respiratoire": 4,
    "respiration genee": 5,
    "gene respiratoire": 4,
    "difficulte a respirer": 5,
    "difficulte respiratoire": 5,
    "difficultes a respirer": 5,
    "saturation basse": 6,
    "spo2 basse": 6,
    "desaturation": 6,
    "malaise": 4,
    "palpitations": 3,
    "sueurs froides": 3,
    "sueurs": 2,
    "probleme cardiaque": 3,
    "problemes cardiaques": 3,
    "probleme cadiaque": 3,
    "problemes cadiaques": 3,
    "souci cardiaque": 3,
    "souci cadiaque": 3,
    "cardiaque": 2,
    "cadiaque": 2,
    "perte de connaissance": 5,
    "confusion": 5,
    "trouble conscience": 5,
    "convulsion": 5,
    "paralysie": 5,
    "faiblesse d un cote": 5,
    "trouble de la parole": 5,
    "somnolence inhabituelle": 4,
    "raideur de nuque": 5,
    "photophobie": 3,
    "meningite": 5,
    "avc": 6,
    "fievre elevee": 4,
    "temperature elevee": 4,
    "fievre": 1,
    "frissons": 2,
    "courbatures": 1,
    "fatigue importante": 1,
    "doliprane": 1,
    "paracetamol": 1,
    "toux": 1,
    "aggravation": 3,
    "vomissements": 2,
    "vomissement": 2,
    "mal au ventre": 1,
    "douleur au ventre": 2,
    "douleur estomac": 2,
    "douleur abdominale": 2,
    "diarrhee": 1,
    "sang dans les selles": 5,
    "ventre dur": 4,
    "appendicite": 4,
    "gastro enterite": 2,
    "gastro-enterite": 2,
    "intoxication alimentaire": 2,
    "nausee": 1,
    "nausees": 1,
    "constipation": 1,
    "douleur main": 2,
    "douleur mains": 2,
    "douleur aux mains": 2,
    "main douloureuse": 2,
    "mains douloureuses": 2,
    "mal main": 2,
    "mal mains": 2,
    "mal au main": 2,
    "mal aux main": 2,
    "mal a la main": 2,
    "mal aux mains": 2,
    "douleur poignet": 2,
    "douleur bras": 2,
    "douleur jambe": 2,
    "douleur articulation": 2,
    "douleurs articulaires": 2,
    "main gonflee": 2,
    "membre gonfle": 2,
    "articulation gonflee": 2,
    "faux mouvement": 2,
    "effort inhabituel": 1,
    "deformation": 5,
    "entorse": 3,
    "foulure": 2,
    "contusion": 2,
    "tendinite": 1,
    "fracture": 5,
    "antecedents": 1,
    "asthme": 1,
    "sifflement": 2,
    "douleur a la respiration": 3,
    "douleur respiratoire": 4,
    "crachats": 1,
    "diabete": 1,
    "immunodepression": 2,
    "grossesse": 2,
    "personne agee": 2,
    "deshydratation": 3,
    "alteration etat general": 3,
    "mal de gorge": 1,
    "difficulte a avaler": 2,
    "difficulte avaler": 2,
    "otite": 1,
    "sinusite": 1,
    "brulure urinaire": 2,
    "brulures en urinant": 2,
    "infection urinaire": 2,
    "urines troubles": 1,
    "douleur lombaire": 4,
    "sang urines": 3,
    "sang dans les urines": 3,
    "pyelonephrite": 4,
    "eruption cutanee": 2,
    "urticaire": 2,
    "chaleur locale": 2,
    "suintement": 2,
    "gonflement des levres": 5,
    "reaction allergique": 3,
    "purpura": 5,
    "marbrures": 4,
    "grippe": 1,
    "gonflement visage": 5,
    "maladie chronique": 1,
}


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9<>=.,% ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def flatten_text(*values: Any) -> str:
    parts: list[str] = []
    for value in values:
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif isinstance(value, tuple):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return normalize_text(" ".join(parts))


AFFIRMATIVE_ANSWERS = {
    "oui",
    "oui beaucoup",
    "beaucoup",
    "fort",
    "forte",
    "tres fort",
    "tres forte",
    "intense",
    "douleur",
    "gene",
    "grave",
}

NEGATIVE_ANSWERS = {
    "non",
    "non pas",
    "pas",
    "aucun",
    "aucune",
    "rien",
    "pas du tout",
}


def _answer_is_negative(answer: str) -> bool:
    text = normalize_text(answer)
    if text in NEGATIVE_ANSWERS:
        return True
    if text.startswith("pas de ") or text.startswith("pas d ") or text.startswith("aucun ") or text.startswith("aucune "):
        return True
    return any(
        pattern in text
        for pattern in [
            "pas de douleur",
            "pas douleur",
            "pas gene",
            "pas d essoufflement",
            "pas essouffle",
            "aucune douleur",
            "aucune gene",
            "non pas de",
        ]
    )


def _answer_is_short_affirmative(answer: str) -> bool:
    text = normalize_text(answer)
    words = text.split()
    if text in AFFIRMATIVE_ANSWERS:
        return True
    if len(words) <= 3 and any(word in text for word in ["oui", "douleur", "beaucoup", "fort", "intense", "gene"]):
        return True
    return False


def _answer_has_intensity(answer: str) -> bool:
    return bool(re.search(r"\b(10|[1-9])\s*(?:/|sur)\s*(?:10|dix)\b", normalize_text(answer)))


def infer_contextual_symptoms(
    patient_case: Any,
    asked_questions: list[str] | tuple[str, ...] | None,
    patient_responses: list[str] | tuple[str, ...] | None,
) -> list[str]:
    """
    Transform short patient answers into clinically meaningful symptoms using
    the question that produced them. Raw answers stay untouched for display.
    """
    questions = list(asked_questions or [])
    responses = list(patient_responses or [])
    inferred: list[str] = []

    for question, answer in zip(questions, responses):
        q = normalize_text(question)
        a = normalize_text(answer)
        if not a:
            continue

        if _answer_is_negative(a):
            if "parler" in q and "respir" in q:
                inferred.append(
                    f"reponse contextualisee: question={q}; reponse={a} ; "
                    "respiration tres genee ; ne peut pas parler"
                )
            if any(term in q for term in ["bouger", "plier", "utiliser", "marcher"]):
                inferred.append(
                    f"reponse contextualisee: question={q}; reponse={a} ; "
                    "impossible de bouger ; mobilite reduite"
                )
            continue

        short_affirmative = _answer_is_short_affirmative(a)
        mentions_pain = has_positive_keyword(a, "douleur") or has_positive_keyword(a, "mal") or short_affirmative
        mentions_severity = any(term in a for term in ["beaucoup", "fort", "forte", "intense", "grave", "tres"])
        parts = [f"reponse contextualisee: reponse={a}"]

        if "respir" in q and ("douleur" in q or "profondement" in q) and mentions_pain:
            parts.extend(["douleur respiratoire", "douleur a la respiration"])
            if mentions_severity:
                parts.append("douleur respiratoire intense")

        if "respir" in q and any(term in a for term in ["essouffle", "souffle court", "respire mal", "gene"]):
            parts.extend(["respiration genee", "essoufflement"])

        if "parler" in q and any(term in a for term in ["non", "difficile", "pas bien", "mots"]):
            parts.extend(["respiration tres genee", "ne peut pas parler"])

        if "saturation" in q and re.search(r"\b(8[0-9]|9[0-4])\s*%?\b", a):
            parts.append("saturation basse")

        if ("evolution" in q or "depuis" in q) and any(term in a for term in ["aggrave", "rapid", "vite", "empire"]):
            parts.append("aggravation rapide")

        if ("intensite" in q or "echelle" in q or "sur 10" in q) and _answer_has_intensity(a):
            parts.append(a)

        if "neurolog" in q and short_affirmative and any(term in q for term in ["parole", "vision", "faiblesse", "equilibre", "visage"]):
            if "parole" in q:
                parts.append("trouble de la parole")
            if "faiblesse" in q:
                parts.append("faiblesse d un cote")
            if "vision" in q:
                parts.append("trouble de la vision")
            if "equilibre" in q:
                parts.append("trouble de l equilibre")

        if len(parts) > 1:
            inferred.append(" ; ".join(parts))

    return inferred


def build_contextual_clinical_text(
    patient_case: Any,
    patient_responses: list[str] | tuple[str, ...] | None = None,
    asked_questions: list[str] | tuple[str, ...] | None = None,
) -> str:
    try:
        inferred = infer_contextual_symptoms(patient_case, asked_questions, patient_responses)
    except Exception:
        inferred = []
    return flatten_text(patient_case, patient_responses or [], inferred)


def _is_negated_context(prefix: str) -> bool:
    words = prefix.split()
    tail = " ".join(words[-10:])
    short_tail = " ".join(words[-6:])
    if " mais " not in f" {short_tail} " and re.search(
        r"\b(?:pas|aucun|aucune|sans|ni|absence|non)\b(?:\s+\w+){0,4}$",
        short_tail,
    ):
        return True

    negation_patterns = [
        r"(?:pas|absence|sans|aucun|aucune|ni|non)(?:\s+(?:de|d|du|des))?\s*$",
        r"ne\s+presente\s+pas(?:\s+(?:de|d|du|des))?\s*$",
        r"n\s+a\s+pas(?:\s+(?:de|d|du|des))?\s*$",
        r"n\s+ai\s+pas(?:\s+(?:de|d|du|des))?\s*$",
        r"n\s+est\s+pas\s+(?:en)?\s*$",
        r"jamais(?:\s+(?:de|d|du|des))?\s*$",
    ]
    return any(re.search(pattern, tail) for pattern in negation_patterns)


def has_positive_keyword(text: str, keyword: str) -> bool:
    keyword = normalize_text(keyword)
    text = normalize_text(text)

    if keyword not in text:
        return False

    for match in re.finditer(re.escape(keyword), text):
        prefix = text[max(0, match.start() - 100):match.start()]
        if not _is_negated_context(prefix):
            return True

    return False


def has_any_positive_keyword(text: str, keywords: list[str]) -> bool:
    return any(has_positive_keyword(text, keyword) for keyword in keywords)


def _temperature_score(text: str) -> int:
    temperatures = []
    for match in re.finditer(r"\b(3[8-9](?:[,.]\d)?|4[0-1](?:[,.]\d)?)\b", text):
        prefix = text[max(0, match.start() - 100):match.start()]
        if not _is_negated_context(prefix):
            temperatures.append(float(match.group(1).replace(",", ".")))

    if any(temp >= 40 for temp in temperatures):
        return 7
    if any(temp >= 39.5 for temp in temperatures):
        return 5
    if any(temp >= 38 for temp in temperatures):
        return 2
    return 0


def _intensity_score(text: str) -> int:
    values: list[int] = []
    patterns = [
        r"\b(10|[1-9])\s*/\s*10\b",
        r"\b(10|[1-9])\s*sur\s*10\b",
        r"\b(10|[1-9])\s*(?:sur|/)\s*dix\b",
        r"\b(?:intensite|douleur|gene|mal)\s*(?:a|de|:)?\s*(10|[1-9])\b",
        r"\b(?:notee?|evaluee?)\s*(?:a|:)?\s*(10|[1-9])\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            prefix = text[max(0, match.start() - 100):match.start()]
            if not _is_negated_context(prefix):
                values.append(int(match.group(1)))

    if not values:
        return 0

    intensity = max(values)
    if intensity >= 9:
        return 8
    if intensity >= 7:
        return 6
    if intensity >= 4:
        return 4
    if intensity >= 1:
        return 2
    return 0


def _musculoskeletal_red_flag_score(text: str) -> int:
    score = 0
    musculo_context = has_any_positive_keyword(
        text,
        [
            "douleur main",
            "douleur aux mains",
            "douleur bras",
            "douleur jambe",
            "douleur articulation",
            "mal aux mains",
            "poignet",
            "genou",
            "cheville",
            "membre",
        ],
    )
    if has_any_positive_keyword(text, ["aggravation rapide", "s aggrave vite", "aggrave rapidement"]):
        score += 3
    if musculo_context and has_any_positive_keyword(text, ["gonflement important", "tres gonfle", "main gonflee", "membre gonfle"]):
        score += 3
    if has_any_positive_keyword(text, ["traumatisme", "chute", "coup violent", "accident"]):
        score += 4
    if has_any_positive_keyword(
        text,
        [
            "impossible de bouger",
            "impossibilite de bouger",
            "ne peux pas bouger",
            "ne peut pas bouger",
            "difficulte a bouger",
            "mobilite reduite",
        ],
    ):
        score += 5
    return score


def _duration_evolution_score(text: str) -> int:
    score = 0
    if has_any_positive_keyword(
        text,
        [
            "aggravation rapide",
            "s aggrave rapidement",
            "s aggrave vite",
            "empire rapidement",
            "de plus en plus",
            "douleur qui augmente",
        ],
    ):
        score += 4
    if has_any_positive_keyword(text, ["brutal", "apparition brutale", "debut brutal", "soudain"]):
        score += 3
    if re.search(r"\bdepuis\s+(?:plus de\s+)?(?:[3-9]|1[0-9])\s+jours\b", text):
        score += 2
    if re.search(r"\bdepuis\s+(?:plus de\s+)?(?:2|3|4|5|6|7|8|9|1[0-9])\s+semaines\b", text):
        score += 1
    return score


def _transversal_red_flag_score(text: str) -> int:
    score = 0
    if has_any_positive_keyword(text, ["impossibilite de boire", "ne peut pas boire", "vomissements repetes"]):
        score += 4
    if has_any_positive_keyword(text, ["sang dans les selles", "sang selles", "vomissement sanglant", "crache du sang"]):
        score += 5
    if has_any_positive_keyword(text, ["douleur insupportable", "douleur tres forte", "douleur intense"]):
        score += 4
    if has_any_positive_keyword(text, ["membre froid", "main froide", "pied froid", "bleu", "cyanose"]):
        score += 5
    if has_any_positive_keyword(text, ["malaise", "sensation de malaise", "evanouissement", "perte de connaissance"]):
        score += 5
    return score


def _terrain_score(text: str) -> int:
    score = 0
    if has_any_positive_keyword(text, ["immunodepression", "chimiotherapie", "greffe", "corticoides au long cours"]):
        score += 3
    if has_any_positive_keyword(text, ["grossesse", "enceinte", "diabete", "insuffisance cardiaque", "bpco", "maladie chronique"]):
        score += 2
    if has_any_positive_keyword(text, ["personne agee", "plus de 75 ans", "terrain fragile"]):
        score += 2
    return score


def _respiratory_distress_score(text: str) -> int:
    score = 0
    if has_any_positive_keyword(
        text,
        [
            "ne peut pas parler",
            "ne peux pas parler",
            "je ne peux pas parler",
            "impossible de parler",
            "parle par mots",
            "phrases impossibles",
            "respiration tres genee",
        ],
    ):
        score += 8
    if has_any_positive_keyword(text, ["tirage", "levres bleues", "cyanose", "detresse respiratoire"]):
        score += 5
    if has_any_positive_keyword(text, ["aggravation respiratoire", "essoufflement au repos"]):
        score += 3
    return score


def _neurologic_red_flag_score(text: str) -> int:
    score = 0
    if has_any_positive_keyword(text, ["debut brutal", "apparition brutale", "au reveil avec deficit"]):
        score += 3
    if has_any_positive_keyword(text, ["trouble de la parole", "faiblesse d un cote", "asymetrie du visage", "paralysie"]):
        score += 6
    if has_any_positive_keyword(text, ["raideur nuque", "photophobie", "convulsion", "perte de connaissance"]):
        score += 5
    return score


def _cardiac_red_flag_score(text: str) -> int:
    score = 0
    cardiac_context = has_any_positive_keyword(text, CARDIAC_KEYWORDS)
    if not cardiac_context:
        return 0

    if has_any_positive_keyword(text, ["oppression", "serrement poitrine", "poitrine serree", "coeur serre"]):
        score += 3
    if has_any_positive_keyword(text, ["douleur intense", "douleur tres forte", "douleur insupportable"]):
        score += 3
    if has_any_positive_keyword(text, ["douleur bras", "bras gauche", "machoire", "dos", "irradiation"]):
        score += 4
    if has_any_positive_keyword(text, ["essoufflement", "difficulte respiratoire", "souffle court", "saturation basse"]):
        score += 4
    if has_any_positive_keyword(text, ["malaise", "sueurs", "sueurs froides", "perte de connaissance"]):
        score += 4
    return score


def classify_clinical_categories(*values: Any) -> list[str]:
    text = flatten_text(*values)
    categories: list[str] = []

    if has_any_positive_keyword(text, CARDIAC_KEYWORDS):
        categories.append("Cardiaque")

    if has_any_positive_keyword(text, RESPIRATORY_KEYWORDS):
        categories.append("Respiratoire")

    if has_any_positive_keyword(text, DIGESTIVE_KEYWORDS):
        categories.append("Digestif")

    if has_any_positive_keyword(text, NEUROLOGIC_KEYWORDS):
        categories.append("Neurologique")

    if has_any_positive_keyword(text, MUSCULOSKELETAL_KEYWORDS):
        categories.append("Musculo-articulaire")

    if has_any_positive_keyword(text, ENT_KEYWORDS):
        categories.append("ORL")

    if has_any_positive_keyword(text, URINARY_KEYWORDS):
        categories.append("Urinaire")

    if has_any_positive_keyword(text, DERMATOLOGIC_KEYWORDS):
        categories.append("Dermatologique")

    if has_any_positive_keyword(text, FEBRILE_INFECTIOUS_KEYWORDS):
        categories.append("Infectieux/Febrile")

    if has_any_positive_keyword(text, GENERAL_KEYWORDS) and not categories:
        categories.append("General")

    if "Infectieux/Febrile" in categories:
        has_alert_shift = has_any_positive_keyword(
            text,
        [
            "douleur thoracique",
            "douleurs thoraciques",
            "oppression poitrine",
            "oppression thoracique",
            "oppression dans la poitrine",
            "essoufflement",
            "difficulte respiratoire",
                "saturation basse",
                "confusion",
                "perte de connaissance",
                "trouble de la parole",
                "faiblesse d un cote",
                "raideur nuque",
            ],
        )
        if not has_alert_shift:
            categories = ["Infectieux/Febrile", *[category for category in categories if category != "Infectieux/Febrile"]]

    return categories or ["General"]


def classify_clinical_category(*values: Any) -> str:
    return ", ".join(classify_clinical_categories(*values))


def calculate_clinical_score(*values: Any) -> tuple[int, str]:
    text = flatten_text(*values)
    score = 0
    matched_keywords: set[str] = set()

    for keyword, weight in SEVERITY_WEIGHTS.items():
        normalized_keyword = normalize_text(keyword)
        if normalized_keyword in matched_keywords:
            continue
        if has_positive_keyword(text, keyword):
            score += weight
            matched_keywords.add(normalized_keyword)

    score += _temperature_score(text)
    score += _intensity_score(text)
    score += _musculoskeletal_red_flag_score(text)
    score += _duration_evolution_score(text)
    score += _transversal_red_flag_score(text)
    score += _terrain_score(text)
    score += _respiratory_distress_score(text)
    score += _neurologic_red_flag_score(text)
    score += _cardiac_red_flag_score(text)

    if re.search(r"\b(?:spo2|saturation|sat)\s*(?:<|inferieure a|moins de)?\s*(8[0-9]|9[0-2])\s*%?\b", text):
        score += 6
    elif re.search(r"\b(?:spo2|saturation|sat)\s*(?:<|inferieure a|moins de)?\s*(9[3-4])\s*%?\b", text):
        score += 4

    has_cardiac_red_flag = has_any_positive_keyword(
        text,
        [
            "douleur thoracique",
            "douleurs thoraciques",
            "oppression thoracique",
            "oppression poitrine",
            "oppression dans la poitrine",
            "douleur poitrine",
            "douleurs poitrine",
            "douleur au coeur",
            "douleurs au coeur",
            "mal au coeur",
            "maux au coeur",
            "douleur cardiaque",
            "douleurs cardiaques",
            "poitrine serree",
            "coeur serre",
            "probleme cardiaque",
        ],
    )
    has_respiratory_red_flag = has_any_positive_keyword(
        text,
        [
            "essoufflement",
            "souffle court",
            "respiration genee",
            "gene respiratoire",
            "difficulte respiratoire",
            "saturation basse",
            "spo2 basse",
        ],
    )
    has_malaise = has_positive_keyword(text, "malaise") or has_positive_keyword(text, "perte de connaissance")
    has_sueurs = has_positive_keyword(text, "sueurs") or has_positive_keyword(text, "sueurs froides")
    has_febrile_context = has_any_positive_keyword(text, ["fievre", "frissons", "temperature", "etat febrile"])
    has_neuro_red_flag = has_any_positive_keyword(
        text,
        [
            "confusion",
            "trouble conscience",
            "convulsion",
            "paralysie",
            "faiblesse d un cote",
            "trouble de la parole",
            "perte de connaissance",
        ],
    )

    if has_cardiac_red_flag and (has_respiratory_red_flag or has_malaise or has_sueurs):
        score += 4
    elif has_respiratory_red_flag and has_malaise:
        score += 3
    if has_febrile_context and has_respiratory_red_flag:
        score += 2
    if has_neuro_red_flag:
        score += 7

    if score >= 12:
        return score, "critique"
    if score >= 7:
        return score, "eleve"
    if score >= 3:
        return score, "modere"
    return score, "faible"
