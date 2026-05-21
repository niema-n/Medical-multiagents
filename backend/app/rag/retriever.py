
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import logging
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from backend.app.services.performance import perf_timer


logger = logging.getLogger("medical_multiagents.rag")

DOCS_DIR = Path(os.getenv("MEDICAL_DOCS_DIR", "data/medical_docs"))
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "medical_knowledge_base")

CATEGORY_SOURCES = {
    "cardiaque": {"cardiaque.md"},
    "dermatologique": {"dermatologique.md"},
    "digestif": {"digestif.md"},
    "general": {"general.md", "readme.md"},
    "infectieux/febrile": {"infectieux_febrile.md"},
    "musculo-articulaire": {"musculo_articulaire.md"},
    "neurologique": {"neurologique.md"},
    "orl": {"orl.md"},
    "respiratoire": {"respiratoire.md"},
    "urinaire": {"urinaire.md"},
}

CATEGORY_QUERY_TERMS = {
    "cardiaque": "douleur thoracique douleurs thoraciques douleur au coeur douleurs au coeur mal au coeur douleur poitrine oppression thoracique poitrine serree coeur serre douleur cardiaque probleme cardiaque probleme cadiaque angor infarctus insuffisance cardiaque embolie pulmonaire malaise palpitations essoufflement urgence",
    "dermatologique": "eruption cutanee boutons plaques rouges rougeurs urticaire demangeaisons rash allergie eczema infection cutanee",
    "digestif": "douleur abdominale mal au ventre douleur estomac nausees vomissements diarrhee constipation selles gastro-enterite appendicite intoxication alimentaire",
    "infectieux/febrile": "fievre temperature frissons courbatures infection fatigue paracetamol doliprane grippe infection virale signes alerte",
    "musculo-articulaire": "douleur main douleur mains douleur bras douleur jambe articulation gonflement traumatisme chute difficulte bouger entorse fracture mobilite",
    "neurologique": "confusion cephalee mal de tete perte de connaissance convulsion paralysie raideur nuque deficit neurologique trouble equilibre",
    "orl": "mal de gorge angine gorge douloureuse difficulte avaler rhinopharyngite fievre nez oreille sinus",
    "respiratoire": "toux fievre essoufflement respiration genee saturation basse asthme pneumonie bronchite dyspnee sifflement",
    "urinaire": "brulure urinaire cystite douleur urinant infection urinaire douleur lombaire sang urines envie frequente uriner pipi",
    "general": "fievre douleur fatigue malaise aggravation surveillance consultation signes de gravite",
}

CATEGORY_ALIASES = {
    "cardiaque": ["douleur thoracique", "douleurs thoraciques", "douleur au coeur", "douleurs au coeur", "mal au coeur", "douleur poitrine", "oppression thoracique", "poitrine serree", "coeur serre", "douleur cardiaque", "probleme cardiaque", "probleme cadiaque", "souci cardiaque", "souci cadiaque", "cardiaque", "cadiaque", "palpitations", "sueurs froides", "angor", "infarctus", "pericardite", "insuffisance cardiaque", "embolie pulmonaire"],
    "dermatologique": ["eruption cutanee", "boutons", "plaques rouges", "rougeurs", "urticaire", "demangeaisons", "rash", "eczema", "gonflement visage", "reaction allergique", "dermatite", "infection cutanee"],
    "digestif": ["mal au ventre", "douleur abdominale", "douleur estomac", "mal estomac", "nausee", "nausees", "vomissements", "diarrhee", "constipation", "selles", "sang dans les selles", "ventre dur", "appendicite", "gastro enterite", "gastro-enterite", "intoxication alimentaire"],
    "infectieux/febrile": ["fievre", "temperature", "frissons", "courbatures", "infection", "etat febrile", "doliprane", "paracetamol", "grippe", "syndrome infectieux", "purpura", "marbrures"],
    "musculo-articulaire": ["douleur main", "douleur bras", "douleur jambe", "douleur articulation", "articulation", "gonflement", "traumatisme", "chute", "faux mouvement", "effort inhabituel", "entorse", "foulure", "contusion", "tendinite", "fracture", "difficulte a bouger"],
    "neurologique": ["confusion", "mal de tete", "maux de tete", "cephalee", "migraine", "convulsion", "perte de connaissance", "paralysie", "faiblesse d un cote", "trouble de la parole", "trouble de l equilibre", "raideur nuque", "raideur de nuque", "photophobie", "meningite", "avc"],
    "orl": ["mal de gorge", "gorge douloureuse", "angine", "difficulte a avaler", "nez qui coule", "rhinopharyngite", "rhume", "sinusite", "otite", "oreille", "douleur oreille", "sinus"],
    "respiratoire": ["toux", "respiration genee", "gene respiratoire", "essoufflement", "souffle court", "difficulte respiratoire", "difficulte a respirer", "dyspnee", "saturation basse", "asthme", "sifflement", "pneumonie", "bronchite", "bpco", "respire mal", "mal a respirer", "je respire mal"],
    "urinaire": ["brulure urinaire", "brulures urinaires", "brulures en urinant", "douleur en urinant", "infection urinaire", "cystite", "pyelonephrite", "colique nephretique", "envie frequente d uriner", "envies frequentes", "urines troubles", "sang urines", "sang dans les urines", "douleur lombaire", "pipi"],
}


class LocalHashEmbeddings:
    """Small deterministic local embeddings matching index_documents --fallback."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = [token.lower() for token in text.split() if token.strip()]
        for token in tokens:
            index = hash(token) % self.dimensions
            vector[index] += 1.0

        norm = sum(value * value for value in vector) ** 0.5
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _build_embedding_function():
    if os.getenv("RAG_EMBEDDING_MODE", "openai").lower() == "local_hash":
        return LocalHashEmbeddings()

    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise RuntimeError("OpenAI embeddings are not installed.") from exc

    return OpenAIEmbeddings()


def _embedding_mode() -> str:
    return os.getenv("RAG_EMBEDDING_MODE", "openai").lower()


def _collection_name() -> str:
    mode = "local_hash" if _embedding_mode() == "local_hash" else "openai"
    return f"{COLLECTION_NAME}_{mode}"


def _normalise_text(value: str | None) -> str:
    text = str(value or "").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9<>=.,%/ -]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _category_tokens(clinical_category: str | None, query: str | None = None) -> set[str]:
    text = _normalise_text(clinical_category)
    tokens = {
        name
        for name in CATEGORY_SOURCES
        if name != "general" and name in text
    }
    if tokens:
        return tokens

    query_text = _normalise_text(query)
    inferred = {
        category
        for category, aliases in CATEGORY_ALIASES.items()
        if any(alias in query_text for alias in aliases)
    }
    if inferred:
        return inferred
    return {"general"} if "general" in text else set()


def _expand_query(query: str, clinical_category: str | None) -> str:
    tokens = _category_tokens(clinical_category, query)
    additions = [CATEGORY_QUERY_TERMS[token] for token in tokens if token in CATEGORY_QUERY_TERMS]
    return f"{query} {' '.join(additions)}".strip()


def _is_relevant_source(
    source: str,
    clinical_category: str | None,
    include_general: bool = True,
    query: str | None = None,
) -> bool:
    tokens = _category_tokens(clinical_category, query)
    source_name = Path(source).name.lower()
    
    if not tokens:
        return include_general and source_name in CATEGORY_SOURCES["general"]

    allowed_sources = set().union(*(CATEGORY_SOURCES.get(token, set()) for token in tokens))
    if source_name in allowed_sources:
        return True

    return include_general and not (tokens - {"general"}) and source_name in CATEGORY_SOURCES["general"]


@lru_cache(maxsize=1)
def _read_markdown_documents() -> tuple[tuple[Path, str], ...]:
    """Load local Markdown medical documents for lexical fallback retrieval."""
    if not DOCS_DIR.exists():
        return ()

    documents: list[tuple[Path, str]] = []
    for path in DOCS_DIR.rglob("*.md"):
        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError:
            logger.exception("rag_document_read_failed", extra={"path": str(path)})
            continue

        if content:
            documents.append((path, content))

    return tuple(documents)


def _normalise_chunk(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _query_terms(query: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]+", _normalise_text(query)) if len(term) > 3}


def _extract_relevant_excerpt(content: str, query_terms: set[str], max_lines: int = 7, max_chars: int = 850) -> str:
    """Keep only clinically useful lines instead of returning whole documents."""
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return ""

    selected: list[str] = []
    title = next((line for line in lines if line.startswith("#")), "")
    if title:
        selected.append(title)

    for line in lines:
        line_lower = line.lower()
        if line.startswith("#"):
            continue
        if any(term in line_lower for term in query_terms):
            selected.append(line)
        if len(selected) >= max_lines:
            break

    if len(selected) <= 1:
        selected.extend(line for line in lines[:max_lines] if line not in selected)

    excerpt = "\n".join(selected[:max_lines]).strip()
    return excerpt if len(excerpt) <= max_chars else f"{excerpt[:max_chars].rstrip()}\n[Extrait RAG resume]"


def _deduplicate_context(chunks: list[tuple[str, str]]) -> str:
    seen_content: set[str] = set()
    unique: list[str] = []

    for source, content in chunks:
        normalised = _normalise_chunk(content)
        if not normalised:
            continue
        content_fingerprint = normalised[:200]
        if content_fingerprint in seen_content:
            continue
        seen_content.add(content_fingerprint)
        unique.append(content)

    return "\n\n".join(unique)


@lru_cache(maxsize=4)
def _get_vectorstore(collection_name: str, persist_directory: str, embedding_mode: str):
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=_build_embedding_function(),
    )


def _retrieve_with_chroma(query: str, k: int, clinical_category: str | None = None) -> str:
    """Retrieve medical context from ChromaDB using OpenAI embeddings."""
    try:
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise RuntimeError("ChromaDB dependencies are not installed.") from exc

    if not CHROMA_DIR.exists():
        raise RuntimeError("ChromaDB directory does not exist. Run index_documents.py first.")

    vectorstore = _get_vectorstore(
        _collection_name(),
        str(CHROMA_DIR),
        _embedding_mode(),
    )
    docs = vectorstore.similarity_search(query, k=max(k * 3, k))
    filtered_docs = [
        doc
        for doc in docs
        if _is_relevant_source(str(doc.metadata.get("source", "unknown")), clinical_category, include_general=False, query=query)
    ]
    if _category_tokens(clinical_category, query) and not filtered_docs:
        return ""
    docs = filtered_docs[:k] if filtered_docs else docs[:k]
    terms = _query_terms(query)
    return _deduplicate_context([
        (
            str(doc.metadata.get("source", "unknown")),
            f"[RAG source={doc.metadata.get('source', 'unknown')}]\n"
            f"{_extract_relevant_excerpt(doc.page_content, terms)}",
        )
        for doc in docs
    ])


def _retrieve_lexical(query: str, k: int, clinical_category: str | None = None) -> str:
    """Fallback retriever when ChromaDB or OpenAI embeddings are unavailable."""
    query_terms = _query_terms(query)
    specialized_ranked: list[tuple[int, Path, str]] = []
    general_ranked: list[tuple[int, Path, str]] = []

    for path, content in _read_markdown_documents():
        if not _is_relevant_source(path.as_posix(), clinical_category, include_general=True, query=query):
            continue
        content_lower = _normalise_text(content)
        score = sum(1 for term in query_terms if term in content_lower)
        if score:
            source_name = path.name.lower()
            item = (score, path, _extract_relevant_excerpt(content, query_terms))
            if source_name in CATEGORY_SOURCES["general"]:
                general_ranked.append(item)
            else:
                specialized_ranked.append(item)

    ranked = specialized_ranked or general_ranked
    ranked.sort(key=lambda item: item[0], reverse=True)
    return _deduplicate_context([
        (
            path.as_posix(),
            f"[RAG fallback source={path.as_posix()} score={score}]\n{content}",
        )
        for score, path, content in ranked[:k]
    ])


def _retrieve_medical_context_uncached(query: str, k: int = 3, clinical_category: str | None = None) -> str:
    """
    Retrieve relevant local medical context for the diagnostic agent.

    ChromaDB is used when available and indexed. If ChromaDB, OpenAI embeddings,
    or network access are unavailable, the function falls back to a local lexical
    retrieval over Markdown files. This keeps FastAPI and LangGraph stable.
    """
    cleaned_query = _expand_query((query or "").strip(), clinical_category)
    if not cleaned_query:
        return ""

    if _category_tokens(clinical_category, cleaned_query):
        lexical_context = _retrieve_lexical(cleaned_query, k, clinical_category)
        if lexical_context:
            return lexical_context

    if os.getenv("RAG_USE_CHROMA", "true").lower() == "true":
        try:
            context = _retrieve_with_chroma(cleaned_query, k, clinical_category)
            if context:
                return context
        except Exception as exc:
            logger.warning("rag_chroma_unavailable_fallback_lexical: %s", exc)

    return _retrieve_lexical(cleaned_query, k, clinical_category)


def retrieve_medical_context(query: str, k: int = 3, clinical_category: str | None = None) -> str:
    from backend.app.services.hitl_cache import cached_rag_context

    with perf_timer("rag", clinical_category=clinical_category, k=k):
        return cached_rag_context((query or "").strip(), k, clinical_category)


def build_chroma_retriever() -> object:
    """
    Build a Chroma retriever for advanced integrations.

    Most application code should call retrieve_medical_context(), which is safer
    because it includes fallback behavior.
    """
    try:
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise RuntimeError("Install langchain-chroma and chromadb to enable ChromaDB.") from exc

    return Chroma(
        collection_name=_collection_name(),
        persist_directory=str(CHROMA_DIR),
        embedding_function=_build_embedding_function(),
    ).as_retriever(search_kwargs={"k": 3})
