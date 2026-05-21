from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("medical_multiagents.rag.index")

load_dotenv()

DOCS_DIR = Path(os.getenv("MEDICAL_DOCS_DIR", "data/medical_docs"))
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "medical_knowledge_base")


class LocalHashEmbeddings:
    """Small deterministic local embeddings for offline academic development."""

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


def _load_markdown_documents():
    try:
        from langchain_core.documents import Document
    except ImportError as exc:
        raise RuntimeError("langchain-core is required to build documents.") from exc

    documents = []
    for path in DOCS_DIR.rglob("*.md"):
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        documents.append(
            Document(
                page_content=content,
                metadata={"source": path.as_posix(), "filename": path.name},
            )
        )

    return documents


def _build_embedding_function(use_openai: bool):
    if not use_openai:
        return LocalHashEmbeddings()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to .env or run with --fallback for local embeddings."
        )

    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise RuntimeError("Install langchain-openai to use OpenAI embeddings.") from exc

    return OpenAIEmbeddings()


def _collection_name(use_openai: bool) -> str:
    mode = "openai" if use_openai else "local_hash"
    return f"{COLLECTION_NAME}_{mode}"


def index_medical_documents(use_openai: bool = True, reset: bool = False) -> dict[str, int | str]:
    """
    Index local Markdown medical documents into ChromaDB.

    Requires:
    - chromadb
    - langchain-chroma
    - langchain-openai
    - OPENAI_API_KEY
    """
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Medical docs directory not found: {DOCS_DIR}")

    try:
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise RuntimeError(
            "Install Chroma dependencies first: uv add chromadb langchain-chroma"
        ) from exc

    documents = _load_markdown_documents()
    if not documents:
        raise RuntimeError(f"No Markdown documents found in {DOCS_DIR}")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    collection_name = _collection_name(use_openai)
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=str(CHROMA_DIR),
        embedding_function=_build_embedding_function(use_openai),
    )

    if reset:
        try:
            vectorstore.delete_collection()
        except Exception:
            logger.info("rag_collection_reset_skipped", exc_info=True)
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=str(CHROMA_DIR),
            embedding_function=_build_embedding_function(use_openai),
        )

    vectorstore.add_documents(documents)

    return {
        "status": "indexed",
        "documents": len(documents),
        "collection": collection_name,
        "persist_directory": str(CHROMA_DIR),
        "embedding_mode": "openai" if use_openai else "local_hash",
        "reset": str(reset),
    }


if __name__ == "__main__":
    use_openai_embeddings = "--fallback" not in sys.argv
    reset_collection = "--reset" in sys.argv
    result = index_medical_documents(use_openai=use_openai_embeddings, reset=reset_collection)
    logger.info("medical_documents_indexed %s", result)
