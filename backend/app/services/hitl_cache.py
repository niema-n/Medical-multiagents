from __future__ import annotations

from functools import lru_cache


def _key_parts(*values: object) -> tuple[str, ...]:
    return tuple(str(value or "") for value in values)


def consultation_key(
    patient_case: str,
    responses: list[str] | tuple[str, ...] | None = None,
    asked_questions: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Stable cache key for deterministic per-turn clinical helpers."""
    return (
        str(patient_case or ""),
        _key_parts(*(responses or ())),
        _key_parts(*(asked_questions or ())),
    )


@lru_cache(maxsize=512)
def cached_clinical_categories(patient_case: str, responses: tuple[str, ...]) -> tuple[str, ...]:
    from backend.app.services.clinical_scoring import classify_clinical_categories

    if responses:
        return tuple(classify_clinical_categories(patient_case, list(responses)))
    return tuple(classify_clinical_categories(patient_case))


@lru_cache(maxsize=256)
def cached_contextual_symptoms(
    patient_case: str,
    responses: tuple[str, ...],
    asked_questions: tuple[str, ...],
) -> tuple[str, ...]:
    from backend.app.services.clinical_scoring import infer_contextual_symptoms

    return tuple(infer_contextual_symptoms(patient_case, list(asked_questions), list(responses)))


@lru_cache(maxsize=256)
def cached_rag_context(query: str, k: int, clinical_category: str | None) -> str:
    from backend.app.rag.retriever import _retrieve_medical_context_uncached

    return _retrieve_medical_context_uncached(query, k=k, clinical_category=clinical_category)
