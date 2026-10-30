"""Case interrogation: answer from sealed files only, with verified citations."""

from __future__ import annotations

import logging
from hashlib import sha256

from temoin.config import Settings, load_settings
from temoin.model import ChatClient, ModelError, complete_json, default_client
from temoin.prompts import QA_PROMPT_VERSION, build_qa_messages
from temoin.schemas import CaseAnswer, Comparison, Report

logger = logging.getLogger(__name__)

IDK_ANSWER = "Je ne sais pas d'après ce dossier."


def build_qa_cache_key(
    question: str, sale: Report, delivery: Report, comparison: Comparison
) -> str:
    """Derive a cache key from the question, the files and the QA version."""
    raw = (
        f"{question.strip()}|{sale.identity_json()}|{delivery.identity_json()}"
        f"|{comparison.model_dump_json(exclude={'created_at_utc'})}|{QA_PROMPT_VERSION}"
    )
    return sha256(raw.encode()).hexdigest()


def _check_citations(answer: CaseAnswer, sale: Report, delivery: Report) -> None:
    """Validate every citation id; ValueError feeds the model retry."""
    if answer.confident and not answer.citations:
        raise ValueError("a confident answer needs at least one citation")
    known = {
        "sale": {defect.id for defect in sale.defects} | {"challenge"},
        "delivery": {defect.id for defect in delivery.defects} | {"challenge"},
    }
    for citation in answer.citations:
        if citation == "verdict":
            continue
        side, separator, ref = citation.partition(":")
        if not separator or side not in known or ref not in known[side]:
            raise ValueError(
                f"unknown citation {citation!r}: use sale:<id>, delivery:<id> or verdict"
            )


def ask_question(
    question: str,
    *,
    sale: Report,
    delivery: Report,
    comparison: Comparison,
    settings: Settings | None = None,
    client: ChatClient | None = None,
) -> CaseAnswer:
    """Answer from sealed files only; honest fallback when evidence is missing.

    The model proposes an answer with citations, the code disposes: unknown
    ids are sent back once, then any failure degrades to an explicit
    "I don't know" instead of an invented answer.
    """
    if not question.strip():
        raise ValueError("question must not be empty")
    active_settings = settings or load_settings()
    active_client = client or default_client(
        api_key=active_settings.baseten_api_key,
        base_url=active_settings.model_base_url,
        timeout_seconds=active_settings.model_timeout_seconds,
    )

    def _check(checked: CaseAnswer) -> None:
        """Check citations against both constats."""
        _check_citations(checked, sale, delivery)

    try:
        answer = complete_json(
            build_qa_messages(
                sale=sale, delivery=delivery, comparison=comparison, question=question.strip()
            ),
            CaseAnswer,
            client=active_client,
            model_name=active_settings.model_name,
            cache_key=build_qa_cache_key(question, sale, delivery, comparison),
            cache_dir=active_settings.cache_dir,
            max_attempts=active_settings.model_max_attempts,
            retry_wait_seconds=active_settings.model_retry_wait_seconds,
            json_retries=active_settings.model_json_retries,
            extra_check=_check,
        )
    except ModelError as error:
        logger.warning("Case interrogation failed, honest fallback: %s", error)
        return CaseAnswer(answer=IDK_ANSWER, citations=[], confident=False)
    if not answer.confident:
        return answer.model_copy(update={"answer": IDK_ANSWER, "citations": []})
    return answer
