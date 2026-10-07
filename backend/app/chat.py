"""Chat orchestration: history, input checks, RAG, and conversation persistence."""

import re
import uuid

from .config import get_settings
from .guardrails import check_input, classify_question, message
from .rag.query import answer_question, load_index, stream_answer_question
from .schemas import ChatRequest, ChatResponse
from .storage import load_history, save_turn


def handle_chat(request: ChatRequest) -> ChatResponse:
    """Check input, add follow-up context, call RAG, and persist the conversation turn."""
    for event in _chat_events(request, streaming=False):
        if event["event"] == "result":
            return ChatResponse.model_validate(event["data"])
    raise ValueError("The answer could not be completed.")


def stream_chat(request: ChatRequest):
    """Expose progress and supported claims while saving only the final response."""
    yield from _chat_events(request, streaming=True)


def _chat_events(request, *, streaming):
    """Share input routing, follow-up context, and persistence across both APIs."""
    settings = get_settings()
    session_id = request.session_id or uuid.uuid4().hex
    question = request.message.strip()
    if not question:
        raise ValueError("Please enter a question.")
    yield {"event": "start", "data": {"session_id": session_id,
           "language": request.language, "mode": settings.chat_mode}}
    history = load_history(session_id, settings.data_dir)
    rejection = check_input(question, has_history=bool(history), rules_only=settings.chat_mode == "extractive")
    category = "document_qa"
    if rejection is None and settings.chat_mode == "llm":
        # Check setup before spending tokens on classification.
        load_index(settings.data_dir / "index")
        yield {"event": "status", "data": {"phase": "classifying"}}
        category = classify_question(question, history)
        rejection = {"personal_advice": "out_of_scope", "out_of_scope": "out_of_scope",
                     "blocked": "blocked", "uncertain": "clarification_required"}.get(category)
    if rejection:
        answer = message(rejection, request.language)
        if rejection == "clarification_required":
            save_turn(session_id, question, answer, settings.data_dir)
        response = ChatResponse(answer=answer, language=request.language,
                                session_id=session_id, status=rejection, mode=settings.chat_mode)
        yield {"event": "result", "data": response.model_dump()}
        return
    # A deterministic baseline for short/pronominal follow-ups, avoiding an
    # extra model call. The limitation is documented and evaluated explicitly.
    query = question
    previous = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
    if previous and (len(question) < 40 or re.search(r"\b(it|that|those|then)\b|那|它|这个|這個", question, re.I)):
        earlier = [m["content"] for m in history if m["role"] == "user"][-3:-1]
        previous_answer = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), "")
        query = (f"Earlier questions: {'; '.join(earlier)}\nPrevious question: {previous}\n"
                 f"Previous answer (context only, not new evidence): {previous_answer[:1200]}\n"
                 f"Follow-up question: {question}")
    options = {"overview": True} if category == "document_overview" else {}
    yield {"event": "status", "data": {"phase": "retrieving"}}
    if streaming:
        final = None
        for event in stream_answer_question(query, request.language, settings.data_dir / "index", **options):
            if event["event"] == "result":
                final = event["data"]
            else:
                yield event
        if final is None:
            raise ValueError("The answer stream ended before completion.")
        answer, citations, status = final["answer"], final["citations"], final["status"]
    else:
        answer, citations, status = answer_question(query, request.language, settings.data_dir / "index", **options)
    response = ChatResponse(answer=answer, language=request.language, session_id=session_id,
                            citations=citations, status=status, mode=settings.chat_mode)
    save_turn(session_id, question, answer, settings.data_dir)
    yield {"event": "result", "data": response.model_dump()}
