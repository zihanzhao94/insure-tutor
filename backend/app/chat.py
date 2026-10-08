"""Chat orchestration: history, input checks, RAG, and conversation persistence."""

import re
import uuid

from .config import get_settings
from .guardrails import check_input, classify_question, localize, message
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
    """Run the same conversation flow for JSON and SSE callers.

    Load recent turns to understand follow-ups, route the question, retrieve fresh
    PDF evidence, then save only the completed answer under this session ID.
    """
    settings = get_settings()
    session_id = request.session_id or uuid.uuid4().hex
    question = request.message.strip()
    if not question:
        raise ValueError("Please enter a question.")
    yield {"event": "start", "data": {"session_id": session_id,
           "mode": settings.chat_mode}}
    history = load_history(session_id, settings.data_dir, limit=2 * settings.memory_turns)
    rejection = check_input(question, has_history=bool(history), rules_only=settings.chat_mode == "extractive")
    category = "document_qa"
    resolved = _short_topic_request(question)
    implicit = _implicit_brochure_request(question)
    key_points = _is_brief_overview_request(question)
    effective_question = resolved[0] if resolved else question
    if rejection is None and settings.chat_mode == "llm":
        # Check setup before spending tokens on classification.
        load_index(settings.data_dir / "index")
        yield {"event": "status", "data": {"phase": "classifying"}}
        if resolved:
            category = resolved[1]
        elif implicit:
            # In this document tutor, this broad phrasing refers to the supplied
            # brochure; it asks for neither a market comparison nor personal advice.
            category = "document_overview"
        else:
            category = classify_question(question, history)
            if category == "uncertain" and key_points:
                category = "document_overview"
        rejection = {"personal_advice": "out_of_scope", "out_of_scope": "out_of_scope",
                     "blocked": "blocked", "uncertain": "clarification_required"}.get(category)
    if rejection:
        answer = message(rejection, request.ui_language)
        if rejection == "clarification_required":
            save_turn(session_id, question, answer, settings.data_dir)
        response = ChatResponse(answer=answer,
                                session_id=session_id, status=rejection, mode=settings.chat_mode)
        yield {"event": "result", "data": response.model_dump()}
        return
    # Search may include previous user topics; answer generation still receives the
    # latest question separately so it does not answer an earlier turn again.
    if resolved:
        query = effective_question
    elif implicit:
        query = implicit[0]
    else:
        query = _retrieval_query(question, history)
    if implicit:
        options = {"focus": implicit[1]}
    elif key_points:
        options = {"key_points": True}
    else:
        options = {"overview": True} if category == "document_overview" else {}
    if history:
        options.update(current_question=effective_question, conversation=history)
    elif implicit:
        # Keep the user's original wording/script for generation; only search is scoped.
        options["current_question"] = question
    yield {"event": "status", "data": {"phase": "retrieving"}}
    if streaming:
        final = None
        for event in stream_answer_question(query, request.ui_language, settings.data_dir / "index", **options):
            if event["event"] == "result":
                final = event["data"]
            else:
                yield event
        if final is None:
            raise ValueError("The answer stream ended before completion.")
        answer, citations, status = final["answer"], final["citations"], final["status"]
    else:
        answer, citations, status = answer_question(query, request.ui_language, settings.data_dir / "index", **options)
    response = ChatResponse(answer=answer, session_id=session_id,
                            citations=citations, status=status, mode=settings.chat_mode)
    # Streaming previews are never conversation memory; only this final result is.
    save_turn(session_id, question, answer, settings.data_dir)
    yield {"event": "result", "data": response.model_dump()}


def _retrieval_query(question: str, history: list[dict[str, str]]) -> str:
    """Add recent user topics to a likely follow-up; old answers never become search evidence."""
    previous = [item["content"] for item in history if item["role"] == "user"]
    if not previous or not (len(question) < 40 or re.search(
            r"\b(it|that|those|then|this|they|them|same|previous|above)\b|那|它|这个|這個|上述|刚才|剛才|前面|还有|還有",
            question, re.I)):
        return question
    return "Recent questions: " + "; ".join(previous[-3:])[:600] + "\nCurrent question: " + question


def _is_brief_overview_request(question: str) -> bool:
    """Recognize a short request for key points, including a common typo."""
    return bool(re.fullmatch(
        r"(?:what(?:'s| is)|tell me)\s+(?:the\s+)?(?:most\s+import(?:ant)?|main|key)\s+"
        r"(?:part|points?|terms?|features?)(?:\s+of\s+(?:this|the)\s+(?:brochure|document|plan))?[?.!]?",
        question.strip(), re.I))


def _short_topic_request(question: str) -> tuple[str, str] | None:
    """Turn a one-word brochure topic, including a common typo, into a complete question."""
    topic = re.sub(r"[\s?.!。？！，,]+", "", question).casefold()
    benefit_question = ("Which types of benefits does the supplied brochure describe? "
                        "Give a brief high-level list of benefit categories; omit rates, amounts and detailed conditions.")
    english = {
        "benefit": benefit_question,
        "benefits": benefit_question,
        "benifit": benefit_question,
        "benifits": benefit_question,
        "premium": "How do premiums work in the supplied insurance plan?",
        "premiums": "How do premiums work in the supplied insurance plan?",
        "charge": "What charges apply to the supplied insurance plan?",
        "charges": "What charges apply to the supplied insurance plan?",
        "summary": "Summarize the main terms of the supplied brochure.",
    }
    chinese = {
        "保障": "请简要列出这份保险宣传册的保障类别，不展开利率、金额或详细条件。",
        "保费": "这份保险宣传册的保费规则是什么？",
        "費用": "这份保险宣传册有哪些费用？",
        "费用": "这份保险宣传册有哪些费用？",
        "总结": "总结这份保险宣传册的主要条款。",
        "總結": "总结这份保险宣传册的主要条款。",
    }
    resolved = english.get(topic) or chinese.get(topic)
    if resolved:
        return resolved, "document_overview" if topic in {"summary", "总结", "總結"} else "document_qa"
    return None


def _implicit_brochure_request(question: str) -> tuple[str, str] | None:
    """Scope broad "what should I watch for" questions without weakening advice or misuse rules."""
    normalized = re.sub(r"[\s?？。！!]+$", "", localize(question, "zh-Hans").strip())
    if re.fullmatch(
            r"(?:(?:这份|这个|该)(?:文件|文档|宣传册|保单|保险计划))?"
            r"(?:都)?有(?:哪|那)些(?:条件|条款|事项|风险)?(?:值得|需要|要)?"
            r"(?:注意|留意)(?:的)?(?:呢|吗)?", normalized):
        return ("这份保险宣传册有哪些重要限制、费用及风险值得留意？", "cautions")
    return None
