"""Coordinate input checks, history, retrieval, generation, and output checks."""

from .schemas import ChatRequest, ChatResponse


def handle_chat(request: ChatRequest) -> ChatResponse:
    # TODO: Load history and clarify follow-up questions.
    # TODO: Apply input guardrails, retrieve evidence, and generate an answer.
    # TODO: Validate the answer and citations, then persist the conversation.
    raise NotImplementedError("Implement the chat pipeline.")
