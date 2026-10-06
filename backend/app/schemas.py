"""Initial API contracts. Page numbers are one-based PDF file page numbers."""

from typing import Literal

from pydantic import BaseModel, Field

Language = Literal["en", "zh-Hans", "zh-Hant"]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    language: Language = "en"
    session_id: str | None = None


class Citation(BaseModel):
    document_id: str
    filename: str
    pdf_page: int = Field(ge=1)
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    language: Language
    session_id: str
    citations: list[Citation] = Field(default_factory=list)
