"""Initial API contracts. Page numbers are one-based PDF file page numbers."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Language = Literal["en", "zh-Hans", "zh-Hant"]
QuestionCategory = Literal["document_qa", "document_overview", "personal_advice", "out_of_scope", "blocked", "uncertain"]
MAX_CLAIMS = 6


class QuestionIntent(BaseModel):
    """A routing label, not an answer or proof that the PDF contains evidence."""

    model_config = ConfigDict(extra="forbid")
    category: QuestionCategory


INTENT_JSON_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["category"],
    "properties": {"category": {"type": "string", "enum": [
        "document_qa", "document_overview", "personal_advice", "out_of_scope", "blocked", "uncertain"
    ]}},
}


class DocumentPage(BaseModel):
    """Extracted page record; serialize these to data/processed/pages.json."""

    document_id: str
    filename: str
    pdf_page: int = Field(ge=1)
    text: str


class DocumentChunk(DocumentPage):
    """Retrievable passage carrying the original page's source metadata."""

    chunk_id: str
    section: str | None = None
    source_hash: str = ""


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    # Used only for predefined system notices, never to direct generated text.
    ui_language: Language = "en"
    session_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")


class Citation(BaseModel):
    document_id: str
    filename: str
    pdf_page: int = Field(ge=1)
    excerpt: str
    chunk_id: str = ""
    url: str = ""


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    citations: list[Citation] = Field(default_factory=list)
    status: Literal["answered", "insufficient_evidence", "out_of_scope", "blocked", "conflict", "clarification_required"] = "answered"
    mode: Literal["llm", "extractive"] = "llm"


class EvidenceRef(BaseModel):
    """Pointer to a retrieved PDF passage selected to support one answer statement."""

    chunk_id: str
    # The server supplies source text when the model selects only a chunk ID.
    quote: str = ""


class GroundedClaim(BaseModel):
    """One factual statement in the answer, not an insurance benefit claim (理赔)."""

    text: str = Field(min_length=1, max_length=1600)
    evidence: list[EvidenceRef] = Field(min_length=1, max_length=4)

    @field_validator("evidence", mode="before")
    @classmethod
    def accept_source_ids(cls, value):
        """Convert string source IDs into citation objects for consistent validation."""
        # Both representations select the same server-owned source records.
        if isinstance(value, list):
            return [{"chunk_id": item} if isinstance(item, str) else item for item in value]
        return value


class GeneratedAnswer(BaseModel):
    """Model output: a status plus at most six evidence-backed answer statements."""

    status: Literal["answered", "insufficient_evidence", "conflict"]
    claims: list[GroundedClaim] = Field(default_factory=list, max_length=MAX_CLAIMS)


# Constrain API output as well as validating it after generation. Quotes and
# citation metadata are deliberately absent: the backend owns those fields.
ANSWER_JSON_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["status", "claims"],
    "properties": {
        "status": {"type": "string", "enum": ["answered", "insufficient_evidence", "conflict"]},
        "claims": {
            "type": "array", "maxItems": MAX_CLAIMS,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["text", "evidence"],
                "properties": {
                    "text": {"type": "string"},
                    "evidence": {
                        "type": "array", "minItems": 1, "maxItems": 4,
                        "items": {"type": "object", "additionalProperties": False,
                                  "required": ["chunk_id"],
                                  "properties": {"chunk_id": {"type": "string"}}},
                    },
                },
            },
        },
    },
}
