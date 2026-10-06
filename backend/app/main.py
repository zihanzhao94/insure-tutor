"""HTTP entry point. Only health and the API contract are implemented."""

from fastapi import FastAPI, HTTPException

from .chat import handle_chat
from .schemas import ChatRequest, ChatResponse

app = FastAPI(title="InsureTutor", version="0.1.0")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "stage": "scaffold"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        return handle_chat(request)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail="Chat and RAG are not implemented yet.") from exc
