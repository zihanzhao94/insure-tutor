"""FastAPI endpoints and startup indexing."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from .chat import handle_chat
from .config import get_settings
from .model_client import ModelError
from .rag.ingest import ensure_index
from .rag.query import load_index
from .schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepare or load the index at API startup and record common setup errors for health checks."""
    app.state.index_error = None
    try:
        if get_settings().auto_ingest:
            app.state.index_summary = await run_in_threadpool(ensure_index)
        else:
            load_index(get_settings().data_dir / "index")
    except (ValueError, FileNotFoundError, ModelError) as exc:
        # Keep the UI and health endpoint usable when setup needs attention.
        app.state.index_error = str(exc)
        logger.warning("Index initialization needs attention: %s", exc)
    yield


app = FastAPI(title="InsureTutor", version="0.2.0", lifespan=lifespan)


@app.get("/api/health")
def health(response: Response):
    """Return index readiness, source documents, and model configuration status."""
    settings = get_settings()
    ready, documents, count = False, [], 0
    error = getattr(app.state, "index_error", None)
    try:
        index = load_index(settings.data_dir / "index")
        ready, count = True, len(index.chunks)
        documents = sorted({c.filename for c in index.chunks})
    except (ValueError, FileNotFoundError):
        pass
    configured = bool(settings.model_api_key) and not (
        settings.model_provider == "openai" and settings.model_api_key.startswith("sk-ant-"))
    return {"status": "ok" if ready else "setup_required", "index_ready": ready,
            "model_configured": configured, "mode": settings.chat_mode, "provider": settings.model_provider,
            "documents": documents, "chunks": count, "detail": error}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Handle chat requests and translate common failures into HTTP errors."""
    try:
        return handle_chat(request)
    except ModelError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/documents/{filename}")
def document(filename: str):
    # Serve only existing source PDFs, never arbitrary filesystem paths.
    """Serve existing source PDFs while preventing access to arbitrary files."""
    directory = (get_settings().data_dir / "raw").resolve()
    path = (directory / filename).resolve()
    if path.parent != directory or path.suffix.lower() != ".pdf" or not path.is_file():
        raise HTTPException(status_code=404, detail="Document not found.")
    return FileResponse(path, media_type="application/pdf")
