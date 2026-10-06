# InsureTutor

Take-home task for AIDF: a bilingual insurance tutor with document-grounded
answers, citations, and guardrails.

## Current status

This repository is a project scaffold. The frontend shell, backend health
endpoint, initial API contracts, and Docker configuration are present.
RAG, conversation persistence, model integration, and guardrails are TODOs.
`POST /api/chat` returns HTTP 501 until the pipeline is implemented.
The language selector is a placeholder; translated UI and answers are TODOs.

## Structure

```text
frontend/       React + TypeScript frontend (Vite)
backend/app/    FastAPI API, chat orchestration, guardrails, storage
backend/app/rag/  PDF ingestion, chunking, retrieval, generation
backend/tests/  Future module and integration tests
data/raw/      Original PDF source material
data/processed/  Extracted passages and source metadata (generated)
data/index/    Retrieval index (generated)
data/sessions/ Conversation database (generated)
evals/         Source-grounded evaluation cases and runner placeholder
docs/          Assignment and editable architecture diagram
```

## Run with Docker

```sh
cp .env.example .env
docker compose up --build
```

- Frontend: http://localhost:5173
- Backend API documentation: http://localhost:8000/docs
- Health endpoint: http://localhost:8000/api/health

No model key is needed to run the scaffold. The frontend container runs the
Vite development server for this demo scaffold. Generated data is persisted
through bind mounts; original PDFs are mounted read-only.

## Local development

Python 3.11+ and a recent Node.js version are required (Node 22.12+ or 24).

Backend, from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn app.main:app --app-dir backend --reload --port 8000
```

Frontend, in another terminal:

```sh
cd frontend
npm ci
npm run dev
```

Vite proxies `/api` requests to the backend. Docker sets `BACKEND_URL` to
the backend service; local development defaults to `http://127.0.0.1:8000`.
Local Python reads existing environment variables; to load `.env` later,
export its values or add an explicit environment-file loader.

## Architecture and next steps

See [architecture.drawio](docs/architecture.drawio) and the
[assignment](docs/TakeHomeTask-InsureTutor.md).

The backend coordinates conversations, retrieval, generation, and input/output
checks. Original PDFs, the retrieval index, and conversation storage are local.
Model API access is isolated in `model_client.py`; choose the provider and
embedding strategy when implementing ingestion.

1. Add source-grounded questions to `evals/questions.jsonl`.
2. Parse the PDF and retain file identity, language, sections, and PDF page numbers.
3. Implement section-aware chunks, including associated conditions and footnotes.
4. Build the index and verify English/Chinese retrieval against the cases.
5. Generate answers with citations resolved from source metadata.
6. Add conversation history, input/output checks, and the frontend integration.

The supplied brochure is not the full policy contract. Missing information,
non-guaranteed illustrations, and conflicting source statements must be handled
explicitly in the completed tutor.

## Secrets and generated files

Copy `.env.example` to `.env` and keep model credentials on the backend.
`.env`, generated indexes, processed text, sessions, and evaluation results are
excluded from Git. Original source PDFs are kept under `data/raw/`.
