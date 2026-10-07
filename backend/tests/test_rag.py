import json
from pathlib import Path
import numpy as np
import pytest
from app.config import ROOT
from app.model_client import ModelError
from app.rag import ingest, query
from app.schemas import DocumentChunk, DocumentPage

def chunk(page=1, text="The minimum guaranteed account value applies after fifteen years.", number=1):
    return DocumentChunk(document_id="plan", filename="plan.pdf", pdf_page=page,
                         text=text, chunk_id=f"plan:p{page}:c{number}")

def test_supplied_pdf_loads_all_pages_and_keeps_source_metadata():
    source = next((ROOT / "data/raw").glob("*.pdf"))
    pages = ingest.load_pdf(source)
    assert len(pages) == 20
    assert pages[0].pdf_page == 1 and pages[-1].pdf_page == 20
    assert pages[0].filename == source.name
    assert "4%" in pages[7].text
    assert "2022" in pages[7].text
    assert "21" in pages[14].text


def test_split_does_not_cross_pages_and_uses_stable_unique_ids():
    pages = [DocumentPage(document_id="plan", filename="plan.pdf", pdf_page=p,
                          text=(f"Page {p} insurance clause. " * 35)) for p in (1, 2)]
    parts = ingest.split_pages(pages, 150, 30)
    assert len(parts) > 2
    assert len({c.chunk_id for c in parts}) == len(parts)
    assert all(len(c.text) <= 150 and c.filename == "plan.pdf" for c in parts)
    assert all(f"Page {c.pdf_page}" in c.text for c in parts)
    assert len({c.source_hash for c in parts}) == 2
    assert [c.chunk_id for c in parts] == [c.chunk_id for c in ingest.split_pages(pages, 150, 30)]


def test_failed_rebuild_preserves_old_index(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, "embed_texts", lambda texts: [[1., 0.] for _ in texts])
    directory = tmp_path / "index"
    ingest.build_index([chunk()], directory, "first")
    previous = (directory / "index.sqlite").read_bytes()
    def fail(texts):
        raise ModelError("Provider unavailable")
    monkeypatch.setattr(ingest, "embed_texts", fail)
    with pytest.raises(ModelError):
        ingest.build_index([chunk(text="Changed policy conditions.")], directory, "second")
    assert (directory / "index.sqlite").read_bytes() == previous


def test_unchanged_ingestion_reuses_index_without_embedding(monkeypatch, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "plan.pdf").write_bytes(b"test document")
    monkeypatch.setattr(ingest, "load_pdf", lambda path: [DocumentPage(document_id="plan", filename=path.name, pdf_page=1, text="A test insurance clause.")])
    calls = []
    monkeypatch.setattr(ingest, "embed_texts", lambda texts: calls.append(texts) or [[1., 0.] for _ in texts])
    assert ingest.ensure_index()["state"] == "built"
    assert ingest.ensure_index()["state"] == "reused"
    assert len(calls) == 1
    (raw / "plan.pdf").write_bytes(b"changed document")
    assert ingest.ensure_index()["state"] == "built"
    assert len(calls) == 2


def test_retrieval_adds_whole_notes_page_not_only_heading(monkeypatch, tmp_path):
    chunks = [chunk(), chunk(12, "Notes: additional policy conditions."),
              chunk(12, "Withdrawals reduce account value and incur applicable charges.", 2),
              chunk(13, "An unrelated page.")]
    monkeypatch.setattr(ingest, "embed_texts", lambda texts: [[1., 0.], [0., 1.], [0., 1.], [0., 1.]])
    ingest.build_index(chunks, tmp_path / "index")
    monkeypatch.setattr(query, "embed_texts", lambda texts: [[1., 0.]])
    result = query.retrieve("guaranteed account", tmp_path / "index", top_k=1)
    assert {c.chunk_id for c in result} == {c.chunk_id for c in chunks[:3]}


def test_index_rejects_embedding_configuration_change(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, "embed_texts", lambda texts: [[1., 0.]])
    ingest.build_index([chunk()], tmp_path / "index")
    monkeypatch.setenv("EMBEDDING_MODEL", "a-different-model")
    with pytest.raises(ValueError, match="Rebuild"):
        query.load_index(tmp_path / "index")


def test_answer_has_server_resolved_citations(monkeypatch, tmp_path):
    source = chunk(text="The cooling-off period is 21 days, according to the brochure.")
    monkeypatch.setattr(query, "retrieve", lambda *a: [source])
    monkeypatch.setattr(query, "generate_answer", lambda messages: json.dumps({"status": "answered", "claims": [{"text": "The cooling-off period is 21 days.", "evidence": [{"chunk_id": source.chunk_id, "quote": source.text}]}]}))
    answer, citations, status = query.answer_question("insurance cooling-off?", "en", tmp_path)
    assert status == "answered" and "[1]" in answer
    assert citations[0].pdf_page == 1
    assert citations[0].url == "/api/documents/plan.pdf#page=1"


@pytest.mark.parametrize("raw", ['not json', '{"status":"answered","claims":[]}', '{"status":"conflict","claims":[]}'])
def test_malformed_or_empty_answer_fails_closed(monkeypatch, tmp_path, raw):
    monkeypatch.setattr(query, "retrieve", lambda *a: [chunk()])
    monkeypatch.setattr(query, "generate_answer", lambda messages: raw)
    answer, citations, status = query.answer_question("insurance?", "en", tmp_path)
    assert status == "insufficient_evidence" and not citations
