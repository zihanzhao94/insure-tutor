import json
from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace
import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
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
    assert pages[13].pdf_page == 14
    assert pages[13].text.startswith("主要產品說明")
    assert pages[14].text.startswith("提供資料責任")
    assert "100" in pages[13].text and "31" in pages[13].text
    assert "1 4% 0.25%" in pages[7].text
    assert all(amount in pages[16].text for amount in ("5,000", "40,000", "400,000"))


def test_cleanup_preserves_body_numbers_tables_and_footnotes():
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({
        NameObject("/F1"): DictionaryObject({NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})})})
    content = DecodedStreamObject()
    content.set_data(b"""BT /F1 10 Tf 1 0 0 1 16 14 Tm (13) Tj ET
BT /F1 10 Tf 1 0 0 1 40 700 Tm (13) Tj ET
BT /F1 10 Tf 1 0 0 1 40 680 Tm (Policy Year 13: 4% / 0.25%) Tj ET
BT /F1 10 Tf 1 0 0 1 40 660 Tm (100 years; 31 days; USD 5,000.) Tj ET
BT /F1 10 Tf 1 0 0 1 40 20 Tm (Footnote: charges continue.) Tj ET
BT /F1 10 Tf 1 0 0 1 300 14 Tm (21) Tj ET
q 1 0 0 1 570 0 cm BT /F1 10 Tf 1 0 0 1 0 14 Tm (14) Tj ET Q
""")
    page[NameObject("/Contents")] = content
    text = ingest.extract_page_text(page)
    assert text.startswith("13\nPolicy Year 13")
    assert text.count("13") == 2
    assert "14" not in text
    assert "4% / 0.25%" in text and "USD 5,000" in text
    assert "100 years; 31 days" in text
    assert "Footnote: charges continue." in text
    assert text.endswith("21")


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
    previous = (directory / "active_collection.json").read_bytes()
    def fail(texts):
        raise ModelError("Provider unavailable")
    monkeypatch.setattr(ingest, "embed_texts", fail)
    with pytest.raises(ModelError):
        ingest.build_index([chunk(text="Changed policy conditions.")], directory, "second")
    assert (directory / "active_collection.json").read_bytes() == previous
    assert query.load_index(directory).metadata["fingerprint"] == "first"


def test_failed_chroma_write_preserves_published_collection(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, "embed_texts", lambda texts: [[1., 0.] for _ in texts])
    directory = tmp_path / "index"
    ingest.build_index([chunk()], directory, "first")
    previous = (directory / "active_collection.json").read_bytes()
    client = query.get_client(str((directory / "chroma").resolve()))
    monkeypatch.setattr(type(client), "get_max_batch_size", lambda self: 1)
    # The second batch has an invalid dimension, after the first batch succeeds.
    monkeypatch.setattr(ingest, "embed_texts", lambda texts: [[1., 0.], [1., 0., 0.]])
    with pytest.raises(Exception, match="dimension"):
        ingest.build_index([chunk(text="New clause."), chunk(page=2)], directory, "second")
    assert (directory / "active_collection.json").read_bytes() == previous
    assert query.load_index(directory).chunks[0].text == chunk().text
    assert client.count_collections() == 1


def test_chroma_persists_sources_and_cosine_search_across_processes(monkeypatch, tmp_path):
    directory = tmp_path / "index"
    monkeypatch.setattr(ingest, "embed_texts", lambda texts: [[1., 0.], [0., 1.]])
    ingest.build_index([chunk(text="Insurance premiums."), chunk(page=2, text="Policy withdrawals.")], directory)
    script = """
import sys
from pathlib import Path
from app.rag.query import load_index
index = load_index(Path(sys.argv[1]))
matches = index.collection.query(query_embeddings=[[0., 1.]], n_results=1)
assert matches['ids'][0] == ['plan:p2:c1']
assert matches['metadatas'][0][0]['pdf_page'] == 2
assert matches['documents'][0][0] == 'Policy withdrawals.'
assert abs(matches['distances'][0][0]) < 1e-6
"""
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "backend")}
    subprocess.run([sys.executable, "-c", script, str(directory)], env=environment,
                   check=True, capture_output=True, text=True, timeout=30)


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
    assert ingest.ensure_index(force=True)["state"] == "built"
    assert len(calls) == 3


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


def test_overview_interleaves_topics_deduplicates_and_obeys_context_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(query, "load_index", lambda *a: SimpleNamespace(chunks=[
        chunk(page=16, text="Benefits at a glance.")]))
    calls = []
    def retrieve(question, directory, top_k):
        calls.append(question)
        number = len(calls)
        return [chunk(page=number, text=f"Topic {number}: " + "x" * 1000),
                chunk(page=10, text="Shared conditions. " + "y" * 1500)] + [
            chunk(page=number, number=i, text="z" * 1000) for i in range(2, 15)]
    monkeypatch.setattr(query, "retrieve", retrieve)
    sources = query.retrieve_overview(tmp_path)
    assert len(calls) == 5
    assert [source.pdf_page for source in sources[:6]] == [16, 1, 2, 3, 4, 5]
    assert sum(len(source.text) for source in sources) <= 17000
    assert len({source.chunk_id for source in sources}) == len(sources)
    assert sum(source.pdf_page == 10 for source in sources) == 1


def test_overview_answer_uses_topic_evidence_and_existing_validation(monkeypatch, tmp_path):
    source = chunk(text="The cooling-off period is 21 days, according to the brochure.")
    monkeypatch.setattr(query, "retrieve_overview", lambda *a: [source])
    def forbidden(*a):
        raise AssertionError("Overview used single-topic retrieval")
    monkeypatch.setattr(query, "retrieve", forbidden)
    def generate(messages):
        assert "educational overview" in messages[0]["content"]
        return json.dumps({"status": "answered", "claims": [{"text": "Cancellation has a 21-day cooling-off period.", "evidence": [source.chunk_id]}]})
    monkeypatch.setattr(query, "generate_answer", generate)
    answer, citations, status = query.answer_question("Summarize this document", "en", tmp_path, overview=True)
    assert status == "answered" and "21" in answer and citations[0].pdf_page == 1


@pytest.mark.parametrize("repair_succeeds", [True, False])
def test_overview_repairs_bad_citations_once_then_validates_again(monkeypatch, tmp_path, repair_succeeds):
    source = chunk(text="The cooling-off period is 21 days, according to the brochure.")
    monkeypatch.setattr(query, "retrieve_overview", lambda *a: [source])
    calls = []
    def generate(messages):
        calls.append(len(messages))
        if len(calls) == 2:
            assert "failed source/number validation" in messages[-1]["content"]
        days = 21 if len(calls) == 2 and repair_succeeds else 99
        return json.dumps({"status": "answered", "claims": [
            {"text": f"The cooling-off period is {days} days.", "evidence": [source.chunk_id]}]})
    monkeypatch.setattr(query, "generate_answer", generate)
    answer, citations, status = query.answer_question("Summarize this document", "en", tmp_path, overview=True)
    assert calls == [2, 4]
    assert status == ("answered" if repair_succeeds else "insufficient_evidence")
    assert bool(citations) == repair_succeeds
