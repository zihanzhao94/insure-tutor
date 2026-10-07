"""Offline PDF extraction, recursive splitting, and persistent SQLite vector index."""

import argparse
import hashlib
import json
import re
import sqlite3
import unicodedata
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from ..config import get_settings
from ..model_client import embed_texts
from ..schemas import DocumentChunk, DocumentPage

# Bump when extraction, splitting logic, or index format changes require a rebuild.
INDEX_VERSION = 1


def load_pdf(pdf_path: Path) -> list[DocumentPage]:
    """Extract PDF text page by page, preserving filenames and one-based page numbers."""
    reader = PdfReader(pdf_path)
    if reader.is_encrypted and reader.decrypt("") == 0:
        raise ValueError(f"PDF requires a password: {pdf_path.name}")
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = unicodedata.normalize("NFKC", page.extract_text() or "")
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text).strip()
        # save each page with its document ID, filename, page number, and text content
        pages.append(DocumentPage(document_id=pdf_path.stem, filename=pdf_path.name,
                                  pdf_page=number, text=text))
    if not any(page.text for page in pages):
        raise ValueError(f"No selectable text in {pdf_path.name}; OCR is required.")
    return pages


def split_pages(pages: list[DocumentPage], chunk_size: int | None = None,
                chunk_overlap: int | None = None) -> list[DocumentChunk]:
    """Split pages into character-based chunks with source metadata; chunks never cross pages."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size if chunk_size is not None else settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        length_function=len, separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    chunks = []
    for page in pages:
        section = next((line.strip() for line in page.text.splitlines() if len(line.strip()) > 3), None)
        source_hash = hashlib.sha256(page.text.encode()).hexdigest()
        for number, text in enumerate(splitter.split_text(page.text), start=1):
            chunks.append(DocumentChunk(
                **page.model_dump(exclude={"text"}), text=text,
                chunk_id=f"{page.document_id}:p{page.pdf_page}:c{number}",
                section=section, source_hash=source_hash))
    return chunks


def build_index(chunks: list[DocumentChunk], index_dir: Path, fingerprint: str = "") -> None:
    """Embed chunks and atomically save text, vectors, and metadata to SQLite."""
    if not chunks:
        raise ValueError("Cannot build an empty index.")
    settings = get_settings()
    vectors = embed_texts([chunk.text for chunk in chunks])
    index_dir.mkdir(parents=True, exist_ok=True)
    temporary = index_dir / "index.building.sqlite"
    temporary.unlink(missing_ok=True)
    try:
        with sqlite3.connect(temporary) as db:
            db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, record TEXT NOT NULL, vector TEXT NOT NULL)")
            db.executemany("INSERT INTO chunks VALUES (?, ?, ?)", [
                (chunk.chunk_id, chunk.model_dump_json(), json.dumps(vector))
                for chunk, vector in zip(chunks, vectors, strict=True)])
            metadata = {"embedding_model": settings.embedding_model, "embedding_backend": settings.embedding_backend,
                        "dimension": str(len(vectors[0])), "fingerprint": fingerprint, "version": str(INDEX_VERSION)}
            db.executemany("INSERT INTO metadata VALUES (?, ?)", metadata.items())
        temporary.replace(index_dir / "index.sqlite")
    finally:
        temporary.unlink(missing_ok=True)


def ensure_index(force: bool = False) -> dict[str, int | str]:
    """Reuse an unchanged index or rebuild it from PDFs, saving inspectable page and chunk records."""
    settings = get_settings()
    files = sorted((settings.data_dir / "raw").glob("*.pdf"))
    if not files:
        raise ValueError("Put at least one PDF in data/raw/ before starting InsureTutor.")
    digest = hashlib.sha256()
    # Fingerprint PDFs, chunk settings, embedding configuration, and index version.
    for pdf in files:
        digest.update(pdf.name.encode())
        digest.update(pdf.read_bytes())
    digest.update(f"{INDEX_VERSION}:{settings.chunk_size}:{settings.chunk_overlap}:{settings.embedding_backend}:{settings.embedding_model}".encode())
    fingerprint = digest.hexdigest()
    index_dir = settings.data_dir / "index"
    index_path = index_dir / "index.sqlite"
    if index_path.exists() and not force:
        with sqlite3.connect(index_path) as db:
            metadata = dict(db.execute("SELECT key, value FROM metadata"))
            if metadata.get("fingerprint") == fingerprint:
                # Reuse unchanged inputs to avoid repeated document embedding calls.
                return {"state": "reused", "documents": len(files),
                        "chunks": db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]}
    pages = [page for pdf in files for page in load_pdf(pdf)]
    chunks = split_pages(pages)
    processed = settings.data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    for name, records in [("pages", pages), ("chunks", chunks)]:
        (processed / f"{name}.json").write_text(
            json.dumps([record.model_dump() for record in records], ensure_ascii=False, indent=2), encoding="utf-8")
    build_index(chunks, index_dir, fingerprint)
    return {"state": "built", "documents": len(files), "pages": len(pages), "chunks": len(chunks)}


def main() -> None:
    """Parse CLI options, prepare the index, and print the result; --force rebuilds an existing index."""
    parser = argparse.ArgumentParser(description="Build the insurance PDF index.")
    parser.add_argument("--force", action="store_true")
    print(json.dumps(ensure_index(force=parser.parse_args().force), ensure_ascii=False))


if __name__ == "__main__":
    main()
