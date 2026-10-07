"""Offline PDF extraction, recursive splitting, and persistent Chroma vector index."""

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PageObject, PdfReader

from ..config import get_settings
from ..model_client import embed_texts
from ..schemas import DocumentChunk, DocumentPage
from .query import get_client, load_index


def extract_page_text(page: PageObject) -> str:
    """Extract text while omitting standalone page numbers in the bottom corners."""
    fragments = []
    left, bottom, right, _ = (float(value) for value in page.cropbox)

    def collect(text, cm, tm, font, font_size):
        # Use page coordinates, including transforms inside PDF form objects.
        x = tm[4] * cm[0] + tm[5] * cm[2] + cm[4]
        y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
        standalone_number = re.fullmatch(r"[0-9]{1,3}", unicodedata.normalize("NFKC", text).strip())
        footer_corner = bottom <= y <= bottom + 24 and (x <= left + 30 or x >= right - 30)
        if not (standalone_number and footer_corner):
            fragments.append(text)

    page.extract_text(visitor_text=collect)
    text = unicodedata.normalize("NFKC", "".join(fragments))
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text).strip()


def load_pdf(pdf_path: Path) -> list[DocumentPage]:
    """Extract PDF text page by page, preserving filenames and one-based page numbers."""
    reader = PdfReader(pdf_path)
    if reader.is_encrypted and reader.decrypt("") == 0:
        raise ValueError(f"PDF requires a password: {pdf_path.name}")
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = extract_page_text(page)
        # File page numbers belong in metadata, not the embedded passage text.
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
    """Build a Chroma collection and publish it only after every chunk is stored."""
    if not chunks:
        raise ValueError("Cannot build an empty index.")
    settings = get_settings()
    vectors = embed_texts([chunk.text for chunk in chunks])
    if len(vectors) != len(chunks):
        raise ValueError("Embedding count does not match the chunk count.")
    index_dir.mkdir(parents=True, exist_ok=True)
    client = get_client(str((index_dir / "chroma").resolve()))
    name = f"insurance-{uuid4().hex}"
    collection = client.create_collection(
        name=name, embedding_function=None, configuration={"hnsw": {"space": "cosine"}},
        metadata={"embedding_model": settings.embedding_model,
                  "embedding_backend": settings.embedding_backend,
                  "dimension": str(len(vectors[0])), "fingerprint": fingerprint})
    temporary = index_dir / f"{name}.json"
    try:
        batch_size = client.get_max_batch_size()
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            collection.add(
                ids=[chunk.chunk_id for chunk in batch], documents=[chunk.text for chunk in batch],
                embeddings=vectors[start:start + batch_size],
                metadatas=[chunk.model_dump(exclude={"text", "chunk_id"}, exclude_none=True)
                           for chunk in batch])
        # Keep the previous collection usable if embedding or storage fails.
        temporary.write_text(json.dumps({"collection": name}), encoding="utf-8")
        temporary.replace(index_dir / "active_collection.json")
    except Exception:
        client.delete_collection(name)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def ensure_index(force: bool = False) -> dict[str, int | str]:
    """Reuse an unchanged index or rebuild it from PDFs, saving inspectable page and chunk records."""
    settings = get_settings()
    files = sorted((settings.data_dir / "raw").glob("*.pdf"))
    if not files:
        raise ValueError("Put at least one PDF in data/raw/ before starting InsureTutor.")
    digest = hashlib.sha256()
    # Fingerprint PDFs, chunk settings, and embedding configuration.
    for pdf in files:
        digest.update(pdf.name.encode())
        digest.update(pdf.read_bytes())
    digest.update(f"{settings.chunk_size}:{settings.chunk_overlap}:{settings.embedding_backend}:{settings.embedding_model}".encode())
    fingerprint = digest.hexdigest()
    index_dir = settings.data_dir / "index"
    index_path = index_dir / "active_collection.json"
    if index_path.exists() and not force:
        name = json.loads(index_path.read_text(encoding="utf-8"))["collection"]
        collection = get_client(str((index_dir / "chroma").resolve())).get_collection(
            name, embedding_function=None)
        if (collection.metadata or {}).get("fingerprint") == fingerprint:
            # Reuse unchanged inputs to avoid repeated document embedding calls.
            load_index(index_dir)
            return {"state": "reused", "documents": len(files), "chunks": collection.count()}
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
