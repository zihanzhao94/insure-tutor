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

TABLE_START = "[TABLE]"
TABLE_END = "[/TABLE]"
INGEST_FORMAT_VERSION = "table-layout-v1"


def _minimum_sum_insured_table(fragments: list[tuple[str, float, float]]) -> str | None:
    """Restore this brochure's minimum-sum table using the PDF's row/column positions.

    pypdf's reading order separates the age headers from amounts. Cell contents
    still come from the PDF; this only attaches their row and column labels.
    """
    def anchor(pattern: str) -> tuple[float, float] | None:
        return next(((x, y) for text, x, y in fragments
                     if re.fullmatch(pattern, text.strip(), re.I)), None)

    minimum = anchor("Minimum Sum Insured")
    maximum = anchor("Maximum Sum Insured")
    younger = anchor(r"<\s*Age\s*45")
    older = anchor(r"≥\s*Age\s*45")
    fp80 = anchor("FP80/100/130")
    fp180 = anchor("FP180/280")
    hong_kong = anchor("Hong Kong Policies")
    macau = anchor("Macau Policies")
    others = anchor("Others")
    if not all((minimum, maximum, younger, older, fp80, fp180, hong_kong, macau, others)):
        return None
    assert minimum and maximum and younger and older and fp80 and fp180 and hong_kong and macau and others
    left_x, right_x = younger[0], older[0]
    if not (minimum[1] > hong_kong[1] > macau[1] > fp180[1] > others[1] > maximum[1]
            and left_x < right_x):
        return None

    def cell(top: float, bottom: float, left: float, right: float = float("inf")) -> str:
        labels = {"香港保單", "Hong Kong Policies", "澳門保單", "Macau Policies"}
        values = [text.strip().replace("\t", " ") for text, x, y in
                  sorted(fragments, key=lambda item: (-item[2], item[1]))
                  if bottom < y <= top and left - 1 <= x < right - 1
                  and text.strip() and text.strip() not in labels]
        joined = " ".join(values)
        # The PDF abbreviates equal HKD/MOP amounts as "HK$ / MOP120,000".
        return re.sub(r"(HK\$|MOP)\s*/\s*(HK\$|MOP)\s*(\d[\d,]*)",
                      lambda match: f"{match[1]}{match[3]} / {match[2]}{match[3]}", joined)

    # The right-hand FP80/100/130 cell spans the two territories; the Others
    # cell spans both age columns. Repeat each spanning value beside its labels.
    fp80_older = cell(minimum[1] + 1, fp180[1] + 2, right_x)
    rows = [
        ("FP80/100/130, Hong Kong Policies",
         cell(hong_kong[1] + 1, macau[1] + 1, left_x, right_x), fp80_older),
        ("FP80/100/130, Macau Policies",
         cell(macau[1] + 1, fp180[1] + 2, left_x, right_x), fp80_older),
        ("FP180/280", cell(fp180[1] + 1, others[1] + 1, left_x, right_x),
         cell(fp180[1] + 1, others[1] + 1, right_x)),
        ("Others", cell(others[1] + 1, maximum[1] + 1, left_x), None),
    ]
    if any(not young or (old is not None and not old) for _, young, old in rows):
        return None
    lines = ["Minimum Sum Insured (PDF table)", "Plan / territory | Issue age <45 | Issue age ≥45"]
    for label, young, old in rows:
        lines.append(f"{label} | {young} | {old or 'same minimum for both age groups'}")
    return "\n".join(lines)


def _split_table(table: str, chunk_size: int) -> list[str]:
    """Keep each table row intact and repeat its headings if rows need multiple chunks."""
    heading, columns, *rows = table.splitlines()
    prefix = f"{heading}\n{columns}"
    result, current = [], prefix
    for row in rows:
        if len(current) + 1 + len(row) > chunk_size and current != prefix:
            result.append(current)
            current = prefix
        current += "\n" + row
    result.append(current)
    return result


def extract_page_text(page: PageObject) -> str:
    """Extract text while omitting standalone page numbers in the bottom corners."""
    fragments: list[tuple[str, float, float]] = []
    left, bottom, right, _ = (float(value) for value in page.cropbox)

    # pypdf calls this once per text fragment; keep all but footer page numbers.
    def collect(text, cm, tm, font, font_size):
        # Use page coordinates, including transforms inside PDF form objects.
        x = tm[4] * cm[0] + tm[5] * cm[2] + cm[4]
        y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
        standalone_number = re.fullmatch(r"[0-9]{1,3}", unicodedata.normalize("NFKC", text).strip())
        footer_corner = bottom <= y <= bottom + 24 and (x <= left + 30 or x >= right - 30)
        if not (standalone_number and footer_corner):
            fragments.append((text, x, y))

    page.extract_text(visitor_text=collect)
    text = unicodedata.normalize("NFKC", "".join(item[0] for item in fragments))
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text).strip()
    table = _minimum_sum_insured_table(fragments)
    if table:
        # pypdf appends these detached table cells after the page's other text.
        tail = re.search(r"\n<\s*Age\s*45\s*歲\s*≥\s*Age\s*45", text)
        heading = "最低保障額\nMinimum Sum Insured"
        if tail and heading in text:
            text = text[:tail.start()].replace(heading, f"{TABLE_START}\n{table}\n{TABLE_END}", 1)
    return text


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
    size = chunk_size if chunk_size is not None else settings.chunk_size
    # Try to cut at paragraph breaks first, then lines, then Chinese sentence
    # ends, then spaces. Lengths are counted in characters, not tokens.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        length_function=len, separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    chunks = []
    # Outer loop: one page at a time, so no chunk can span two pages.
    for page in pages:
        # Use the page's first real line as a rough section title.
        section = next((line.strip() for line in page.text.splitlines() if len(line.strip()) > 3), None)
        source_hash = hashlib.sha256(page.text.encode()).hexdigest()
        # Inner loop: one page usually yields several chunks. Each chunk copies
        # the page's filename and page number; its ID looks like "doc:p8:c2".
        pieces = re.split(r"(?s)(\[TABLE\].*?\[/TABLE\])", page.text)
        texts = []
        for piece in pieces:
            if piece.startswith(TABLE_START) and piece.endswith(TABLE_END):
                texts.extend(_split_table(piece[len(TABLE_START):-len(TABLE_END)].strip(), size))
            elif piece.strip():
                texts.extend(splitter.split_text(piece.strip()))
        for number, text in enumerate(texts, start=1):
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
    # Only the chunk text is embedded; page numbers are stored beside the vector.
    vectors = embed_texts([chunk.text for chunk in chunks])
    if len(vectors) != len(chunks):
        raise ValueError("Embedding count does not match the chunk count.")
    index_dir.mkdir(parents=True, exist_ok=True)
    client = get_client(str((index_dir / "chroma").resolve()))
    # Build into a new collection; the current one stays active until this succeeds.
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
            # One record per chunk: ID, original text, vector, and metadata
            # (filename, page number, section) used later to build citations.
            collection.add(
                ids=[chunk.chunk_id for chunk in batch], documents=[chunk.text for chunk in batch],
                embeddings=vectors[start:start + batch_size],
                metadatas=[chunk.model_dump(exclude={"text", "chunk_id"}, exclude_none=True)
                           for chunk in batch])
        # Keep the previous collection usable if embedding or storage fails:
        # the active pointer is switched last, in one atomic file replace.
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
    digest.update(f"{INGEST_FORMAT_VERSION}:{settings.chunk_size}:{settings.chunk_overlap}:{settings.embedding_backend}:{settings.embedding_model}".encode())
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
    # Rebuild: extract pages -> split into chunks -> save readable copies -> embed and store.
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
