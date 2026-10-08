"""Compare page-level retrieval across chunk settings without replacing the active index."""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import get_settings  # noqa: E402
from app.model_client import embed_texts  # noqa: E402
from app.rag.ingest import load_pdf, split_pages  # noqa: E402
from app.rag.query import _terms  # noqa: E402
from app.schemas import DocumentPage  # noqa: E402


def rank(query, chunks, chunk_vectors, query_vector):
    """Use the production cosine plus normalized lexical score for all chunks."""
    semantic = chunk_vectors @ query_vector
    terms = _terms(query)
    lexical = np.array([
        sum(min(count, counts[term]) for term, count in terms.items())
        for counts in (_terms(chunk.text) for chunk in chunks)
    ], dtype=float)
    score = semantic + 0.12 * lexical / (lexical.max() or 1)
    return sorted(range(len(chunks)), key=lambda i: -score[i])


def metrics(primary_pages, relevant):
    """Score distinct pages in the five primary chunks, before context expansion."""
    hits = len(set(primary_pages) & relevant)
    return {"hit_at_5": int(hits > 0), "precision_at_5": hits / len(primary_pages) if primary_pages else 0,
            "recall_at_5": hits / len(relevant),
            "reciprocal_rank": next((1 / i for i, page in enumerate(primary_pages, 1)
                                     if page in relevant), 0)}


def evaluate(cases, pages, size, overlap, query_vectors, top_k):
    chunks = split_pages(pages, size, overlap)
    started = time.monotonic()
    vectors = np.asarray(embed_texts([chunk.text for chunk in chunks]), dtype=np.float32)
    embedding_seconds = round(time.monotonic() - started, 2)
    details = []
    for case, query_vector in zip(cases, query_vectors, strict=True):
        ordered = rank(case["query"], chunks, vectors, query_vector)
        primary = ordered[:top_k]
        primary_pages = list(dict.fromkeys(chunks[i].pdf_page for i in primary))
        relevant = set(case["relevant_pages"])
        item = {"id": case["id"], "primary_pages": primary_pages, **metrics(primary_pages, relevant)}
        # Production retrieval adds whole first-three primary pages and disclosure pages.
        selected_docs = {chunks[i].document_id for i in primary}
        selected_pages = {(chunks[i].document_id, chunks[i].pdf_page) for i in primary[:3]}
        contextual = [i for i, chunk in enumerate(chunks) if chunk.document_id in selected_docs and (
            (chunk.document_id, chunk.pdf_page) in selected_pages or
            re.search(r"\bNotes\b|\bKey Product Disclosures\b|附註|重要資料披露", chunk.text))]
        chosen, total = [], 0
        for i in primary + contextual:
            if i not in chosen and total + len(chunks[i].text) <= 17000:
                chosen.append(i)
                total += len(chunks[i].text)
        context_pages = {chunks[i].pdf_page for i in chosen}
        item.update(context_recall=len(context_pages & relevant) / len(relevant),
                    context_precision=len(context_pages & relevant) / len(context_pages) if context_pages else 0,
                    context_chars=total, context_chunks=len(chosen))
        details.append(item)
    averages = {key: round(sum(row[key] for row in details) / len(details), 3)
                for key in ("hit_at_5", "precision_at_5", "recall_at_5", "reciprocal_rank",
                            "context_recall", "context_precision", "context_chars", "context_chunks")}
    return {"chunk_size": size, "chunk_overlap": overlap, "chunk_count": len(chunks),
            "embedded_characters": sum(len(c.text) for c in chunks),
            "embedding_seconds": embedding_seconds, "average": averages, "cases": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[600, 1000, 1400])
    parser.add_argument("--overlap-ratios", nargs="+", type=float, default=[0, 0.15])
    args = parser.parse_args()
    if any(size < 100 for size in args.sizes) or any(not 0 <= ratio < 1 for ratio in args.overlap_ratios):
        parser.error("Sizes must be >= 100 and overlap ratios must be in [0, 1).")
    settings = get_settings()
    cases = [json.loads(line) for line in (ROOT / "evals/retrieval_cases.jsonl").read_text().splitlines() if line]
    raw_files = sorted((settings.data_dir / "raw").glob("*.pdf"))
    processed = settings.data_dir / "processed/pages.json"
    pages = ([page for path in raw_files for page in load_pdf(path)] if raw_files else
             [DocumentPage.model_validate(row) for row in json.loads(processed.read_text())]
             if processed.exists() else [])
    if not pages:
        parser.error("No extracted pages or source PDFs found.")
    query_vectors = np.asarray(embed_texts([case["query"] for case in cases]), dtype=np.float32)
    variants = []
    for size in args.sizes:
        for ratio in args.overlap_ratios:
            overlap = round(size * ratio)
            result = evaluate(cases, pages, size, overlap, query_vectors, settings.top_k)
            variants.append(result)
            avg = result["average"]
            print(f"{size}/{overlap}: chunks={result['chunk_count']} hit@5={avg['hit_at_5']:.3f} "
                  f"recall@5={avg['recall_at_5']:.3f} MRR={avg['reciprocal_rank']:.3f} "
                  f"context_recall={avg['context_recall']:.3f} chars={avg['context_chars']:.0f}", flush=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "embedding_model": settings.embedding_model, "top_k": settings.top_k,
              "case_count": len(cases), "variants": variants}
    output = ROOT / "evals/results" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-retrieval.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
