"""Compare keyword weights for primary ranking on the active index, without changing it."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import get_settings  # noqa: E402
from app.model_client import embed_texts  # noqa: E402
from app.rag.query import _terms, load_index  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", nargs="+", type=float, default=[0, 0.05, 0.12, 0.2, 0.3, 0.5, 1.0])
    args = parser.parse_args()
    settings = get_settings()
    index = load_index(settings.data_dir / "index")
    # Reuse the stored chunk vectors; only the questions are embedded here.
    records = index.collection.get(include=["embeddings", "metadatas"])
    vectors = np.asarray(records["embeddings"], dtype=np.float32)
    pages = [metadata["pdf_page"] for metadata in records["metadatas"]]
    text = {chunk.chunk_id: chunk.text for chunk in index.chunks}
    chunk_terms = [_terms(text[chunk_id]) for chunk_id in records["ids"]]
    cases = [json.loads(line) for line in (ROOT / "evals/retrieval_cases.jsonl").read_text().splitlines() if line]
    query_vectors = np.asarray(embed_texts([case["query"] for case in cases]), dtype=np.float32)

    print("weight  precision@5  recall@5  MRR")
    for weight in args.weights:
        precision, recall, reciprocal = [], [], []
        for case, query_vector in zip(cases, query_vectors, strict=True):
            # Same formula as production retrieval, with the weight as the only variable.
            terms = _terms(case["query"])
            lexical = np.array([sum(min(count, counts[term]) for term, count in terms.items())
                                for counts in chunk_terms], dtype=float)
            score = vectors @ query_vector + weight * lexical / (lexical.max() or 1)
            primary = np.argsort(-score)[:settings.top_k]
            primary_pages = list(dict.fromkeys(pages[i] for i in primary))
            relevant = set(case["relevant_pages"])
            hits = len(set(primary_pages) & relevant)
            precision.append(hits / len(primary_pages))
            recall.append(hits / len(relevant))
            reciprocal.append(next((1 / rank for rank, page in enumerate(primary_pages, 1)
                                    if page in relevant), 0))
        print(f"{weight:<7g} {np.mean(precision):.3f}        {np.mean(recall):.3f}     {np.mean(reciprocal):.3f}")


if __name__ == "__main__":
    main()
