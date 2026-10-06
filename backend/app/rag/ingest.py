"""Offline ingestion: raw PDF -> extracted text -> chunks -> retrieval index."""

# TODO: Extract text and tables, preserving document IDs and PDF page numbers.
# TODO: Save intermediate chunks under data/processed/.
# TODO: Embed and index chunks under data/index/; avoid re-indexing unchanged files.
