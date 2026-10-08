# Retrieval parameter comparison

Measured on 2026-10-07 against the supplied 20-page brochure using
`text-embedding-3-small`, `TOP_K=5`, the production page-preserving splitter,
cosine similarity and lexical weight 0.12. Nine questions have manually checked
relevant **PDF pages** in [retrieval_cases.jsonl](retrieval_cases.jsonl).
The script re-embeds each split and leaves the active Chroma index untouched.
Raw per-question output: `evals/results/20261007T140318Z-retrieval.json` (local,
ignored by Git). Reproduce with `.venv/bin/python evals/retrieval_sweep.py`.

| Chunk size / overlap (characters) | Chunks | Page Precision@5 | Page Recall@5 | MRR | Context page precision | Mean context characters |
|---|---:|---:|---:|---:|---:|---:|
| 600 / 0 | 72 | 0.528 | 1.000 | 0.944 | 0.446 | 9,275 |
| 600 / 90 | 79 | 0.537 | 1.000 | 0.944 | 0.441 | 9,222 |
| 1000 / 0 | 50 | 0.513 | 1.000 | **1.000** | 0.422 | **8,699** |
| **1000 / 150 (current)** | 53 | **0.541** | 1.000 | **1.000** | **0.457** | 9,824 |
| 1400 / 0 | 37 | 0.457 | 1.000 | 0.944 | 0.393 | 10,398 |
| 1400 / 210 | 37 | 0.500 | 1.000 | 0.889 | 0.422 | 10,178 |

Precision, recall and MRR use distinct pages among the five primary retrieved
chunks. Expanded-context precision counts distinct relevant pages divided by
all distinct pages sent to generation. Expanded-context recall was 1.000 for
all six combinations. The current 1000/150 setting has the strongest measured
page precision and tied-best MRR, so there is no evidence from this small set
to change it. The zero-overlap 1000-character setting sends about 11% less
context, but its precision is lower; a larger evaluation set is needed before
removing overlap. The 1400-character variants rank relevant pages less well.

These are coarse page labels from one brochure. A relevant page can contain
irrelevant chunks or fail to support the exact answer. All variants reaching
full Recall@5 on nine questions is a ceiling effect, not proof of general
retrieval quality. We did not compare other embedding models, `TOP_K`, lexical
weights or multilingual query distributions. End-to-end status/phrase/page smoke
checks live in [run_eval.py](run_eval.py); neither score is a semantic
faithfulness or answer-relevance grade. Manual source review remains necessary,
especially for exclusions, waiting periods and financial guarantees.

The evaluation follows the retrieval/response split and metrics described in
[All-in-RAG system evaluation](https://datawhalechina.github.io/all-in-rag/#/chapter6/18_system_evaluation).
Its [tools chapter](https://datawhalechina.github.io/all-in-rag/#/chapter6/19_common_tools)
describes RAGAS, LlamaIndex evaluation and Phoenix. This project uses a small
framework-independent retrieval script because the current pipeline is direct
FastAPI/Chroma code and has page labels. RAGAS-style LLM judging can be added
after preparing reference answers and reviewing the judge on insurance claims.
