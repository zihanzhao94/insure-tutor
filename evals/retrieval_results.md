# Retrieval parameter comparison

Measured again on 2026-10-17 after restoring the PDF's minimum-sum table,
against the supplied 20-page brochure using
`text-embedding-3-small`, `TOP_K=5`, the production page-preserving splitter,
cosine similarity and lexical weight 0.12. Nine questions have manually checked
relevant **PDF pages** in [retrieval_cases.jsonl](retrieval_cases.jsonl).
The script re-embeds each split and leaves the active Chroma index untouched.
Raw per-question output: `evals/results/20261017T014841Z-retrieval.json` (local,
ignored by Git). Reproduce with `.venv/bin/python evals/retrieval_sweep.py`.

| Chunk size / overlap (characters) | Chunks | Page Precision@5 | Page Recall@5 | MRR | Context page precision | Mean context characters |
|---|---:|---:|---:|---:|---:|---:|
| 600 / 0 | 72 | 0.519 | 1.000 | 0.944 | 0.441 | 9,097 |
| 600 / 90 | 79 | 0.537 | 1.000 | 0.944 | 0.452 | 9,229 |
| 1000 / 0 | 50 | 0.513 | 1.000 | **1.000** | 0.422 | **8,642** |
| **1000 / 150 (current)** | 54 | **0.541** | 1.000 | **1.000** | **0.457** | 9,837 |
| 1400 / 0 | 38 | 0.467 | 1.000 | 0.944 | 0.398 | 10,176 |
| 1400 / 210 | 38 | 0.500 | 1.000 | 0.889 | 0.422 | 10,067 |

Precision, recall and MRR use distinct pages among the five primary retrieved
chunks. Expanded-context precision counts distinct relevant pages divided by
all distinct pages sent to generation. Expanded-context recall was 1.000 for
all six combinations. The current 1000/150 setting has the strongest measured
page precision and tied-best MRR, so there is no evidence from this small set
to change it. The zero-overlap 1000-character setting sends about 11% less
context, but its precision is lower; a larger evaluation set is needed before
removing overlap. The 1400-character variants rank relevant pages less well.

## Keyword weight comparison

Measured again on 2026-10-17 on the active 1000/150 index with the same nine
questions. Only the keyword weight
changed in `score = cosine similarity + weight * normalized keyword score`.
Reproduce with `.venv/bin/python evals/lexical_weight_sweep.py`.

| Keyword weight | Page Precision@5 | Page Recall@5 | MRR |
|---|---:|---:|---:|
| 0 (vector only) | 0.578 | 0.944 | 1.000 |
| 0.04 | **0.596** | **1.000** | **1.000** |
| 0.05 | **0.596** | **1.000** | **1.000** |
| **0.12 (current)** | 0.541 | **1.000** | **1.000** |
| 0.2 | 0.513 | **1.000** | **1.000** |
| 0.3 | 0.502 | **1.000** | **1.000** |
| 0.5 | 0.502 | **1.000** | **1.000** |
| 1.0 | 0.448 | **1.000** | **1.000** |

Vector-only ranking misses a relevant page for "停止缴付保费会怎样？"; keyword
weighting recovers it. Precision then falls as the
weight grows, because chunks that share words without being relevant enter the
top five. The weight matters more than a tie-breaker would: the mean cosine gap
between the first and fifth chunk is about 0.08, smaller than the 0.12 maximum
bonus. Weights 0.04 and 0.05 tied for the best page precision on these nine
questions, with the same recall and MRR as 0.12.

Before the table-extraction change, on 2026-10-15, one live run of the 28 end-to-end cases per weight passed 26/28
at 0.04, 27/28 at 0.05, and 28/28 at the current 0.12. The failures at 0.04
were the English and Traditional Chinese overviews; at 0.05, the English
overview. Each returned `insufficient_evidence`. Three further English overview
tries passed 0/3 at 0.05 and 2/3 at 0.12. Model output varies, so these small
samples do not prove that 0.12 is generally better. It remains in use pending
more labeled questions and repeated answer-quality checks.

## Limits of these measurements

These are coarse page labels from one brochure. A relevant page can contain
irrelevant chunks or fail to support the exact answer. All chunk variants
reaching full Recall@5 on nine questions is a ceiling effect, not proof of
general retrieval quality. We did not compare other embedding models, `TOP_K`
or multilingual query distributions. End-to-end status/phrase/page smoke
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
