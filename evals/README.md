# Evaluation

With the backend running, use:

```sh
.venv/bin/python evals/run_eval.py
# A smaller subset:
.venv/bin/python evals/run_eval.py --ids rate_en cooling_hant conflict_en injection_en
```

These calls use the configured embedding/chat APIs. The thirty cases cover
English, Simplified and Traditional Chinese, policy conditions, follow-ups,
missing evidence, a bilingual table discrepancy, unrelated questions, personal
buying advice and prompt injection, plus document overviews, paraphrased advice,
topic changes after history, ambiguous input and legitimate exclusion questions.
They also check a short "Introduce this policy" answer and an educational
investment-risk question, which should explain brochure-backed trade-offs
without making a personal buy/invest decision.
Page numbers are one-based PDF file pages.
`prerequisite_questions` sends multiple completed turns with the same session ID
before checking the final question.
The case's `language` sets `ui_language`, which selects the fixed notices and is
sent to generation as the answer language. An explicit language request in the
question can still override it.

The runner checks response status, at least one expected source page, selected
answer phrases/numbers, optional source-page groups and forbidden phrases, and
empty citations for refusals. Each source-page group requires at least one cited
page from that group; the two new cases use these groups to require both a
benefit/return source and a charge/risk source. A missing-detail case accepts either a refusal or an explicitly sourced
explanation that the brochure does not specify those details. These checks are
smoke checks, **not a correctness score**: a wrong answer can contain the right
words, and a correct paraphrase can fail a phrase check. Read the saved answers
and exact source excerpts under `results/` and manually assess all conditions,
amounts, interpretation, translation, and contradictions. Reports are ignored
by Git. Unit tests in `backend/tests/` use mocked models and make no API calls.

For chunk-size/overlap comparisons, run:

```sh
.venv/bin/python evals/retrieval_sweep.py
```

The nine page-labeled queries are in `retrieval_cases.jsonl`. The sweep uses the
production splitter, embeddings and cosine-plus-keyword ranking for six parameter
combinations, without changing the active Chroma index. Precision@5, Recall@5
and MRR score distinct PDF pages among the five primary chunks. Context metrics
include the application's page and disclosure expansion. The raw JSON report is
written under `results/`; see `retrieval_results.md` for the comparison. These
page labels are coarse: they do not grade which sentence within a page supports
an answer. They also do not measure answer faithfulness or language quality.

To compare keyword weights on the active index with the same queries, run:

```sh
.venv/bin/python evals/lexical_weight_sweep.py
```

It reuses the stored chunk vectors and embeds only the nine questions.
