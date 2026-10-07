# Evaluation

With the backend running, use:

```sh
.venv/bin/python evals/run_eval.py
# A smaller subset:
.venv/bin/python evals/run_eval.py --ids rate_en cooling_hant conflict_en injection_en
```

These calls use the configured embedding/chat APIs. The fourteen cases cover
English, Simplified and Traditional Chinese, policy conditions, follow-ups,
missing evidence, a bilingual table discrepancy, unrelated questions, personal
buying advice and prompt injection. Page numbers are one-based PDF file pages.

The runner checks response status, at least one expected source page, selected
answer phrases/numbers, and empty citations for refusals. A missing-detail case accepts either a refusal or an explicitly sourced
explanation that the brochure does not specify those details. These checks are
smoke checks, **not a correctness score**: a wrong answer can contain the right
words, and a correct paraphrase can fail a phrase check. Read the saved answers
and exact source excerpts under `results/` and manually assess all conditions,
amounts, interpretation, translation, and contradictions. Reports are ignored
by Git. Unit tests in `backend/tests/` use mocked models and make no API calls.
