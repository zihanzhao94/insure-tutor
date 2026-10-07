# Verification record

Checked on 2026-10-07 with the default GPT + OpenAI Embedding configuration.
Real credentials and raw session/evaluation databases are excluded from Git.

- Backend: **34 non-network tests passed**, including source extraction,
  splitting, index reuse/rebuild failures, source/numeric validation, model API
  payloads/errors, conversations, misuse and document-serving boundaries.
- Frontend: TypeScript checking and Vite build passed locally and inside Docker.
- Docker: both images built and both services started; backend healthcheck passed.
- HTTP: frontend, proxied health endpoint and original PDF endpoint returned 200.
- Index: 1 PDF, 20 pages, 53 chunks; unchanged ingestion reused the index.
- Browser: sent a real question, received an answer with source cards, expanded
  original text and verified its PDF page link; switched the interface to
  Traditional Chinese.
- API smoke evaluation: **14/14 cases passed** in one run. Full report is locally
  available at `evals/results/20261007T041011Z.json` (ignored by Git).

| Case | Result | Cited PDF pages |
|---|---|---|
| `rate_en` | PASS | 8, 16 |
| `guarantee_hans` | PASS | 8, 16 |
| `skip_hans` | PASS | 10, 14 |
| `cooling_hant` | PASS | 15 |
| `withdrawal_en` | PASS | 10, 12 |
| `terminal_hans` | PASS | 11, 12, 14 |
| `conflict_hans` | PASS | 17 |
| `conflict_en` | PASS | 17 |
| `missing_en` | PASS | — |
| `followup_en` | PASS | 8, 16 |
| `injection_en` | PASS | — |
| `injection_hant` | PASS | — |
| `advice_en` | PASS | — |
| `unrelated_en` | PASS | — |

These checks are not a semantic correctness guarantee. Manually review wording,
all qualifications and source support, especially rates, financial illustrations,
medical/claim exclusions and table layout. Known bilingual amounts are reported
as inconsistent; detailed benefit-claim documentation is not inferred from
surrender or cooling-off rules. A missing-information answer can either refuse
or explicitly explain the absence with a source reference.
