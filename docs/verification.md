# Verification record

Checked on 2026-10-07 after migrating to local Chroma and cleaning printed
page-number footers, with the default
GPT + OpenAI Embedding configuration.
Real credentials and raw session/evaluation databases are excluded from Git.

- Backend: **37 non-network tests passed**, including source extraction,
  splitting, index reuse/rebuild failures, source/numeric validation, model API
  payloads/errors, conversations, misuse and document-serving boundaries.
  Chroma-specific checks verify cross-process persistence, cosine search,
  page metadata, and preservation of the active index after a partial write fails.
- Frontend: TypeScript checking and Vite build passed locally and inside Docker
  earlier the same day; frontend code is unchanged by this migration.
- Docker: rebuilt the backend image with Chroma and started both services;
  backend healthcheck passed.
- HTTP: frontend, proxied health endpoint and original PDF endpoint returned 200.
- Index: local Chroma contains 1 PDF, 20 pages, 53 chunks, with 1,536-dimensional
  OpenAI embeddings; unchanged ingestion reused the index. Existing conversation
  SQLite storage and the unused old vector file remain local.
- Browser (earlier the same day): sent a real question, received an answer with source cards, expanded
  original text and verified its PDF page link; switched the interface to
  Traditional Chinese.
- API smoke evaluation: **14/14 cases passed** in one run. Full report is locally
  available at `evals/results/20261007T064940Z.json` (ignored by Git).

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

## Source display update

Also checked on 2026-10-07: frontend TypeScript/Vite builds passed locally and
inside Docker after grouping source cards by PDF page. A browser question returned
three references grouped into two page cards with the original numbers and PDF
page links preserved. Expanded text had a 218px scrollable content area for
1,800px of extracted text. Keyboard collapse and English/Simplified/Traditional
Chinese source controls were verified. See [source-cards.jpg](source-cards.jpg).

## PDF file page references

Printed page-number cleanup was checked against all 20 source pages and a
synthetic PDF containing body numbers, table values, footnotes and transformed
footer text. Only standalone numeric fragments in bottom corners were removed;
the policy amounts, rates, waiting periods and body numbers were retained.
Rendered PDF page 14 confirmed that its printed `13` is a footer.

The Chroma index was rebuilt with `--force`, retained 53 chunks and was reused
on the next ingestion. The backend Docker service restarted healthy and the
14-case API evaluation passed again. Browser verification confirmed the
`PDF page 14` citation label/link, cleaned excerpts and preserved `100`/`31`
policy numbers. See [pdf-page-references.jpg](pdf-page-references.jpg).
