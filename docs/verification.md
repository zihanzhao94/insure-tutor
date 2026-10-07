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

## AI intent routing

Checked on 2026-10-07 with the existing GPT/OpenAI configuration:

- **56 non-network backend tests passed.** New checks cover validated category
  labels, malformed classifier output, history-aware routing, clarification,
  extractive-mode compatibility, overview coverage/budget and one bounded
  citation-repair attempt followed by the same evidence validation.
- Frontend TypeScript/Vite build passed. Both Docker services were rebuilt;
  backend health and the frontend/proxied health endpoints passed.
- The initial 22-case live run passed **21/22** checks, including all 14 existing
  cases. English overview generation failed its source/number checks. Report:
  `evals/results/20261007T084426Z.json` (ignored).
- The expanded eight-case run passed **7/8** before the bounded overview repair
  was added. Paraphrased personal advice, advice after history, unrelated topics
  after history, ambiguous input and a legitimate suicide-exclusion question
  passed. Report: `evals/results/20261007T084737Z.json` (ignored).
- After adding summary-table coverage, clearer topic/citation instructions and
  a single repair attempt, the three overview cases passed twice. The final
  run passed **3/3**, citing PDF pages 8, 12, 14, 15 and 16:
  `evals/results/20261007T085255Z.json` (ignored).

The original screenshot question now returns a sourced overview in Simplified
and Traditional Chinese, rather than failing a required-keyword check.
Only the `uncertain` category requests clarification; a clear question with no
source evidence follows retrieval and returns an insufficient-evidence response.
Provider or malformed-category failures return safe API errors.

These are smoke checks, not classification accuracy or semantic correctness
scores. Manual review still found summaries that paraphrase terminal-illness
exclusions too broadly, including a Chinese wording that can wrongly imply
mental illness itself is excluded. The overview prompt explicitly cautions
against this, but source-ID and number validation cannot enforce that meaning.
Exact exclusions, waiting-period conditions and financial guarantees still need
review against the PDF; no independent semantic output judge was added.

## Streaming and inline citations

Checked on 2026-10-07 with the same GPT/OpenAI configuration:

- **86 non-network backend tests passed**, including 30 streaming cases.
  Checks cover genuine incremental emission, closed/escaped JSON claims,
  rejection of unsupported sources/numbers/quotes, authoritative final rejection,
  one overview repair with reset, provider refusals/truncation/disconnection,
  SSE errors and completed-only conversation persistence.
- **11 frontend tests passed** for UTF-8 and SSE fragmentation, CRLF boundaries,
  incomplete/error responses, reset/final replacement, cancellation propagation
  into fetch and reader cleanup, and safe inline citation URLs.
  TypeScript/Vite builds passed locally and inside Docker.
- Both Docker services restarted; the backend was healthy. The frontend,
  proxied health endpoint and original PDF endpoint returned 200. The existing
  Chroma index was reused; no ingestion or embedding migration was needed.
- **8/8 live SSE smoke cases passed**: Simplified Chinese cooling-off,
  Traditional Chinese overview, missing claim-document evidence, injection,
  unrelated input, bilingual table conflict, English guarantee and its follow-up.
  Report: `evals/results/20261007T092144Z-streaming.json` (ignored).
  The overview's first validated paragraph arrived at **7.996 seconds** and its
  final response at **13.119 seconds**. Successful answer deltas concatenated to
  the final answer; source links and reference numbers matched their metadata.
- Browser checks confirmed English and Traditional Chinese inline references,
  language switching without rewriting old messages, and removal of raw-source
  cards. Clicking reference 1 opened the PDF viewer at **page 8 of 20**.
  See [inline-citations.jpg](inline-citations.jpg).

Streaming exposes validated paragraphs rather than raw model tokens. A final
failure can replace an earlier supported preview. Stop restores the composer
immediately; request identity guards ignore late updates from a canceled turn.
Provider reads can finish their current read before transport resources release.
References request PDF file pages, with no paragraph highlighting.

Manual review of the live overview still found overly broad terminal-illness
exclusion wording. The existing semantic limitation remains: source and numeric
validation establish provenance, not the correctness of every interpretation.
