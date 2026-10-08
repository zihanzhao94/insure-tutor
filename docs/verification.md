# Verification record

## Assignment checklist review (2026-10-08)

Compared the repository with [the take-home task](TakeHomeTask-InsureTutor.md).
The chat, document-grounded answers with PDF references, scope controls,
English/Chinese interface, Docker startup, `.env.example`, and README run and
architecture instructions are present. `docker compose config --quiet` passed;
the running backend was healthy with one indexed PDF and 53 chunks; the frontend
returned HTTP 200. `.env` and local evaluation results are ignored by Git.
The latest checks passed 105 backend tests, 17 frontend tests, and the frontend
TypeScript/Vite build.

The remaining quality risk is semantic: source-ID, quote, and number checks cannot establish
that every insurance condition is paraphrased correctly. Manually review
representative answers against the PDF, especially exclusions, waiting periods,
financial guarantees and bilingual discrepancies. Chinese script choice is a
model instruction rather than a deterministic guarantee; Simplified/Traditional
output is a bonus in the assignment.

## Short follow-up and SSE regression (2026-10-08)

- Reproduced the screenshot's two-turn sequence on the old backend:
  `what is the most import part` and `benifit` both returned
  `clarification_required` despite sharing a session ID.
- The backend now treats the first wording as a request for concise key points
  and resolves one-word brochure topics, including `benifit`, to complete
  questions. The original user wording remains in SQLite history.
- The two new live API cases passed **2/2** on the rebuilt Docker backend.
  Repeating the same sequence through `/api/chat/stream` returned `answered`
  for both turns, with 4 and 6 validated `delta` events respectively.
- The backend unit suite passed **104 tests** after this change. The local
  report is `evals/results/20261008T022403Z.json` (ignored by Git).

## Multi-turn memory and retrieval evaluation (2026-10-07)

- The backend now supplies the latest three completed turns to intent routing
  and answer generation. A three-turn question about the 2.5% account-value
  guarantee answered correctly with citations to PDF pages 8 and 16. Recent
  assistant answers remain context only; fresh passages support claims.
- Six chunk-size/overlap settings were re-embedded and compared on nine
  page-labeled questions. All had page Recall@5 of 1.000. The current
  1000/150 setting had page Precision@5 of 0.541 and MRR of 1.000; the
  1400/210 setting had Precision@5 of 0.500 and MRR of 0.889. See
  [the retrieval comparison](../evals/retrieval_results.md). These are
  small, coarse page-level measurements, not a semantic answer score.
- **100 backend tests passed.** The frontend's 17 tests and production build
  passed; the frontend changes visible in the working tree predated this task.
- The Docker backend was rebuilt and restarted from the updated source. Its
  `/api/health` endpoint reported an index-ready 53-chunk collection, and the
  running container reported `MEMORY_TURNS=3`.
- A current-source live API subset on port 8001 initially passed **10/11**
  smoke checks. The terminal-illness case returned insufficient evidence even
  though retrieval found pages 11, 12 and 14. After adding one bounded
  source/number repair attempt and tightening its prompt, that case and the
  three-turn memory case passed **2/2** on the updated server. Reports are
  local and ignored by Git: `20261007T140018Z.json` and
  `20261007T140246Z.json` under `evals/results/`.
- Manual review of the successful terminal-illness response found an
  unnecessary exclusion list. Although its cited pages contain the listed
  terms, source and numeric checks do not establish that every paraphrase or
  implication is correct. This remains a semantic-review item, especially for
  mental-state wording; the smoke check does not certify it.

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

## Interface and answer language separation

Checked on 2026-10-07 with the same GPT/OpenAI configuration:

- **92 non-network backend tests passed**. Added checks cover both buffered and
  streamed answers: the interface locale is absent from the generation payload,
  and model text is preserved without Chinese script conversion. Overview repair
  feedback explicitly preserves the original end-user language preference.
- **11 frontend tests passed**; TypeScript/Vite build passed. The request now
  sends `ui_language` for fixed notices only, and the selector is labelled
  "Interface language" in English and both Chinese interfaces.
- Both Docker images were rebuilt, and the updated backend was rebuilt again
  after the prompt adjustment. The existing index was reused.
- **5/5 live SSE language smoke checks passed**: English interface with a Chinese
  question, Chinese interface with an English question, Simplified interface with
  a Traditional question, an explicit request for Traditional Chinese, and the
  exact screenshot question `这个文档的主要内容总结一下` in an English interface.
  The screenshot question returned a Simplified Chinese overview with source
  references. Report: `evals/results/20261007T100128Z-auto-language.json` (ignored).

These checks establish Chinese versus English response behavior, not guaranteed
script matching. The Simplified cooling-off question returned Traditional Chinese
despite the prompt's default; the application preserves that model output. The
Traditional-question and explicit-Traditional cases returned Traditional Chinese.
Fixed notices and disclaimers remain in the interface locale. The live overview
still paraphrased some exclusions too broadly; the semantic limitation recorded
above remains. No independent semantic judge or source-display change was added.

## Clickable source previews

Checked on 2026-10-07:

- **53 targeted backend tests passed** for RAG and streaming. A citation selected
  by chunk ID retains the server-owned original excerpt, filename, page and URL.
- **14 frontend tests passed**, including line-wrap cleanup, Chinese spacing,
  preservation of policy figures and numeric rows, Unicode-safe truncation,
  and selecting a claim-related passage instead of an unrelated chunk opening.
  TypeScript/Vite build passed; both Docker services are running and the backend
  is healthy. No new model call or index rebuild was required.
- Browser checks confirmed that an inline citation opens a dialog with filename,
  PDF file page, original snippet and **View PDF**. The snippet uses natural
  wrapping instead of the PDF's broken line layout. English, Simplified and
  Traditional interface labels were checked without changing answer/source text.
- The close button, Escape and backdrop dismissed the dialog. Keyboard Tab from
  the close button reached **View PDF**. Its link opened the original PDF viewer
  at **page 15 of 20**. See [citation-preview.png](citation-preview.png).

The preview is a shortened original source passage, selected using shared terms
and numbers. It does not translate the PDF, reconstruct tables, highlight an exact
PDF paragraph or independently prove the cited claim's meaning. The complete
source remains accessible through the page link.

## Dynamic source-library display

Checked on 2026-10-07:

- **14 frontend tests passed** and TypeScript/Vite build passed. No backend
  logic or source-index data was changed.
- The rebuilt Docker UI no longer contains the three fixed example questions,
  a fixed product heading, or an assumed document-language label.
- English and Simplified Chinese browser checks confirmed the welcome copy,
  actual indexed filename `FLEXI-ULife Prime Saver 23.15.41.pdf`, and its original
  PDF link. See [source-library.png](source-library.png).

The source library uses `/api/health`'s active-index document list. Changed files
require ingestion and a page refresh; this check did not replace the supplied
PDF or evaluate a different insurance product. Brochure-specific backend prompts
and conflict rules remain, as explained in the README.
