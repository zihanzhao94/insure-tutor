import { copy } from "../copy";
import type { Citation, Language } from "../types";
export interface NumberedCitation { citation: Citation; number: number; }
export default function SourceCard({ sources, language }: { sources: NumberedCitation[]; language: Language }) {
  const t = copy(language);
  const { citation } = sources[0];
  return <section className="source-card" aria-label={`${t.page} ${citation.pdf_page}`}>
    <div className="source-heading">
      <div className="source-meta">
        <div className="source-title"><span className="source-references">{sources.map(({ number }) =>
          <span className="source-number" key={number}>[{number}]</span>)}</span>
          <span className="source-page">{t.page} {citation.pdf_page}</span></div>
        <small>{citation.filename}</small>
      </div>
      <a className="source-link" href={citation.url} target="_blank" rel="noreferrer"
        aria-label={`${t.open}: ${citation.filename}, ${t.page} ${citation.pdf_page}`}>{t.openPage} <span aria-hidden="true">↗</span></a>
    </div>
    <details className="source-details">
      <summary><span className="source-show">{t.viewPassages}</span><span className="source-hide">{t.hidePassages}</span><span className="source-chevron" aria-hidden="true">⌄</span></summary>
      <p className="source-note">{t.extractionNote}</p>
      <div className="source-passages" tabIndex={0} role="region" aria-label={`${t.sources}, ${t.page} ${citation.pdf_page}`}>
        {sources.map(({ citation: source, number }) => <div className="source-passage" key={number}>
          <p className="source-passage-label">{t.reference} [{number}]</p>
          <blockquote>{source.excerpt}</blockquote>
        </div>)}
      </div>
    </details>
  </section>;
}
