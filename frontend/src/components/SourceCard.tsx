import { copy } from "../copy";
import type { Citation, Language } from "../types";
export default function SourceCard({ citation, number, language }: { citation: Citation; number: number; language: Language }) {
  const t = copy(language);
  return <details className="source-card">
    <summary><span className="source-number">{number}</span><span>{t.page} {citation.pdf_page}<small>{citation.filename}</small></span></summary>
    <blockquote>{citation.excerpt}</blockquote>
    <a href={citation.url} target="_blank" rel="noreferrer">{t.open} ↗</a>
  </details>;
}
