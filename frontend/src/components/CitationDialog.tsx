import { useEffect, useId, useRef } from "react";
import { createPortal } from "react-dom";
import { sourceSnippet } from "../citationParts";
import { copy } from "../copy";
import type { Citation, Language } from "../types";

/** Show original evidence with native focus trapping and Escape dismissal. */
export default function CitationDialog({ citation, number, claim, language, onClose }: {
  citation: Citation; number: number; claim: string; language: Language; onClose: () => void;
}) {
  const t = copy(language);
  const title = useId();
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    if (dialog.current && !dialog.current.open) dialog.current.showModal();
  }, []);
  function dismiss() {
    dialog.current?.close();
    onClose();
  }
  return createPortal(<dialog ref={dialog} className="citation-dialog" aria-labelledby={title}
    onCancel={event => { event.preventDefault(); dismiss(); }} onClose={onClose}
    onClick={event => { if (event.target === event.currentTarget) dismiss(); }}>
    <div className="citation-dialog-content">
      <div className="citation-dialog-header">
        <h2 id={title}>{t.reference} [{number}]</h2>
        <button type="button" className="citation-close" aria-label={t.close} autoFocus onClick={dismiss}>×</button>
      </div>
      <p className="citation-filename">{citation.filename}</p>
      <p className="citation-page">{t.page} {citation.pdf_page}</p>
      <p className="citation-excerpt-label">{t.sourceExcerpt}</p>
      <blockquote className="citation-snippet">{sourceSnippet(citation.excerpt, claim)}</blockquote>
      <p className="citation-layout-note">{t.sourceLayoutNote}</p>
      <a className="citation-pdf" href={citation.url} target="_blank" rel="noreferrer">{t.viewPdf} <span aria-hidden="true">↗</span></a>
    </div>
  </dialog>, document.body);
}
