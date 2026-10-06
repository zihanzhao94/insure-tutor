import type { Citation } from "../types";

export default function SourceCard({ citation }: { citation: Citation }) {
  // TODO: Link to the original PDF page once document serving is implemented.
  return (
    <details>
      <summary>{citation.filename} — PDF page {citation.pdf_page}</summary>
      <blockquote>{citation.excerpt}</blockquote>
    </details>
  );
}
