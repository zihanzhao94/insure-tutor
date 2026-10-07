import { citationParts } from "../citationParts";
import { copy } from "../copy";
import type { Citation, Language } from "../types";

export default function AnswerText({ answer, citations, language }: {
  answer: string; citations: Citation[]; language: Language;
}) {
  const t = copy(language);
  return <div className="answer-text">{citationParts(answer, citations).map((part, index) => {
    if (part.type === "text") return part.text;
    const label = `${t.reference} ${part.number}: ${part.citation.filename}, ${t.page} ${part.citation.pdf_page}`;
    return <a className="inline-citation" key={index} href={part.citation.url} target="_blank" rel="noreferrer"
      title={label} aria-label={label}>{part.number}</a>;
  })}</div>;
}
