import { useState } from "react";
import { citationParts } from "../citationParts";
import { copy } from "../copy";
import type { Citation, Language } from "../types";
import CitationDialog from "./CitationDialog";

export default function AnswerText({ answer, citations, language }: {
  answer: string; citations: Citation[]; language: Language;
}) {
  const t = copy(language);
  const [selected, setSelected] = useState<{ number: number; claim: string } | null>(null);
  const citation = selected === null ? undefined : citations[selected.number - 1];
  let claim = "";
  return <><div className="answer-text">{citationParts(answer, citations).map((part, index) => {
    if (part.type === "text") { claim = part.text.split("\n\n").at(-1) || ""; return part.text; }
    const citedClaim = claim;
    const label = `${t.reference} ${part.number}: ${part.citation.filename}, ${t.page} ${part.citation.pdf_page}`;
    return <button type="button" className="inline-citation" key={index}
      title={label} aria-label={label} aria-haspopup="dialog"
      onClick={() => setSelected({ number: part.number, claim: citedClaim })}>{part.number}</button>;
  })}</div>
    {citation && selected !== null && <CitationDialog citation={citation} number={selected.number} claim={selected.claim}
      language={language} onClose={() => setSelected(null)} />}
  </>;
}
