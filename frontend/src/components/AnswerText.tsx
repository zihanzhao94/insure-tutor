import { useEffect, useRef, useState } from "react";
import { citationParts } from "../citationParts";
import { copy } from "../copy";
import type { Citation, Language } from "../types";
import CitationDialog from "./CitationDialog";
import { ProgressiveText } from "../progressiveText";

export default function AnswerText({ answer, citations, language, animate = false }: {
  answer: string; citations: Citation[]; language: Language; animate?: boolean;
}) {
  const t = copy(language);
  const [selected, setSelected] = useState<number | null>(null);
  const citation = selected === null ? undefined : citations[selected - 1];
  const [visible, setVisible] = useState("");
  const playback = useRef<ProgressiveText | null>(null);
  const content = useRef<HTMLDivElement>(null);
  useEffect(() => {
    playback.current = new ProgressiveText(setVisible);
    return () => playback.current?.dispose();
  }, []);
  useEffect(() => {
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    playback.current?.update(answer, animate && !reducedMotion && !document.hidden);
  }, [answer, animate]);
  useEffect(() => {
    const conversation = content.current?.closest<HTMLElement>(".conversation");
    if (conversation && conversation.scrollHeight - conversation.scrollTop - conversation.clientHeight < 64) {
      conversation.scrollTop = conversation.scrollHeight;
    }
  }, [visible]);
  // Rejected/corrected responses must never leave an old preview on screen.
  const displayed = animate && answer.startsWith(visible) ? visible : answer;
  return <><div ref={content} className="answer-text" aria-busy={displayed !== answer}>{citationParts(displayed, citations).map((part, index) => {
    if (part.type === "text") return part.text;
    const label = `${t.reference} ${part.number}: ${part.citation.filename}, ${t.page} ${part.citation.pdf_page}`;
    return <button type="button" className="inline-citation" key={index}
      title={label} aria-label={label} aria-haspopup="dialog"
      onClick={() => setSelected(part.number)}>{part.number}</button>;
  })}{displayed !== answer && <span className="typing-cursor" aria-hidden="true" />}</div>
    {citation && selected !== null && <CitationDialog citation={citation} number={selected}
      language={language} onClose={() => setSelected(null)} />}
  </>;
}
