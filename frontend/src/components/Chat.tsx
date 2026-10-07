import { useEffect, useRef, useState } from "react";
import { getHealth, sendMessage } from "../api";
import { copy } from "../copy";
import type { ChatResponse, Citation, Health, Language } from "../types";
import SourceCard, { type NumberedCitation } from "./SourceCard";
interface Turn { question: string; response?: ChatResponse; }

function groupSources(citations: Citation[]): NumberedCitation[][] {
  // Group by PDF page while preserving the answer's citation numbers.
  const pages = new Map<string, NumberedCitation[]>();
  citations.forEach((citation, index) => {
    const key = JSON.stringify([citation.filename, citation.pdf_page]);
    const sources = pages.get(key) || [];
    sources.push({ citation, number: index + 1 });
    pages.set(key, sources);
  });
  return [...pages.values()];
}
export default function Chat({ language }: { language: Language }) {
  const t = copy(language);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [checking, setChecking] = useState(true);
  const [error, setError] = useState("");
  const session = useRef<string | undefined>(undefined);
  const latest = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  async function checkHealth() {
    setChecking(true);
    try { setHealth(await getHealth()); setHealthError(false); }
    catch { setHealthError(true); }
    finally { setChecking(false); }
  }
  useEffect(() => { void checkHealth(); }, []);
  useEffect(() => { if (turns.length) latest.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }); }, [turns, busy]);
  const ready = health?.index_ready && (health.model_configured || health.mode === "extractive");
  async function submit() {
    const text = question.trim();
    if (!text || busy || !ready) return;
    setBusy(true); setError(""); setQuestion("");
    setTurns(previous => [...previous, { question: text }]);
    try {
      const response = await sendMessage({ message: text, language, session_id: session.current });
      session.current = response.session_id;
      setTurns(previous => previous.map((turn, i) => i === previous.length - 1 ? { ...turn, response } : turn));
    } catch (err) {
      setError(err instanceof Error ? err.message : t.error);
      setQuestion(text);
      setTurns(previous => previous.slice(0, -1));
    } finally { setBusy(false); input.current?.focus(); }
  }
  function reset() { session.current = undefined; setTurns([]); setError(""); setQuestion(""); input.current?.focus(); }
  return <div className="layout">
    <section className="chat-panel" aria-label={t.question}>
      <div className="panel-toolbar"><span className={`status-dot ${ready ? "ready" : ""}`}>{checking ? t.checking : ready ? t.ready : t.setup}</span>
        <button className="text-button" onClick={reset} disabled={busy || !turns.length}>{t.newChat} ↗</button></div>
      {(!ready && !checking || healthError) && <div className="notice" role="status"><p>{healthError ? t.connection : health?.detail || t.setup}</p><button onClick={() => void checkHealth()} disabled={checking}>{t.retry}</button></div>}
      <div className="conversation" aria-live="polite" aria-busy={busy}>
        {!turns.length && <div className="welcome"><p className="eyebrow">{t.eyebrow}</p><h1>{t.title}</h1><p className="intro">{t.introduction}</p>
          <p className="suggestions-label">{t.suggestions}</p><div className="suggestions">{t.examples.map(example => <button key={example} onClick={() => { setQuestion(example); input.current?.focus(); }}>{example}<span>↗</span></button>)}</div>
        </div>}
        {turns.map((turn, i) => <article className="turn" key={i}>
          <div className="question"><span className="message-label">{t.you}</span><p>{turn.question}</p></div>
          {turn.response && <div className="answer"><div className="answer-header"><span className="message-label">{t.assistant}</span><span className={`answer-status ${turn.response.status}`}>{t.statuses[turn.response.status]}</span></div>
            {turn.response.mode === "extractive" && <p className="mode-label">{t.extracted}</p>}
            <div className="answer-text">{turn.response.answer}</div>
            {!!turn.response.citations.length && <div className="sources"><h3>{t.sources}</h3>{groupSources(turn.response.citations).map(sources => <SourceCard key={JSON.stringify([sources[0].citation.filename, sources[0].citation.pdf_page])} sources={sources} language={language} />)}</div>}
          </div>}
        </article>)}
        {busy && <div className="thinking" role="status"><span className="pulse" />{t.thinking}</div>}
        <div ref={latest} />
      </div>
      <form className="composer" onSubmit={event => { event.preventDefault(); void submit(); }}>
        {error && <p className="error" role="alert">{error}</p>}
        <label className="sr-only" htmlFor="question">{t.question}</label>
        <textarea id="question" ref={input} rows={3} maxLength={4000} placeholder={t.placeholder} value={question} disabled={busy} onChange={event => setQuestion(event.target.value)} onKeyDown={event => {
          if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void submit(); }
        }} />
        <div className="composer-actions"><span>{question.length} / 4000</span><button className="send-button" disabled={busy || !ready || !question.trim()}>{t.send}<span>↑</span></button></div>
      </form>
    </section>
    <aside className="document-panel"><p className="eyebrow">{t.sourceLibrary}</p><div className="document-icon" aria-hidden="true">PDF</div><h2>FLEXI-ULife<br />Prime Saver</h2><p className="document-type">{t.documentType}</p>
      {health?.documents.map(filename => <a className="document-link" key={filename} href={`/api/documents/${encodeURIComponent(filename)}`} target="_blank" rel="noreferrer">{t.open} ↗</a>)}
      <div className="side-note"><h3>{t.about}</h3><p>{t.aboutText}</p></div><p className="disclaimer">{t.note}</p>
    </aside>
  </div>;
}
