import { useEffect, useRef, useState } from "react";
import { getHealth, sendMessageStream } from "../api";
import { copy } from "../copy";
import type { ChatResponse, Citation, Health, Language } from "../types";
import type { StreamPhase } from "../stream";
import AnswerText from "./AnswerText";
interface Draft {
  answer: string; citations: Citation[]; language: Language;
  mode: ChatResponse["mode"]; phase: StreamPhase;
}
interface Turn { question: string; response?: ChatResponse; draft?: Draft; }
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
  const activeRequest = useRef<AbortController | null>(null);
  const latest = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  async function checkHealth() {
    setChecking(true);
    try { setHealth(await getHealth()); setHealthError(false); }
    catch { setHealthError(true); }
    finally { setChecking(false); }
  }
  useEffect(() => { void checkHealth(); }, []);
  useEffect(() => () => activeRequest.current?.abort(), []);
  useEffect(() => { if (!busy && turns.length) input.current?.focus(); }, [busy]);
  useEffect(() => { if (turns.length) latest.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }); }, [turns, busy]);
  const ready = health?.index_ready && (health.model_configured || health.mode === "extractive");
  async function submit() {
    const text = question.trim();
    if (!text || activeRequest.current || !ready) return;
    const controller = new AbortController();
    activeRequest.current = controller;
    setBusy(true); setError(""); setQuestion("");
    setTurns(previous => [...previous, { question: text, draft: {
      answer: "", citations: [], language, mode: health?.mode || "llm", phase: "classifying",
    } }]);
    function updateDraft(update: (draft: Draft) => Draft) {
      if (activeRequest.current !== controller || controller.signal.aborted) return;
      setTurns(previous => previous.map((turn, i) => i === previous.length - 1 && turn.draft
        ? { ...turn, draft: update(turn.draft) } : turn));
    }
    try {
      const response = await sendMessageStream({ message: text, language, session_id: session.current }, event => {
        if (activeRequest.current !== controller || controller.signal.aborted) return;
        switch (event.event) {
          case "start":
            session.current = event.data.session_id;
            updateDraft(draft => ({ ...draft, language: event.data.language, mode: event.data.mode }));
            break;
          case "status":
            updateDraft(draft => ({ ...draft, phase: event.data.phase }));
            break;
          case "delta":
            updateDraft(draft => ({ ...draft, answer: draft.answer + event.data.text, citations: event.data.citations }));
            break;
          case "reset":
            updateDraft(draft => ({ ...draft, answer: "", citations: [] }));
            break;
        }
      }, controller.signal);
      if (activeRequest.current !== controller) return;
      controller.signal.throwIfAborted();
      session.current = response.session_id;
      // The final result replaces provisional text, including a failed repair.
      setTurns(previous => previous.map((turn, i) => i === previous.length - 1 ? { question: turn.question, response } : turn));
    } catch (err) {
      if (activeRequest.current !== controller) return;
      if (!controller.signal.aborted) setError(err instanceof Error ? err.message : t.error);
      setQuestion(text);
      setTurns(previous => previous.slice(0, -1));
    } finally {
      if (activeRequest.current === controller) {
        activeRequest.current = null;
        setBusy(false);
      }
    }
  }
  function stop() {
    const controller = activeRequest.current;
    if (!controller) return;
    // Restore the composer immediately; ignore any late transport callbacks.
    activeRequest.current = null;
    controller.abort();
    setBusy(false);
    setQuestion(turns.at(-1)?.question || "");
    setTurns(previous => previous.slice(0, -1));
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
        {turns.map((turn, i) => {
          const answer = turn.response || turn.draft;
          return <article className="turn" key={i}>
          <div className="question"><span className="message-label">{t.you}</span><p>{turn.question}</p></div>
          {!!answer?.answer && <div className="answer"><div className="answer-header"><span className="message-label">{t.assistant}</span>
            {turn.response && <span className={`answer-status ${turn.response.status}`}>{t.statuses[turn.response.status]}</span>}</div>
            {answer.mode === "extractive" && <p className="mode-label">{t.extracted}</p>}
            <AnswerText answer={answer.answer} citations={answer.citations} language={language} />
            {!!answer.citations.length && <p className="citation-hint">{t.citationHint}</p>}
          </div>}
        </article>;
        })}
        {busy && <div className="thinking" role="status"><span className="pulse" />{t.phases[turns.at(-1)?.draft?.phase || "classifying"]}</div>}
        <div ref={latest} />
      </div>
      <form className="composer" onSubmit={event => { event.preventDefault(); void submit(); }}>
        {error && <p className="error" role="alert">{error}</p>}
        <label className="sr-only" htmlFor="question">{t.question}</label>
        <textarea id="question" ref={input} rows={3} maxLength={4000} placeholder={t.placeholder} value={question} disabled={busy} onChange={event => setQuestion(event.target.value)} onKeyDown={event => {
          if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void submit(); }
        }} />
        <div className="composer-actions"><span>{question.length} / 4000</span>{busy
          ? <button className="send-button" type="button" onClick={stop}>{t.stop}<span aria-hidden="true">■</span></button>
          : <button className="send-button" disabled={!ready || !question.trim()}>{t.send}<span>↑</span></button>}</div>
      </form>
    </section>
    <aside className="document-panel"><p className="eyebrow">{t.sourceLibrary}</p><div className="document-icon" aria-hidden="true">PDF</div><h2>FLEXI-ULife<br />Prime Saver</h2><p className="document-type">{t.documentType}</p>
      {health?.documents.map(filename => <a className="document-link" key={filename} href={`/api/documents/${encodeURIComponent(filename)}`} target="_blank" rel="noreferrer">{t.open} ↗</a>)}
      <div className="side-note"><h3>{t.about}</h3><p>{t.aboutText}</p></div><p className="disclaimer">{t.note}</p>
    </aside>
  </div>;
}
