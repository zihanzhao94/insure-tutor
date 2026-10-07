import type { Citation } from "./types";

export type AnswerPart =
  | { type: "text"; text: string }
  | { type: "citation"; number: number; citation: Citation };

/** Resolve known server-owned citations and leave all other text untouched. */
export function citationParts(answer: string, citations: Citation[]): AnswerPart[] {
  const parts: AnswerPart[] = [];
  const markers = /\[([1-9]\d*)\]/g;
  let start = 0;
  for (const match of answer.matchAll(markers)) {
    const number = Number(match[1]);
    const citation = citations[number - 1];
    if (!citation || !/^\/api\/documents\/[^?#]+#page=[1-9]\d*$/.test(citation.url)) continue;
    if (match.index > start) parts.push({ type: "text", text: answer.slice(start, match.index) });
    parts.push({ type: "citation", number, citation });
    start = match.index + match[0].length;
  }
  if (start < answer.length) parts.push({ type: "text", text: answer.slice(start) });
  return parts;
}

/** Compact extracted text without translating or rewriting the source. */
export function sourceSnippet(excerpt: string, claim = ""): string {
  const lines = excerpt.split(/\r\n?|\n/).map(line => line.replace(/\s+/g, " ").trim()).filter(Boolean);
  const numericRow = (line: string) => /\d/.test(line) && /^[\d\s.,%$€£¥()+/−–—:-]+$/.test(line);
  // Keep extracted numeric rows separate instead of inventing table columns.
  let text = lines.map((line, i) => line + (i === lines.length - 1 ? ""
    : numericRow(line) || numericRow(lines[i + 1]) ? "\n" : " ")).join("")
    .replace(/(\p{Script=Han}) +(?=\p{Script=Han})/gu, "$1");
  // Prefer a source sentence sharing the claim's terms; keep an original substring.
  const stopWords = new Set(["the", "and", "for", "with", "from", "that", "this", "are", "was", "will", "have", "has", "its"]);
  const terms = [...new Set((claim.toLowerCase().match(/\d+(?:,\d{3})*(?:\.\d+)?%?|[\p{Script=Latin}]{3,}|\p{Script=Han}{2}/gu) || [])
    .filter(term => !stopWords.has(term)))];
  const starts = [0, ...[...text.matchAll(/(?:[。！？\n]|[.!?](?=\s))\s*/g)].map(match => match.index + match[0].length)];
  const score = (start: number) => {
    const window = Array.from(text.slice(start)).slice(0, 420).join("").toLowerCase();
    return terms.reduce((total, term) => {
      const position = window.indexOf(term);
      return total + (position < 0 ? 0 : (/^\d/.test(term) ? 4 : 1) * (1 - position / 420));
    }, 0);
  };
  let start = 0;
  for (const candidate of starts) if (score(candidate) > score(start)) start = candidate;
  const prefix = start ? "…" : "";
  text = text.slice(start);
  const characters = Array.from(text);
  if (characters.length <= 420) return prefix + text;
  const preview = characters.slice(0, 420).join("");
  // Prefer a complete sentence, avoiding decimal points in amounts and rates.
  const endings = [...preview.matchAll(/[。！？]|[.!?](?=\s|$)/g)];
  const last = endings.at(-1)?.index;
  if (last !== undefined && last >= 240) return prefix + preview.slice(0, last + 1) + "…";
  // Do not cut through an English word or a numeric value.
  return prefix + preview.replace(/[\p{Script=Latin}\d.,%/-]+$/u, "").trimEnd() + "…";
}
