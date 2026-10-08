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
export function sourceSnippet(excerpt: string): string {
  const lines = excerpt.split(/\r\n?|\n/).map(line => line.replace(/\s+/g, " ").trim()).filter(Boolean);
  const numericRow = (line: string) => /\d/.test(line) && /^[\d\s.,%$€£¥()+/−–—:-]+$/.test(line);
  // Keep extracted numeric rows separate instead of inventing table columns.
  const text = lines.map((line, i) => line + (i === lines.length - 1 ? ""
    : numericRow(line) || numericRow(lines[i + 1]) ? "\n" : " ")).join("")
    .replace(/(\p{Script=Han}) +(?=\p{Script=Han})/gu, "$1");
  const characters = Array.from(text);
  if (characters.length <= 420) return text;
  const preview = characters.slice(0, 420).join("");
  // Prefer a complete sentence, avoiding decimal points in amounts and rates.
  const endings = [...preview.matchAll(/[。！？]|[.!?](?=\s|$)/g)];
  const last = endings.at(-1)?.index;
  if (last !== undefined && last >= 240) return preview.slice(0, last + 1) + "…";
  // Do not cut through an English word or a numeric value.
  return preview.replace(/[\p{Script=Latin}\d.,%/-]+$/u, "").trimEnd() + "…";
}
