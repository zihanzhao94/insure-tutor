import type { Citation } from "./types";

export type AnswerPart =
  | { type: "text"; text: string }
  | { type: "citation"; number: number; citation: Citation };

/** Link known server-owned citations and leave all other text untouched. */
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
