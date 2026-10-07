import type { ChatResponse, Citation } from "./types";

export type StreamPhase = "classifying" | "retrieving" | "generating" | "checking";
export type ChatStreamEvent =
  | { event: "start"; data: { session_id: string; mode: ChatResponse["mode"] } }
  | { event: "status"; data: { phase: StreamPhase } }
  | { event: "delta"; data: { text: string; citations: Citation[] } }
  | { event: "reset"; data: Record<string, never> }
  | { event: "result"; data: ChatResponse };

/** Assemble SSE frames across arbitrary network and CRLF boundaries. */
export class SseParser {
  private buffer = "";
  private event = "message";
  private data: string[] = [];

  constructor(private receive: (event: string, data: string) => void) {}

  push(text: string) {
    this.buffer += text;
    while (true) {
      const boundary = this.buffer.search(/[\r\n]/);
      if (boundary < 0) return;
      // A CR at the end might be the first half of a split CRLF.
      if (this.buffer[boundary] === "\r" && boundary === this.buffer.length - 1) return;
      const line = this.buffer.slice(0, boundary);
      const width = this.buffer.slice(boundary, boundary + 2) === "\r\n" ? 2 : 1;
      this.buffer = this.buffer.slice(boundary + width);
      if (!line) {
        const event = this.event;
        const data = this.data;
        this.event = "message";
        this.data = [];
        if (data.length) this.receive(event, data.join("\n"));
      } else if (!line.startsWith(":")) {
        const colon = line.indexOf(":");
        const field = colon < 0 ? line : line.slice(0, colon);
        const value = colon < 0 ? "" : line.slice(colon + 1).replace(/^ /, "");
        if (field === "event") this.event = value;
        if (field === "data") this.data.push(value);
      }
    }
  }

  finish() {
    // At EOF a trailing CR is a complete line ending, not a split CRLF.
    if (this.buffer.endsWith("\r")) this.push("\n");
  }
}

/** Read validated increments; only a final result makes a response complete. */
export async function readChatStream(
  body: ReadableStream<Uint8Array>,
  receive: (event: ChatStreamEvent) => void,
  signal?: AbortSignal,
): Promise<ChatResponse> {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let result: ChatResponse | undefined;
  const parser = new SseParser((event, raw) => {
    if (result) return;
    // Comments/heartbeats and unknown event types carry no answer content.
    if (!["start", "status", "delta", "reset", "result", "error"].includes(event)) return;
    const data = JSON.parse(raw);
    if (event === "error") throw new Error(typeof data?.detail === "string" ? data.detail : "The answer could not be completed.");
    if (event === "result") {
      if (typeof data?.answer !== "string" || typeof data?.session_id !== "string" || !Array.isArray(data?.citations)) {
        throw new Error("The answer stream returned an invalid result.");
      }
      result = data as ChatResponse;
    } else if (event === "delta" && (typeof data?.text !== "string" || !Array.isArray(data?.citations))) {
      throw new Error("The answer stream returned an invalid update.");
    }
    receive({ event, data } as ChatStreamEvent);
  });
  const cancel = () => { void reader.cancel(signal?.reason).catch(() => {}); };
  signal?.addEventListener("abort", cancel, { once: true });
  try {
    while (!result) {
      signal?.throwIfAborted();
      const { value, done } = await reader.read();
      signal?.throwIfAborted();
      parser.push(done ? decoder.decode() : decoder.decode(value, { stream: true }));
      if (done) parser.finish();
      if (done && !result) throw new Error("The answer stream ended before completion. Please try again.");
    }
    return result;
  } finally {
    signal?.removeEventListener("abort", cancel);
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
