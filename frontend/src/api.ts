import type { ChatRequest, ChatResponse, Health } from "./types";
import { readChatStream, type ChatStreamEvent } from "./stream";

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, { ...options, signal: AbortSignal.timeout(120_000) });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(typeof body?.detail === "string" ? body.detail : `Request failed (${response.status}).`);
  }
  return response.json();
}
export const getHealth = () => request<Health>("/api/health");

/** Consume genuine SSE updates while keeping timeout and cancellation bounded. */
export async function sendMessageStream(
  body: ChatRequest,
  receive: (event: ChatStreamEvent) => void,
  signal?: AbortSignal,
): Promise<ChatResponse> {
  const controller = new AbortController();
  const cancel = () => controller.abort(signal?.reason);
  signal?.addEventListener("abort", cancel, { once: true });
  if (signal?.aborted) cancel();
  const timeout = window.setTimeout(() => controller.abort(new DOMException("The request timed out. Please try again.", "TimeoutError")), 120_000);
  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST", headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body), signal: controller.signal,
    });
    if (!response.ok) {
      const error = await response.json().catch(() => null);
      throw new Error(typeof error?.detail === "string" ? error.detail : `Request failed (${response.status}).`);
    }
    if (!response.body || !response.headers.get("content-type")?.includes("text/event-stream")) {
      throw new Error("The server did not return an answer stream.");
    }
    return await readChatStream(response.body, receive, controller.signal);
  } finally {
    window.clearTimeout(timeout);
    signal?.removeEventListener("abort", cancel);
  }
}
