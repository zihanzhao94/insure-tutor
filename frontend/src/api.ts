import type { ChatRequest, ChatResponse, Health } from "./types";

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, { ...options, signal: AbortSignal.timeout(120_000) });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(typeof body?.detail === "string" ? body.detail : `Request failed (${response.status}).`);
  }
  return response.json();
}
export const getHealth = () => request<Health>("/api/health");
export const sendMessage = (body: ChatRequest) => request<ChatResponse>("/api/chat", {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});
