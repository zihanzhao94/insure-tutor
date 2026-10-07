export type Language = "en" | "zh-Hans" | "zh-Hant";
export interface Citation {
  document_id: string; filename: string; pdf_page: number;
  excerpt: string; chunk_id: string; url: string;
}
export interface ChatRequest { message: string; ui_language: Language; session_id?: string; }
export interface ChatResponse {
  answer: string; session_id: string; citations: Citation[];
  status: "answered" | "insufficient_evidence" | "out_of_scope" | "blocked" | "conflict" | "clarification_required";
  mode: "llm" | "extractive";
}
export interface Health {
  status: string; index_ready: boolean; model_configured: boolean;
  mode: "llm" | "extractive"; provider: string; documents: string[];
  chunks: number; detail: string | null;
}
