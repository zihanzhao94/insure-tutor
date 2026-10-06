export type Language = "en" | "zh-Hans" | "zh-Hant";

export interface Citation {
  document_id: string;
  filename: string;
  pdf_page: number;
  excerpt: string;
}

export interface ChatRequest {
  message: string;
  language: Language;
  session_id?: string;
}

export interface ChatResponse {
  answer: string;
  language: Language;
  session_id: string;
  citations: Citation[];
}
