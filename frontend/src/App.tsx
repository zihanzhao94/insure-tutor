import { useState } from "react";
import Chat from "./components/Chat";
import LanguageSelect from "./components/LanguageSelect";
import type { Language } from "./types";

export default function App() {
  const [language, setLanguage] = useState<Language>("en");
  return (
    <main>
      <header><h1>InsureTutor</h1></header>
      <LanguageSelect value={language} onChange={setLanguage} />
      <Chat />
    </main>
  );
}
