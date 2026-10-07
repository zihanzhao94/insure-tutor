import { useEffect, useState } from "react";
import Chat from "./components/Chat";
import LanguageSelect from "./components/LanguageSelect";
import { copy } from "./copy";
import type { Language } from "./types";

export default function App() {
  const [language, setLanguage] = useState<Language>("en");
  const t = copy(language);
  useEffect(() => { document.documentElement.lang = language; }, [language]);
  return <main>
    <header className="topbar">
      <a className="brand" href="/" aria-label="InsureTutor home"><span className="brand-mark">i</span><span>InsureTutor<small>{t.subtitle}</small></span></a>
      <LanguageSelect value={language} onChange={setLanguage} />
    </header>
    <Chat language={language} />
    <footer>{t.local}</footer>
  </main>;
}
