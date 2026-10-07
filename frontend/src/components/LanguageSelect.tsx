import { copy } from "../copy";
import type { Language } from "../types";
interface Props { value: Language; onChange: (language: Language) => void; }
export default function LanguageSelect({ value, onChange }: Props) {
  return <label className="language-select">{copy(value).language}
    <select value={value} onChange={event => onChange(event.target.value as Language)}>
      <option value="en">English</option><option value="zh-Hans">简体中文</option><option value="zh-Hant">繁體中文</option>
    </select>
  </label>;
}
