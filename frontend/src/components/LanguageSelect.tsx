import type { Language } from "../types";

interface Props {
  value: Language;
  onChange: (language: Language) => void;
}

export default function LanguageSelect({ value, onChange }: Props) {
  // TODO: Translate interface copy and pass the selection to the chat request.
  return (
    <label>
      Answer language{" "}
      <select value={value} onChange={(event) => onChange(event.target.value as Language)}>
        <option value="en">English</option>
        <option value="zh-Hans">简体中文</option>
        <option value="zh-Hant">繁體中文</option>
      </select>
    </label>
  );
}
