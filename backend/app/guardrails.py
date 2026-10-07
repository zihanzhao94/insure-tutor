"""Deterministic input checks and output evidence validation.

These rules reduce common failures; they are not a proof of semantic correctness.
The model must additionally follow the grounded-answer system prompt.
"""

import re
import unicodedata
from decimal import Decimal

from opencc import OpenCC

from .schemas import DocumentChunk, GeneratedAnswer, Language

_simplifier = OpenCC("t2s")
_traditional = OpenCC("s2t")

MESSAGES = {
    "en": {
        "blocked": "I cannot ignore the tutor's rules, reveal secrets, or invent policy terms or citations. Please ask about the supplied insurance plan.",
        "out_of_scope": "I can explain the supplied FLEXI-ULife Prime Saver brochure. I cannot give personal buying advice or answer unrelated questions.",
        "insufficient_evidence": "The supplied brochure does not provide enough evidence to answer this reliably. Please consult the full policy terms or the insurer.",
        "conflict": "The source contains inconsistent information. I cannot select one figure as definitive; please confirm it with the insurer.",
        "disclaimer": "Based on the supplied brochure, not the full policy contract. Figures described as current or assumed may be outdated; this is an explanation, not personal insurance advice.",
        "extractive": "Retrieved passages (source text only; generated answers are disabled):",
    },
    "zh-Hans": {
        "blocked": "我不能忽略规则、透露密钥，或编造保单条款与引用。请询问所提供保险计划的内容。",
        "out_of_scope": "我可以解释所提供的首选灵活万用寿险计划资料，无法提供个人投保建议或回答无关问题。",
        "insufficient_evidence": "所提供的宣传册没有足够证据可靠地回答这个问题。请查阅完整保单条款或向保险公司确认。",
        "conflict": "来源材料存在不一致，无法把其中一个数字当作确定结论。请向保险公司确认。",
        "disclaimer": "回答依据所提供的宣传册，完整保障以保单条款为准。材料中的现时或假设数字可能已过时；此处为资料解释，不是个人投保建议。",
        "extractive": "检索到的原文（当前关闭了模型生成回答）：",
    },
}


def localize(text: str, language: Language) -> str:
    if language == "zh-Hans":
        return _simplifier.convert(text)
    if language == "zh-Hant":
        return _traditional.convert(text)
    return text


def message(kind: str, language: Language) -> str:
    values = MESSAGES["en" if language == "en" else "zh-Hans"]
    return localize(values[kind], language)


def check_input(question: str, has_history: bool = False) -> str | None:
    text = _simplifier.convert(question).lower()
    injection = r"ignore.{0,30}(instructions|rules|prompt)|reveal.{0,30}(prompt|secret|key)|(?:invent|fabricate|fake).{0,30}(citation|policy|term)|忽略.{0,20}(指令|规则|提示)|(?:透露|显示|泄露).{0,15}(密钥|提示词)|(?:伪造|编造).{0,15}(条款|引用|保单)"
    if re.search(injection, text):
        return "blocked"
    advice = r"should i (?:buy|invest)|recommend.{0,30}(?:buy|invest)|我.{0,15}(?:该买|应该买|适合买)|(?:帮我|替我).{0,12}(?:决定|选择).{0,10}(?:保险|投保)"
    if re.search(advice, text):
        return "out_of_scope"
    unrelated = r"\b(?:recipe|pasta|weather|football|president|javascript|poem|joke)\b|菜谱|食谱|天气|足球|总统|写代码|笑话"
    if re.search(unrelated, text):
        return "out_of_scope"
    domain = r"insurance|policy|premium|benefit|interest|guarantee|withdraw|surrender|coverage|insured|illness|death|grace|cancel|charge|account value|cash value|flexi|rate|exclusion|hkd|usd|mop|cooling|claim|unemployment|maturity|levy|issue age|保单|保险|寿险|保费|保障|利率|保证|提款|提取|缴费|交费|缴付|退保|身故|现金|病症|投保|供款|冷静期|失业|费用|不保|港币|港元|美元|万用|派息|回报|保额|理赔"
    if not has_history and not re.search(domain, text):
        return "out_of_scope"
    return None


def _canonical(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).lower()


def _numbers(text: str) -> set[str]:
    raw = re.findall(r"(?<![\d.])\d+(?:,\d{3})*(?:\.\d+)?", text)
    # Numeric formatting can change across languages: 4, 4.0 and 4.00 agree.
    return {str(Decimal(value.replace(",", "")).normalize()) for value in raw}


def validate_answer(answer: GeneratedAnswer, evidence: list[DocumentChunk]) -> None:
    """Reject invented source IDs, non-verbatim quotes, and unsupported numbers.

    An exact quotation verifies provenance, not whether every paraphrase follows
    logically. The eval set and manual review still matter.
    """
    sources = {chunk.chunk_id: chunk for chunk in evidence}
    if answer.status in {"answered", "conflict"} and not answer.claims:
        raise ValueError("An answered response must include supported claims.")
    for claim in answer.claims:
        quotes = []
        for ref in claim.evidence:
            chunk = sources.get(ref.chunk_id)
            if chunk is None:
                raise ValueError("The answer contains an unverifiable source ID.")
            if not ref.quote:
                ref.quote = chunk.text
            if len(ref.quote) < 12 or _canonical(ref.quote) not in _canonical(chunk.text):
                raise ValueError("The answer contains an unverifiable citation or quote.")
            quotes.append(ref.quote)
        supported = {number.replace(",", "") for number in _numbers(" ".join(quotes))}
        claimed = {number.replace(",", "") for number in _numbers(claim.text)}
        if not claimed <= supported:
            raise ValueError("The answer contains a number absent from its quoted evidence.")


def source_conflict(question: str, evidence: list[DocumentChunk], language: Language) -> GeneratedAnswer | None:
    """Check the supplied brochure's bilingual sum-insured-change row.

    Compare extracted values, never correct a source typo by guessing. This is a
    narrowly scoped check for a known table discrepancy, not a general detector.
    """
    text = _simplifier.convert(question).lower()
    if not re.search(r"(?:increase|decrease|change).*(?:sum insured|cover)|(?:增加|减少|更改|变更).*(?:保额|保障额)", text):
        return None
    for chunk in evidence:
        zh = re.search(r"每次更改之最低金額為([\d,]+)美元\s*/\s*([\d,]+)港元\s*/\s*([\d,]+)\s*澳門元", chunk.text)
        en = re.search(r"minimum amount of increase\s*/\s*decrease is US\$([\d,]+)\s*/\s*HK\$([\d,]+)\s*/\s*MOP([\d,]+)", chunk.text)
        if not zh or not en or zh.groups() == en.groups():
            continue
        chinese = f"USD {zh[1]} / HKD {zh[2]} / MOP {zh[3]}"
        english = f"USD {en[1]} / HKD {en[2]} / MOP {en[3]}"
        if language == "en":
            explanation = f"For the minimum increase/decrease in sum insured, the Chinese row states {chinese}, while the English row states {english}. The HKD and MOP amounts disagree; the brochure does not establish which version is correct. Confirm the amount with the insurer."
        else:
            explanation = f"增加或减少保障额的最低金额，中文一行写为 {chinese}，英文一行写为 {english}。港元与澳门元金额不一致，宣传册无法确定哪一版正确，请向保险公司确认。"
        return GeneratedAnswer.model_validate({"status": "conflict", "claims": [{"text": explanation, "evidence": [{"chunk_id": chunk.chunk_id}]}]})
    return None
