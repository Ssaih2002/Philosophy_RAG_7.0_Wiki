from typing import List

from google import genai

from .config import GEMINI_API_KEY, GEMINI_AUX_MODEL


client = genai.Client(api_key=GEMINI_API_KEY)


def extract_keywords_from_question(question: str) -> List[str]:
    prompt = f"""从下面的问题中抽取用于原文关键词检索的术语，3~8个。
要求：
1) 仅输出逗号分隔短语
2) 不要编号、不要解释、不要引号
3) 优先哲学专名、概念、术语

问题：
{question}
"""
    response = client.models.generate_content(
        model=GEMINI_AUX_MODEL,
        contents=prompt,
    )
    text = (response.text or "").strip()
    terms: List[str] = []
    for seg in text.replace("，", ",").replace(";", ",").split(","):
        s = seg.strip().strip("'\"")
        if len(s) >= 2:
            terms.append(s)
    out: List[str] = []
    seen = set()
    for t in terms:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out[:12]
