from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .llm_router import generate_answer
from .web_search import WebSearchResult, web_search_wikipedia_multi
from .config import OPENAI_API_KEY, DEEPSEEK_API_KEY, OPENAI_MODEL_PRIMARY, DEEPSEEK_MODEL_PRIMARY


def _format_sources(results: List[WebSearchResult]) -> str:
    if not results:
        return "（无）"
    lines: List[str] = []
    for i, r in enumerate(results, start=1):
        snip = (r.snippet or "").strip().replace("\n", " ").strip()
        if len(snip) > 320:
            snip = snip[:320].rstrip() + "…"
        lines.append(f"[{i}] {r.title}\n- URL: {r.url}\n- 摘要: {snip}")
    return "\n\n".join(lines)


def build_standard_qa_prompt(question: str, *, web_results: List[WebSearchResult]) -> str:
    """
    标准问答（不依赖本地语料/向量库）：仅允许基于联网搜索结果回答。
    """
    sources_block = _format_sources(web_results)
    q = (question or "").strip()
    return f"""你是一名严谨的研究助理。你将基于「联网搜索结果」回答用户问题。

## 最高优先级规则（强制）
- 只允许使用下方 Sources 中的信息来回答；**不得**凭空补全细节、不得编造出处、不得把猜测说成事实。
- 若 Sources 不足以支持某个结论：明确写「目前检索证据不足」并说明缺口；可以给出下一步检索建议。
- 遇到争议/口径不一：分别呈现不同来源的说法，并指出差异点与可能原因（时间、定义、样本、版本）。
- 输出必须「言之有物」：给出关键定义、结论、证据链与限定条件；避免空泛套话。

## 输出格式（建议但不强制）
- 先给结论要点（3–7 条）
- 再解释推理过程与证据（引用 Sources 的编号）
- 最后给「参考来源」列表（按编号列出 URL）

## Sources（联网搜索结果）
{sources_block}

## 用户问题
{q}
"""


def answer_standard_qa(
    question: str,
    *,
    provider: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
    web_max_results: int = 6,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    # Wikipedia-only (English + German)
    # 注意：web_search 内部会把 query 收敛成短关键词，避免把整段 prompt 当作 srsearch
    results = web_search_wikipedia_multi(
        question, max_results=web_max_results, langs=["en", "de", "zh"]
    )
    prompt = build_standard_qa_prompt(question, web_results=results)
    # Gemini 在部分地区会返回 FAILED_PRECONDITION（不支持的地理位置）。
    # 标准问答不应因此让整个接口 500：尽量自动切到其他供应商；否则返回可操作的错误提示。
    model_used = ""
    try:
        text, model_used = generate_answer(
            prompt=prompt,
            provider=provider,
            model=model,
            temperature=float(temperature),
            max_output_tokens=int(max_output_tokens),
        )
    except Exception as e:
        msg = str(e)
        # 尝试跨供应商降级（仅当 key 已配置）
        if (
            (provider or "").lower() == "gemini"
            and ("User location is not supported" in msg or "FAILED_PRECONDITION" in msg)
        ):
            if (OPENAI_API_KEY or "").strip():
                text, model_used = generate_answer(
                    prompt=prompt,
                    provider="openai",
                    model=OPENAI_MODEL_PRIMARY,
                    temperature=float(temperature),
                    max_output_tokens=int(max_output_tokens),
                )
            elif (DEEPSEEK_API_KEY or "").strip():
                text, model_used = generate_answer(
                    prompt=prompt,
                    provider="deepseek",
                    model=DEEPSEEK_MODEL_PRIMARY,
                    temperature=float(temperature),
                    max_output_tokens=int(max_output_tokens),
                )
            else:
                text = (
                    "标准问答模式已完成 Wikipedia 检索，但当前所选 Gemini API 在你的地区不可用（FAILED_PRECONDITION）。\n\n"
                    "解决方式：\n"
                    "1) 在前端把“回答模型”切到 OpenAI 或 DeepSeek；并在 `src/config.py` 填好对应 API Key。\n"
                    "2) 或为 Gemini 配置可用的网络出口（代理/可用地区）。\n"
                )
                model_used = "gemini:unavailable_location"
        else:
            # 其他异常：同样不要 500，给出最小可行动信息
            text = f"标准问答模式生成失败：{msg}"
            model_used = f"{provider}:{model}"

    meta: Dict[str, Any] = {
        "answer_model": model_used,
        "web_search": {
            "engine": "wikipedia(en,de,zh)",
            "query": (question or "").strip(),
            "results": [
                {"title": r.title, "url": r.url, "snippet": r.snippet} for r in results
            ],
        },
    }
    # 标准问答不返回本地 docs（避免误导为“来自语料库引用”）
    return text, [], meta

