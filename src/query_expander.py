from google import genai
import time
import random

from .config import (
    GEMINI_API_KEY,
    GEMINI_AUX_MODEL,
    GEMINI_AUX_FALLBACK_MODEL,
    GEMINI_RETRY_MAX_ATTEMPTS,
    GEMINI_RETRY_BASE_SECONDS,
    GEMINI_RETRY_JITTER_SECONDS,
)

client = genai.Client(api_key=GEMINI_API_KEY)

def _is_retryable_llm_error(exc: Exception) -> bool:
    s = str(exc).lower()
    retry_marks = [
        "503",
        "unavailable",
        "high demand",
        "resource_exhausted",
        "429",
        "deadline exceeded",
        "timed out",
        "connection reset",
        "remoteprotocolerror",
        "server disconnected without sending a response",
        "connection aborted",
        "broken pipe",
    ]
    return any(m in s for m in retry_marks)


def expand_query(query):
    prompt = f"""
Rewrite the following philosophy question into 3 different search queries.
The goal is to retrieve relevant philosophical texts.

Question:
{query}

Queries:
"""
    candidates = [GEMINI_AUX_MODEL]
    if GEMINI_AUX_FALLBACK_MODEL and GEMINI_AUX_FALLBACK_MODEL != GEMINI_AUX_MODEL:
        candidates.append(GEMINI_AUX_FALLBACK_MODEL)

    text = ""
    last_err: Exception | None = None
    for model_name in candidates:
        attempts = max(1, int(GEMINI_RETRY_MAX_ATTEMPTS))
        for attempt in range(1, attempts + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                text = response.text or ""
                last_err = None
                break
            except Exception as e:
                last_err = e
                if (not _is_retryable_llm_error(e)) or attempt >= attempts:
                    break
                backoff = float(GEMINI_RETRY_BASE_SECONDS) * (2 ** (attempt - 1))
                jitter = random.uniform(0.0, float(GEMINI_RETRY_JITTER_SECONDS))
                time.sleep(backoff + jitter)
        if text:
            break

    # 如果扩写彻底失败，退化为只用原 query（不让检索链路直接崩）
    if not text:
        return [query]
    queries = [query]
    for line in text.split("\n"):
        line = line.strip()
        if len(line) > 5:
            queries.append(line)
    return list(set(queries))
