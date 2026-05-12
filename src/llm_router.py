import time
import random
import threading
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .config import (
    OPENAI_API_KEY,
    DEEPSEEK_API_KEY,
    OPENAI_BASE_URL,
    DEEPSEEK_BASE_URL,
    OPENAI_MODEL_PRIMARY,
    OPENAI_MODEL_SECONDARY,
    DEEPSEEK_MODEL_PRIMARY,
    GEMINI_ANSWER_MODEL,
    GEMINI_FALLBACK_MODELS,
    GEMINI_RETRY_MAX_ATTEMPTS,
    GEMINI_RETRY_BASE_SECONDS,
    GEMINI_RETRY_JITTER_SECONDS,
    OPENAI_MAX_CONCURRENCY,
    OPENAI_RETRY_MAX_ATTEMPTS,
    OPENAI_RETRY_BASE_SECONDS,
    OPENAI_RETRY_JITTER_SECONDS,
    OPENAI_RETRY_MAX_SLEEP_SECONDS,
)
from .llm_gemini import generate_with_retry_and_fallback, is_quota_exhausted_error, is_retryable_llm_error


def _gemini_fallback_chain(primary: str) -> List[str]:
    """
    Per-model cascade (each model gets up to GEMINI_RETRY_MAX_ATTEMPTS tries in llm_gemini):
    - gemini-3.1-pro -> 2.5-pro -> 2.5-flash
    - gemini-2.5-pro -> 2.5-flash
    - gemini-2.5-flash -> (none)
    Unknown ids fall back to GEMINI_FALLBACK_MODELS from config (excluding primary).
    """
    pl = (primary or "").strip().lower()
    if pl in ("gemini-3.1-pro", "gemini-3-pro", "gemini-3.1-pro-preview", "gemini-3-pro-preview"):
        return ["gemini-2.5-pro", "gemini-2.5-flash"]
    if pl == "gemini-2.5-pro":
        return ["gemini-2.5-flash"]
    if pl == "gemini-2.5-flash":
        return []
    out: List[str] = []
    for m in GEMINI_FALLBACK_MODELS:
        if m and str(m).strip().lower() != pl:
            out.append(str(m).strip())
    return out
from .net_proxy import apply_proxy_env, get_proxy_url

apply_proxy_env()


def _sleep_backoff(attempt: int) -> None:
    backoff = float(GEMINI_RETRY_BASE_SECONDS) * (2 ** (attempt - 1))
    jitter = random.uniform(0.0, float(GEMINI_RETRY_JITTER_SECONDS))
    time.sleep(backoff + jitter)


def _http_retryable(exc: Exception) -> bool:
    return is_retryable_llm_error(exc)


class OpenAIHTTPError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        retry_after_seconds: Optional[float],
        response_text: str,
    ):
        super().__init__(message)
        self.status_code = int(status_code)
        self.retry_after_seconds = retry_after_seconds
        self.response_text = response_text


_openai_sem = threading.Semaphore(max(1, int(OPENAI_MAX_CONCURRENCY)))
_AUTO_COOLDOWNS: Dict[str, float] = {}
_AUTO_COOLDOWNS_LOCK = threading.Lock()
_AUTO_QUOTA_COOLDOWN_SECONDS = 6 * 60 * 60


def _cooldown_key(provider: str, model: str) -> str:
    return f"{(provider or '').strip().lower()}:{(model or '').strip()}"


def _is_auto_cooled(provider: str, model: str) -> bool:
    key = _cooldown_key(provider, model)
    now = time.time()
    with _AUTO_COOLDOWNS_LOCK:
        until = float(_AUTO_COOLDOWNS.get(key) or 0.0)
        if until <= now:
            _AUTO_COOLDOWNS.pop(key, None)
            return False
        return True


def _mark_auto_cooldown(provider: str, model: str, seconds: float = _AUTO_QUOTA_COOLDOWN_SECONDS) -> None:
    key = _cooldown_key(provider, model)
    with _AUTO_COOLDOWNS_LOCK:
        _AUTO_COOLDOWNS[key] = max(float(_AUTO_COOLDOWNS.get(key) or 0.0), time.time() + max(60.0, seconds))


def _openai_url() -> str:
    return f"{str(OPENAI_BASE_URL).rstrip('/')}/responses"


def _preview_text(text: str, limit: int = 360) -> str:
    out = (text or "").strip().replace("\r", " ").replace("\n", " ")
    if len(out) > limit:
        out = out[:limit] + "…"
    return out


def _parse_retry_after_seconds(headers: httpx.Headers) -> Optional[float]:
    ra = (headers.get("retry-after") or "").strip()
    if not ra:
        return None
    # Retry-After can be seconds or HTTP-date.
    try:
        return float(ra)
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(ra)
        now = dt.now(tz=dt.tzinfo)
        return max(0.0, (dt - now).total_seconds())
    except Exception:
        return None


def _sleep_openai_backoff(attempt: int, retry_after_seconds: Optional[float]) -> None:
    base = float(OPENAI_RETRY_BASE_SECONDS) * (2 ** (attempt - 1))
    jitter = random.uniform(0.0, float(OPENAI_RETRY_JITTER_SECONDS))
    sleep_s = base + jitter
    if retry_after_seconds is not None:
        # Respect server guidance if it's larger than our local backoff.
        sleep_s = max(sleep_s, float(retry_after_seconds))
    sleep_s = min(float(OPENAI_RETRY_MAX_SLEEP_SECONDS), max(0.0, sleep_s))
    time.sleep(sleep_s)


def _openai_response_text(data: Dict[str, Any]) -> str:
    text = ""
    if not isinstance(data, dict):
        return text
    direct = data.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct

    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or []
        if isinstance(content, str):
            text += content
            continue
        if not isinstance(content, list):
            continue
        for c in content:
            if isinstance(c, str):
                text += c
            elif isinstance(c, dict):
                if c.get("type") in ("output_text", "text", "message"):
                    text += c.get("text", "") or c.get("content", "") or ""
                elif "text" in c:
                    text += c.get("text") or ""
    return text


def _openai_error_retryable(exc: Exception) -> bool:
    if isinstance(exc, OpenAIHTTPError):
        if exc.status_code in (408, 409, 425, 429, 500, 502, 503, 504):
            return True
        body = (exc.response_text or "").lower()
        if any(x in body for x in ("rate_limit", "server_error", "temporarily", "overloaded")):
            return True
        return False
    return _http_retryable(exc)


def _openai_error_summary(exc: Exception) -> str:
    if isinstance(exc, OpenAIHTTPError):
        body = _preview_text(exc.response_text or "", 520)
        return f"OpenAI HTTP {exc.status_code}; retry_after={exc.retry_after_seconds}; body={body}"
    return _preview_text(str(exc), 520)


def _openai_candidate_chain(selected_model: str) -> List[str]:
    primary = (selected_model or OPENAI_MODEL_PRIMARY or "").strip()
    candidates: List[str] = []
    if primary:
        candidates.append(primary)
    secondary = (OPENAI_MODEL_SECONDARY or "").strip()
    if secondary and secondary not in candidates and primary != secondary:
        candidates.append(secondary)
    return candidates or [OPENAI_MODEL_PRIMARY]


def _openai_responses_once(prompt: str, model: str, *, max_output_tokens: int) -> Tuple[str, str]:
    key = (OPENAI_API_KEY or "").strip()
    if not key:
        raise RuntimeError("缺少 OPENAI_API_KEY（在 src/config.py 中配置），无法调用 OpenAI。")

    model = (model or "").strip()
    if not model:
        raise RuntimeError("OpenAI 模型名为空。")

    url = _openai_url()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": int(max_output_tokens),
    }
    with _openai_sem:
        proxy = get_proxy_url()
        try:
            client = httpx.Client(timeout=180.0, proxy=proxy) if proxy else httpx.Client(timeout=180.0)
        except TypeError:
            # Older httpx versions use `proxies=`
            client = (
                httpx.Client(timeout=180.0, proxies=proxy) if proxy else httpx.Client(timeout=180.0)
            )
        with client:
            r = client.post(url, headers=headers, json=payload)

    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        status = int(getattr(r, "status_code", 0) or 0)
        text = ""
        try:
            text = r.text or ""
        except Exception:
            text = ""
        retry_after = _parse_retry_after_seconds(r.headers)
        raise OpenAIHTTPError(
            f"OpenAI HTTP {status} for {url}. body={text[:2000]}",
            status_code=status,
            retry_after_seconds=retry_after,
            response_text=text,
        ) from e

    try:
        data = r.json()
    except Exception as e:
        raise RuntimeError(f"OpenAI 返回非 JSON 响应：{_preview_text(getattr(r, 'text', '') or '')}") from e
    text = _openai_response_text(data)
    if not text:
        raise RuntimeError(f"OpenAI 返回为空（未提取到 output_text）。response={_preview_text(str(data), 800)}")
    return text, f"openai:{model}"


def generate_answer_via_openai_responses(prompt: str, model: str, *, max_output_tokens: int) -> Tuple[str, str]:
    """OpenAI 单模型：重试（不含降级）。"""
    attempts = max(1, int(OPENAI_RETRY_MAX_ATTEMPTS or GEMINI_RETRY_MAX_ATTEMPTS))
    last_err: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            print(f"[openai] request model={model}, attempt={attempt}/{attempts}")
            return _openai_responses_once(prompt, model, max_output_tokens=max_output_tokens)
        except Exception as e:
            last_err = e
            print(
                f"[openai] error model={model}, attempt={attempt}/{attempts}, "
                f"retryable={_openai_error_retryable(e)}: {_openai_error_summary(e)}"
            )
            if attempt >= attempts or not _openai_error_retryable(e):
                break
            if isinstance(e, OpenAIHTTPError):
                _sleep_openai_backoff(attempt, e.retry_after_seconds)
            else:
                _sleep_backoff(attempt)
    raise last_err or RuntimeError("OpenAI 调用失败。")


def generate_answer_via_openai_with_fallback(
    prompt: str,
    primary_model: str,
    fallback_model: Optional[str],
    *,
    max_output_tokens: int,
) -> Tuple[str, str]:
    """
    OpenAI：按候选链重试并降级。每个模型会先完成自己的重试预算，再切换下一模型。
    """
    last_err: Optional[Exception] = None
    candidates = _openai_candidate_chain(primary_model)
    if fallback_model and fallback_model not in candidates:
        candidates.append(fallback_model)
    for model_name in candidates:
        attempts = max(1, int(OPENAI_RETRY_MAX_ATTEMPTS or GEMINI_RETRY_MAX_ATTEMPTS))
        for attempt in range(1, attempts + 1):
            try:
                print(f"[openai] request model={model_name}, attempt={attempt}/{attempts}")
                return _openai_responses_once(prompt, model_name, max_output_tokens=max_output_tokens)
            except Exception as e:
                last_err = e
                retryable = _openai_error_retryable(e)
                print(
                    f"[openai] error model={model_name}, attempt={attempt}/{attempts}, "
                    f"retryable={retryable}: {_openai_error_summary(e)}"
                )
                if attempt >= attempts or not retryable:
                    break
                if isinstance(e, OpenAIHTTPError):
                    _sleep_openai_backoff(attempt, e.retry_after_seconds)
                else:
                    _sleep_backoff(attempt)
        # 当前模型失败且还有下一个候选，就切换下一模型继续
        if model_name != candidates[-1]:
            print(
                f"[openai] switching fallback model: {model_name} -> "
                f"{candidates[candidates.index(model_name)+1]} after error: {_openai_error_summary(last_err) if last_err else 'unknown'}"
            )
    raise RuntimeError(f"OpenAI 调用失败。last_error={_openai_error_summary(last_err) if last_err else 'unknown'}") from last_err

def generate_answer_via_deepseek_chat(prompt: str, model: str) -> Tuple[str, str]:
    key = (DEEPSEEK_API_KEY or "").strip()
    if not key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY（在 src/config.py 中配置），无法调用 DeepSeek。")

    url = f"{DEEPSEEK_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    model = (model or "").strip()
    allowed = ("deepseek-v4-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner")
    if model not in allowed:
        raise RuntimeError(
            f"DeepSeek 模型名不合法：{model!r}。可用：{', '.join(allowed)}"
        )
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }

    attempts = max(1, int(GEMINI_RETRY_MAX_ATTEMPTS))
    last_err: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            proxy = get_proxy_url()
            try:
                client = httpx.Client(timeout=180.0, proxy=proxy) if proxy else httpx.Client(timeout=180.0)
            except TypeError:
                client = (
                    httpx.Client(timeout=180.0, proxies=proxy)
                    if proxy
                    else httpx.Client(timeout=180.0)
                )
            with client:
                r = client.post(url, headers=headers, json=payload)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = ""
                try:
                    body = r.text or ""
                except Exception:
                    body = ""
                raise RuntimeError(
                    f"DeepSeek HTTP {r.status_code} for {url}. body={body[:2000]}"
                ) from e
            data = r.json()
            text = ""
            if isinstance(data, dict):
                choices = data.get("choices") or []
                if choices and isinstance(choices, list) and isinstance(choices[0], dict):
                    msg = choices[0].get("message") or {}
                    if isinstance(msg, dict):
                        text = msg.get("content") or ""
            if not text:
                raise RuntimeError("DeepSeek 返回为空（未提取到 message.content）。")
            return text, f"deepseek:{model}"
        except Exception as e:
            last_err = e
            if attempt >= attempts or not _http_retryable(e):
                break
            _sleep_backoff(attempt)
    raise last_err or RuntimeError("DeepSeek 调用失败。")


def _auto_candidate_chain(selected_model: str = "") -> List[Tuple[str, str]]:
    selected = (selected_model or "").strip()
    if selected and selected.lower() != "auto":
        if ":" in selected:
            p, m = selected.split(":", 1)
            return [(p.strip().lower(), m.strip())]
        return [("gemini", selected)]
    candidates: List[Tuple[str, str]] = []
    for m in [GEMINI_ANSWER_MODEL] + _gemini_fallback_chain(GEMINI_ANSWER_MODEL):
        if m and ("gemini", m) not in candidates:
            candidates.append(("gemini", m))
    for m in [OPENAI_MODEL_PRIMARY, OPENAI_MODEL_SECONDARY]:
        if m and ("openai", m) not in candidates:
            candidates.append(("openai", m))
    for m in [DEEPSEEK_MODEL_PRIMARY, "deepseek-v4-pro", "deepseek-v4-flash"]:
        if m and ("deepseek", m) not in candidates:
            candidates.append(("deepseek", m))
    return candidates


def generate_answer_auto(
    *,
    prompt: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
    trace_id: str = "",
    stage: str = "",
) -> Tuple[str, str]:
    last_err: Optional[Exception] = None
    for provider, model_name in _auto_candidate_chain(model):
        if _is_auto_cooled(provider, model_name):
            print(f"[auto-llm] skip cooled model {provider}:{model_name}")
            continue
        try:
            print(f"[auto-llm] trying {provider}:{model_name}")
            text, used = generate_answer(
                prompt=prompt,
                provider=provider,
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                trace_id=trace_id,
                stage=stage,
            )
            if provider == "gemini" and used != f"gemini:{model_name}":
                _mark_auto_cooldown(provider, model_name, seconds=30 * 60)
            return text, used
        except Exception as e:
            last_err = e
            if is_quota_exhausted_error(e):
                _mark_auto_cooldown(provider, model_name)
                print(f"[auto-llm] quota cooldown {provider}:{model_name}: {e}")
                continue
            if _http_retryable(e):
                print(f"[auto-llm] retryable failure; switching candidate {provider}:{model_name}: {e}")
                continue
            print(f"[auto-llm] non-retryable failure; switching candidate {provider}:{model_name}: {e}")
            continue
    raise RuntimeError(f"Auto 模型调用失败。last_error={last_err}") from last_err


def generate_answer(
    *,
    prompt: str,
    provider: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
    trace_id: str = "",
    stage: str = "",
) -> Tuple[str, str]:
    """
    统一路由：返回 (text, provider:model_used)
    """
    p = (provider or "gemini").strip().lower()
    m_raw = (model or "").strip()

    if p == "auto" or m_raw.lower() == "auto":
        return generate_answer_auto(
            prompt=prompt,
            model=m_raw,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            trace_id=trace_id,
            stage=stage,
        )

    if p == "openai":
        # OpenAI：选定模型先重试；若配置了 secondary，则失败后降级。
        m = m_raw or OPENAI_MODEL_PRIMARY
        if OPENAI_MODEL_SECONDARY:
            return generate_answer_via_openai_with_fallback(
                prompt,
                m,
                OPENAI_MODEL_SECONDARY,
                max_output_tokens=max_output_tokens,
            )
        return generate_answer_via_openai_responses(
            prompt, m, max_output_tokens=max_output_tokens
        )
    if p == "deepseek":
        m = m_raw or DEEPSEEK_MODEL_PRIMARY
        return generate_answer_via_deepseek_chat(prompt, m)

    # gemini
    if m_raw:
        primary = m_raw
    else:
        primary = GEMINI_ANSWER_MODEL

    fallbacks = _gemini_fallback_chain(primary)

    st = (stage or "").strip() or "llm.gemini"
    text, used = generate_with_retry_and_fallback(
        prompt=prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        primary_model=primary,
        fallback_models=fallbacks,
        trace_id=trace_id,
        stage=st,
    )
    return text, f"gemini:{used}"

