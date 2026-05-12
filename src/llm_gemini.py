import errno
import random
import time
from typing import List, Optional, Tuple

import httpx
from google import genai

from .config import (
    GEMINI_API_KEY,
    GEMINI_RETRY_MAX_ATTEMPTS,
    GEMINI_RETRY_BASE_SECONDS,
    GEMINI_RETRY_JITTER_SECONDS,
)
from .net_proxy import apply_proxy_env

apply_proxy_env()

client = genai.Client(api_key=GEMINI_API_KEY)

def _norm_model_id(model: str) -> str:
    """
    Accept both forms:
    - gemini-3.1-pro-preview
    - models/gemini-3.1-pro-preview
    The google-genai client expects `models/...`.
    """
    m = (model or "").strip()
    if not m:
        return m
    if m.startswith("models/"):
        return m
    # tolerate users passing "gemini/..." or stray prefixes
    if m.startswith("/"):
        m = m.lstrip("/")
    return f"models/{m}"


def is_retryable_llm_error(exc: Exception) -> bool:
    """网络抖动、代理断连、服务端临时不可用等：应重试（仍可能最终失败）。"""
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.ConnectTimeout,
            httpx.RemoteProtocolError,
            httpx.ReadError,
            ConnectionError,
            TimeoutError,
        ),
    ):
        return True

    if isinstance(exc, OSError):
        we = getattr(exc, "winerror", None)
        if we in (10054, 10053, 10060, 10061):
            return True
        en = getattr(exc, "errno", None)
        if en in (
            errno.ECONNRESET,
            errno.ECONNREFUSED,
            errno.ETIMEDOUT,
            errno.EPIPE,
            errno.EHOSTUNREACH,
            errno.ENETUNREACH,
        ):
            return True

    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, Exception) and cause is not exc:
        if is_retryable_llm_error(cause):
            return True

    s = str(exc).lower()
    retry_marks = [
        "503",
        "unavailable",
        "high demand",
        "resource_exhausted",
        "429",
        "deadline exceeded",
        "timed out",
        "timeout",
        "connection reset",
        "10054",
        "winerror",
        "connecterror",
        "connection aborted",
        "broken pipe",
        "remoteprotocolerror",
        "server disconnected without sending a response",
        # 中文 Windows / 代理常见提示
        "远程主机",
        "强迫关闭",
        "无法连接",
    ]
    return any(m in s for m in retry_marks)


def describe_llm_error(exc: Exception) -> str:
    s = str(exc or "")
    low = s.lower()
    if not (GEMINI_API_KEY or "").strip():
        return "missing GEMINI_API_KEY"
    if "404" in low or "not found" in low or "invalid model" in low:
        return "model_not_found_or_invalid_id"
    if "failed_precondition" in low or "user location is not supported" in low:
        return "region_or_account_precondition_failed"
    if "permission" in low or "403" in low or "api key" in low:
        return "permission_or_api_key_rejected"
    if "429" in low or "resource_exhausted" in low or "quota" in low:
        return "quota_or_rate_limit"
    if "503" in low or "unavailable" in low or "high demand" in low or "overloaded" in low:
        return "service_unavailable_or_overloaded"
    if any(x in low for x in ("proxy", "connect", "connection", "timed out", "timeout", "10054", "10061", "远程主机", "无法连接")):
        return "network_or_proxy_failure"
    if "empty response text" in low:
        return "empty_response_text"
    return "unknown_gemini_error"


def is_quota_exhausted_error(exc: Exception) -> bool:
    """Hard quota/rate-limit failures should switch model instead of retrying locally."""
    s = str(exc or "").lower()
    if "resource_exhausted" not in s and "429" not in s and "quota" not in s:
        return False
    hard_marks = [
        "generate_requests_per_model_per_day",
        "generaterequestsperdayperprojectpermodel",
        "quota exceeded",
        "exceeded your current quota",
        "retrydelay",
        "retry in",
    ]
    return any(m in s for m in hard_marks)


def generate_with_retry_and_fallback(
    *,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    primary_model: str,
    fallback_models: Optional[List[str]] = None,
    trace_id: str = "",
    stage: str = "llm.gemini",
) -> Tuple[str, str]:
    candidates: List[str] = [primary_model] + [
        m for m in (fallback_models or []) if m and m != primary_model
    ]
    last_error: Optional[Exception] = None

    for model_name in candidates:
        attempts = max(1, int(GEMINI_RETRY_MAX_ATTEMPTS))
        for attempt in range(1, attempts + 1):
            try:
                t0 = time.time()
                model_id = _norm_model_id(model_name)
                if not (GEMINI_API_KEY or "").strip():
                    raise RuntimeError("缺少 GEMINI_API_KEY（在 src/config.py 中配置），无法调用 Gemini。")
                if trace_id:
                    try:
                        from .trace import log_stage

                        log_stage(
                            trace_id=trace_id,
                            stage=stage,
                            event="request",
                            extra={"model": model_id, "attempt": f"{attempt}/{attempts}"},
                        )
                    except Exception:
                        pass
                else:
                    print(
                        f"Sending request to Gemini... model={model_id}, attempt={attempt}/{attempts}"
                    )
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config={
                        "temperature": temperature,
                        "max_output_tokens": max_output_tokens,
                    },
                )
                text = getattr(response, "text", None)
                if not text:
                    raise RuntimeError("Gemini returned empty response text.")
                dt_ms = int((time.time() - t0) * 1000)
                if trace_id:
                    try:
                        from .trace import log_stage

                        log_stage(
                            trace_id=trace_id,
                            stage=stage,
                            event="response",
                            ms=dt_ms,
                            extra={"model": model_id, "chars": len(text or "")},
                        )
                    except Exception:
                        pass
                else:
                    print(f"Received response from Gemini model={model_id}")
                return text, model_name
            except Exception as e:
                last_error = e
                retryable = is_retryable_llm_error(e)
                diagnosis = describe_llm_error(e)
                hard_quota = is_quota_exhausted_error(e)
                if hard_quota:
                    if trace_id:
                        try:
                            from .trace import log_stage

                            log_stage(
                                trace_id=trace_id,
                                stage=stage,
                                event="quota_skip",
                                extra={
                                    "model": model_name,
                                    "attempt": f"{attempt}/{attempts}",
                                    "diagnosis": diagnosis,
                                    "err": str(e),
                                },
                            )
                        except Exception:
                            pass
                    else:
                        print(
                            f"Gemini quota exhausted model={model_name}; "
                            f"skip remaining retries and switch fallback: {e}"
                        )
                    break
                if (not retryable) or attempt >= attempts:
                    if trace_id:
                        try:
                            from .trace import log_stage

                            log_stage(
                                trace_id=trace_id,
                                stage=stage,
                                event="error",
                                extra={
                                    "model": model_name,
                                    "attempt": f"{attempt}/{attempts}",
                                    "retryable": retryable,
                                    "diagnosis": diagnosis,
                                    "err": str(e),
                                },
                            )
                        except Exception:
                            pass
                    else:
                        print(
                            f"Gemini call failed model={model_name}, attempt={attempt}/{attempts}, "
                            f"retryable={retryable}, diagnosis={diagnosis}: {e}"
                        )
                    break
                backoff = float(GEMINI_RETRY_BASE_SECONDS) * (2 ** (attempt - 1))
                jitter = random.uniform(0.0, float(GEMINI_RETRY_JITTER_SECONDS))
                sleep_s = backoff + jitter
                if trace_id:
                    try:
                        from .trace import log_stage

                        log_stage(
                            trace_id=trace_id,
                            stage=stage,
                            event="retry",
                            extra={"model": model_name, "sleep_s": f"{sleep_s:.2f}", "diagnosis": diagnosis, "err": str(e)},
                        )
                    except Exception:
                        pass
                else:
                    print(
                        f"Gemini transient error model={model_name}, diagnosis={diagnosis}, retry in {sleep_s:.2f}s: {e}"
                    )
                time.sleep(sleep_s)

        if model_name != candidates[-1]:
            nxt = candidates[candidates.index(model_name) + 1]
            if trace_id:
                try:
                    from .trace import log_stage

                    log_stage(
                        trace_id=trace_id,
                        stage=stage,
                        event="fallback",
                        extra={"from": model_name, "to": nxt},
                    )
                except Exception:
                    pass
            else:
                print(
                    f"Switching fallback model after {attempts} failed attempts: "
                    f"{model_name} -> {nxt}"
                )

    if last_error:
        raise last_error
    raise RuntimeError("Gemini request failed without explicit error.")

