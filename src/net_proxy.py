import os
from typing import Optional

from .config import HTTP_PROXY_URL, HTTPS_PROXY_URL, SOCKS_PROXY_URL


def apply_proxy_env() -> None:
    """
    Apply proxy settings for libraries that respect env vars (httpx, google-genai, etc).
    Only sets env vars when config provides a non-empty proxy URL.
    """
    http_p = (HTTP_PROXY_URL or "").strip()
    https_p = (HTTPS_PROXY_URL or "").strip()
    socks_p = (SOCKS_PROXY_URL or "").strip()
    if http_p:
        os.environ["HTTP_PROXY"] = http_p
        os.environ["http_proxy"] = http_p
    if https_p:
        os.environ["HTTPS_PROXY"] = https_p
        os.environ["https_proxy"] = https_p
    if socks_p:
        # Many HTTP stacks honor ALL_PROXY for socks proxy.
        os.environ["ALL_PROXY"] = socks_p
        os.environ["all_proxy"] = socks_p


def get_proxy_url() -> Optional[str]:
    """
    Return a single proxy URL for httpx.Client(proxy=...).
    Prefer HTTPS proxy if set.
    """
    https_p = (HTTPS_PROXY_URL or "").strip()
    if https_p:
        return https_p
    http_p = (HTTP_PROXY_URL or "").strip()
    return http_p or None

