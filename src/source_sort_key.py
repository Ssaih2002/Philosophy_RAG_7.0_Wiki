"""
文献名排序：中文 → 汉语拼音；日文 → 平假名（近似五十音序）；其它 → NFC + casefold。

说明：
- 文内同时含假名或常见日式标点（如 「」・）时，整串按日文读音规则处理。
- 纯汉字且无法判断语种时（无假名、无日文标点），按汉语拼音处理；纯日文汉字文件名可能被
  排在「中文拼音」区块中，这是无元数据时的固有歧义，可依赖文件名中补假名规避。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List

# 平假名 / 片假名 / 半角片假名 等
_KANA_RE = re.compile(
    r"[\u3040-\u309f"  # Hiragana
    r"\u30a0-\u30ff"  # Katakana
    r"\u31f0-\u31ff"  # Katakana phonetic extensions
    r"\uff66-\uff9f]"  # Halfwidth Katakana
)

# CJK 统一表意 + 兼容区（常用文件名足够）
_CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")

# 常见于日文题名，中文亦可能用但概率较低，用作「偏日文」弱信号
_JP_MARKERS = (
    "\u30fb",  # ・
    "\u300c",
    "\u300d",  # 「」
    "\u300e",
    "\u300f",  # 『』
    "\u3010",
    "\u3011",  # 【】
)


def _has_kana(s: str) -> bool:
    return bool(_KANA_RE.search(s))


def _has_cjk(s: str) -> bool:
    return bool(_CJK_RE.search(s))


def _has_jp_marker(s: str) -> bool:
    return any(m in s for m in _JP_MARKERS)


def _bucket(name: str) -> int:
    """
    分区（先排出的在前）：
      0 — 无 CJK、无假名：西文/数字/符号等，按 NFC+casefold。
      1 — 有汉字但无「日文线索」：按汉语拼音。
      2 — 含假名或常见日文标点：按平假名读音（五十音序）。
    纯日文汉字且无假名、无标点线索的文件会落在第 1 区（按拼音），见模块文档。
    """
    s = name or ""
    if _has_kana(s) or _has_jp_marker(s):
        return 2
    if _has_cjk(s):
        return 1
    return 0


def _pypinyin_key(s: str) -> str:
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError:
        return unicodedata.normalize("NFC", s).casefold()
    parts = lazy_pinyin(s, style=Style.NORMAL, errors=lambda frag: list(frag))
    return "".join(parts) if parts else ""


def _kakasi_hiragana_key(s: str) -> str:
    try:
        from pykakasi import kakasi as _kakasi_cls
    except ImportError:
        return unicodedata.normalize("NFC", s).casefold()
    kks = _kakasi_cls()
    out: List[str] = []
    for seg in kks.convert(s):
        out.append(str(seg.get("hira") or seg.get("orig") or ""))
    return "".join(out)


def _ascii_key(s: str) -> str:
    return unicodedata.normalize("NFC", s).casefold()


def _reading_sort_key(name: str) -> tuple:
    s = name or ""
    b = _bucket(s)
    if b == 2:
        return (b, _kakasi_hiragana_key(s), s)
    if b == 1:
        return (b, _pypinyin_key(s), s)
    return (b, _ascii_key(s), s)


def sort_library_sources_by_reading(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 中文拼音 / 日文五十音(平假名序) / 其它 排序文献列表。"""
    if not items:
        return items
    return sorted(items, key=lambda d: _reading_sort_key(d.get("source") or ""))
