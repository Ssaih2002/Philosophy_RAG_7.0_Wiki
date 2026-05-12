import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    ANSWER_STYLE_RETRIEVAL_OVERRIDES,
    GEMINI_ANSWER_TEMPERATURE,
    ANSWER_MAX_OUTPUT_TOKENS_DEFAULT,
    ANSWER_MAX_OUTPUT_TOKENS_ULTRA,
    CURRENT_PROFILE,
    PROFILE_SETTINGS,
    MAX_FINAL_K,
    MAX_KEYWORD_HITS,
    MIN_CHUNKS_PER_PRIMARY_SOURCE,
    PRIMARY_SOURCE_COUNT,
)
from .embedder import Embedder
from .vector_store import VectorStore
from .query_expander import expand_query
from .keyword_extractor import extract_keywords_from_question
from .sparse_retriever import SparseRetriever, build_sparse_query
from .hybrid_retrieval import weighted_reciprocal_rank_fusion
from .term_merger import merge_terms
from .reranker import CrossEncoderReranker
from .citation import build_context
from .academic_prompt import (
    STYLE_CITE_PATCH,
    STYLE_CONCEPT_MAP,
    STYLE_STANDARD_QA,
    STYLE_SEP,
    build_prompt,
    normalize_answer_style,
)
from .llm_router import generate_answer
from .standard_qa import answer_standard_qa
from .library_manager import normalize_library_id
from .memory_conflict import compute_conflict_hint
from .memory_store import append_event, load_recent_turns, prepare_history_block_for_prompt
from .wiki_manager import read_user_wiki, update_user_wiki_async
from .trace import log_stage
from . import config
from .config import (
    AUTO_B2_ENABLED,
    AUTO_B2_MIN_LIBRARY_LANG_SHARE,
    AUTO_B2_MAX_TARGET_LANGS,
    WIKI_LLM_PROVIDER,
    WIKI_LLM_MODEL,
)
from .library_language import load_cached_language_profile, top_languages
from .auto_b2 import generate_b2


def _split_concept_queries(question: str, merged_terms: List[str]) -> List[str]:
    """概念梳理模式：用关键词与子串检索，避免把整句当作单一扩写问题。"""
    out: List[str] = []
    seen: set = set()
    for t in merged_terms:
        t = (t or "").strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    q = (question or "").strip()
    if q:
        tmp = q
        for sep in [",", "，", "、", "\n", "\r", ";", "；", "|"]:
            tmp = tmp.replace(sep, "\n")
        for line in tmp.split("\n"):
            t = line.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
    return out


def replace_source_refs(text, docs):
    """
    Replace occurrences like 'Source 7' or '[Source 7]' in the model output
    with concrete citations '(filename, p. page)' using the retrieved docs.
    """

    def repl(match):
        idx_str = match.group(2)
        try:
            idx = int(idx_str) - 1
        except ValueError:
            return match.group(1)
        if 0 <= idx < len(docs):
            src = docs[idx].get("source", "Unknown")
            page = docs[idx].get("page", "Unknown")
            return f"({src}, p. {page})"
        return match.group(1)

    pattern = re.compile(r"(\[?\s*[Ss]ource\s+(\d+)\s*]?)")
    return pattern.sub(repl, text)


def sanitize_citations(text: str, docs: List[Dict[str, Any]]) -> str:
    """
    Keep only verifiable citations that exist in retrieved docs.
    Any unknown citation '(x, p. y)' is replaced to avoid fabricated references.
    """
    def _norm_src(s: str) -> str:
        return (s or "").strip().lower()

    def _norm_page(p: str) -> str:
        # Be tolerant to common formatting differences:
        # "12", "12 ", "p.12", "12-13", "12–13", "12—13", "12/13"
        s = (p or "").strip().lower()
        s = s.replace("pp.", "").replace("p.", "").replace("p ", "")
        s = s.replace("–", "-").replace("—", "-")
        s = re.sub(r"\s+", "", s)
        return s

    # Build per-source allowed pages to allow range matching.
    allowed_by_source: Dict[str, set] = {}
    for d in docs:
        src0 = str(d.get("source", "Unknown"))
        page0 = str(d.get("page", "Unknown"))
        src = _norm_src(src0)
        page = _norm_page(page0)
        if not src:
            continue
        allowed_by_source.setdefault(src, set()).add(page)

    pattern = re.compile(r"\(([^()]+),\s*p\.\s*([^)]+)\)")

    def _to_int(x: str) -> Optional[int]:
        m = re.search(r"\d+", x or "")
        if not m:
            return None
        try:
            return int(m.group(0))
        except Exception:
            return None

    def repl(match):
        src_raw = match.group(1).strip()
        page_raw = match.group(2).strip()

        src = _norm_src(src_raw)
        page = _norm_page(page_raw)
        allowed_pages = allowed_by_source.get(src, set())

        # If source doesn't match but only one source exists in evidence, align to it.
        if not allowed_pages and len(allowed_by_source) == 1:
            only_src = next(iter(allowed_by_source.keys()))
            allowed_pages = allowed_by_source.get(only_src, set())

        # Exact match
        if page in allowed_pages:
            return f"({src_raw}, p. {page_raw})"

        # Range / composite page tolerant match: accept if any token matches.
        # Examples: "12-13" matches "12" or "13"; "12/13" matches "12" etc.
        tokens = re.split(r"[-/]+", page) if page else []
        tokens = [t for t in tokens if t]
        if any(t in allowed_pages for t in tokens):
            return f"({src_raw}, p. {page_raw})"

        # Substring fallback: "12-13" should match stored "12" (or vice versa)
        for ap in allowed_pages:
            if not ap:
                continue
            if ap in page or page in ap:
                return f"({src_raw}, p. {page_raw})"

        # Numeric tolerance: allow +-1 page drift.
        p_int = _to_int(page)
        if p_int is not None:
            for ap in allowed_pages:
                a_int = _to_int(ap)
                if a_int is None:
                    continue
                if abs(a_int - p_int) <= 1:
                    return f"({src_raw}, p. {page_raw})"

        return "(unverified citation removed)"

    return pattern.sub(repl, text)


def _doc_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "text": row["text"],
        "page": row.get("page"),
        "source": row.get("source", "Unknown"),
        "chunk_id": row.get("chunk_id"),
    }


def _dedupe_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for d in docs:
        cid = d.get("chunk_id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(d)
    return out


def _enforce_source_coverage(
    base_docs: List[Dict[str, Any]],
    keyword_hit_docs: List[Dict[str, Any]],
    *,
    per_source_keep: int,
    source_count: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Force balanced source coverage from keyword hits.
    Select top N sources by hit count, then keep up to per_source_keep docs
    from each selected source, and prepend them to the final candidate list.
    """
    if per_source_keep <= 0 or source_count <= 0:
        return base_docs, {"enabled": False}

    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for d in keyword_hit_docs:
        src = d.get("source") or "Unknown"
        by_source.setdefault(src, []).append(d)
    if len(by_source) < source_count:
        return base_docs, {"enabled": False, "reason": "insufficient_sources"}

    ranked_sources = sorted(by_source.keys(), key=lambda s: len(by_source[s]), reverse=True)
    selected_sources = ranked_sources[:source_count]

    forced: List[Dict[str, Any]] = []
    forced_counts: Dict[str, int] = {}
    for src in selected_sources:
        picked = by_source[src][:per_source_keep]
        forced.extend(picked)
        forced_counts[src] = len(picked)

    merged = _dedupe_docs(forced + base_docs)
    return merged, {
        "enabled": True,
        "selected_sources": selected_sources,
        "per_source_keep": per_source_keep,
        "forced_counts": forced_counts,
        "forced_total": len(forced),
    }


def _balance_docs_by_source_filters(
    docs: List[Dict[str, Any]],
    filters: List[str],
    k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    多个「限定文献」子串时，按子串路数均分条数（每路内保持 docs 原有顺序）。
    每条文档只归入「命中的最长子串」对应的路，避免短子串抢占同属一路径的更长限定。
    """
    filters = [f.strip() for f in filters if f and str(f).strip()]
    if k <= 0 or not docs or len(filters) < 2:
        return docs[: max(0, k)], {"enabled": False}

    buckets: Dict[str, List[Dict[str, Any]]] = {f: [] for f in filters}
    rest: List[Dict[str, Any]] = []
    idx = {f: i for i, f in enumerate(filters)}
    for d in docs:
        src = d.get("source") or ""
        hits = [f for f in filters if f in src]
        if not hits:
            rest.append(d)
            continue
        # 最长子串优先：避免「短文件名」把另一条路径里更长、更具体的限定串全部吃掉
        # （例如先填 MEGA②I 再填 MEGA②II 时，后者路径里仍含前者子串）。
        best = max(hits, key=lambda f: (len(f), idx[f]))
        buckets[best].append(d)

    n = len(filters)
    base, rem = divmod(k, n)
    alloc = [base + (1 if i < rem else 0) for i in range(n)]

    out: List[Dict[str, Any]] = []
    actual: Dict[str, int] = {}
    for i, f in enumerate(filters):
        chunk = buckets[f][: alloc[i]]
        actual[f] = len(chunk)
        out.extend(chunk)

    if len(out) < k:
        more: List[Dict[str, Any]] = []
        for i, f in enumerate(filters):
            more.extend(buckets[f][alloc[i] :])
        more.extend(rest)
        seen = {d.get("chunk_id") for d in out if d.get("chunk_id")}
        for d in more:
            if len(out) >= k:
                break
            cid = d.get("chunk_id")
            if cid and cid in seen:
                continue
            if cid:
                seen.add(cid)
            out.append(d)
    return out[:k], {
        "enabled": True,
        "target_per_filter": dict(zip(filters, alloc)),
        "actual_per_filter": actual,
    }


def _detect_required_language(question: str) -> str:
    q = (question or "").strip()
    q_lower = q.lower()
    english_directives = (
        "answer in english",
        "respond in english",
        "reply in english",
        "use english",
        "in english",
    )
    chinese_directives = (
        "用中文",
        "中文回答",
        "简体中文",
        "用汉语",
        "请用中文",
    )
    if any(x in q_lower for x in english_directives):
        return "English"
    if any(x in q for x in chinese_directives):
        return "Simplified Chinese"

    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", q))
    latin_count = len(re.findall(r"[A-Za-z]", q))
    if latin_count and not cjk_count:
        return "English"
    if cjk_count and not latin_count:
        return "Simplified Chinese"
    if latin_count or cjk_count:
        # Mixed input often contains Chinese names/titles inside an English question.
        # Choose the dominant writing system instead of letting one CJK character force Chinese.
        if latin_count >= max(12, cjk_count * 2):
            return "English"
        if cjk_count >= max(4, latin_count):
            return "Simplified Chinese"
        return "English" if latin_count >= cjk_count else "Simplified Chinese"
    return "same as question"


def _empty_retrieval_message(required_language: str) -> str:
    if (required_language or "").strip().lower().startswith("english"):
        return (
            "[No retrieved excerpts] The current index did not return passages relevant to the input, "
            "so I cannot generate corpus-grounded citations or footnotes. Please confirm that Ingest has "
            "completed, check that the selected retrieval profile and `data` vector/sparse stores match the "
            "corpus, or retry with adjusted keywords / filename filters."
        )
    return (
        "【检索结果为空】当前索引中未检索到与输入相关的文献片段，"
        "无法从语料侧生成引用与脚注。建议：确认已完成 Ingest，检查当前检索配置（quality / fast）与 "
        "`data` 下语料、向量库路径是否一致，或调整关键词 / 限定文件名后重试。"
    )


def _detect_question_lang_code(question: str) -> str:
    """
    Coarse language code used for Auto+B2 routing: zh/en/de.
    """
    q = (question or "").strip()
    if re.search(r"[\u4e00-\u9fff]", q):
        return "zh"
    # very rough DE hint
    if any(ch in q for ch in ("ä", "ö", "ü", "ß", "Ä", "Ö", "Ü")):
        return "de"
    return "en"


class RAGEngine:
    def __init__(self):
        print("Initializing RAG system...")
        self._swap_lock = threading.Lock()
        self.profile = CURRENT_PROFILE
        self.params = dict(PROFILE_SETTINGS[self.profile])
        self.embedder = Embedder(self.params["EMBEDDING_MODEL"])
        # Cache per-library stores to avoid repeated heavy init (Chroma/SQLite open, schema checks).
        self._store_lock = threading.Lock()
        self._vs_cache: Dict[Tuple[str, str], VectorStore] = {}
        self._sparse_cache: Dict[Tuple[str, str], SparseRetriever] = {}
        # Default stores (backward compatible)
        self.db = VectorStore(self.profile)
        self.sparse = SparseRetriever(self.profile)
        self._vs_cache[(self.profile, "default")] = self.db
        self._sparse_cache[(self.profile, "default")] = self.sparse
        self.reranker = CrossEncoderReranker(self.params["RERANKER_MODEL"])

    def _get_vector_store(self, *, profile: str, library_id: str) -> VectorStore:
        key = (str(profile), normalize_library_id(library_id))
        with self._store_lock:
            v = self._vs_cache.get(key)
            if v is not None:
                return v
            v = VectorStore(key[0], library_id=key[1])
            self._vs_cache[key] = v
            # keep cache bounded (very small LRU-ish: drop an arbitrary oldest)
            if len(self._vs_cache) > 16:
                try:
                    self._vs_cache.pop(next(iter(self._vs_cache.keys())))
                except Exception:
                    pass
            return v

    def _get_sparse(self, *, profile: str, library_id: str) -> SparseRetriever:
        key = (str(profile), normalize_library_id(library_id))
        with self._store_lock:
            s = self._sparse_cache.get(key)
            if s is not None:
                return s
            s = SparseRetriever(key[0], library_id=key[1])
            self._sparse_cache[key] = s
            if len(self._sparse_cache) > 16:
                try:
                    self._sparse_cache.pop(next(iter(self._sparse_cache.keys())))
                except Exception:
                    pass
            return s

    def get_profile(self) -> str:
        return self.profile

    def get_profile_options(self) -> List[str]:
        return list(PROFILE_SETTINGS.keys())

    def switch_profile(self, profile: str) -> bool:
        if profile not in PROFILE_SETTINGS:
            return False
        self.profile = profile
        self.params = dict(PROFILE_SETTINGS[profile])
        self.embedder = Embedder(self.params["EMBEDDING_MODEL"])
        # Reset caches when profile changes to avoid mixing persistence dirs/collection names.
        with self._store_lock:
            self._vs_cache.clear()
            self._sparse_cache.clear()
        self.db = VectorStore(self.profile)
        self.sparse = SparseRetriever(self.profile)
        self._vs_cache[(self.profile, "default")] = self.db
        self._sparse_cache[(self.profile, "default")] = self.sparse
        self.reranker = CrossEncoderReranker(self.params["RERANKER_MODEL"])
        return True

    def _dense_docs(
        self,
        question: str,
        source_filters: Optional[List[str]] = None,
        *,
        db: Optional[VectorStore] = None,
        embedder: Optional[Embedder] = None,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        db = db or self.db
        embedder = embedder or self.embedder
        search_k = int(k if k is not None else self.params["SEARCH_K"])
        # NOTE: query expansion is handled at a higher level (parallel aux step).
        # Keep this method purely dense retrieval for a single query.
        queries = [question]
        seen = set()
        ordered: List[Dict[str, Any]] = []
        for q in queries:
            emb = embedder.encode([q])[0]
            results = db.search(
                emb,
                k=search_k,
                source_filters=source_filters,
            )
            for text, meta in zip(
                results["documents"][0],
                results["metadatas"][0],
            ):
                cid = meta.get("chunk_id")
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                ordered.append(
                    {
                        "chunk_id": cid,
                        "text": text,
                        "page": meta.get("page"),
                        "source": meta.get("source", "Unknown"),
                    }
                )
        return ordered

    def _dense_docs_from_queries(
        self,
        queries: List[str],
        source_filters: Optional[List[str]] = None,
        *,
        db: Optional[VectorStore] = None,
        embedder: Optional[Embedder] = None,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """对多个短查询分别做稠密检索并合并去重（用于概念梳理）。"""
        db = db or self.db
        embedder = embedder or self.embedder
        search_k = int(k if k is not None else self.params["SEARCH_K"])
        ordered: List[Dict[str, Any]] = []
        seen_ids: set = set()
        for q in queries:
            q = (q or "").strip()
            if not q:
                continue
            emb = embedder.encode([q])[0]
            results = db.search(
                emb,
                k=search_k,
                source_filters=source_filters,
            )
            for text, meta in zip(
                results["documents"][0],
                results["metadatas"][0],
            ):
                cid = meta.get("chunk_id")
                if not cid or cid in seen_ids:
                    continue
                seen_ids.add(cid)
                ordered.append(
                    {
                        "chunk_id": cid,
                        "text": text,
                        "page": meta.get("page"),
                        "source": meta.get("source", "Unknown"),
                    }
                )
        return ordered

    def _effective_retrieval_params(self, style_norm: str) -> Dict[str, Any]:
        p = dict(self.params)
        for key, val in ANSWER_STYLE_RETRIEVAL_OVERRIDES.get(style_norm, {}).items():
            p[key] = val
        return p

    def _resolve_keyword_terms(
        self,
        question: str,
        keyword_terms: Optional[List[str]],
        auto_extract_keywords: bool,
        *,
        max_terms: int = 12,
        auto_terms_override: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        user = [t.strip() for t in (keyword_terms or []) if t and str(t).strip()]
        auto: List[str] = []
        if auto_terms_override is not None:
            auto = [t.strip() for t in (auto_terms_override or []) if t and str(t).strip()]
        elif auto_extract_keywords:
            try:
                auto = extract_keywords_from_question(question)
            except Exception:
                auto = []
        merged = merge_terms(user_terms=user, auto_terms=auto, max_terms=max_terms)

        if merged["user_terms"] and merged["auto_terms"]:
            source = "merged"
        elif merged["user_terms"]:
            source = "user"
        elif merged["auto_terms"]:
            source = "auto"
        else:
            source = "question"

        return {
            "term_source": source,
            **merged,
        }

    def reload_sparse(self):
        # NOTE: with multi-library support, sparse is per-library; hot-reload is handled per request.
        self.sparse.reload()

    def _retrieve_impl(
        self,
        question: str,
        keyword_terms: Optional[List[str]] = None,
        source_filters: Optional[List[str]] = None,
        auto_extract_keywords: bool = True,
        use_hybrid: bool = True,
        use_rerank: bool = True,
        use_sep_reference: bool = False,
        answer_style: str = "哲学论述",
        retrieval_ablation: Optional[str] = None,
        retrieval_final_k_override: Optional[int] = None,
        *,
        library_id: str = "default",
        # Multi-library Auto+B2 can pass these to avoid per-library B2 calls.
        dense_queries_override: Optional[List[str]] = None,
        extra_sparse_terms: Optional[List[str]] = None,
        auto_b2_meta_override: Optional[Dict[str, Any]] = None,
        trace_id: str = "",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        library_id = normalize_library_id(library_id)
        db = self._get_vector_store(profile=self.profile, library_id=library_id)
        sparse = self._get_sparse(profile=self.profile, library_id=library_id)

        style_norm = normalize_answer_style(answer_style)
        rp = self._effective_retrieval_params(style_norm)
        rrf_dense_w = float(rp.get("RRF_DENSE_WEIGHT", 1.0))
        rrf_sparse_w = float(rp.get("RRF_SPARSE_WEIGHT", 1.0))
        # 评测消融：禁用 SEP，避免与主实验混淆
        if retrieval_ablation:
            use_sep_reference = False
        normalized_sources = [
            s.strip() for s in (source_filters or []) if s and str(s).strip()
        ]
        term_budget = 22 if style_norm == STYLE_CONCEPT_MAP else 12
        # --- Parallel aux steps (do NOT reduce quality) ---
        # 1) keyword extraction (LLM) 2) query expansion (LLM) 3) Auto+B2 (LLM, conditional)
        aux_timeout = float(getattr(config, "AUTO_B2_AUX_TIMEOUT_SECONDS", 2.5) or 2.5)
        # keep a small buffer for keyword/query expansion; they can be a bit slower than b2 sometimes
        kw_timeout = max(1.5, aux_timeout + 0.8)
        ex_timeout = max(1.5, aux_timeout + 0.8)

        auto_terms: Optional[List[str]] = None
        expanded_queries: Optional[List[str]] = None

        # Prepare Auto+B2 submission decision (lightweight, no LLM yet)
        b2_should_run = False
        b2_target_langs: List[str] = []
        b2_lp_dist: Optional[Dict[str, Any]] = None
        q_lang = ""
        if (
            bool(AUTO_B2_ENABLED)
            and not retrieval_ablation
            and style_norm not in (STYLE_STANDARD_QA, STYLE_SEP)
            and not dense_queries_override
        ):
            try:
                q_lang = _detect_question_lang_code(question)
                lp = load_cached_language_profile(self.profile, library_id, data_dir="data")
                if lp and lp.dist:
                    tops = top_languages(
                        lp.dist,
                        min_share=float(AUTO_B2_MIN_LIBRARY_LANG_SHARE),
                        max_langs=int(AUTO_B2_MAX_TARGET_LANGS),
                    )
                    b2_target_langs = [x for x in tops if x and x != q_lang]
                    if b2_target_langs:
                        b2_should_run = True
                        b2_lp_dist = lp.dist
            except Exception:
                b2_should_run = False

        def _safe_expand() -> List[str]:
            return expand_query(question)

        def _safe_keywords() -> List[str]:
            return extract_keywords_from_question(question)

        def _safe_b2():
            return generate_b2(question=question, target_langs=b2_target_langs)

        b2 = None
        auto_b2_meta: Dict[str, Any] = dict(auto_b2_meta_override or {"enabled": False})
        auto_b2_preview: Dict[str, Any] = {}

        with ThreadPoolExecutor(max_workers=3) as ex:
            fut_kw = None
            fut_qx = None
            fut_b2 = None
            if auto_extract_keywords:
                if trace_id:
                    log_stage(trace_id=trace_id, stage="retrieve.keyword_extract", event="start")
                fut_kw = ex.submit(_safe_keywords)
            if style_norm != STYLE_CONCEPT_MAP:
                # concept mode uses different splitting; expansion is less useful there
                if trace_id:
                    log_stage(trace_id=trace_id, stage="retrieve.query_expand", event="start")
                fut_qx = ex.submit(_safe_expand)
            if b2_should_run:
                if trace_id:
                    log_stage(
                        trace_id=trace_id,
                        stage="retrieve.auto_b2",
                        event="start",
                        extra={"q_lang": q_lang, "target_langs": ",".join(b2_target_langs)},
                    )
                fut_b2 = ex.submit(_safe_b2)

            # collect keyword extraction
            if fut_kw is not None:
                t0 = time.time()
                try:
                    auto_terms = fut_kw.result(timeout=kw_timeout)
                except FutureTimeoutError:
                    auto_terms = []
                except Exception:
                    auto_terms = []
                if trace_id:
                    log_stage(
                        trace_id=trace_id,
                        stage="retrieve.keyword_extract",
                        event="ok",
                        ms=int((time.time() - t0) * 1000),
                        extra={"n": len(auto_terms or [])},
                    )

            # collect query expansion
            if fut_qx is not None:
                t0 = time.time()
                try:
                    expanded_queries = fut_qx.result(timeout=ex_timeout)
                except FutureTimeoutError:
                    expanded_queries = [question]
                except Exception:
                    expanded_queries = [question]
                # ensure original question is included and keep it bounded
                expanded_queries = list(dict.fromkeys([question] + list(expanded_queries or [])))[:12]
                if trace_id:
                    log_stage(
                        trace_id=trace_id,
                        stage="retrieve.query_expand",
                        event="ok",
                        ms=int((time.time() - t0) * 1000),
                        extra={"n": len(expanded_queries or [])},
                    )

            # collect Auto+B2
            if fut_b2 is not None:
                t0 = time.time()
                try:
                    b2 = fut_b2.result(timeout=max(1.0, aux_timeout + 0.5))
                except FutureTimeoutError:
                    b2 = None
                    auto_b2_meta = {"enabled": False, "error": "timeout"}
                except Exception:
                    b2 = None
                    auto_b2_meta = {"enabled": False, "error": "auto_b2_failed"}
                if b2 and (getattr(b2, "dense_queries", None) or getattr(b2, "sparse_terms", None)):
                    auto_b2_meta = {
                        "enabled": True,
                        "question_lang": q_lang,
                        "library_lang_profile": b2_lp_dist,
                        "target_langs": list(getattr(b2, "target_langs", b2_target_langs) or b2_target_langs),
                        "aux_model_used": getattr(b2, "model_used", ""),
                        "cached": bool(getattr(b2, "cached", False)),
                        "dense_queries_count": len(getattr(b2, "dense_queries", []) or []),
                        "sparse_terms_count": len(getattr(b2, "sparse_terms", []) or []),
                    }
                    auto_b2_preview = {
                        "dense_queries": list(dict.fromkeys([question] + list(getattr(b2, "dense_queries", []) or [])))[:12],
                        "sparse_terms": list(getattr(b2, "sparse_terms", []) or [])[:24],
                        "glossary": dict(list((getattr(b2, "glossary", {}) or {}).items())[:18]),
                    }
                if trace_id:
                    log_stage(
                        trace_id=trace_id,
                        stage="retrieve.auto_b2",
                        event="ok",
                        ms=int((time.time() - t0) * 1000),
                        extra={"enabled": bool(auto_b2_meta.get("enabled")), "cached": bool(auto_b2_meta.get("cached"))},
                    )

        # Merge keyword terms using the (possibly) precomputed auto terms
        term_info = self._resolve_keyword_terms(
            question,
            keyword_terms,
            auto_extract_keywords,
            max_terms=term_budget,
            auto_terms_override=auto_terms,
        )
        merged_terms = term_info["merged_terms"]
        term_source = term_info["term_source"]
        concept_cq: List[str] = []
        sparse_terms_for_fts = merged_terms

        # Apply extra sparse terms (from multi-library B2 reuse) and local Auto+B2 results.
        if extra_sparse_terms:
            merged_terms = list(dict.fromkeys(list(merged_terms) + list(extra_sparse_terms)))[: max(12, term_budget)]
            sparse_terms_for_fts = merged_terms
        if b2 and getattr(b2, "sparse_terms", None):
            merged_terms = list(dict.fromkeys(list(merged_terms) + list(getattr(b2, "sparse_terms", []) or [])))[: max(12, term_budget)]
            sparse_terms_for_fts = merged_terms
        # Dense query override precedence:
        # caller override > Auto+B2 dense queries > query expansion
        if dense_queries_override:
            auto_b2_meta.setdefault("enabled", True)
        elif b2 and getattr(b2, "dense_queries", None):
            dense_queries_override = list(dict.fromkeys([question] + list(getattr(b2, "dense_queries", []) or [])))[:12]
        elif expanded_queries:
            dense_queries_override = list(expanded_queries)

        if style_norm == STYLE_CONCEPT_MAP:
            concept_cq = _split_concept_queries(question, merged_terms)
            seen_ft = set()
            fts_list: List[str] = []
            for t in list(merged_terms) + list(concept_cq):
                tt = (t or "").strip()
                if not tt or tt.lower() in seen_ft:
                    continue
                seen_ft.add(tt.lower())
                fts_list.append(tt)
            sparse_terms_for_fts = fts_list[:36]
            fallback_q = " ".join(concept_cq) if concept_cq else (question or "")
            keyword_query = build_sparse_query(sparse_terms_for_fts, fallback_q)
            dense_list = []
            if concept_cq:
                t0 = time.time()
                dense_list = self._dense_docs_from_queries(
                    concept_cq,
                    source_filters=normalized_sources,
                    db=db,
                    embedder=self.embedder,
                    k=rp["SEARCH_K"],
                )
                if trace_id:
                    log_stage(
                        trace_id=trace_id,
                        stage="retrieve.dense",
                        event="ok",
                        ms=int((time.time() - t0) * 1000),
                        extra={"mode": "concept_queries", "n": len(dense_list), "k": int(rp["SEARCH_K"])},
                    )
        else:
            keyword_query = build_sparse_query(merged_terms, question)
            if dense_queries_override:
                t0 = time.time()
                dense_list = self._dense_docs_from_queries(
                    dense_queries_override,
                    source_filters=normalized_sources,
                    db=db,
                    embedder=self.embedder,
                    k=rp["SEARCH_K"],
                )
                if trace_id:
                    log_stage(
                        trace_id=trace_id,
                        stage="retrieve.dense",
                        event="ok",
                        ms=int((time.time() - t0) * 1000),
                        extra={"mode": "queries_override", "n": len(dense_list), "k": int(rp["SEARCH_K"])},
                    )
            else:
                t0 = time.time()
                dense_list = self._dense_docs(
                    question,
                    source_filters=normalized_sources,
                    db=db,
                    embedder=self.embedder,
                    k=rp["SEARCH_K"],
                )
                if trace_id:
                    log_stage(
                        trace_id=trace_id,
                        stage="retrieve.dense",
                        event="ok",
                        ms=int((time.time() - t0) * 1000),
                        extra={"mode": "single", "n": len(dense_list), "k": int(rp["SEARCH_K"])},
                    )
        dense_ids = [d["chunk_id"] for d in dense_list if d.get("chunk_id")]
        dense_by_id = {d["chunk_id"]: d for d in dense_list if d.get("chunk_id")}

        # Optional SEP reference (weak corpus): retrieve from sep profile and blend with small weight.
        sep_weight = 0.35
        sep_max_docs = 6
        sep_search_k = 16
        sep_dense_ids: List[str] = []
        sep_by_id: Dict[str, Dict[str, Any]] = {}
        if use_sep_reference:
            try:
                sep_params = dict(PROFILE_SETTINGS.get("sep") or {})
                sep_embed_model = str(sep_params.get("EMBEDDING_MODEL") or self.params.get("EMBEDDING_MODEL"))
                # Reuse main embedder when model matches to avoid double heavy loads.
                sep_embedder = self.embedder if sep_embed_model == self.embedder.model_name else Embedder(sep_embed_model)
                sep_db = self._get_vector_store(profile="sep", library_id="default")
                # Same expansion logic as main dense, but only keep ids + text/meta from results.
                for q in expand_query(question):
                    emb = sep_embedder.encode([q])[0]
                    results = sep_db.search(emb, k=int(sep_search_k), source_filters=None)
                    for text, meta in zip(results["documents"][0], results["metadatas"][0]):
                        cid0 = meta.get("chunk_id") or meta.get("id") or meta.get("chunk")  # tolerate variations
                        if not cid0:
                            continue
                        cid = f"sep::{cid0}"
                        if cid in sep_by_id:
                            continue
                        sep_by_id[cid] = {
                            "chunk_id": cid,
                            "text": text,
                            "page": meta.get("page"),
                            "source": meta.get("source", "SEP"),
                            "_is_sep": True,
                        }
                sep_dense_ids = list(sep_by_id.keys())
            except Exception:
                # If SEP DB not present / broken, just skip it silently.
                sep_dense_ids = []
                sep_by_id = {}

        sparse_ids: List[str] = []
        keyword_hit_docs: List[Dict[str, Any]] = []
        if use_hybrid and sparse.is_ready():
            t0 = time.time()
            keyword_hits_full = sparse.search(
                keyword_query,
                k=MAX_KEYWORD_HITS,
                source_filters=normalized_sources,
            )
            if trace_id:
                log_stage(
                    trace_id=trace_id,
                    stage="retrieve.sparse",
                    event="ok",
                    ms=int((time.time() - t0) * 1000),
                    extra={"hits": len(keyword_hits_full), "k": int(MAX_KEYWORD_HITS), "ready": True},
                )
            sparse_ids = [
                h["chunk_id"]
                for h in keyword_hits_full[: rp["SPARSE_K"]]
                if h.get("chunk_id")
            ]
            keyword_hit_docs = [_doc_from_row(h) for h in keyword_hits_full]
        keyword_source_counter = Counter(
            (d.get("source") or "Unknown") for d in keyword_hit_docs
        )
        keyword_source_stats = [
            {"source": src, "count": int(cnt)}
            for src, cnt in keyword_source_counter.most_common()
        ]

        hybrid_ok = bool(use_hybrid and sparse.is_ready() and sparse_ids)
        fused: List[Any] = []

        if retrieval_ablation == "dense_only":
            dense_cap = int(rp.get("HYBRID_TOP_N") or 0)
            if dense_cap <= 0:
                dense_cap = int(rp.get("SEARCH_K") or 0)
            if dense_cap <= 0:
                dense_cap = len(dense_ids)
            top_ids = dense_ids[:dense_cap]
        elif retrieval_ablation == "sparse_only":
            sc = int(rp.get("SPARSE_K") or 0) or 34
            top_ids = sparse_ids[:sc]
        elif retrieval_ablation == "merge_no_rrf":
            cap = int(rp.get("HYBRID_TOP_N") or 0) or 28
            if sparse_ids:
                top_ids = list(dict.fromkeys(sparse_ids + dense_ids))[:cap]
            else:
                top_ids = dense_ids[:cap] if dense_ids else []
        elif hybrid_ok:
            if use_sep_reference and sep_dense_ids:
                t0 = time.time()
                fused = weighted_reciprocal_rank_fusion(
                    [dense_ids, sparse_ids, sep_dense_ids],
                    [rrf_dense_w, rrf_sparse_w, float(sep_weight)],
                    rrf_k=rp["RRF_K"],
                )
                if trace_id:
                    log_stage(
                        trace_id=trace_id,
                        stage="retrieve.rrf",
                        event="ok",
                        ms=int((time.time() - t0) * 1000),
                        extra={"lists": 3, "top_n": int(rp["HYBRID_TOP_N"])},
                    )
            else:
                t0 = time.time()
                fused = weighted_reciprocal_rank_fusion(
                    [dense_ids, sparse_ids],
                    [rrf_dense_w, rrf_sparse_w],
                    rrf_k=rp["RRF_K"],
                )
                if trace_id:
                    log_stage(
                        trace_id=trace_id,
                        stage="retrieve.rrf",
                        event="ok",
                        ms=int((time.time() - t0) * 1000),
                        extra={"lists": 2, "top_n": int(rp["HYBRID_TOP_N"])},
                    )
            fused_ids = [cid for cid, _ in fused[: rp["HYBRID_TOP_N"]]]
            # RRF 融合序优先（语义+词面联合），再补纯向量 / 纯关键词召回，避免 Full 与 sparse_only 仅差「前缀顺序」。
            top_ids = list(dict.fromkeys(fused_ids + dense_ids + sparse_ids))
        else:
            if use_sep_reference and sep_dense_ids:
                fused = weighted_reciprocal_rank_fusion(
                    [dense_ids, sep_dense_ids],
                    [1.0, float(sep_weight)],
                    rrf_k=rp["RRF_K"],
                )
                top_ids = [cid for cid, _ in fused[: rp["HYBRID_TOP_N"]]]
                # Keep any remaining main dense ids as tail for coverage.
                top_ids = list(dict.fromkeys(top_ids + dense_ids))
            else:
                fused = []
                # Non-hybrid path: use HYBRID_TOP_N as a dense candidate cap when set,
                # otherwise fall back to SEARCH_K (SEP profile sets HYBRID_TOP_N=0).
                dense_cap = int(rp.get("HYBRID_TOP_N") or 0)
                if dense_cap <= 0:
                    dense_cap = int(rp.get("SEARCH_K") or 0)
                if dense_cap <= 0:
                    dense_cap = len(dense_ids)
                top_ids = dense_ids[:dense_cap]

        # If SEP reference is enabled, ensure a small number of SEP candidates
        # are considered before truncation/rerank. This keeps SEP as a weak
        # reference (still capped by sep_max_docs) while avoiding the common case
        # where all early candidates are swallowed by main dense hits.
        if use_sep_reference and sep_dense_ids:
            try:
                need = max(6, int(sep_max_docs) * 2)
            except Exception:
                need = 12
            top_ids = list(dict.fromkeys(sep_dense_ids[:need] + top_ids))

        docs: List[Dict[str, Any]] = []
        sep_kept = 0
        for cid in top_ids:
            if cid.startswith("sep::"):
                if sep_kept >= int(sep_max_docs):
                    continue
                d = sep_by_id.get(cid)
                if d:
                    docs.append(_doc_from_row(d))
                    sep_kept += 1
                continue
            row = sparse.get_doc(cid)
            if row:
                docs.append(_doc_from_row(row))
            elif cid in dense_by_id:
                docs.append(_doc_from_row(dense_by_id[cid]))

        coverage_meta = {"enabled": False}
        if keyword_hit_docs:
            docs, coverage_meta = _enforce_source_coverage(
                docs,
                keyword_hit_docs,
                per_source_keep=MIN_CHUNKS_PER_PRIMARY_SOURCE,
                source_count=PRIMARY_SOURCE_COUNT,
            )

        baseline_final_k = max(1, min(int(rp["FINAL_K"]), int(MAX_FINAL_K)))
        forced_need = 0
        if coverage_meta.get("enabled"):
            forced_need = int(MIN_CHUNKS_PER_PRIMARY_SOURCE) * int(PRIMARY_SOURCE_COUNT)
        effective_final_k = max(baseline_final_k, forced_need)
        effective_final_k = min(effective_final_k, int(MAX_FINAL_K))
        # 评测 run_retrieval_only 需要超过「哲学论述」FINAL_K 的条数时，抬高最终截断上限（仍受 MAX_FINAL_K 约束）
        if retrieval_final_k_override is not None:
            o = max(1, min(int(retrieval_final_k_override), int(MAX_FINAL_K)))
            effective_final_k = max(effective_final_k, o)
        rerank_enabled = bool(use_rerank and len(docs) > 1)
        rerank_question = question
        if style_norm == STYLE_CONCEPT_MAP:
            rerank_question = (
                " ".join(sparse_terms_for_fts)
                if sparse_terms_for_fts
                else (question or "").strip() or keyword_query
            )
        multi_sf = bool(normalized_sources and len(normalized_sources) >= 2)
        src_balance_meta: Dict[str, Any] = {"enabled": False}
        if rerank_enabled:
            try:
                cand_pre = min(
                    len(docs),
                    max(
                        int(rp["RERANK_CANDIDATES"]),
                        effective_final_k * max(2, len(normalized_sources)),
                    ),
                )
                to_rerank = docs[:cand_pre]
                if multi_sf:
                    to_rerank, src_balance_meta = _balance_docs_by_source_filters(
                        docs, normalized_sources, cand_pre
                    )
                t0 = time.time()
                docs = self.reranker.rerank(
                    question=rerank_question,
                    docs=to_rerank,
                    top_k=effective_final_k,
                )
                if trace_id:
                    log_stage(
                        trace_id=trace_id,
                        stage="retrieve.rerank",
                        event="ok",
                        ms=int((time.time() - t0) * 1000),
                        extra={"cands": len(to_rerank), "top_k": int(effective_final_k)},
                    )
            except Exception:
                rerank_enabled = False
                if multi_sf:
                    docs, src_balance_meta = _balance_docs_by_source_filters(
                        docs, normalized_sources, effective_final_k
                    )
                else:
                    docs = docs[:effective_final_k]
        else:
            if multi_sf:
                docs, src_balance_meta = _balance_docs_by_source_filters(
                    docs, normalized_sources, effective_final_k
                )
            else:
                docs = docs[:effective_final_k]

        eff_keywords = (
            sparse_terms_for_fts if style_norm == STYLE_CONCEPT_MAP else merged_terms
        )
        meta = {
            "profile": self.profile,
            "retrieval_ablation": retrieval_ablation,
            "keywords_used": eff_keywords,
            "source_filters_used": normalized_sources,
            "user_terms_used": term_info["user_terms"],
            "auto_terms_used": term_info["auto_terms"],
            "dropped_terms": term_info["dropped_terms"],
            "keyword_query": keyword_query,
            "term_source": term_source,
            "hybrid": hybrid_ok,
            "reranked": rerank_enabled,
            "sep_reference_enabled": bool(use_sep_reference and bool(sep_dense_ids)),
            "sep_weight": float(sep_weight),
            "sep_max_docs": int(sep_max_docs),
            "sep_docs_kept": int(sep_kept),
            "keyword_hit_docs": keyword_hit_docs,
            "keyword_source_stats": keyword_source_stats,
            "coverage_enforced": bool(coverage_meta.get("enabled", False)),
            "answer_style_canonical": style_norm,
            "debug": {
                "profile": self.profile,
                "search_k": rp["SEARCH_K"],
                "final_k": effective_final_k,
                "retrieval_final_k_override": retrieval_final_k_override,
                "final_k_configured": rp["FINAL_K"],
                "final_k_cap": MAX_FINAL_K,
                "retrieval_params_effective": {
                    "SEARCH_K": rp["SEARCH_K"],
                    "FINAL_K": rp["FINAL_K"],
                    "SPARSE_K": rp["SPARSE_K"],
                    "HYBRID_TOP_N": rp["HYBRID_TOP_N"],
                    "RERANK_CANDIDATES": rp["RERANK_CANDIDATES"],
                    "RRF_DENSE_WEIGHT": rrf_dense_w,
                    "RRF_SPARSE_WEIGHT": rrf_sparse_w,
                },
                "concept_dense_queries": concept_cq
                if style_norm == STYLE_CONCEPT_MAP
                else [],
                "auto_b2": auto_b2_meta,
                "auto_b2_preview": auto_b2_preview,
                "source_filters": normalized_sources,
                "library_id": library_id,
                "chroma_collection": db.collection_name,
                "chroma_path": db.db_path,
                "sparse_db_path": sparse.path,
                "dense_top_ids": dense_ids[:12],
                "sparse_top_ids": sparse_ids[:12],
                "fused_top_ids": [cid for cid, _ in fused[:12]],
                "sep_dense_top_ids": sep_dense_ids[:12],
                "sep_docs_kept": int(sep_kept),
                "retrieved_before_rerank": len(top_ids),
                "final_docs": len(docs),
                "keyword_hits_count": len(keyword_hit_docs),
                "keyword_hits_cap": MAX_KEYWORD_HITS,
                "keyword_source_stats_top10": keyword_source_stats[:10],
                "coverage": coverage_meta,
                "source_filter_balance": src_balance_meta,
            },
        }
        return docs, meta

    def retrieve(
        self,
        question: str,
        keyword_terms: Optional[List[str]] = None,
        source_filters: Optional[List[str]] = None,
        auto_extract_keywords: bool = True,
        use_hybrid: bool = True,
        use_rerank: bool = True,
        use_sep_reference: bool = False,
        answer_style: str = "哲学论述",
        retrieval_ablation: Optional[str] = None,
        retrieval_final_k_override: Optional[int] = None,
        *,
        library_ids: Optional[List[str]] = None,
        library_weights: Optional[List[float]] = None,
        trace_id: str = "",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        # Single-library default (backward compatible)
        lids = [normalize_library_id(x) for x in (library_ids or []) if x and str(x).strip()]
        if not lids:
            return self._retrieve_impl(
                question,
                keyword_terms=keyword_terms,
                source_filters=source_filters,
                auto_extract_keywords=auto_extract_keywords,
                use_hybrid=use_hybrid,
                use_rerank=use_rerank,
                use_sep_reference=use_sep_reference,
                answer_style=answer_style,
                retrieval_ablation=retrieval_ablation,
                retrieval_final_k_override=retrieval_final_k_override,
                library_id="default",
                trace_id=trace_id,
            )

        # Multi-library: per-library retrieve (no rerank), then weighted RRF fuse, then optional global rerank.
        wts: List[float] = []
        if library_weights and len(library_weights) == len(lids):
            wts = [float(x) for x in library_weights]
        else:
            wts = [1.0 for _ in lids]

        per_docs: List[List[Dict[str, Any]]] = []
        per_meta: List[Dict[str, Any]] = []
        doc_by_ns: Dict[str, Dict[str, Any]] = {}
        ranked_lists: List[List[str]] = []

        # Candidate budget: take more before global rerank/truncation
        style_norm = normalize_answer_style(answer_style)
        rp = self._effective_retrieval_params(style_norm)
        cand_cap = int(rp.get("RERANK_CANDIDATES") or 40)
        cand_cap = max(cand_cap, int(rp.get("HYBRID_TOP_N") or 0) or cand_cap)
        cand_cap = min(cand_cap, int(MAX_FINAL_K))

        # Multi-library Auto+B2: compute once per target language group, then reuse per library.
        auto_b2_multi: Dict[str, Any] = {"enabled": False}
        auto_b2_multi_preview: Dict[str, Any] = {}
        b2_by_langs: Dict[Tuple[str, ...], Any] = {}
        per_lib_b2_meta: Dict[str, Any] = {}
        try:
            style_norm = normalize_answer_style(answer_style)
            if (
                bool(AUTO_B2_ENABLED)
                and not retrieval_ablation
                and style_norm not in (STYLE_STANDARD_QA, STYLE_SEP)
            ):
                q_lang = _detect_question_lang_code(question)
                for lid in lids:
                    lp = load_cached_language_profile(self.profile, lid, data_dir="data")
                    if not lp or not lp.dist:
                        continue
                    tops = top_languages(
                        lp.dist,
                        min_share=float(AUTO_B2_MIN_LIBRARY_LANG_SHARE),
                        max_langs=int(AUTO_B2_MAX_TARGET_LANGS),
                    )
                    target_langs = tuple([x for x in tops if x and x != q_lang])
                    if not target_langs:
                        continue
                    if target_langs not in b2_by_langs:
                        b2_by_langs[target_langs] = generate_b2(
                            question=question,
                            target_langs=list(target_langs),
                        )
                    b2 = b2_by_langs.get(target_langs)
                    per_lib_b2_meta[lid] = {
                        "question_lang": q_lang,
                        "library_lang_profile": lp.dist,
                        "target_langs": list(target_langs),
                        "enabled": bool(b2 and (b2.dense_queries or b2.sparse_terms)),
                        "cached": bool(getattr(b2, "cached", False)) if b2 else False,
                        "aux_model_used": getattr(b2, "model_used", "") if b2 else "",
                    }
                auto_b2_multi = {
                    "enabled": bool(per_lib_b2_meta),
                    "question_lang": q_lang,
                    "groups": {
                        ",".join(k): {
                            "enabled": bool(v and (v.dense_queries or v.sparse_terms)),
                            "cached": bool(getattr(v, "cached", False)) if v else False,
                            "aux_model_used": getattr(v, "model_used", "") if v else "",
                            "dense_queries_count": len(getattr(v, "dense_queries", []) or []) if v else 0,
                            "sparse_terms_count": len(getattr(v, "sparse_terms", []) or []) if v else 0,
                        }
                        for k, v in b2_by_langs.items()
                    },
                }
                # Preview for UI/debug (bounded; keyed by lang group)
                auto_b2_multi_preview = {
                    ",".join(k): {
                        "dense_queries": list(dict.fromkeys([question] + list(getattr(v, "dense_queries", []) or [])))[:12]
                        if v
                        else [],
                        "sparse_terms": list(getattr(v, "sparse_terms", []) or [])[:24] if v else [],
                        "glossary": dict(list((getattr(v, "glossary", {}) or {}).items())[:18]) if v else {},
                    }
                    for k, v in b2_by_langs.items()
                }
        except Exception:
            auto_b2_multi = {"enabled": False, "error": "auto_b2_multi_failed"}

        for lid in lids:
            b2_dense_override = None
            b2_sparse_terms = None
            b2_meta_override = None
            meta0 = per_lib_b2_meta.get(lid) if per_lib_b2_meta else None
            if meta0 and meta0.get("enabled"):
                tl = tuple(meta0.get("target_langs") or [])
                b2 = b2_by_langs.get(tl)
                if b2:
                    b2_dense_override = list(dict.fromkeys([question] + list(b2.dense_queries or [])))[:12]
                    b2_sparse_terms = list(b2.sparse_terms or [])[:40]
                    b2_meta_override = {
                        "enabled": True,
                        "mode": "multi_library_reuse",
                        **meta0,
                        "dense_queries_count": len(b2.dense_queries or []),
                        "sparse_terms_count": len(b2.sparse_terms or []),
                    }
            docs_i, meta_i = self._retrieve_impl(
                question,
                keyword_terms=keyword_terms,
                source_filters=source_filters,
                auto_extract_keywords=auto_extract_keywords,
                use_hybrid=use_hybrid,
                use_rerank=False,  # global rerank later
                use_sep_reference=use_sep_reference,
                answer_style=answer_style,
                retrieval_ablation=retrieval_ablation,
                retrieval_final_k_override=max(cand_cap, 60),
                library_id=lid,
                dense_queries_override=b2_dense_override,
                extra_sparse_terms=b2_sparse_terms,
                auto_b2_meta_override=b2_meta_override,
                trace_id=trace_id,
            )
            # namespace ids to avoid collisions (chunk_0 etc)
            ns_list: List[str] = []
            out_docs_i: List[Dict[str, Any]] = []
            for d in docs_i:
                raw = str(d.get("chunk_id") or "")
                ns = f"{lid}::{raw}"
                dd = dict(d)
                dd["library_id"] = lid
                dd["raw_chunk_id"] = raw
                dd["chunk_id"] = ns
                out_docs_i.append(dd)
                doc_by_ns[ns] = dd
                ns_list.append(ns)
            per_docs.append(out_docs_i)
            per_meta.append(meta_i)
            ranked_lists.append(ns_list)

        fused = weighted_reciprocal_rank_fusion(ranked_lists, wts, rrf_k=int(rp.get("RRF_K") or 60))
        fused_ids = [cid for cid, _ in fused]
        fused_docs = [doc_by_ns[cid] for cid in fused_ids if cid in doc_by_ns]

        # Global rerank across fused docs
        rerank_enabled = bool(use_rerank and len(fused_docs) > 1)
        if rerank_enabled:
            try:
                top_cands = fused_docs[:cand_cap]
                ranked = self.reranker.rerank(
                    question, top_cands, top_k=min(len(top_cands), cand_cap)
                )
                fused_docs = ranked + fused_docs[cand_cap:]
            except Exception:
                pass

        # Final truncate uses same logic as single (FINAL_K with coverage constraints already inside per-library),
        # but for multi-library we cap to effective FINAL_K only.
        baseline_final_k = max(1, min(int(rp["FINAL_K"]), int(MAX_FINAL_K)))
        if retrieval_final_k_override is not None:
            baseline_final_k = max(baseline_final_k, int(retrieval_final_k_override))
        fused_docs = fused_docs[:baseline_final_k]

        meta: Dict[str, Any] = {
            "profile": self.profile,
            "library_ids": lids,
            "library_weights": wts,
            "hybrid": bool(use_hybrid),
            "reranked": bool(rerank_enabled),
            "answer_style": answer_style,
            "debug": {
                "libraries": per_meta,
                "cross_library_fused_top_ids": fused_ids[:20],
                "auto_b2_multi": auto_b2_multi,
                "auto_b2_multi_preview": auto_b2_multi_preview,
            },
        }
        return fused_docs, meta

    def answer(
        self,
        question: str,
        user_id: str = "default",
        conversation_id: str = "default",
        memory: bool = True,
        history_max_turns: int = 10,
        history_max_chars: int = 12000,
        wiki_max_chars: int = 3500,
        use_concept_graph: bool = True,
        use_concept_index: bool = False,
        keyword_terms: Optional[List[str]] = None,
        source_filters: Optional[List[str]] = None,
        auto_extract_keywords: bool = True,
        use_hybrid: bool = True,
        use_rerank: bool = True,
        use_sep_reference: bool = False,
        answer_style: str = "哲学论述",
        llm_provider: str = "gemini",
        llm_model: str = "gemini-3.1-pro-preview",
        ultra_long_answer: bool = False,
        library_ids: Optional[List[str]] = None,
        library_weights: Optional[List[float]] = None,
        *,
        retrieval_ablation: Optional[str] = None,
        skip_citation_sanitize: bool = False,
        _allow_sep_swap: bool = True,
        trace_id: str = "",
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        style_norm = normalize_answer_style(answer_style)
        if trace_id:
            log_stage(trace_id=trace_id, stage="answer", event="start", extra={"style": style_norm, "memory": bool(memory)})

        max_out = (
            int(ANSWER_MAX_OUTPUT_TOKENS_ULTRA)
            if ultra_long_answer
            else int(ANSWER_MAX_OUTPUT_TOKENS_DEFAULT)
        )

        uid = (user_id or "default").strip() or "default"
        cid = (conversation_id or "default").strip() or "default"

        history_block = ""
        user_wiki_block = ""
        if memory:
            if trace_id:
                log_stage(trace_id=trace_id, stage="answer.memory", event="start")
            try:
                msgs = load_recent_turns(
                    user_id=uid,
                    conversation_id=cid,
                    max_turns=int(history_max_turns),
                    max_chars=max(int(history_max_chars), 0),
                )
                history_block = prepare_history_block_for_prompt(
                    msgs,
                    history_max_chars=int(history_max_chars),
                )
            except Exception:
                history_block = ""
            try:
                user_wiki_block = read_user_wiki(
                    user_id=uid,
                    max_chars=int(wiki_max_chars),
                    question=(question or ""),
                )
            except Exception:
                user_wiki_block = ""
            try:
                append_event(
                    user_id=uid,
                    conversation_id=cid,
                    role="user",
                    content=(question or ""),
                    meta={"answer_style": answer_style},
                )
            except Exception:
                pass
            if trace_id:
                log_stage(trace_id=trace_id, stage="answer.memory", event="ok", extra={"history_chars": len(history_block or ""), "wiki_chars": len(user_wiki_block or "")})

        # 标准问答：不依赖向量库/本地语料，只做联网搜索 + 调用模型。
        if style_norm == STYLE_STANDARD_QA:
            # If memory exists, prefer answering from conversation continuity instead of web-only QA rules.
            # This avoids the standard QA constraint ("only Sources") which would otherwise ignore history.
            if memory and (history_block or user_wiki_block):
                mem_prompt = f"""You are a careful assistant.\n\nUse the following context for continuity and user preferences.\n- This context is NOT a citable source. Do not fabricate citations.\n- If the answer is explicitly contained in the conversation history, use it directly.\n- If not contained, say you don't know and suggest what to provide next.\n\nUser wiki (may be empty):\n{(user_wiki_block or '').strip() or '(empty)'}\n\nConversation history (recent turns; may be empty):\n{(history_block or '').strip() or '(empty)'}\n\nUser question:\n{(question or '').strip()}\n\nAnswer:\n"""
                response_text, model_used = generate_answer(
                    prompt=mem_prompt,
                    provider=llm_provider,
                    model=llm_model,
                    temperature=min(0.35, float(GEMINI_ANSWER_TEMPERATURE)),
                    max_output_tokens=max_out,
                    trace_id=trace_id,
                    stage="answer.llm_call",
                )
                docs0 = []
                meta0 = {"answer_model": model_used, "retrieval_skipped": True, "standard_qa_memory_mode": True}
            else:
                response_text, docs0, meta0 = answer_standard_qa(
                    question,
                    provider=llm_provider,
                    model=llm_model,
                    temperature=min(0.35, float(GEMINI_ANSWER_TEMPERATURE)),
                    max_output_tokens=max_out,
                    web_max_results=6,
                )
            meta0.update(
                {
                    "profile": self.profile,
                    "answer_style": answer_style,
                    "answer_style_canonical": style_norm,
                    "answer_max_output_tokens": max_out,
                    "retrieval_skipped": True,
                }
            )
            if memory:
                try:
                    append_event(
                        user_id=uid,
                        conversation_id=cid,
                        role="assistant",
                        content=(response_text or ""),
                        meta={"answer_model": f"{llm_provider}:{llm_model}", "answer_style": answer_style},
                    )
                except Exception:
                    pass
                # async update user wiki (A版)
                try:
                    update_user_wiki_async(
                        user_id=uid,
                        question=(question or ""),
                        answer=(response_text or ""),
                        llm_provider=WIKI_LLM_PROVIDER,
                        llm_model=WIKI_LLM_MODEL,
                    )
                except Exception:
                    pass
            return response_text, docs0, meta0

        # SEP 模式：强制使用 sep profile（读取 data/chroma_db_sep/ 里的向量库）。
        # 为了复用现有检索链路，这里用锁保护下的“临时 profile swap”，避免并发状态互相污染。
        if style_norm == STYLE_SEP and _allow_sep_swap:
            with self._swap_lock:
                old = (self.profile, self.params, self.embedder, self.db, self.sparse, self.reranker)
                try:
                    ok = self.switch_profile("sep")
                    if not ok:
                        raise RuntimeError("SEP profile 不存在：请检查 src/config.py 的 PROFILE_SETTINGS 是否包含 'sep'。")
                    # SEP 默认不启用稀疏/混合：用户若另行提供 sparse_fts_sep.db，也可手动在前端勾选开关尝试。
                    answer2, docs2, meta2 = self.answer(
                        question,
                        user_id=uid,
                        conversation_id=cid,
                        memory=memory,
                        history_max_turns=history_max_turns,
                        history_max_chars=history_max_chars,
                        wiki_max_chars=wiki_max_chars,
                        use_concept_graph=use_concept_graph,
                        use_concept_index=use_concept_index,
                        keyword_terms=None,
                        # SEP 模式下默认不继承“限定文件名”过滤，避免把 SEP 全过滤空。
                        source_filters=None,
                        auto_extract_keywords=False,
                        use_hybrid=False,
                        use_rerank=use_rerank,
                        answer_style=answer_style,
                        llm_provider=llm_provider,
                        llm_model=llm_model,
                        ultra_long_answer=ultra_long_answer,
                        retrieval_ablation=retrieval_ablation,
                        skip_citation_sanitize=skip_citation_sanitize,
                        _allow_sep_swap=False,
                    )
                    meta2["sep_mode"] = True
                    meta2["sep_profile"] = "sep"
                    meta2.setdefault("sep_hint", "请将 Chroma 向量库放到 data/chroma_db_sep/ 后再使用本模式。")
                    return answer2, docs2, meta2
                finally:
                    (self.profile, self.params, self.embedder, self.db, self.sparse, self.reranker) = old

        eff_auto = auto_extract_keywords
        if style_norm == STYLE_CONCEPT_MAP:
            eff_auto = False

        docs, meta = self.retrieve(
            question,
            keyword_terms=keyword_terms,
            source_filters=source_filters,
            auto_extract_keywords=eff_auto,
            use_hybrid=use_hybrid,
            use_rerank=use_rerank,
            use_sep_reference=use_sep_reference,
            answer_style=answer_style,
            library_ids=library_ids,
            library_weights=library_weights,
            retrieval_ablation=retrieval_ablation,
            trace_id=trace_id,
        )
        required_language = _detect_required_language(question)
        if not docs:
            msg = _empty_retrieval_message(required_language)
            meta["retrieval_empty"] = True
            meta["answer_style"] = answer_style
            meta["answer_model"] = ""
            meta["answer_max_output_tokens"] = 0
            meta["required_language"] = required_language
            return msg, [], meta

        context = build_context(docs)
        meta["required_language"] = required_language

        conflict_note = ""
        if memory:
            try:
                conflict_note = compute_conflict_hint(
                    history_block=history_block,
                    user_wiki_block=user_wiki_block,
                    docs=docs,
                )
            except Exception:
                conflict_note = ""

        concept_graph_block = ""
        concept_index_block = ""
        if memory and use_concept_graph:
            try:
                from .concept_graph import format_subgraph_for_prompt

                concept_graph_block = format_subgraph_for_prompt(uid, question)
            except Exception:
                concept_graph_block = ""
        if memory and use_concept_index:
            try:
                from .concept_vector_store import concept_search_block

                concept_index_block = concept_search_block(
                    self.embedder,
                    uid,
                    question,
                    top_k=6,
                )
            except Exception:
                concept_index_block = ""

        prompt = build_prompt(
            question,
            context,
            required_language=required_language,
            answer_style=answer_style,
            history_block=history_block,
            user_wiki_block=user_wiki_block,
            conflict_note=conflict_note,
            concept_graph_block=concept_graph_block,
            concept_index_block=concept_index_block,
            wiki_max_chars=int(wiki_max_chars),
            history_max_chars=int(history_max_chars),
        )
        gen_temp = float(GEMINI_ANSWER_TEMPERATURE)
        if normalize_answer_style(answer_style) == STYLE_CITE_PATCH:
            gen_temp = min(0.35, gen_temp)

        response_text, model_used = generate_answer(
            prompt=prompt,
            provider=llm_provider,
            model=llm_model,
            temperature=gen_temp,
            max_output_tokens=max_out,
            trace_id=trace_id,
            stage="answer.llm_call",
        )
        print(f"Answer generation completed with model={model_used}")
        cleaned_text = replace_source_refs(response_text, docs)
        if not skip_citation_sanitize:
            cleaned_text = sanitize_citations(cleaned_text, docs)
        meta["answer_style"] = answer_style
        meta["citation_sanitize_skipped"] = bool(skip_citation_sanitize)
        meta["answer_model"] = model_used
        meta["answer_max_output_tokens"] = max_out
        if memory:
            try:
                append_event(
                    user_id=uid,
                    conversation_id=cid,
                    role="assistant",
                    content=(cleaned_text or ""),
                    meta={"answer_model": model_used, "answer_style": answer_style},
                )
            except Exception:
                pass
            try:
                update_user_wiki_async(
                    user_id=uid,
                    question=(question or ""),
                    answer=(cleaned_text or ""),
                    llm_provider=WIKI_LLM_PROVIDER,
                    llm_model=WIKI_LLM_MODEL,
                )
            except Exception:
                pass
        return cleaned_text, docs, meta
