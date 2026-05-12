import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
# NOTE:
# This project calls the public Gemini API via `google-genai` (v1beta).
# For Gemini 3.1 Pro, the correct model id is `gemini-3.1-pro-preview`
# (NOT `gemini-3.1-pro`, which will 404 on v1beta).
GEMINI_ANSWER_MODEL = "gemini-3.1-pro-preview"
# 辅助模型用于问题扩写等检索侧任务：优先稳定/速度
GEMINI_AUX_MODEL = "gemini-2.5-flash"
# 未在 llm_router._gemini_fallback_chain 中单独列出的 Gemini 模型名，按此顺序降级（各模型内仍受 GEMINI_RETRY_MAX_ATTEMPTS 约束）
GEMINI_FALLBACK_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]
# query expander（问题扩写）失败时的兜底模型
GEMINI_AUX_FALLBACK_MODEL = "gemini-2.5-flash"
# 单模型最大重试次数（含首次请求，>=1）
GEMINI_RETRY_MAX_ATTEMPTS = 3
# 指数退避基数秒（实际等待 = base * 2^(attempt-1) + 抖动）
GEMINI_RETRY_BASE_SECONDS = 1.2
# 每次重试的最大随机抖动秒，避免并发雪崩
GEMINI_RETRY_JITTER_SECONDS = 0.6
# 回答输出上限：默认更稳（12288）；手动开启“超长回答”才使用 24576
ANSWER_MAX_OUTPUT_TOKENS_DEFAULT = 12288
ANSWER_MAX_OUTPUT_TOKENS_ULTRA = 24576
GEMINI_ANSWER_TEMPERATURE = 0.7

# --- Wiki LLM (user.md updates) ---
# Keep wiki updates fast/stable and independent from the answer model.
WIKI_LLM_PROVIDER = "gemini"
WIKI_LLM_MODEL = "gemini-2.5-flash"

# --- Session title suggestion (local multi-session UI) ---
SESSION_TITLE_LLM_PROVIDER = "gemini"
SESSION_TITLE_LLM_MODEL = "gemini-2.5-flash"
SESSION_TITLE_MAX_OUTPUT_TOKENS = 48

# --- User wiki (user.md)：合并写盘 + 强化/遗忘 ---
# 模型仍输出「候选全文」，程序与旧稿合并：候选未写出的旧要点默认保留。
WIKI_UPDATE_MAX_OUTPUT_TOKENS = 8192
# 距上次 wiki 活动时间超过该天数 → 本轮不执行遗忘，并刷新所有要点的强化时间（长期未登录保护）
WIKI_ABSENCE_GRACE_DAYS = 45.0
# 遗忘条件（须同时满足）：距上次强化超过 N 天，且距上次强化已超过 M 次 wiki 更新轮次
WIKI_FORGET_MIN_DAYS = 120.0
WIKI_FORGET_MIN_ROUNDS = 8
# Hypotheses 可用更短的阈值（更易过期）
WIKI_HYPOTHESIS_FORGET_MIN_DAYS = 90.0
WIKI_HYPOTHESIS_FORGET_MIN_ROUNDS = 6

# 每 N 次 wiki 写入后，触发一次“整理/压实”：规范主要写在提示词里；
# 后端只做结构性校验，避免用脆弱的 token/bullet 校验误拒合理合并。
WIKI_COMPACT_EVERY_N_WRITES = 5
WIKI_COMPACT_MAX_OUTPUT_TOKENS = 4096

# --- Auto + B2 (cross-lingual retrieve) ---
# When enabled, the system may generate multilingual query variants and term lists
# (using GEMINI_AUX_MODEL) based on each library's cached language profile.
AUTO_B2_ENABLED = True
# Trigger only when dominant library language differs from the question language.
AUTO_B2_MIN_LIBRARY_LANG_SHARE = 0.60
# At most N target languages to expand into (e.g., ["de","en"]).
AUTO_B2_MAX_TARGET_LANGS = 2
# LLM timeout budget for the auxiliary translation/term step; if exceeded, fall back to original query.
AUTO_B2_AUX_TIMEOUT_SECONDS = 2.5
# Cache directory under data/ for B2 outputs (JSON).
AUTO_B2_CACHE_DIR = "data/cache/auto_b2"

# --- 其他供应商（key 直接在此处配置）---

# 可选：代理（v2ray / Clash 等）
# - HTTP(S) 代理（常见 33210）
# - SOCKS5 代理（常见 33211；会写入 ALL_PROXY，很多库会使用）
# 留空则直连。
HTTP_PROXY_URL = "http://127.0.0.1:33210"
HTTPS_PROXY_URL = "http://127.0.0.1:33210"
SOCKS_PROXY_URL = ""

# OpenAI Responses API base
OPENAI_BASE_URL = "https://api.openai.com/v1"

# DeepSeek OpenAI-compatible base
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# OpenAI 主力模型（按你当前可用模型）
OPENAI_MODEL_PRIMARY = "gpt-5.1"
OPENAI_MODEL_SECONDARY = "gpt-5-mini"

# OpenAI 侧：并发与重试（避免 429 雪崩）
# - 并发过高时会更容易触发 429；Web 端多用户同时问答建议设为 1~3
OPENAI_MAX_CONCURRENCY = 2
# - OpenAI 429/503 等可恢复错误的最大重试次数（含首次请求，>=1）
OPENAI_RETRY_MAX_ATTEMPTS = 5
# - 指数退避基数秒（实际等待 = base * 2^(attempt-1) + 抖动；同时会尊重 Retry-After）
OPENAI_RETRY_BASE_SECONDS = 1.5
# - 每次重试的最大随机抖动秒
OPENAI_RETRY_JITTER_SECONDS = 0.8
# - 单次 sleep 上限（秒），避免等待过久卡死请求
OPENAI_RETRY_MAX_SLEEP_SECONDS = 30.0

# DeepSeek 主力模型
DEEPSEEK_MODEL_PRIMARY = "deepseek-v4-flash"
DEEPSEEK_MODEL_SECONDARY = "deepseek-v4-pro"

CHROMA_PATH = "data/chroma_db"
SPARSE_DB_PATH = "data/sparse_fts.db"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Avoid unbounded context expansion; allow large but controlled values.
MAX_FINAL_K = 200
# Full keyword-hit list cap for side panel and API payload safety.
MAX_KEYWORD_HITS = 2000
# When multiple sources are involved, keep at least N chunks per source.
MIN_CHUNKS_PER_PRIMARY_SOURCE = 15
# Number of primary sources to enforce in balanced coverage mode.
PRIMARY_SOURCE_COUNT = 2

RETRIEVAL_PROFILE = "quality"  # "quality" | "fast" | "sep"

# --- Hugging Face cache / local models ---
# To avoid repeated slow downloads, consider setting HF_HOME to a persistent directory
# (e.g., project-local "data/hf_cache"). See tools/prefetch_models.py.

# --- Reranker runtime ---
# "auto" => use CUDA if available, else CPU
# "cuda" => force CUDA (will fall back to CPU if not available)
# "cpu"  => force CPU
RERANKER_DEVICE = "auto"

# 在「检索 profile」基础上，按回答风格再覆盖（键与 academic_prompt 中文 value 一致）
ANSWER_STYLE_RETRIEVAL_OVERRIDES = {
    "概念梳理": {
        "FINAL_K": 28,
        "SEARCH_K": 36,
        "SPARSE_K": 64,
        "HYBRID_TOP_N": 56,
        "RERANK_CANDIDATES": 52,
    },
    "文献综述": {
        "FINAL_K": 18,
        "SEARCH_K": 32,
        "SPARSE_K": 52,
        "HYBRID_TOP_N": 44,
        "RERANK_CANDIDATES": 40,
    },
    "哲学论述": {
        "FINAL_K": 16,
        "SEARCH_K": 30,
        "SPARSE_K": 48,
        "HYBRID_TOP_N": 40,
        "RERANK_CANDIDATES": 36,
        # Full 模式 RRF：略抬高稠密（语义）相对稀疏（词面），便于与 sparse_only 拉开差异
        "RRF_DENSE_WEIGHT": 1.35,
        "RRF_SPARSE_WEIGHT": 1.0,
    },
    "盲审审稿": {
        "FINAL_K": 14,
        "SEARCH_K": 28,
        "SPARSE_K": 44,
        "HYBRID_TOP_N": 38,
        "RERANK_CANDIDATES": 34,
    },
    "引文补注": {
        "FINAL_K": 20,
        "SEARCH_K": 32,
        "SPARSE_K": 52,
        "HYBRID_TOP_N": 44,
        "RERANK_CANDIDATES": 40,
    },
    "学术分析": {
        "FINAL_K": 14,
        "SEARCH_K": 26,
        "SPARSE_K": 40,
        "HYBRID_TOP_N": 34,
        "RERANK_CANDIDATES": 30,
    },
    "简洁作答": {
        "FINAL_K": 8,
        "SEARCH_K": 18,
        "SPARSE_K": 28,
        "HYBRID_TOP_N": 24,
        "RERANK_CANDIDATES": 22,
    },
}

PROFILE_SETTINGS = {
    "quality": {
        # default profile: multilingual quality (CN/EN/DE)
        "EMBEDDING_MODEL": "BAAI/bge-m3",
        "RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",
        "SEARCH_K": 22,
        "FINAL_K": 10,
        "SPARSE_K": 34,
        "HYBRID_TOP_N": 28,
        "RRF_K": 60,
        "RERANK_CANDIDATES": 26,
        # 无风格覆盖时的默认：RRF 中稠密略重于稀疏（可被 ANSWER_STYLE_RETRIEVAL_OVERRIDES 覆盖）
        "RRF_DENSE_WEIGHT": 1.15,
        "RRF_SPARSE_WEIGHT": 1.0,
    },
    "fast": {
        # lighter profile: faster but less multilingual robustness
        "EMBEDDING_MODEL": "BAAI/bge-small-en",
        "RERANKER_MODEL": "BAAI/bge-reranker-base",
        "SEARCH_K": 12,
        "FINAL_K": 5,
        "SPARSE_K": 18,
        "HYBRID_TOP_N": 14,
        "RRF_K": 60,
        "RERANK_CANDIDATES": 14,
    },
    # Stanford Encyclopedia of Philosophy（SEP）专用 profile：
    # 你可以将预先构建好的 Chroma 向量库放到 data/chroma_db_sep/（目录名由 CHROMA_PATH + _sep 推导）。
    # 建议使用与构建向量库相同的 embedding 模型，否则检索会明显变差。
    "sep": {
        "EMBEDDING_MODEL": "BAAI/bge-m3",
        "RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",
        "SEARCH_K": 26,
        "FINAL_K": 10,
        "SPARSE_K": 0,
        "HYBRID_TOP_N": 0,
        "RRF_K": 60,
        "RERANK_CANDIDATES": 24,
    },
}

CURRENT_PROFILE = RETRIEVAL_PROFILE if RETRIEVAL_PROFILE in PROFILE_SETTINGS else "quality"
_P = PROFILE_SETTINGS[CURRENT_PROFILE]

EMBEDDING_MODEL = _P["EMBEDDING_MODEL"]
RERANKER_MODEL = _P["RERANKER_MODEL"]
SEARCH_K = _P["SEARCH_K"]
FINAL_K = _P["FINAL_K"]
SPARSE_K = _P["SPARSE_K"]
HYBRID_TOP_N = _P["HYBRID_TOP_N"]
RRF_K = _P["RRF_K"]
RERANK_CANDIDATES = _P["RERANK_CANDIDATES"]
