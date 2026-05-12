# Philosophy RAG（Hybrid + Rerank）

面向哲学与多语种（中/英/德等）文献的**本地知识库问答系统**：内置可一键 Ingest 的语料流水线（语义切分→向量化→Chroma 持久化→SQLite FTS5 词面索引），查询侧支持**稠密语义检索 + 稀疏关键词检索 + RRF 融合 + Cross-Encoder 精排**，生成侧可切换 **Gemini / OpenAI / DeepSeek** 等多家模型，并对**引用做可核验清洗**（只保留能在本次证据片段中对齐的页码/来源）。

除基础 RAG 外，还提供：**多轮会话记忆 + 用户 Wiki**（持久化 JSONL/Markdown，前端可只读查看）、可选**概念图/概念向量域**辅助提示、**多库（library）加权融合检索**、文献上传与**可恢复删除（trash）**、以及「标准问答（联网）」与「SEP 专用向量库/参照」等模式，便于做严肃的文献核对、综述写作与研究型问答。

此外，系统支持按库语言分布进行 **Auto+B2 跨语检索增强**：当库内文献以德/英为主而用户用中文提问时，会用辅助模型生成多语稠密查询与术语锚点，让向量检索与 FTS5 词面检索同时“对齐语种”；多库场景下会按语言组复用该步骤，避免按库重复调用导致延迟爆炸。

**一键启动**：已安装 **Python 3.11** 时，在仓库根目录双击 `start_app.bat`（Windows）或 `start_app_mac.command`（macOS）；会检查/创建 `.venv`、安装依赖、**启动 API 后端（8000）+ 本地静态页服务（5173）**，并在浏览器打开 `http://127.0.0.1:5173/frontend.html`（避免 `file://` 下 ES 模块不执行、只剩兜底脚本）。翻译子程序有独立入口：Windows 双击 `start_translation.bat`，打开 `http://127.0.0.1:5173/translation_frontend.html`。首次请在 `src/config.py` 中配置 API Key（见下文）。

---

## 功能概览

| 能力 | 说明 |
|------|------|
| 多轮记忆与 Wiki | 每用户会话 NDJSON 历史、可注入 Wiki 页（衰减与访问加权，见 `src/wiki_manager.py`） |
| 概念层 | 可选概念图（NetworkX）与独立概念向量域（`src/concept_graph.py`、`src/concept_vector_store.py`） |
| 流式回答 | `POST /api/answer/stream` 返回 NDJSON 行（与标准 `/api/answer` 同请求体思路） |
| 混合检索 | 稠密语义 + FTS5 词面，RRF（`src/hybrid_retrieval.py`）融合 |
| 精排 | `sentence-transformers` Cross-Encoder（profile 可配） |
| 查询侧 | 问题扩写、用户关键词与自动抽词合并（`query_expander` / `term_merger`） |
| 跨语检索（Auto+B2） | 基于库语言分布自动生成多语查询与术语锚点，提升中文问德文/英文库的稳定性（见 `src/auto_b2.py`、`src/library_language.py`） |
| 风格 | 多种回答风格，检索预算可按风格覆盖（`ANSWER_STYLE_RETRIEVAL_OVERRIDES`） |
| 引用 | Prompt 约束 + 生成后 `(source, p. page)` 与证据对齐（`sanitize_citations`） |
| 标准问答 | 无本地语料时，仅基于 Wikipedia（英/德）联网检索（`standard_qa`） |
| SEP | 独立 `sep` profile + 「参照 SEP」弱融合（`data/chroma_db_sep/`） |
| EPUB | 支持 `.epub` ingest；以 `chapter-x:para-y` 生成伪页码用于引用定位（`src/epub_loader.py`） |

---

## 架构（检索到生成）

1. **扩写**：`expand_query` 生成多条稠密查询（失败则退回原问句）。
2. **稠密**：当前 profile 的 `VectorStore` → Chroma `query`。
3. **关键词**：合并用户词与自动词 → `build_sparse_query` → FTS5。
4. **融合**：`reciprocal_rank_fusion` / `weighted_reciprocal_rank_fusion`（SEP 参照时为加权 RRF）。
5. **重排**：可选 Cross-Encoder，截断至 `FINAL_K` 有效上限。
6. **生成**：`build_context` + `build_prompt` → `generate_answer`。
7. **清洗**：`replace_source_refs` → `sanitize_citations`。

`ingest` 流水线：`document_loader`（支持 `pdf/docx/json/epub`）→ `semantic_chunker` → `Embedder` → Chroma + `SparseRetriever.rebuild`，`chunk_id` 对齐。

---

## 目录结构（核心）

```text
Philosophy_UP/                 # 仓库根目录名可自定
├── start_app.bat / start_app_mac.command
├── start_translation.bat       # 翻译子程序独立一键启动
├── run_backend.bat
├── ingest.py
├── ingest_single_tmp.py
├── merge_profile.py / merge_tmp_to_quality.bat
├── web_app.py                 # FastAPI
├── frontend.html
├── translation_frontend.html   # 独立翻译工作台（不嵌入主前端）
├── translation_frontend.js
├── eval_frontend.html         # 评测控制台（调单次 /api/eval/single）
├── chat.py                    # 终端交互（简易）
├── requirements.txt
├── tools/
│   ├── ensure_torch_accel.py  # 可选：尝试安装 CUDA 版 PyTorch
│   └── inspect_chroma_sqlite.py  # （可选）查看 Chroma 集合名与 embeddings 条数
├── data/
│   ├── chroma_db_quality/ | chroma_db_fast/ | chroma_db_sep/
│   ├── sparse_fts_quality.db | sparse_fts_fast.db | sparse_fts_sep.db（可选）
│   ├── pdf/                   # 语料
│   └── uploads/               # Web 上传
├── data/eval/                 # 示例评测 JSONL
└── src/
    ├── eval/                  # 批量评测 CLI + 指标
    ├── config.py
    ├── rag_engine.py
    ├── vector_store.py
    ├── sparse_retriever.py
    ├── hybrid_retrieval.py
    ├── embedder.py
    ├── reranker.py
    ├── ingest_pipeline.py
    └── …
```

---

## 安装

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 配置（`src/config.py`）

- **API Key**（按需）：`GEMINI_API_KEY`、`OPENAI_API_KEY`、`DEEPSEEK_API_KEY`。勿将真实 Key 提交到公开仓库；泄露后请轮换。
- **Gemini 模型与降级**：默认主模型为 `GEMINI_ANSWER_MODEL = "gemini-3.1-pro"`。生成路径在 `src/llm_gemini.py` 中对**每个候选模型**最多重试 `GEMINI_RETRY_MAX_ATTEMPTS` 次（默认 3），再换下一档；降级链由 `src/llm_router.py` 的 `_gemini_fallback_chain` 决定：选 `gemini-3.1-pro` 时为 **3.1-pro → 2.5-pro → 2.5-flash**；选 `gemini-2.5-pro` 时为 **2.5-pro → 2.5-flash**；选 `gemini-2.5-flash` 则不再降级。若 Google 侧模型字符串有变，只需在请求体 `llm_model` 或配置里改成当前可用的 id（亦可把 `gemini-3-pro-preview` 等别名加入 `_gemini_fallback_chain` 的首档判断）。
- **代理（可选）**：`HTTP_PROXY_URL` / `HTTPS_PROXY_URL` / `SOCKS_PROXY_URL`（SOCKS 需 `httpx[socks]`）。本机调试若遇 `127.0.0.1` 被代理成 502，请为 `NO_PROXY` 增加 `localhost,127.0.0.1`。
- **检索 Profile**：`RETRIEVAL_PROFILE = "quality" | "fast" | "sep"`  
  - `quality`：默认多语言，`BAAI/bge-m3` + `bge-reranker-v2-m3`。  
  - `fast`：更轻量。  
  - `sep`：用于**仅查 SEP 向量库**或配合「斯坦福哲学百科模式」；向量目录为 `data/chroma_db_sep/`，集合名一般为 `philosophy_sep`；若你是用 `quality` ingest 后整库挪到 `chroma_db_sep/`，集合可能仍为 `philosophy_quality`，程序会自动选用有数据的集合（见 `vector_store.py`）。

切换 profile 后须对**该 profile 重新 Ingest**，否则 embedding 与索引不一致。

---

## 启动

### 后端

```powershell
python -m uvicorn web_app:app --reload --host 127.0.0.1 --port 8000
```

健康检查：`GET http://127.0.0.1:8000/api/health` → `{"status":"ok"}`。

### 前端（推荐用 HTTP 打开）

用 **`start_app` 已会自动起 `http.server 5173`**；或手动：

```powershell
.\run_frontend_static.bat
REM 或: python -m http.server 5173 --bind 127.0.0.1
```

浏览器打开：`http://127.0.0.1:5173/frontend.html`（勿用 `file://`，否则部分环境只剩 ES5 兜底）。

### 翻译子程序独立前端

翻译子程序不嵌入主问答前端。Windows 下双击：

```powershell
.\start_translation.bat
```

脚本会启动同一个 FastAPI 后端和静态页服务，并打开：

```text
http://127.0.0.1:5173/translation_frontend.html
```

### 关于 `main.js`（主脚本外置）

为提高兼容性，前端主逻辑已从 `frontend.html` 的**超长内联脚本**迁移到外部文件 `main.js`，由页面底部的加载器以 `type="module"` 动态加载。

- **好处**：规避某些环境/策略对超长内联脚本的静默拦截或跳过；也更利于浏览器缓存与调试。
- **如果仍看到 ES5 兜底**：先确认 `http://127.0.0.1:5173/main.js` 能直接打开；若打不开，多半是静态服务器没起来或被安全软件拦截。

若 **5173 端口被占用**，先关闭旧窗口或改 `run_frontend_static.bat` 中的端口号，并在 `frontend.html` 的 `API_BASE` 仍指向 `8000`（仅静态端口可变）。

---

## Ingest

`ingest.py` 与 `POST /api/ingest` 使用当前 **运行时 profile**（见 `web_app` 中 `rag_engine.get_profile()`），流程见 `src/ingest_pipeline.py`：全量重建 Chroma 与对应 `sparse_fts_<profile>.db`。

---

## SEP（斯坦福哲学百科）说明

1. **独立向量库**：`data/chroma_db_sep/`（Chroma 持久化目录整份）。  
2. **「参照 SEP」**：在主库检索之外检索 SEP，加权 RRF 融合；`sep` 片段条数有上限（代码中 `sep_max_docs` 等）。  
3. **「斯坦福哲学百科模式」**：临时切换到 `sep` profile 检索，再生成；**不继承主界面「限定文件名」**，避免把 SEP 库滤空。  
4. **`sep` profile** 下 `HYBRID_TOP_N` 可能为 0（表示不走混合）；检索逻辑已对「仅稠密」路径做上限回退，避免候选被截成空。

若 SEP 检索仍异常：优先检查 `data/chroma_db_sep/` 是否存在、以及是否确实包含向量（目录下应有 `chroma.sqlite3` 等文件）。
必要时可运行：

```powershell
python tools/inspect_chroma_sqlite.py
```

核对集合名与 embeddings 条数是否为 0。

---

## 翻译子程序（长文/书籍）

新增 `translate.py` 可把单篇文章或一本书作为翻译项目处理：先读取文档并生成全书概览与术语表草稿，用户确认术语表后，再按章节/分块调用 Gemini / OpenAI / DeepSeek 翻译，最后导出 `.txt` 或 `.docx`。

推荐流程：

```powershell
python translate.py prepare "data/library_docs/book.pdf" --target zh-CN --provider gemini
python translate.py glossary <project_id> --draft
python translate.py confirm-glossary <project_id>
python translate.py translate <project_id>
python translate.py export <project_id> --format docx
```

也可以一键运行（会自动确认模型生成的术语表草稿）：

```powershell
python translate.py run "data/library_docs/book.pdf" --target zh-CN --provider openai --model gpt-5.1 --format txt
```

中间文件保存在 `data/translations/<project_id>/`：`state.json` 记录项目状态，`glossary.draft.json` 是术语表草稿，`glossary.json` 是确认后的项目术语表，`translations/` 保存每个分块译文。翻译过程可续跑；失败后再次执行 `python translate.py translate <project_id>` 会跳过已有分块。

长期术语库保存在 `data/translations/global_glossary.json`。确认某个项目术语表时，术语会自动合并进长期库；新项目概览和分块翻译都会参考长期库，但项目术语表优先覆盖长期库，适合处理同一术语在不同作者/语境下的特殊译法。可随时查看和回写：

```powershell
python translate.py global-glossary --target zh-CN > global_glossary.edit.json
python translate.py save-global-glossary global_glossary.edit.json
```

独立翻译前端 `translation_frontend.html` 提供“查看长期术语库”和“保存长期术语库”按钮，可直接把 JSON 加载到术语表编辑框中修正；主问答前端 `frontend.html` 不包含翻译子程序。

设计上，概览阶段默认**不建立持久向量库**。翻译任务更需要稳定的章节结构、全书摘要、术语表和已译上下文，而不是随机检索证据；持久向量库只适合后续反复问答、跨章节查出处或特别复杂的术语追踪场景。第一版保留扩展空间，但默认用结构化 JSON 状态控制上下文。

---

## API 摘要

- `POST /api/answer`：请求体见 `web_app.QuestionRequest`。除检索字段外，支持 `user_id`、`conversation_id`、`memory`、历史与 Wiki 注入上限（`history_max_turns` / `history_max_chars` / `wiki_max_chars`）、概念层开关（`use_concept_graph` / `use_concept_index`）、`llm_provider` / `llm_model`（Gemini 可选 `gemini-3.1-pro`、`gemini-2.5-pro`、`gemini-2.5-flash` 等）。`source_filters` 为 **source 元数据子串**；填多条时与评测相同，会**尽量均分**各子串对应文献的条数。成功响应含 `conversation_id`（用于多轮延续）。  
- `POST /api/answer/stream`：同上业务参数，响应为 **NDJSON** 流（进度与最终文本分段输出，适合长回答）。  
- `POST /api/memory/concepts/add`、`POST /api/memory/graph/edge`：维护用户概念与图边（供检索与 prompt 注入）。  
- `GET/POST /api/profile`：切换检索 profile。  
- `POST /api/ingest`、`POST /api/ingest/stream`：ingest。  
- `POST /api/upload`：上传至 `data/uploads/`。
- `POST /api/translation/prepare`、`GET /api/translation/projects`、`POST /api/translation/projects/{project_id}/glossary`、`POST /api/translation/projects/{project_id}/translate`、`POST /api/translation/projects/{project_id}/export`：长文翻译项目接口。
- `GET/PUT /api/translation/glossary`：长期全局翻译术语库，可用 `target_language` 查询参数过滤目标语言。

响应含 `docs`、`keyword_hit_docs`、`debug`（dense/sparse/fused id、chroma 路径等）、SEP 相关元数据等。

---

## 检索参数调优（`PROFILE_SETTINGS` / `ANSWER_STYLE_RETRIEVAL_OVERRIDES`）

- `SEARCH_K`：稠密召回条数。  
- `SPARSE_K`：稀疏取前若干命中。  
- `HYBRID_TOP_N`：融合列表截断（混合路径）；`sep` 专用 profile 可能为 0，由实现回退到 `SEARCH_K`。  
- `FINAL_K`：进入上下文的片段数上限（另受 `MAX_FINAL_K` 等约束）。  
- `RRF_K`：RRF 平滑常数（默认 60）。

---

## 依赖说明

见 `requirements.txt`：`chromadb`、`sentence-transformers`、`fastapi`、`uvicorn`、`google-genai`、`httpx[socks]`、`networkx`（概念图）、文档解析与可选 NLP/LDA/可视化库等。稀疏检索为 **SQLite FTS5**，不依赖 `rank-bm25`。记忆与 Wiki 以文件为主，无额外重型服务依赖。

另：前端「库内文献（sources）」列表支持 **多语言排序**（见 `web_app.py` 与 `src/source_sort_key.py`）：
- **中文**：按 **拼音**（`pypinyin`）
- **日文**：按 **五十音序**（将标题转为平假名后排序；`pykakasi`）
- **其它语言/符号**：按 Unicode 规范化（NFC）+ 不区分大小写的字典序

如你是从旧环境升级，请确保已安装新增依赖：

```powershell
python -m pip install -r requirements.txt
```

---

## 常见问题

| 现象 | 处理 |
|------|------|
| 混合检索未生效 | 确认已 Ingest 且存在 `sparse_fts_<profile>.db`。 |
| SEP 无结果 | 确认 `data/chroma_db_sep/` 存在且集合内有向量（目录下应有 `chroma.sqlite3` 等文件）。 |
| 侧边栏无片段但正文有回答 | 多为前端 `module` 未执行而走了兜底；用 `http://127.0.0.1:5173/frontend.html` 打开并强刷；或看页面「前端自检」提示。 |
| 引用被替换为 `(unverified citation removed)` | 该页码/来源未出现在本次 `docs` 证据中，属预期防护。 |
| Ingest 很慢/占盘大 | 大库+大模型为正常现象；`fast` profile 可减轻负担。 |

---
