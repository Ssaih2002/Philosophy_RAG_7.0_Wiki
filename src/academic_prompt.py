"""
回答风格（answer_style）约定：

- API / 前端请优先使用下列「中文 value」（名称日后若要改，只改此处常量即可）。
- 仍接受旧版英文 value，由 normalize_answer_style() 统一映射。

已在前端下架、代码中仍兼容的风格：学术分析、简洁作答（见 DEPRECATED_STYLES）。
"""

# --- 主模式（推荐给 API / 前端的中文 value）---
STYLE_PHILOSOPHER = "哲学论述"
STYLE_REVIEW = "盲审审稿"
STYLE_CITE_PATCH = "引文补注"
STYLE_CONCEPT_MAP = "概念梳理"
STYLE_LITERATURE_REVIEW = "文献综述"
STYLE_STANDARD_QA = "标准问答"
STYLE_SEP = "斯坦福哲学百科模式"

# --- 已弃用（前端不再展示；保留兼容与 prompt 分支）---
STYLE_ACADEMIC = "学术分析"  # 对应旧 english: academic
STYLE_CONCISE = "简洁作答"  # 对应旧 english: concise

DEPRECATED_STYLES = frozenset({STYLE_ACADEMIC, STYLE_CONCISE})


def normalize_answer_style(answer_style: str) -> str:
    """将英文或别名统一为内部使用的中文 canonical style（先匹配专名，避免误映射）。"""
    if not answer_style:
        return STYLE_PHILOSOPHER
    s = str(answer_style).strip()
    key = s.lower()

    if s in (STYLE_STANDARD_QA, "标准问答模式", "问答", "问答模式") or key in ("qa", "standard_qa"):
        return STYLE_STANDARD_QA
    if s in (STYLE_SEP, "SEP", "斯坦福哲学百科", "斯坦福哲学百科模式") or key in ("sep", "stanford_encyclopedia"):
        return STYLE_SEP
    if s in (STYLE_CONCEPT_MAP, "关键词谱系") or key == "concept_map":
        return STYLE_CONCEPT_MAP
    if (
        s in (STYLE_LITERATURE_REVIEW, "综述", "文献回顾")
        or key in ("literature_review", "lit_review")
    ):
        return STYLE_LITERATURE_REVIEW
    if (
        s in (STYLE_CITE_PATCH, "仅补脚注", "补引用")
        or key == "cite_patch"
    ):
        return STYLE_CITE_PATCH
    if s in (STYLE_REVIEW, "盲审") or key == "review":
        return STYLE_REVIEW
    if key == "academic" or s == STYLE_ACADEMIC:
        return STYLE_ACADEMIC
    if key == "concise" or s == STYLE_CONCISE:
        return STYLE_CONCISE
    if (
        s in (STYLE_PHILOSOPHER, "哲学沉思者")
        or key == "philosophical"
    ):
        return STYLE_PHILOSOPHER

    return STYLE_PHILOSOPHER


def _length_and_coverage_block(style: str) -> str:
    """强制长答与多脚注（按输出语言计正文规模，不含脚注列表）。"""
    if style == STYLE_CONCEPT_MAP:
        return """
## Length and Coverage (concept mapping; mandatory)

- Main body length: at least **4000 output-language characters or 2600 English words**, target **4500-6500 output-language characters or 3000-4300 English words** (`## Footnotes` excluded). If shorter, continue expanding until complete.
- Footnotes: at least **22** valid entries, prioritizing different sources and page numbers from Sources; cover most relevant excerpts where possible.
- Structure: at least **5** second-level headings (`## ...`), with multiple developed paragraphs per section and frequent `(...quotation or close paraphrase...)[n]` evidence.
- Single-text / multi-text: cover the planned sections fully; for multiple texts, include a dedicated section on differences in usage and conceptual genealogy.
- Do not output a mere outline or say "space is limited"; incorporate useful evidence from Sources.
"""
    if style == STYLE_LITERATURE_REVIEW:
        return """
## Length and Coverage (literature review; mandatory)

- Main body length: at least **3600 output-language characters or 2300 English words**, target **4200-5600 output-language characters or 2700-3700 English words** (`## Footnotes` excluded).
- Footnotes: at least **16** entries, covering different sources and page numbers where possible; avoid repeating one source without need.
- Structure: at least **5** second-level headings (`## ...`) covering research question/scope, research trajectory/periodization, core controversies, methods/evidence assessment, and gaps/future issues.
- Method: compare positions, argumentative strength, and evidence types instead of listing views.
"""
    if style == STYLE_PHILOSOPHER:
        return """
## Length and Coverage (philosophical argument; mandatory)

- Main body length: at least **3200 output-language characters or 2100 English words**, target **3800-5200 output-language characters or 2500-3400 English words** (footnotes excluded).
- Footnotes: at least **14** entries across the argument; support core claims with excerpts.
- Develop the argument in full paragraphs; do not substitute a short answer.
"""
    if style == STYLE_REVIEW:
        return """
## Length (blind review)

- Target at least **3000 output-language characters or 1900 English words** unless the submitted draft is very short; still cover all review dimensions.
- Use `[n]` footnotes when quoting the user's draft or corpus evidence; target at least **8** footnotes unless evidence is too sparse, and say so.
"""
    if style == STYLE_CITE_PATCH:
        return """
## Length (citation patch)

- The body length follows the user's draft; do not add or remove wording just to meet a length target.
- Add as many footnotes as the verifiable claims support; cite every traceable claim where possible.
"""
    if style in (STYLE_ACADEMIC, STYLE_CONCISE):
        return """
## Length

- Academic / concise modes: target **2200-3500 output-language characters or 1400-2300 English words**; include at least **10** footnotes unless Sources are too sparse.
"""
    return ""


def _footnote_rules_block() -> str:
    return """
## Citation and Footnote Format (mandatory for all modes)

Use in-text bracketed footnote markers in Markdown:

0. **Number consistency (mandatory)**: footnotes must start at `[1]` and increase strictly in order of first appearance. Do not skip, repeat, or back-reference numbers. Before finalizing, verify that the last in-text number equals the number of footnote-list entries.

1. **Main body**: when using or quoting a source, give a quotation or close paraphrase followed immediately by a footnote marker:
   `(...quotation or close paraphrase...)[n]`
   Use ASCII brackets and Arabic numerals such as `[1]`, `[2]`.

2. **Footnote list**: after the main body, use a second-level heading localized to the required output language (for English: `## Footnotes`; for Chinese: `## 脚注`), then list:
   `[n] (source filename or Cite as name, p. page)` followed by a short explanation of the source and its relation to the body.
   Source names and page numbers **must** come from `Cite as: (..., p. ...)` in Sources; never fabricate them.

3. **Forbidden**: do not use standalone `(filename, p. x)` citations in the body without a matching `[n]` footnote entry unless a mode explicitly says otherwise. Do not use arbitrary labels such as `Source 1` or `[Source 3]`.

4. **No retrieved Sources**: if Sources are empty or only contain the placeholder notice, briefly explain in the required output language that no relevant excerpts were retrieved from the current index and suggest checking Ingest, keywords, or filename filters. **Do not** invent sources or footnotes.
"""


def _clip(s: str, max_chars: int) -> str:
    t = (s or "").strip()
    if max_chars <= 0:
        return ""
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1].rstrip() + "…"


def build_prompt(
    question,
    context,
    required_language="same as question",
    answer_style="哲学论述",
    *,
    history_block: str = "",
    user_wiki_block: str = "",
    conflict_note: str = "",
    concept_graph_block: str = "",
    concept_index_block: str = "",
    wiki_max_chars: int = 3500,
    history_max_chars: int = 12000,
):
    style = normalize_answer_style(answer_style)

    style_block = f"""
Style mode: {style}
Write with maximal depth, strong conceptual architecture, and sustained argument.
Use fully developed paragraphs and avoid brief outline-like responses.
Differentiate analysis from mere summary; foreground tensions and conceptual commitments.
"""
    if style == STYLE_ACADEMIC:
        # DEPRECATED for frontend: 仍保留分支供旧 API 使用
        style_block = f"""
Style mode: {style} (deprecated in UI)
Write in a rigorous academic tone with explicit concepts and argument structure.
Prefer clarity and textual precision over rhetorical flourish.
"""
    elif style == STYLE_CONCISE:
        # DEPRECATED for frontend
        style_block = f"""
Style mode: {style} (deprecated in UI)
Be clear and focused while still preserving core argument steps and key footnotes.
Use shorter sections.
"""
    elif style == STYLE_PHILOSOPHER:
        style_block = f"""
Style mode: {style}
You compose as a philosopher writing for an expert audience: maximal conceptual depth,
sustained argument, explicit distinctions, and dialectical structure.
Avoid outline-style bullet substituting for real analysis; use full paragraphs.
"""
    elif style == STYLE_REVIEW:
        style_block = f"""
Style mode: {style}
You are an extremely strict dissertation blind reviewer.
Assume the user pasted a full draft section (possibly thousands of words) and wants severe quality control.
Tone: strict, unsparing, direct; but still constructive and professional.
Follow the same footnote convention as other modes when you quote their draft or attribute to corpus.
"""
    elif style == STYLE_CITE_PATCH:
        style_block = f"""
Style mode: {style}
The user's message in \"User input\" is a **draft to be annotated only**. Your job:

1. **Reproduce the user's draft verbatim** as the main body: **do not** change wording, order of paragraphs,
   punctuation for grammar, typos, or structure **except** inserting footnote markers and optional minimal
   clarifying brackets ONLY if absolutely necessary for disambiguation (prefer zero such edits; if in doubt, do not edit).

2. Where a claim or sentence can be supported by Sources, append after the relevant span the excerpt in parentheses
   followed by `[n]`, as: `（…verbatim or tight paraphrase from Sources…）[n]`. If the sentence already quotes the source,
   still add `[n]` after that parenthetical quote.

3. If a sentence **cannot** be tied to Sources, do **not** invent a footnote; you may append `[待核]` once per problematic
   sentence at most, without rewriting the sentence.

4. Then output `## 脚注` with each `[n]` matching the Cite as lines from Sources.

If Sources are empty, output only a short notice per global rules — do not fabricate annotated body.
"""
    elif style == STYLE_CONCEPT_MAP:
        style_block = f"""
Style mode: {style}
The user's input is **keyword-focused** (not necessarily a full question). You must:

1. Base the answer **primarily on direct quotation** from Sources; every major point should anchor to quoted lines
   followed by `[n]` in the required format.

2. If Sources overwhelmingly come from **one** document: explain how the concept(s) function **within that text**
   (definition, argumentative role, nearby concepts).

3. If Sources span **multiple** documents: contrast how the concept(s) differ across texts and, if appropriate,
   sketch a brief **conceptual genealogy** (who uses it how; tensions; lineage).

4. Do not substitute a general encyclopedia definition for textual analysis; state explicitly when Sources are thin.

5. Use the same `## 脚注` block at the end with full Cite as references.
"""
    elif style == STYLE_LITERATURE_REVIEW:
        style_block = f"""
Style mode: {style}
Write as a rigorous literature reviewer for an academic journal.
Organize the answer by research themes/controversies rather than by isolated author summaries.
For each theme, compare positions, evaluate evidence quality, and identify what remains unresolved.
Conclude with a synthetic assessment of research gaps and actionable future directions.
"""

    review_block = ""
    output_block = """
## Output Structure (philosophical argument / default)

1. Main body: several developed paragraphs that advance the argument; citations use `(...quotation or close paraphrase...)[n]`.
2. Footnote section: localized heading (`## Footnotes` in English, `## 脚注` in Chinese), with each `[n]` matching the body.
3. A brief closing paragraph may name limits or open problems where appropriate.
"""

    if style == STYLE_REVIEW:
        review_block = """
## Review Mode（盲审审稿标准）

You are reviewing the user's own text as if it were a doctoral dissertation under blind review.
Be highly demanding; do NOT flatter.

Inspect: (1) thesis (2) logic (3) concepts (4) structure (5) evidence/citation (6) method (7) language (8) format.

For each major issue: brief quote or paraphrase from user draft → why it fails → concrete fix → optional rewrite.
Prioritize by severity.
When referring to corpus evidence in Sources, use the same `[n]` footnote convention.
"""
        output_block = """
## Output Structure (blind review; mandatory)

1. Overall verdict (2-4 sentences) and main risks.
2. High-severity issues.
3. Medium-severity issues.
4. Language / style。
5. Format / citation (use footnotes for the user's draft or corpus evidence where needed).
6. Prioritized revision plan。
7. Optional rewritten paragraph examples.
8. Footnote section localized to the required output language.
"""

    elif style == STYLE_CITE_PATCH:
        output_block = """
## Output Structure (citation patch; mandatory)

1. **Body**: reproduce the user's draft exactly, only inserting `(...quotation...)[n]` where support exists; do not polish or reorder it.
2. Footnote section localized to the required output language, matching every `[n]`.
"""

    elif style == STYLE_CONCEPT_MAP:
        output_block = """
## Output Structure (concept mapping; mandatory)

1. Opening note (not counted among the five sections): input keywords and corpus scope (single text / multiple texts).
2. At least five `##` sections: term definition and usage; argumentative role; links to other concepts; single-text tension or multi-text comparison; conceptual genealogy or summary of differences.
3. Each section must contain multiple developed paragraphs with dense `(...quotation...)[n]` evidence.
4. Footnote section localized to the required output language and matching the coverage requirements above.
"""
    elif style == STYLE_LITERATURE_REVIEW:
        output_block = """
## Output Structure (literature review; mandatory)

1. Research question and review scope: define the problem, corpus boundary, and criteria.
2. Research trajectory and stages: organize developments by period or problem-field.
3. Core controversies and positions: compare claims and disagreements across sources.
4. Methods, evidence, and argumentative quality: assess strengths and limits.
5. Gaps and future issues: provide actionable questions.
6. Footnote section localized to the required output language, using Sources' Cite as lines.
"""

    elif style == STYLE_ACADEMIC:
        output_block = """
## Output Structure (academic analysis; deprecated in UI)

Same as the default structure, with a more restrained tone; footnote format is unchanged.
"""

    elif style == STYLE_CONCISE:
        output_block = """
## Output Structure (concise answer; deprecated in UI)

Shorter main body plus a localized footnote section; do not omit footnote requirements.
"""

    context_block = context.strip() if context else ""
    if not context_block:
        context_block = (
            "(No excerpts retrieved — follow the no-retrieval rule in the required output language; do not fabricate sources.)"
        )

    hb = _clip(history_block, history_max_chars)
    wb = _clip(user_wiki_block, wiki_max_chars)
    cn = (conflict_note or "").strip()
    cgraph = (concept_graph_block or "").strip()
    cidx = (concept_index_block or "").strip()

    footnote_block = _footnote_rules_block()
    length_block = _length_and_coverage_block(style)

    conflict_section = ""
    if cn:
        conflict_section = f"**Conflict hint (heuristic):** {cn}\n\n---\n"

    prompt = f"""
You are an elite philosophical researcher writing for publication.

Your primary task depends on Style mode below, using the provided Sources (excerpts) as evidence unless Sources are empty.

{style_block}

---

{length_block}

---

{footnote_block}

---

## Primary Objective

Use the provided excerpts as the main evidential basis. When excerpts are insufficient, state so clearly before any general knowledge.

---

## Evidence Priority Rule

1. Treat excerpts as primary evidence.
2. Do not attribute claims to texts unless clearly supported.
3. Never fabricate textual evidence or page numbers.
4. When excerpts are insufficient: distinguish (a) excerpt-supported from (b) general knowledge.

---

## Text Coverage Requirement

When multiple excerpts appear: synthesize where relevant; explain disagreements between sources explicitly.

---

## Comparative Interpretation

When texts disagree: compare positions and conceptual differences; do not conflate incompatible views.

---

{review_block}

---

## Language Rule

MANDATORY LANGUAGE OUTPUT: {required_language}
This requirement overrides style names, examples, retrieved excerpt language, user wiki, and conversation history.
Write all narrative text, headings, section labels, review labels, no-retrieval notices, and footnote explanations in the mandatory language.
Keep quoted source text in its original language when quoting, but explain it in the mandatory language.
For English output, do not use Chinese section titles such as `正文` or `脚注`; use English labels such as `Main body` and `Footnotes`.
For Simplified Chinese output, use 简体中文 for explanations and headings.

---

## Output Structure

{output_block}

---

## Context Priority (Mandatory)

Resolve information in this strict order when they conflict:

1. **Sources (below)** — sole basis for factual claims, quotations, and footnotes `[n]`.
2. **Concept index / graph (if present)** — structured hints only; not citable unless the same claim appears in Sources.
3. **User wiki + conversation history** — continuity, preferences, and working definitions; **never** override Sources for evidence.

If User Context appears to contradict Sources, **follow Sources for citations** and add **one short sentence** naming the tension (do not hide the conflict).

{conflict_section}

## User Context (Non-evidence)

The following blocks are for *conversation continuity, preferences, and optional concept hints only*.
They are NOT evidence and MUST NOT be cited as sources. All citations/footnotes MUST come from `Sources:` excerpts.

User wiki (stable profile; may be empty):
{wb or "(empty)"}

Conversation history (recent turns; may be empty):
{hb or "(empty)"}

Concept graph (structured relations; may be empty):
{cgraph or "(empty)"}

Concept layer index (separate vector domain; non-citable; may be empty):
{cidx or "(empty)"}

---

Sources:
{context_block}

User input:
{question}

Answer:
"""
    return prompt
