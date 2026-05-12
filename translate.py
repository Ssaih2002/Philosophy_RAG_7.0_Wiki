from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from src.translation_pipeline import (
    export_project,
    list_translation_projects,
    load_glossary,
    load_global_glossary,
    load_project,
    prepare_translation_project,
    project_paths,
    run_translation,
    save_glossary,
    save_global_glossary,
    translate_project,
)


def _ensure_stdio_utf8() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _print_event(ev: Dict[str, Any]) -> None:
    typ = ev.get("type")
    if typ == "load_start":
        print(f"[load] {ev.get('source_path')}")
    elif typ == "chunks_ready":
        print(f"[chunks] chapters={ev.get('chapters')} chunks={ev.get('chunks')}")
    elif typ == "overview_start":
        print("[overview] calling LLM...")
    elif typ == "overview_done":
        print(f"[overview] done model={ev.get('model_used')} terms={ev.get('terms')}")
    elif typ == "translate_chunk_start":
        print(f"[translate] ch{ev.get('chapter_index')} chunk{ev.get('chunk_index')}...")
    elif typ == "translate_chunk_done":
        print(
            f"[translate] {ev.get('translated_chunks')}/{ev.get('total_chunks')} "
            f"model={ev.get('model_used')}"
        )
    elif typ == "prepared":
        print(f"[prepared] project_id={ev.get('project_id')}")


def cmd_prepare(args: argparse.Namespace) -> None:
    state = prepare_translation_project(
        args.source,
        target_language=args.target,
        provider=args.provider,
        model=args.model,
        project_id=args.project_id,
        chunk_chars=args.chunk_chars,
        call_llm=not args.no_llm,
        emit=_print_event,
    )
    paths = project_paths(state["project_id"])
    print(f"Project: {state['project_id']}")
    print(f"State: {paths.state}")
    print(f"Draft glossary: {paths.glossary_draft}")
    print("Review the draft glossary, then confirm with:")
    print(f"  python translate.py confirm-glossary {state['project_id']}")


def cmd_show(args: argparse.Namespace) -> None:
    state = load_project(args.project_id)
    print(json.dumps(state, ensure_ascii=False, indent=2))


def cmd_list(args: argparse.Namespace) -> None:
    projects = list_translation_projects()
    if not projects:
        print("No translation projects.")
        return
    for p in projects:
        prog = p.get("progress") or {}
        print(
            f"{p.get('project_id')} | {p.get('status')} | {p.get('target_language')} | "
            f"{prog.get('translated_chunks', 0)}/{prog.get('total_chunks', 0)} | {p.get('source_name')}"
        )


def cmd_glossary(args: argparse.Namespace) -> None:
    glossary = load_glossary(args.project_id, confirmed=not args.draft)
    print(json.dumps(glossary, ensure_ascii=False, indent=2))


def cmd_confirm_glossary(args: argparse.Namespace) -> None:
    if args.file:
        glossary = json.loads(Path(args.file).read_text(encoding="utf-8"))
    else:
        glossary = load_glossary(args.project_id, confirmed=False)
    saved = save_glossary(args.project_id, glossary)
    print(f"Confirmed {len(saved.get('terms') or [])} terms for project {args.project_id}.")
    print("Merged into long-term global glossary: data/translations/global_glossary.json")


def cmd_global_glossary(args: argparse.Namespace) -> None:
    glossary = load_global_glossary(target_language=args.target)
    print(json.dumps(glossary, ensure_ascii=False, indent=2))


def cmd_save_global_glossary(args: argparse.Namespace) -> None:
    glossary = json.loads(Path(args.file).read_text(encoding="utf-8"))
    saved = save_global_glossary(glossary)
    print(f"Saved global glossary with {len(saved.get('terms') or [])} terms.")


def cmd_translate(args: argparse.Namespace) -> None:
    state = translate_project(
        args.project_id,
        provider=args.provider,
        model=args.model,
        max_chunks=args.max_chunks,
        concurrency=args.concurrency,
        resume=not args.no_resume,
        emit=_print_event,
    )
    prog = state.get("progress") or {}
    print(f"Status: {state.get('status')} ({prog.get('translated_chunks', 0)}/{prog.get('total_chunks', 0)})")


def cmd_export(args: argparse.Namespace) -> None:
    out = export_project(args.project_id, output_format=args.format, output_path=args.output or "")
    print(f"Exported: {out}")


def cmd_run(args: argparse.Namespace) -> None:
    state = run_translation(
        args.source,
        target_language=args.target,
        provider=args.provider,
        model=args.model,
        output_format=args.format,
        project_id=args.project_id,
        concurrency=args.concurrency,
        emit=_print_event,
    )
    print(f"Project: {state.get('project_id')}")
    print(f"Exported: {state.get('last_export')}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Long-form translation pipeline for Philosophy_UP.")
    sub = p.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="Load a document, generate overview and draft glossary.")
    prep.add_argument("source", help="Path to .pdf/.docx/.json/.epub")
    prep.add_argument("--target", default="zh-CN", help="Target language, e.g. zh-CN, en, de")
    prep.add_argument("--provider", default="gemini", choices=["gemini", "openai", "deepseek", "auto"])
    prep.add_argument("--model", default="", help="Model id; empty uses project defaults")
    prep.add_argument("--project-id", default="", help="Optional stable project id")
    prep.add_argument("--chunk-chars", type=int, default=3600)
    prep.add_argument("--no-llm", action="store_true", help="Only load and chunk; skip overview/glossary LLM call")
    prep.set_defaults(func=cmd_prepare)

    show = sub.add_parser("show", help="Print project state JSON.")
    show.add_argument("project_id")
    show.set_defaults(func=cmd_show)

    ls = sub.add_parser("list", help="List translation projects.")
    ls.set_defaults(func=cmd_list)

    gloss = sub.add_parser("glossary", help="Print glossary JSON.")
    gloss.add_argument("project_id")
    gloss.add_argument("--draft", action="store_true", help="Show draft glossary even if confirmed exists")
    gloss.set_defaults(func=cmd_glossary)

    confirm = sub.add_parser("confirm-glossary", help="Confirm draft or edited glossary before translation.")
    confirm.add_argument("project_id")
    confirm.add_argument("--file", default="", help="Optional edited glossary JSON file")
    confirm.set_defaults(func=cmd_confirm_glossary)

    gg = sub.add_parser("global-glossary", help="Print the long-term global glossary JSON.")
    gg.add_argument("--target", default="", help="Optional target language filter, e.g. zh-CN")
    gg.set_defaults(func=cmd_global_glossary)

    sgg = sub.add_parser("save-global-glossary", help="Replace the long-term global glossary from a JSON file.")
    sgg.add_argument("file", help="Path to edited global glossary JSON")
    sgg.set_defaults(func=cmd_save_global_glossary)

    tr = sub.add_parser("translate", help="Translate pending chunks.")
    tr.add_argument("project_id")
    tr.add_argument("--provider", default="", choices=["", "gemini", "openai", "deepseek", "auto"])
    tr.add_argument("--model", default="")
    tr.add_argument("--max-chunks", type=int, default=0, help="Limit chunks for smoke tests; 0 means all")
    tr.add_argument("--concurrency", type=int, default=1, help="Parallel segment workers, 1-20")
    tr.add_argument("--no-resume", action="store_true", help="Re-translate chunks even if cached")
    tr.set_defaults(func=cmd_translate)

    exp = sub.add_parser("export", help="Export translated chunks to txt/docx.")
    exp.add_argument("project_id")
    exp.add_argument("--format", default="txt", choices=["txt", "docx"])
    exp.add_argument("--output", default="")
    exp.set_defaults(func=cmd_export)

    run = sub.add_parser("run", help="Prepare, auto-confirm draft glossary, translate, and export.")
    run.add_argument("source", help="Path to .pdf/.docx/.json/.epub")
    run.add_argument("--target", default="zh-CN")
    run.add_argument("--provider", default="gemini", choices=["gemini", "openai", "deepseek", "auto"])
    run.add_argument("--model", default="")
    run.add_argument("--format", default="txt", choices=["txt", "docx"])
    run.add_argument("--project-id", default="")
    run.add_argument("--concurrency", type=int, default=1, help="Parallel segment workers, 1-20")
    run.set_defaults(func=cmd_run)

    return p


def main() -> None:
    _ensure_stdio_utf8()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
