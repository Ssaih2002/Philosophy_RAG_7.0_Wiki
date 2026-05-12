from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime
from typing import Any, Dict, List


def _ensure_stdio_utf8() -> None:
    """在 Windows 控制台使用 UTF-8，避免中文路径与 JSON 摘要乱码（Python 3.7+）。"""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

from src.rag_engine import RAGEngine

from .discrimination import ablation_discrimination_report, gold_lift_summary
from .runner import EvalItem, load_items, run_retrieval_only, write_jsonl


def main() -> None:
    _ensure_stdio_utf8()
    p = argparse.ArgumentParser(description="Philosophy RAG batch evaluation.")
    p.add_argument("--input", "-i", required=True, help="JSONL 评测集")
    p.add_argument(
        "--output",
        "-o",
        required=True,
        help="主输出 JSONL 文件名（与 --out-dir 联用时只写文件名即可，如 result.jsonl）",
    )
    p.add_argument(
        "--out-dir",
        default="",
        metavar="DIR",
        help="若指定，则写入 DIR/<运行子目录>/<output 的文件名>；子目录名由 --run-name 决定，默认可自动生成时间戳，避免多次运行互相覆盖",
    )
    p.add_argument(
        "--run-name",
        default="",
        metavar="NAME",
        help="与 --out-dir 联用：子目录名；省略则用时间戳 YYYYMMDD_HHMMSS",
    )
    p.add_argument(
        "--suite",
        default="ablation",
        choices=("ablation", "terminology"),
        help="ablation=多检索模式; terminology=同 ablation 但仅 experiment=terminology，默认每条 30 片段",
    )
    p.add_argument(
        "--modes",
        default="dense_only,sparse_only,merge_no_rrf,full_rrf",
        help="ablation/terminology: 逗号分隔；默认即 Baseline1/2/Hybrid/Full（dense_only, sparse_only, merge_no_rrf, full_rrf）。亦可用别名 baseline1,baseline2,hybrid,full",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=None,
        metavar="N",
        help="每条返回片段条数上限；省略时消融默认 10、术语默认 30",
    )
    p.add_argument(
        "--doc-preview-chars",
        type=int,
        default=400,
        metavar="N",
        help="每条片段写入 JSON 的正文字符数；0 表示写入全文（文件会变大，便于人工审阅）",
    )
    p.add_argument("--experiment-filter", default="", help="仅保留指定 experiment 字段的行")
    args = p.parse_args()

    top_n = args.top_n if args.top_n is not None else (30 if args.suite == "terminology" else 10)

    if args.out_dir:
        run_name = args.run_name.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
        out_base = os.path.join(os.path.normpath(args.out_dir), run_name)
        os.makedirs(out_base, exist_ok=True)
        output_path = os.path.join(out_base, os.path.basename(args.output))
    else:
        output_path = os.path.normpath(args.output)
        out_base = os.path.dirname(output_path) or "."
        if out_base and out_base != ".":
            os.makedirs(out_base, exist_ok=True)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    items = load_items(args.input)
    if args.experiment_filter:
        items = [x for x in items if x.experiment == args.experiment_filter]
    if args.suite == "terminology":
        items = [x for x in items if x.experiment == "terminology"]

    engine = RAGEngine()
    out_rows: List[Dict[str, Any]] = []

    if args.suite in ("ablation", "terminology"):
        for it in items:
            if not it.question.strip():
                continue
            for mode in modes:
                row = run_retrieval_only(
                    engine,
                    item=it,
                    mode=mode,
                    top_n=top_n,
                    doc_preview_chars=args.doc_preview_chars,
                )
                out_rows.append(row)

    write_jsonl(output_path, out_rows)

    hits = [r["metrics"]["hit@5"] for r in out_rows if r.get("metrics", {}).get("hit@5") is not None]
    mrrs = [r["metrics"]["mrr"] for r in out_rows if r.get("metrics", {}).get("mrr") is not None]
    summary: Dict[str, Any] = {"suite": args.suite, "n_rows": len(out_rows)}
    if hits:
        summary["mean_hit@5_global"] = statistics.mean(hits)
    if mrrs:
        summary["mean_mrr_global"] = statistics.mean(mrrs)

    if args.suite in ("ablation", "terminology"):
        disc = ablation_discrimination_report(out_rows, k=min(10, max(1, top_n)))
        summary["hybrid_discrimination"] = {
            k: v
            for k, v in disc.items()
            if k not in ("per_question", "interpretation_low_jaccard", "interpretation_sparse_fraction", "interpretation_hybrid_lift_proxy", "interpretation_rrf_vs_concat")
        }
        summary["hybrid_discrimination_notes"] = {
            "low_jaccard_full_vs_dense": disc.get("interpretation_low_jaccard"),
            "sparse_overlap": disc.get("interpretation_sparse_fraction"),
            "full_not_in_dense": disc.get("interpretation_hybrid_lift_proxy"),
            "merge_vs_rrf": disc.get("interpretation_rrf_vs_concat"),
        }
        gl = gold_lift_summary(out_rows)
        if gl.get("mean_hit5_lift_full_rrf_minus_dense") is not None or gl.get(
            "mean_mrr_lift_full_rrf_minus_dense"
        ) is not None:
            summary["gold_lift_full_rrf_vs_dense"] = gl
        with open(output_path + ".discrimination.json", "w", encoding="utf-8") as f:
            json.dump(disc, f, ensure_ascii=False, indent=2)

    with open(output_path + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("输出文件：")
    print(" ", output_path)
    print(" ", output_path + ".summary.json")
    if args.suite in ("ablation", "terminology"):
        print(" ", output_path + ".discrimination.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
