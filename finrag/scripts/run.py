"""命令行入口。

用法：
  python scripts/run.py build                    构建索引
  python scripts/run.py ask "出差住宿费标准"       单次问答（带引用溯源）
  python scripts/run.py eval --mode hybrid        全量评测
  python scripts/run.py compare                   三种检索模式对比
  python scripts/run.py ablate                    分块策略消融实验
  python scripts/run.py ask --json                输出 JSON，便于接入其他系统
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finrag import (  # noqa: E402
    Index,
    Retriever,
    evaluate,
    load_cases,
    load_documents,
    structure_aware_chunks,
)
from finrag.api_embedder import build_embedder  # noqa: E402

DOCS = ROOT / "docs"
EVAL = ROOT / "data" / "eval_set.json"
INDEX_DIR = ROOT / "artifacts" / "index"


def _index(kind: str) -> Index:
    return Index.build(load_documents(DOCS), embedder=build_embedder(kind))


def cmd_build(args: argparse.Namespace) -> None:
    emb = build_embedder(args.embedder)
    chunks = load_documents(DOCS)
    idx = Index.build(chunks, embedder=emb)
    idx.save(INDEX_DIR)
    print(f"文档数: {len({c.doc_id for c in chunks})}")
    print(f"分块数: {len(chunks)}")
    print(f"向量维度: {emb.dim}")
    print(f"索引已保存: {INDEX_DIR}")


def cmd_ask(args: argparse.Namespace) -> None:
    idx = _index(args.embedder)
    retr = Retriever(idx)
    hits = retr.search(args.question, top_k=args.top_k, mode=args.mode)

    if args.json:
        print(
            json.dumps(
                {
                    "question": args.question,
                    "mode": args.mode,
                    "results": [
                        {
                            "rank": i + 1,
                            "score": round(h.rrf_score, 5),
                            "bm25_rank": h.bm25_rank,
                            "vec_rank": h.vec_rank,
                            "section": h.chunk.section_no,
                            "heading": h.chunk.heading,
                            "doc": h.chunk.doc_title,
                            "excerpt": h.chunk.text[:200],
                        }
                        for i, h in enumerate(hits)
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print(f"问题: {args.question}")
    print(f"检索模式: {args.mode}   返回 {len(hits)} 条\n" + "=" * 66)
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h.chunk.doc_title} > {h.chunk.heading}"
              f"   条款号={h.chunk.section_no or '-'}"
              f"  RRF={h.rrf_score:.5f}")
        print(f"    {h.chunk.text[:120]}...")
        print()


def cmd_eval(args: argparse.Namespace) -> None:
    idx = _index(args.embedder)
    cases = load_cases(EVAL)
    rep = evaluate(idx, cases, mode=args.mode, top_k=args.top_k)
    print(rep.summary())

    out = ROOT / "artifacts" / f"eval_{args.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "metrics": {
                    "recall_at_1": rep.recall_at_1,
                    "recall_at_3": rep.recall_at_3,
                    "recall_at_5": rep.recall_at_5,
                    "mrr": rep.mrr,
                    "keyword_coverage": rep.keyword_coverage,
                    "citation_precision": rep.citation_precision,
                    "hallucination_rate": rep.hallucination_rate,
                },
                "cases": [r.__dict__ for r in rep.results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n明细已写入: {out}")


def cmd_compare(args: argparse.Namespace) -> None:
    idx = _index(args.embedder)
    cases = load_cases(EVAL)
    rows = []
    print(f"{'模式':<10}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'MRR':>8}{'关键词覆盖':>12}")
    print("-" * 56)
    for mode in ("bm25", "vector", "hybrid"):
        r = evaluate(idx, cases, mode=mode, top_k=args.top_k)
        rows.append(
            {
                "mode": mode,
                "recall@1": r.recall_at_1,
                "recall@3": r.recall_at_3,
                "recall@5": r.recall_at_5,
                "mrr": r.mrr,
                "keyword_coverage": r.keyword_coverage,
            }
        )
        print(
            f"{mode:<10}{r.recall_at_1:>8.3f}{r.recall_at_3:>8.3f}"
            f"{r.recall_at_5:>8.3f}{r.mrr:>8.3f}{r.keyword_coverage:>12.3f}"
        )
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "mode_compare.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def cmd_ablate(args: argparse.Namespace) -> None:
    """分块策略消融：max_len 与 min_len 对召回的影响。

    这是 RAG 调优里最常见的第一个实验——分块粒度直接决定召回上限。
    """
    from finrag import HashingEmbedder

    cases = load_cases(EVAL)
    print(f"{'max_len':>9}{'min_len':>9}{'块数':>7}{'R@1':>9}{'R@5':>9}{'MRR':>9}")
    print("-" * 52)
    results = []
    for min_len in (20, 40, 60):
        for max_len in (300, 600, 900):
            chunks: list = []
            for p in sorted(DOCS.glob("*.md")):
                text = p.read_text(encoding="utf-8")
                title = text.split("\n", 1)[0].lstrip("# ").strip() or p.stem
                chunks.extend(
                    structure_aware_chunks(
                        text, doc_id=p.stem, doc_title=title,
                        min_len=min_len, max_len=max_len,
                    )
                )
            if not chunks:
                continue
            idx = Index.build(chunks, embedder=HashingEmbedder())
            rep = evaluate(idx, cases, mode="bm25", top_k=args.top_k)
            results.append(
                {
                    "min_len": min_len,
                    "max_len": max_len,
                    "n_chunks": len(chunks),
                    "recall@1": rep.recall_at_1,
                    "recall@5": rep.recall_at_5,
                    "mrr": rep.mrr,
                }
            )
            print(
                f"{max_len:>9}{min_len:>9}{len(chunks):>7}"
                f"{rep.recall_at_1:>9.3f}{rep.recall_at_5:>9.3f}{rep.mrr:>9.3f}"
            )
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "ablation_chunking.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    best = max(results, key=lambda x: (x["recall@5"], x["mrr"]))
    print(f"\n最优配置: min_len={best['min_len']} max_len={best['max_len']}"
          f"  R@5={best['recall@5']:.3f} MRR={best['mrr']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="财务制度 RAG 检索系统")
    ap.add_argument("--embedder", default="auto", choices=["auto", "api", "local", "hash"])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build").set_defaults(func=cmd_build)

    p_ask = sub.add_parser("ask")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=5)
    p_ask.add_argument("--mode", default="hybrid", choices=["bm25", "vector", "hybrid"])
    p_ask.add_argument("--json", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--mode", default="hybrid", choices=["bm25", "vector", "hybrid"])
    p_eval.add_argument("--top-k", type=int, default=5)
    p_eval.set_defaults(func=cmd_eval)

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--top-k", type=int, default=5)
    p_cmp.set_defaults(func=cmd_compare)

    p_abl = sub.add_parser("ablate")
    p_abl.add_argument("--top-k", type=int, default=5)
    p_abl.set_defaults(func=cmd_ablate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
