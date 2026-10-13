"""RAG 评测器：量化检索质量，回答"我的改动到底有没有用"。

指标：
  recall@k   —— 正确条款是否出现在前 k 条召回里（检索召回率）
  MRR        —— 第一个正确结果排名的倒数均值（排序质量）
  引用正确率 —— 返回的引用中，有多少确实对应该问题的 gold 条款
  幻觉率     —— 答案中的关键数字未在引用原文中出现的比例
  关键词覆盖率 —— gold_keywords 被答案覆盖的比例（回答完整性）

关键点：recall@k 和 MRR 是检索层指标，关键词覆盖率和幻觉率是生成层指标，
必须分开看。检索对了但答案没提，等于白干——这是新手最容易忽略的地方。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .retrieval import Answer, GroundedQA, Index, Retriever


@dataclass
class CaseResult:
    case_id: str
    question: str
    category: str
    difficulty: str
    gold_sections: list[str]
    hit_sections: list[str]
    reciprocal_rank: float
    recall_at_1: int
    recall_at_3: int
    recall_at_5: int
    keyword_coverage: float
    citation_precision: float
    hallucinated: int
    grounded: bool
    top_answer: str


@dataclass
class EvalReport:
    mode: str
    top_k: int
    total: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    keyword_coverage: float
    citation_precision: float
    hallucination_rate: float
    by_difficulty: dict[str, dict[str, float]]
    results: list[CaseResult]

    def summary(self) -> str:
        lines = [
            f"检索模式: {self.mode}   top_k={self.top_k}   用例数={self.total}",
            "-" * 62,
            f"  recall@1              {self.recall_at_1:.3f}",
            f"  recall@3              {self.recall_at_3:.3f}",
            f"  recall@5              {self.recall_at_5:.3f}",
            f"  MRR                   {self.mrr:.3f}",
            f"  引用正确率             {self.citation_precision:.3f}",
            f"  关键词覆盖率(生成质量) {self.keyword_coverage:.3f}",
            f"  数字幻觉率            {self.hallucination_rate:.3f}",
            "-" * 62,
            "分难度 recall@5:",
        ]
        for diff, m in sorted(self.by_difficulty.items()):
            lines.append(f"  {diff:<10} {m['recall_at_5']:.3f}   (n={int(m['n'])})")
        lines.append("")
        lines.append("按类目 recall@5:")
        for cat, m in sorted(self.by_category.items()):
            lines.append(f"  {cat:<10} {m['recall_at_5']:.3f}   (n={int(m['n'])})")
        return "\n".join(lines)


def _norm(s: str) -> str:
    return s.replace(" ", "").replace("　", "")


def evaluate(
    index: Index,
    cases: Sequence[dict],
    mode: str = "hybrid",
    top_k: int = 5,
) -> EvalReport:
    retr = Retriever(index)
    qa = GroundedQA(index, retr)
    results: list[CaseResult] = []

    for c in cases:
        gold = set(c["gold"])
        hits = retr.search(c["question"], top_k=max(top_k, 5), mode=mode)
        hit_secs = [h.chunk.section_no for h in hits if h.chunk.section_no]

        rr = 0.0
        for i, h in enumerate(hits, start=1):
            if h.chunk.section_no in gold:
                rr = 1.0 / i
                break

        ans: Answer = qa.answer(c["question"], top_k=top_k)
        ans_norm = _norm(ans.answer)
        kws = c.get("gold_keywords", [])
        covered = sum(1 for k in kws if _norm(k) in ans_norm)
        kw_cov = covered / len(kws) if kws else 1.0

        cited_secs = [h.chunk.section_no for h in ans.hits if h.chunk.section_no]
        cite_hit = sum(1 for s in cited_secs if s in gold)
        cite_prec = cite_hit / len(cited_secs) if cited_secs else 0.0

        hallucinated = 0 if ans.grounded else 1

        results.append(
            CaseResult(
                case_id=c["id"],
                question=c["question"],
                category=c.get("category", "-"),
                difficulty=c.get("difficulty", "-"),
                gold_sections=sorted(gold),
                hit_sections=hit_secs[:5],
                reciprocal_rank=rr,
                recall_at_1=int(bool(hit_secs[:1] and hit_secs[0] in gold)),
                recall_at_3=int(bool(set(hit_secs[:3]) & gold)),
                recall_at_5=int(bool(set(hit_secs[:5]) & gold)),
                keyword_coverage=kw_cov,
                citation_precision=cite_prec,
                hallucinated=hallucinated,
                grounded=ans.grounded,
                top_answer=ans.answer[:80],
            )
        )

    n = max(len(results), 1)

    def agg(key: str) -> float:
        return sum(getattr(r, key) for r in results) / n

    by_diff: dict[str, dict[str, float]] = {}
    for r in results:
        b = by_diff.setdefault(r.difficulty, {"recall_at_5": 0.0, "n": 0.0})
        b["recall_at_5"] += r.recall_at_5
        b["n"] += 1
    for b in by_diff.values():
        b["recall_at_5"] /= b["n"]

    by_cat: dict[str, dict[str, float]] = {}
    for r in results:
        b = by_cat.setdefault(r.category, {"recall_at_5": 0.0, "n": 0.0})
        b["recall_at_5"] += r.recall_at_5
        b["n"] += 1
    for b in by_cat.values():
        b["recall_at_5"] /= b["n"]

    rep = EvalReport(
        mode=mode,
        top_k=top_k,
        total=len(results),
        recall_at_1=agg("recall_at_1"),
        recall_at_3=agg("recall_at_3"),
        recall_at_5=agg("recall_at_5"),
        mrr=agg("reciprocal_rank"),
        keyword_coverage=agg("keyword_coverage"),
        citation_precision=agg("citation_precision"),
        hallucination_rate=sum(r.hallucinated for r in results) / n,
        by_difficulty=by_diff,
        results=results,
    )
    rep.by_category = by_cat  # type: ignore[attr-defined]
    return rep


def load_cases(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["cases"]
