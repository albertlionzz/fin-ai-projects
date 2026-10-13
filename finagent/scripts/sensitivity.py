"""scale 与权重的敏感性分析。

回答的问题
----------
README 里有一条诚实清单写着"风险分权重是手工设定，没有数据支撑，应做敏感性分析"。
这个脚本就是把那句话兑现：**给出"分数在什么参数下会改变结论"的边界。**

为什么衡量指标是「档位翻转」而不是「分数变化」
------------------------------------------
分数是连续量，改任何参数它都会变——"分数变了"这句话信息量为零。
真正决定行动的是**档位**（→ 建议动作）：极高=暂缓出具结论，高=专项函证，
中=书面问询，低=常规复核。参数敏感性要回答的是：

    在多大的参数扰动范围内，建议动作保持不变？

超出这个范围，说明当前结论对参数敏感，那么"参数取 20"这件事
就不能只靠经验值撑着，必须补历史样本回测。

网格设计
--------
两个维度：
    scale       饱和函数尺度，{10, 15, 20, 25, 30, 40, 50}
    w_high/w_warn  重大与关注的权重比，{1.5, 2.0, 2.5, 3.0, 5.0}

`scale` 和 `w_high/w_warn` 不是等价的：前者改变整条曲线的形状，
后者只改变不同严重度之间的相对贡献。分开扫才能看出是哪一类参数在驱动翻转。

用法
----
    python scripts/sensitivity.py                     # 控制台摘要
    python scripts/sensitivity.py --md artifacts/sensitivity.md
    python scripts/sensitivity.py --json-out artifacts/sensitivity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finagent import (  # noqa: E402
    AnalysisResult,
    Finding,
    Level,
    Orchestrator,
    StatementSeries,
    load_benchmarks,
    load_pair,
    load_statement,
)
from finagent.models import DEFAULT_POLICY, ScorePolicy  # noqa: E402


SCALES: tuple[float, ...] = (10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0)
WEIGHT_RATIOS: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 5.0)

GRADE_ORDER = ["低", "中", "高", "极高"]


# ---------------------------------------------------------------- 场景构建


def _scenario(
    name: str,
    note: str,
    cur_path: str,
    prev_path: str,
    period_paths: Sequence[str] | None = None,
    benchmark_path: str | None = None,
) -> dict[str, Any]:
    """构造一个敏感性场景：一次完整的分析结果 + 说明。

    场景是**已经跑完的分析结果**，不是原始数据。这样敏感性分析
    只在评分函数上做网格，不会因为重跑规则引擎而引入无关变量。
    """
    cur, prev = load_pair(cur_path, prev_path)
    series: StatementSeries | None = None
    if period_paths:
        series = StatementSeries(statements=[load_statement(p) for p in period_paths])
        cur, prev = series.latest, series.previous

    bm = load_benchmarks(benchmark_path) if benchmark_path else None
    orch = Orchestrator()
    if bm is not None:
        orch.audit.engine.benchmarks = bm
    result = orch.analyze(cur, prev, series=series)

    return {
        "name": name,
        "note": note,
        "result": result,
        "baseline_grade": result.risk_grade(),
        "baseline_score": round(result.risk_score(), 2),
        "n_findings": len(result.findings),
        "n_high": result.high_count,
        "n_warn": result.warn_count,
    }


def _synthetic_boundary_scenario() -> dict[str, Any]:
    """合成边界场景：5 项关注、0 项重大。

    为什么要有一个合成场景：前三个真实场景的档位都被**严重度维度**主导
    （有重大 → 不低于高；无重大 → 中），分数维度的切点根本没能上场，
    跑出来的"100% 稳定"有一部分是设计使然，不是参数稳健的证据。

    真正检验分数切点的是**贴着切点的样本**。5 项关注在默认参数下
    raw = 5×4 = 20，得分 63.2，距「高」线 65 只差 1.8 分——
    这是全网格里最可能翻档的构型。它不是真实公司，是探针。
    """
    findings = [
        Finding(
            rule_id=f"SYNTH_WARN_{i}",
            level=Level.WARN,
            title=f"合成关注项 {i}",
            detail="用于参数敏感性探针的合成发现。",
        )
        for i in range(1, 6)
    ]
    result = AnalysisResult(
        company="合成边界样本",
        period="探针",
        findings=findings,
    )
    return {
        "name": "合成边界（5 项关注，贴 65 分切点）",
        "note": "合成探针，非真实公司。用于隔离分数维度的影响——"
                "真实场景的档位多由严重度维度决定，测不出切点敏感性。",
        "result": result,
        "baseline_grade": result.risk_grade(),
        "baseline_score": round(result.risk_score(), 2),
        "n_findings": len(result.findings),
        "n_high": result.high_count,
        "n_warn": result.warn_count,
    }


def build_scenarios() -> list[dict[str, Any]]:
    """默认场景集：三类典型形态 + 一个空白对照 + 一个分数维度探针。"""
    st = ROOT / "data" / "statements"
    bmp = ROOT / "data" / "benchmarks" / "industry_benchmarks.json"
    return [
        _scenario(
            name="示例制造 2025A（7 类异常全命中）",
            note="极端情形：重大 4 项。用于检验高分段是否稳定——"
                 "若这类场景都会因参数翻档，说明评分函数在最需要区分的区间失效。",
            cur_path=st / "示例制造_2025A.json",
            prev_path=st / "示例制造_2024A.json",
        ),
        _scenario(
            name="长恒电子 2025A（4 期慢性恶化）",
            note="中等情形：重大 2 项、关注 0 项。这是饱和函数得分天然偏低的区间"
                 "（重大项少），最可能因 scale 变化而翻档。",
            cur_path=st / "长恒电子_2025A.json",
            prev_path=st / "长恒电子_2024A.json",
            period_paths=[
                st / "长恒电子_2022A.json",
                st / "长恒电子_2023A.json",
                st / "长恒电子_2024A.json",
                st / "长恒电子_2025A.json",
            ],
        ),
        _scenario(
            name="恒润重工 2025A（行业校准后 2 项关注）",
            note="轻微情形：0 项重大、2 项关注。无重大异常时档位完全由分数决定，"
                 "是检验分数档切点的纯净样本。",
            cur_path=st / "恒润重工_2025A.json",
            prev_path=st / "恒润重工_2024A.json",
            benchmark_path=str(bmp),
        ),
        _synthetic_boundary_scenario(),
    ]


# ---------------------------------------------------------------- 网格计算


def _policies() -> list[ScorePolicy]:
    """网格参数组合。

    权重只扫**比值**不扫绝对值：w_high 与 w_warn 同时翻倍，
    饱和函数的输入 raw 也翻倍，等价于把 scale 减半——两个维度会重复计数。
    扫比值才是正交的。
    """
    out: list[ScorePolicy] = []
    for ratio in WEIGHT_RATIOS:
        w_warn = 4.0
        w_high = w_warn * ratio
        for scale in SCALES:
            out.append(
                replace(
                    DEFAULT_POLICY,
                    scale=scale,
                    w_high=w_high,
                    w_warn=w_warn,
                )
            )
    return out


def sweep(scenarios: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """对每个场景跑完整参数网格，返回每格的分数与档位。"""
    rows: list[dict[str, Any]] = []
    for sc in scenarios:
        result: AnalysisResult = sc["result"]
        cells: list[dict[str, Any]] = []
        n_flip = 0
        for p in _policies():
            score = result.risk_score(p)
            grade = result.risk_grade(p)
            flipped = grade != sc["baseline_grade"]
            if flipped:
                n_flip += 1
            cells.append(
                {
                    "scale": p.scale,
                    "ratio": round(p.w_high / p.w_warn, 2),
                    "score": round(score, 2),
                    "grade": grade,
                    "flipped": flipped,
                }
            )
        total = len(cells)
        rows.append(
            {
                "name": sc["name"],
                "note": sc["note"],
                "baseline_grade": sc["baseline_grade"],
                "baseline_score": sc["baseline_score"],
                "n_findings": sc["n_findings"],
                "n_high": sc["n_high"],
                "n_warn": sc["n_warn"],
                "n_cells": total,
                "n_flip": n_flip,
                "stability": round(1.0 - n_flip / total, 4) if total else 1.0,
                "cells": cells,
            }
        )
    return rows


def flip_boundary(row: dict[str, Any]) -> dict[str, Any]:
    """找出单个场景的翻转边界：每个 scale 下，比值要多大才翻档。

    返回的结构直接回答"参数在什么范围内结论不变"：
        {"scale": 20.0, "safe_ratios": [1.5, 2.0, 2.5], "flip_from": 3.0}

    `flip_from` 是第一个翻档的比值；若整行都不翻，则为 None。
    """
    by_scale: dict[float, list[dict[str, Any]]] = {}
    for c in row["cells"]:
        by_scale.setdefault(c["scale"], []).append(c)

    boundaries: list[dict[str, Any]] = []
    for scale in sorted(by_scale):
        cells = sorted(by_scale[scale], key=lambda c: c["ratio"])
        safe = [c["ratio"] for c in cells if not c["flipped"]]
        flip_from = next((c["ratio"] for c in cells if c["flipped"]), None)
        boundaries.append(
            {
                "scale": scale,
                "safe_ratios": safe,
                "flip_from": flip_from,
                "grades": [c["grade"] for c in cells],
            }
        )
    return {"name": row["name"], "by_scale": boundaries}


def stable_region(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """全局稳健区间：在所有场景下都不翻档的参数组合。

    这是敏感性分析真正的产出。如果稳健区间为空，说明当前参数组合
    是"恰好能让这批样本得到当前结论"的，必须补样本回测——这不是缺陷，
    是必须承认的局限。
    """
    n_scen = len(rows)
    if n_scen == 0:
        return {"n_scenarios": 0, "stable_cells": [], "share": 0.0}

    # 用 (scale, ratio) 做键，统计每个组合在几个场景下翻档
    flip_count: dict[tuple[float, float], int] = {}
    for r in rows:
        for c in r["cells"]:
            key = (c["scale"], c["ratio"])
            flip_count[key] = flip_count.get(key, 0) + (1 if c["flipped"] else 0)

    total_cells = len(flip_count)
    stable = sorted(k for k, v in flip_count.items() if v == 0)
    return {
        "n_scenarios": n_scen,
        "n_cells": total_cells,
        "n_stable": len(stable),
        "share": round(len(stable) / total_cells, 4) if total_cells else 0.0,
        "stable_cells": [
            {"scale": s, "ratio": r, "is_default": s == DEFAULT_POLICY.scale and r == round(DEFAULT_POLICY.w_high / DEFAULT_POLICY.w_warn, 2)}
            for s, r in stable
        ],
    }


# ---------------------------------------------------------------- 输出


def render_console(rows: Sequence[dict[str, Any]], region: dict[str, Any]) -> str:
    L: list[str] = []
    L.append("=" * 78)
    L.append(" 风险分参数敏感性分析")
    L.append(" 衡量指标：档位是否翻转（分数永远在变，档位才决定建议动作）")
    L.append("=" * 78)
    L.append("")
    L.append(f"{'场景':<38}{'基准档位':<8}{'翻档格数':>10}{'稳定性':>10}")
    L.append("-" * 78)
    for r in rows:
        L.append(
            f"{r['name'][:36]:<38}{r['baseline_grade']:<8}"
            f"{r['n_flip']:>6}/{r['n_cells']:<4}{r['stability']:>9.0%}"
        )
    L.append("")

    L.append("各 scale 下的权重比安全区（不翻档的 w_high/w_warn 比值）")
    L.append("-" * 78)
    for r in rows:
        b = flip_boundary(r)
        parts = []
        for item in b["by_scale"]:
            if item["safe_ratios"]:
                parts.append(f"scale={item['scale']:g} → 比值 ≤ {max(item['safe_ratios']):g}")
            else:
                parts.append(f"scale={item['scale']:g} → 无安全区")
        L.append(f"  {r['name'][:34]}")
        L.append(f"    " + "；".join(parts))
        L.append("")

    L.append(f"全局稳健区间：{region['n_stable']}/{region['n_cells']} 个参数组合"
             f"（{region['share']:.0%}）在全部 {region['n_scenarios']} 个场景下不翻档")
    default_in = any(c.get("is_default") for c in region["stable_cells"])
    L.append(f"  当前默认参数（scale={DEFAULT_POLICY.scale:g}, "
             f"w_high/w_warn={DEFAULT_POLICY.w_high / DEFAULT_POLICY.w_warn:g}）"
             f"{'在稳健区间内' if default_in else '**不在稳健区间内**'}")
    return "\n".join(L)


def render_markdown(rows: Sequence[dict[str, Any]], region: dict[str, Any]) -> str:
    L: list[str] = []
    L.append("# 风险分参数敏感性分析")
    L.append("")
    L.append("> 衡量指标是**档位是否翻转**，不是分数变化。分数是连续量，")
    L.append("> 改任何参数它都会变；真正决定行动的是档位——极高=暂缓出具结论、")
    L.append("> 高=专项函证、中=书面问询、低=常规复核。")
    L.append("> 参数敏感性要回答的是：**在多大的参数扰动范围内，建议动作保持不变。**")
    L.append("")
    L.append(f"网格：scale ∈ {{{', '.join(f'{s:g}' for s in SCALES)}}}，"
             f"w_high/w_warn ∈ {{{', '.join(f'{r:g}' for r in WEIGHT_RATIOS)}}}"
             f"（{len(SCALES) * len(WEIGHT_RATIOS)} 个组合）")
    L.append("")

    L.append("## 一、场景与基准")
    L.append("")
    L.append("| 场景 | 发现数 | 重大 | 关注 | 基准分数 | 基准档位 |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        L.append(
            f"| {r['name']} | {r['n_findings']} | {r['n_high']} | {r['n_warn']} "
            f"| {r['baseline_score']:.2f} | {r['baseline_grade']} |"
        )
    L.append("")
    for r in rows:
        L.append(f"- **{r['name']}**：{r['note']}")
    L.append("")

    L.append("## 二、逐场景稳定性")
    L.append("")
    L.append("| 场景 | 基准档位 | 翻档格数 | 稳定性 |")
    L.append("|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['name']} | {r['baseline_grade']} | {r['n_flip']}/{r['n_cells']} | {r['stability']:.0%} |")
    L.append("")

    L.append("## 三、档位矩阵（scale × 权重比）")
    L.append("")
    for r in rows:
        L.append(f"### {r['name']}（基准 {r['baseline_grade']}）")
        L.append("")
        header = "| scale \\ 比值 | " + " | ".join(f"{x:g}" for x in WEIGHT_RATIOS) + " |"
        L.append(header)
        L.append("|" + "---|" * (len(WEIGHT_RATIOS) + 1))
        for s in SCALES:
            cells = [c for c in r["cells"] if c["scale"] == s]
            cells.sort(key=lambda c: c["ratio"])
            marks = []
            for c in cells:
                g = c["grade"]
                marks.append(f"**{g}**" if c["flipped"] else g)
            L.append(f"| {s:g} | " + " | ".join(marks) + " |")
        L.append("")
        L.append("> 加粗 = 该参数组合下档位与基准不同。")
        L.append("")

    L.append("## 四、翻转边界")
    L.append("")
    for r in rows:
        b = flip_boundary(r)
        L.append(f"**{r['name']}**")
        L.append("")
        if r["n_high"] == 0:
            L.append("> 本场景无重大异常，风险分只由关注项贡献，"
                     "`w_high/w_warn` 比值不进入计算——整行同值是数学必然，"
                     "不是稳定性证据。此场景检验的只有 scale 与切点。")
            L.append("")
        L.append("| scale | 不翻档的权重比 | 首个翻档比值 |")
        L.append("|---|---|---|")
        for item in b["by_scale"]:
            safe = "、".join(f"{x:g}" for x in item["safe_ratios"]) or "—"
            L.append(f"| {item['scale']:g} | {safe} | {item['flip_from'] if item['flip_from'] else '—'} |")
        L.append("")

    L.append("## 五、全局稳健区间")
    L.append("")
    L.append(f"- 参数组合总数：**{region['n_cells']}**")
    L.append(f"- 在全部 {region['n_scenarios']} 个场景下均不翻档：**{region['n_stable']}**"
             f"（{region['share']:.0%}）")
    if region["stable_cells"]:
        L.append("- 稳健组合：" + "、".join(
            f"scale={c['scale']:g}/比值={c['ratio']:g}" for c in region["stable_cells"]
        ))
    default_in = any(c.get("is_default") for c in region["stable_cells"])
    L.append("")
    L.append(f"当前默认参数 `scale={DEFAULT_POLICY.scale:g}`、"
             f"`w_high/w_warn={DEFAULT_POLICY.w_high / DEFAULT_POLICY.w_warn:g}`："
             + ("**位于稳健区间内**" if default_in else "**不在稳健区间内**"))
    L.append("")

    L.append("## 六、结论与后续")
    L.append("")
    unstable = [r for r in rows if r["n_flip"] > 0]
    if unstable:
        L.append("以下场景存在参数敏感区，说明其档位结论在当前样本下不够稳健：")
        L.append("")
        for r in unstable:
            L.append(f"- {r['name']}：{r['n_flip']}/{r['n_cells']} 个组合翻档"
                     f"（稳定性 {r['stability']:.0%}）")
        L.append("")
        L.append("这不是缺陷，是**必须承认的局限**：手工权重在没有历史样本前")
        L.append("只能是可辩护的起点。下一步应收集标注样本（专家判过的档位），")
        L.append("用样本校准 scale 与权重，而不是继续手调。")
    else:
        L.append("全部场景在整个参数网格内档位稳定，说明当前档位结论对")
        L.append("scale 与权重比不敏感——但这只覆盖了已构造的场景，")
        L.append("不代表全市场情形。补样本后应重跑本脚本。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("*本报告由 scripts/sensitivity.py 自动生成。分析基于已完成的规则判定结果，"
              "网格仅作用于评分函数，不重新执行规则引擎。*")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(prog="finagent-sensitivity")
    ap.add_argument("--md", default=None, help="输出 Markdown 报告路径")
    ap.add_argument("--json-out", default=None, help="输出 JSON 结果路径")
    args = ap.parse_args()

    scenarios = build_scenarios()
    rows = sweep(scenarios)
    region = stable_region(rows)

    print(render_console(rows, region))

    if args.md:
        out = Path(args.md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(rows, region), encoding="utf-8")
        print(f"\n[ok] Markdown 报告已写入：{out}")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "grid": {
                "scales": list(SCALES),
                "weight_ratios": list(WEIGHT_RATIOS),
                "default_policy": DEFAULT_POLICY.to_dict(),
            },
            "scenarios": [
                {k: v for k, v in r.items() if k != "cells"} | {"cells": r["cells"]}
                for r in rows
            ],
            "stable_region": region,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] JSON 结果已写入：{out}")

    # 退出码：默认参数不在稳健区间内时返回 1，便于 CI 拦截
    return 0 if any(c.get("is_default") for c in region["stable_cells"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
