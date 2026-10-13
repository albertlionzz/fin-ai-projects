"""输出渲染：把 AnalysisResult 变成 Markdown 报告或控制台摘要。

渲染层与逻辑层严格分离——这样同一份分析结果可以出网页、出 Word、
出飞书卡片，而不必改任何判定逻辑。
"""

from __future__ import annotations

from typing import Sequence

from .models import AnalysisResult, Finding, FinancialStatement, Level


_LEVEL_MARK = {Level.HIGH: "🔴", Level.WARN: "🟡", Level.INFO: "⚪"}


def _fmt_ratio(v: float | None, suffix: str = "%") -> str:
    return "—" if v is None else f"{v:.2f}{suffix}"


# ---------------------------------------------------------------- 控制台


def render_console(result: AnalysisResult) -> str:
    """终端摘要。面向快速查看，不追求完整。"""
    lines: list[str] = []
    bar = "=" * 62
    lines.append(bar)
    lines.append(f" {result.company} · {result.period}")
    lines.append(bar)
    lines.append(f" 风险分 {result.risk_score():.1f}/100 [{result.risk_grade()}]    "
                 f"重大 {result.high_count}  关注 {result.warn_count}   合计 {len(result.findings)}")
    lines.append("")
    lines.append(result.summary)
    if result.findings:
        lines.append("")
        lines.append(f" 建议：{result.risk_action()}")
    lines.append("")

    if not result.findings:
        lines.append(" 未触发任何异常规则。")
    else:
        for i, f in enumerate(result.findings, 1):
            mark = _LEVEL_MARK.get(f.level, "·")
            lines.append(f"{mark} [{f.level.label}] {f.title}   ({f.rule_id})")
            lines.append(f"    {f.detail}")
            if f.basis:
                lines.append(f"    依据：{f.basis}")
            lines.append("")

    if result.trace:
        lines.append("-" * 62)
        lines.append(" 执行轨迹")
        for t in result.trace:
            flag = "OK " if t.ok else "ERR"
            ms = f"{t.duration_ms:.1f}ms" if t.duration_ms else "—"
            lines.append(f"  [{flag}] {t.agent:<10} {t.action:<20} {ms:>10}  {t.detail}")

    return "\n".join(lines)


# ---------------------------------------------------------------- Markdown


def render_markdown(
    result: AnalysisResult,
    cur: FinancialStatement | None = None,
    prev: FinancialStatement | None = None,
) -> str:
    """完整 Markdown 报告。这是交付给用户的主产物。"""
    md: list[str] = []
    md.append(f"# {result.company} · 财务报表异常分析")
    md.append("")
    md.append(f"> 报告期：**{result.period}**　｜　风险分：**{result.risk_score():.1f} / 100**"
              f"（等级：**{result.risk_grade()}**）")
    md.append("")

    # --- 概览 ---
    md.append("## 一、分析概览")
    md.append("")
    md.append(result.summary)
    md.append("")
    md.append(f"- 重大异常（需专项核查）：**{result.high_count}** 项")
    md.append(f"- 关注事项（需解释）：**{result.warn_count}** 项")
    md.append(f"- 发现合计：**{len(result.findings)}** 项")
    md.append(f"- 风险等级：**{result.risk_grade()}**（{result.risk_score():.1f}/100）")
    md.append("")
    if result.findings:
        md.append(f"> **处置建议**：{result.risk_action()}。")
        md.append("")

    # --- 关键指标 ---
    if cur is not None and prev is not None:
        md.append("## 二、关键财务指标")
        md.append("")
        md.append("| 指标 | " + prev.period + " | " + cur.period + " | 变动 |")
        md.append("|---|---|---|---|")
        pairs = [
            ("毛利率", prev.gross_margin(), cur.gross_margin(), "%"),
            ("净利率", prev.net_margin(), cur.net_margin(), "%"),
            ("资产负债率", prev.debt_ratio(), cur.debt_ratio(), "%"),
            ("流动比率", prev.current_ratio(), cur.current_ratio(), ""),
            ("经营现金流/净利润", prev.ocf_to_profit(), cur.ocf_to_profit(), ""),
            ("ROE", prev.roe(), cur.roe(), "%"),
        ]
        for name, pv, cv, suf in pairs:
            if pv is None and cv is None:
                continue
            delta = "—" if (pv is None or cv is None) else f"{cv - pv:+.2f}{suf}"
            md.append(f"| {name} | {_fmt_ratio(pv, suf)} | {_fmt_ratio(cv, suf)} | {delta} |")
        md.append("")

    # --- 异常明细 ---
    md.append("## 三、异常项明细")
    md.append("")
    if not result.findings:
        md.append("本期未触发任何异常规则。各科目同比变动均在观察线内，勾稽关系平衡。")
        md.append("")
    else:
        for i, f in enumerate(result.findings, 1):
            md.append(f"### {i}. {f.title}")
            md.append("")
            md.append(f"**严重度**：{f.level.label}　**规则编号**：`{f.rule_id}`"
                      f"　**涉及科目**：{', '.join(f'`{k}`' for k in f.items) if f.items else '—'}")
            md.append("")
            md.append(f.detail)
            md.append("")
            if f.basis:
                md.append(f"*判定依据*：{f.basis}")
                md.append("")

    # --- 研判 ---
    md.append("## 四、成因研判与核查方向")
    md.append("")
    md.append(result.reasoning or "（未生成研判内容）")
    md.append("")

    # --- 轨迹 ---
    if result.trace:
        md.append("## 五、执行轨迹")
        md.append("")
        md.append("| Agent | 动作 | 耗时 | 状态 | 说明 |")
        md.append("|---|---|---|---|---|")
        for t in result.trace:
            ms = f"{t.duration_ms:.1f}ms" if t.duration_ms else "—"
            md.append(f"| `{t.agent}` | `{t.action}` | {ms} | {'成功' if t.ok else '失败'} | {t.detail} |")
        md.append("")

    md.append("---")
    md.append("")
    md.append("*本报告由 finagent 自动生成。异常判定来自确定性规则引擎，可复算、可追溯；"
              "结论仅基于所提供的报表数据，不构成投资建议。*")
    md.append("")
    md.append("*风险分口径：各发现按严重度加权（重大 10 / 关注 4 / 提示 1）后经饱和函数 "
              "`(1-e^(-x/20))×100` 归一，单调递增不触顶。等级映射："
              "≥85 极高、≥65 高、≥35 中、其余低。权重与阈值应随历史样本回测校准。*")
    return "\n".join(md)
