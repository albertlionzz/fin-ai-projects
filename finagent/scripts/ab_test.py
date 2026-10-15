"""A/B 对比：模板研判 vs LLM 研判，用同一批 findings。

回答的问题
----------
README 的诚实清单里有一条："未接真实 LLM 验证，`OpenAICompatLLM` 已实现但未用
真实模型跑过，生成质量未知。" 这个脚本就是兑现那句话的工具。

对比的四把尺子（全部可机器判定）
------------------------------
    事实保真度   输出里每个数字必须能在 findings 里找到出处
    覆盖率       每条 finding 是否都被提及
    复述率       finding.detail 被原样抄进去的比例（模板刻意做到 ~0）
    可复算性     同一 prompt 跑 k 次输出是否逐字节一致

用法
----
    # 框架自检（无需网络、无需 key）：用确定性 stub 跑通全链路
    python scripts/ab_test.py --stub good

    # 证明幻觉检测真的有效：hostile stub 会故意编一个数字
    python scripts/ab_test.py --stub hostile

    # 真实对比：接任意 OpenAI 兼容端点
    export FINAGENT_LLM_KEY=sk-xxx
    python scripts/ab_test.py --llm api --runs 3 --md artifacts/ab_test.md

    # 本地 llama-server（Bonsai / Qwen 都走这条）
    python scripts/ab_test.py --llm api \
        --llm-api http://127.0.0.1:8080/v1/chat/completions \
        --llm-model qwen3-27b --llm-key dummy --runs 3

关于 stub 的重要说明
------------------
`--stub` 模式**不是真实对比结果**，它只验证框架本身能跑、
指标能算、幻觉检测能抓到问题。报告里会明确标注 STUB。
真实结论必须用 `--llm api` 跑出来——没有 key 时这个脚本
会直接说明缺什么，而不是给一份看起来像结论的数字。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finagent import (  # noqa: E402
    NullLLM,
    OpenAICompatLLM,
    Orchestrator,
    ReportAgent,
    StatementSeries,
    load_benchmarks,
    load_pair,
    load_statement,
)
from finagent.abcompare import ComparisonRow, evaluate, extract_numbers  # noqa: E402
from finagent.models import Level as _L  # noqa: E402


# ---------------------------------------------------------------- Stub LLM
#
# 为什么要有 stub：没有网络、没有 key 时，A/B 框架本身必须可验证。
# 一个跑不起来的评测框架比没有框架更糟——它会让人以为对比已经做过了。
#
# 两种 stub 对应两种我们要测的能力：
#   good    行为像"守规矩的模型"：覆盖全、数字都有出处、有少量复述
#   hostile 行为像"爱发挥的模型"：覆盖全，但编造一个数字
# hostile 的存在是为了证明保真度检测真的能抓到幻觉——
# 一个永远返回 1.0 的指标是没有价值的。


_TITLE_RE = re.compile(r"^\s*\d+\.\s*\[(?P<level>[^\]]+)\]\s*(?P<title>.+?)（规则\s*(?P<rule>[A-Z_]+)）", re.M)
_FACT_RE = re.compile(r"^\s*事实：(?P<fact>.+)$", re.M)


@dataclass
class StubLLM:
    """确定性的离线模型替身。

    它模拟的是"一个读了 findings 然后写研读结论的模型"：
    从 prompt 里抽出标题与事实句，重新组织成一段叙述。
    所有数字都取自 prompt 本身，因此天然 grounded。

    mode:
        good    守规矩：覆盖全部标题，引用事实句，不造数字
        hostile 爱发挥：同样覆盖，但额外编一个 prompt 里不存在的数字
               （用于验证保真度检测真的能抓到幻觉）
    """

    mode: str = "good"
    name: str = "stub-good"
    _cache: dict[str, str] = field(default_factory=dict, repr=False)

    def complete(self, prompt: str, system: str = "") -> str:
        if prompt not in self._cache:
            self._cache[prompt] = self._render(prompt)
        return self._cache[prompt]

    # --- 内部 ---

    @staticmethod
    def _titles(prompt: str) -> list[tuple[str, str]]:
        """从 prompt 抽出 [(规则编号, 标题), ...]。

        ReportAgent._findings_block 的输出格式是固定的，这里按同一格式解析。
        解析是 stub 的职责——它就是在演"读 prompt 的模型"。
        """
        return [(m.group("rule"), m.group("title").strip()) for m in _TITLE_RE.finditer(prompt)]

    @staticmethod
    def _facts(prompt: str) -> list[str]:
        return [m.group("fact").strip() for m in _FACT_RE.finditer(prompt)]

    def _render(self, prompt: str) -> str:
        titles = self._titles(prompt)
        facts = self._facts(prompt)
        if not titles:
            return "本期未触发任何异常规则，各科目同比变动均在观察线内，无需专项核查。"

        n = len(titles)
        parts = [
            f"本期共识别 {n} 项异常，逐项说明如下。",
            "",
        ]
        for k, (rule, title) in enumerate(titles, 1):
            parts.append(f"{k}. {title}")
            fact = facts[k - 1] if k - 1 < len(facts) else ""
            if fact:
                parts.append(f"   {fact}")
            parts.append("   建议核查方向：调取该项对应的原始凭证与业务合同核对。")
            parts.append("")

        if self.mode == "hostile":
            # 故意注入一个 prompt 里不存在的数字。
            # 保真度检测必须把它抓出来——抓不到说明指标是假的。
            parts.append("综合来看，公司整体风险敞口约为 87.4%，需引起高度重视。")
        return "\n".join(parts)


# ---------------------------------------------------------------- 场景


@dataclass
class Scenario:
    name: str
    company: str
    period: str
    findings: list[Finding]
    note: str = ""

    @property
    def n_high(self) -> int:
        return sum(1 for f in self.findings if f.level == _L.HIGH)

    @property
    def n_warn(self) -> int:
        return sum(1 for f in self.findings if f.level == _L.WARN)


def build_scenarios() -> list[Scenario]:
    st = ROOT / "data" / "statements"
    bmp = ROOT / "data" / "benchmarks" / "industry_benchmarks.json"
    out: list[Scenario] = []

    # 场景一：两期，7 类异常（含 4 项重大）
    cur, prev = load_pair(st / "示例制造_2025A.json", st / "示例制造_2024A.json")
    r = Orchestrator().analyze(cur, prev)
    out.append(Scenario(
        name="示例制造 2025A（重大 4 / 关注 3）",
        company=r.company, period=r.period, findings=list(r.findings),
        note="高密度场景：检验模型在多项重大异常下是否仍能逐条覆盖且不串号。",
    ))

    # 场景二：四期趋势
    series = StatementSeries(statements=[
        load_statement(st / f"长恒电子_{p}.json") for p in ("2022A", "2023A", "2024A", "2025A")
    ])
    r2 = Orchestrator().analyze(series.latest, series.previous, series=series)
    out.append(Scenario(
        name="长恒电子 2025A（4 期趋势，重大 2）",
        company=r2.company, period=r2.period, findings=list(r2.findings),
        note="含持续性升级与 QUALITY_DRIFT：detail 里带跨期说明，数字更多。",
    ))

    # 场景三：行业校准
    bm = load_benchmarks(bmp)
    cur3, prev3 = load_pair(st / "恒润重工_2025A.json", st / "恒润重工_2024A.json")
    orch3 = Orchestrator()
    orch3.audit.engine.benchmarks = bm
    r3 = orch3.analyze(cur3, prev3)
    out.append(Scenario(
        name="恒润重工 2025A（行业校准，关注 2）",
        company=r3.company, period=r3.period, findings=list(r3.findings),
        note="观察线来源不同（industry:p75 vs absolute），检验模型是否混淆阈值口径。",
    ))

    # 场景四：零发现（空输入边界）
    out.append(Scenario(
        name="空白场景（0 项发现）",
        company="无异常样本", period="探针", findings=[],
        note="边界：无 findings 时模型会不会凭空发明异常。这是最危险的失效模式。",
    ))
    return out


# ---------------------------------------------------------------- 运行


def run_reasoner(
    agent: ReportAgent,
    sc: Scenario,
    runs: int,
) -> list[str]:
    """同一批 findings 重复跑 runs 次，收集全部输出。"""
    return [agent.reason(sc.company, sc.period, sc.findings) for _ in range(runs)]


def sweep(scenarios: Sequence[Scenario], runs: int, llm_mode: str, stub_mode: str) -> tuple[list[ComparisonRow], dict]:
    rows: list[ComparisonRow] = []
    provenance: dict = {}

    # A 侧：模板（NullLLM → 确定性聚合模板）
    for sc in scenarios:
        agent = ReportAgent(llm=NullLLM())
        outs = run_reasoner(agent, sc, runs)
        rows.append(evaluate(sc.name, "模板（NullLLM）", outs, sc.findings))

    # B 侧：LLM 或 stub
    if llm_mode == "api":
        llm = OpenAICompatLLM()
        label = f"LLM（{llm.model}）"
        # 出处必须随报告落盘：同一个指标换个模型可能是另一个结论，
        # 没有模型/端点/温度的 A/B 报告不可复算，等于没有报告。
        provenance = {
            "model": llm.model,
            "base_url": llm.base_url,
            "temperature": llm.temperature,
            "runs": runs,
        }
    else:
        llm = StubLLM(mode=stub_mode, name=f"stub-{stub_mode}")
        label = f"STUB-{stub_mode}（框架自检）"
    for sc in scenarios:
        agent = ReportAgent(llm=llm)
        outs = run_reasoner(agent, sc, runs)
        rows.append(evaluate(sc.name, label, outs, sc.findings))

    return rows, provenance


# ---------------------------------------------------------------- 输出


def _dw(s: str) -> int:
    """字符串的显示宽度（CJK 记 2，ASCII 记 1）。

    中文场景必须自己算宽度：Python 的 `:<34` 按字符数补空格，
    而终端按显示宽度排版，纯用格式说明符会让含中文的列全部错位。
    """
    import unicodedata

    w = 0
    for ch in s:
        w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return w


def _pad(s: str, width: int) -> str:
    """按显示宽度左对齐补空格。超宽则截断并加省略号。"""
    if _dw(s) <= width:
        return s + " " * (width - _dw(s))
    # 超宽：截到 width-1 个显示宽度，补一个 …
    out: list[str] = []
    used = 0
    for ch in s:
        cw = 2 if unicodedata_east_asian(ch) else 1
        if used + cw > width - 1:
            break
        out.append(ch)
        used += cw
    return "".join(out) + " " * max(0, width - 1 - used) + "…"


def unicodedata_east_asian(ch: str) -> bool:
    import unicodedata

    return unicodedata.east_asian_width(ch) in ("W", "F")


def render_console(rows: Sequence[ComparisonRow], is_stub: bool) -> str:
    L: list[str] = []
    L.append("=" * 96)
    L.append(" 研判质量 A/B 对比")
    if is_stub:
        L.append(" ⚠ STUB 模式：仅验证框架可跑，非真实模型对比结论")
    L.append("=" * 96)
    L.append("")
    L.append(
        f"{_pad('场景', 36)}{_pad('研判器', 30)}"
        f"{'保真':>7}{'覆盖':>7}{'复述':>7}{'可复算':>8}{'字数':>7}"
    )
    L.append("-" * 96)
    for r in rows:
        L.append(
            f"{_pad(r.scenario, 36)}{_pad(r.reasoner, 30)}"
            f"{r.fidelity.score:>6.0%}{r.coverage.score:>7.0%}"
            f"{r.restatement:>7.0%}{r.determinism:>8.0%}{r.length:>7}"
        )
    L.append("")

    bad = [r for r in rows if r.fidelity.score < 1.0]
    if bad:
        L.append("事实保真度未满分的输出（需人工核对的数字）：")
        for r in bad:
            L.append(f"  - {r.scenario} / {r.reasoner}")
            L.append(f"    未标注出处：{r.fidelity.ungrounded}")
        L.append("")
    else:
        L.append("全部输出的事实保真度均为 100%——输出中的每个数字都能在 findings 里找到出处。")
        L.append("")

    tpl = [r for r in rows if r.reasoner.startswith("模板")]
    llm_rows = [r for r in rows if not r.reasoner.startswith("模板")]
    if tpl and llm_rows:
        L.append("模板 vs 对手（均值）：")
        L.append("-" * 92)
        for label, group in (("模板", tpl), ("对手", llm_rows)):
            if not group:
                continue
            n = len(group)
            L.append(
                f"  {label:<8}保真 {sum(r.fidelity.score for r in group)/n:>6.1%}"
                f"　覆盖 {sum(r.coverage.score for r in group)/n:>6.1%}"
                f"　复述 {sum(r.restatement for r in group)/n:>6.1%}"
                f"　可复算 {sum(r.determinism for r in group)/n:>6.1%}"
                f"　字数 {sum(r.length for r in group)/n:>7.0f}"
            )
    return "\n".join(L)


def render_markdown(rows: Sequence[ComparisonRow], is_stub: bool, runs: int, provenance: dict | None = None) -> str:
    L: list[str] = []
    L.append("# 研判质量 A/B 对比")
    L.append("")
    if is_stub:
        L.append("> ⚠ **STUB 模式**：本报告仅验证评测框架可运行、指标可计算、")
        L.append("> 幻觉检测可生效。**不是真实模型对比结论。**")
        L.append("> 真实结论需执行 `python scripts/ab_test.py --llm api`。")
    else:
        L.append("> 真实模型对比。同一批 findings，两个研判器，各跑 "
                 f"{runs} 次取首次输出作为评分对象。")
    if provenance:
        L.append(">")
        L.append(f"> **模型出处**：`{provenance['model']}`　端点 `{provenance['base_url']}`　"
                 f"temperature {provenance['temperature']}　每场景 {provenance['runs']} 次")
    L.append("")
    L.append("## 一、对比维度")
    L.append("")
    L.append("| 维度 | 衡量什么 | 为什么重要 |")
    L.append("|---|---|---|")
    L.append("| 事实保真度 | 输出中每个数字能否在 findings 里找到出处 | 财务场景里一个编造的数字就是事故 |")
    L.append("| 覆盖率 | 每条 finding 是否都被提及 | 漏报比啰嗦危险 |")
    L.append("| 复述率 | finding.detail 被原样抄入的比例 | 模板刻意做到 ~0；模型只换措辞重抄则增量价值为零 |")
    L.append("| 可复算性 | 同 prompt 跑 k 次输出是否逐字节一致 | 项目核心主张：结论要可复算 |")
    L.append("")

    L.append("## 二、结果")
    L.append("")
    L.append("| 场景 | 研判器 | 保真度 | 覆盖率 | 复述率 | 可复算性 | 字数 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(
            f"| {r.scenario} | {r.reasoner} | {r.fidelity.score:.0%} "
            f"| {r.coverage.score:.0%} | {r.restatement:.0%} "
            f"| {r.determinism:.0%} | {r.length} |"
        )
    L.append("")

    bad = [r for r in rows if r.fidelity.score < 1.0]
    L.append("## 三、未标注出处的数字")
    L.append("")
    if not bad:
        L.append("全部输出的事实保真度均为 100%。")
    else:
        L.append("以下输出包含无法在 findings 中找到出处的数字，需人工核对：")
        L.append("")
        for r in bad:
            L.append(f"- **{r.scenario} / {r.reasoner}**：{r.fidelity.ungrounded}")
    L.append("")

    miss = [r for r in rows if r.coverage.missing]
    L.append("## 四、未覆盖的发现")
    L.append("")
    if not miss:
        L.append("全部输出均覆盖了所有 finding。")
    else:
        for r in miss:
            L.append(f"- **{r.scenario} / {r.reasoner}**：漏掉 {r.coverage.missing}")
    L.append("")

    L.append("## 五、读数说明")
    L.append("")
    L.append("- **保真度 < 100% 是硬伤**，不是扣分项。财务结论里出现无出处的数字，")
    L.append("  整份报告的可信度都要打问号。")
    L.append("- **可复算性 100% 只属于模板**。这是设计使然，不是模板更聪明——")
    L.append("  它说明了为什么判定环节不能用 LLM：措辞环节能容忍波动，判定环节不能。")
    L.append("- **复述率高低要结合场景看**。模板接近 0 是因为它做跨发现聚合；")
    L.append("  模型如果复述率高，说明它只是在「换措辞重抄」，没有提供聚合价值。")
    L.append("")
    L.append("---")
    L.append("")
    L.append(f"*本报告由 scripts/ab_test.py 自动生成，每场景重复 {runs} 次。*")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(prog="finagent-ab-test")
    ap.add_argument("--llm", default="stub", help="stub(默认，框架自检) / api(真实模型)")
    ap.add_argument("--stub", default="good", choices=["good", "hostile"],
                    help="stub 模式下的行为：good=守规矩 / hostile=故意编数字")
    ap.add_argument("--runs", type=int, default=1, help="每个场景重复次数，用于测可复算性")
    ap.add_argument("--md", default=None, help="输出 Markdown 报告路径")
    ap.add_argument("--json-out", default=None, help="输出 JSON 结果路径")
    args = ap.parse_args()

    if args.llm == "api" and args.runs > 1:
        print(f"[提示] --runs {args.runs} 会对每个场景发起 {args.runs} 次真实 API 调用，"
              f"共 {args.runs * 4} 次。", file=sys.stderr)

    scenarios = build_scenarios()
    rows, provenance = sweep(scenarios, args.runs, args.llm, args.stub)
    is_stub = args.llm != "api"

    print(render_console(rows, is_stub))

    if args.md:
        out = Path(args.md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(rows, is_stub, args.runs, provenance), encoding="utf-8")
        print(f"\n[ok] Markdown 报告已写入：{out}")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "mode": "stub" if is_stub else "api",
                    "stub_mode": args.stub if is_stub else None,
                    "runs": args.runs,
                    "provenance": provenance or None,
                    "rows": [r.to_dict() for r in rows],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[ok] JSON 结果已写入：{out}")

    # 退出码：出现保真度不足即返回 1，便于 CI 拦截
    return 0 if all(r.fidelity.score >= 1.0 for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
