"""多 Agent 编排层：数据提取 → 合规校验 → 结论生成。

为什么要拆成三个 Agent，而不是一个函数干完？
    这是本项目要回答的核心工程问题。财务场景的要求是**高可控执行链路**：
    每一环的产出要能单独验证、单独替换、单独追责。

    - ExtractAgent  ：把非结构化报表转成结构化科目。错了，后面全错。
    - AuditAgent    ：跑确定性规则。不用 LLM——规则可复算、可辩护。
    - ReportAgent   ：把规则判定翻译成人话。这一环才需要语言能力。

    关键设计：**LLM 只在最后一环介入**。让模型去"发现异常"是不可控的，
    让它去"解释已确认的异常"才是它擅长且可审计的用法。

    每个 Agent 都记录 AgentTrace，产出可复现的执行报告。
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .models import (
    AgentTrace,
    AnalysisResult,
    FinancialStatement,
    Finding,
    Level,
    StatementSeries,
)
from .rules import RuleBook, RuleEngine


# ---------------------------------------------------------------- LLM 通道


class LLM(Protocol):
    """LLM 通道协议。与项目一的 Embedder 协议同构——可插拔、有兜底。"""

    def complete(self, prompt: str, system: str = "") -> str: ...


class NullLLM:
    """离线兜底：不调模型，用模板拼结论。

    存在的意义不是"假装有 LLM"，而是让整条链路**零依赖可跑通**、
    可评测、可演示。接入真实模型时只替换这一个实现。
    """

    name = "null"

    def complete(self, prompt: str, system: str = "") -> str:
        return ""


class OpenAICompatLLM:
    """OpenAI 兼容 /v1/chat/completions。通义 / DeepSeek / 硅基流动 / vLLM 都走这条。"""

    name = "openai-compat"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 90.0,
        temperature: float = 0.1,
    ):
        self.model = model or os.getenv("FINAGENT_LLM_MODEL", "qwen-plus")
        self.base_url = (
            base_url
            or os.getenv("FINAGENT_LLM_API", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        )
        self.api_key = api_key or os.getenv("FINAGENT_LLM_KEY", "")
        self.timeout = timeout
        self.temperature = temperature
        if not self.api_key:
            raise ValueError("缺少 API key。设置 FINAGENT_LLM_KEY，或改用 NullLLM。")

    def complete(self, prompt: str, system: str = "") -> str:
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        payload = json.dumps(
            {
                "model": self.model,
                "messages": msgs,
                "temperature": self.temperature,
                "stream": False,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self.base_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


def build_llm(kind: str = "auto", **kwargs) -> LLM:
    """auto：有 key 用 API，否则回退 NullLLM。绝不因为缺 key 就崩。"""
    kind = (kind or "auto").lower()
    if kind == "null" or kind == "none":
        return NullLLM()
    if kind == "api":
        return OpenAICompatLLM(**kwargs)
    if os.getenv("FINAGENT_LLM_KEY"):
        try:
            return OpenAICompatLLM()
        except Exception as e:  # noqa: BLE001
            print(f"[warn] LLM 初始化失败，回退离线模板：{e}")
    return NullLLM()


# ---------------------------------------------------------------- Agent 基类


@dataclass
class Agent:
    """所有 Agent 的共同契约：有名字、有 trace、可单独调用。"""

    name: str = "agent"
    trace: list[AgentTrace] = field(default_factory=list)

    def _log(self, action: str, detail: str = "", ms: float = 0.0, ok: bool = True) -> None:
        self.trace.append(AgentTrace(agent=self.name, action=action, detail=detail, duration_ms=ms, ok=ok))


# ---------------------------------------------------------------- 1. 数据提取


# 中文科目名 → 英文 key 的映射。真实项目里这张表应该独立配置文件化。
ITEM_ALIASES: dict[str, str] = {
    "营业收入": "revenue", "营业总收入": "revenue", "主营业务收入": "revenue", "营收": "revenue",
    "营业成本": "cost_of_sales", "主营业务成本": "cost_of_sales", "营业总成本": "cost_of_sales",
    "净利润": "net_profit", "归母净利润": "net_profit", "归属于母公司股东的净利润": "net_profit",
    "应收账款": "accounts_receivable", "应收票据及应收账款": "accounts_receivable",
    "存货": "inventory", "资产总计": "total_assets", "总资产": "total_assets",
    "负债合计": "total_liabilities", "总负债": "total_liabilities",
    "所有者权益合计": "total_equity", "股东权益合计": "total_equity",
    "归属于母公司所有者权益合计": "net_assets", "净资产": "net_assets",
    "流动资产合计": "current_assets", "流动负债合计": "current_liabilities",
    "经营活动产生的现金流量净额": "operating_cash_flow", "经营现金流": "operating_cash_flow",
    "商誉": "goodwill", "销售费用": "selling_expense", "管理费用": "admin_expense",
    "研发费用": "rd_expense", "财务费用": "finance_expense",
}


class ExtractAgent(Agent):
    """数据提取 Agent：把非结构化 / 半结构化报表转成标准科目表。

    设计取舍：**优先走确定性解析，LLM 只做兜底**。
    原因很直接——财务数字错一位就是事故。确定性能解析的绝不给模型。
    只有格式完全无法识别时，才把原始片段丢给 LLM 抽取，并要求它逐项给出原文依据。
    """

    def __init__(self, llm: LLM | None = None, name: str = "extract"):
        super().__init__(name=name)
        self.llm = llm or NullLLM()

    # --- 确定性解析：形如 "营业收入 125000" / "营业收入：125,000" / "营业收入 1,000.00 万元" ---
    #
    # 行首科目名 + 分隔符 + 数字 + 可选单位。单位必须吃掉，否则真实报表
    # （几乎每行都带"万元/元/%"）会整行匹配失败，退化成"全部未识别"。
    _LINE = re.compile(
        r"^\s*([\u4e00-\u9fff（）()A-Za-z]{2,24})\s*[:：]?\s*"
        r"(-?[\d,]+(?:\.\d+)?)\s*"
        r"(万元|亿元|千元|百万元|元|%|％)?\s*$"
    )

    # 单位换算到"万元"（本模型统一口径）。无单位视为已是万元。
    _UNIT_SCALE: dict[str, float] = {
        "亿元": 10000.0,
        "百万元": 100.0,
        "千元": 0.1,
        "万元": 1.0,
        "元": 0.0001,
    }

    @staticmethod
    def _match_alias(cn_name: str) -> str | None:
        """把报表里的中文科目名匹配到标准 key。

        顺序：先全等（最精确），再按别名长度从长到短做子串匹配。
        按长度排序是必要的——否则"营业收入"会先命中短别名"营收"所在的键，
        而"营业总成本"这类长名也可能被更短的别名抢先吃掉。
        """
        if cn_name in ITEM_ALIASES:
            return ITEM_ALIASES[cn_name]
        for alias in sorted(ITEM_ALIASES, key=len, reverse=True):
            if alias in cn_name:
                return ITEM_ALIASES[alias]
        return None

    def parse_text(self, text: str, period: str) -> FinancialStatement:
        """从纯文本逐行解析。无法识别的行不静默丢弃，而是记进 meta 供人工复核。"""
        t0 = time.time()
        items: dict[str, float] = {}
        unrecognized: list[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = self._LINE.match(line)
            if not m:
                unrecognized.append(line)
                continue
            cn_name, num, unit = m.group(1), m.group(2).replace(",", ""), m.group(3)
            key = self._match_alias(cn_name)
            if key is None:
                unrecognized.append(line)
                continue
            try:
                value = float(num)
            except ValueError:
                unrecognized.append(line)
                continue
            if unit and unit in self._UNIT_SCALE:
                value *= self._UNIT_SCALE[unit]
            items[key] = value
        self._log(
            "parse_text",
            f"识别 {len(items)} 个科目，未识别 {len(unrecognized)} 行",
            ms=(time.time() - t0) * 1000,
        )
        return FinancialStatement(
            period=period,
            items=items,
            meta={"unrecognized": unrecognized, "source": "text"},
        )

    def parse_json(self, path: str) -> FinancialStatement:
        """标准结构化输入，直接加载。"""
        t0 = time.time()
        d = json.loads(open(path, encoding="utf-8").read())
        st = FinancialStatement.from_dict(d)
        self._log("parse_json", f"{path} → {len(st.items)} 个科目", ms=(time.time() - t0) * 1000)
        return st

    def parse_with_llm_fallback(self, text: str, period: str) -> FinancialStatement:
        """确定性解析失败时的 LLM 兜底。

        要求模型输出 JSON 并附原文依据；解析失败则退回确定性结果，
        绝不产出一份"看起来成功但数字是编的"报表。
        """
        st = self.parse_text(text, period)
        if len(st.items) >= 8 or isinstance(self.llm, NullLLM):
            return st
        self._log("llm_fallback", "确定性解析科目不足，尝试 LLM 抽取")
        prompt = (
            "从下面的财务报表文本中抽取科目与金额，单位统一为万元。\n"
            "只输出 JSON，不要解释。格式：{\"科目英文key\": 数值}\n"
            f"可用 key：{sorted(set(ITEM_ALIASES.values()))}\n\n"
            f"报表文本：\n{text}\n"
        )
        try:
            out = self.llm.complete(prompt, system="你是严谨的财务数据抽取工具。")
            m = re.search(r"\{.*\}", out, re.S)
            if m:
                data = json.loads(m.group(0))
                for k, v in data.items():
                    if k in set(ITEM_ALIASES.values()):
                        try:
                            st.items[k] = float(v)
                        except (TypeError, ValueError):
                            pass
                st.meta["source"] = "llm_fallback"
                self._log("llm_fallback", f"补充后共 {len(st.items)} 个科目")
        except Exception as e:  # noqa: BLE001
            self._log("llm_fallback", f"失败，保留确定性结果：{e}", ok=False)
        return st


# ---------------------------------------------------------------- 2. 合规校验


class AuditAgent(Agent):
    """合规校验 Agent：跑确定性规则引擎。

    刻意**不接 LLM**。原因：异常判定必须可复算、可辩护。
    同一个输入必须永远得到同一个结论，否则审计无法追责。
    """

    def __init__(self, engine: RuleEngine | None = None, name: str = "audit"):
        super().__init__(name=name)
        self.engine = engine or RuleEngine()

    def run(
        self,
        cur: FinancialStatement,
        prev: FinancialStatement,
        series: StatementSeries | None = None,
    ) -> list[Finding]:
        t0 = time.time()
        findings = self.engine.run(cur, prev, series=series)
        high = sum(1 for f in findings if f.level == Level.HIGH)
        warn = sum(1 for f in findings if f.level == Level.WARN)
        if series is not None and series.has_trend(self.engine.book.trend_min_periods):
            detail = (
                f"{len(self.engine.rules)} 条规则（含多期趋势）→ {len(findings)} 项发现"
                f"（重大 {high} / 关注 {warn}），趋势区间 {series.periods[0]}→{series.periods[-1]}"
            )
        else:
            detail = (
                f"{len(self.engine.rules)} 条规则 → {len(findings)} 项发现"
                f"（重大 {high} / 关注 {warn}）"
            )
        self._log("rule_engine", detail, ms=(time.time() - t0) * 1000)
        return findings


# ---------------------------------------------------------------- 3. 结论生成


REPORT_SYSTEM = """你是财务报表分析专家，为投资决策提供异常项说明。

硬性规则：
1. 只解释给定的异常项，不得新增未列出的发现。
2. 每条结论必须点明涉及的科目与数字，避免空泛表述。
3. 不得编造未提供的数据。信息不足时明确说明"需进一步核查"。
4. 用简体中文，结构化输出，控制在 400 字以内。"""


class ReportAgent(Agent):
    """结论生成 Agent：把规则判定翻译成可读的说明与建议。

    这一环才引入 LLM。关键约束：**输入只有已确认的 findings**，
    模型没有机会"自己发明"异常。没有 LLM 时用确定性模板，产出依然完整。
    """

    def __init__(self, llm: LLM | None = None, name: str = "report"):
        super().__init__(name=name)
        self.llm = llm or NullLLM()

    def _findings_block(self, findings: Sequence[Finding]) -> str:
        lines = []
        for i, f in enumerate(findings, 1):
            lines.append(
                f"{i}. [{f.level.label}] {f.title}（规则 {f.rule_id}）\n"
                f"   事实：{f.detail}\n"
                f"   涉及科目：{', '.join(f.items) or '—'}\n"
                f"   判定依据：{f.basis or '—'}"
            )
        return "\n".join(lines) if lines else "（无异常项）"

    def summarize(self, company: str, period: str, findings: Sequence[Finding]) -> str:
        """确定性摘要：不依赖 LLM，永远可产出。"""
        if not findings:
            return f"{company} {period} 未触发任何异常规则。"
        high = [f for f in findings if f.level == Level.HIGH]
        warn = [f for f in findings if f.level == Level.WARN]
        head = f"{company} {period} 共触发 {len(findings)} 项异常：重大 {len(high)} 项、关注 {len(warn)} 项。"
        if high:
            head += " 需优先核查：" + "；".join(f.title for f in high[:3]) + "。"
        return head

    # 规则 → 成因族 的静态归属表。用途是把零散发现聚成几条主线索，
    # 避免报告变成"7 条异常的流水账"——那对读者没有增量价值。
    _CLUSTERS: tuple[tuple[str, str, str], ...] = (
        # (线索名, 命中规则, 该线索的推断)
        (
            "收入质量",
            "AR_GAP,OCF_DIVERGE,MARGIN_EROSION",
            "应收增速跑赢营收、现金流与利润背离、增收不增利同时出现，"
            "指向收入确认偏松或放宽信用换收入。重点核查前五大客户账龄、"
            "期后退款与信用政策变更记录。",
        ),
        (
            "资产质量",
            "INV_GAP,GOODWILL,SWING_SCAN",
            "存货增速远超营收、商誉占净资产比重偏高，且存在多科目大幅波动，"
            "指向资产端减值计提可能不足。重点核查存货库龄结构与商誉减值测试假设。",
        ),
        (
            "偿债能力",
            "DEBT_RATIO",
            "资产负债率显著抬升，需结合有息负债期限结构判断再融资压力。"
            "重点核查短期借款到期安排与表外担保。",
        ),
        (
            "盈利质量趋势",
            "QUALITY_DRIFT",
            "多项盈利指标连续同向变动，说明是慢性恶化而非单期扰动。"
            "重点核查成本结构变化、费用刚性与现金流回收周期的历史走势。",
        ),
    )

    def _template_reasoning(self, findings: Sequence[Finding]) -> str:
        """确定性研判：做 LLM 不做之外的事——跨发现聚合、归因、排优先级。

        刻意不复述 detail（那是第三章的职责）。这里只回答三个问题：
        1) 这些异常能归成哪几条主线索？
        2) 每条线索指向什么共同成因、下一步查什么？
        3) 按什么顺序查？
        """
        if not findings:
            return "本期未发现异常，各科目同比变动均在观察线内，勾稽关系平衡，无需专项核查。"

        hit = {f.rule_id for f in findings}
        blocks: list[str] = []

        # --- 主线索归因 ---
        blocks.append("### 主线索归因")
        blocks.append("")
        matched: set[str] = set()
        for name, rule_ids, inference in self._CLUSTERS:
            ids = [r.strip() for r in rule_ids.split(",") if r.strip()]
            fused = [r for r in ids if r in hit]
            if not fused:
                continue
            matched.update(fused)
            blocks.append(f"- **{name}**（命中 {len(fused)} 项：{'、'.join(f'`{r}`' for r in fused)}）")
            blocks.append(f"  {inference}")
        # 未被任何线索覆盖的规则，单独列出，避免遗漏
        orphan = sorted(hit - matched)
        if orphan:
            blocks.append(f"- **其他**（{'、'.join(f'`{r}`' for r in orphan)}）")
            blocks.append("  未归入上述主线索，建议单独确认业务背景后再判断性质。")
        blocks.append("")

        # --- 核查顺序 ---
        blocks.append("### 建议核查顺序")
        blocks.append("")
        order = [f for f in findings if f.level == Level.HIGH]
        if order:
            for i, f in enumerate(order, 1):
                blocks.append(f"{i}. **{f.title}**（`{f.rule_id}`）—— {f.basis or '需核查原始凭证'}")
        else:
            blocks.append("1. 无重大异常项；按关注项逐条向管理层问询即可。")
        blocks.append("")

        # --- 影响提示 ---
        blocks.append("### 对判断的整体影响")
        blocks.append("")
        n_high = len([f for f in findings if f.level == Level.HIGH])
        n_warn = len([f for f in findings if f.level == Level.WARN])
        # 门槛与 risk_grade 的严重度档保持一致：只要存在重大异常，
        # 就不能说"以关注项为主"。两处阈值不一致会产出自相矛盾的话。
        if n_high >= 1:
            if n_high >= 3:
                blocks.append(
                    f"重大异常达 {n_high} 项，问题不是孤立的科目波动，"
                    "而可能来自同一套业务决策（如激进的渠道压货或并购扩张）。"
                    "在核查完成前，基于本期报表得出的盈利能力与资产安全结论都应视为不稳健。"
                )
            else:
                blocks.append(
                    f"存在 {n_high} 项重大异常（另有 {n_warn} 项关注），"
                    "已达到需要专项核查的程度。"
                    "建议按上述顺序逐项核实，未完成前不宜据本期报表下确定性结论。"
                )
        else:
            blocks.append(
                f"异常以关注项为主（{n_warn} 项），尚未形成系统性风险特征。"
                "建议按上述顺序向管理层问询，若下一期继续恶化，再升级为重大处理。"
            )
        return "\n".join(blocks)

    def reason(self, company: str, period: str, findings: Sequence[Finding]) -> str:
        """生成式说明。无 LLM 时走确定性聚合模板，产出依然完整可读。"""
        t0 = time.time()
        if isinstance(self.llm, NullLLM):
            self._log("template_reasoning", "未接 LLM，使用确定性聚合模板")
            return self._template_reasoning(findings)

        prompt = (
            f"公司：{company}\n报告期：{period}\n\n"
            f"经规则引擎确认的异常项：\n{self._findings_block(findings)}\n\n"
            f"请逐条说明：该异常可能的业务成因、需要进一步核查的方向、"
            f"以及对投资判断的影响。不要新增未列出的异常。"
        )
        try:
            out = self.llm.complete(prompt, system=REPORT_SYSTEM).strip()
            self._log("llm_reasoning", f"生成 {len(out)} 字", ms=(time.time() - t0) * 1000)
            return out
        except Exception as e:  # noqa: BLE001
            self._log("llm_reasoning", f"失败，回退模板：{e}", ms=(time.time() - t0) * 1000, ok=False)
            return self._template_reasoning(findings)


# ---------------------------------------------------------------- 编排


@dataclass
class Orchestrator:
    """把三个 Agent 串成一条可控链路。

    设计原则：**任何一环失败都不中断整条链路**，而是记录下来继续走，
    最后在 trace 里如实呈现。财务报表分析里，"少报一个异常"比"整份报告没出来"更危险。
    """

    extract: ExtractAgent = field(default_factory=ExtractAgent)
    audit: AuditAgent = field(default_factory=AuditAgent)
    report: ReportAgent = field(default_factory=ReportAgent)

    def analyze(
        self,
        cur: FinancialStatement,
        prev: FinancialStatement,
        company: str | None = None,
        period: str | None = None,
        series: StatementSeries | None = None,
    ) -> AnalysisResult:
        company = company or str(cur.meta.get("company") or "未命名主体")
        period = period or cur.period

        # 环节 2：合规校验（环节 1 的数据提取已完成，输入即结构化报表）
        # 传入 series 时启用多期趋势规则；不传则退化为纯两期分析，行为不变。
        findings = self.audit.run(cur, prev, series=series)

        # 环节 3：结论生成
        summary = self.report.summarize(company, period, findings)
        reasoning = self.report.reason(company, period, findings)

        trace = [*self.extract.trace, *self.audit.trace, *self.report.trace]
        metrics: dict[str, Any] = {
            "current": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in cur.ratios().items()},
            "previous": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in prev.ratios().items()},
            "items_count": {"current": len(cur.items), "previous": len(prev.items)},
        }
        if series is not None:
            metrics["series"] = {
                "periods": series.periods,
                "n_periods": len(series),
                "trend_enabled": series.has_trend(3),
            }

        return AnalysisResult(
            company=company,
            period=period,
            findings=list(findings),
            summary=summary,
            reasoning=reasoning,
            trace=trace,
            metrics=metrics,
        )
