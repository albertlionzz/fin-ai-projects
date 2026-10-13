"""财务异常检测的规则引擎：把"会计常识"写成可执行、可测试的规则。

为什么要有这一层？
    LLM 能看出"这个数字不对劲"，但说不清"依据哪条准则、阈值多少、
    为什么是重大异常"。财务场景里，结论必须可追溯、可复算、可辩护。
    规则引擎负责给出**可解释的判定**，LLM 只负责把判定翻译成人话。

设计要点：
    1. 规则与阈值分离：阈值集中在 RuleBook，方便按行业/规模调整。
    2. 每条规则返回 Finding，带 level / 依据 / 涉及科目，不给"是/否"黑盒。
    3. 纯标准库，零依赖；全部规则可单测。

会计依据说明（阈值来源）：
    - 应收账款增速远超营收增速：经典的"收入质量恶化"信号，
      常见经验阈值是应收增速 - 营收增速 > 20pct（参考《财务报表分析》通行口径）。
    - 存货增速远超营收增速：存货积压或跌价准备计提不足的信号。
    - 毛利率异常波动：超过 5pct 的同比变动需要解释（多数行业的正常经营波动区间）。
    - 经营现金流 / 净利润：长期低于 0.8 说明利润没有现金支撑（"纸面利润"）。
    - 资产负债率：超过 70% 进入高杠杆区间（制造业通行观察线）。
    - 商誉占净资产比：超过 30% 意味减值风险敞口大。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

from .models import FinancialStatement, Finding, Level


# ---------------------------------------------------------------- 规则簿


@dataclass
class RuleBook:
    """所有阈值集中在这里。换行业 / 换规模时只调这里，不动规则逻辑。

    默认值针对 A 股制造业中型企业，是可辩护的起点，不是绝对真理。
    """

    # 增速剪刀差（百分点）
    receivable_gap_warn: float = 20.0
    receivable_gap_high: float = 40.0
    inventory_gap_warn: float = 20.0
    inventory_gap_high: float = 40.0

    # 毛利率同比变动（百分点）
    gross_margin_warn: float = 5.0
    gross_margin_high: float = 10.0

    # 经营现金流 / 净利润
    ocf_ratio_warn: float = 0.8
    ocf_ratio_high: float = 0.5

    # 资产负债率（%）
    debt_ratio_warn: float = 70.0
    debt_ratio_high: float = 80.0

    # 商誉 / 净资产（%）
    goodwill_warn: float = 30.0
    goodwill_high: float = 50.0

    # 净利率（%）低于此值视为微利，异常容忍度应提高
    thin_margin: float = 2.0

    # 单科目变动显著性（%，用于勾稽与波动扫描）
    line_item_swing_warn: float = 50.0
    line_item_swing_high: float = 100.0


# ---------------------------------------------------------------- 规则实现
#
# 每条规则签名统一：(当前期, 上期, 规则簿) -> Finding | None
# 返回 None 表示该规则未触发。


def rule_receivable_growth_gap(
    cur: FinancialStatement, prev: FinancialStatement, rb: RuleBook
) -> Finding | None:
    """应收账款增速 vs 营收增速：收入质量的核心信号。

    若应收增长显著快于营收，说明收入可能靠放宽信用政策换来的，
    回款风险上升，甚至有提前确认收入的嫌疑。
    """
    g_recv = cur.yoy("accounts_receivable", prev)
    g_rev = cur.yoy("revenue", prev)
    if g_recv is None or g_rev is None:
        return None
    gap = g_recv - g_rev
    if gap < rb.receivable_gap_warn:
        return None
    level = Level.HIGH if gap >= rb.receivable_gap_high else Level.WARN
    return Finding(
        rule_id="AR_GAP",
        level=level,
        title="应收账款增速显著快于营业收入",
        detail=(
            f"应收账款同比 {g_recv:+.1f}%，营业收入同比 {g_rev:+.1f}%，"
            f"剪刀差 {gap:.1f} 个百分点（观察线 {rb.receivable_gap_warn:.0f}pct）。"
            f"收入可能依赖放宽信用政策实现，回款质量存疑。"
        ),
        items=["accounts_receivable", "revenue"],
        evidence={"receivable_yoy": round(g_recv, 2), "revenue_yoy": round(g_rev, 2), "gap_pct": round(gap, 2)},
        basis="应收增速与营收增速的剪刀差是收入质量恶化的经典先行指标",
    )


def rule_inventory_growth_gap(
    cur: FinancialStatement, prev: FinancialStatement, rb: RuleBook
) -> Finding | None:
    """存货增速 vs 营收增速：存货积压 / 跌价准备不足。"""
    g_inv = cur.yoy("inventory", prev)
    g_rev = cur.yoy("revenue", prev)
    if g_inv is None or g_rev is None:
        return None
    gap = g_inv - g_rev
    if gap < rb.inventory_gap_warn:
        return None
    level = Level.HIGH if gap >= rb.inventory_gap_high else Level.WARN
    return Finding(
        rule_id="INV_GAP",
        level=level,
        title="存货增速显著快于营业收入",
        detail=(
            f"存货同比 {g_inv:+.1f}%，营业收入同比 {g_rev:+.1f}%，剪刀差 {gap:.1f} 个百分点。"
            f"存在存货积压或跌价准备计提不充分的风险。"
        ),
        items=["inventory", "revenue"],
        evidence={"inventory_yoy": round(g_inv, 2), "revenue_yoy": round(g_rev, 2), "gap_pct": round(gap, 2)},
        basis="存货增速持续高于营收增速通常指向滞销或减值计提不足",
    )


def rule_gross_margin_swing(
    cur: FinancialStatement, prev: FinancialStatement, rb: RuleBook
) -> Finding | None:
    """毛利率同比大幅变动，需要合理解释。"""
    gm_cur, gm_prev = cur.gross_margin(), prev.gross_margin()
    if gm_cur is None or gm_prev is None:
        return None
    delta = gm_cur - gm_prev
    if abs(delta) < rb.gross_margin_warn:
        return None
    level = Level.HIGH if abs(delta) >= rb.gross_margin_high else Level.WARN
    direction = "上升" if delta > 0 else "下降"
    return Finding(
        rule_id="GM_SWING",
        level=level,
        title=f"毛利率同比大幅{direction}",
        detail=(
            f"毛利率由 {gm_prev:.2f}% 变为 {gm_cur:.2f}%，变动 {delta:+.2f} 个百分点"
            f"（观察线 ±{rb.gross_margin_warn:.0f}pct）。需核实成本结转或产品结构是否发生变化。"
        ),
        items=["revenue", "cost_of_sales"],
        evidence={"gross_margin_prev": round(gm_prev, 2), "gross_margin_cur": round(gm_cur, 2), "delta_pct": round(delta, 2)},
        basis="毛利率异常波动需与产品结构、原材料价格、成本结转口径变化相互印证",
    )


def rule_cash_profit_divergence(
    cur: FinancialStatement, prev: FinancialStatement, rb: RuleBook
) -> Finding | None:
    """经营现金流 / 净利润：利润是否有现金支撑。"""
    ratio = cur.ocf_to_profit()
    if ratio is None:
        return None
    if ratio >= rb.ocf_ratio_warn:
        return None
    level = Level.HIGH if ratio < rb.ocf_ratio_high else Level.WARN
    return Finding(
        rule_id="OCF_DIVERGE",
        level=level,
        title="经营现金流与净利润背离",
        detail=(
            f"经营活动现金流 / 净利润 = {ratio:.2f}（健康观察线 ≥ {rb.ocf_ratio_warn:.1f}）。"
            f"利润的现金含量偏低，需关注是否存在纸面利润或回款恶化。"
        ),
        items=["operating_cash_flow", "net_profit"],
        evidence={"ocf_to_profit": round(ratio, 3)},
        basis="长期看经营现金流应匹配净利润，持续背离是盈余质量偏弱的直接证据",
    )


def rule_debt_ratio(
    cur: FinancialStatement, prev: FinancialStatement, rb: RuleBook
) -> Finding | None:
    """资产负债率过高。"""
    dr = cur.debt_ratio()
    if dr is None or dr < rb.debt_ratio_warn:
        return None
    level = Level.HIGH if dr >= rb.debt_ratio_high else Level.WARN
    prev_dr = prev.debt_ratio()
    trend = ""
    if prev_dr is not None:
        trend = f"，上年同期 {prev_dr:.2f}%（{dr - prev_dr:+.2f}pct）"
    return Finding(
        rule_id="DEBT_RATIO",
        level=level,
        title="资产负债率处于高位",
        detail=(
            f"资产负债率 {dr:.2f}%（观察线 {rb.debt_ratio_warn:.0f}%）{trend}。"
            f"偿债压力与再融资能力需一并评估。"
        ),
        items=["total_liabilities", "total_assets"],
        evidence={"debt_ratio": round(dr, 2), "prev_debt_ratio": round(prev_dr, 2) if prev_dr is not None else None},
        basis="资产负债率是杠杆水平与偿债风险的基础度量",
    )


def rule_goodwill_ratio(
    cur: FinancialStatement, prev: FinancialStatement, rb: RuleBook
) -> Finding | None:
    """商誉占净资产比重过高，减值风险敞口大。"""
    gw = cur.get("goodwill")
    ne = cur.get("net_assets")
    if gw is None or ne is None or ne <= 0:
        return None
    ratio = gw / ne * 100.0
    if ratio < rb.goodwill_warn:
        return None
    level = Level.HIGH if ratio >= rb.goodwill_high else Level.WARN
    return Finding(
        rule_id="GOODWILL",
        level=level,
        title="商誉占净资产比重偏高",
        detail=(
            f"商誉 {gw:,.0f} 万元，占净资产 {ratio:.2f}%（观察线 {rb.goodwill_warn:.0f}%）。"
            f"若被并购标的业绩不达预期，存在集中减值风险。"
        ),
        items=["goodwill", "net_assets"],
        evidence={"goodwill": gw, "net_assets": ne, "goodwill_ratio": round(ratio, 2)},
        basis="商誉/净资产是并购减值风险敞口的惯用度量",
    )


def rule_margin_erosion(
    cur: FinancialStatement, prev: FinancialStatement, rb: RuleBook
) -> Finding | None:
    """营收增长但净利率下滑：增收不增利。"""
    g_rev = cur.yoy("revenue", prev)
    nm_cur, nm_prev = cur.net_margin(), prev.net_margin()
    if g_rev is None or nm_cur is None or nm_prev is None:
        return None
    if g_rev <= 0 or nm_cur >= nm_prev:
        return None
    drop = nm_prev - nm_cur
    level = Level.HIGH if drop >= rb.gross_margin_high else Level.WARN
    return Finding(
        rule_id="MARGIN_EROSION",
        level=level,
        title="营业收入增长但净利率下滑",
        detail=(
            f"营业收入同比 {g_rev:+.1f}%，净利率由 {nm_prev:.2f}% 降至 {nm_cur:.2f}%"
            f"（-{drop:.2f}pct）。规模扩张未转化为盈利改善，需核查费用或成本端。"
        ),
        items=["revenue", "net_profit"],
        evidence={"revenue_yoy": round(g_rev, 2), "net_margin_prev": round(nm_prev, 2), "net_margin_cur": round(nm_cur, 2)},
        basis="增收不增利是盈利质量恶化的典型形态",
    )


def rule_balance_sheet_identity(
    cur: FinancialStatement, prev: FinancialStatement, rb: RuleBook
) -> Finding | None:
    """勾稽校验：资产 = 负债 + 所有者权益。

    这不是财务分析规则，而是**数据可信度检查**。
    勾稽不平时，后面所有分析都不可信——必须先报出来。
    """
    ta, tl = cur.get("total_assets"), cur.get("total_liabilities")
    te = cur.get("total_equity")
    if ta is None or tl is None or te is None:
        return None
    diff = ta - (tl + te)
    tol = max(abs(ta) * 0.001, 1.0)   # 千分之一容差，容忍四舍五入
    if abs(diff) <= tol:
        return None
    return Finding(
        rule_id="BS_IDENTITY",
        level=Level.HIGH,
        title="资产负债表勾稽不平",
        detail=(
            f"资产总计 {ta:,.0f}，负债与所有者权益合计 {tl + te:,.0f}，"
            f"差额 {diff:,.0f}（容差 {tol:,.0f}）。数据存在录入或口径问题，"
            f"在此之前的所有分析结论均不可采信。"
        ),
        items=["total_assets", "total_liabilities", "total_equity"],
        evidence={"diff": round(diff, 2), "tolerance": round(tol, 2)},
        basis="会计恒等式是财报数据可信度的底线校验",
    )


def rule_line_item_swing(
    cur: FinancialStatement, prev: FinancialStatement, rb: RuleBook
) -> Finding | None:
    """科目同比大幅波动扫描：找出所有异常科目，供人工逐项核查。

    注意这条规则与上面几条会有重叠（比如应收已单独报过）。
    重叠是刻意的——专项规则给"为什么重要"，扫描规则给"全面覆盖"，
    两者回答的是不同问题。
    """
    swings: list[tuple[str, float]] = []
    for item, cur_v in cur.items.items():
        prev_v = prev.items.get(item)
        if prev_v is None or prev_v == 0 or cur_v is None:
            continue
        chg = (cur_v - prev_v) / abs(prev_v) * 100.0
        if abs(chg) >= rb.line_item_swing_warn:
            swings.append((item, chg))
    if not swings:
        return None
    swings.sort(key=lambda x: -abs(x[1]))
    top = swings[:6]
    level = (
        Level.HIGH
        if any(abs(c) >= rb.line_item_swing_high for _, c in swings)
        else Level.WARN
    )
    listed = "；".join(f"{k} {c:+.1f}%" for k, c in top)
    return Finding(
        rule_id="SWING_SCAN",
        level=level,
        title=f"存在 {len(swings)} 个科目同比大幅波动",
        detail=(
            f"波动超过 ±{rb.line_item_swing_warn:.0f}% 的科目：{listed}"
            f"{'…' if len(swings) > len(top) else ''}。建议逐项核查原始凭证。"
        ),
        items=[k for k, _ in top],
        evidence={"swings": {k: round(c, 2) for k, c in top}, "total_flagged": len(swings)},
        basis="科目异常波动扫描用于全面覆盖，避免专项规则遗漏",
    )


# ---------------------------------------------------------------- 规则集


DEFAULT_RULES: tuple[Callable[..., Finding | None], ...] = (
    rule_balance_sheet_identity,      # 先做数据可信度校验
    rule_receivable_growth_gap,
    rule_inventory_growth_gap,
    rule_gross_margin_swing,
    rule_cash_profit_divergence,
    rule_debt_ratio,
    rule_goodwill_ratio,
    rule_margin_erosion,
    rule_line_item_swing,
)


@dataclass
class RuleEngine:
    """规则的执行容器。rules 可替换，便于做规则的增删实验。"""

    book: RuleBook = field(default_factory=RuleBook)
    rules: tuple[Callable[..., Finding | None], ...] = DEFAULT_RULES

    def run(
        self, cur: FinancialStatement, prev: FinancialStatement
    ) -> list[Finding]:
        """跑全部规则，按严重度倒序返回（HIGH 在前）。"""
        out: list[Finding] = []
        for rule in self.rules:
            try:
                f = rule(cur, prev, self.book)
            except Exception as e:  # noqa: BLE001 — 单条规则失败不应拖垮整轮
                out.append(
                    Finding(
                        rule_id=f"ERR_{rule.__name__}",
                        level=Level.INFO,
                        title=f"规则 {rule.__name__} 执行失败",
                        detail=f"该规则被跳过，其余规则不受影响。错误：{e}",
                        items=[],
                        basis="规则隔离机制",
                    )
                )
                continue
            if f is not None:
                out.append(f)
        out.sort(key=lambda f: (-int(f.level), f.rule_id))
        return out
