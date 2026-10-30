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
from typing import Any, Callable, Sequence

from .benchmarks import BenchmarkBook
from .models import FinancialStatement, Finding, Level, StatementSeries


# ---------------------------------------------------------------- 盈利质量指标表
#
# QUALITY_DRIFT（多项盈利质量指标连续同向恶化）的指标清单：名称、重要性权重、
# 恶化方向、取值函数。放在 RuleBook 之前，因为 RuleBook 的默认权重表由它生成，
# 配置校验也拿它当参照系。
#
# 权重排序的依据：
#   毛利率       竞争地位与定价权的直接体现，盈利质量恶化的领先指标；
#   净利率       盈利的最终结果，受毛利率驱动，另含费用与非经常项信息；
#   经营现金流/净利润  利润的含金量（应计质量），"纸面利润"的照妖镜；
#   资产负债率   偿债安全边际，与"盈利质量"相距较远；
#   流动比率     短期偿付能力，同上，只作佐证不作主证。
# 前两项度量盈利本身故权重最高，第三项检验盈利质量居次，后两项关乎偿债
# 安全而非盈利质量，权重减半。
#
# 曾经的写法是 5 项等权计数（trend_min_agree：≥3 项同向即触发）。等权的
# 隐含假设是"五项一样重要"——实测思维下这不成立："经营现金流+杠杆+流动性"
# 三项齐坏，不该与"毛利率+净利率"两项齐坏判成同一严重度，后者是更强的信号。
QUALITY_DRIFT_INDICATORS: tuple[tuple[str, float, int, Callable], ...] = (
    # (指标名, 重要性权重, 恶化方向, 取值函数)；方向 -1 = 越小越坏，+1 = 越大越坏
    ("毛利率", 3.0, -1, lambda s: s.gross_margin()),
    ("净利率", 2.0, -1, lambda s: s.net_margin()),
    ("经营现金流/净利润", 2.0, -1, lambda s: s.ocf_to_profit()),
    ("资产负债率", 1.0, +1, lambda s: s.debt_ratio()),
    ("流动比率", 1.0, -1, lambda s: s.current_ratio()),
)


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

    # 由盈转亏后的亏损深度（净利率，%）。转亏本身是定性变化，一律要报；
    # 深度达到该值判重大，否则判关注。分界线取 5 的理由：
    # 净利率 -5% 意味着每 100 元收入亏 5 元，已不是"微利转微亏"的边界状态，
    # 而是需要解释成本结构或资产减值的一次实质亏损。
    net_loss_high: float = 5.0

    # 单科目变动显著性（%，用于勾稽与波动扫描）
    line_item_swing_warn: float = 50.0
    line_item_swing_high: float = 100.0

    # 多期趋势：信号需连续出现多少期才启用趋势规则。
    # 3 是下限——2 期只能做同比，谈不上"趋势"。
    trend_min_periods: int = 3

    # 持续性升级：最近几个相邻期配对**全部**触发才算"持续"。
    # 与 trend_min_periods 拆开的理由：前者是"数据够不够做趋势分析"的门槛，
    # 后者是"多高的复发才算结构性问题"的判定标准，两者没有必然联系。
    # 曾经的写法是 n_pairs = trend_min_periods - 1，即一个参数承载两个语义——
    # 想放宽持续性判定（比如"连续三年"要 3 对）就不得不连带把启用门槛
    # 从 3 期抬到 4 期，而后者会让只有 3 期数据的公司连趋势规则都跑不上。
    # 默认 2 与拆分前逐字等价（3 - 1 = 2）。
    persistence_min_pairs: int = 2

    # 盈利质量同向恶化（QUALITY_DRIFT）：5 项指标的重要性权重表。
    # 默认值见上方 QUALITY_DRIFT_INDICATORS 的排序说明。
    # 与等权计数（旧 trend_min_agree）的差别：加权后"毛利率+净利率"两项齐坏
    # （3+2=5）即触发，而"现金流+杠杆+流动性"三项齐坏（2+1+1=4）不触发。
    quality_drift_weights: dict[str, float] = field(
        default_factory=lambda: {n: w for n, w, _d, _g in QUALITY_DRIFT_INDICATORS}
    )

    # 加权得分门槛。默认 5 的含义：
    #   最强两指标组合（毛利率 3 + 净利率 2）恰好达标——两个盈利维度连续同向
    #   走坏即构成恶化通道，不需要弱指标凑数；
    #   纯弱指标组合够不着——等权时代它们凑够 3 项就能触发，正是要纠正的方向。
    # 与权重表配合保证"至少 2 项同向"：单项最高权重 3 < 5，
    # 单项指标再重也不会单独触发（见规则内的结构性底线）。
    quality_drift_min_score: float = 5.0

    def __post_init__(self) -> None:
        # 明显不合法的配置要当场炸，不能静默跑出没有意义的结果——
        # 与 parse_amount 认不出格式就 raise 是同一条纪律。
        if self.trend_min_periods < 2:
            raise ValueError(
                f"trend_min_periods 至少为 2（两期才能做一次同比），"
                f"收到 {self.trend_min_periods}"
            )
        if self.persistence_min_pairs < 1:
            raise ValueError(
                f"persistence_min_pairs 至少为 1（看几对相邻期），"
                f"收到 {self.persistence_min_pairs}"
            )
        # 权重表的键必须与指标表一一对应：少一个键意味着该指标被静默剔除，
        # 多一个键意味着配置里躺着永远不生效的权重——两种都要当场炸。
        expected = {n for n, _w, _d, _g in QUALITY_DRIFT_INDICATORS}
        got = set(self.quality_drift_weights)
        if got != expected:
            raise ValueError(
                f"quality_drift_weights 的键必须与指标表一致"
                f"（缺 {sorted(expected - got)}，多 {sorted(got - expected)}）"
            )
        if any(w <= 0 for w in self.quality_drift_weights.values()):
            raise ValueError(
                f"quality_drift_weights 的权重必须为正数，"
                f"收到 {self.quality_drift_weights}"
            )
        if self.quality_drift_min_score <= 0:
            raise ValueError(
                f"quality_drift_min_score 必须为正数，"
                f"收到 {self.quality_drift_min_score}"
            )


# ---------------------------------------------------------------- 行业校准辅助
#
# 规则与阈值分离之后，第二个分离是"判定策略"与"参照数据"分离：
#     RuleBook      判定策略（阈值该定多少，人拍的）
#     BenchmarkBook 参照数据（同行实际分布，随样本更新）
# 两者都不要求改规则逻辑，但生命周期不同，所以分开存放。


def _benchmark_aware(fn):
    """标记该规则支持行业基准校准。

    与 `_series_rule` 同一套思路：用标记而非签名内省，
    让 RuleEngine 在分发时知道要不要多传一个 benchmarks 参数。
    """
    fn.is_benchmark_aware = True
    return fn


@dataclass(frozen=True)
class CalibratedLine:
    """一次行业校准的结果。

    `warn_src == "absolute"` 时表示没校准上（无簿 / 行业不在表里 /
    样本量不足 / 该指标没有分位数），此时 warn/high 就是绝对线原值，
    note 为空——四条规则据此保证"没基准时行为逐字不变"。
    """

    warn: float
    high: float
    warn_src: str                 # "absolute" | "industry:p75"
    high_src: str                 # "absolute" | "industry:p90" | "industry:p75x2.00"
    note: str = ""                # 同行区间描述，拼进 detail；绝对线下为空
    p50: float | None = None      # 行业中位数（机器可读）
    p90: float | None = None      # 行业 p90（机器可读）

    @property
    def calibrated(self) -> bool:
        return self.warn_src != "absolute"


def _industry_line(
    bm: BenchmarkBook | None,
    cur: FinancialStatement,
    metric: str,
    abs_warn: float,
    abs_high: float,
    q: str = "p75",
) -> CalibratedLine:
    """按行业分位校准观察线，返回关注线 / 重大线及其来源。

    四条规则，缺一不可：

    1. **无基准簿 / 行业不在表里 / 样本量不足 → 原样返回绝对线**。
       这是向后兼容底线：不传 benchmarks 时，所有规则行为与改造前逐字一致。

    2. **关注线取 p75，重大线取 p90，两端都 grounded 在同行分布上**。
       p75 = "比四分之三的同行更极端，需要解释"（关注级）；
       p90 = "行业最差十分之一，需要专项核查"（重大级）。
       早年重大线按 `abs_high / abs_warn` 的比例推导，那是拍倍数不是看分布：
       食品饮料 ar_growth_gap 实测 p90 = 73.3，而比例线只有 47.3——
       按比例线会把行业尾部 10% 的公司判成"只到关注级"。
       p90 缺失时（老簿子只存了 p75）退回比例推导，行为可预期。

    3. **必须分别披露两条线的来源**。warn_src / high_src 写进 evidence，
       财务工具不能悄悄换阈值：读者看到"观察线 12pct"时，
       必须能回答"这个 12 是哪来的、重大线 20 又是哪来的"。

    4. **必须把整条区间给读者，不能只给一根线**。单说"超过 p75"会被读成
       "绝对偏高"，而按定义任一行约 25% 的公司落在 p75 之上——
       行业中位数才是"常态"的锚点。note 里带上 p50 / p75 / p90，
       读者能自己定位公司处于哪个区间，而不是只剩一个孤零零的阈值。
    """
    industry = cur.meta.get("industry")
    if bm is None or not industry:
        return CalibratedLine(float(abs_warn), float(abs_high), "absolute", "absolute")
    warn, warn_src = bm.line(industry, metric, abs_warn, q)
    if warn_src == "absolute":
        return CalibratedLine(float(abs_warn), float(abs_high), "absolute", "absolute")

    b = bm.get(industry, metric)
    p50 = b.at("p50") if b is not None else None
    p90 = b.at("p90") if b is not None else None
    if p90 is None:
        ratio = (abs_high / abs_warn) if abs_warn else 2.0
        high, high_src = warn * ratio, f"industry:{q}x{ratio:.2f}"
    else:
        high, high_src = p90, "industry:p90"
    return CalibratedLine(
        warn=warn,
        high=high,
        warn_src=warn_src,
        high_src=high_src,
        note=_band_note(industry, p50, warn, p90),
        p50=p50,
        p90=p90,
    )


def _band_note(
    industry: str | None, p50: float | None, p75: float, p90: float | None
) -> str:
    """同行分布的区间描述。缺哪个分位就不写哪个，不补默认值。"""
    parts = []
    if p50 is not None:
        parts.append(f"中位 {p50:.1f}")
    parts.append(f"p75 {p75:.1f}")
    if p90 is not None:
        parts.append(f"p90 {p90:.1f}")
    return f"；已按{industry}行业分位校准（{'，'.join(parts)}）"


def _line_evidence(line: CalibratedLine) -> dict[str, Any]:
    """四条规则共用的 evidence 片段：阈值、两条线的来源、行业区间。

    单独抽出来是因为这四个字段必须在每条规则里逐字一致——
    复核人要能跨规则比较"这条的 p90 和那条的 p90 是不是同一本簿子"。
    """
    return {
        "line_warn": round(line.warn, 2),
        "line_high": round(line.high, 2),
        "line_source": line.warn_src,
        "line_high_source": line.high_src,
        "industry_p50": None if line.p50 is None else round(line.p50, 2),
        "industry_p90": None if line.p90 is None else round(line.p90, 2),
    }


# ---------------------------------------------------------------- 规则实现
#
# 每条规则签名统一：(当前期, 上期, 规则簿) -> Finding | None
# 返回 None 表示该规则未触发。


@_benchmark_aware
def rule_receivable_growth_gap(
    cur: FinancialStatement,
    prev: FinancialStatement,
    rb: RuleBook,
    benchmarks: BenchmarkBook | None = None,
) -> Finding | None:
    """应收账款增速 vs 营收增速：收入质量的核心信号。

    若应收增长显著快于营收，说明收入可能靠放宽信用政策换来的，
    回款风险上升，甚至有提前确认收入的嫌疑。

    行业校准理由：快速消费品靠渠道铺货，应收增速快于营收近乎常态；
    软件开发多为预收款，应收本就少。同一条 20pct 线衡量两者必错一头。
    """
    g_recv = cur.yoy("accounts_receivable", prev)
    g_rev = cur.yoy("revenue", prev)
    if g_recv is None or g_rev is None:
        return None
    gap = g_recv - g_rev
    line = _industry_line(
        bm=benchmarks, cur=cur, metric="ar_growth_gap",
        abs_warn=rb.receivable_gap_warn, abs_high=rb.receivable_gap_high,
    )
    warn, high = line.warn, line.high
    if gap < warn:
        return None
    level = Level.HIGH if gap >= high else Level.WARN
    return Finding(
        rule_id="AR_GAP",
        level=level,
        title="应收账款增速显著快于营业收入",
        detail=(
            f"应收账款同比 {g_recv:+.1f}%，营业收入同比 {g_rev:+.1f}%，"
            f"剪刀差 {gap:.1f} 个百分点（观察线 {warn:.0f}pct{line.note}）。"
            f"收入可能依赖放宽信用政策实现，回款质量存疑。"
        ),
        items=["accounts_receivable", "revenue"],
        evidence={
            "receivable_yoy": round(g_recv, 2),
            "revenue_yoy": round(g_rev, 2),
            "gap_pct": round(gap, 2),
            **_line_evidence(line),
        },
        basis="应收增速与营收增速的剪刀差是收入质量恶化的经典先行指标",
    )


@_benchmark_aware
def rule_inventory_growth_gap(
    cur: FinancialStatement,
    prev: FinancialStatement,
    rb: RuleBook,
    benchmarks: BenchmarkBook | None = None,
) -> Finding | None:
    """存货增速 vs 营收增速：存货积压 / 跌价准备不足。

    行业校准理由：工程机械备货周期长、在产订单多，存货增速高于营收是常态；
    软件开发的"存货"基本只剩实施成本，波动天然小。
    """
    g_inv = cur.yoy("inventory", prev)
    g_rev = cur.yoy("revenue", prev)
    if g_inv is None or g_rev is None:
        return None
    gap = g_inv - g_rev
    line = _industry_line(
        bm=benchmarks, cur=cur, metric="inv_growth_gap",
        abs_warn=rb.inventory_gap_warn, abs_high=rb.inventory_gap_high,
    )
    warn, high = line.warn, line.high
    if gap < warn:
        return None
    level = Level.HIGH if gap >= high else Level.WARN
    return Finding(
        rule_id="INV_GAP",
        level=level,
        title="存货增速显著快于营业收入",
        detail=(
            f"存货同比 {g_inv:+.1f}%，营业收入同比 {g_rev:+.1f}%，剪刀差 {gap:.1f} 个百分点"
            f"（观察线 {warn:.0f}pct{line.note}）。"
            f"存在存货积压或跌价准备计提不充分的风险。"
        ),
        items=["inventory", "revenue"],
        evidence={
            "inventory_yoy": round(g_inv, 2),
            "revenue_yoy": round(g_rev, 2),
            "gap_pct": round(gap, 2),
            **_line_evidence(line),
        },
        basis="存货增速持续高于营收增速通常指向滞销或减值计提不足",
    )


@_benchmark_aware
def rule_gross_margin_swing(
    cur: FinancialStatement,
    prev: FinancialStatement,
    rb: RuleBook,
    benchmarks: BenchmarkBook | None = None,
) -> Finding | None:
    """毛利率同比大幅变动，需要合理解释。

    行业校准理由：软件开发的毛利率对版本发布节奏极敏感，单期 5pct 波动
    未必异常；快速消费品的成本结构刚性，波动通常来自真实经营变化。
    """
    gm_cur, gm_prev = cur.gross_margin(), prev.gross_margin()
    if gm_cur is None or gm_prev is None:
        return None
    delta = gm_cur - gm_prev
    line = _industry_line(
        bm=benchmarks, cur=cur, metric="gross_margin_delta",
        abs_warn=rb.gross_margin_warn, abs_high=rb.gross_margin_high,
    )
    warn, high = line.warn, line.high
    if abs(delta) < warn:
        return None
    level = Level.HIGH if abs(delta) >= high else Level.WARN
    direction = "上升" if delta > 0 else "下降"
    return Finding(
        rule_id="GM_SWING",
        level=level,
        title=f"毛利率同比大幅{direction}",
        detail=(
            f"毛利率由 {gm_prev:.2f}% 变为 {gm_cur:.2f}%，变动 {delta:+.2f} 个百分点"
            f"（观察线 ±{warn:.0f}pct{line.note}）。需核实成本结转或产品结构是否发生变化。"
        ),
        items=["revenue", "cost_of_sales"],
        evidence={
            "gross_margin_prev": round(gm_prev, 2),
            "gross_margin_cur": round(gm_cur, 2),
            "delta_pct": round(delta, 2),
            **_line_evidence(line),
        },
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


@_benchmark_aware
def rule_debt_ratio(
    cur: FinancialStatement,
    prev: FinancialStatement,
    rb: RuleBook,
    benchmarks: BenchmarkBook | None = None,
) -> Finding | None:
    """资产负债率过高。

    行业校准理由：这是行业差异最大的一项。房地产与建筑施工 78% 是常态，
    软件开发 35% 已属偏高。用统一的 70% 线，前者永远不报警，后者天天误报。
    """
    dr = cur.debt_ratio()
    if dr is None:
        return None
    line = _industry_line(
        bm=benchmarks, cur=cur, metric="debt_ratio",
        abs_warn=rb.debt_ratio_warn, abs_high=rb.debt_ratio_high,
    )
    warn, high = line.warn, line.high
    if dr < warn:
        return None
    level = Level.HIGH if dr >= high else Level.WARN
    prev_dr = prev.debt_ratio()
    trend = ""
    if prev_dr is not None:
        trend = f"，上年同期 {prev_dr:.2f}%（{dr - prev_dr:+.2f}pct）"
    return Finding(
        rule_id="DEBT_RATIO",
        level=level,
        title="资产负债率处于高位",
        detail=(
            f"资产负债率 {dr:.2f}%（观察线 {warn:.0f}%{line.note}）{trend}。"
            f"偿债压力与再融资能力需一并评估。"
        ),
        items=["total_liabilities", "total_assets"],
        evidence={
            "debt_ratio": round(dr, 2),
            "prev_debt_ratio": round(prev_dr, 2) if prev_dr is not None else None,
            **_line_evidence(line),
        },
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


def rule_net_margin_turn(
    cur: FinancialStatement, prev: FinancialStatement, rb: RuleBook
) -> Finding | None:
    """净利率由正转负：由盈转亏。

    为什么必须单独成条，而不是靠 MARGIN_EROSION 或 QUALITY_DRIFT 兜住
    ------------------------------------------------------------------
    三条规则的语义各不相同，实测中各自漏的东西不一样：

        MARGIN_EROSION   要求营收增长（g_rev > 0）。营收下滑时不触发，
                         即"减收更减利"这种更常见的恶化形态没人管；
        QUALITY_DRIFT    三期趋势规则，且加权后净利率只占 2 分。
                         净利率(2)+杠杆(1)+流动性(1)=4 < 5 的转亏会被抑制；
                         更重要的是，转亏是**单年事件**，要求三期数据本身就错配——
                         等三期齐了，亏损已经发生两年了。

    实测（911 家 A 股，2024A→2025A）有 79 家净利率由正转负，其中 31 家
    只落到泛化的 SWING_SCAN（"存在 N 个科目同比大幅波动"），最深一例
    净利率由 2.31% 掉到 -17.37%。这不是阈值松紧问题，是缺一条规则。

    为什么不分「亏损扩大」（负数→更负）
    ----------------------------------
    那属于"恶化幅度"范畴，与转亏的"性质变化"不同：一家常年亏损的公司
    净利率从 -3% 到 -6% 不改变定性判断，需要的是幅度阈值而非零轴穿越。
    留作已知局限，不在本条规则里混入两个语义。
    """
    nm_prev, nm_cur = prev.net_margin(), cur.net_margin()
    if nm_prev is None or nm_cur is None:
        return None
    if not (nm_prev > 0 >= nm_cur):
        return None

    depth = -nm_cur                     # 亏损深度（正数）
    delta = nm_cur - nm_prev            # 负值，降幅
    level = Level.HIGH if depth >= rb.net_loss_high else Level.WARN

    g_rev = cur.yoy("revenue", prev)
    if g_rev is None:
        shape = "营收数据缺失，无法判断是缩量还是增量亏损"
    elif g_rev < 0:
        shape = f"同期营业收入同比 {g_rev:+.1f}%，属于收入下滑叠加盈利转亏"
    else:
        shape = f"同期营业收入同比 {g_rev:+.1f}%，属于增收却转亏，需查成本费用端"

    return Finding(
        rule_id="NET_MARGIN_TURN",
        level=level,
        title=f"净利率由正转负（{nm_prev:.2f}% → {nm_cur:.2f}%）",
        detail=(
            f"净利率由 {nm_prev:.2f}% 降至 {nm_cur:.2f}%，下降 {-delta:.2f} 个百分点，"
            f"已由盈转亏（亏损深度 {depth:.2f}%）。{shape}。"
            f"这是盈利性质的改变，不是幅度波动，必须核查非经常性损益、"
            f"资产减值与费用确认是否集中释放。"
        ),
        items=["revenue", "net_profit"],
        evidence={
            "net_margin_prev": round(nm_prev, 2),
            "net_margin_cur": round(nm_cur, 2),
            "delta_pct": round(delta, 2),
            "loss_depth": round(depth, 2),
            "revenue_yoy": None if g_rev is None else round(g_rev, 2),
        },
        basis="由盈转亏是盈利性质的定性变化，与幅度阈值属于不同维度，需独立设线",
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


# ---------------------------------------------------------------- 多期趋势规则
#
# 两条规则签名与上面不同：接收整个序列而非相邻两期。
# 用 is_series_rule 标记区分，RuleEngine 据此选择调用方式。


def _series_rule(fn):
    fn.is_series_rule = True
    return fn


def _escalate(level: Level) -> Level:
    """提升一级严重度。已在最高级则保持不变。"""
    order = [Level.INFO, Level.WARN, Level.HIGH]
    i = min(order.index(level) + 1, len(order) - 1)
    return order[i]


def _persistence_note(periods: list[str], escalated: bool) -> str:
    span = f"{periods[0]}→{periods[-1]}"
    if escalated:
        return (
            f"该信号在 {span} 连续 {len(periods) - 1} 个报告期重复出现，"
            f"非孤立波动，已升级为{Level.HIGH.label}。"
        )
    return (
        f"该信号在 {span} 连续 {len(periods) - 1} 个报告期重复出现，"
        f"非孤立波动，属结构性问题。"
    )


def escalate_persistent_findings(
    findings: list[Finding],
    series: StatementSeries,
    rb: RuleBook,
    benchmarks: BenchmarkBook | None = None,
) -> list[Finding]:
    """对跨期持续出现的信号升级严重度。

    这是多期趋势的核心机制，刻意做成"升级原发现"而不是"新增一条发现"：
    新增会导致同一个问题在报告里出现两次，风险分也会被重复计算，
    读者看到的是"4 项异常"实际只有 2 个问题。

    判定标准：某条规则在最近 `persistence_min_pairs` 个相邻期配对上**全部**触发。
    "连续"而非"累计"——中间断一期说明情况有变化，不能算持续。

    `persistence_min_pairs` 与 `trend_min_periods` 是两个独立的参数：
    前者管"复发几次算持续"（判定标准），后者管"几期数据才配做趋势分析"
    （启用门槛）。拆分前两者绑死（n_pairs = trend_min_periods - 1），
    想把持续性判定放宽到 3 对就不得不把启用门槛抬到 4 期。

    benchmarks 必须与主引擎用的是同一本：持续性判定要重跑历史配对，
    若这里用绝对线而主引擎用了行业校准线，两边的触发集不一致，
    升级结论就不可信了。
    """
    if not series.has_trend(rb.trend_min_periods):
        return findings

    n_pairs = rb.persistence_min_pairs
    recent = series.trailing_pairs(n_pairs)
    if len(recent) < n_pairs:
        return findings

    # 每条规则在多少个最近配对里触发过
    fired: dict[str, int] = {}
    for prev_s, cur_s in recent:
        for rule in _PAIR_RULES:
            try:
                if getattr(rule, "is_benchmark_aware", False):
                    f = rule(cur_s, prev_s, rb, benchmarks=benchmarks)
                else:
                    f = rule(cur_s, prev_s, rb)
            except Exception:  # noqa: BLE001 — 与主引擎一致的隔离策略
                continue
            if f is not None:
                fired[f.rule_id] = fired.get(f.rule_id, 0) + 1

    out: list[Finding] = []
    for f in findings:
        if fired.get(f.rule_id, 0) == n_pairs:
            new_level = _escalate(f.level)
            escalated = new_level != f.level
            f.level = new_level
            f.detail = f.detail + _persistence_note(series.periods[-(n_pairs + 1):], escalated)
            f.basis = (f.basis or "") + "；跨期持续性信号比单期跳变更能说明问题是结构性的"
            f.confidence = min(1.0, f.confidence + 0.1)
        out.append(f)
    out.sort(key=lambda x: (-int(x.level), x.rule_id))
    return out


@_series_rule
def rule_quality_deterioration(
    series: StatementSeries, rb: RuleBook
) -> Finding | None:
    """多项盈利质量指标连续同向恶化（按重要性加权判定）。

    与持续性升级互补：上面那条管"已触阈值的信号是否持续"，
    这条管"没触阈值但一直在缓慢变坏"的情况。
    毛利率每年降 1pct，单期看都够不上观察线，但连续四年就是明确的恶化通道——
    这种"温水煮青蛙"式的问题恰恰是单期分析最容易漏掉的。

    加权而非等权：5 项指标的重要性不同（见 QUALITY_DRIFT_INDICATORS）。
    等权计数时代，"经营现金流+杠杆+流动性"三项齐坏与"毛利率+净利率"两项
    齐坏同严重度；按权重后前者得分 4 不触发、后者得分 5 触发。

    结构性底线（不是可调参数）：至少 2 项指标同向才判定。单项指标再重也
    不算"多项同向"——单一毛利率连降四年可能是定价策略调整等正常经营行为，
    跨指标 corroboration 才是这条规则的存在理由。
    """
    if not series.has_trend(rb.trend_min_periods):
        return None

    sts = series.statements
    weights = rb.quality_drift_weights
    # 按配置的权重降序迭代：披露时最重要的指标排最前，且不依赖传入字典的顺序。
    # 权重并列时回退到指标表次序（即默认重要性排序），不用字符串排序——
    # 并列时谁前谁后应当是设计意图，不是 Unicode 码点。
    table_order = {name: i for i, (name, _w, _d, _g) in enumerate(QUALITY_DRIFT_INDICATORS)}
    ordered = sorted(
        QUALITY_DRIFT_INDICATORS, key=lambda t: (-weights[t[0]], table_order[t[0]])
    )

    worsening: list[tuple[str, float]] = []
    for name, _default_weight, direction, getter in ordered:
        vals = [getter(s) for s in sts]
        if any(v is None for v in vals):
            continue
        # 要求每一期都**实际**朝恶化方向走：严格大于 0。
        # 早先用 >= 0（允许持平），结果四期完全相同的数据也被判为"同向恶化"——
        # 持平既不是改善也不是恶化，报出来就是误报。
        deltas = [(vals[i + 1] - vals[i]) * direction for i in range(len(vals) - 1)]
        if all(d > 0 for d in deltas):
            worsening.append((name, weights[name]))

    score = sum(w for _name, w in worsening)
    if len(worsening) < 2 or score < rb.quality_drift_min_score:
        return None

    listed = "、".join(f"{name}（权重 {w:g}）" for name, w in worsening)
    return Finding(
        rule_id="QUALITY_DRIFT",
        level=Level.HIGH,
        title=f"{len(worsening)} 项盈利质量指标连续 {len(sts)} 期同向恶化",
        detail=(
            f"{listed} 在 {sts[0].period}→{sts[-1].period} 期间持续走坏，"
            f"每一期都朝同一方向变动；按重要性加权得分 {score:g}"
            f"（门槛 {rb.quality_drift_min_score:g}）。"
            f"单项看可能均未触及时值观察线，"
            f"但连续同向变动已构成明确的恶化通道，属于结构性问题。"
        ),
        items=["revenue", "net_profit", "operating_cash_flow"],
        evidence={
            "worsening": [name for name, _w in worsening],
            "weights": {name: w for name, w in worsening},
            "score": score,
            "min_score": rb.quality_drift_min_score,
            "periods": series.periods,
        },
        basis=(
            "单期阈值法会系统性漏掉'每期小幅恶化'型问题；"
            "多指标同向连续变动是识别慢性恶化的补充维度"
        ),
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
    rule_net_margin_turn,             # 定性变化：由盈转亏，两期即可判
    rule_line_item_swing,
    rule_quality_deterioration,       # 多期趋势：最后跑，需要完整序列
)

# 两期规则集合，供持续性升级复用（顺序与 DEFAULT_RULES 前段一致）
_PAIR_RULES: tuple[Callable[..., Finding | None], ...] = tuple(
    r for r in DEFAULT_RULES if not getattr(r, "is_series_rule", False)
)


@dataclass
class RuleEngine:
    """规则的执行容器。rules 可替换，便于做规则的增删实验。

    benchmarks 挂在引擎上而不是 RuleBook 里：RuleBook 是判定策略，
    BenchmarkBook 是参照数据，两者生命周期不同。策略是人拍的、稳定；
    行业分布随样本更新、会变。混在一个 dataclass 里会让"改阈值"
    和"换数据源"两件事纠缠不清。
    """

    book: RuleBook = field(default_factory=RuleBook)
    rules: tuple[Callable[..., Finding | None], ...] = DEFAULT_RULES
    benchmarks: BenchmarkBook | None = None

    def run(
        self,
        cur: FinancialStatement,
        prev: FinancialStatement,
        series: StatementSeries | None = None,
        benchmarks: BenchmarkBook | None = None,
    ) -> list[Finding]:
        """跑全部规则，按严重度倒序返回（HIGH 在前）。

        series 可选：传入多期序列时，额外执行趋势规则并对持续信号升级严重度。
        只传两期时行为与之前完全一致——这是为了让现有调用方无感升级。

        benchmarks 可选：覆盖引擎级默认值，便于同一次运行里对比
        "绝对线 vs 行业校准线"（A/B 场景）。不传则用构造时的那一本。
        """
        bm = benchmarks if benchmarks is not None else self.benchmarks
        out: list[Finding] = []
        for rule in self.rules:
            try:
                if getattr(rule, "is_series_rule", False):
                    if series is None or not series.has_trend(self.book.trend_min_periods):
                        continue
                    f = rule(series, self.book)
                elif getattr(rule, "is_benchmark_aware", False):
                    f = rule(cur, prev, self.book, benchmarks=bm)
                else:
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

        if series is not None:
            out = escalate_persistent_findings(out, series, self.book, benchmarks=bm)

        out.sort(key=lambda f: (-int(f.level), f.rule_id))
        return out
