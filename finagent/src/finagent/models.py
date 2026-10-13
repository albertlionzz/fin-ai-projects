"""项目二的数据模型：财务报表、异常发现、分析结论。

与项目一（finrag）的关系：
    finrag 解决"从制度文档里查到条款"；
    finagent 解决"从财务数字里发现异常并说明理由"。
    两者共用同一套工程约定：纯标准库、dataclass 驱动、可独立单测。

科目命名统一用英文 key，避免中文键在不同报表里的写法漂移
（"营业收入" / "营业总收入" / "主营业务收入"）。映射表见 data/statements。
"""

from __future__ import annotations

import json
import math as _math
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterable, Sequence


# ---------------------------------------------------------------- 严重度


class Level(IntEnum):
    """异常严重度。用 IntEnum 是为了可直接排序。

    取值顺序即排序优先级，越大越严重。
    """

    INFO = 0    # 提示：仅供参考
    WARN = 1    # 关注：需要解释
    HIGH = 2    # 重大：需要专项核查

    @property
    def label(self) -> str:
        return {0: "提示", 1: "关注", 2: "重大"}[int(self)]


# ---------------------------------------------------------------- 报表


@dataclass
class FinancialStatement:
    """一期财务报表（资产负债表 + 利润表 + 现金流量表的关键科目）。

    单位统一为**万元**。所有可比性处理（口径调整、追溯重述）由上游完成，
    本模型只负责结构化承载与比率计算。

    period 形如 "2025A" / "2024A"，用于报告标注。
    """

    period: str
    items: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # --- 取值 ---

    def get(self, key: str, default: float | None = None) -> float | None:
        v = self.items.get(key)
        return default if v is None else v

    def require(self, key: str) -> float:
        v = self.items.get(key)
        if v is None:
            raise KeyError(f"报表 {self.period} 缺少必需科目：{key}")
        return v

    def has(self, key: str) -> bool:
        return self.items.get(key) is not None

    # --- 同比 / 变动 ---

    def yoy(self, key: str, prev: "FinancialStatement") -> float | None:
        """同比增速（%）。基期为 0 或缺失时返回 None——不做无意义的除法。"""
        cur_v = self.get(key)
        prev_v = prev.get(key)
        if cur_v is None or prev_v is None or prev_v == 0:
            return None
        return (cur_v - prev_v) / abs(prev_v) * 100.0

    def delta_pp(self, key: str, prev: "FinancialStatement") -> float | None:
        """变动（百分点）。用于比率类科目。"""
        cur_v = self.get(key)
        prev_v = prev.get(key)
        if cur_v is None or prev_v is None:
            return None
        return cur_v - prev_v

    # --- 比率 ---

    def gross_margin(self) -> float | None:
        """毛利率（%）。"""
        rev = self.get("revenue")
        cost = self.get("cost_of_sales")
        if rev is None or cost is None or rev == 0:
            return None
        return (rev - cost) / rev * 100.0

    def net_margin(self) -> float | None:
        """净利率（%）。"""
        rev = self.get("revenue")
        np_ = self.get("net_profit")
        if rev is None or np_ is None or rev == 0:
            return None
        return np_ / rev * 100.0

    def debt_ratio(self) -> float | None:
        """资产负债率（%）。"""
        tl = self.get("total_liabilities")
        ta = self.get("total_assets")
        if tl is None or ta is None or ta == 0:
            return None
        return tl / ta * 100.0

    def current_ratio(self) -> float | None:
        """流动比率。"""
        ca = self.get("current_assets")
        cl = self.get("current_liabilities")
        if ca is None or cl is None or cl == 0:
            return None
        return ca / cl

    def ocf_to_profit(self) -> float | None:
        """经营现金流 / 净利润。净利润为负时不计算（比值为负无解释力）。"""
        ocf = self.get("operating_cash_flow")
        np_ = self.get("net_profit")
        if ocf is None or np_ is None or np_ <= 0:
            return None
        return ocf / np_

    def roe(self) -> float | None:
        """净资产收益率（%）。"""
        np_ = self.get("net_profit")
        ne = self.get("net_assets")
        if np_ is None or ne is None or ne == 0:
            return None
        return np_ / ne * 100.0

    def ratios(self) -> dict[str, float | None]:
        """一次取全部派生指标，便于报告与对比。"""
        return {
            "gross_margin": self.gross_margin(),
            "net_margin": self.net_margin(),
            "debt_ratio": self.debt_ratio(),
            "current_ratio": self.current_ratio(),
            "ocf_to_profit": self.ocf_to_profit(),
            "roe": self.roe(),
        }

    # --- 序列化 ---

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FinancialStatement":
        return cls(
            period=d["period"],
            items={k: float(v) for k, v in d.get("items", {}).items() if v is not None},
            meta=d.get("meta", {}),
        )


# ---------------------------------------------------------------- 发现


@dataclass
class Finding:
    """一条异常发现。必须带依据，否则在财务场景里没有说服力。"""

    rule_id: str
    level: Level
    title: str
    detail: str
    items: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    basis: str = ""              # 判定依据（会计逻辑 / 准则口径）
    confidence: float = 1.0      # 规则置信度，供 LLM 加权参考

    def to_dict(self) -> dict:
        d = asdict(self)
        d["level"] = int(self.level)
        d["level_label"] = self.level.label
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Finding":
        d = dict(d)
        d["level"] = Level(int(d.get("level", 0)))
        d.pop("level_label", None)
        return cls(**d)


# ---------------------------------------------------------------- 结论文档


@dataclass
class AgentTrace:
    """Agent 的执行轨迹。作品集里这一项很关键——它证明流程是可控、可复现的。

    duration_ms 刻意不进入 to_dict()：它是性能观测值，每次运行都不同。
    把它塞进结构化输出会让"同一输入必须得到同一输出"这条可复算承诺失效
    （下游做 diff 时会被毫秒数噪声淹没）。控制台渲染仍会显示它。
    """

    agent: str
    action: str
    detail: str = ""
    duration_ms: float = 0.0
    ok: bool = True

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "action": self.action,
            "detail": self.detail,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class ScorePolicy:
    """风险分的全部可调参数。

    抽出来是为了两件事：

    1. **敏感性分析有真实代码路径可走**。参数如果硬编码在方法里，
       做敏感性就只能另写一份公式——那份副本会悄悄和主线逻辑漂移，
       跑出来的"稳健区间"是假的。
    2. **让"参数该定多少"成为一个可单独讨论的对象**。与 RuleBook
       阈值集中管理是同一个思路：改参数不改逻辑。

    `DEFAULT_POLICY` 就是长期以来的线上行为，任何人不传参数时结果不变。
    """

    # 饱和函数的尺度。越大则同样 raw 分越低、曲线越平。
    scale: float = 20.0

    # 严重度权重
    w_high: float = 10.0
    w_warn: float = 4.0
    w_info: float = 1.0

    # 分数 → 档位 的切点
    grade_very_high: float = 85.0
    grade_high: float = 65.0
    grade_mid: float = 35.0

    # 重大异常数量 → 档位 的门槛
    sev_very_high: int = 4     # 重大达到几项，档位不得低于「极高」
    sev_high: int = 1          # 重大达到几项，档位不得低于「高」

    def to_dict(self) -> dict:
        return {
            "scale": self.scale,
            "weights": {"high": self.w_high, "warn": self.w_warn, "info": self.w_info},
            "grade_cutoffs": {
                "very_high": self.grade_very_high,
                "high": self.grade_high,
                "mid": self.grade_mid,
            },
            "severity_counts": {"very_high": self.sev_very_high, "high": self.sev_high},
        }


DEFAULT_POLICY = ScorePolicy()


@dataclass
class AnalysisResult:
    """一次完整分析的产出。"""

    company: str
    period: str
    findings: list[Finding] = field(default_factory=list)
    summary: str = ""
    reasoning: str = ""
    trace: list[AgentTrace] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.level == Level.HIGH)

    @property
    def warn_count(self) -> int:
        return sum(1 for f in self.findings if f.level == Level.WARN)

    def risk_score(self, policy: ScorePolicy | None = None) -> float:
        """0~100 的风险分。

        用饱和函数而非线性累加：线性累加在重大异常多时会直接顶到 100 被封顶，
        导致"4 项重大"和"8 项重大"拿到同样的分数，失去区分度。
        取 1-exp(-x/scale) 形式，单调递增但永不触顶，天然保留了边际区分能力。

        scale 取 20 是经验值：按现在权重，约 3 项重大 + 2 项关注能到 70 分区间。
        这个参数应该随历史样本回测校准——在没有样本之前，任何精确的加权都是伪精确。
        校准方法见 scripts/sensitivity.py。
        """
        p = policy or DEFAULT_POLICY
        if not self.findings:
            return 0.0
        w = {Level.HIGH: p.w_high, Level.WARN: p.w_warn, Level.INFO: p.w_info}
        raw = sum(w[f.level] * f.confidence for f in self.findings)
        return (1.0 - _math.exp(-raw / p.scale)) * 100.0

    def risk_grade(self, policy: ScorePolicy | None = None) -> str:
        """把 0~100 的风险分映射到可执行档位。

        取两个维度的较高者，缺一不可：

        1. **分数档**：按饱和函数得分切分，反映异常的总量与集中度。
        2. **严重度档**：按重大异常的数量切分。**只要存在 1 项重大，
           档位不得低于「高」**——重大异常的定义就是"需要专项核查"，
           让使用者去"常规复核"是逻辑矛盾。

        早先版本只按分数切档，出现过"2 项重大异常却建议就关注项问询"
        的自相矛盾输出（重大项少时饱和函数得分偏低）。这是设计缺陷，
        不是阈值问题——**建议动作必须与最严重发现的等级自洽。**
        """
        p = policy or DEFAULT_POLICY
        if not self.findings:
            return "低"

        order = ["低", "中", "高", "极高"]

        # 维度一：分数档
        s = self.risk_score(p)
        if s >= p.grade_very_high:
            by_score = "极高"
        elif s >= p.grade_high:
            by_score = "高"
        elif s >= p.grade_mid:
            by_score = "中"
        else:
            by_score = "低"

        # 维度二：严重度档（重大异常数量）
        n_high = self.high_count
        if n_high >= p.sev_very_high:
            by_severity = "极高"
        elif n_high >= p.sev_high:
            by_severity = "高"
        else:
            by_severity = "中"

        return order[max(order.index(by_score), order.index(by_severity))]

    def risk_action(self) -> str:
        """与档位配套的建议动作，直接写进报告结论区。"""
        g = self.risk_grade()
        return {
            "极高": "建议暂缓出具结论，先对全部重大异常逐项核查原始凭证",
            "高": "建议对重大异常项做专项函证或走访核实后再定稿",
            "中": "建议就关注项向管理层提出书面问询",
            "低": "未见系统性异常，可常规复核",
        }[g]

    def to_dict(self) -> dict:
        return {
            "company": self.company,
            "period": self.period,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "reasoning": self.reasoning,
            "trace": [t.to_dict() for t in self.trace],
            "metrics": self.metrics,
            "risk_score": round(self.risk_score(), 2),
            "risk_grade": self.risk_grade(),
            "risk_action": self.risk_action(),
        }


# ---------------------------------------------------------------- 加载


def load_statement(path: str | Path) -> FinancialStatement:
    return FinancialStatement.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_pair(
    cur_path: str | Path, prev_path: str | Path
) -> tuple[FinancialStatement, FinancialStatement]:
    """加载当期与上期。顺序容易搞反，这里统一约定：先当期后上期。"""
    return load_statement(cur_path), load_statement(prev_path)


# ---------------------------------------------------------------- 多期序列


def period_sort_key(period: str) -> tuple:
    """把 "2025A" / "2025Q1" 之类的期间标签转成可排序的键。

    不能直接按字符串排："2025Q1" vs "2024A" 比字符串会得到错误顺序
    （'Q' > 'A'）。做法是抽出其中的数字按数值比，非数字部分做次级排序。
    抽不出数字时退化为字符串比较，保证不抛异常。
    """
    import re as _re

    nums = tuple(int(n) for n in _re.findall(r"\d+", period))
    return (nums, period)


@dataclass
class StatementSeries:
    """按时间顺序排列的多期报表序列。

    存在的理由：**单期对比看不出趋势，趋势才是结构性问题的证据。**
    "今年应收增速突然超营收 40pct" 可能是一次性渠道压货；
    "连续三年都超 40pct" 几乎一定是收入确认政策出了问题。
    前者要问，后者要查。

    约定：内部始终按时间正序存放（最早在前，最新在后），
    调用方传进来的顺序不影响结果。
    """

    statements: list[FinancialStatement] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.statements = sorted(self.statements, key=lambda s: period_sort_key(s.period))

    # --- 基本属性 ---

    @property
    def latest(self) -> FinancialStatement | None:
        return self.statements[-1] if self.statements else None

    @property
    def previous(self) -> FinancialStatement | None:
        return self.statements[-2] if len(self.statements) >= 2 else None

    @property
    def periods(self) -> list[str]:
        return [s.period for s in self.statements]

    def __len__(self) -> int:
        return len(self.statements)

    def __iter__(self):
        return iter(self.statements)

    def __getitem__(self, i: int) -> FinancialStatement:
        return self.statements[i]

    # --- 派生结构 ---

    def pairs(self) -> list[tuple[FinancialStatement, FinancialStatement]]:
        """全部相邻期配对，顺序为 (上期, 当期)。"""
        return list(zip(self.statements, self.statements[1:]))

    def trailing_pairs(self, n: int) -> list[tuple[FinancialStatement, FinancialStatement]]:
        """最近 n 个相邻期配对。期数不足时返回全部，由调用方判断是否够用。"""
        return self.pairs()[-n:] if n > 0 else []

    def has_trend(self, min_periods: int) -> bool:
        """序列是否足以做趋势判断：至少要有 min_periods 期报表。

        2 期只能做同比，谈不上趋势。趋势判断的最低要求是 3 期。
        """
        return len(self.statements) >= min_periods

    def to_dict(self) -> dict:
        return {
            "periods": self.periods,
            "statements": [s.to_dict() for s in self.statements],
        }


def build_series(paths: Sequence[str | Path]) -> StatementSeries:
    """从多个报表路径构造序列。重复期间后者覆盖前者。"""
    seen: dict[str, FinancialStatement] = {}
    for p in paths:
        st = load_statement(p)
        seen[st.period] = st
    return StatementSeries(statements=list(seen.values()))
