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
    """Agent 的执行轨迹。作品集里这一项很关键——它证明流程是可控、可复现的。"""

    agent: str
    action: str
    detail: str = ""
    duration_ms: float = 0.0
    ok: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


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

    def risk_score(self) -> float:
        """0~100 的风险分。

        用饱和函数而非线性累加：线性累加在重大异常多时会直接顶到 100 被封顶，
        导致"4 项重大"和"8 项重大"拿到同样的分数，失去区分度。
        取 1-exp(-x/scale) 形式，单调递增但永不触顶，天然保留了边际区分能力。

        scale 取 20 是经验值：按现在权重，约 3 项重大 + 2 项关注能到 70 分区间。
        这个参数应该随历史样本回测校准——在没有样本之前，任何精确的加权都是伪精确。
        """
        if not self.findings:
            return 0.0
        w = {Level.HIGH: 10.0, Level.WARN: 4.0, Level.INFO: 1.0}
        raw = sum(w[f.level] * f.confidence for f in self.findings)
        scale = 20.0
        return (1.0 - _math.exp(-raw / scale)) * 100.0

    def risk_grade(self) -> str:
        """把 0~100 的风险分映射到可执行档位。

        分数本身只是标量，读者需要的是"该做什么"。档位设计原则：
        宁可把边界定得保守，也不要给出模棱两可的中间态。
        """
        s = self.risk_score()
        if not self.findings:
            return "低"
        if s >= 85:
            return "极高"
        if s >= 65:
            return "高"
        if s >= 35:
            return "中"
        return "低"

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
