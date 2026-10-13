"""行业基准：把"全市场统一观察线"换成"行业内相对位置"的参照系。

为什么需要这一层
--------------
`AR_GAP` 的 20pct 观察线对全市场一视同仁，但行业之间的差异是结构性的：
快速消费品靠渠道铺货，应收增速快于营收几乎是常态；工程机械销售回款慢，
但增速差通常不大。用同一条线衡量所有行业，要么在快消误报连篇，
要么在重工漏掉真问题。

设计取舍（三条，都是踩过坑才定下来的）
----------------------------------
1. **行业数据缺失时行为绝对不变**。没有基准簿、行业不在表里、或样本量不足，
   一律回退绝对观察线。这是向后兼容的底线——`BenchmarkBook` 默认 `None`，
   老调用方零改动，测试 `test_no_benchmarks_matches_absolute` 钉住。

2. **校准后的观察线必须披露来源**。Finding 的 evidence 里记录 `line_source`，
   detail 里注明是行业分位校准。财务工具绝不能悄悄换阈值——
   读者看到"观察线 12pct"时必须能回答"这个 12 是哪来的"。

3. **样本量不足的基准不可信**。`n_peers < min_peers` 时视为无数据。
   3 个同行算出来的分位数是噪声，不是参照系。默认门槛 5。

与 RuleBook 的关系
----------------
RuleBook 是**判定策略**（阈值该定多少，是人拍的）；BenchmarkBook 是**参照数据**
（同行实际分布，是会随样本更新的）。两者都不要求改规则逻辑，
但生命周期不同，所以分开存放：规则簿挂在 `RuleBook`，基准簿挂在 `RuleEngine`。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# 规则使用的分位数档位。p75 = "比四分之三的同行更极端"，
# 对应"需要解释"的关注级；p95 留给调用方自行取用。
DEFAULT_QUANTILE = "p75"

# 基准簿最低同行样本量。低于此值的分位数视为噪声。
DEFAULT_MIN_PEERS = 5


@dataclass
class IndustryBenchmark:
    """某个行业在某一个决策指标上的同行分布。

    percentiles 的键形如 "p50" / "p75" / "p90" / "p95"，
    值是同行样本在该指标上的分位数（单位与指标一致）。
    """

    industry: str
    metric: str
    n_peers: int
    percentiles: dict[str, float] = field(default_factory=dict)
    source: str = ""
    as_of: str = ""

    def at(self, q: str) -> float | None:
        """取指定分位数的值。档位不存在时返回 None，由调用方决定回退。"""
        v = self.percentiles.get(q)
        return None if v is None else float(v)

    @property
    def available_quantiles(self) -> list[str]:
        return sorted(self.percentiles)

    def to_dict(self) -> dict:
        return {
            "industry": self.industry,
            "metric": self.metric,
            "n_peers": self.n_peers,
            "percentiles": dict(self.percentiles),
            "source": self.source,
            "as_of": self.as_of,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IndustryBenchmark":
        return cls(
            industry=str(d["industry"]),
            metric=str(d["metric"]),
            n_peers=int(d.get("n_peers", 0)),
            percentiles={k: float(v) for k, v in (d.get("percentiles") or {}).items()},
            source=str(d.get("source", "")),
            as_of=str(d.get("as_of", "")),
        )


@dataclass
class BenchmarkBook:
    """行业基准簿：(行业, 指标) → 同行分布。

    不可变语义：构造后不再修改。加载入口见 `load_benchmarks()`。
    """

    entries: dict[tuple[str, str], IndustryBenchmark] = field(default_factory=dict)
    min_peers: int = DEFAULT_MIN_PEERS
    source: str = ""
    as_of: str = ""

    # --- 查询 ---

    def get(self, industry: str | None, metric: str) -> IndustryBenchmark | None:
        """查基准。行业缺失、条目缺失、或样本量不足，统一返回 None。

        样本量门槛在这一层统一执行，规则里不需要各自判断——
        "3 个同行的分位数不能用"这条规则只应有一个实现。
        """
        if not industry:
            return None
        b = self.entries.get((industry, metric))
        if b is None or b.n_peers < self.min_peers:
            return None
        return b

    def line(
        self,
        industry: str | None,
        metric: str,
        fallback: float,
        q: str = DEFAULT_QUANTILE,
    ) -> tuple[float, str]:
        """取校准后的观察线，并同时返回来源说明。

        返回 (观察线, 来源)。来源形如 "absolute" 或 "industry:p75"，
        调用方必须把它写进 Finding 的 evidence——这是披露义务，不是可选项。
        """
        b = self.get(industry, metric)
        if b is None:
            return float(fallback), "absolute"
        v = b.at(q)
        if v is None:
            return float(fallback), "absolute"
        return v, f"industry:{q}"

    def industries(self) -> list[str]:
        return sorted({ind for ind, _ in self.entries})

    def metrics(self) -> list[str]:
        return sorted({m for _, m in self.entries})

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)

    # --- 序列化 ---

    def to_dict(self) -> dict:
        return {
            "schema": "finagent-industry-benchmarks/1",
            "source": self.source,
            "as_of": self.as_of,
            "min_peers": self.min_peers,
            "industries": {
                industry: {
                    "n_peers": self._n_peers_of(industry),
                    "metrics": {
                        metric: b.percentiles
                        for (ind, metric), b in sorted(self.entries.items())
                        if ind == industry
                    },
                }
                for industry in self.industries()
            },
        }

    def _n_peers_of(self, industry: str) -> int:
        for (ind, _), b in self.entries.items():
            if ind == industry:
                return b.n_peers
        return 0


def load_benchmarks(path: str | Path) -> BenchmarkBook:
    """从 JSON 加载行业基准簿。

    期望格式：
        {
          "schema": "finagent-industry-benchmarks/1",
          "source": "...", "as_of": "2025-12-31", "min_peers": 5,
          "industries": {
            "工程机械": {
              "n_peers": 42,
              "metrics": {
                "ar_growth_gap": {"p50": 8.0, "p75": 12.0, "p90": 20.0}
              }
            }
          }
        }

    解析策略：**容忍部分损坏**。单个行业的单个指标解析失败只跳过该指标，
    不让整个基准簿加载失败——参照数据是增强项，不该成为单点故障。
    """
    d: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    book = BenchmarkBook(
        min_peers=int(d.get("min_peers", DEFAULT_MIN_PEERS)),
        source=str(d.get("source", "")),
        as_of=str(d.get("as_of", "")),
    )
    industries = d.get("industries") or {}
    if not isinstance(industries, dict):
        return book

    for industry, body in industries.items():
        if not isinstance(body, dict):
            continue
        n_peers = int(body.get("n_peers", 0) or 0)
        metrics = body.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue
        for metric, dist in metrics.items():
            if not isinstance(dist, dict):
                continue
            try:
                pct = {str(k): float(v) for k, v in dist.items()}
            except (TypeError, ValueError):
                continue    # 跳过这一个坏指标，其余照常加载
            book.entries[(str(industry), str(metric))] = IndustryBenchmark(
                industry=str(industry),
                metric=str(metric),
                n_peers=n_peers,
                percentiles=pct,
                source=book.source,
                as_of=book.as_of,
            )
    return book
