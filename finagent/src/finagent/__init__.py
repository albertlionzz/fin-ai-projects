"""finagent · 财务报表差异分析 Agent（项目二）

一组分工协作的 Agent，把两份财务报表的差异，转化为可追溯、可辩护的异常清单。

模块：
    models.py      报表 / 发现 / 结论的数据模型与财务比率
    rules.py       规则引擎：把会计常识写成可执行、可测的判定
    benchmarks.py  行业基准：把统一观察线换成行业内相对位置的参照系
    agents.py      三个分工 Agent（提取 / 校验 / 生成）与编排层
    render.py      输出渲染（Markdown / 控制台）
    workpaper.py   审计工作底稿导出（Markdown / 可选 xlsx）

设计立场：
    确定性优先。规则引擎负责判定，LLM 只负责解释。
    财务场景里，"结论可复算、可追责"比"看起来聪明"重要得多。

作者：转型学习项目 · 项目二
"""

from .agents import (
    Agent,
    AuditAgent,
    ExtractAgent,
    LLM,
    NullLLM,
    OpenAICompatLLM,
    Orchestrator,
    ReportAgent,
    build_llm,
)
from .benchmarks import (
    BenchmarkBook,
    IndustryBenchmark,
    load_benchmarks,
)
from .models import (
    AgentTrace,
    AnalysisResult,
    FinancialStatement,
    Finding,
    Level,
    StatementSeries,
    build_series,
    load_pair,
    load_statement,
    period_sort_key,
)
from .render import render_console, render_markdown
from .workpaper import WorkingPaper, build_working_paper, render_workpaper_markdown

__all__ = [
    "Agent",
    "AgentTrace",
    "AnalysisResult",
    "AuditAgent",
    "BenchmarkBook",
    "ExtractAgent",
    "FinancialStatement",
    "Finding",
    "IndustryBenchmark",
    "LLM",
    "Level",
    "NullLLM",
    "OpenAICompatLLM",
    "Orchestrator",
    "ReportAgent",
    "StatementSeries",
    "WorkingPaper",
    "build_llm",
    "build_series",
    "build_working_paper",
    "load_benchmarks",
    "load_pair",
    "load_statement",
    "period_sort_key",
    "render_console",
    "render_markdown",
    "render_workpaper_markdown",
]

__version__ = "0.2.0"
