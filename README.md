# 财务场景 AI 应用 · 两个可运行内核

> 两个共享同一套工程约定的项目：**一个负责把制度文档查准，一个负责把财务数字看穿。**
> 都是纯标准库、零第三方依赖，`git clone` 后直接能跑出数字。

## 为什么是这两个项目

财务/审计场景对 AI 的要求和通用场景不一样：

- **答案要能追责** —— 说"住宿费能报 800"，得指得出是哪一条
- **结论要能复算** —— 同一个人隔一周、换一个人，看同一份报表必须得到同一个结论
- **数字不能幻觉** —— 金额错一位就是事故

这两个项目分别针对"查得准"和"判得稳"做了工程化设计。

## 项目一 · finrag

**财务制度 RAG 检索系统**

| | |
|---|---|
| 解决的问题 | 从制度文档里**查到条款**并给出可追溯的引用 |
| 核心产物 | 带条款溯源的答案 + 可量化的评测指标 |
| 评测方式 | recall@k / MRR / 引用正确率 / 数字幻觉率 |
| 实测 | BM25 recall@1 **0.960**，数字幻觉率 **0.000** |

针对财务场景做了专门设计：条款号溯源、数字幻觉检测、引用精确率。
这些不是通用 RAG 教程里有的东西。

```
finrag/
├── src/finrag/retrieval.py    结构感知分块 + BM25 + 向量通道 + RRF 融合 + 幻觉检测
├── src/finrag/api_embedder.py 可插拔真实 embedding（API / 本地 / 哈希三层降级）
├── src/finrag/evaluate.py     recall@k / MRR / 引用正确率 / 幻觉率
├── scripts/run.py             CLI：build / ask / eval / compare / ablate
├── tests/test_retrieval.py    34 个单元测试
├── docs/                      财务制度语料（2 篇，28 个条款块）
└── data/eval_set.json         25 条人工标注用例
```

## 项目二 · finagent

**财务报表差异分析 Agent**

| | |
|---|---|
| 解决的问题 | 从两期或多期财务数字里**发现异常**并说明依据 |
| 核心产物 | 带回溯依据的异常清单 + 风险分与等级 |
| 评测方式 | 规则召回率 / 误报率 / 可复算性 |
| 实测 | 植入 7 类异常全命中，88 个测试通过；4 期趋势模式可识别单期阈值法漏掉的慢性恶化 |

**核心立场：确定性优先，LLM 只在最后一环介入。**

判定环节刻意不接 LLM——判异常靠规则，写结论才用模型。
这不是保守，是把不可靠组件放在伤害最小的位置。

```
finagent/
├── src/finagent/models.py   财务指标 / 多期序列 / 风险分（饱和函数）/ 等级映射
├── src/finagent/rules.py    RuleBook 集中阈值 + 10 条规则 + 持续性升级 + RuleEngine
├── src/finagent/agents.py   Extract / Audit / Report 三 Agent + Orchestrator
├── src/finagent/render.py   终端摘要 + Markdown 报告（多期横向铺开）
├── scripts/run.py           CLI：analyze / extract / rules / batch
└── tests/test_analysis.py   88 个单元测试
```

## 共享的工程约定

这两个项目不是各写各的，它们共用同一套约定——这是刻意的：

1. **纯标准库，零依赖**。`git clone` 就能跑，不需要环境配置。降低审阅成本。
2. **dataclass 驱动**。数据结构先定义清楚，逻辑围绕数据展开。
3. **可插拔 Protocol**。`finrag` 的 `Embedder` 和 `finagent` 的 `LLM` 是同一个设计思路：
   核心逻辑不依赖具体外部服务，接口稳定，实现可换。
4. **自带标注集**。没有评测集，"我改进了"就无法量化。两个项目都有 gold 标注。
5. **诚实清单**。README 里明确写已知局限，包括"哪些参数是经验值而非调优结果"。

## 快速开始

```bash
# 环境：Python 3.13+，无需 pip install 任何东西

# --- 项目一 ---
cd finrag
export PYTHONPATH=src
python scripts/run.py build
python scripts/run.py ask "出差住宿费在一线城市能报多少"
python scripts/run.py eval --mode bm25
python -m unittest discover -s tests

# --- 项目二 ---
cd ../finagent
export PYTHONPATH=src

# 两期模式（7 类植入异常全命中）
python scripts/run.py analyze \
    --cur  data/statements/示例制造_2025A.json \
    --prev data/statements/示例制造_2024A.json \
    --md   artifacts/report.md

# 多期模式（≥3 期自动启用趋势规则与持续性升级）
python scripts/run.py analyze \
    --periods data/statements/长恒电子_2022A.json \
              data/statements/长恒电子_2023A.json \
              data/statements/长恒电子_2024A.json \
              data/statements/长恒电子_2025A.json \
    --md   artifacts/report_trend.md

python -m unittest discover -s tests
```

## 各项目的完整说明

每个项目目录下都有独立 README，包含架构图、实测指标、关键设计决策，
以及一节「**面试会问的三个问题**」——把容易引起质疑的地方提前讲清楚。

---

**这两个项目的价值不在代码，在于它们证明了一件事**：

懂业务的人做 AI 应用，第一反应不是"用大模型做 XX"，
而是"**哪些环节必须确定性，模型只能在哪儿介入**"。

规则引擎判异常、LLM 写结论，这个分工不是为了保守，
是因为**一份不能被复算的财务分析报告，在专业场景里等于没有价值。**
