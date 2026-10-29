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
| 核心产物 | 带回溯依据的异常清单 + 风险分与等级 + 审计底稿 |
| 评测方式 | 规则召回率 / 误报率 / 可复算性 / 研判质量四维指标 |
| 实测 | 植入 7 类异常全命中，178 个测试通过；4 期趋势模式可识别单期阈值法漏掉的慢性恶化；行业校准可双向调整灵敏度；35 组参数里 24 组不翻档；行业基准已从 auditdog.cn 真实抓取（食品饮料 111 家、电子 485 家、计算机 315 家，均排除 ST，共 911 家），44 家公司判定因校准改变；重大线从"拍倍数"改为行业 p90 后 22 处判定翻转（11:11 对称，证明比例线与分布形状无关） |

**核心立场：确定性优先，LLM 只在最后一环介入。**

判定环节刻意不接 LLM——判异常靠规则，写结论才用模型。
这不是保守，是把不可靠组件放在伤害最小的位置。

```
finagent/
├── src/finagent/
│   ├── models.py      财务指标 / 多期序列 / 风险分（饱和函数）/ 等级映射 / ScorePolicy
│   ├── rules.py       RuleBook 集中阈值 + 10 条规则 + 持续性升级 + 行业区间校准（p75 关注 / p90 重大）+ RuleEngine
│   ├── benchmarks.py  行业基准簿（分位数查询 + 同行样本量门槛）
│   ├── workpaper.py   审计底稿（Markdown 永远可用；xlsx 需 openpyxl）
│   ├── abcompare.py   研判质量四维指标（保真 / 覆盖 / 复述 / 可复算）
│   ├── agents.py      Extract / Audit / Report 三 Agent + Orchestrator
│   └── render.py      终端摘要 + Markdown 报告（多期横向铺开）
├── scripts/
│   ├── run.py         CLI：analyze / workpaper / extract / rules / batch
│   ├── sensitivity.py 风险分参数敏感性分析（scale × 权重比网格）
│   ├── fetch_benchmarks.py      从 auditdog.cn 抓三大报表，构建真实行业基准簿
│   ├── build_fixture_from_raw.py 把爬虫缓存还原成 finagent statement JSON
│   └── ab_test.py     研判质量 A/B（--stub 框架自检 / --api 真实模型）
└── tests/test_analysis.py   178 个单元测试
```

四项能力递进，每一项都解决上一项暴露的问题：

| 能力 | 解决什么 |
|---|---|
| 多期趋势 | 单期阈值法看不见"每期小幅走坏、期期够不上观察线"的慢性恶化 |
| 行业基准 | 全市场统一线对快消太松、对建筑施工太紧，必然同时误报与漏报 |
| 参数敏感性 | 手工权重不知道在多大扰动下结论还成立；只盯分数变化会得到"处处敏感"的假结论 |
| A/B 框架 | 接入 LLM 前先定义"什么叫更好"，否则无法判断模型值不值得用 |

## 共享的工程约定

这两个项目不是各写各的，它们共用同一套约定——这是刻意的：

1. **纯标准库，零必需依赖**。`git clone` 就能跑，不需要环境配置。降低审阅成本。
   finagent 的 xlsx 导出是唯一用到第三方包（openpyxl）的路径，且是**可选**：
   缺依赖时明确报错并指向 Markdown 备选，不静默降级。
2. **dataclass 驱动**。数据结构先定义清楚，逻辑围绕数据展开。
3. **可插拔 Protocol**。`finrag` 的 `Embedder` 和 `finagent` 的 `LLM` 是同一个设计思路：
   核心逻辑不依赖具体外部服务，接口稳定，实现可换。
4. **自带标注集**。没有评测集，"我改进了"就无法量化。两个项目都有 gold 标注。
5. **诚实清单**。README 里明确写已知局限，包括"哪些参数是经验值而非调优结果"，
   以及"哪些场景其实没测到"。

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

# 行业校准（同一份数据，全市场口径零异常、工程机械口径两项关注）
python scripts/run.py analyze \
    --cur  data/statements/恒润重工_2025A.json \
    --prev data/statements/恒润重工_2024A.json \
    --benchmarks data/benchmarks/industry_benchmarks.json \
    --md   artifacts/report_bench.md

# 审计底稿（Markdown 无需第三方包；xlsx 需 pip install openpyxl）
python scripts/run.py workpaper --ref A-3 \
    --cur  data/statements/示例制造_2025A.json \
    --prev data/statements/示例制造_2024A.json \
    --md   artifacts/wp.md

# 参数敏感性分析
python scripts/sensitivity.py --md artifacts/sensitivity.md

python -m unittest discover -s tests
```

> finagent 的 xlsx 导出是唯一需要第三方包的路径（openpyxl）。
> 缺依赖时它会**明确报错并说明改用 Markdown**，不会静默降级——
> 要的是表格、拿到的是文本，比直接报错更危险。

## 各项目的完整说明

每个项目目录下都有独立 README，包含架构图、实测指标、关键设计决策，
以及一节「**面试会问的三个问题**」——把容易引起质疑的地方提前讲清楚。

---

**这两个项目的价值不在代码，在于它们证明了一件事**：

懂业务的人做 AI 应用，第一反应不是"用大模型做 XX"，
而是"**哪些环节必须确定性，模型只能在哪儿介入**"。

规则引擎判异常、LLM 写结论，这个分工不是为了保守，
是因为**一份不能被复算的财务分析报告，在专业场景里等于没有价值。**
