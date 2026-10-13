# finagent · 财务报表差异分析 Agent

> 三 Agent 分工的财务异常识别内核。
> **确定性优先，LLM 只在最后一环介入**——判异常靠规则，写结论才用模型。
> 零第三方依赖，`git clone` 后直接能跑出命中清单与风险分。

## 这是什么

不是"让 AI 读财报"的 demo，是一个**规则可复算、结论可辩护**的分析流水线。
它回答的问题是：

> 「两期财务报表摆在一起，哪些差异是真的异常，依据是什么，先查哪个？」

与项目一的关系：

| | finrag（项目一） | finagent（项目二） |
|---|---|---|
| 解决的问题 | 从制度文档里**查到条款** | 从财务数字里**发现异常** |
| 核心产物 | 带引用的答案 | 带依据的异常清单 |
| 评测方式 | recall@k / MRR / 幻觉率 | 规则召回率 / 误报率 |
| 共享约定 | 纯标准库、dataclass 驱动、可插拔 Protocol、可独立单测 | 同上 |

两个项目共用同一套工程约定。项目一的 `Embedder` Protocol 和项目二的 `LLM` Protocol
是同一个设计思路：**核心逻辑不依赖具体外部服务，接口稳定，实现可换。**

## 快速开始

```bash
# 环境：Python 3.13+，无需 pip install 任何东西
export PYTHONPATH=src        # Windows: set PYTHONPATH=src

# 1. 分析两期报表（示例数据已刻意植入 7 类异常）
python scripts/run.py analyze \
    --cur  data/statements/示例制造_2025A.json \
    --prev data/statements/示例制造_2024A.json \
    --md   artifacts/report_2025A.md \
    --json-out artifacts/report_2025A.json

# 2. 分析多期报表（≥3 期自动启用趋势规则与持续性升级）
python scripts/run.py analyze \
    --periods data/statements/长恒电子_2022A.json \
              data/statements/长恒电子_2023A.json \
              data/statements/长恒电子_2024A.json \
              data/statements/长恒电子_2025A.json \
    --md   artifacts/report_长恒电子_2025A.md

# 3. 行业校准观察线（AR_GAP / INV_GAP / GM_SWING / DEBT_RATIO 改按同行业 p75 分位）
#    报表 meta.industry 缺失或行业不在表里时自动回退绝对线，来源记入 evidence.line_source
python scripts/run.py analyze \
    --cur  data/statements/恒润重工_2025A.json \
    --prev data/statements/恒润重工_2024A.json \
    --benchmarks data/benchmarks/industry_benchmarks.json \
    --md   artifacts/report_恒润重工_2025A.md

# 4. 导出审计工作底稿（Markdown 永远可用；xlsx 需 openpyxl）
python scripts/run.py workpaper --ref A-3 \
    --cur  data/statements/示例制造_2025A.json \
    --prev data/statements/示例制造_2024A.json \
    --md   artifacts/wp_A-3.md \
    --xlsx artifacts/wp_A-3.xlsx

# 5. 风险分参数敏感性分析（scale × 权重比网格，衡量档位是否翻转）
python scripts/sensitivity.py --md artifacts/sensitivity.md

# 6. 研判质量 A/B 对比（--stub 为框架自检；--api 走真实 OpenAI 兼容端点）
python scripts/ab_test.py --stub good --md artifacts/ab_test.md
python scripts/ab_test.py --api --model qwen-plus --md artifacts/ab_test_api.md

# 7. 查看已注册的规则与阈值
python scripts/run.py rules

# 8. 从文本报表抽取结构化科目（确定性解析，无 LLM 也可用）
python scripts/run.py extract --text data/raw/示例公司2025年报摘录.txt --period 2025A

# 9. 批量分析一个目录（按 <主体>_<YYYYA>.json 自动配对相邻两期）
python scripts/run.py batch --dir data/statements --md-dir artifacts/batch

# 10. 单元测试
python -m unittest discover -s tests
```

## 架构

```
两期或多期报表 JSON
   │
   ├─ ExtractAgent          科目抽取（确定性正则；科目<8 时才启用 LLM 兜底）
   │      └─ 未识别行记入 meta.unrecognized，不静默丢弃
   │
   ├─ AuditAgent            异常判定（★ 刻意不接 LLM）
   │      └─ RuleEngine.run()   10 条规则 → Finding[]
   │             ├─ 单条规则失败不中断整轮，记 ERR_<name> 后继续
   │             ├─ 传入 benchmarks 时按行业分位校准 4 条规则的观察线
   │             └─ 传入 series 时额外执行：
   │                  ├─ 序列规则（QUALITY_DRIFT 等多期同向恶化）
   │                  └─ 持续性升级（跨期反复命中的发现提升一级严重度）
   │
   ├─ ReportAgent           结论生成（★ 唯一引入 LLM 的环节）
   │      ├─ summarize()   确定性摘要
   │      └─ reason()      有 LLM 走生成；无 LLM 走确定性聚合模板
   │
   └─ Orchestrator          串成链路，任何一环失败不中断，trace 如实呈现
          │
      render.py              终端摘要 / Markdown 报告
      workpaper.py           审计底稿（Markdown + xlsx，五张表）
      benchmarks.py          行业基准簿（分位数查询 + 样本量门槛）
      abcompare.py           研判质量四维指标（保真 / 覆盖 / 复述 / 可复算）
```

### 为什么判定环节刻意不用 LLM

财务报表分析的第一要求不是"说得好听"，是**同一个输入必须得到同一个结论**。
LLM 做不到这一点。所以分工是：

- **规则引擎判"是不是异常"**——阈值写在 `RuleBook` 里，可复算、可辩护、可被审计质疑
- **LLM 只写"这意味着什么"**——输入限定为已确认的 findings，模型没有机会自己发明异常

这不是"保守"，是**把不可靠组件放在伤害最小的位置**。
一个在判定环节胡说八道的模型会让整份报告失去价值；一个在措辞环节发挥不佳的模型，
最坏结果只是文字不够漂亮，事实依然准确。

## 规则清单（10 条）

| 规则编号 | 名称 | 观察线（关注 / 重大） | 依据 |
|---|---|---|---|
| `BS_IDENTITY` | 资产负债表勾稽校验 | 容差 0.1% | 资产 = 负债 + 所有者权益 |
| `AR_GAP` | 应收增速 vs 营收增速 | 20pct / 40pct | 剪刀差是收入质量恶化先行指标 |
| `INV_GAP` | 存货增速 vs 营收增速 | 20pct / 40pct | 增速持续高于营收指向滞销 |
| `GROSS_MARGIN` | 毛利率大幅变动 | 5pct / 10pct | 异常升或降都需解释 |
| `OCF_DIVERGE` | 经营现金流 / 净利润 | 0.8 / 0.5 | 现金含量低指向纸面利润 |
| `DEBT_RATIO` | 资产负债率 | 70% / 80% | 杠杆与偿债风险基础度量 |
| `GOODWILL` | 商誉 / 净资产 | 30% / 50% | 并购减值风险敞口 |
| `MARGIN_EROSION` | 增收不增利 | 净利率降 2pct | 盈利质量恶化典型形态 |
| `SWING_SCAN` | 科目同比大幅波动扫描 | 50% / 100% | 全面覆盖，防专项规则遗漏 |
| `QUALITY_DRIFT` | 多项盈利质量指标连续同向恶化 | 5 项指标中 ≥3 项连续同向 | 单期够不上观察线、多期同向走是慢性恶化 |

其中 `AR_GAP` / `INV_GAP` / `GROSS_MARGIN` / `DEBT_RATIO` 四条**支持行业校准**：
传入基准簿后，观察线取该行业的 p75 分位而非全市场固定值（重大线只按原有比例推导，
不跟着行业样本漂移）。校准结果一律写进 `Finding.evidence.line_source`——
财务工具不能悄悄换阈值。

前 9 条是**两期规则**（接收 `cur, prev`）；`QUALITY_DRIFT` 是**序列规则**（接收整个 `series`），
只有传入 ≥3 期报表时才会执行。序列规则与两期规则通过 `is_series_rule` 标记区分，
`RuleEngine.run()` 据此分派，互不干扰。

全部阈值集中在 `RuleBook` 一个 dataclass 里。**改阈值不需要碰任何规则逻辑**——
这是为了让"参数该定多少"成为一个可以单独讨论、单独回测的问题。
与趋势相关的两个阈值同样在这里：`trend_min_periods = 3`（至少几期才算有趋势，
同时决定持续性升级看最近几期配对）与 `trend_min_agree = 3`（`QUALITY_DRIFT` 要求
5 项指标里至少几项连续同向才报）。

## 实测结果（示例数据，7 类异常全命中）

```
风险分 92.6/100 [极高]    重大 4  关注 3   合计 7

🔴 AR_GAP       应收 +64.3% vs 营收 +8.0%，剪刀差 56.3pct
🔴 INV_GAP      存货 +60.0% vs 营收 +8.0%，剪刀差 52.0pct
🔴 OCF_DIVERGE  经营现金流/净利润 = 0.38（线 ≥0.8）
🔴 SWING_SCAN   10 个科目波动超 ±50%
🟡 DEBT_RATIO   资产负债率 73.33%（上年 54.76%，+18.57pct）
🟡 GOODWILL     商誉占净资产 40.38%（线 30%）
🟡 MARGIN_EROSION  净利率 8.20% → 5.83%
```

132 个单元测试全部通过（含 1 条需 openpyxl 的底稿渲染测试，缺依赖时自动跳过）。
测试分十一层：模型层、序列层、规则层、引擎层、行业基准层、评分参数层、底稿层、
A/B 指标层、抽取层、编排层、渲染层、研判质量、端到端评测集。

### 多期模式实测（长恒电子，4 期，慢性恶化型）

`data/statements/长恒电子_2022A..2025A.json` 是第二套 fixture，刻意设计成
**每一期都够不上时值观察线、但连续四期构成明确恶化通道**：

| 指标 | 2022A | 2023A | 2024A | 2025A |
|---|---|---|---|---|
| 毛利率 | 29.00% | 27.98% | 26.67% | 25.00% |
| 净利率 | 8.60% | — | 6.76% | 5.54% |
| 资产负债率 | 50.00% | — | — | 57.27% |
| 经营现金流/净利润 | 1.14 | — | — | 0.82 |
| 流动比率 | 1.73 | — | — | 1.36 |

两期模式下这套数据**只能报出 1 项关注**（`MARGIN_EROSION`，净利率降 1.22pct，
够不上 2pct 重大线）——净利率三年从 8.60% 掉到 5.54% 这件事，
在两期视角里只是一个轻飘飘的"关注"。
切到 4 期模式：

```
风险分 63.2/100 [高]    重大 2  关注 0   合计 2

🔴 MARGIN_EROSION  净利率 6.76% → 5.54%（-1.22pct）
                   该信号在 2023A→2025A 连续 2 个报告期重复出现，
                   非孤立波动，已升级为重大
🔴 QUALITY_DRIFT   毛利率、净利率、经营现金流/净利润、资产负债率、流动比率
                   在 2022A→2025A 期间持续走坏，每一期都朝同一方向变动
```

注意 `MARGIN_EROSION` 那条：**它不是新增的发现，是原有发现被升级了。**
单期只看 2025A 时它是"关注"级（净利率降 1.22pct，够不上 2pct 重大线）；
但它在 2023A→2024A、2024A→2025A 两个相邻配对都触发，说明是结构性的，
于是升级为重大并在 detail 里追加持续性说明。**报告里同一个问题只出现一次，
风险分也不会双算。**

### 怎么读这个分数（面试会问的三个问题）

**Q：92.6 是怎么算出来的？为什么不是把异常数量加权求和？**

第一版就是加权求和，然后发现它有个致命缺陷：
4 项重大 = 4×10 = 40，8 项重大 = 80，一旦超过 10 分就 `min(100)` 封顶。
结果 **"4 项重大"和"8 项重大"拿到同样的 100 分**，分数完全失去区分度。

改成饱和函数 `(1 - e^(-raw/20)) × 100`：单调递增，永不触顶。

| 重大项数 | 修前 | 修后 |
|---|---|---|
| 4 | 100.00 | 86.47 |
| 8 | 100.00 | 98.17 |
| 差值 | **0.00** | **11.70** |

这个缺陷是我自己跑数据时发现的——**没有意识到"封顶"是一个 bug 的人，
不会去做这个对比表。** 评分函数必须保留边际区分能力，这是排序问题的基本要求。

**Q：scale=20 这个参数凭什么？**

老实说：**这是经验值，不是调优结果。** 当前权重下约 3 项重大 + 2 项关注能到 70 分区间，
符合直觉上"相当不妙但不至于灾难"的定位。它应该随历史样本回测校准。
在没有样本之前，任何精确到小数点的加权都是伪精确——**所以我把它写成常量并标注了这一点，
而不是假装它是经过验证的。**

**Q：为什么要有风险等级，不能只给分数吗？**

分数是标量，读者需要的是"该做什么"。所以每个分数映射到档位并附带建议动作：

| 分数 | 等级 | 建议动作 |
|---|---|---|
| ≥85 | 极高 | 暂缓出具结论，先逐项核查重大异常原始凭证 |
| ≥65 | 高 | 对重大异常做专项函证或走访核实后再定稿 |
| ≥35 | 中 | 就关注项向管理层提出书面问询 |
| <35 | 低 | 未见系统性异常，可常规复核 |

**Q：分数和严重度打架时听谁的？**

听严重度的。`risk_grade()` 取「分数档」与「严重度档」的较高者，
且**存在 1 项重大，档位就不得低于「高」**。

这条规则是修 bug 修出来的：4 期分析跑出"重大 2 项、关注 0 项、风险分 63.2"，
档位却落在「中」，建议动作是"就关注项向管理层提出书面问询"——
**一份报告告诉读者"去问那 0 项关注"，这是自相矛盾。**
饱和函数在重大项少时得分天然偏低（3 项重大约 70 分），
纯按分数切档会把"数量少但性质严重"的情况漏掉。

测试 `test_grade_never_contradicts_severity` 钉住了这个约束：
`by_score` 与 `by_severity` 取较高者，档位永远不会低于严重度所要求的水平。

### 行业校准实测（恒润重工，同一份数据两种口径）

`data/statements/恒润重工_2025A.json` 是第三套 fixture，`meta.industry = 工程机械`，
刻意设计成**全市场口径零异常、同行业口径两项进最差四分之一**：

| 指标 | 实测值 | 绝对线 | 工程机械 p75 |
|---|---|---|---|
| 应收剪刀差 | 15.0pct | 20（不触发） | **12（触发）** |
| 资产负债率 | 62.0% | 70（不触发） | **60（触发）** |

不传基准簿时输出"未触发任何异常规则"；传入后输出 2 项关注，
detail 里明确写出 `已按工程机械行业 p75 分位校准`，
`evidence.line_source` 记录机器可读的 `industry:p75`。

反向也同样成立——测试 `test_calibration_can_suppress_a_finding` 构造了一个
建筑施工公司（该行业剪刀差 p75 = 26pct，比全市场线还松），
绝对线下 gap 25pct 会报警，行业线下不报。**校准是双向的，
不是只用来提高灵敏度。**

### 参数敏感性实测（35 个组合，69% 稳健）

`scripts/sensitivity.py` 对 scale 与权重比跑网格，衡量指标是**档位是否翻转**，
不是分数变化——分数是连续量，改任何参数它都会变；
真正决定行动的是档位（极高=暂缓出具结论、高=专项函证、中=书面问询、低=常规复核）。

结果（`artifacts/sensitivity.md`）：

| 场景 | 基准档位 | 翻档格数 | 稳定性 |
|---|---|---|---|
| 示例制造（重大 4 / 关注 3） | 极高 | 0/35 | 100% |
| 长恒电子（重大 2 / 关注 0） | 高 | 5/35 | 86% |
| 恒润重工（关注 2，行业校准） | 中 | 0/35 | 100% |
| 合成边界（5 项关注贴 65 切点） | 中 | 10/35 | 71% |

全局 24/35 个组合在全部场景下不翻档，**默认参数（scale=20、比值 2.5）位于稳健区间内**。
合成边界那一行是最有信息量的：它在 scale ≤15 时全部翻档，说明
"分数恰好落在切点附近"的结论本就该被视为不稳健——
这正是不做敏感性分析就永远发现不了的事。

顺带一个诚实的注脚：无重大异常的场景里，`w_high/w_warn` 比值根本不进入计算
（风险分只由关注项贡献），矩阵整行同值是数学必然，不是稳定性证据。
报告里对这类场景单独标注了这一点，避免把"没测到"读成"很稳健"。

## 关键设计决策

### 单条规则崩溃不能让整轮分析停摆

`RuleEngine.run()` 用 try/except 包住每条规则，失败就记一条 `ERR_<rule>` 然后继续。

在财务场景里，**"少报一个异常"比"整份报告没出来"更危险**。
一个规则的边界条件写错（比如除零）不应该让其余 8 条规则的结果一起消失。
测试 `test_broken_rule_does_not_break_engine` 用一条必抛异常的假规则钉住了这个行为。

### 未识别的行要记下来，不能静默丢弃

`ExtractAgent.parse_text()` 逐行正则匹配中文科目名，匹配不上的行全部记入
`meta["unrecognized"]`，CLI 会打印到 stderr 提醒人工核对。

静默丢弃是财务数据处理里最危险的行为——你不知道自己漏了什么。
宁可让使用者看到"有 12 行没认出来"，也不要给他一份看似完整实则残缺的报表。

顺带一个实测踩到的坑：初版正则写成"整行必须以数字结尾"，
结果真实报表（行末几乎都带"万元"）**一行都解析不出来**，
10 个科目全部落入未识别列表。这个 bug 单看代码不容易发现——
必须拿真实格式的行去跑。现在正则允许行末单位，并统一换算到万元口径
（亿元 ×10000、千元 ×0.1、元 ×0.0001）。
测试 `test_line_with_unit_suffix_is_parsed` 与 `test_unit_conversion_to_wan_yuan` 钉住了它。

### 研判章节不能是明细的复述

第一版 `ReportAgent.reason()` 在无 LLM 时就是：

```python
parts = [f"### {f.title}\n{f.detail}" for f in findings]
```

这导致报告的第三章和第四章内容完全一样——**篇幅上去了，信息量为零**。
在作品集里这会被一眼看穿。

改写成真正的跨发现聚合：用静态归属表把规则聚成"收入质量 / 资产质量 / 偿债能力"
三条主线索，每条给出共同成因推断和核查方向，再排出核查优先级。
测试 `test_reasoning_does_not_restate_details` 把"复述即失分"这条约束固化下来。

**这是零依赖模板的价值所在**：不是"没有 LLM 时的降级方案"，
而是"LLM 不该做的那部分"——聚合与归因是确定性的，交给模型反而不可靠。

### 勾稽校验为什么排在第一

`BS_IDENTITY` 是 `DEFAULT_RULES` 里的第一条。因为**如果报表本身不平，
后面所有比率分析都建立在错误数据上**。先验证数据可信，再分析数据含义——
这个顺序不能反。

### 持续性信号要升级原有发现，不能新增一条

用户对这个功能的需求原话是：

> 「连续三年应收增速超营收 20pct」比「今年突然超一次」更值得警惕

直觉上会想"那就再报一条持续性 Finding"。但这样做有两个后果：

1. **风险分双算**——同一个应收问题，单期一次 + 持续性一次，`risk_score` 会加两遍
2. **报告里同一问题出现两次**——读者会以为是两个独立问题，核查顺序也被打乱

所以实现是**在原 Finding 上提升一级严重度并追加说明**：

```python
f.level = _escalate(f.level)          # WARN → HIGH，HIGH 不越界
f.detail += "该信号在 2023A→2025A 连续 2 个报告期重复出现，非孤立波动，已升级为重大。"
f.confidence = min(1.0, f.confidence + 0.1)
```

配套的两个判定口径：

- **"连续"不是"累计"**：要求最近 N-1 个相邻期配对**全部**触发。中间中断一期不计——
  一次坏、一次好、一次坏不是趋势，是波动。
- **重大不越界**：`_escalate()` 到 HIGH 就停。持续性只能把"关注"提升到"重大"，
  不能造出一个更严重的档位——动作上无法区分。

测试 `test_escalates_when_signal_persists` / `test_interrupted_signal_does_not_escalate`
（中断一期不升级）/ `test_escalation_is_idempotent_on_level` 三条分别钉住这三个口径。

### 为什么趋势判定要求至少 3 期

两期只能算一次同比，任何"趋势"都无从谈起。`RuleBook.trend_min_periods = 3`
意味着：2 期输入时序列规则全部静默跳过，`escalate_persistent_findings` 也直接返回——
**退化成一个纯两期分析，行为与改造前完全一致。**

这不是"少了个功能"，是**边界条件必须显式**。传 2 期却试图做趋势判定，
只会得到"看起来做了其实没做"的假象。

## 目录结构

```
finagent/
├── src/finagent/
│   ├── models.py       FinancialStatement / StatementSeries / Finding / AnalysisResult / 风险分与等级 / ScorePolicy
│   ├── rules.py        RuleBook（全部阈值）+ 10 条规则 + RuleEngine + 持续性升级 + 行业校准
│   ├── benchmarks.py   IndustryBenchmark / BenchmarkBook / load_benchmarks（分位数查询 + 样本量门槛）
│   ├── workpaper.py    WorkingPaper / Procedure / CrossReference / Signoff + md/xlsx 渲染
│   ├── abcompare.py    研判质量四维指标（保真 / 覆盖 / 复述 / 可复算）
│   ├── agents.py       LLM Protocol / ExtractAgent / AuditAgent / ReportAgent / Orchestrator
│   ├── render.py       console + markdown 两种渲染（多期时横向铺开全部期间）
│   └── __init__.py
├── scripts/
│   ├── run.py          CLI：analyze / workpaper / extract / rules / batch
│   ├── sensitivity.py  风险分参数敏感性分析（scale × 权重比网格）
│   └── ab_test.py      研判质量 A/B 对比（--stub 框架自检 / --api 真实模型）
├── tests/test_analysis.py  132 个单元测试
├── data/
│   ├── statements/
│   │   ├── 示例制造_2024A/2025A.json      两期 fixture（7 类异常，gold 集断言用）
│   │   ├── 长恒电子_2022A..2025A.json     四期 fixture（慢性恶化，趋势规则用）
│   │   └── 恒润重工_2024A/2025A.json      两期 fixture（行业校准，meta.industry=工程机械）
│   └── benchmarks/industry_benchmarks.json  5 个行业 × 4 项指标的分位数
└── artifacts/          生成的报告、底稿、敏感性与 A/B 结果
```

## 已知局限（诚实清单）

1. **报表科目靠正则解析**：格式一变就要改别名表。真实场景应接 XBRL 或财 reporting 系统导出。
2. **单位靠行末文本判断**：若报表在表头统一标注"单位：万元"而每行不带单位，
   当前会按"无单位即万元"处理——恰好对，但这属于巧合而非设计。需要在解析前显式传入表头单位。
3. **规则阈值未经历史回测**：观察线来自会计常识，不是从样本里拟合出来的。上线前必须校准。
4. **风险分权重是手工设定**：重大 10 / 关注 4 / 提示 1 的比值没有数据支撑。
   已做敏感性分析（`scripts/sensitivity.py`）：35 个组合里 24 个不翻档，
   默认参数在稳健区间内。但这只覆盖了已构造的 4 个场景，不等于全市场稳健。
5. **持续性判定只覆盖最近 2 个相邻配对**：`trend_min_periods = 3` 派生出
   `n_pairs = 2`，即"最近两期都命中才算持续"，对应的是**连续两年**（两次同比）的持续性。
   原始需求里"连续三年应收增速超营收"这种更严的场景需要把 `trend_min_periods` 调到 4。
   另外这个名字同时控制"至少几期才启用趋势"和"看几个配对"两件事，
   属于一个参数承载两个语义，拆分得更干净的做法是拆成 `trend_min_periods` 与 `persistence_min_pairs`。
6. **持续性升级只看单条规则的跨期命中**：不识别"两条不同规则轮流出现"的组合型持续信号
   （比如一期 `AR_GAP`、一期 `INV_GAP` 交替）。组合持续性是比单规则持续更弱的信号，
   但确实存在。
7. **QUALITY_DRIFT 的 5 项指标权重相同**：毛利率连续走坏和流动比率连续走坏被同等看待。
   真实场景里前者的信息量显然更大，需要按指标重要性加权。
8. **未接真实 LLM 验证**：`OpenAICompatLLM` 与 A/B 框架已就绪，但尚未用真实模型跑过。
   跑法：`python scripts/ab_test.py --api --model qwen-plus`
   （通义 / DeepSeek / 硅基流动 / vLLM 任一 OpenAI 兼容端点，参数走 `FINAGENT_LLM_API` 与 `FINAGENT_LLM_KEY`）。
   在此之前，所有 `--stub` 输出只证明框架能跑，不构成模型质量结论。
9. **行业基准是模拟数据**：`data/benchmarks/industry_benchmarks.json` 的 5 个行业 × 4 项指标
   是为了演示校准机制而构造的，不是从真实同行样本算出来的。接入前必须换成真实数据源，
   并补上样本口径（是否含 ST / 是否剔除异常值 / 取数期间）。
   此外只有 4 条规则支持校准，`OCF_DIVERGE`、`GOODWILL` 等仍是全市场固定线。
10. **基准簿按 p75 单分位校准**：只用了第 75 百分位，没用 p50/p90 构成区间。
    更稳妥的做法是给出"行业中位 ~ 高分位"的带，让判定落在带内时不报、超出上界才报。
11. **底稿的程序区是按执行动作反推的**：勾稽、分析、趋势三项程序由实际跑过的环节生成，
    但"追加函证""实地走访"这类需要人工触发的程序不会自动出现，
    底稿仍需要人工补程序、签编制人/复核人名。

## 下一步（按优先级）

1. **接真实 LLM 跑 A/B**：框架与四维指标已就绪，缺的只是一次真实运行。
   重点不是看模型"写得好不好"，是看它在**事实保真度**上能否稳定 100%——
   保真度掉下来的模型，写得再漂亮也不能用。
2. 给 `QUALITY_DRIFT` 的 5 项指标加重要性权重（毛利率 > 净利率 > 现金流 > 杠杆 > 流动性）
3. 把行业基准换成真实数据源，并从单分位（p75）升级为区间（p50~p90）
4. 收集专家标注样本，用样本校准 scale 与权重，替代当前的手工设定
5. ~~补多期趋势规则~~ **已完成**——序列规则 `QUALITY_DRIFT`、持续性升级、CLI `--periods`、4 期 fixture
6. ~~引入行业基准~~ **已完成**——`benchmarks.py` + 4 条规则校准 + 双向验证测试
7. ~~scale 与权重敏感性分析~~ **已完成**——`scripts/sensitivity.py`，35 组合网格
8. ~~输出审计底稿格式~~ **已完成**——`workpaper.py`，Markdown + xlsx 五张表

---

**这个项目的价值不在代码，在于它证明了一件事**：

懂会计的人做 AI 应用，第一反应不是"用大模型分析财报"，
而是"**哪些环节必须确定性，模型只能在哪儿介入**"。

规则引擎判异常、LLM 写结论——这个分工不是为了保守，
是因为**一份不能被复算的财务分析报告，在专业场景里等于没有价值。**
