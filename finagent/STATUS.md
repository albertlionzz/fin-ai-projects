# finagent 交付状态纪要（STATUS）

> 最后更新：2026-11-04｜维护人：飞哥 / WorkBuddy
> 本文是 MVP 封板时的交付快照，长期细节见 `README.md` 与 `docs/设计说明.md`。

## 0. 一句话定位

finagent 是一个 **A 股财报研判工具**：抓取上市公司三大报表 → 行业基准校准阈值
→ 11 条规则引擎跑异常 → 生成审计式底稿。当前是**结构完整、可独立运行的 MVP**，
已封板。

## 1. 封板已交付

| 模块 | 状态 | 说明 |
|---|---|---|
| 行业基准簿 | ✅ 3 行业 / 893 家 | 食品饮料 103 / 电子 485 / 计算机 305，0 异常 0 告警 |
| 规则引擎 | ✅ 11 条规则 | 含 `NET_MARGIN_TURN`（本次新增） |
| 单元测试 | ✅ 185 个全绿 | 覆盖解析、规则、A/B、口径修复 |
| 权重跨行业复核 | ✅ 2 行业 | 食品饮料 + 电子，翻转方向无例外 |
| 规则缺口研究 | ✅ §15 + §15.7 | 一次真缺口（`NET_MARGIN_TURN`）+ 一次负结果 |
| A/B 真实 LLM 验收 | ✅ 本地 qwen3:4b | 事实保真 4/4 100%、覆盖率 100%、复述率 0% |

## 2. 三个最有价值的实测结论

1. **权重量化（§14）**：`QUALITY_DRIFT` 加权后，食品饮料翻转 5 处（17%）、
   电子翻转 6 处（20%）。得分全为整数，触发集合被切成 **4 分 / 5 分两簇**，
   不存在"4.5 分临界"的模糊地带。跨两行业翻转方向**无例外**。
2. **`NET_MARGIN_TURN`（§15）**：全样本 911 家，净利率由正转负 79 家，
   补规则前只覆盖 48 家，**补后 79/79 全覆盖**；65 家由负转正不误报。
3. **应收/存货慢性趋势（§15.7，负结果）**：596 家有三年数据的公司里，
   应收 18 家慢性（10 家已被 `AR_GAP` 抓住）、存货 15 家慢性（7 家已被 `INV_GAP`
   抓住），漏报全在 20pct 阈值之下且多为良性。**不加规则**——这是"查过了、
   设计够用"的负结论，留给后来者省一次重复调研。

## 3. 已知局限（诚实清单摘要，完整版见 README §已知局限 1–14）

| 编号 | 局限 | 影响 | 关掉它的路径 |
|---|---|---|---|
| **7** | `QUALITY_DRIFT` 权重未经专家校准，是推理起点 | 权重合理性缺第三方背书 | **把 `.scratch/annotation_answers.md` 的默认值改成你的判断 → 重跑 `score_annotations.py`** |
| 8 | 真实 LLM 只跑过本地 4B，未跑云端大模型 | 保真度达标不能外推到云端 | `scripts/ab_test.py --llm api`（见 README） |
| 9 | 基准是第三方站点 **AI 抽取**结果，非审计数据 | 金额 3~5 位有效数字、站点入库不全 | 接 XBRL / 财 reporting 导出 |
| 14 | `NET_MARGIN_TURN` 只覆盖"过零轴"，不覆盖"亏损扩大" | −3% → −6% 这类幅度恶化无人管 | 新规则或并入幅度阈值 |

> **当前唯一卡飞哥的口子是局限 7**：权重仍是我的推理起点，不是你的判断。
>
> 已用一组**忠实默认值**（= 现行推理起点）填好 `.scratch/annotation_answers.md`
> 并跑通判卷器 `score_annotations.py`：Q10 归一后逐项回到 3/2/2/1/1，
> 30 家样本两边都触发 6 家，**零变化基线**——链路自洽，局限 7 仍未关闭。
> 后续只需改答案文件里任一行的值再重跑即可看到差异。
> 另备 `.scratch/weight_sensitivity.py`：6 组权重 + 8 档门槛对照，
> 结论是"改权重换的是哪几家、改门槛换的是有几家"，详见设计说明 §14.6。

## 4. 暂缓 / 未做（非必需，源站限流所限）

- **机械设备（净 521 家）、医药生物（净 461 家）抓取暂停**：数据源
  `bbs.auditdog.cn` 对我的 IP 持续返回 `HTTP 429`，并发抓取会触发更严厉限流。
  已抓的 **~244 家机械设备缓存留盘**（0 个 `.404` 永久跳过标记），
  限流解除后 `python scripts/fetch_benchmarks.py --industries 机械设备 --force`
  会复用缓存、只补缺口。**绝不再并发**。
- **计算机三年趋势复核未完成**：缺 2023AR 三期缓存（卡网络），暂缓。
- **权重专家校准**：见局限 7。

## 5. 下一步（按优先级）

1. **飞哥填权重问卷** → 我跑 `score_annotations.py` 出校准方案（关局限 7）。
2. **源站限流解除后**串行补抓 `机械设备` → `医药生物`（单行业、足够延迟）。
3. **云端大模型跑 A/B**（关局限 8）。

## 6. 怎么跑

```bash
# 安装（项目自带 venv 指引见 README）
pip install -e .

# 跑测试
python -m unittest discover -s tests -q        # 185 个，全绿

# 看规则清单 / 跑单家分析
python scripts/run.py rules
python scripts/run.py analyze <股票代码> <报告期> <公司名> <行业>

# 抓取行业基准（需联网；已抓的会自动复用缓存，--force 仅绕过请求数护栏）
python scripts/fetch_benchmarks.py --industries 食品饮料 --force --delay 1.5

# A/B 真实 LLM 验收（本地或云端）
python scripts/ab_test.py --llm local --runs 3
# 云端：
# FINAGENT_LLM_API=<端点> FINAGENT_LLM_MODEL=qwen-plus FINAGENT_LLM_KEY=<key> \
#   python scripts/ab_test.py --llm api --runs 3
```

## 7. 仓库状态

- 远端：`albertlionzz/fin-ai-projects`（SSH 推送，密钥 `~/.ssh/id_ed25519_finagent`）
- 本地未跟踪：`data/benchmarks/raw/`（HTML 缓存，上 GB）、`build_report.json`、
  `.scratch/`（调研脚本）——均已在 `.gitignore`，不入库。
- 当前 HEAD 即封板点；`data/benchmarks/industry_benchmarks.auditdog.json`
  （真实基准簿）仅含 3 行业，as_of 2026-10-27。
