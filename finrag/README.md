# finrag · 财务制度 RAG 检索系统

> 一个可运行、可评测、可解释的企业文档问答检索内核。
> 零第三方依赖，`git clone` 后直接能跑出 recall@1 / MRR 数字。

## 这是什么

不是 demo，是一个**带评测集的检索内核**。它回答的问题是：
「企业财务/制度类文档问答，怎么做才能让答案可信、可追责、可量化调优。」

针对财务场景做了专门设计：条款号溯源、数字幻觉检测、引用精确率。
这些不是通用 RAG 教程里有的东西——**这是懂会计的人才知道要做的东西**。

## 快速开始

```bash
# 环境：Python 3.13+，无需 pip install 任何东西

# 1. 构建索引
python scripts/run.py build

# 2. 提问（带条款溯源）
python scripts/run.py ask "出差住宿费在一线城市能报多少"

# 3. 全量评测（25 条标注用例）
python scripts/run.py eval --mode bm25

# 4. 三种检索模式对比
python scripts/run.py compare

# 5. 分块策略消融实验
python scripts/run.py ablate

# 6. 单元测试
python -m unittest discover -s tests
```

## 架构

```
docs/*.md              财务制度语料（Markdown，条款带编号）
   │
   ├─ structure_aware_chunks()   结构感知分块（按标题层级，非定长硬切）
   │      └─ 保留 heading_path + section_no + char offset
   │
   └─ Index.build()
          ├─ BM25      稀疏通道：关键词精确匹配（金额、条款号）
          └─ Embedder  稠密通道：语义相似（可插拔）
                 │
      Retriever.search()
          ├─ 双路召回  各取 candidate_k=30
          └─ RRF 融合  只看排名，免疫量纲差异
                 │
      GroundedQA.answer()
          ├─ 抽取式回答（离线可用）
          ├─ 数字幻觉检测  答案数字 ∉ 原文 → grounded=False
          └─ 引用溯源      chunk_id → 原文 offset
                 │
      evaluate()
          ├─ recall@1/3/5   检索召回
          ├─ MRR             排序质量
          ├─ 引用正确率      引用是否指向正确条款
          ├─ 关键词覆盖率    生成完整性
          └─ 数字幻觉率      生成可靠性
```

## 实测指标（25 条人工标注用例）

| 检索模式 | recall@1 | recall@3 | recall@5 | MRR | 关键词覆盖率 |
|---------|---------|---------|---------|-----|------------|
| BM25     | 0.960   | 1.000   | 1.000   | 0.980 | 0.860 |
| 向量     | 0.840   | 0.920   | 0.960   | 0.881 | 0.860 |
| 混合     | 0.840   | 0.960   | 0.960   | 0.900 | 0.860 |

生成层指标：引用正确率 0.224，数字幻觉率 **0.000**。

### 怎么读这张表（面试会问的三个问题）

**Q：为什么 BM25 反而比混合好？**
因为当前的 `HashingEmbedder` 是哈希向量，只有字面重合度，**没有语义能力**。
它把"住宿费"这个字面串当成相似度，在 RRF 里引入了噪声，把正确条款从第 1 位挤到第 2 位。
这是**真实缺陷，不是 bug**——恰好证明了"必须换真实 embedding"。

**Q：混合检索不是标配吗？**
是，但前提是两个通道都有信息量。当其中一路是噪声，融合只会拉低性能。
正确做法是**先用评测集验证每个通道**，再决定要不要融合。盲目上混合是常见的错误。

**Q：怎么换成真向量？**
设置环境变量即可，**不需要改任何检索逻辑**：

```bash
# 方式 A：OpenAI 兼容 API（通义/硅基流动/DeepSeek/本地 vLLM 都行）
export FINRAG_EMBEDDING_KEY=sk-xxx
export FINRAG_EMBEDDING_API=https://api.siliconflow.cn/v1/embeddings
export FINRAG_EMBEDDING_MODEL=BAAI/bge-m3
python scripts/run.py compare

# 方式 B：本地模型（8G 显存建议先 bge-small-zh）
pip install sentence-transformers
python scripts/run.py compare --embedder local
```

换完必须**重跑 `compare` 对比指标**。数字涨了才说明这个模型对你的场景有效——
"换个更强的就一定更好"是错的，bge-m3 在金融术语短文本上未必打得过 bge-large-zh。

## 关键设计决策

### 为什么按标题结构分块，不用定长切

财务制度是「章-节-条」三层结构。定长硬切会把"差旅标准"和"超标处理"切开，
召回的片段语义不完整，答案就答不全。结构感知分块让每个 chunk 自带完整上下文。

代价是对文档格式有依赖——所以**先评测再调参**：

```
max_len  min_len     块数      R@1      R@5      MRR
    300       20     28    0.960    1.000    0.980
    600       40     28    0.960    1.000    0.980
    900       60     27    0.960    1.000    0.980
```

**诚实的结论：这批语料条款切分干净，分块参数不敏感，瓶颈不在分块而在向量质量。**
很多教程会写"调分块参数能提升 20%"，在这个场景下是假的。

### 为什么用 RRF 不用分数加权

BM25 分数无上界（可以到几十），余弦相似度固定在 [0,1]。
直接加权要归一化，而归一化方式引入分布假设，换个语料就得重调。
RRF 只看排名（第1名得 1/61，第2名得 1/62），天然免疫量纲问题。

### 数字幻觉检测

财务出错的杀伤力几乎都在数字上。策略：抽取答案中所有数字，
与引用原文比对，找不到就标记 `grounded=False`。

实现时踩过坑：章节编号「3.2」「第一章」会被误判成金额，导致幻觉率虚高到 1.0。
已用正则排除 `数字.数字` 形式的章节号。**这就是为什么要有评测集——
没有评测，你根本发现不了自己这个 bug，只会以为模型全在幻觉。**

### 引用精确率只有 0.224，说明什么

因为 top_k=5 里有 5 条引用，只有一条指向 gold 条款。
这个数字低是**正常的**——引用列表本身不是答案，模型在 top-1 就该命中。
真正该盯的是 recall@1 和 MRR。

## 目录结构

```
finrag/
├── src/finrag/
│   ├── retrieval.py       分块 + BM25 + 向量通道 + RRF + 带引用问答 + 幻觉检测
│   ├── api_embedder.py    可插拔真实 embedding（API / 本地 / 哈希三层降级）
│   ├── evaluate.py        recall@k / MRR / 引用正确率 / 幻觉率
│   └── __init__.py
├── scripts/run.py         CLI 入口：build / ask / eval / compare / ablate
├── tests/test_retrieval.py  34 个单元测试，含端到端 smoke
├── docs/                  财务制度语料（2 篇，28 个条款块）
├── data/eval_set.json     25 条标注用例（gold 条款号 + 关键词 + 难度）
└── artifacts/             索引与实验结果
```

## 已知局限（诚实清单）

1. **向量通道无语义**：当前用哈希向量兜底，必须换真实 embedding 才能体现混合检索价值。
2. **回答是抽取式**：未接 LLM，只是取最高分片段首句。接入后关键词覆盖率应显著提升。
3. **语料仅 28 块**：规模太小，评测指标的置信区间偏窄。生产需扩到百块以上。
4. **无并发与缓存**：单机单线程，面向面试演示而非生产负载。
5. **分块依赖 Markdown 标题**：扫描版 PDF 需先 OCR + 结构还原。

## 下一步（按优先级）

1. 接真实 embedding（`bge-m3` 或 API），重跑 `compare` 验证混合检索是否反超 BM25
2. 接 LLM 做生成，用 `data/eval_set.json` 的 `gold_keywords` 量化关键词覆盖率提升
3. 加 hybrid 权重调节（RRF_K 可调），扫参看是否存在最优值
4. 语料扩到 10 万字：把公司全套财务制度、HR 制度、报销单据样本灌进去
5. 加 FastAPI 服务层，让 HR 周报/月报生成直接调用

---

**这个项目的价值不在代码，在于它证明了一件事**：
懂会计 + 会一点 Python 的人，能做出比"调 API 的文科生"更懂业务可靠性的 AI 系统。
AI 应用的门槛在业务理解，不在调包。
