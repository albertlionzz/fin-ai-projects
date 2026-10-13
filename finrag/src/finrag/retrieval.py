"""财务制度 RAG 检索层：分块 + 混合检索 + 引用溯源。

设计要点（每一条都是面试会追问的点）：

1. 分块策略：按 Markdown 标题做结构感知切分，保留 heading path 作为元数据。
   为什么不按固定字符数硬切？因为财务制度是"条款-款项-标准"三层结构，
   硬切会把"差旅标准"和"例外审批条件"切开，检索时召回的片段语义不完整。

2. 混合检索：BM25（稀疏、关键词精确）+ 向量（稠密、语义相近）用 RRF 融合。
   财务场景里"住宿费 480 元"这种精确数字 BM25 完胜向量；
   "出差住宿怎么报销"这种泛化问法向量完胜 BM25。两者必须都要。

3. 引用溯源：每个 chunk 带 doc_id / heading / 原文 offset，
   answer 里能定位回原文片段。这是财务场景的生命线——答案错了要能追责到条款。

无第三方依赖，纯标准库。向量通道留了 Embedder 协议，
接入真实 embedding API 时只需实现 protocol，不必改上层逻辑。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, Protocol, Sequence

# ---------------------------------------------------------------- 数据结构


@dataclass
class Chunk:
    """一个知识片段。offset 保留原文位置，用于引用溯源。"""

    chunk_id: str
    doc_id: str
    doc_title: str
    heading: str
    heading_path: list[str]
    text: str
    char_start: int
    char_end: int
    section_no: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


@dataclass
class Hit:
    """一条检索结果，带融合分数与各通道原始排名。"""

    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    vec_rank: int | None = None
    rrf_score: float = 0.0

    @property
    def cite(self) -> str:
        return f"{self.chunk.doc_title} > {self.chunk.heading}"


@dataclass
class Answer:
    """带引用的答案。citations 指向具体 chunk_id。"""

    question: str
    answer: str
    citations: list[str] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    grounded: bool = True
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- 分词

_CJK = re.compile(r"[\u4e00-\u9fff]")
_TOKEN = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z]+|\d+(?:\.\d+)?")


def tokenize(text: str) -> list[str]:
    """中文按字 + 双字 bigram 切，英文数字按词切。

    单字保证召回（"费"能命中"费用报销"），bigram 保证精度（"报销"作为一个 token）。
    财务文本里大量金额、比例、科目代码，双字粒度比纯单字噪声低很多。
    """
    tokens = _TOKEN.findall(text.lower())
    grams: list[str] = []
    for i, tok in enumerate(tokens):
        grams.append(tok)
        if _CJK.match(tok) and i + 1 < len(tokens) and _CJK.match(tokens[i + 1]):
            grams.append(tok + tokens[i + 1])
    return grams


# ---------------------------------------------------------------- 分块


_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def structure_aware_chunks(
    text: str,
    doc_id: str,
    doc_title: str,
    min_len: int = 40,
    max_len: int = 600,
) -> list[Chunk]:
    """按 Markdown 标题层级做结构感知分块。

    标题本身不单独成块，而是作为 heading 元数据附着到它下面的所有块。
    这样"第三章 差旅费管理 / 3.2 住宿标准"这一整段会作为一个 chunk，
    检索到时人能立刻看到上下文，模型也不会丢条款层级。
    """
    lines = text.split("\n")
    chunks: list[Chunk] = []
    path: list[str] = []
    buf: list[str] = []
    buf_start = 0
    running = 0  # 记录 buf 第一行在原文中的起始偏移

    def flush() -> None:
        nonlocal buf, buf_start, running
        body = "\n".join(buf).strip()
        if len(body) >= min_len or (body and chunks and len(body) >= 12):
            section_no = ""
            if path:
                m = re.match(r"^([\d.]+)", path[-1])
                section_no = m.group(1) if m else ""
            cid = hashlib.sha1(
                f"{doc_id}|{path}|{body[:80]}|{buf_start}".encode("utf-8")
            ).hexdigest()[:12]
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    heading=path[-1] if path else doc_title,
                    heading_path=list(path),
                    text=body,
                    char_start=buf_start,
                    char_end=buf_start + len(body),
                    section_no=section_no,
                )
            )
        buf = []
        running += 0

    offset = 0
    for line in lines:
        line_start = offset
        offset += len(line) + 1  # +1 for '\n'
        m = _HEADING.match(line)
        if m:
            flush()
            buf_start = offset
            level = len(m.group(1))
            title = m.group(2).strip()
            path = path[: level - 1] + [title]
        else:
            if not buf:
                buf_start = line_start
            buf.append(line)
    flush()

    # 超长块再按 max_len 二次切分（按句读，避免切断语义）
    out: list[Chunk] = []
    for c in chunks:
        if len(c.text) <= max_len:
            out.append(c)
            continue
        for part, off in _split_long(c.text, max_len):
            cid = hashlib.sha1(f"{c.chunk_id}|{off}".encode()).hexdigest()[:12]
            out.append(
                Chunk(
                    chunk_id=cid,
                    doc_id=c.doc_id,
                    doc_title=c.doc_title,
                    heading=c.heading,
                    heading_path=list(c.heading_path),
                    text=part,
                    char_start=c.char_start + off,
                    char_end=c.char_start + off + len(part),
                    section_no=c.section_no,
                )
            )
    return out


_SENT_END = re.compile(r"(?<=[。！？；;!?\n])")


def _split_long(text: str, max_len: int) -> list[tuple[str, int]]:
    """超长块按句边界切，返回 (片段, 在原文中的偏移)。"""
    pieces: list[tuple[str, int]] = []
    cur, cur_off = "", 0
    pos = 0
    for sent in _SENT_END.split(text):
        if not sent:
            continue
        if len(cur) + len(sent) > max_len and cur:
            pieces.append((cur.strip(), cur_off))
            cur, cur_off = "", pos
        if not cur:
            cur_off = pos
        cur += sent
        pos += len(sent)
    if cur.strip():
        pieces.append((cur.strip(), cur_off))
    return pieces


# ---------------------------------------------------------------- BM25


class BM25:
    """Okapi BM25。中文短文本上 k1=1.5, b=0.75 是稳健默认值。"""

    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = list(corpus)
        self.n = len(self.corpus)
        self.doc_len = [len(d) for d in self.corpus]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.tf: list[Counter] = [Counter(d) for d in self.corpus]
        df: Counter = Counter()
        for t in self.tf:
            df.update(t.keys())
        # BM25+ 平滑，避免高频词 idf 变负
        self.idf: dict[str, float] = {
            w: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for w, c in df.items()
        }

    def scores(self, query: Sequence[str]) -> list[float]:
        out = [0.0] * self.n
        qtf = Counter(query)
        for w, qn in qtf.items():
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(w, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                out[i] += idf * (f * (self.k1 + 1) / denom) * (1 + 0.3 * (qn - 1))
        return out

    def top_k(self, query_tokens: Sequence[str], k: int) -> list[tuple[int, float]]:
        sc = self.scores(query_tokens)
        ranked = sorted(enumerate(sc), key=lambda x: -x[1])
        return [(i, s) for i, s in ranked[:k] if s > 0]


# ---------------------------------------------------------------- 向量通道


class Embedder(Protocol):
    """向量通道协议。接真实 embedding API 时实现它即可，上层不用改。"""

    dim: int

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """零依赖的确定性哈希向量，作为向量通道的可运行替身。

    刻意不假装它是语义向量——它的作用是让整条 RRF 链路今天就能跑通、能评测。
    换成真 embedding（bge-m3 / text-embedding-3-small）时，只替换这一个类。
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = tokenize(text)
        for tok in toks:
            h = hashlib.md5(tok.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "big") % self.dim
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------- 索引


@dataclass
class Index:
    chunks: list[Chunk]
    bm25: BM25
    embedder: Embedder
    matrix: list[list[float]]

    @classmethod
    def build(
        cls,
        chunks: Sequence[Chunk],
        embedder: Embedder | None = None,
    ) -> "Index":
        emb = embedder or HashingEmbedder()
        corpus = [tokenize(c.text + " " + " ".join(c.heading_path)) for c in chunks]
        return cls(
            chunks=list(chunks),
            bm25=BM25(corpus),
            embedder=emb,
            matrix=emb.encode([c.text + " " + " ".join(c.heading_path) for c in chunks]),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "chunks.jsonl").write_text(
            "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in self.chunks),
            encoding="utf-8",
        )
        (p / "config.json").write_text(
            json.dumps({"n": len(self.chunks), "dim": self.embedder.dim}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder | None = None) -> "Index":
        p = Path(path)
        chunks = [
            Chunk.from_dict(json.loads(line))
            for line in (p / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls.build(chunks, embedder)


# ---------------------------------------------------------------- 融合检索


RRF_K = 60  # RRF 平滑常数，60 是社区标准取值


def rrf_fuse(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion。

    为什么用 RRF 而不是分数加权归一化？因为 BM25 分数和余弦相似度
    量纲完全不同（一个是 unbounded 正数，一个是 [0,1]），归一化会引入
    分布假设。RRF 只看排名，天然免疫量纲问题——这是它被广泛使用的根本原因。
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking, start=1):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (k + rank)
    return fused


class Retriever:
    """混合检索器：BM25 + 向量，RRF 融合。"""

    def __init__(self, index: Index, candidate_k: int = 30):
        self.index = index
        self.candidate_k = candidate_k

    def search(self, query: str, top_k: int = 5, mode: str = "hybrid") -> list[Hit]:
        chunks = self.index.chunks
        if not chunks:
            return []

        if mode == "bm25":
            ranked = self.index.bm25.top_k(tokenize(query), self.candidate_k)
            bm_rank = {i: r for r, (i, _) in enumerate(ranked, 1)}
            vec_rank: dict[int, int] = {}
        elif mode == "vector":
            qv = self.index.embedder.encode([query])[0]
            sims = [cosine(qv, mv) for mv in self.index.matrix]
            ranked = sorted(enumerate(sims), key=lambda x: -x[1])[: self.candidate_k]
            vec_rank = {i: r for r, (i, _) in enumerate(ranked, 1)}
            bm_rank = {}
        else:
            bm_ranked = self.index.bm25.top_k(tokenize(query), self.candidate_k)
            bm_rank = {i: r for r, (i, _) in enumerate(bm_ranked, 1)}
            qv = self.index.embedder.encode([query])[0]
            sims = [cosine(qv, mv) for mv in self.index.matrix]
            vec_ranked = sorted(enumerate(sims), key=lambda x: -x[1])[: self.candidate_k]
            vec_rank = {i: r for r, (i, _) in enumerate(vec_ranked, 1)}

        rankings = [list(bm_rank.keys()), list(vec_rank.keys())]
        fused = rrf_fuse(rankings)

        bm_score = (
            dict(enumerate(self.index.bm25.scores(tokenize(query))))
            if mode != "vector"
            else {}
        )
        qv_cache = self.index.embedder.encode([query])[0]
        vec_score_all = [cosine(qv_cache, mv) for mv in self.index.matrix]

        hits = [
            Hit(
                chunk=chunks[i],
                score=bm_score.get(i, 0.0),
                bm25_rank=bm_rank.get(i),
                vec_rank=vec_rank.get(i),
                rrf_score=s,
            )
            for i, s in fused.items()
        ]
        # 标题路径越深的块轻微加权：财务条款里层级信息即重要性
        hits.sort(key=lambda h: -(h.rrf_score * (1 + 0.03 * len(h.chunk.heading_path))))
        return hits[:top_k]


# ---------------------------------------------------------------- 生成（有引用）


ANSWER_PROMPT = """你是财务制度问答助手。只依据给定的资料片段回答问题。

硬性规则：
1. 每个结论后面必须标注来源编号，如 [1]、[2]。
2. 资料片段中没有明确写到的内容，一律回答"制度中未明确规定"。
3. 不要编造金额、比例、日期。宁可说不明确，也不要猜。
4. 回答控制在 150 字以内，用简体中文。

资料片段：
{context}

问题：{question}

回答："""


class GroundedQA:
    """带引用与幻觉检测的问答器。

    幻觉检测策略：抽取答案中的所有数字，与引用片段原文做比对。
    数字在原文中找不到 → 标记 ungrounded。
    这是财务场景最实用的检测——因为财务出错的杀伤力几乎都在数字上。
    """

    def __init__(self, index: Index, retriever: Retriever):
        self.index = index
        self.retriever = retriever

    def _format_context(self, hits: Sequence[Hit]) -> str:
        return "\n\n".join(
            f"[{i+1}] {h.chunk.doc_title} > {h.chunk.heading}\n{h.chunk.text}"
            for i, h in enumerate(hits)
        )

    def answer(self, question: str, top_k: int = 5) -> Answer:
        hits = self.retriever.search(question, top_k=top_k)
        if not hits:
            return Answer(
                question=question,
                answer="制度中未明确规定。",
                grounded=False,
                notes=["检索无召回，可能是知识库为空或问题超出范围"],
            )

        # 离线抽取式回答：取排名最高片段的首句作为核心答案
        lead = hits[0].chunk.text.strip().split("\n")[0]
        lead = re.sub(r"^[\d.]+\s*", "", lead).strip()
        answer_text = lead[:300]
        notes: list[str] = []
        if len(hits) > 1:
            others = "；".join(h.chunk.heading for h in hits[1:3])
            answer_text += f"（相关条款：{others}）"
            notes.append("未接LLM，当前为抽取式回答；接 API 后由模型基于片段生成")
        # 幻觉检测只看核心答案，不看附加的条款索引
        grounded, reason = self._check_grounded(lead, hits)
        if not grounded:
            notes.append(reason)

        return Answer(
            question=question,
            answer=answer_text,
            citations=[h.chunk.chunk_id for h in hits],
            hits=list(hits),
            grounded=grounded,
            notes=notes,
        )

    _NUM = re.compile(r"(?<![\d.])\d+(?:\.\d+)?%?(?![\d.])")
    # 章节编号（3.2 / 2.1.1）不是金额，需排除，否则幻觉检测会全线误报
    _SECTION = re.compile(r"\d+(?:\.\d+)+")

    def _extract_nums(self, text: str) -> set[str]:
        cleaned = self._SECTION.sub(" ", text)
        return set(self._NUM.findall(cleaned))

    def _check_grounded(self, answer: str, hits: Sequence[Hit]) -> tuple[bool, str]:
        src = " ".join(h.chunk.text for h in hits)
        src_nums = self._extract_nums(src)
        ans_nums = self._extract_nums(answer)
        bad = sorted(n for n in ans_nums if n not in src_nums)
        if bad:
            return False, f"数字幻觉风险：{bad} 未出现在引用片段中"
        return True, ""


def load_documents(docs_dir: str | Path) -> list[Chunk]:
    """读取 docs/ 下所有 markdown 文档并分块。"""
    chunks: list[Chunk] = []
    for p in sorted(Path(docs_dir).glob("*.md")):
        text = p.read_text(encoding="utf-8")
        title = text.split("\n", 1)[0].lstrip("# ").strip() or p.stem
        chunks.extend(structure_aware_chunks(text, doc_id=p.stem, doc_title=title))
    return chunks
