"""财务制度 RAG 检索系统（零依赖可运行版）。

模块:
  retrieval.py  分块 / BM25 / 向量通道 / RRF 融合 / 带引用问答
  evaluate.py   recall@k / MRR / 引用正确率 / 幻觉率
  api_embedder.py 可选：接真实 embedding API 的向量通道

作者: 转型学习项目
"""

from .retrieval import (
    Answer,
    BM25,
    Chunk,
    Embedder,
    GroundedQA,
    HashingEmbedder,
    Hit,
    Index,
    Retriever,
    load_documents,
    rrf_fuse,
    structure_aware_chunks,
    tokenize,
)
from .evaluate import EvalReport, evaluate, load_cases

__all__ = [
    "Answer",
    "BM25",
    "Chunk",
    "Embedder",
    "EvalReport",
    "GroundedQA",
    "HashingEmbedder",
    "Hit",
    "Index",
    "Retriever",
    "evaluate",
    "load_cases",
    "load_documents",
    "rrf_fuse",
    "structure_aware_chunks",
    "tokenize",
]
