"""可插拔的真实 Embedding 通道。

哈希向量（HashingEmbedder）没有语义能力——它只能把字面重合当成相似。
接真实 embedding 后，这里是唯一需要改的地方，上层检索逻辑零改动。

三种接法，按成本从低到高：

1. 本地 bge-m3 / gte（需 torch + sentence-transformers，8G显存可跑小模型）
   pip install sentence-transformers
2. OpenAI 兼容 API（通义/DeepSeek/vLLM 自建都走这条路）
   设 FINRAG_EMBEDDING_API 即可，无需改代码
3. 离线兜底：HashingEmbedder（默认，保证零依赖可跑）

关键：换 embedder 后必须重跑 evaluate.py 对比指标，
数字涨了才说明这个模型对你的场景有效——不是"换个更强的就一定更好"。
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Sequence

from .retrieval import Embedder


class OpenAICompatEmbedder(Embedder):
    """调用 OpenAI 兼容 /v1/embeddings 接口。

    兼容：OpenAI、硅基流动、通义千问、DeepSeek、Ollama、vLLM 自建。
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        dim: int = 1024,
        timeout: float = 60.0,
    ):
        self.model = model or os.getenv("FINRAG_EMBEDDING_MODEL", "bge-m3")
        self.base_url = (
            base_url
            or os.getenv("FINRAG_EMBEDDING_API", "https://api.siliconflow.cn/v1/embeddings")
        ).rstrip("/")
        self.api_key = api_key or os.getenv("FINRAG_EMBEDDING_KEY", "")
        self.dim = dim
        self.timeout = timeout
        if not self.api_key:
            raise ValueError(
                "缺少 API key。设置环境变量 FINRAG_EMBEDDING_KEY，"
                "或改用 LocalEmbedder / HashingEmbedder。"
            )

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = json.dumps({"model": self.model, "input": list(texts)}).encode("utf-8")
        req = urllib.request.Request(
            self.base_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        vecs = [item["embedding"] for item in data["data"]]
        self.dim = len(vecs[0]) if vecs else self.dim
        return vecs


class LocalEmbedder(Embedder):
    """本地 sentence-transformers 模型。适合离线与隐私敏感场景。

    中文场景推荐：BAAI/bge-m3（多语言，1024维）、BAAI/bge-large-zh-v1.5
    8G 显存建议 bge-small-zh-v1.5（512维，约 400MB）先跑通，再换大的。
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer  # 延迟导入

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self.model.encode(
            list(texts),
            batch_size=16,
            normalize_embeddings=True,  # 归一化后点积即余弦
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]


def build_embedder(kind: str = "auto", **kwargs):
    """按名取 embedder。auto 策略：API → 本地模型 → 哈希兜底。"""
    kind = (kind or "auto").lower()
    if kind == "api":
        return OpenAICompatEmbedder(**kwargs)
    if kind == "local":
        return LocalEmbedder(**kwargs)
    if kind in ("hash", "none"):
        from .retrieval import HashingEmbedder

        return HashingEmbedder()

    if os.getenv("FINRAG_EMBEDDING_KEY"):
        try:
            return OpenAICompatEmbedder()
        except Exception as e:  # noqa: BLE001
            print(f"[warn] API embedder 初始化失败，回退哈希：{e}")
    try:
        return LocalEmbedder()
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 本地模型不可用（{e}），回退哈希向量。")
        from .retrieval import HashingEmbedder

        return HashingEmbedder()
