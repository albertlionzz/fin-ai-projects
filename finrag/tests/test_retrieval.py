"""单元测试：零依赖跑。

  python -m unittest discover -s tests -v
  或
  python tests/test_retrieval.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finrag import (  # noqa: E402
    BM25,
    Index,
    Retriever,
    load_cases,
    load_documents,
    structure_aware_chunks,
    tokenize,
)
from finrag.retrieval import HashingEmbedder, cosine, rrf_fuse  # noqa: E402

DOCS = ROOT / "docs"
EVAL = ROOT / "data" / "eval_set.json"


class TestTokenize(unittest.TestCase):
    def test_cjk_bigram(self):
        toks = tokenize("住宿费")
        self.assertIn("住", toks)
        self.assertIn("住宿", toks)
        self.assertIn("宿费", toks)

    def test_numbers_kept(self):
        self.assertIn("650", tokenize("不超过 650 元"))

    def test_empty(self):
        self.assertEqual(tokenize(""), [])


class TestChunking(unittest.TestCase):
    def setUp(self):
        self.text = (DOCS / "差旅费管理制度.md").read_text(encoding="utf-8")

    def test_section_numbers_extracted(self):
        chunks = structure_aware_chunks(self.text, "t", "测试")
        secs = {c.section_no for c in chunks}
        self.assertIn("3.1", secs, "应提取住宿费标准条款号")
        self.assertIn("2.1", secs)

    def test_heading_path_populated(self):
        chunks = structure_aware_chunks(self.text, "t", "测试")
        c = next(c for c in chunks if c.section_no == "3.1")
        self.assertTrue(c.heading_path, "heading_path 不应为空")
        self.assertIn("住宿费标准", c.heading)
        self.assertTrue(
            any("第三章" in h for h in c.heading_path),
            f"heading_path 应含章级标题，实际={c.heading_path}",
        )

    def test_offsets_valid(self):
        chunks = structure_aware_chunks(self.text, "t", "测试")
        for c in chunks:
            self.assertLessEqual(c.char_start, c.char_end)
            self.assertGreaterEqual(c.char_end, c.char_start + len(c.text) - 5)

    def test_max_len_respected(self):
        chunks = structure_aware_chunks(self.text, "t", "测试", max_len=200)
        long = [c for c in chunks if len(c.text) > 200]
        self.assertEqual(long, [], "超长块应被二次切分")

    def test_chunk_id_unique(self):
        chunks = structure_aware_chunks(self.text, "t", "测试")
        ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_empty_doc(self):
        self.assertEqual(structure_aware_chunks("", "t", "x", min_len=100), [])


class TestBM25(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            tokenize("伙食补助按出差自然日计算，每人每天 120 元"),
            tokenize("市内交通费每人每天包干 80 元"),
            tokenize("住宿费一线城市每晚不超过 450 元"),
        ]
        self.bm = BM25(self.corpus)

    def test_scores_shape(self):
        self.assertEqual(len(self.bm.scores(tokenize("住宿费"))), 3)

    def test_relevant_doc_ranks_first(self):
        top = self.bm.top_k(tokenize("住宿费标准"), k=1)
        self.assertTrue(top)
        self.assertEqual(top[0][0], 2, "住宿费应命中第3篇")

    def test_unseen_term_returns_empty(self):
        self.assertEqual(self.bm.top_k(tokenize("量子纠缠态"), k=3), [])

    def test_idf_positive(self):
        self.assertTrue(all(v > 0 for v in self.bm.idf.values()))


class TestRRF(unittest.TestCase):
    def test_both_rankings_contribute(self):
        f = rrf_fuse([[0, 1, 2], [2, 1, 0]])
        self.assertIn(0, f)
        self.assertIn(2, f)
        self.assertGreater(f[0], 0)

    def test_agreement_scores_higher(self):
        # 两个通道都排第一的文档应得分最高
        f = rrf_fuse([[5, 6], [6, 5]])
        self.assertEqual(f[5], f[6])

    def test_single_ranking(self):
        f = rrf_fuse([[1, 2, 3]])
        self.assertEqual(len(f), 3)


class TestEmbedder(unittest.TestCase):
    def setUp(self):
        self.emb = HashingEmbedder(dim=128)

    def test_normalized(self):
        v = self.emb.encode(["住宿费标准"])[0]
        self.assertAlmostEqual(sum(x * x for x in v) ** 0.5, 1.0, places=5)

    def test_deterministic(self):
        a = self.emb.encode(["测试文本"])[0]
        b = self.emb.encode(["测试文本"])[0]
        self.assertEqual(a, b)

    def test_self_similarity_highest(self):
        v = self.emb.encode(["住宿费标准"])[0]
        same = cosine(v, self.emb.encode(["住宿费标准"])[0])
        other = cosine(v, self.emb.encode(["差旅审批流程"])[0])
        self.assertGreater(same, other)

    def test_empty_input(self):
        self.assertEqual(self.emb.encode([]), [])


class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = Index.build(load_documents(DOCS), embedder=HashingEmbedder(256))
        cls.retr = Retriever(cls.index)

    def test_index_built(self):
        self.assertGreater(len(self.index.chunks), 20)
        self.assertEqual(len(self.index.matrix), len(self.index.chunks))

    def test_search_returns_hits(self):
        hits = self.retr.search("住宿费标准", top_k=5)
        self.assertEqual(len(hits), 5)
        for h in hits:
            self.assertIsNotNone(h.chunk.chunk_id)

    def test_bm25_finds_correct_section(self):
        hits = self.retr.search("一线城市住宿费限额", top_k=3, mode="bm25")
        self.assertIn("3.1", [h.chunk.section_no for h in hits])

    def test_top_k_respected(self):
        self.assertEqual(len(self.retr.search("报销", top_k=2)), 2)

    def test_cite_string(self):
        h = self.retr.search("伙食补助", top_k=1)[0]
        self.assertIn(">", h.cite)

    def test_empty_index_safe(self):
        empty = Index.build([], embedder=HashingEmbedder(64))
        self.assertEqual(Retriever(empty).search("任何问题"), [])

    def test_gibberish_query_does_not_crash(self):
        hits = self.retr.search("量子引力波色动力学", top_k=3)
        self.assertIsInstance(hits, list)


class TestEvalSet(unittest.TestCase):
    def setUp(self):
        self.cases = load_cases(EVAL)

    def test_case_count(self):
        self.assertGreaterEqual(len(self.cases), 20)

    def test_required_fields(self):
        for c in self.cases:
            self.assertIn("id", c)
            self.assertIn("question", c)
            self.assertIsInstance(c["gold"], list)
            self.assertTrue(c["gold"], f"{c['id']} 缺 gold")

    def test_ids_unique(self):
        ids = [c["id"] for c in self.cases]
        self.assertEqual(len(ids), len(set(ids)))

    def test_difficulty_values(self):
        allowed = {"easy", "medium", "hard"}
        for c in self.cases:
            self.assertIn(c["difficulty"], allowed)


class TestEndToEnd(unittest.TestCase):
    """端到端冒烟：必须 recall@5 达标，否则说明检索链路坏了。"""

    def setUp(self):
        self.index = Index.build(load_documents(DOCS), embedder=HashingEmbedder(256))
        self.cases = load_cases(EVAL)

    def test_recall_at_5_above_threshold(self):
        from finrag import evaluate

        rep = evaluate(self.index, self.cases, mode="bm25", top_k=5)
        self.assertGreaterEqual(
            rep.recall_at_5, 0.9,
            f"recall@5={rep.recall_at_5:.3f} 低于 0.9，检索链路可能已损坏",
        )

    def test_mrr_above_threshold(self):
        from finrag import evaluate

        rep = evaluate(self.index, self.cases, mode="bm25", top_k=5)
        self.assertGreaterEqual(rep.mrr, 0.85)

    def test_no_hallucination_on_gold(self):
        """gold 问题的答案数字必须全部能在原文找到。"""
        from finrag import evaluate

        rep = evaluate(self.index, self.cases, mode="bm25", top_k=5)
        self.assertEqual(rep.hallucination_rate, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
