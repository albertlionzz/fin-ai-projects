"""单元测试：零依赖跑。

  python -m unittest discover -s tests -v
  或
  python tests/test_analysis.py

测试分三层：
  1. 模型层：财务比率算得对不对
  2. 规则层：每条规则的触发边界（含不触发的情形）
  3. 编排层：整条链路能跑通且产出完整
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finagent import (  # noqa: E402
    AnalysisResult,
    ExtractAgent,
    Finding,
    FinancialStatement,
    Level,
    NullLLM,
    Orchestrator,
    ReportAgent,
    StatementSeries,
    build_series,
    load_benchmarks,
    load_pair,
    render_console,
    render_markdown,
)
from finagent.abcompare import (  # noqa: E402
    check_coverage,
    check_fidelity,
    check_restatement,
    determinism_ratio,
    evaluate,
    extract_numbers,
    number_pool,
)
from finagent.benchmarks import BenchmarkBook, IndustryBenchmark  # noqa: E402
from finagent.models import DEFAULT_POLICY, ScorePolicy  # noqa: E402
from finagent.rules import (  # noqa: E402
    QUALITY_DRIFT_INDICATORS,
    RuleBook,
    RuleEngine,
    rule_balance_sheet_identity,
    rule_cash_profit_divergence,
    rule_debt_ratio,
    rule_goodwill_ratio,
    rule_gross_margin_swing,
    rule_inventory_growth_gap,
    rule_margin_erosion,
    rule_receivable_growth_gap,
)
from finagent.workpaper import (  # noqa: E402
    build_working_paper,
    render_workpaper_markdown,
)

STMT_DIR = ROOT / "data" / "statements"
BENCH_PATH = ROOT / "data" / "benchmarks" / "industry_benchmarks.json"
RB = RuleBook()


def mk(period: str, **items: float) -> FinancialStatement:
    """构造测试用报表。"""
    return FinancialStatement(period=period, items=dict(items))


# ---------------------------------------------------------------- 模型层


class TestRatios(unittest.TestCase):
    def test_gross_margin(self):
        st = mk("T", revenue=1000.0, cost_of_sales=700.0)
        self.assertAlmostEqual(st.gross_margin(), 30.0, places=6)

    def test_gross_margin_zero_revenue_returns_none(self):
        """营收为 0 时不能做除法，必须返回 None 而不是崩或给 inf。"""
        self.assertIsNone(mk("T", revenue=0.0, cost_of_sales=10.0).gross_margin())

    def test_net_margin(self):
        st = mk("T", revenue=2000.0, net_profit=180.0)
        self.assertAlmostEqual(st.net_margin(), 9.0, places=6)

    def test_debt_ratio(self):
        st = mk("T", total_liabilities=600.0, total_assets=1000.0)
        self.assertAlmostEqual(st.debt_ratio(), 60.0, places=6)

    def test_ocf_to_profit_normal(self):
        st = mk("T", operating_cash_flow=900.0, net_profit=1000.0)
        self.assertAlmostEqual(st.ocf_to_profit(), 0.9, places=6)

    def test_ocf_to_profit_negative_profit_returns_none(self):
        """净利润为负时比值无解释力，应返回 None。"""
        self.assertIsNone(mk("T", operating_cash_flow=100.0, net_profit=-500.0).ocf_to_profit())

    def test_yoy(self):
        cur = mk("2025A", revenue=1100.0)
        prev = mk("2024A", revenue=1000.0)
        self.assertAlmostEqual(cur.yoy("revenue", prev), 10.0, places=6)

    def test_yoy_zero_base_returns_none(self):
        """基期为 0 时增速无定义，返回 None 而不是抛异常。"""
        cur = mk("2025A", revenue=1100.0)
        prev = mk("2024A", revenue=0.0)
        self.assertIsNone(cur.yoy("revenue", prev))

    def test_missing_item_returns_none(self):
        self.assertIsNone(mk("T", revenue=100.0).net_margin())

    def test_current_ratio(self):
        st = mk("T", current_assets=800.0, current_liabilities=400.0)
        self.assertAlmostEqual(st.current_ratio(), 2.0, places=6)

    def test_roe(self):
        st = mk("T", net_profit=150.0, net_assets=1000.0)
        self.assertAlmostEqual(st.roe(), 15.0, places=6)


class TestRiskScore(unittest.TestCase):
    def test_no_findings_is_zero(self):
        r = AnalysisResult(company="X", period="T")
        self.assertEqual(r.risk_score(), 0.0)

    def test_score_monotonic_and_never_caps(self):
        """关键性质：异常越多分越高，且永远不触顶——否则多报异常就失去区分度。"""
        def score(n: int) -> float:
            r = AnalysisResult(
                company="X", period="T",
                findings=[Finding(rule_id=f"R{i}", level=Level.HIGH, title="t", detail="d") for i in range(n)],
            )
            return r.risk_score()

        s4, s8, s16 = score(4), score(8), score(16)
        self.assertLess(s4, s8)
        self.assertLess(s8, s16)
        self.assertLess(s16, 100.0, "饱和函数不应触顶 100")

    def test_high_outweighs_warn(self):
        hi = AnalysisResult(company="X", period="T",
                            findings=[Finding(rule_id="A", level=Level.HIGH, title="t", detail="d")])
        wa = AnalysisResult(company="X", period="T",
                            findings=[Finding(rule_id="B", level=Level.WARN, title="t", detail="d")])
        self.assertGreater(hi.risk_score(), wa.risk_score())

    def test_grade_thresholds(self):
        """档位 = 分数档与严重度档的较高者。

        不能只看分数：饱和函数在重大项少时得分偏低（2 项重大仅 63 分），
        若只按分数切档会出现"有重大异常却建议常规复核"的自相矛盾。
        """
        def grade(n_high: int, n_warn: int = 0) -> str:
            fs = [Finding(rule_id=f"H{i}", level=Level.HIGH, title="t", detail="d") for i in range(n_high)]
            fs += [Finding(rule_id=f"W{i}", level=Level.WARN, title="t", detail="d") for i in range(n_warn)]
            return AnalysisResult(company="X", period="T", findings=fs).risk_grade()

        self.assertEqual(grade(0), "低")          # 无发现
        self.assertEqual(grade(0, 3), "中")       # 仅关注项，分数 45
        self.assertEqual(grade(1), "高")          # 1 项重大 → 不低于高（分数仅 39）
        self.assertEqual(grade(2), "高")          # 2 项重大，分数 63 但仍属高
        self.assertEqual(grade(3), "高")          # 3 项重大
        self.assertEqual(grade(4), "极高")        # 4 项重大 → 极高

    def test_grade_never_contradicts_severity(self):
        """硬约束：存在重大异常时，建议动作不得是"常规复核"或"仅问询关注项"。

        这条钉住的是曾经的真实缺陷——输出里写着"2 项重大"，
        处置建议却是"建议就关注项向管理层提出书面问询"。
        """
        for n_high in (1, 2, 3, 4):
            r = AnalysisResult(
                company="X", period="T",
                findings=[Finding(rule_id=f"H{i}", level=Level.HIGH, title="t", detail="d")
                          for i in range(n_high)],
            )
            self.assertIn(r.risk_grade(), ("高", "极高"), f"{n_high} 项重大时档位过低")
            self.assertNotIn("常规复核", r.risk_action())
            self.assertNotIn("关注项", r.risk_action())

    def test_grade_never_contradicts_score(self):
        """遍历组合，确保等级单调不回落——分数升而等级降是逻辑错误。"""
        order = {"低": 0, "中": 1, "高": 2, "极高": 3}
        prev_rank, prev_score = -1, -1.0
        for n in range(0, 13):
            r = AnalysisResult(company="X", period="T",
                               findings=[Finding(rule_id=f"H{i}", level=Level.HIGH, title="t", detail="d")
                                         for i in range(n)])
            rank, score = order[r.risk_grade()], r.risk_score()
            self.assertGreaterEqual(rank, prev_rank, f"{n} 项重大时等级回落")
            self.assertGreater(score, prev_score, f"{n} 项重大时分数未递增")
            prev_rank, prev_score = rank, score

    def test_action_present_for_every_grade(self):
        for n in (0, 1, 3, 8):
            r = AnalysisResult(company="X", period="T",
                               findings=[Finding(rule_id=f"H{i}", level=Level.HIGH, title="t", detail="d")
                                         for i in range(n)])
            self.assertTrue(r.risk_action().strip(), f"档位 {r.risk_grade()} 缺少建议动作")


# ---------------------------------------------------------------- 规则层


class TestRuleReceivableGap(unittest.TestCase):
    def test_triggers_high_when_gap_large(self):
        cur = mk("2025A", accounts_receivable=170.0, revenue=1080.0)
        prev = mk("2024A", accounts_receivable=100.0, revenue=1000.0)
        f = rule_receivable_growth_gap(cur, prev, RB)
        self.assertIsNotNone(f)
        self.assertEqual(f.level, Level.HIGH)      # 70% - 8% = 62pct > 40
        self.assertEqual(f.rule_id, "AR_GAP")

    def test_triggers_warn_in_middle_band(self):
        # 应收 +28%，营收 +0% → 28pct，落在 20~40 之间
        cur = mk("2025A", accounts_receivable=128.0, revenue=1000.0)
        prev = mk("2024A", accounts_receivable=100.0, revenue=1000.0)
        f = rule_receivable_growth_gap(cur, prev, RB)
        self.assertIsNotNone(f)
        self.assertEqual(f.level, Level.WARN)

    def test_not_triggered_when_aligned(self):
        cur = mk("2025A", accounts_receivable=110.0, revenue=1100.0)
        prev = mk("2024A", accounts_receivable=100.0, revenue=1000.0)
        self.assertIsNone(rule_receivable_growth_gap(cur, prev, RB))

    def test_not_triggered_when_receivable_falls(self):
        """应收下降是好事，不能因为营收也在降就误报。"""
        cur = mk("2025A", accounts_receivable=50.0, revenue=500.0)
        prev = mk("2024A", accounts_receivable=100.0, revenue=1000.0)
        self.assertIsNone(rule_receivable_growth_gap(cur, prev, RB))

    def test_missing_item_returns_none(self):
        cur = mk("2025A", revenue=1000.0)
        prev = mk("2024A", accounts_receivable=100.0, revenue=1000.0)
        self.assertIsNone(rule_receivable_growth_gap(cur, prev, RB))


class TestRuleInventoryGap(unittest.TestCase):
    def test_triggers(self):
        cur = mk("2025A", inventory=160.0, revenue=1080.0)
        prev = mk("2024A", inventory=100.0, revenue=1000.0)
        f = rule_inventory_growth_gap(cur, prev, RB)
        self.assertIsNotNone(f)
        self.assertEqual(f.rule_id, "INV_GAP")

    def test_not_triggered(self):
        cur = mk("2025A", inventory=105.0, revenue=1100.0)
        prev = mk("2024A", inventory=100.0, revenue=1000.0)
        self.assertIsNone(rule_inventory_growth_gap(cur, prev, RB))


class TestRuleGrossMargin(unittest.TestCase):
    def test_triggers_on_large_drop(self):
        cur = mk("2025A", revenue=1000.0, cost_of_sales=800.0)   # 20%
        prev = mk("2024A", revenue=1000.0, cost_of_sales=700.0)  # 30%
        f = rule_gross_margin_swing(cur, prev, RB)
        self.assertIsNotNone(f)
        self.assertEqual(f.rule_id, "GM_SWING")

    def test_triggers_on_large_rise(self):
        """毛利率大幅上升同样需要解释，不能只报下降。"""
        cur = mk("2025A", revenue=1000.0, cost_of_sales=600.0)   # 40%
        prev = mk("2024A", revenue=1000.0, cost_of_sales=700.0)  # 30%
        self.assertIsNotNone(rule_gross_margin_swing(cur, prev, RB))

    def test_not_triggered_on_small_change(self):
        cur = mk("2025A", revenue=1000.0, cost_of_sales=707.0)   # 29.3%
        prev = mk("2024A", revenue=1000.0, cost_of_sales=700.0)  # 30%
        self.assertIsNone(rule_gross_margin_swing(cur, prev, RB))


class TestRuleCashProfit(unittest.TestCase):
    def test_triggers_when_diverged(self):
        cur = mk("2025A", operating_cash_flow=300.0, net_profit=1000.0)
        prev = mk("2024A", operating_cash_flow=900.0, net_profit=1000.0)
        f = rule_cash_profit_divergence(cur, prev, RB)
        self.assertIsNotNone(f)
        self.assertEqual(f.level, Level.HIGH)   # 0.3 < 0.5

    def test_not_triggered_when_healthy(self):
        cur = mk("2025A", operating_cash_flow=950.0, net_profit=1000.0)
        prev = mk("2024A", operating_cash_flow=900.0, net_profit=1000.0)
        self.assertIsNone(rule_cash_profit_divergence(cur, prev, RB))


class TestRuleDebtRatio(unittest.TestCase):
    def test_triggers(self):
        cur = mk("2025A", total_liabilities=750.0, total_assets=1000.0)
        prev = mk("2024A", total_liabilities=500.0, total_assets=1000.0)
        f = rule_debt_ratio(cur, prev, RB)
        self.assertIsNotNone(f)
        self.assertEqual(f.level, Level.WARN)   # 75% 在 70~80 之间

    def test_triggers_high(self):
        cur = mk("2025A", total_liabilities=850.0, total_assets=1000.0)
        prev = mk("2024A", total_liabilities=500.0, total_assets=1000.0)
        self.assertEqual(rule_debt_ratio(cur, prev, RB).level, Level.HIGH)

    def test_not_triggered(self):
        cur = mk("2025A", total_liabilities=400.0, total_assets=1000.0)
        prev = mk("2024A", total_liabilities=500.0, total_assets=1000.0)
        self.assertIsNone(rule_debt_ratio(cur, prev, RB))


class TestRuleGoodwill(unittest.TestCase):
    def test_triggers(self):
        cur = mk("2025A", goodwill=400.0, net_assets=1000.0)
        prev = mk("2024A", goodwill=100.0, net_assets=1000.0)
        f = rule_goodwill_ratio(cur, prev, RB)
        self.assertIsNotNone(f)
        self.assertEqual(f.rule_id, "GOODWILL")

    def test_not_triggered_when_low(self):
        cur = mk("2025A", goodwill=100.0, net_assets=1000.0)
        self.assertIsNone(rule_goodwill_ratio(cur, mk("2024A"), RB))

    def test_not_triggered_when_net_assets_negative(self):
        """净资产为负时该比值无意义，应跳过而不是算出负数比例。"""
        cur = mk("2025A", goodwill=400.0, net_assets=-100.0)
        self.assertIsNone(rule_goodwill_ratio(cur, mk("2024A"), RB))


class TestRuleMarginErosion(unittest.TestCase):
    def test_triggers_when_growth_without_profit(self):
        cur = mk("2025A", revenue=1200.0, net_profit=60.0)    # 5%
        prev = mk("2024A", revenue=1000.0, net_profit=100.0)  # 10%
        f = rule_margin_erosion(cur, prev, RB)
        self.assertIsNotNone(f)
        self.assertEqual(f.rule_id, "MARGIN_EROSION")

    def test_not_triggered_when_revenue_falls(self):
        """营收下滑时净利率下降是自然结果，不属于"增收不增利"。"""
        cur = mk("2025A", revenue=800.0, net_profit=40.0)
        prev = mk("2024A", revenue=1000.0, net_profit=100.0)
        self.assertIsNone(rule_margin_erosion(cur, prev, RB))

    def test_not_triggered_when_margin_improves(self):
        cur = mk("2025A", revenue=1200.0, net_profit=150.0)
        prev = mk("2024A", revenue=1000.0, net_profit=100.0)
        self.assertIsNone(rule_margin_erosion(cur, prev, RB))


class TestRuleBalanceSheetIdentity(unittest.TestCase):
    def test_passes_when_balanced(self):
        cur = mk("2025A", total_assets=1000.0, total_liabilities=600.0, total_equity=400.0)
        self.assertIsNone(rule_balance_sheet_identity(cur, mk("2024A"), RB))

    def test_triggers_when_unbalanced(self):
        cur = mk("2025A", total_assets=1000.0, total_liabilities=600.0, total_equity=300.0)
        f = rule_balance_sheet_identity(cur, mk("2024A"), RB)
        self.assertIsNotNone(f)
        self.assertEqual(f.level, Level.HIGH)
        self.assertEqual(f.rule_id, "BS_IDENTITY")

    def test_tolerates_rounding(self):
        """千分之一以内的舍入误差不应报警。"""
        cur = mk("2025A", total_assets=1000.0, total_liabilities=600.0, total_equity=399.9)
        self.assertIsNone(rule_balance_sheet_identity(cur, mk("2024A"), RB))

    def test_skips_when_items_missing(self):
        cur = mk("2025A", total_assets=1000.0)
        self.assertIsNone(rule_balance_sheet_identity(cur, mk("2024A"), RB))


# ---------------------------------------------------------------- 引擎层


class TestRuleEngine(unittest.TestCase):
    def test_sorted_by_level_desc(self):
        cur = mk("2025A", revenue=1000.0, cost_of_sales=800.0, total_liabilities=850.0,
                 total_assets=1000.0, operating_cash_flow=100.0, net_profit=500.0)
        prev = mk("2024A", revenue=1000.0, cost_of_sales=700.0, total_liabilities=500.0,
                  total_assets=1000.0, operating_cash_flow=450.0, net_profit=500.0)
        out = RuleEngine().run(cur, prev)
        levels = [int(f.level) for f in out]
        self.assertEqual(levels, sorted(levels, reverse=True), "结果应按严重度倒序")

    def test_broken_rule_does_not_break_engine(self):
        """单条规则抛异常时，其余规则必须继续跑——这是可用性底线。"""
        def broken(cur, prev, rb):
            raise RuntimeError("故意失败")

        def ok(cur, prev, rb):
            return Finding(rule_id="OK", level=Level.WARN, title="正常", detail="d")

        eng = RuleEngine(rules=(broken, ok))
        out = eng.run(mk("2025A"), mk("2024A"))
        self.assertEqual(len(out), 2, "失败规则应被记录而非中断")
        self.assertTrue(any(f.rule_id == "OK" for f in out))
        self.assertTrue(any(f.rule_id.startswith("ERR_") for f in out))

    def test_custom_rules_replaceable(self):
        def only_one(cur, prev, rb):
            return Finding(rule_id="ONLY", level=Level.INFO, title="t", detail="d")

        out = RuleEngine(rules=(only_one,)).run(mk("2025A"), mk("2024A"))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].rule_id, "ONLY")


# ---------------------------------------------------------------- 多期趋势层


def mk_series(*specs: tuple[str, dict]) -> StatementSeries:
    """快速构造序列。specs 为 (期间, 科目字典)。"""
    return StatementSeries(statements=[mk(p, **items) for p, items in specs])


class TestStatementSeries(unittest.TestCase):
    def test_sorted_chronologically_regardless_of_input_order(self):
        """传入乱序也必须按时间正序——调用方不该为排序负责。"""
        s = mk_series(
            ("2025A", {"revenue": 300.0}),
            ("2023A", {"revenue": 100.0}),
            ("2024A", {"revenue": 200.0}),
        )
        self.assertEqual(s.periods, ["2023A", "2024A", "2025A"])
        self.assertEqual(s.latest.period, "2025A")
        self.assertEqual(s.previous.period, "2024A")

    def test_quarterly_periods_sort_by_number_not_string(self):
        """'2025Q1' 按字符串比 '2024A' 大会排到后面——必须按数字比。"""
        s = mk_series(
            ("2025Q1", {"revenue": 100.0}),
            ("2024A", {"revenue": 90.0}),
        )
        self.assertEqual(s.periods, ["2024A", "2025Q1"])

    def test_pairs_are_consecutive(self):
        s = mk_series(
            ("2022A", {"revenue": 1.0}),
            ("2023A", {"revenue": 2.0}),
            ("2024A", {"revenue": 3.0}),
        )
        self.assertEqual([(a.period, b.period) for a, b in s.pairs()],
                         [("2022A", "2023A"), ("2023A", "2024A")])

    def test_trailing_pairs_takes_last_n(self):
        s = mk_series(
            ("2022A", {"revenue": 1.0}), ("2023A", {"revenue": 2.0}),
            ("2024A", {"revenue": 3.0}), ("2025A", {"revenue": 4.0}),
        )
        got = [(a.period, b.period) for a, b in s.trailing_pairs(2)]
        self.assertEqual(got, [("2023A", "2024A"), ("2024A", "2025A")])

    def test_has_trend_requires_min_periods(self):
        """2 期只能做同比，谈不上趋势——这是 trend_min_periods=3 的依据。"""
        two = mk_series(("2024A", {"revenue": 1.0}), ("2025A", {"revenue": 2.0}))
        three = mk_series(("2023A", {"revenue": 1.0}), ("2024A", {"revenue": 2.0}),
                          ("2025A", {"revenue": 3.0}))
        self.assertFalse(two.has_trend(3))
        self.assertTrue(three.has_trend(3))


class TestPersistentEscalation(unittest.TestCase):
    """持续性升级：这是多期趋势的核心机制。"""

    def _persistent(self) -> StatementSeries:
        """应收增速连续四期跑赢营收，剪刀差始终超 20pct。"""
        return mk_series(
            ("2022A", {"revenue": 100000.0, "accounts_receivable": 19000.0,
                       "inventory": 14000.0, "cost_of_sales": 71000.0}),
            ("2023A", {"revenue": 104000.0, "accounts_receivable": 24000.0,
                       "inventory": 16000.0, "cost_of_sales": 74900.0}),
            ("2024A", {"revenue": 108000.0, "accounts_receivable": 30000.0,
                       "inventory": 18500.0, "cost_of_sales": 79200.0}),
            ("2025A", {"revenue": 112000.0, "accounts_receivable": 37500.0,
                       "inventory": 21500.0, "cost_of_sales": 84000.0}),
        )

    def test_escalates_when_signal_persists(self):
        s = self._persistent()
        eng = RuleEngine()
        base = eng.run(s.latest, s.previous)
        with_trend = eng.run(s.latest, s.previous, series=s)

        base_ar = [f for f in base if f.rule_id == "AR_GAP"][0]
        trend_ar = [f for f in with_trend if f.rule_id == "AR_GAP"][0]
        self.assertEqual(base_ar.level, Level.WARN, "单期应在观察线内判关注")
        self.assertEqual(trend_ar.level, Level.HIGH, "持续出现应升级为重大")
        self.assertIn("重复出现", trend_ar.detail, "升级须说明是持续性导致的")

    def test_no_escalation_without_series(self):
        """不传序列时行为必须与改造前完全一致——这是向后兼容底线。"""
        s = self._persistent()
        out = RuleEngine().run(s.latest, s.previous)
        for f in out:
            self.assertNotIn("重复出现", f.detail)
            self.assertNotIn("跨期持续性", f.basis)

    def test_interrupted_signal_does_not_escalate(self):
        """中间断一期不算持续——'连续'与'累计'必须区分。

        这是刻意设计：2023 触线、2024 恢复、2025 再触线，
        说明情况有变化，不能按"结构性问题"处理。
        """
        s = mk_series(
            ("2022A", {"revenue": 100000.0, "accounts_receivable": 19000.0, "cost_of_sales": 71000.0}),
            ("2023A", {"revenue": 104000.0, "accounts_receivable": 26000.0, "cost_of_sales": 74900.0}),
            ("2024A", {"revenue": 108000.0, "accounts_receivable": 25000.0, "cost_of_sales": 79200.0}),
            ("2025A", {"revenue": 112000.0, "accounts_receivable": 36000.0, "cost_of_sales": 84000.0}),
        )
        out = RuleEngine().run(s.latest, s.previous, series=s)
        ar = [f for f in out if f.rule_id == "AR_GAP"][0]
        self.assertNotIn("重复出现", ar.detail, "中断一期不得判定为持续")

    def test_escalation_is_idempotent_on_level(self):
        """重大已是最高级，升级不得越界，也不得重复追加说明文字。"""
        s = self._persistent()
        out = RuleEngine().run(s.latest, s.previous, series=s)
        for f in out:
            self.assertLessEqual(int(f.level), int(Level.HIGH))
            self.assertEqual(f.detail.count("重复出现"), 1)

    def test_two_periods_skips_trend_entirely(self):
        """只有两期时趋势规则必须静默跳过，不得报错或产出半截结论。"""
        s = mk_series(
            ("2024A", {"revenue": 100000.0, "accounts_receivable": 19000.0, "cost_of_sales": 71000.0}),
            ("2025A", {"revenue": 104000.0, "accounts_receivable": 26000.0, "cost_of_sales": 74900.0}),
        )
        out = RuleEngine().run(s.latest, s.previous, series=s)
        self.assertFalse(any(f.rule_id == "QUALITY_DRIFT" for f in out))
        self.assertTrue(all("重复出现" not in f.detail for f in out))


class TestPersistencePairsSplit(unittest.TestCase):
    """trend_min_periods 与 persistence_min_pairs 的拆分。

    拆分前一个参数承载两个语义：n_pairs = trend_min_periods - 1。
    想把持续性判定收紧到 3 对（"连续三年"），就不得不把启用门槛抬到
    4 期——副作用是 3 期数据的公司连趋势规则都跑不上了。
    """

    def _persistent(self) -> StatementSeries:
        return mk_series(
            ("2022A", {"revenue": 100000.0, "accounts_receivable": 19000.0,
                       "inventory": 14000.0, "cost_of_sales": 71000.0}),
            ("2023A", {"revenue": 104000.0, "accounts_receivable": 24000.0,
                       "inventory": 16000.0, "cost_of_sales": 74900.0}),
            ("2024A", {"revenue": 108000.0, "accounts_receivable": 30000.0,
                       "inventory": 18500.0, "cost_of_sales": 79200.0}),
            ("2025A", {"revenue": 112000.0, "accounts_receivable": 37500.0,
                       "inventory": 21500.0, "cost_of_sales": 84000.0}),
        )

    def test_default_is_behavior_preserving(self):
        """默认 2 与拆分前逐字等价（旧式 trend_min_periods - 1 = 2）。"""
        self.assertEqual(RuleBook().trend_min_periods, 3)
        self.assertEqual(RuleBook().persistence_min_pairs, 2)

    def test_stricter_persistence_does_not_raise_enable_gate(self):
        """核心性质：收紧持续性判定，3 期公司照样跑趋势规则。

        旧写法做不到这一点——trend_min_periods=4 会把启用门槛一起抬到 4 期。
        """
        three = mk_series(
            ("2023A", {"revenue": 100000.0, "accounts_receivable": 19000.0,
                       "inventory": 14000.0, "cost_of_sales": 71000.0}),
            ("2024A", {"revenue": 104000.0, "accounts_receivable": 24000.0,
                       "inventory": 16000.0, "cost_of_sales": 74900.0}),
            ("2025A", {"revenue": 108000.0, "accounts_receivable": 30000.0,
                       "inventory": 18500.0, "cost_of_sales": 79200.0}),
        )
        rb = RuleBook(persistence_min_pairs=3)
        self.assertEqual(rb.trend_min_periods, 3, "启用门槛没被连带抬高")
        out = RuleEngine(book=rb).run(three.latest, three.previous, series=three)
        ar = [f for f in out if f.rule_id == "AR_GAP"]
        self.assertTrue(ar, "3 期照样跑规则并触发 AR_GAP")
        self.assertNotIn("重复出现", ar[0].detail,
                         "凑不齐 3 对，不得升级")

    def test_three_pairs_still_escalates(self):
        """4 期有 3 对且全部触发时，3 对门槛下仍应升级。"""
        s = self._persistent()
        out = RuleEngine(book=RuleBook(persistence_min_pairs=3)).run(
            s.latest, s.previous, series=s
        )
        ar = [f for f in out if f.rule_id == "AR_GAP"][0]
        self.assertEqual(ar.level, Level.HIGH)

    def test_invalid_config_raises(self):
        """明显非法的配置当场炸，不静默跑出没有意义的结果。"""
        with self.assertRaises(ValueError):
            RuleBook(trend_min_periods=1)
        with self.assertRaises(ValueError):
            RuleBook(persistence_min_pairs=0)

    def test_render_follows_metrics_not_series_shape(self):
        """渲染层读 metrics.trend_enabled，不自己拿 series 重算门槛。

        改了 RuleBook 让趋势没跑起来时，控制台不得显示"趋势区间"——
        显示层只反映事实，不重新定义事实。
        """
        from finagent.models import AnalysisResult

        s = self._persistent()
        result = AnalysisResult(
            company="测试", period="2025A",
            metrics={"series": {"periods": s.periods, "n_periods": 4,
                                "trend_enabled": False}},
        )
        out = render_console(result, series=s)
        self.assertNotIn("趋势区间", out, "引擎没跑趋势，控制台不许显示趋势区间")


class TestQualityDrift(unittest.TestCase):
    """同向恶化：捕捉单期够不上观察线、但一直在变坏的问题。"""

    def _drifting(self) -> StatementSeries:
        """毛利率/净利率/OCF/负债率/流动比率四期全部同向走坏，但每期幅度都很小。"""
        return mk_series(
            ("2022A", {"revenue": 100000.0, "cost_of_sales": 71000.0, "net_profit": 8600.0,
                       "operating_cash_flow": 9800.0, "total_liabilities": 75000.0,
                       "total_assets": 150000.0, "current_assets": 78000.0,
                       "current_liabilities": 45000.0}),
            ("2023A", {"revenue": 104000.0, "cost_of_sales": 74900.0, "net_profit": 8100.0,
                       "operating_cash_flow": 8400.0, "total_liabilities": 80100.0,
                       "total_assets": 154500.0, "current_assets": 77000.0,
                       "current_liabilities": 47500.0}),
            ("2024A", {"revenue": 108000.0, "cost_of_sales": 79200.0, "net_profit": 7300.0,
                       "operating_cash_flow": 6800.0, "total_liabilities": 86600.0,
                       "total_assets": 159500.0, "current_assets": 75500.0,
                       "current_liabilities": 50500.0}),
            ("2025A", {"revenue": 112000.0, "cost_of_sales": 84000.0, "net_profit": 6200.0,
                       "operating_cash_flow": 5100.0, "total_liabilities": 94500.0,
                       "total_assets": 165000.0, "current_assets": 73500.0,
                       "current_liabilities": 54000.0}),
        )

    def test_fires_on_sustained_multi_indicator_worsening(self):
        out = RuleEngine().run(
            self._drifting().latest, self._drifting().previous, series=self._drifting()
        )
        drift = [f for f in out if f.rule_id == "QUALITY_DRIFT"]
        self.assertEqual(len(drift), 1, "四期同向恶化应命中 QUALITY_DRIFT")
        self.assertEqual(drift[0].level, Level.HIGH)
        self.assertIn("毛利率", drift[0].detail)

    def test_does_not_fire_when_directions_conflict(self):
        """指标方向不一致时不构成"恶化通道"——不能为了报而报。"""
        s = mk_series(
            ("2022A", {"revenue": 100000.0, "cost_of_sales": 71000.0, "net_profit": 8000.0,
                       "operating_cash_flow": 9000.0, "total_liabilities": 70000.0,
                       "total_assets": 150000.0, "current_assets": 80000.0,
                       "current_liabilities": 40000.0}),
            ("2023A", {"revenue": 104000.0, "cost_of_sales": 73000.0, "net_profit": 8400.0,
                       "operating_cash_flow": 9600.0, "total_liabilities": 71000.0,
                       "total_assets": 154000.0, "current_assets": 82000.0,
                       "current_liabilities": 41000.0}),
            ("2024A", {"revenue": 108000.0, "cost_of_sales": 75000.0, "net_profit": 8800.0,
                       "operating_cash_flow": 10200.0, "total_liabilities": 72000.0,
                       "total_assets": 158000.0, "current_assets": 84000.0,
                       "current_liabilities": 42000.0}),
        )
        out = RuleEngine().run(s.latest, s.previous, series=s)
        self.assertFalse(any(f.rule_id == "QUALITY_DRIFT" for f in out))

    def test_flat_series_does_not_fire(self):
        """持平不算改善也不算恶化——四期完全一致不应触发。"""
        flat = dict(revenue=100000.0, cost_of_sales=71000.0, net_profit=8000.0,
                    operating_cash_flow=9000.0, total_liabilities=70000.0,
                    total_assets=150000.0, current_assets=80000.0,
                    current_liabilities=40000.0)
        s = mk_series(("2022A", flat), ("2023A", dict(flat)),
                      ("2024A", dict(flat)), ("2025A", dict(flat)))
        out = RuleEngine().run(s.latest, s.previous, series=s)
        self.assertFalse(any(f.rule_id == "QUALITY_DRIFT" for f in out))

    def test_real_four_period_fixture(self):
        """长恒电子四期数据：验证真实 fixture 上的端到端行为。"""
        paths = [
            STMT_DIR / f"长恒电子_{p}.json" for p in ("2022A", "2023A", "2024A", "2025A")
        ]
        s = build_series(paths)
        out = RuleEngine().run(s.latest, s.previous, series=s)
        drift = [f for f in out if f.rule_id == "QUALITY_DRIFT"]
        self.assertTrue(drift, "长恒电子应触发同向恶化")
        self.assertEqual(drift[0].evidence["score"], 9.0,
                         "五项全坏得分应为 3+2+2+1+1=9")


class TestQualityDriftWeights(unittest.TestCase):
    """重要性权重：等权计数的隐含假设"五项一样重要"不成立。

    等权时代（trend_min_agree=3）两类组合被判成同一严重度：
      A. 毛利率 + 净利率齐坏        —— 两个盈利维度同向走坏，强信号，却不触发（才 2 项）
      B. 现金流 + 杠杆 + 流动性齐坏  —— 三个弱指标凑数，触发
    加权后 A 得分 5 触发、B 得分 4 不触发。下面每条测试钉一个侧面。
    """

    # 四期健康基座：毛利率 29%、净利率 8.6%、OCF/净利 1.14、负债率 50%、流动比率 1.73
    BASE = dict(revenue=100000.0, cost_of_sales=71000.0, net_profit=8600.0,
                operating_cash_flow=9800.0, total_liabilities=75000.0,
                total_assets=150000.0, current_assets=78000.0,
                current_liabilities=45000.0)

    def _series(self, *changes: dict) -> StatementSeries:
        """四期序列：首期用 BASE，之后每期叠加 changes 里的科目覆盖。"""
        specs = [("2022A", dict(self.BASE))]
        for i, ch in enumerate(changes):
            specs.append((f"202{3 + i}A", {**self.BASE, **ch}))
        return mk_series(*specs)

    def _drift(self, s: StatementSeries):
        out = RuleEngine().run(s.latest, s.previous, series=s)
        return [f for f in out if f.rule_id == "QUALITY_DRIFT"]

    def test_strong_pair_fires_weak_triple_does_not(self):
        """核心行为：毛利率+净利率（5 分）触发；现金流+杠杆+流动性（4 分）不触发。"""
        # 毛利率、净利率逐年走坏，其余三项持平或改善
        strong = self._series(
            {"cost_of_sales": 72000.0, "net_profit": 8200.0},
            {"cost_of_sales": 73000.0, "net_profit": 7800.0},
            {"cost_of_sales": 74000.0, "net_profit": 7400.0},
        )
        drift = self._drift(strong)
        self.assertEqual(len(drift), 1, "两个盈利维度齐坏应触发")
        self.assertEqual(drift[0].evidence["score"], 5.0)
        self.assertEqual(drift[0].evidence["worsening"], ["毛利率", "净利率"])
        self.assertEqual(drift[0].evidence["weights"], {"毛利率": 3.0, "净利率": 2.0})
        self.assertIn("门槛 5", drift[0].detail)
        self.assertIn("权重", drift[0].detail)

        # 净利率、负债率、流动比率逐年走坏，毛利率与 OCF/净利改善或持平
        weak = self._series(
            {"net_profit": 8200.0, "total_liabilities": 77000.0,
             "current_assets": 77000.0, "current_liabilities": 46000.0},
            {"net_profit": 7800.0, "total_liabilities": 79000.0,
             "current_assets": 75500.0, "current_liabilities": 47500.0},
            {"net_profit": 7400.0, "total_liabilities": 81000.0,
             "current_assets": 74000.0, "current_liabilities": 49000.0},
        )
        self.assertFalse(self._drift(weak),
                         "三项弱指标凑数（4 分）不得触发——等权时代会误触发")

    def test_single_indicator_never_fires_regardless_of_weight(self):
        """结构性底线：单项指标再重也不算"多项同向"。

        只坏毛利率（权重 3，最高）：可能是定价策略调整等单一因素，
        跨指标 corroboration 才是这条规则的存在理由。
        """
        only_gm = self._series(
            {"cost_of_sales": 72000.0},
            {"cost_of_sales": 73000.0},
            {"cost_of_sales": 74000.0},
        )
        self.assertFalse(self._drift(only_gm), "单项毛利率连降不得触发")

    def test_equal_weights_reproduce_old_counting(self):
        """把权重全部设 1、门槛设 3，行为回到等权计数时代。

        证明新参数化是旧行为的超集：不是只能加权，而是可以配出等权。
        """
        rb = RuleBook(
            quality_drift_weights={n: 1.0 for n, _w, _d, _g in QUALITY_DRIFT_INDICATORS},
            quality_drift_min_score=3.0,
        )
        weak = self._series(
            {"net_profit": 8200.0, "total_liabilities": 77000.0,
             "current_assets": 77000.0, "current_liabilities": 46000.0},
            {"net_profit": 7800.0, "total_liabilities": 79000.0,
             "current_assets": 75500.0, "current_liabilities": 47500.0},
            {"net_profit": 7400.0, "total_liabilities": 81000.0,
             "current_assets": 74000.0, "current_liabilities": 49000.0},
        )
        out = RuleEngine(book=rb).run(weak.latest, weak.previous, series=weak)
        drift = [f for f in out if f.rule_id == "QUALITY_DRIFT"]
        self.assertEqual(len(drift), 1, "等权 3/5 时三项弱指标应触发（旧行为）")
        self.assertEqual(drift[0].evidence["score"], 3.0)

    def test_custom_weights_reorder_disclosure(self):
        """自定义权重后，披露顺序跟随权重降序，不跟随指标表顺序。"""
        rb = RuleBook(
            quality_drift_weights={
                "毛利率": 1.0, "净利率": 1.0, "经营现金流/净利润": 3.0,
                "资产负债率": 1.0, "流动比率": 1.0,
            },
            quality_drift_min_score=4.0,
        )
        # OCF/净利 + 负债率 + 流动比率齐坏：等权下才 3 分，自定义后 3+1+1=5 分
        s = self._series(
            {"operating_cash_flow": 8400.0, "total_liabilities": 77000.0,
             "current_assets": 77000.0, "current_liabilities": 46000.0},
            {"operating_cash_flow": 6800.0, "total_liabilities": 79000.0,
             "current_assets": 75500.0, "current_liabilities": 47500.0},
            {"operating_cash_flow": 5100.0, "total_liabilities": 81000.0,
             "current_assets": 74000.0, "current_liabilities": 49000.0},
        )
        out = RuleEngine(book=rb).run(s.latest, s.previous, series=s)
        drift = [f for f in out if f.rule_id == "QUALITY_DRIFT"]
        self.assertEqual(len(drift), 1)
        self.assertEqual(drift[0].evidence["worsening"],
                         ["经营现金流/净利润", "资产负债率", "流动比率"],
                         "披露顺序应按权重降序")

    def test_default_weights_follow_documented_ranking(self):
        """默认权重表即文档承诺的重要性排序。"""
        self.assertEqual(
            list(RuleBook().quality_drift_weights),
            ["毛利率", "净利率", "经营现金流/净利润", "资产负债率", "流动比率"],
        )
        self.assertEqual(RuleBook().quality_drift_min_score, 5.0)

    def test_invalid_weight_config_raises(self):
        """键对不上、权重非正、门槛非正都要当场炸。"""
        full = {n: 1.0 for n, _w, _d, _g in QUALITY_DRIFT_INDICATORS}
        with self.assertRaises(ValueError):      # 少一个键
            RuleBook(quality_drift_weights={k: v for k, v in full.items()
                                            if k != "毛利率"})
        with self.assertRaises(ValueError):      # 多一个键
            RuleBook(quality_drift_weights={**full, "商誉/净资产": 1.0})
        with self.assertRaises(ValueError):      # 零权重
            RuleBook(quality_drift_weights={**full, "毛利率": 0.0})
        with self.assertRaises(ValueError):      # 负门槛
            RuleBook(quality_drift_min_score=-1.0)


class TestSeriesIntegration(unittest.TestCase):
    """多期输入在编排层与渲染层的贯通。"""

    def setUp(self):
        self.paths = [
            STMT_DIR / f"长恒电子_{p}.json" for p in ("2022A", "2023A", "2024A", "2025A")
        ]
        self.series = build_series(self.paths)
        self.result = Orchestrator().analyze(
            self.series.latest, self.series.previous, series=self.series,
            company="长恒电子",
        )

    def test_metrics_carry_series_info(self):
        self.assertEqual(self.result.metrics["series"]["n_periods"], 4)
        self.assertTrue(self.result.metrics["series"]["trend_enabled"])
        self.assertEqual(self.result.metrics["series"]["periods"],
                         ["2022A", "2023A", "2024A", "2025A"])

    def test_console_shows_trend_span(self):
        out = render_console(self.result, series=self.series)
        self.assertIn("趋势区间", out)
        self.assertIn("2022A", out)
        self.assertIn("2025A", out)

    def test_markdown_has_trend_section(self):
        md = render_markdown(self.result, self.series.latest, self.series.previous,
                             series=self.series)
        self.assertIn("多期趋势", md)
        self.assertIn("趋势口径", md)

    def test_markdown_trend_table_lists_all_periods(self):
        md = render_markdown(self.result, self.series.latest, self.series.previous,
                             series=self.series)
        for p in self.series.periods:
            self.assertIn(p, md, f"趋势表缺少期间 {p}")

    def test_two_period_render_has_no_trend_section(self):
        """两期报告不得出现趋势章节——避免给读者"分析了趋势"的错觉。"""
        cur, prev = load_pair(STMT_DIR / "示例制造_2025A.json",
                              STMT_DIR / "示例制造_2024A.json")
        r = Orchestrator().analyze(cur, prev, company="示例制造股份")
        md = render_markdown(r, cur, prev)
        self.assertNotIn("多期趋势", md)
        self.assertNotIn("趋势口径", md)

    def test_series_analysis_is_deterministic(self):
        a = Orchestrator().analyze(self.series.latest, self.series.previous,
                                   series=self.series).to_dict()
        b = Orchestrator().analyze(self.series.latest, self.series.previous,
                                   series=self.series).to_dict()
        self.assertEqual(a, b, "多期分析必须可复算")


# ---------------------------------------------------------------- 提取层


class TestExtractAgent(unittest.TestCase):
    def test_parse_text_basic(self):
        text = "营业收入：125,000\n净利润 8,200\n应收账款：21,000"
        st = ExtractAgent(llm=NullLLM()).parse_text(text, "2025A")
        self.assertEqual(st.get("revenue"), 125000.0)
        self.assertEqual(st.get("net_profit"), 8200.0)
        self.assertEqual(st.get("accounts_receivable"), 21000.0)

    def test_unrecognized_lines_recorded_not_dropped(self):
        """无法识别的行必须留痕——静默丢弃是财务数据处理的严重缺陷。"""
        text = "营业收入：1000\n【附注】本表未经审计\n其他综合收益 500"
        st = ExtractAgent(llm=NullLLM()).parse_text(text, "2025A")
        self.assertEqual(len(st.meta["unrecognized"]), 2)
        self.assertIn("【附注】本表未经审计", st.meta["unrecognized"])

    def test_negative_numbers(self):
        st = ExtractAgent(llm=NullLLM()).parse_text("净利润 -3,500", "2025A")
        self.assertEqual(st.get("net_profit"), -3500.0)

    def test_line_with_unit_suffix_is_parsed(self):
        """真实报表行末几乎都带单位。带单位就解析不出来 = 整个解析器形同虚设。

        这是实测发现的真 bug：初版正则要求整行以数字结尾，导致
        "营业收入 1,000.00 万元" 整行被判为未识别。
        """
        text = "营业收入 1,000.00 万元\n应收账款 300.00 万元\n存货 200.00 万元"
        st = ExtractAgent(llm=NullLLM()).parse_text(text, "2025A")
        self.assertEqual(st.get("revenue"), 1000.0)
        self.assertEqual(st.get("accounts_receivable"), 300.0)
        self.assertEqual(st.get("inventory"), 200.0)
        self.assertEqual(st.meta["unrecognized"], [], "带单位的标准行不应进入未识别列表")

    def test_unit_conversion_to_wan_yuan(self):
        """口径统一为万元：亿元 × 10000，千元 × 0.1，元 × 0.0001。"""
        st = ExtractAgent(llm=NullLLM()).parse_text(
            "资产总计 1.80 亿元\n负债合计 500 千元\n商誉 3,000,000 元", "2025A"
        )
        self.assertAlmostEqual(st.get("total_assets"), 18000.0)
        self.assertAlmostEqual(st.get("total_liabilities"), 50.0)
        self.assertAlmostEqual(st.get("goodwill"), 300.0)

    def test_alias_longest_match_wins(self):
        """最长别名优先：'营业总成本' 不能因为含 '营业成本' 的短别名而被误吃到 cost_of_sales。

        这里两个别名实际指向同一 key，所以断言的是不抛错且取到正确 key；
        真正的风险是长名被更短的其他 key 别名截胡，故同时验证 '归属于母公司股东的净利润'。
        """
        st = ExtractAgent(llm=NullLLM()).parse_text(
            "营业总成本 900 万元\n归属于母公司股东的净利润 120 万元", "2025A"
        )
        self.assertEqual(st.get("cost_of_sales"), 900.0)
        self.assertEqual(st.get("net_profit"), 120.0)

    def test_percentage_line_does_not_leak_into_items(self):
        """带 % 的行应被识别为对应科目（数值原样），单位 % 不参与金额换算。"""
        st = ExtractAgent(llm=NullLLM()).parse_text("销售费用 84.6 %", "2025A")
        self.assertAlmostEqual(st.get("selling_expense"), 84.6)


# ---------------------------------------------------------------- 编排层


class TestOrchestrator(unittest.TestCase):
    def test_full_pipeline_runs(self):
        cur = mk("2025A", revenue=1080.0, cost_of_sales=826.0, net_profit=63.0,
                 accounts_receivable=345.0, inventory=240.0,
                 total_assets=1950.0, total_liabilities=1430.0, total_equity=520.0,
                 net_assets=520.0, operating_cash_flow=24.0, goodwill=210.0)
        cur.meta["company"] = "测试主体"
        prev = mk("2024A", revenue=1000.0, cost_of_sales=720.0, net_profit=82.0,
                  accounts_receivable=210.0, inventory=150.0,
                  total_assets=1680.0, total_liabilities=920.0, total_equity=760.0,
                  net_assets=760.0, operating_cash_flow=91.0, goodwill=42.0)
        r = Orchestrator().analyze(cur, prev)
        self.assertEqual(r.company, "测试主体")
        self.assertGreater(len(r.findings), 0, "应命中多项异常")
        self.assertTrue(r.summary)
        self.assertTrue(r.reasoning)
        self.assertGreater(len(r.trace), 0, "应有执行轨迹")
        self.assertGreater(r.risk_score(), 0.0)

    def test_no_llm_still_produces_report(self):
        """零依赖原则：没有 LLM 也必须产出完整报告。"""
        cur = mk("2025A", revenue=1000.0, cost_of_sales=700.0, net_profit=100.0)
        prev = mk("2024A", revenue=1000.0, cost_of_sales=700.0, net_profit=100.0)
        r = Orchestrator().analyze(cur, prev)
        self.assertTrue(r.summary)
        self.assertTrue(r.reasoning)

    def test_clean_statements_produce_no_findings(self):
        """对照期数据应基本干净——若健康数据也报一堆异常，说明阈值失准。"""
        cur = mk("2025A", revenue=1050.0, cost_of_sales=756.0, net_profit=88.0,
                 accounts_receivable=220.0, inventory=155.0,
                 total_assets=1700.0, total_liabilities=930.0, total_equity=770.0,
                 net_assets=770.0, operating_cash_flow=95.0, goodwill=42.0)
        prev = mk("2024A", revenue=1000.0, cost_of_sales=720.0, net_profit=82.0,
                  accounts_receivable=210.0, inventory=150.0,
                  total_assets=1680.0, total_liabilities=920.0, total_equity=760.0,
                  net_assets=760.0, operating_cash_flow=91.0, goodwill=42.0)
        r = Orchestrator().analyze(cur, prev)
        self.assertEqual(r.high_count, 0, f"健康数据不应有重大异常，实际：{[f.title for f in r.findings]}")


# ---------------------------------------------------------------- 渲染层


class TestRender(unittest.TestCase):
    def setUp(self):
        self.cur, self.prev = load_pair(
            STMT_DIR / "示例制造_2025A.json",
            STMT_DIR / "示例制造_2024A.json",
        )
        self.result = Orchestrator().analyze(self.cur, self.prev, company="示例制造股份")

    def test_markdown_has_all_sections(self):
        md = render_markdown(self.result, self.cur, self.prev)
        for section in ["分析概览", "关键财务指标", "异常项明细", "成因研判", "执行轨迹"]:
            self.assertIn(section, md, f"报告缺少章节：{section}")

    def test_markdown_deterministic(self):
        """同样的输入必须产出完全一致的报告——可复算性要求。"""
        a = render_markdown(self.result, self.cur, self.prev)
        b = render_markdown(self.result, self.cur, self.prev)
        self.assertEqual(a, b)

    def test_console_render_nonempty(self):
        out = render_console(self.result)
        self.assertIn("示例制造股份", out)
        self.assertIn("风险分", out)

    def test_json_roundtrip(self):
        d = self.result.to_dict()
        self.assertIn("findings", d)
        self.assertIn("risk_score", d)
        for f in d["findings"]:
            self.assertIn("level_label", f, "序列化后应带可读严重度标签")


# ---------------------------------------------------------------- 研判质量
#
# 这里是上一版最容易糊弄过去的地方：模板把 findings 的 detail 原样复述一遍
# 就当作"成因研判"，报告篇幅上去了但信息量为零。以下测试把这种行为钉死。


class TestReasoningQuality(unittest.TestCase):
    def setUp(self):
        cur, prev = load_pair(
            STMT_DIR / "示例制造_2025A.json",
            STMT_DIR / "示例制造_2024A.json",
        )
        self.result = Orchestrator().analyze(cur, prev, company="示例制造股份")

    def test_reasoning_does_not_restate_details(self):
        """研判不能是明细的复述——否则第四章与第三章重复，是无效篇幅。"""
        reasoning = self.result.reasoning
        for f in self.result.findings:
            self.assertNotIn(
                f.detail, reasoning,
                f"研判章节原样复制了 {f.rule_id} 的明细文本，应改为跨发现归因",
            )

    def test_reasoning_has_attribution_structure(self):
        """研判必须回答'这些异常归成哪几条线索'，而不是逐条罗列。"""
        r = self.result.reasoning
        self.assertIn("主线索归因", r)
        self.assertIn("建议核查顺序", r)
        self.assertIn("对判断的整体影响", r)

    def test_reasoning_clusters_cover_most_rules(self):
        """刻意植入的异常应能落入既有线索族，'其他'不应成为主要成分。"""
        reasoning = self.result.reasoning
        self.assertTrue(
            "收入质量" in reasoning or "资产质量" in reasoning,
            "至少应识别出收入质量或资产质量中的一条线索",
        )
        hit_ids = {f.rule_id for f in self.result.findings}
        # AR_GAP/OCF_DIVERGE/MARGIN_EROSION 三条应聚成收入质量线索
        self.assertGreaterEqual(len(hit_ids & {"AR_GAP", "OCF_DIVERGE", "MARGIN_EROSION"}), 2)

    def test_reasoning_stable_across_runs(self):
        cur, prev = load_pair(
            STMT_DIR / "示例制造_2025A.json", STMT_DIR / "示例制造_2024A.json"
        )
        r1 = Orchestrator().analyze(cur, prev).reasoning
        r2 = Orchestrator().analyze(cur, prev).reasoning
        self.assertEqual(r1, r2, "确定性研判不应有随机成分")

    def test_empty_findings_reasoning_is_clean(self):
        cur = mk("2025A", revenue=1050.0, cost_of_sales=756.0, net_profit=88.0,
                 accounts_receivable=220.0, inventory=155.0,
                 total_assets=1700.0, total_liabilities=930.0, total_equity=770.0,
                 net_assets=770.0, operating_cash_flow=95.0, goodwill=42.0)
        prev = mk("2024A", revenue=1000.0, cost_of_sales=720.0, net_profit=82.0,
                  accounts_receivable=210.0, inventory=150.0,
                  total_assets=1680.0, total_liabilities=920.0, total_equity=760.0,
                  net_assets=760.0, operating_cash_flow=91.0, goodwill=42.0)
        r = Orchestrator().analyze(cur, prev)
        self.assertEqual(r.findings, [])
        self.assertIn("未发现异常", r.reasoning)


# ---------------------------------------------------------------- 端到端
#
# 评测集：对刻意植入的异常做 gold 标注，验证规则引擎的召回。
# 与项目一的 eval_set.json 同一思路——没有标注集，"我改了规则"就无法量化。


EXPECTED_RULES_2025A = {"AR_GAP", "INV_GAP", "OCF_DIVERGE", "DEBT_RATIO", "GOODWILL", "MARGIN_EROSION", "SWING_SCAN"}


class TestEvalSet(unittest.TestCase):
    """规则召回验证：植入的异常必须全部被命中。"""

    def setUp(self):
        cur, prev = load_pair(
            STMT_DIR / "示例制造_2025A.json",
            STMT_DIR / "示例制造_2024A.json",
        )
        self.result = Orchestrator().analyze(cur, prev, company="示例制造股份")
        self.hit_rules = {f.rule_id for f in self.result.findings}

    def test_recall_of_planted_anomalies(self):
        missing = EXPECTED_RULES_2025A - self.hit_rules
        self.assertFalse(missing, f"未命中 gold 标注的规则：{sorted(missing)}")

    def test_no_unexpected_high_severity(self):
        """不该出现的重大异常要暴露出来——防误报。"""
        unexpected = {f.rule_id for f in self.result.findings if f.level == Level.HIGH} - EXPECTED_RULES_2025A
        self.assertFalse(unexpected, f"出现 gold 之外的重大异常：{sorted(unexpected)}")

    def test_balance_sheet_identity_not_flagged(self):
        """测试数据勾稽是平的，不应报数据可信度问题。"""
        self.assertNotIn("BS_IDENTITY", self.hit_rules)


# ---------------------------------------------------------------- 行业基准层

class TestIndustryBenchmark(unittest.TestCase):
    """单个指标的同行分布。"""

    def setUp(self):
        self.b = IndustryBenchmark(
            industry="工程机械", metric="ar_growth_gap", n_peers=42,
            percentiles={"p50": 6.0, "p75": 12.0, "p90": 20.0},
        )

    def test_at_returns_exact_quantile(self):
        self.assertEqual(self.b.at("p75"), 12.0)

    def test_at_unknown_quantile_returns_none(self):
        self.assertIsNone(self.b.at("p99"))

    def test_available_quantiles_is_sorted_property(self):
        self.assertEqual(self.b.available_quantiles, ["p50", "p75", "p90"])

    def test_to_dict_roundtrip(self):
        back = IndustryBenchmark.from_dict(self.b.to_dict())
        self.assertEqual(back.at("p90"), 20.0)
        self.assertEqual(back.n_peers, 42)
        self.assertEqual(back.industry, "工程机械")


class TestBenchmarkBook(unittest.TestCase):
    """基准簿：查询、样本量门槛、容错加载。"""

    @classmethod
    def setUpClass(cls):
        cls.book = load_benchmarks(BENCH_PATH)

    def test_fixture_loaded(self):
        self.assertTrue(self.book)
        self.assertIn("工程机械", self.book.industries())

    def test_get_returns_none_for_unknown_industry(self):
        self.assertIsNone(self.book.get("航天军工", "ar_growth_gap"))

    def test_get_returns_none_below_min_peers(self):
        """3 个同行的分位数是噪声，不是基准。"""
        self.assertIsNone(self.book.get("小众仪器制造", "ar_growth_gap"))
        self.assertEqual(self.book.get("工程机械", "ar_growth_gap").n_peers, 42)

    def test_line_calibrates_from_industry(self):
        line, src = self.book.line("工程机械", "ar_growth_gap", 20.0)
        self.assertEqual(line, 12.0)
        self.assertEqual(src, "industry:p75")

    def test_line_falls_back_to_absolute(self):
        line, src = self.book.line("航天军工", "ar_growth_gap", 20.0)
        self.assertEqual((line, src), (20.0, "absolute"))

    def test_line_falls_back_when_metric_missing(self):
        line, src = self.book.line("工程机械", "no_such_metric", 20.0)
        self.assertEqual((line, src), (20.0, "absolute"))

    def test_loader_tolerates_corrupt_entry(self):
        """单个行业损坏不该让整本簿加载失败——参照数据是增强项。"""
        import json
        import tempfile

        bad = {
            "schema": "finagent-industry-benchmarks/1",
            "min_peers": 5,
            "industries": {
                "坏行业": {"n_peers": 10, "metrics": "这不是字典"},
                "工程机械": {
                    "n_peers": 42,
                    "metrics": {"ar_growth_gap": {"p75": 12.0}},
                },
            },
        }
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as fh:
            json.dump(bad, fh, ensure_ascii=False)
            tmp = fh.name
        try:
            book = load_benchmarks(tmp)
            self.assertIn("工程机械", book.industries())
            self.assertNotIn("坏行业", book.industries())
        finally:
            Path(tmp).unlink(missing_ok=True)


class TestBenchmarkedRules(unittest.TestCase):
    """规则接行业基准：两个方向都要成立，且不传时逐字不变。"""

    @classmethod
    def setUpClass(cls):
        cls.book = load_benchmarks(BENCH_PATH)
        cls.cur, cls.prev = load_pair(
            STMT_DIR / "恒润重工_2025A.json", STMT_DIR / "恒润重工_2024A.json"
        )

    def _run(self, benchmarks=None):
        return RuleEngine().run(self.cur, self.prev, benchmarks=benchmarks)

    def test_absolute_lines_produce_nothing_for_this_fixture(self):
        """恒润重工按全市场口径是干净的——这是它的设计前提。"""
        self.assertEqual(self._run(), [])

    def test_calibration_raises_sensitivity(self):
        """同行业口径下，两项进入最差四分之一，应当报出来。"""
        hits = {f.rule_id for f in self._run(self.book)}
        self.assertEqual(hits, {"AR_GAP", "DEBT_RATIO"})

    def test_calibrated_findings_are_warn_not_high(self):
        for f in self._run(self.book):
            self.assertEqual(f.level, Level.WARN)

    def test_evidence_discloses_line_source(self):
        """换了阈值必须留痕：读者要能回答"这个 12 是哪来的"。"""
        for f in self._run(self.book):
            self.assertEqual(f.evidence.get("line_source"), "industry:p75")
            self.assertIn("line_warn", f.evidence)

    def test_absolute_findings_record_absolute_source(self):
        cur, prev = load_pair(
            STMT_DIR / "示例制造_2025A.json", STMT_DIR / "示例制造_2024A.json"
        )
        # 示例制造没有 industry，即便传了基准簿也应走绝对线
        for f in RuleEngine().run(cur, prev, benchmarks=self.book):
            self.assertEqual(f.evidence.get("line_source", "absolute"), "absolute")

    def test_calibration_can_suppress_a_finding(self):
        """反方向：绝对线报警但行业常态更宽松时，不应报。

        建筑施工的应收剪刀差 p75 = 26pct，比全市场观察线 20pct 还松。
        gap 25pct 在绝对线下触发，在行业线下不触发。
        """
        cur = mk(
            "2025A",
            accounts_receivable=1250.0, revenue=1000.0,
            inventory=200.0, total_assets=2000.0,
            total_liabilities=400.0, total_equity=1600.0,
        )
        prev = mk(
            "2024A",
            accounts_receivable=1000.0, revenue=1000.0,
            inventory=200.0, total_assets=2000.0,
            total_liabilities=400.0, total_equity=1600.0,
        )
        cur.meta["industry"] = "建筑施工"
        f = rule_receivable_growth_gap(cur, prev, RB)
        self.assertIsNotNone(f, "绝对线下 gap 25pct 应触发")
        self.assertEqual(f.evidence["line_source"], "absolute")

        f2 = rule_receivable_growth_gap(cur, prev, RB, benchmarks=self.book)
        self.assertIsNone(f2, "建筑施工 p75=26pct，gap 25pct 属于行业常态，不该报")

    def test_high_line_comes_from_industry_p90(self):
        """重大线取行业 p90，不再按 abs_high/abs_warn 比例拍倍数。

        食品饮料实测 ar_growth_gap 的 p90 = 73.3，而比例线只有 47.3——
        按比例线会把行业尾部 10% 的公司判成"只到关注级"。
        工程机械 fixture：p50=6 / p75=12 / p90=20。
        """
        cur = mk(
            "2025A",
            accounts_receivable=2000.0, revenue=1000.0,
            inventory=200.0, total_assets=2000.0,
            total_liabilities=400.0, total_equity=1600.0,
        )
        prev = mk(
            "2024A",
            accounts_receivable=1000.0, revenue=1000.0,
            inventory=200.0, total_assets=2000.0,
            total_liabilities=400.0, total_equity=1600.0,
        )
        cur.meta["industry"] = "工程机械"
        f = rule_receivable_growth_gap(cur, prev, RB, benchmarks=self.book)
        self.assertEqual(f.level, Level.HIGH)          # gap 100% 远超 p90=20
        self.assertEqual(f.evidence["line_warn"], 12.0)
        self.assertEqual(f.evidence["line_high"], 20.0)
        self.assertEqual(f.evidence["line_source"], "industry:p75")
        self.assertEqual(f.evidence["line_high_source"], "industry:p90")
        self.assertEqual(f.evidence["industry_p50"], 6.0)
        self.assertEqual(f.evidence["industry_p90"], 20.0)

    def test_warn_band_is_p75_to_p90(self):
        """[p75, p90) 区间是关注级：比四分之三的同行极端，但没进最差十分之一。"""
        cur = mk(
            "2025A",
            accounts_receivable=1160.0, revenue=1000.0,   # gap = 16pct
            inventory=200.0, total_assets=2000.0,
            total_liabilities=400.0, total_equity=1600.0,
        )
        prev = mk(
            "2024A",
            accounts_receivable=1000.0, revenue=1000.0,
            inventory=200.0, total_assets=2000.0,
            total_liabilities=400.0, total_equity=1600.0,
        )
        cur.meta["industry"] = "工程机械"
        f = rule_receivable_growth_gap(cur, prev, RB, benchmarks=self.book)
        self.assertIsNotNone(f)
        self.assertEqual(f.level, Level.WARN, "16pct 在 p75=12 与 p90=20 之间")
        self.assertIn("中位 6.0", f.detail, "区间描述要把中位数一起给读者")
        self.assertIn("p90 20.0", f.detail)

    def test_high_line_falls_back_to_ratio_without_p90(self):
        """老簿子只有 p75 时，重大线退回比例推导，行为可预期。"""
        import json
        import tempfile

        thin = {
            "schema": "finagent-industry-benchmarks/1",
            "min_peers": 5,
            "industries": {
                "工程机械": {"n_peers": 42, "metrics": {"ar_growth_gap": {"p75": 12.0}}},
            },
        }
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as fh:
            json.dump(thin, fh, ensure_ascii=False)
            tmp = fh.name
        try:
            book = load_benchmarks(tmp)
            cur = mk(
                "2025A",
                accounts_receivable=2000.0, revenue=1000.0,
                inventory=200.0, total_assets=2000.0,
                total_liabilities=400.0, total_equity=1600.0,
            )
            prev = mk(
                "2024A",
                accounts_receivable=1000.0, revenue=1000.0,
                inventory=200.0, total_assets=2000.0,
                total_liabilities=400.0, total_equity=1600.0,
            )
            cur.meta["industry"] = "工程机械"
            f = rule_receivable_growth_gap(cur, prev, RB, benchmarks=book)
            self.assertEqual(f.evidence["line_warn"], 12.0)
            self.assertEqual(f.evidence["line_high"], 24.0, "比例线 = 12 × (40/20)")
            self.assertEqual(f.evidence["line_high_source"], "industry:p75x2.00")
            self.assertIsNone(f.evidence["industry_p50"])
            self.assertIsNone(f.evidence["industry_p90"])
            self.assertIn("p75 12.0", f.detail, "缺的分位不写，不补默认值")
            self.assertNotIn("中位", f.detail)
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_absolute_lines_have_no_band(self):
        """没校准上时 evidence 里不出现行业区间，来源全是 absolute。"""
        cur = mk(
            "2025A",
            accounts_receivable=1250.0, revenue=1000.0,
            inventory=200.0, total_assets=2000.0,
            total_liabilities=400.0, total_equity=1600.0,
        )
        prev = mk(
            "2024A",
            accounts_receivable=1000.0, revenue=1000.0,
            inventory=200.0, total_assets=2000.0,
            total_liabilities=400.0, total_equity=1600.0,
        )
        cur.meta["industry"] = "航天军工"          # 不在簿里
        f = rule_receivable_growth_gap(cur, prev, RB, benchmarks=self.book)
        self.assertEqual(f.evidence["line_source"], "absolute")
        self.assertEqual(f.evidence["line_high_source"], "absolute")
        self.assertEqual(f.evidence["line_high"], 40.0)
        self.assertIsNone(f.evidence["industry_p50"])
        self.assertNotIn("分位校准", f.detail)


# ---------------------------------------------------------------- 评分参数层

class TestScorePolicy(unittest.TestCase):
    """ScorePolicy：敏感性分析必须走真实代码路径。"""

    def _result(self, levels):
        findings = [Finding(rule_id=f"R{i}", level=lv, title=f"t{i}", detail="d") for i, lv in enumerate(levels)]
        return AnalysisResult(company="x", period="2025A", findings=findings)

    def test_default_policy_matches_legacy_numbers(self):
        """不传策略时，分数与改造前逐字一致。"""
        import math

        r = self._result([Level.HIGH] * 4 + [Level.WARN] * 3)
        raw = 4 * 10.0 + 3 * 4.0
        expect = (1 - math.exp(-raw / 20.0)) * 100
        self.assertAlmostEqual(r.risk_score(), expect, places=9)
        self.assertEqual(DEFAULT_POLICY.scale, 20.0)

    def test_larger_scale_lowers_score_monotonically(self):
        r = self._result([Level.HIGH] * 3)
        small = r.risk_score(ScorePolicy(scale=10.0))
        mid = r.risk_score(ScorePolicy(scale=20.0))
        large = r.risk_score(ScorePolicy(scale=50.0))
        self.assertGreater(small, mid)
        self.assertGreater(mid, large)

    def test_weights_change_ranking(self):
        r = self._result([Level.WARN] * 6)
        # 关注权重拉到 10，分数应逼近 6 项重大的情形
        highish = r.risk_score(ScorePolicy(w_warn=10.0, w_high=10.0))
        self.assertGreater(highish, 90.0)

    def test_grade_cutoffs_are_configurable(self):
        """6 项关注：raw=24，分数 69.88，默认切点下判「高」。"""
        r = self._result([Level.WARN] * 6)
        self.assertEqual(r.risk_grade(), "高")
        strict = r.risk_grade(ScorePolicy(grade_very_high=99.0, grade_high=95.0))
        self.assertEqual(strict, "中")

    def test_severity_floor_is_configurable(self):
        """1 项重大：默认地板是高；把地板抬到 2 项后回落到中。"""
        r = self._result([Level.HIGH])
        self.assertEqual(r.risk_grade(ScorePolicy(sev_high=1)), "高")
        self.assertEqual(r.risk_grade(ScorePolicy(sev_high=2)), "中")

    def test_policy_serializes_for_reporting(self):
        d = DEFAULT_POLICY.to_dict()
        self.assertEqual(d["scale"], 20.0)
        self.assertEqual(d["weights"]["high"], 10.0)
        self.assertEqual(d["severity_counts"]["high"], 1)


# ---------------------------------------------------------------- 底稿层

class TestWorkingPaper(unittest.TestCase):
    """审计底稿：结构完整、结论不被改写、序列化可往返。"""

    @classmethod
    def setUpClass(cls):
        cur, prev = load_pair(
            STMT_DIR / "示例制造_2025A.json", STMT_DIR / "示例制造_2024A.json"
        )
        cls.result = Orchestrator().analyze(cur, prev, company="示例制造股份")
        cls.wp = build_working_paper(
            cls.result, ref="A-3", prepared_by="张三", reviewed_by="李四"
        )

    def test_header_fields_complete(self):
        self.assertEqual(self.wp.ref, "A-3")
        self.assertEqual(self.wp.entity, "示例制造股份")
        self.assertEqual(self.wp.period, "2025A")
        self.assertEqual(self.wp.signoff.prepared_by, "张三")
        self.assertEqual(self.wp.signoff.reviewed_by, "李四")

    def test_procedures_cover_executed_work(self):
        codes = [p.code for p in self.wp.procedures]
        self.assertIn("P1", codes)
        self.assertIn("P2", codes)

    def test_findings_carried_over_verbatim(self):
        self.assertEqual(len(self.wp.findings), len(self.result.findings))
        self.assertEqual(
            [f.rule_id for f in self.wp.findings],
            [f.rule_id for f in self.result.findings],
        )

    def test_cross_references_present(self):
        refs = self.wp.index_refs()
        self.assertIn("A-3", refs)

    def test_conclusion_is_derived_not_rewritten(self):
        """底稿结论由分析结果原样派生，不另做加工。"""
        c = self.wp.conclusion
        self.assertIn("示例制造股份", c)
        self.assertIn("2025A", c)
        self.assertIn("7 项异常", c)
        self.assertIn("重大 4 项", c)
        self.assertIn("92.6", c)
        self.assertIn("极高", c)
        # 建议动作也必须与档位一致
        self.assertIn(self.result.risk_action(), c)

    def test_markdown_has_all_five_sections(self):
        md = render_workpaper_markdown(self.wp)
        for h in ("一、", "二、", "三、", "四、", "五、"):
            self.assertIn(h, md)
        self.assertIn("编制人", md)
        self.assertIn("复核人", md)

    def test_markdown_marks_high_severity(self):
        md = render_workpaper_markdown(self.wp)
        self.assertIn("重大", md)

    def test_to_dict_roundtrip(self):
        from finagent.workpaper import WorkingPaper

        back = WorkingPaper.from_dict(self.wp.to_dict())
        self.assertEqual(back.ref, self.wp.ref)
        self.assertEqual(len(back.findings), len(self.wp.findings))
        self.assertEqual(back.signoff.prepared_by, self.wp.signoff.prepared_by)

    def test_xlsx_renders_five_sheets(self):
        openpyxl = pytest_import("openpyxl")
        if openpyxl is None:
            self.skipTest("openpyxl 未安装")
        import tempfile

        from finagent.workpaper import render_workpaper_xlsx

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "wp.xlsx"
            render_workpaper_xlsx(self.wp, str(out))
            wb = openpyxl.load_workbook(out)
            self.assertEqual(
                wb.sheetnames, ["底稿首页", "审计过程", "审计发现", "交叉索引", "审计结论"]
            )
            self.assertEqual(wb["审计发现"].max_row, len(self.wp.findings) + 1)

    def test_trend_note_appears_when_series_given(self):
        series = build_series(
            [
                STMT_DIR / "示例制造_2024A.json",
                STMT_DIR / "示例制造_2025A.json",
            ]
        )
        wp = build_working_paper(self.result, series=series)
        p2 = [p for p in wp.procedures if p.code == "P2"][0]
        self.assertIn("趋势", p2.scope_ + p2.result + p2.method)

    def test_xlsx_requires_openpyxl_explicitly(self):
        """缺依赖必须明确报错，不能静默降级成 Markdown。

        静默降级的危险在于：调用方要的是表格文件，
        拿到的却是文本，还以为交付完成了。
        """
        import builtins

        from finagent import workpaper as wp_mod

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "openpyxl":
                raise ImportError("No module named 'openpyxl'")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = fake_import
        try:
            with self.assertRaises(RuntimeError) as ctx:
                wp_mod.render_workpaper_xlsx(self.wp, "/tmp/should_not_exist.xlsx")
        finally:
            builtins.__import__ = real_import
        self.assertIn("openpyxl", str(ctx.exception))
        self.assertIn("render_workpaper_markdown", str(ctx.exception))


# ---------------------------------------------------------------- A/B 指标层

class TestAbMetrics(unittest.TestCase):
    """A/B 对比的四类指标：保真、覆盖、复述、可复算。"""

    @classmethod
    def setUpClass(cls):
        cur, prev = load_pair(
            STMT_DIR / "示例制造_2025A.json", STMT_DIR / "示例制造_2024A.json"
        )
        cls.findings = Orchestrator().analyze(cur, prev).findings

    def test_extract_numbers_handles_cjk_and_signs(self):
        vals = extract_numbers("剪刀差 56.3pct，下降 -2.1 个百分点，共 7 项")
        self.assertIn(56.3, vals)
        self.assertIn(-2.1, vals)
        self.assertIn(7.0, vals)

    def test_number_pool_includes_evidence_and_counts(self):
        pool = number_pool(self.findings, n_high=4, n_warn=3)
        # 计数类数字必须合法，否则所有"重大 4 项"都会被判成幻觉
        self.assertIn(4.0, pool)
        self.assertIn(7.0, pool)
        # evidence 里的数字同样合法
        self.assertIn(64.3, pool)  # 应收同比

    def test_fidelity_accepts_grounded_text(self):
        text = "应收账款同比 64.3%，剪刀差 56.3 个百分点，共 7 项异常，重大 4 项。"
        rep = check_fidelity(text, self.findings, n_high=4, n_warn=3)
        self.assertEqual(rep.score, 1.0)
        self.assertEqual(rep.ungrounded, [])

    def test_fidelity_flags_fabricated_number(self):
        text = "整体风险敞口约为 87.4%，需引起高度重视。"
        rep = check_fidelity(text, self.findings, n_high=4, n_warn=3)
        self.assertLess(rep.score, 1.0)
        self.assertIn(87.4, rep.ungrounded)

    def test_coverage_counts_covered_titles(self):
        full = "\n".join(f.title for f in self.findings)
        rep = check_coverage(full, self.findings)
        self.assertEqual(rep.score, 1.0)
        self.assertEqual(check_coverage("完全无关的一段话", self.findings).score, 0.0)

    def test_restatement_detects_verbatim_copy(self):
        copy = "\n".join(f.detail for f in self.findings)
        self.assertGreater(check_restatement(copy, self.findings), 0.9)

    def test_restatement_near_zero_for_aggregation(self):
        agg = "本期异常集中在收入质量与现金流两个方向，建议按顺序核查原始凭证。"
        self.assertLess(check_restatement(agg, self.findings), 0.1)

    def test_determinism_ratio(self):
        """取最常见输出的占比，不是"是否与首次相同"。"""
        self.assertEqual(determinism_ratio(["a", "a", "a"]), 1.0)
        self.assertEqual(determinism_ratio(["a", "b"]), 0.5)
        self.assertEqual(determinism_ratio(["a", "a", "b"]), 2 / 3)
        self.assertEqual(determinism_ratio([]), 1.0)

    def test_evaluate_produces_comparable_row(self):
        """用真实模板研判器跑一遍，四个指标都应落在合法区间。"""
        agent = ReportAgent(llm=NullLLM())
        out = agent.reason("示例制造股份", "2025A", self.findings)
        self.assertTrue(out, "模板研判不应为空")
        row = evaluate("场景", "确定性模板", [out], self.findings)
        self.assertEqual(row.n_runs, 1)
        self.assertTrue(0.0 <= row.fidelity.score <= 1.0)
        self.assertTrue(0.0 <= row.coverage.score <= 1.0)
        self.assertTrue(0.0 <= row.restatement <= 1.0)
        self.assertGreater(row.length, 0)

    def test_template_reasoner_is_fully_grounded(self):
        """确定性模板的每个数字都必须来自 findings——这是它的立身之本。"""
        agent = ReportAgent(llm=NullLLM())
        out = agent.reason("示例制造股份", "2025A", self.findings)
        rep = check_fidelity(out, self.findings, n_high=4, n_warn=3)
        self.assertEqual(rep.score, 1.0, f"模板出现无依据数字：{rep.ungrounded}")


# 本地 Ollama qwen3:4b-instruct-2507-q4_K_M 对示例制造场景的真实输出
# （2026-10-16，temperature 0.1）。逐条覆盖了全部 7 项发现且数字全部有出处，
# 但旧版覆盖率检查器（精确标题匹配）对它报 0%——它把标题
# "应收账款增速显著快于营业收入" 改写成了 "应收账款增速远超营收"。
# 这段输出因此固化为回归样本：改写措辞不得被判成漏报。
_REAL_LLM_OUTPUT_07FINDINGS = """1. **应收账款增速远超营收**（+64.3% vs +8.0%）
   成因：信用政策放宽，客户赊账激增。
   核查：回款周期、客户信用评级、坏账计提。
   影响：回款风险高，利润兑现能力存疑。

2. **存货增速远超营收**（+60.0% vs +8.0%）
   成因：产销量下滑或库存管理失当。
   核查：存货结构、跌价准备、销售合同执行。
   影响：存货减值风险上升，资产质量恶化。

3. **经营现金流/净利润=0.38**
   成因：利润含大量非现金收益或回款延迟。
   核查：费用确认时点、应收款项回款情况。
   影响：盈余质量弱，现金流支撑不足。

4. **10科目波动超±50%**
   成因：单笔大额交易或会计政策调整。
   核查：原始凭证、费用归属、会计分录。
   影响：数据可信度存疑，需逐项验证。

5. **资产负债率73.33%**
   成因：债务扩张，或资产结构偏重有息负债。
   核查：债务结构、融资用途、偿债计划。
   影响：财务杠杆高，再融资压力大。

6. **商誉占净资产40.38%**
   成因：并购标的估值偏高或业绩未达预期。
   核查：并购对价、标的业绩承诺。
   影响：减值风险集中，盈利稳定性受威胁。

7. **净利率由8.20%降至5.83%**
   成因：成本上升或销售费用扩张。
   核查：成本结构、费用率变动。
   影响：盈利改善乏力，扩张不可可持续。"""


class TestCoverageParaphrase(unittest.TestCase):
    """覆盖率的同义改写路径：真实模型不会照抄标题。

    精确标题匹配对 stub 恒 100%（stub 按构造照抄），对真实模型恒 0——
    这个指标的主要用户群恰好被它漏掉。修复后三条路径：
    忽略空白的标题匹配 / 规则编号 / 独有汉字≥80% 或独有指标数字。
    """

    @classmethod
    def setUpClass(cls):
        cur, prev = load_pair(
            STMT_DIR / "示例制造_2025A.json", STMT_DIR / "示例制造_2024A.json"
        )
        cls.findings = Orchestrator().analyze(cur, prev).findings

    def test_real_llm_output_is_fully_covered(self):
        """回归样本：qwen3:4b 的真实输出逐条覆盖 7 项，指标必须认下来。"""
        rep = check_coverage(_REAL_LLM_OUTPUT_07FINDINGS, self.findings)
        self.assertEqual(rep.score, 1.0, f"被误判为漏报：{rep.missing}")

    def test_dropped_synonym_and_space_still_covered(self):
        """"5项…4期恶化" vs 标题 "5 项…4 期同向恶化"：空格与个别字词丢失算覆盖。"""
        drift = Finding(
            rule_id="QUALITY_DRIFT", level=Level.HIGH,
            title="5 项盈利质量指标连续 4 期同向恶化",
            detail="五项指标连续同向走坏。",
            evidence={"score": 9.0, "periods": ["2022A", "2025A"]},
        )
        other = Finding(
            rule_id="MARGIN_EROSION", level=Level.HIGH,
            title="营业收入增长但净利率下滑", detail="净利率下滑。",
        )
        text = "异常2：5项盈利质量指标连续4期恶化，反映经营效率下降。"
        rep = check_coverage(text, [drift, other])
        self.assertNotIn("QUALITY_DRIFT", rep.missing)
        self.assertIn("MARGIN_EROSION", rep.missing)

    def test_distinctive_number_path(self):
        """措辞整个改写、只留下指标数字时，独有数字命中也算覆盖。"""
        ocf = Finding(
            rule_id="OCF_DIVERGE", level=Level.WARN,
            title="经营现金流与净利润背离",
            detail="经营现金流/净利润 0.38 低于观察线 0.8。",
            evidence={"ocf_ratio": 0.38, "line_warn": 0.8},
        )
        debt = Finding(
            rule_id="DEBT_RATIO", level=Level.WARN,
            title="资产负债率处于高位",
            detail="资产负债率 73.33% 处于高位。",
            evidence={"debt_ratio": 73.33, "line_warn": 70.0},
        )
        # 只复述数字、完全不用标题措辞
        text = "现金利润转化率仅 0.38，杠杆水平 73.33%，均需关注。"
        rep = check_coverage(text, [ocf, debt])
        self.assertEqual(rep.score, 1.0, f"被误判为漏报：{rep.missing}")

    def test_shared_words_do_not_cross_cover(self):
        """共享词不构成证据："净利率下滑"不得覆盖"毛利率下滑"。"""
        gm = Finding(rule_id="GM", level=Level.WARN, title="毛利率下滑",
                     detail="毛利率由 29.0% 降至 26.0%。", evidence={"delta": -3.0})
        nm = Finding(rule_id="NM", level=Level.WARN, title="净利率下滑",
                     detail="净利率由 8.6% 降至 7.4%。", evidence={"delta": -1.2})
        rep = check_coverage("净利率由 8.6% 降至 7.4%，盈利走弱。", [gm, nm])
        self.assertIn("GM", rep.missing)
        self.assertNotIn("NM", rep.missing)

    def test_weak_numbers_do_not_count_as_evidence(self):
        """年份与计数类小整数没有区分度，不能作为"提及过"的证据。"""
        a = Finding(rule_id="A", level=Level.WARN, title="净利率下滑",
                    detail="净利率由 8.20% 降至 5.83%。",
                    evidence={"periods": ["2024A", "2025A"]})
        b = Finding(rule_id="B", level=Level.WARN, title="盈利质量指标连续同向恶化",
                    detail="多项指标连续同向走坏。", evidence={"score": 9.0})
        text = "2025A 期间共连续 3 期观察，未见异常。"
        rep = check_coverage(text, [a, b])
        self.assertIn("B", rep.missing, "报告期与计数类'3'证明不了讨论过 B")


def _load_fetch_module():
    """把 scripts/fetch_benchmarks.py 当模块加载。

    它是脚本不是包内模块（站点抓取逻辑不该进核心库），但解析与口径这些纯函数
    值得被测试钉住——尤其是金额格式与 gross_margin_delta 的绝对值口径。
    """
    import importlib.util

    path = ROOT / "scripts" / "fetch_benchmarks.py"
    spec = importlib.util.spec_from_file_location("finagent_fetch_benchmarks", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestBenchmarkFeed(unittest.TestCase):
    """fetch_benchmarks.py 的纯函数：解析、口径、分位数（零网络）。"""

    @classmethod
    def setUpClass(cls):
        cls.fb = _load_fetch_module()

    # --- 金额解析 ---

    def test_parse_amount_formats(self):
        p = self.fb.parse_amount
        self.assertEqual(p("4,237.02 亿"), 423702000000.0)
        self.assertEqual(p("42 亿"), 4200000000.0)          # 无小数位
        self.assertEqual(p("87,407,775.73"), 87407775.73)   # 小额给全精度
        self.assertEqual(p("-9.63 亿"), -963000000.0)       # 负值
        self.assertEqual(p("35,515,000"), 35515000.0)
        self.assertEqual(p("1.5 万亿"), 1.5e12)

    def test_parse_amount_null_markers(self):
        for s in ("—", "-", "", "   ", "N/A"):
            self.assertIsNone(self.fb.parse_amount(s))

    def test_parse_amount_rejects_unknown(self):
        """认不出的格式必须炸，不能静默变 None——基准会悄悄失真。"""
        for s in ("96.1%", "0.49 倍", "abc", "1.2 亿美元"):
            with self.assertRaises(ValueError):
                self.fb.parse_amount(s)

    # --- 行抽取 ---

    def test_extract_rows_takes_first_value_per_row(self):
        """科目名与值之间可能夹"注 ↗"链接，切段取首个数值单元格才对。"""
        html = (
            '<li id="row-INVENTORY" class="x"><span>存货</span>'
            '<a title="查看该科目年报附注原文" href="/notes">注</a>'
            '<span class="tabular-nums">945.26 亿</span></li>'
            '<li id="row-CONTRACT_ASSET"><span>合同资产</span>'
            '<span class="tabular-nums">—</span></li>'
        )
        rows = self.fb.extract_rows(html, {"INVENTORY", "CONTRACT_ASSET"})
        self.assertEqual(rows["INVENTORY"], "945.26 亿")
        self.assertEqual(rows["CONTRACT_ASSET"], "—")

    def test_extract_rows_ignores_rsc_duplicate(self):
        """页面同时含 SSR HTML 与 RSC payload，同一行出现两次，取首次。"""
        row = ('<li id="row-TOTAL_ASSETS"><span>资产总计</span>'
               '<span class="tabular-nums">106.91 亿</span></li>')
        html = row + "<script>" + row + "</script>"
        rows = self.fb.extract_rows(html, {"TOTAL_ASSETS"})
        self.assertEqual(rows["TOTAL_ASSETS"], "106.91 亿")

    # --- 指标口径 ---

    def test_gross_margin_delta_uses_absolute_value(self):
        """规则比的是 abs(delta)，基准就必须是绝对变动的分布。"""
        bs = {"TOTAL_ASSETS": "100 亿", "TOTAL_LIABILITIES": "50 亿",
              "ACCOUNTS_RECE": "10 亿", "INVENTORY": "20 亿"}
        is_cur = {"OPERATE_INCOME": "80 亿", "OPERATE_COST": "60 亿"}    # 毛利率 25%
        is_prev = {"OPERATE_INCOME": "80 亿", "OPERATE_COST": "64 亿"}   # 毛利率 20%
        m = self.fb.compute_metrics(bs, bs, is_cur, is_prev)
        self.assertAlmostEqual(m["gross_margin_delta"], 5.0, places=6)
        m2 = self.fb.compute_metrics(bs, bs, is_prev, is_cur)            # 反方向同正
        self.assertAlmostEqual(m2["gross_margin_delta"], 5.0, places=6)

    def test_growth_gap_math(self):
        bs_cur = {"TOTAL_ASSETS": "100 亿", "TOTAL_LIABILITIES": "50 亿",
                  "ACCOUNTS_RECE": "15 亿", "INVENTORY": "20 亿"}
        bs_prev = {"ACCOUNTS_RECE": "10 亿", "INVENTORY": "18 亿"}
        is_cur = {"OPERATE_INCOME": "80 亿", "OPERATE_COST": "60 亿"}
        is_prev = {"OPERATE_INCOME": "100 亿", "OPERATE_COST": "70 亿"}
        m = self.fb.compute_metrics(bs_cur, bs_prev, is_cur, is_prev)
        self.assertAlmostEqual(m["ar_growth_gap"], 70.0, places=6)       # +50% −(−20%)
        self.assertAlmostEqual(m["inv_growth_gap"], 31.111111, places=4)

    def test_materiality_floor_drops_immaterial_base(self):
        """基期应收只占基期营收 0.005% 时增速无经济意义，不进分布。"""
        bs_cur = {"TOTAL_ASSETS": "100 亿", "TOTAL_LIABILITIES": "50 亿",
                  "ACCOUNTS_RECE": "1000 万", "INVENTORY": "20 亿"}
        bs_prev = {"ACCOUNTS_RECE": "50 万", "INVENTORY": "18 亿"}
        is_cur = {"OPERATE_INCOME": "80 亿", "OPERATE_COST": "60 亿"}
        is_prev = {"OPERATE_INCOME": "100 亿", "OPERATE_COST": "70 亿"}
        m = self.fb.compute_metrics(bs_cur, bs_prev, is_cur, is_prev)
        self.assertNotIn("ar_growth_gap", m)
        self.assertIn("inv_growth_gap", m)      # 存货 18 亿是重大的，照算

    def test_missing_income_statement_leaves_debt_ratio(self):
        bs = {"TOTAL_ASSETS": "106.91 亿", "TOTAL_LIABILITIES": "78.3 亿"}
        m = self.fb.compute_metrics(bs, None, None, None)
        self.assertAlmostEqual(m["debt_ratio"], 73.239173, places=4)
        self.assertNotIn("gross_margin_delta", m)

    # --- 分位数 ---

    def test_percentile_matches_known_values(self):
        vals = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(self.fb.percentile(vals, 0), 1.0)
        self.assertEqual(self.fb.percentile(vals, 50), 2.5)
        self.assertEqual(self.fb.percentile(vals, 75), 3.25)
        self.assertEqual(self.fb.percentile(vals, 100), 4.0)
        self.assertIsNone(self.fb.percentile([], 75))
        self.assertEqual(self.fb.percentile([7.0], 75), 7.0)

    # --- 抽样与 ST ---

    def test_stride_sample_spans_roe_range(self):
        """/data 按 ROE 降序，取前 N 家会得到高 ROE 偏斜的样本。"""
        items = list(range(100))
        picked = self.fb.stride_sample(items, 10)
        self.assertEqual(len(picked), 10)
        self.assertIn(0, picked)
        self.assertIn(99, picked)
        self.assertEqual(self.fb.stride_sample(items, 0), items)
        self.assertEqual(self.fb.stride_sample(items, 200), items)

    def test_st_detection(self):
        for n in ("*ST西发", "ST绝味", "*ST春天"):
            self.assertTrue(self.fb.is_st(n))
        for n in ("贵州茅台", "东鹏饮料", "宁德时代"):
            self.assertFalse(self.fb.is_st(n))

    def test_excluded_st_codes_from_universe(self):
        """ST 排除名单必须从枚举全量算。crawl_industry 收到的是过滤后清单，
        在它内部统计会恒得到 []——"默认排除 ST"将在构建报告里无据可查
        （2026-10-16 实测：电子、食品饮料两个行业的 excluded_st 都是 []，
        而实际分别排除了 10 家与 11 家）。"""
        universe = [
            {"code": "600519", "name": "贵州茅台"},
            {"code": "600530", "name": "*ST西发"},
            {"code": "000752", "name": "ST绝味"},
        ]
        self.assertEqual(
            self.fb.excluded_st_codes(universe, include_st=False),
            ["600530", "000752"],
        )
        self.assertEqual(self.fb.excluded_st_codes(universe, include_st=True), [])

    # --- 现金流量表通道 ---

    def test_extract_all_rows_returns_rows_outside_bs_is(self):
        """extract_rows 只回调用方点名的行；cf 页面的行 ID 事先不确定，
        必须有一条全量抽取的通道，否则经营现金流永远取不到。"""
        html = (
            '<li id="row-OPERATE_INCOME"><span>营业收入</span>'
            '<span class="tabular-nums">324.73 亿</span></li>'
            '<li id="row-NETCASH_OPERATE"><span>经营活动产生的现金流量净额</span>'
            '<span class="tabular-nums">45.93 亿</span></li>'
        )
        all_rows = self.fb.extract_all_rows(html)
        self.assertEqual(all_rows["NETCASH_OPERATE"], "45.93 亿")
        # 对照：按 BS/IS 行 ID 过滤时，现金流行会被丢掉
        filtered = self.fb.extract_rows(html, self.fb.BS_ROWS)
        self.assertNotIn("NETCASH_OPERATE", filtered)

    def test_fetch_statement_cf_uses_cache_without_network(self):
        """kind="cf" 的缓存命名与解析通道。有缓存就不能发请求——
        这一点同时被"测试不联网"本身验证。"""
        import tempfile

        html = ('<li id="row-NETCASH_OPERATE"><span>经营活动产生的现金流量净额</span>'
                '<span class="tabular-nums">45.93 亿</span></li>')
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td)
            (raw / "600600_2025AR_cf.html").write_text(html, encoding="utf-8")
            rows = self.fb.fetch_statement("600600", "2025AR", False, raw, 0.0, kind="cf")
            self.assertEqual(rows["NETCASH_OPERATE"], "45.93 亿")

    def test_fetch_statement_bs_is_still_works(self):
        """重构后回归：原有 income 布尔参数的口径不能变。"""
        import tempfile

        bs = ('<li id="row-INVENTORY"><span>存货</span>'
              '<span class="tabular-nums">32.37 亿</span></li>')
        isr = ('<li id="row-OPERATE_INCOME"><span>营业收入</span>'
               '<span class="tabular-nums">324.73 亿</span></li>')
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td)
            (raw / "600600_2025AR_bs.html").write_text(bs, encoding="utf-8")
            (raw / "600600_2025AR_is.html").write_text(isr, encoding="utf-8")
            self.assertEqual(
                self.fb.fetch_statement("600600", "2025AR", False, raw, 0.0)["INVENTORY"],
                "32.37 亿",
            )
            self.assertEqual(
                self.fb.fetch_statement("600600", "2025AR", True, raw, 0.0)["OPERATE_INCOME"],
                "324.73 亿",
            )

    # --- 网络层健壮性 ---

    def test_incomplete_read_is_not_oserror(self):
        """钉住 2026-10 电子抓取崩溃的根因：IncompleteRead 的继承链是
        IncompleteRead → HTTPException → Exception，与 OSError 无关，
        `except (URLError, OSError)` 结构上接不住它。"""
        import http.client

        exc = http.client.IncompleteRead(b"7645 bytes read")
        self.assertIsInstance(exc, http.client.HTTPException)
        self.assertNotIsInstance(exc, OSError)

    def test_http_get_retries_incomplete_read(self):
        """站点偶发断流必须走重试，不能一次打挂整轮抓取。

        电子行业 1940 次请求约 95 分钟，曾因第 71 秒一次 IncompleteRead 全废。
        """
        import http.client
        import tempfile
        from unittest import mock

        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b"<html>ok</html>"

        state = {"n": 0}

        def flaky(url, timeout=None):
            state["n"] += 1
            if state["n"] < 3:
                raise http.client.IncompleteRead(b"7645 bytes read")
            return FakeResp()

        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "sub" / "x.html"
            with mock.patch.object(self.fb.urllib.request, "urlopen", side_effect=flaky), \
                 mock.patch.object(self.fb.time, "sleep"):
                body = self.fb.http_get("https://x.test/", cache, 0.0, retries=3)
            self.assertEqual(body, "<html>ok</html>")
            self.assertEqual(state["n"], 3)
            self.assertTrue(cache.exists())

    def test_http_get_gives_up_cleanly_after_retries(self):
        """重试耗尽要抛带上下文的 RuntimeError，不写缓存也不写 404 标记。"""
        import http.client
        import tempfile
        from unittest import mock

        def dead(url, timeout=None):
            raise http.client.IncompleteRead(b"7645 bytes read")

        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "x.html"
            with mock.patch.object(self.fb.urllib.request, "urlopen", side_effect=dead), \
                 mock.patch.object(self.fb.time, "sleep"):
                with self.assertRaises(RuntimeError) as ctx:
                    self.fb.http_get("https://x.test/", cache, 0.0, retries=3)
            self.assertIn("x.test", str(ctx.exception))
            self.assertIn("IncompleteRead", str(ctx.exception))
            self.assertFalse(cache.exists())
            self.assertFalse(Path(str(cache) + ".404").exists())


class TestFixtureBridge(unittest.TestCase):
    """build_fixture_from_raw.py：把爬虫缓存还原成 finagent statement。"""

    @classmethod
    def setUpClass(cls):
        import importlib.util

        path = ROOT / "scripts" / "build_fixture_from_raw.py"
        spec = importlib.util.spec_from_file_location("finagent_build_fixture", path)
        cls.br = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.br)

    def test_period_label_normalized(self):
        """站点用 2025AR（Annual Report），fixture 用 2025A，两者必须对齐，
        否则 period_sort_key 之后多期排序会出现不可预期的次序。"""
        self.assertEqual(self.br.build("x", "2025AR", "x", "")["period"], "2025A")
        self.assertEqual(self.br.build("x", "2024AR", "x", "")["period"], "2024A")
        self.assertEqual(self.br.build("x", "2025Q1", "x", "")["period"], "2025Q1")
        self.assertEqual(self.br.build("x", "2025A", "x", "")["period"], "2025A")

    def test_mapping_covers_every_consumed_item(self):
        """耦合不变量：模型与规则读到的每个科目，桥接表都必须能产出。

        这条测试钉的是"规则新增了一个科目消费，fixture 却悄悄没有"——
        那种缺失不会报错，只会让规则永远不触发，看报告完全看不出。
        """
        import re

        consumed: set[str] = set()
        for f in ("src/finagent/models.py", "src/finagent/rules.py"):
            src = (ROOT / f).read_text(encoding="utf-8")
            consumed |= set(re.findall(r'yoy\("([a-z_0-9]+)"', src))
            consumed |= set(re.findall(r'\.(?:get|has|require)\("([a-z_0-9]+)"', src))
        # 这些是 dict/属性/meta 访问，不是报表科目
        consumed -= {"meta", "items", "level", "period", "source", "note", "industry"}
        produced = set(self.br.BS_MAP.values()) | set(self.br.IS_MAP.values()) | set(
            self.br.CF_MAP.values()
        )
        gap = consumed - produced
        self.assertEqual(gap, set(), f"规则消费但桥接表未覆盖：{sorted(gap)}")

    def test_mapping_has_no_duplicate_target(self):
        """两个站点行映射到同一个科目会静默互相覆盖。"""
        seen: dict[str, str] = {}
        for mapping in (self.br.BS_MAP, self.br.IS_MAP, self.br.CF_MAP):
            for rid, field in mapping.items():
                self.assertNotIn(field, seen, f"{field} 被 {seen.get(field)} 与 {rid} 重复映射")
                seen[field] = rid

    def test_build_reports_missing_fields_honestly(self):
        """没有缓存又不想联网时，缺失要写进 meta.note，不能默认填 0。"""
        doc = self.br.build("999999", "2025AR", "不存在的公司", "")
        self.assertEqual(doc["items"], {})
        self.assertIn("bs(整表缺失)", doc["meta"]["note"])
        self.assertIn("is(整表缺失)", doc["meta"]["note"])
        self.assertIn("cf(整表缺失)", doc["meta"]["note"])


def pytest_import(name: str):
    """缺依赖时返回 None，让测试跳过而非报错。"""
    try:
        return __import__(name)
    except ImportError:
        return None


if __name__ == "__main__":
    unittest.main(verbosity=2)
