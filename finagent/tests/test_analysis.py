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
    load_pair,
    render_console,
    render_markdown,
)
from finagent.rules import (  # noqa: E402
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

STMT_DIR = ROOT / "data" / "statements"
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
        """档位边界必须与分数一致，避免出现'92 分却判低风险'这种自相矛盾。"""
        def grade(n_high: int, n_warn: int = 0) -> str:
            fs = [Finding(rule_id=f"H{i}", level=Level.HIGH, title="t", detail="d") for i in range(n_high)]
            fs += [Finding(rule_id=f"W{i}", level=Level.WARN, title="t", detail="d") for i in range(n_warn)]
            return AnalysisResult(company="X", period="T", findings=fs).risk_grade()

        self.assertEqual(grade(0), "低")          # 无发现
        self.assertEqual(grade(1), "中")          # 39.35
        self.assertEqual(grade(2), "中")          # 63.21
        self.assertEqual(grade(3), "高")          # 77.69
        self.assertEqual(grade(4), "极高")        # 86.47

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
