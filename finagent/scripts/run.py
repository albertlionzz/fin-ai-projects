"""命令行入口。

用法：
  # 两期分析（原有行为）
  python scripts/run.py analyze --cur data/statements/示例制造_2025A.json \
                                --prev data/statements/示例制造_2024A.json --md out.md

  # 多期分析（启用趋势规则，顺序不限）
  python scripts/run.py analyze --periods data/statements/长恒电子_2022A.json \
                                           data/statements/长恒电子_2023A.json \
                                           data/statements/长恒电子_2024A.json \
                                           data/statements/长恒电子_2025A.json \
                                --md out.md

  # 行业基准校准（观察线换成同行业 p75 分位）
  python scripts/run.py analyze --cur data/statements/恒润重工_2025A.json \
                                --prev data/statements/恒润重工_2024A.json \
                                --benchmarks data/benchmarks/industry_benchmarks.json

  # 导出审计工作底稿（Markdown 永可用；xlsx 需 openpyxl）
  python scripts/run.py workpaper --cur data/statements/示例制造_2025A.json \
                                  --prev data/statements/示例制造_2024A.json \
                                  --ref A-3 --xlsx artifacts/A-3.xlsx

  python scripts/run.py extract --text report.txt --period 2025A
  python scripts/run.py rules                     列出全部注册规则
  python scripts/run.py batch --dir data/statements  批量分析（按 *_YYYYA.json 配对）
  python scripts/run.py analyze ... --json         输出 JSON，便于接入其他系统

其他脚本：
  python scripts/sensitivity.py   风险分参数敏感性分析
  python scripts/fetch_benchmarks.py   抓真实行业基准（auditdog.cn 三大报表）
  python scripts/ab_test.py       模板 vs LLM 研判质量 A/B
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finagent import (  # noqa: E402
    ExtractAgent,
    Orchestrator,
    ReportAgent,
    StatementSeries,
    build_llm,
    build_working_paper,
    load_benchmarks,
    load_pair,
    load_statement,
    render_console,
    render_markdown,
    render_workpaper_markdown,
)
from finagent.rules import DEFAULT_RULES, RuleBook  # noqa: E402


def _require_file(path: str, what: str, hint: str = "") -> None:
    """文件不存在时给出可操作的提示，而不是裸 traceback。"""
    if Path(path).exists():
        return
    print(f"[错误] 找不到{what}：{path}", file=sys.stderr)
    print(f"       当前工作目录：{Path.cwd()}", file=sys.stderr)
    if hint:
        print(f"       提示：{hint}", file=sys.stderr)
    raise SystemExit(2)


_PATH_HINT = "CLI 示例路径均相对 finagent/ 目录，请先 cd 过去，或改用绝对路径。"


def _load_inputs(args: argparse.Namespace) -> tuple[object, object, StatementSeries | None]:
    """统一处理两期 / 多期两种输入方式。返回 (cur, prev, series)。"""
    if args.periods:
        if len(args.periods) < 2:
            print("[错误] --periods 至少需要 2 期报表", file=sys.stderr)
            raise SystemExit(2)
        for p in args.periods:
            _require_file(p, "报表文件", _PATH_HINT)
        series = StatementSeries(statements=[load_statement(p) for p in args.periods])
        return series.latest, series.previous, series
    if args.cur and args.prev:
        for p in (args.cur, args.prev):
            _require_file(p, "报表文件", _PATH_HINT)
        return (*load_pair(args.cur, args.prev), None)
    print(
        "[错误] 请指定 --cur 与 --prev（两期分析），或用 --periods 传多期报表（启用趋势分析）",
        file=sys.stderr,
    )
    raise SystemExit(2)


def cmd_analyze(args: argparse.Namespace) -> int:
    cur, prev, series = _load_inputs(args)

    bm = None
    if args.benchmarks:
        _require_file(
            args.benchmarks, "行业基准簿",
            "可用的示例文件：data/benchmarks/industry_benchmarks.json",
        )
        bm = load_benchmarks(args.benchmarks)
    llm = build_llm(args.llm)
    orch = Orchestrator(
        extract=ExtractAgent(llm=llm),
        report=ReportAgent(llm=llm),
    )
    result = orch.analyze(cur, prev, company=args.company, series=series, benchmarks=bm)

    # --json-out 落盘（结构化结果，供下游程序消费）
    if args.json_out:
        out_j = Path(args.json_out)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        out_j.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[ok] JSON 结果已写入：{out_j}")

    # --json 控制台打印结构化结果（保持向后兼容的纯开关语义）
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print(render_console(result, series=series))

    if args.md:
        out = Path(args.md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(result, cur, prev, series=series), encoding="utf-8")
        print(f"\n[ok] Markdown 报告已写入：{out}")
    return 0


def cmd_workpaper(args: argparse.Namespace) -> int:
    """导出审计工作底稿。Markdown 永可用；xlsx 需要 openpyxl（可选依赖）。"""
    cur, prev, series = _load_inputs(args)

    bm = None
    if args.benchmarks:
        _require_file(
            args.benchmarks, "行业基准簿",
            "可用的示例文件：data/benchmarks/industry_benchmarks.json",
        )
        bm = load_benchmarks(args.benchmarks)
    result = Orchestrator().analyze(
        cur, prev, company=args.company, series=series, benchmarks=bm
    )
    wp = build_working_paper(
        result,
        ref=args.ref,
        prepared_by=args.prepared_by,
        reviewed_by=args.reviewed_by,
        series=series,
    )

    md_text = render_workpaper_markdown(wp)
    out = Path(args.md or f"artifacts/workpaper_{wp.ref}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md_text, encoding="utf-8")
    print(f"[ok] 底稿（Markdown）已写入：{out}")

    if args.json_out:
        out_j = Path(args.json_out)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        out_j.write_text(json.dumps(wp.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] 底稿（JSON）已写入：{out_j}")

    if args.xlsx:
        try:
            from finagent.workpaper import render_workpaper_xlsx

            render_workpaper_xlsx(wp, args.xlsx)
            print(f"[ok] 底稿（xlsx）已写入：{args.xlsx}")
        except RuntimeError as e:
            print(f"[提示] {e}", file=sys.stderr)
            return 3
    return 0


def _read_json(path: str, what: str):
    """读取报表 JSON，文件缺失时给出可操作的提示而非裸 traceback。"""
    p = Path(path)
    if not p.exists():
        print(
            f"[错误] 找不到{what}：{path}\n"
            f"       当前工作目录：{Path.cwd()}\n"
            f"       提示：CLI 里的示例路径都相对 finagent/ 目录，请先 cd 过去，或用绝对路径。",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return load_statement(str(p))


def cmd_extract(args: argparse.Namespace) -> int:
    _require_file(
        args.text, "报表文本",
        "可用的示例文件：data/raw/示例公司2025年报摘录.txt",
    )
    text = Path(args.text).read_text(encoding="utf-8")
    st = ExtractAgent(llm=build_llm(args.llm)).parse_with_llm_fallback(text, args.period)
    print(json.dumps(st.to_dict(), ensure_ascii=False, indent=2))
    if st.meta.get("unrecognized"):
        print(f"\n[注意] {len(st.meta['unrecognized'])} 行未能识别，需人工核对：", file=sys.stderr)
        for line in st.meta["unrecognized"][:10]:
            print(f"  - {line}", file=sys.stderr)
    return 0


def cmd_rules(_: argparse.Namespace) -> int:
    rb = RuleBook()
    print("已注册规则（执行顺序即数据可信度优先级）：\n")
    for i, r in enumerate(DEFAULT_RULES, 1):
        doc = (r.__doc__ or "").strip().split("\n")[0]
        print(f"{i:>2}. {r.__name__:<32} {doc}")
    print("\n当前阈值：")
    for k, v in json.loads(json.dumps(rb.__dict__, default=str)).items():
        print(f"  {k:<24} {v}")
    return 0


_PAIR = re.compile(r"^(?P<name>.+)_(?P<period>\d{4}A)\.json$")


def cmd_batch(args: argparse.Namespace) -> int:
    """批量分析：按主体名分组，取相邻两期配对。"""
    d = Path(args.dir)
    groups: dict[str, list[tuple[str, Path]]] = {}
    for p in sorted(d.glob("*.json")):
        m = _PAIR.match(p.name)
        if not m:
            continue
        groups.setdefault(m.group("name"), []).append((m.group("period"), p))

    llm = build_llm(args.llm)
    rows: list[tuple[str, str, int, int, float]] = []
    for name, files in sorted(groups.items()):
        files.sort()
        for (pp, ppath), (cp, cpath) in zip(files, files[1:]):
            cur, prev = load_pair(cpath, ppath)
            result = Orchestrator(
                report=ReportAgent(llm=llm)
            ).analyze(cur, prev, company=name)
            rows.append((name, f"{pp}→{cp}", result.high_count, result.warn_count, result.risk_score()))
            if args.md_dir:
                od = Path(args.md_dir)
                od.mkdir(parents=True, exist_ok=True)
                (od / f"{name}_{cp}.md").write_text(
                    render_markdown(result, cur, prev), encoding="utf-8"
                )

    print(f"{'主体':<16}{'区间':<16}{'重大':>6}{'关注':>6}{'风险分':>10}")
    print("-" * 56)
    for name, span, hi, wa, sc in rows:
        print(f"{name:<16}{span:<16}{hi:>6}{wa:>6}{sc:>10.1f}")
    if not rows:
        print("（未找到可配对的报表，命名需形如 <主体>_<YYYYA>.json）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="finagent")
    ap.add_argument("--llm", default="auto", help="auto(默认) / api / null")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="分析两期或多期报表的差异与异常")
    a.add_argument("--cur", default=None, help="当期报表 JSON（两期模式）")
    a.add_argument("--prev", default=None, help="上期报表 JSON（两期模式）")
    a.add_argument(
        "--periods",
        nargs="+",
        default=None,
        metavar="JSON",
        help="多期报表路径，至少 2 个。传入即启用多期趋势分析，"
             "顺序不限（内部按期排序）",
    )
    a.add_argument(
        "--benchmarks",
        default=None,
        metavar="JSON",
        help="行业基准簿路径。传入后，AR_GAP / INV_GAP / GM_SWING / DEBT_RATIO "
             "的观察线改为同行业 p75 分位；报表 meta.industry 缺失或行业不在表里时"
             "自动回退绝对线，并在 Finding.evidence.line_source 中记录来源",
    )
    a.add_argument("--company", default=None, help="主体名称（缺省取报表 meta）")
    a.add_argument("--md", default=None, help="输出 Markdown 报告的路径")
    a.add_argument("--json", action="store_true", help="在控制台打印 JSON 而非摘要")
    a.add_argument("--json-out", default=None, help="把 JSON 结果写入指定路径")
    a.set_defaults(func=cmd_analyze)

    w = sub.add_parser("workpaper", help="导出审计工作底稿")
    w.add_argument("--cur", default=None, help="当期报表 JSON（两期模式）")
    w.add_argument("--prev", default=None, help="上期报表 JSON（两期模式）")
    w.add_argument("--periods", nargs="+", default=None, metavar="JSON",
                   help="多期报表路径，至少 2 个")
    w.add_argument("--benchmarks", default=None, metavar="JSON", help="行业基准簿路径")
    w.add_argument("--company", default=None, help="主体名称")
    w.add_argument("--ref", default="A-3", help="底稿编号，如 A-3 / Z1-1")
    w.add_argument("--prepared-by", default="", help="编制人")
    w.add_argument("--reviewed-by", default="", help="复核人")
    w.add_argument("--md", default=None, help="底稿 Markdown 输出路径")
    w.add_argument("--json-out", default=None, help="底稿 JSON 输出路径")
    w.add_argument("--xlsx", default=None, help="底稿 xlsx 输出路径（需要 openpyxl）")
    w.set_defaults(func=cmd_workpaper)

    e = sub.add_parser("extract", help="从文本报表抽取结构化科目")
    e.add_argument("--text", required=True)
    e.add_argument("--period", required=True)
    e.set_defaults(func=cmd_extract)

    r = sub.add_parser("rules", help="列出注册规则与阈值")
    r.set_defaults(func=cmd_rules)

    b = sub.add_parser("batch", help="批量分析目录下的报表")
    b.add_argument("--dir", required=True)
    b.add_argument("--md-dir", default=None)
    b.set_defaults(func=cmd_batch)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
