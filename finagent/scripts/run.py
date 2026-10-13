"""命令行入口。

用法：
  python scripts/run.py analyze --cur <当期.json> --prev <上期.json>
  python scripts/run.py analyze --cur data/statements/示例制造_2025A.json \
                                --prev data/statements/示例制造_2024A.json --md out.md
  python scripts/run.py extract --text report.txt --period 2025A
  python scripts/run.py rules                     列出全部注册规则
  python scripts/run.py batch --dir data/statements  批量分析（按 *_YYYYA.json 配对）
  python scripts/run.py analyze ... --json         输出 JSON，便于接入其他系统
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
    build_llm,
    load_pair,
    render_console,
    render_markdown,
)
from finagent.rules import DEFAULT_RULES, RuleBook  # noqa: E402


def cmd_analyze(args: argparse.Namespace) -> int:
    cur, prev = load_pair(args.cur, args.prev)
    llm = build_llm(args.llm)
    orch = Orchestrator(
        extract=ExtractAgent(llm=llm),
        report=ReportAgent(llm=llm),
    )
    result = orch.analyze(cur, prev, company=args.company)

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

    print(render_console(result))

    if args.md:
        out = Path(args.md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(result, cur, prev), encoding="utf-8")
        print(f"\n[ok] Markdown 报告已写入：{out}")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
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

    a = sub.add_parser("analyze", help="分析两份报表的差异与异常")
    a.add_argument("--cur", required=True, help="当期报表 JSON")
    a.add_argument("--prev", required=True, help="上期报表 JSON")
    a.add_argument("--company", default=None, help="主体名称（缺省取报表 meta）")
    a.add_argument("--md", default=None, help="输出 Markdown 报告的路径")
    a.add_argument("--json", action="store_true", help="在控制台打印 JSON 而非摘要")
    a.add_argument("--json-out", default=None, help="把 JSON 结果写入指定路径")
    a.set_defaults(func=cmd_analyze)

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
