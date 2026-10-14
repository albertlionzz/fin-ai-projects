#!/usr/bin/env python3
"""把 fetch_benchmarks.py 抓下来的原始报表 HTML 转成 finagent 的 statement JSON。

存在的理由：审计底稿的结论必须能被独立复算。爬虫只产出基准分位数，
被分析公司自己的两期报表也要能同源还原，否则"行业校准"这条链上
数据来源不统一，读者无法验证校准用的样本和被测公司是不是一套口径。

用法：
    python scripts/build_fixture_from_raw.py --code 600600 \
        --cur 2025AR --prev 2024AR --industry 食品饮料 \
        --name 青岛啤酒 --out data/statements

只读 data/benchmarks/raw/ 下的缓存文件，不发任何网络请求。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# 复用爬虫的解析器：两处各写一套 HTML 解析必然漂移
_spec = importlib.util.spec_from_file_location("fb", HERE / "fetch_benchmarks.py")
fb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fb)  # type: ignore[union-attr]

RAW_DIR = HERE.parent / "data" / "benchmarks" / "raw"

# 站点行 ID -> finagent 科目名。
# 营收/成本用 OPERATE_INCOME/OPERATE_COST（与爬虫算毛利率变动同源）；
# 净利润/净资产取归母口径，两者配套，ROE 才不会分子分母错配。
BS_MAP = {
    "TOTAL_ASSETS": "total_assets",
    "TOTAL_LIABILITIES": "total_liabilities",
    "TOTAL_EQUITY": "total_equity",
    "TOTAL_PARENT_EQUITY": "net_assets",
    "TOTAL_CURRENT_ASSETS": "current_assets",
    "TOTAL_CURRENT_LIAB": "current_liabilities",
    "ACCOUNTS_RECE": "accounts_receivable",
    "INVENTORY": "inventory",
    "GOODWILL": "goodwill",
}
IS_MAP = {
    "OPERATE_INCOME": "revenue",
    "OPERATE_COST": "cost_of_sales",
    "PARENT_NETPROFIT": "net_profit",
    "SALE_EXPENSE": "selling_expense",
    "MANAGE_EXPENSE": "admin_expense",
    "RESEARCH_EXPENSE": "rd_expense",
    "FINANCE_EXPENSE": "finance_expense",
}
CF_MAP = {
    "NETCASH_OPERATE": "operating_cash_flow",
}


# 需要的全部站点行 ID。extract_rows 按传入的 ID 过滤，
# 所以必须把三张表映射表里的键并集一次传全，不能只传爬虫那 4 个。
WANTED_ROWS = sorted(set(BS_MAP) | set(IS_MAP) | set(CF_MAP))


def _load(code: str, period: str, kind: str, fetch_missing: bool = False) -> dict[str, str]:
    """kind ∈ {bs, is, cf}。缺文件时返回空 dict，由调用方决定是否致命。

    fetch_missing=True 且本地没有缓存时，现抓一张（走爬虫的磁盘缓存，
    下次同一代码就是纯本地操作）。主爬虫不抓现金流，所以演示公司要补。
    """
    path = RAW_DIR / f"{code}_{period}_{kind}.html"
    if not path.exists():
        if not fetch_missing:
            return {}
        rows = fb.fetch_statement(code, period, False, RAW_DIR, 1.0, kind=kind)
        return rows or {}
    buf = path.read_text(encoding="utf-8", errors="replace")
    if kind == "cf":
        return fb.extract_all_rows(buf)
    return fb.extract_rows(buf, WANTED_ROWS)


def _to_float(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return fb.parse_amount(s)
    except ValueError:
        return None


def build(code: str, period: str, name: str, industry: str, fetch_missing: bool = False) -> dict:
    # 站点用 2025AR（Annual Report），finagent fixture 用 2025A，统一后者
    label = period[:-1] if period.endswith("AR") else period
    items: dict[str, float] = {}
    missing: list[str] = []
    for kind, mapping in (("bs", BS_MAP), ("is", IS_MAP), ("cf", CF_MAP)):
        rows = _load(code, period, kind, fetch_missing)
        if not rows:
            missing.append(f"{kind}(整表缺失)")
            continue
        for rid, field in mapping.items():
            v = _to_float(rows.get(rid))
            if v is None:
                missing.append(f"{field}({rid})")
            else:
                items[field] = v
    return {
        "period": label,
        "meta": {
            "company": name,
            "unit": "元",
            "industry": industry,
            "source": (
                "auditdog.cn 三大报表 Beta（站点自上市公司公开年报 AI 抽取），"
                f"经 fetch_benchmarks.py 抓取缓存 {code}_{period}_*.html 还原；"
                "非审计数据。"
            ),
            "note": (
                f"由 scripts/build_fixture_from_raw.py 从原始 HTML 生成；"
                f"缺失字段：{'、'.join(missing) if missing else '无'}"
            ),
        },
        "items": items,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="从爬虫缓存生成 finagent statement JSON")
    ap.add_argument("--code", required=True, help="6 位证券代码")
    ap.add_argument("--cur", default="2025AR")
    ap.add_argument("--prev", default="2024AR")
    ap.add_argument("--industry", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--out", default=str(HERE.parent / "data" / "statements"))
    ap.add_argument(
        "--fetch-missing",
        action="store_true",
        help="本地没有的报表页现抓一张（默认只用缓存，缺就记缺失）",
    )
    a = ap.parse_args()

    if not a.name:
        # 从构建报告里取公司名，避免手输与 /data 页面不一致
        rep = HERE.parent / "data" / "benchmarks" / "build_report.json"
        if rep.exists():
            for comp in json.loads(rep.read_text(encoding="utf-8"))["industries"].values():
                if a.code in comp["companies"]:
                    a.name = comp["companies"][a.code]["name"]
                    break
        a.name = a.name or a.code

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for period in (a.cur, a.prev):
        doc = build(a.code, period, a.name, a.industry, a.fetch_missing)
        # 文件名用归一化后的期间标签（2025A），与既有 fixture 命名一致
        p = out_dir / f"{a.name}_{doc['period']}.json"
        p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        n = len(doc["items"])
        note = doc["meta"]["note"]
        print(f"[产出] {p}  ({n} 个科目；{note})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
