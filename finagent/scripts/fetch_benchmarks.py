#!/usr/bin/env python3
"""从 auditdog.cn 抓取三大报表，构建 finagent 行业基准簿（真实数据版）。

为什么单独一个文件
------------------
`data/benchmarks/industry_benchmarks.json` 是模拟 fixture，测试和演示都钉在它的
具体数值上（工程机械 ar_growth_gap p75=12.0、小众仪器制造 n_peers=3 触发样本量
门槛）。真实数据必须落到另一个文件，否则测试的确定性就没了——两者可以并存，
用 `--benchmarks` 指哪个就用哪个。

数据源
------
bbs.auditdog.cn（陈版主实务问答检索）的三大报表 Beta，数据由站点从上市公司
公开年报 AI 抽取：

  * `/data?industry=<行业>&period=<YYYYAR>&tab=financial&page=N`
    每页 50 家，按 ROE 降序，给公司名与五项概览（营收/归母净利/总资产/
    经营现金流/ROE）。行业取申万一级（31 个），站点同时支持二级行业。
  * `/statements/<code>?period=<YYYYAR>`                 资产负债表
  * `/statements/<code>?period=<YYYYAR>&statement=income` 利润表
    行项目有稳定 ID，单期单值（页面不含上年同期列，两期要发两次请求）。

指标口径（与 finagent 规则严格一致，见 models.py / rules.py）
------------------------------------------------------------
  debt_ratio         = 负债合计 / 资产总计 × 100
  gross_margin_delta = |毛利率_本期 − 毛利率_上期|
                       ↑ 规则里比的是 `abs(delta)`，所以基准必须用**绝对变动**
                       的分布；直接用带符号值算 p75 会得到 +0.8 这种一侧阈值，
                       把一半公司的负向大幅波动判成"正常"。
  ar_growth_gap      = 应收账款同比 − 营业收入同比（百分点）
  inv_growth_gap     = 存货同比 − 营业收入同比（百分点）
  增长类指标设材料性门槛：基期科目 < 基期营收的 0.1% 时，该增速无经济意义，
  不进分布——否则"应收从 10 万到 500 万"会把 p90/p95 打成噪声。

默认排除 ST 公司（*ST西发、ST绝味等），退市风险与重组会让财务分布失真；
排除名单写进构建报告，不用 `--include-st` 看不到但不代表没发生。

礼貌与稳健
---------
默认每次请求间隔 0.8s、失败重试 3 次；404 落标记文件不重复请求；HTML 全量
落盘缓存，改指标口径重跑零网络开销；预估请求数超过 2000 需 `--force`。

用法
----
  # 干跑：只枚举公司清单，零报表请求
  python scripts/fetch_benchmarks.py --industries 食品饮料 --dry-run

  # 试点：按 ROE 跨度均匀抽 12 家
  python scripts/fetch_benchmarks.py --industries 食品饮料 --limit 12

  # 全量
  python scripts/fetch_benchmarks.py --industries 电子,计算机 --delay 1.0
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BASE = "https://bbs.auditdog.cn"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

PAGE_SIZE = 50          # /data 每页公司数
MAX_PAGES = 25          # 单行业枚举上限（1250 家）
REQUEST_GUARD = 2000    # 预估请求数超过此值需 --force

# 申万一级行业（站点行业下拉的完整清单）
SW_L1 = (
    "食品饮料", "农林牧渔", "基础化工", "钢铁", "有色金属", "电子",
    "家用电器", "纺织服饰", "轻工制造", "医药生物", "公用事业", "交通运输",
    "房地产", "商贸零售", "社会服务", "综合", "建筑材料", "建筑装饰",
    "电力设备", "国防军工", "计算机", "传媒", "通信", "银行", "非银金融",
    "汽车", "机械设备", "煤炭", "石油石化", "环保", "美容护理",
)

BS_ROWS = {"TOTAL_ASSETS", "TOTAL_LIABILITIES", "ACCOUNTS_RECE", "INVENTORY"}
IS_ROWS = {"OPERATE_INCOME", "OPERATE_COST", "TOTAL_OPERATE_INCOME"}

METRICS = ("ar_growth_gap", "inv_growth_gap", "debt_ratio", "gross_margin_delta")
QUANTILES = (50, 75, 90, 95)

# 基期科目 / 基期营收 的材料性门槛
MATERIALITY_RATIO = 0.001

NULL_MARKERS = {"", "-", "—", "–", "N/A", "None", "null", "－"}

_AMOUNT_RE = re.compile(r"^(-?[\d,]+(?:\.\d+)?)\s*(万亿|亿|万|千)?$")
_ROW_MARK_RE = re.compile(r'id="row-([A-Z0-9_]+)"')
_VALUE_RE = re.compile(r'tabular-nums[^"]*">([^<]*)<')
_COMPANY_LINK_RE = re.compile(
    r'href="/indicators/(\d{6})\?period=([0-9A-Z]{6})">([^<]+)</a>'
)

METRIC_DEFINITIONS = {
    "debt_ratio": "负债合计 / 资产总计 × 100（%）",
    "gross_margin_delta": (
        "|毛利率_本期 − 毛利率_上期|（百分点）。取绝对值是因为规则 AR/GM 判定"
        "比较的是 abs(delta)；带符号分布的分位数不构成有效阈值。"
    ),
    "ar_growth_gap": "应收账款同比 − 营业收入同比（百分点）",
    "inv_growth_gap": "存货同比 − 营业收入同比（百分点）",
    "_growth_materiality": (
        f"增长类指标的材料性门槛：基期科目 < 基期营收的 {MATERIALITY_RATIO:.1%} 时"
        "不计入分布（基期几乎不存在的科目，其增速无经济意义）。"
    ),
    "_n_peers": (
        "行业级 n_peers 取该行业保留指标的最小同行数（最弱环节口径）；"
        "各指标自己的同行数见 build_report.json。"
    ),
}


# ---------------------------------------------------------------- 基础解析


def parse_amount(text: str | None) -> float | None:
    """把 "4,237.02 亿" / "87,407,775.73" / "—" 解析成 float（单位：元）。

    无法识别的内容抛 ValueError 而不是静默返回 None——静默跳过一个坏值会让
    基准悄悄失真，宁可让爬虫当场停下来。
    """
    s = (text or "").strip()
    if s in NULL_MARKERS:
        return None
    m = _AMOUNT_RE.match(s)
    if not m:
        raise ValueError(f"无法解析的金额格式：{text!r}")
    num = float(m.group(1).replace(",", ""))
    unit = m.group(2)
    mult = {"万亿": 1e12, "亿": 1e8, "万": 1e4, "千": 1e3}.get(unit, 1.0)
    # 源值最多精确到"分"，round 到 2 位只是抹掉 423702000000.00006 这类
    # 浮点尾差，不损失任何有效信息。
    return round(num * mult, 2)


def row_value(rows: dict[str, str] | None, rid: str) -> float | None:
    """取某一行的数值；行不存在或为空值标记时返回 None。"""
    if not rows:
        return None
    raw = rows.get(rid)
    if raw is None:
        return None
    return parse_amount(raw)


def extract_rows(html: str, wanted: set[str]) -> dict[str, str]:
    """按行 ID 切段，取每段第一个 tabular-nums 单元格的原始字符串。

    页面同时含 SSR HTML 与 RSC payload，同一行会出现两次；按行 ID 取首次出现。
    切段而不是固定窗口，是为了避免长科目名把窗口撑爆后抓到下一行的值。
    """
    marks = list(_ROW_MARK_RE.finditer(html))
    out: dict[str, str] = {}
    for i, mk in enumerate(marks):
        rid = mk.group(1)
        if rid not in wanted or rid in out:
            continue
        end = marks[i + 1].start() if i + 1 < len(marks) else len(html)
        vm = _VALUE_RE.search(html, mk.start(), end)
        if vm:
            out[rid] = vm.group(1)
    return out


def percentile(values: list[float], p: float) -> float | None:
    """线性插值分位数（与 numpy 默认 linear 方法一致）。"""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ---------------------------------------------------------------- 网络层


def http_get(url: str, cache: Path, delay: float, retries: int = 3) -> str | None:
    """带磁盘缓存的 GET。返回 None 表示 404（站点没有这个资源）。

    404 写 `<cache>.404` 标记文件，重跑时不再打这个地址——否则每次重跑都要
    把全部缺失公司再问一遍。
    """
    miss = cache.with_name(cache.name + ".404")
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="replace")
    if miss.exists():
        return None

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(body, encoding="utf-8")
            if delay > 0:
                time.sleep(delay)
            return body
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                miss.parent.mkdir(parents=True, exist_ok=True)
                miss.write_text("", encoding="utf-8")
                if delay > 0:
                    time.sleep(delay)
                return None
            last_err = exc
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            http.client.HTTPException,
        ) as exc:
            # HTTPException 不是 OSError 子类：站点偶发断流时 `resp.read()` 抛
            # `http.client.IncompleteRead`（继承链 IncompleteRead → HTTPException
            # → Exception），曾整轮 95 分钟的电子抓取在第 71 秒被它一次断流打挂。
            last_err = exc
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"请求失败（重试 {retries} 次后放弃）：{url} -> {last_err}")


def parse_data_table(html: str, period: str) -> list[dict]:
    """解析 /data 页的公司表格行：代码、名称、五项概览单元格。"""
    rows: list[dict] = []
    for chunk in re.split(r"<tr", html)[1:]:
        m = _COMPANY_LINK_RE.search(chunk)
        if not m or m.group(2) != period:
            continue
        cells = re.findall(r'tabular-nums[^"]*">([^<]*)<', chunk)
        rows.append({"code": m.group(1), "name": m.group(3).strip(), "cells": cells})
    return rows


def enumerate_industry(
    industry: str, period: str, raw_dir: Path, delay: float
) -> list[dict]:
    """翻页拿全行业公司清单。"""
    out: list[dict] = []
    page = 1
    while page <= MAX_PAGES:
        url = (
            f"{BASE}/data?industry={urllib.parse.quote(industry)}"
            f"&period={period}&tab=financial&page={page}"
        )
        cache = raw_dir / f"list_{industry}_p{page}.html"
        html = http_get(url, cache, delay)
        if html is None:
            break
        rows = parse_data_table(html, period)
        if not rows:
            break
        out.extend(rows)
        if len(rows) < PAGE_SIZE:
            break
        page += 1
    return out


def extract_all_rows(html: str) -> dict[str, str]:
    """抽取页面上全部 `id="row-XXX"` 行的第一个数值单元格。

    extract_rows 只保留调用方点名的行；这里反其道而行之，用于行 ID 事先
    不确定的场景（现金流量表、以及排查新行业是否换了行名）。
    """
    import re

    found = re.findall(
        r'id="row-([A-Z_0-9]+)"[^>]*>(.{0,600}?)tabular-nums[^"]*">([^<]*)<', html, re.S
    )
    rows: dict[str, str] = {}
    for rid, _mid, val in found:
        rows.setdefault(rid, val)
    return rows


def fetch_statement(
    code: str,
    period: str,
    income: bool,
    raw_dir: Path,
    delay: float,
    kind: str | None = None,
) -> dict[str, str] | None:
    """抓一张报表页并抽行。返回 None 表示该期间没有数据（404）。

    kind 默认由 income 推导（bs/is）。显式传 "cf" 抓现金流量表，
    供 build_fixture_from_raw.py 补经营现金流科目用——基准分位数不算
    现金流，所以主爬虫不抓，省掉 50% 的请求。
    """
    k = kind or ("is" if income else "bs")
    if k == "cf":
        stmt = "&statement=cashflow"
    elif k == "is":
        stmt = "&statement=income"
    else:
        stmt = ""
    cache = raw_dir / f"{code}_{period}_{k}.html"
    url = f"{BASE}/statements/{code}?period={period}{stmt}"
    html = http_get(url, cache, delay)
    if html is None:
        return None
    if k == "cf":
        return extract_all_rows(html)
    return extract_rows(html, IS_ROWS if k == "is" else BS_ROWS)


# ---------------------------------------------------------------- 指标计算


def gross_margin(rows: dict[str, str] | None) -> float | None:
    rev = row_value(rows, "OPERATE_INCOME")
    cost = row_value(rows, "OPERATE_COST")
    if rev is None or cost is None or rev == 0:
        return None
    return (rev - cost) / rev * 100.0


def compute_metrics(
    bs_cur: dict[str, str] | None,
    bs_prev: dict[str, str] | None,
    is_cur: dict[str, str] | None,
    is_prev: dict[str, str] | None,
) -> dict[str, float]:
    """按 finagent 口径算 4 项指标；数据不齐的项直接不出现。"""
    out: dict[str, float] = {}

    ta = row_value(bs_cur, "TOTAL_ASSETS")
    tl = row_value(bs_cur, "TOTAL_LIABILITIES")
    if ta and tl is not None:
        out["debt_ratio"] = tl / ta * 100.0

    gm_cur, gm_prev = gross_margin(is_cur), gross_margin(is_prev)
    if gm_cur is not None and gm_prev is not None:
        out["gross_margin_delta"] = abs(gm_cur - gm_prev)

    rev_cur = row_value(is_cur, "OPERATE_INCOME")
    rev_prev = row_value(is_prev, "OPERATE_INCOME")
    if rev_cur and rev_prev:
        rev_yoy = (rev_cur - rev_prev) / abs(rev_prev) * 100.0
        for key, rid in (("ar", "ACCOUNTS_RECE"), ("inv", "INVENTORY")):
            cur_v = row_value(bs_cur, rid)
            prev_v = row_value(bs_prev, rid)
            if cur_v is None or not prev_v:
                continue
            if rev_prev > 0 and abs(prev_v) < MATERIALITY_RATIO * rev_prev:
                continue
            yoy = (cur_v - prev_v) / abs(prev_v) * 100.0
            out[f"{key}_growth_gap"] = yoy - rev_yoy
    return out


def cross_check(cells: list[str], bs_cur, is_cur) -> dict[str, float]:
    """用 /data 概览列交叉校验报表解析值，返回相对偏差。

    /data 的营收列对应营业总收入、总资产列对应资产总计；两边都来自同一份
    年报的不同抽取路径，偏差超过 1% 说明某一路解析错了。

    营收必须同口径对比：/data 概览列是营业总收入，报表侧也要取
    TOTAL_OPERATE_INCOME 而不能取 OPERATE_INCOME（营业收入）。两家口径在
    有金融业务的公司本就不同——指南针营业总收入 21.46 亿 vs 营业收入
    15.39 亿（差 28%），新大陆差 4%，茅台差 1.9%（财务公司利息收入计入
    营业总收入、不计入营业收入）。混口径会让这类公司全部误报"解析错误"。
    """
    checks: dict[str, float] = {}
    try:
        dp_rev = parse_amount(cells[0]) if len(cells) > 0 else None
        dp_ta = parse_amount(cells[2]) if len(cells) > 2 else None
        st_rev = row_value(is_cur, "TOTAL_OPERATE_INCOME")
        st_ta = row_value(bs_cur, "TOTAL_ASSETS")
    except ValueError:
        return checks
    if dp_rev and st_rev:
        checks["revenue"] = abs(st_rev - dp_rev) / dp_rev
    if dp_ta and st_ta:
        checks["total_assets"] = abs(st_ta - dp_ta) / dp_ta
    return checks


def is_st(name: str) -> bool:
    return "ST" in name.upper()


def excluded_st_codes(companies: list[dict], include_st: bool) -> list[str]:
    """枚举阶段被 ST 过滤挡在爬取之外的公司代码。

    必须在枚举全量上算：crawl_industry 收到的是过滤后的清单，
    它自己统计的 excluded_st 恒为空——"默认排除 ST"这件事会在构建报告里
    无据可查（2026-10-16 修复：此前两个行业的 excluded_st 都是 []，
    而电子实际排除了 10 家、食品饮料 11 家）。
    """
    if include_st:
        return []
    return [c["code"] for c in companies if is_st(c["name"])]


# ---------------------------------------------------------------- 主流程


def stride_sample(items: list, limit: int) -> list:
    """等跨距抽样。

    `/data` 按 ROE 降序，"取前 N 家"等于只取高 ROE 群体，试点数字会整体偏乐观。
    下标按 linspace 取，保证首尾都进样本——试点要的是跨 ROE 谱系各取一家。
    """
    if limit <= 0 or limit >= len(items):
        return list(items)
    if limit == 1:
        return [items[len(items) // 2]]
    idx = [round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)]
    return [items[i] for i in idx]


def crawl_industry(
    industry: str,
    companies: list[dict],
    cur: str,
    prev: str,
    raw_dir: Path,
    delay: float,
) -> tuple[list[dict], dict]:
    """逐个公司抓两期两张表，算指标。返回 (公司结果, 行业统计)。"""
    results: list[dict] = []
    stats = {
        "enumerated": len(companies),
        "crawled": 0,
        "missing_statements": [],
        "parse_errors": [],
        "flagged_checks": [],
    }
    for comp in companies:
        code, name = comp["code"], comp["name"]
        rec: dict = {"code": code, "name": name, "status": "ok", "metrics": {}}
        try:
            # 本期资产负债表先抓：没有它这家公司就没有抓下去的意义，
            # 剩下 3 个请求不必再打。
            bs_cur = fetch_statement(code, cur, False, raw_dir, delay)
            if bs_cur is None:
                rec["status"] = "missing_statements"
                stats["missing_statements"].append(code)
                results.append(rec)
                continue
            bs_prev = fetch_statement(code, prev, False, raw_dir, delay)
            is_cur = fetch_statement(code, cur, True, raw_dir, delay)
            is_prev = fetch_statement(code, prev, True, raw_dir, delay)
        except (ValueError, RuntimeError) as exc:
            rec["status"] = "error"
            rec["error"] = str(exc)
            stats["parse_errors"].append(code)
            results.append(rec)
            continue

        try:
            metrics = compute_metrics(bs_cur, bs_prev, is_cur, is_prev)
            checks = cross_check(comp["cells"], bs_cur, is_cur)
        except ValueError as exc:
            rec["status"] = "parse_error"
            rec["error"] = str(exc)
            stats["parse_errors"].append(code)
            results.append(rec)
            continue

        rec["metrics"] = metrics
        if not metrics:
            rec["status"] = "no_metrics"
        rec["raw"] = {
            "total_assets": row_value(bs_cur, "TOTAL_ASSETS"),
            "total_liabilities": row_value(bs_cur, "TOTAL_LIABILITIES"),
            "accounts_receivable": row_value(bs_cur, "ACCOUNTS_RECE"),
            "inventory": row_value(bs_cur, "INVENTORY"),
            "revenue": row_value(is_cur, "OPERATE_INCOME"),
            "cost_of_sales": row_value(is_cur, "OPERATE_COST"),
        }
        rec["checks"] = checks
        if any(v > 0.01 for v in checks.values()):
            stats["flagged_checks"].append(code)
        stats["crawled"] += 1
        results.append(rec)
    return results, stats


def build_book(
    per_industry: dict[str, list[dict]],
    min_peers: int,
    existing: dict | None = None,
) -> tuple[dict, dict]:
    """汇总成基准簿 JSON + 构建报告。

    existing 是同一条 --out 路径上已有的簿。**必须合并而不是覆盖**：
    基准簿按行业累加，跑一个行业就冲掉上一个行业的数据是不可接受的损失
    （电子 485 家的抓取成本是 95 分钟，被一次误覆盖全废）。
    合并规则：本次抓到的行业整体替换，未抓到的行业原样保留，
    每个行业块带自己的 as_of，读者能看出哪批数据是旧的。
    """
    today = date.today().isoformat()
    industries: dict[str, dict] = {}
    if existing:
        prev = existing.get("industries") or {}
        if isinstance(prev, dict):
            industries.update(prev)      # 先保留旧行业，下面再覆盖本次的
    book: dict = {
        "schema": "finagent-industry-benchmarks/1",
        "source": (
            "auditdog.cn 三大报表 Beta（站点自上市公司公开年报 AI 抽取），"
            "2025AR/2024AR；非审计数据，仅作行业参照系，不作为审计证据。"
            "按行业分次累加，各行业抓取日期见该行业块内的 as_of。"
        ),
        "as_of": today,
        "min_peers": min_peers,
        "metric_definitions": METRIC_DEFINITIONS,
        "industries": industries,
    }
    report: dict = {"industries": {}}
    for industry, results in per_industry.items():
        per_metric: dict[str, list[float]] = {m: [] for m in METRICS}
        for rec in results:
            for m, v in rec.get("metrics", {}).items():
                if v is not None:
                    per_metric.setdefault(m, []).append(v)

        block: dict[str, dict] = {}
        detail: dict[str, dict] = {}
        ns: list[int] = []
        for m in METRICS:
            vals = per_metric.get(m, [])
            detail[m] = {
                "n": len(vals),
                "min": round(min(vals), 2) if vals else None,
                "max": round(max(vals), 2) if vals else None,
                **{f"p{q}": round(percentile(vals, q), 2) if vals else None for q in QUANTILES},
            }
            if len(vals) < min_peers:
                continue
            ns.append(len(vals))
            block[m] = {f"p{q}": round(percentile(vals, q), 2) for q in QUANTILES}
        if not block:
            continue
        industries[industry] = {
            "n_peers": min(ns),
            "as_of": today,
            "metrics": block,
        }
        report["industries"][industry] = {"metrics": detail}
    return book, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="从 auditdog.cn 抓三大报表，构建 finagent 行业基准簿",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--industries", required=True,
                    help="逗号分隔的行业名（申万一级），或 all")
    ap.add_argument("--cur", default="2025AR", help="本期，默认 2025AR")
    ap.add_argument("--prev", default="2024AR", help="上期，默认 2024AR")
    ap.add_argument("--limit", type=int, default=0,
                    help="每行业抽样公司数（按 ROE 跨度均匀抽），0 表示全量")
    ap.add_argument("--include-st", action="store_true",
                    help="保留 ST 公司（默认排除）")
    ap.add_argument("--delay", type=float, default=0.8, help="请求间隔秒数")
    ap.add_argument("--min-peers", type=int, default=5, help="指标最低同行数")
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "benchmarks" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "data" / "benchmarks" / "industry_benchmarks.auditdog.json"))
    ap.add_argument("--report", default=str(ROOT / "data" / "benchmarks" / "build_report.json"))
    ap.add_argument("--force", action="store_true",
                    help=f"预估请求数超过 {REQUEST_GUARD} 时强制执行")
    ap.add_argument("--dry-run", action="store_true",
                    help="只枚举公司清单并估算请求数，不抓报表")
    args = ap.parse_args(argv)

    if args.industries.strip().lower() == "all":
        industries = list(SW_L1)
    else:
        industries = [s.strip() for s in args.industries.split(",") if s.strip()]
    unknown = [i for i in industries if i not in SW_L1]
    if unknown:
        print(f"[提示] 非申万一级行业（按站点二级行业处理）：{unknown}")

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1) 枚举
    universe: dict[str, list[dict]] = {}
    for ind in industries:
        comps = enumerate_industry(ind, args.cur, raw_dir, args.delay)
        universe[ind] = comps
        n_st = sum(1 for c in comps if is_st(c["name"]))
        print(f"[枚举] {ind}: {len(comps)} 家（含 ST {n_st} 家）")

    # 2) 估算请求数
    plan: dict[str, list[dict]] = {}
    n_companies = 0
    for ind, comps in universe.items():
        kept = [c for c in comps if args.include_st or not is_st(c["name"])]
        picked = stride_sample(kept, args.limit)
        plan[ind] = picked
        n_companies += len(picked)
    total = n_companies * 4    # 每家公司：资产负债×2 期 + 利润表×2 期
    print(f"[计划] {n_companies} 家公司 × 4 请求 = {total} 次报表请求"
          f"（每 50 家公司另有 1 次清单请求）")
    if total > REQUEST_GUARD and not args.force:
        print(f"[中止] 预估请求数 {total} 超过 {REQUEST_GUARD}，"
              f"确认无误请加 --force", file=sys.stderr)
        return 2
    if args.dry_run:
        for ind, comps in plan.items():
            print(f"  {ind}: {len(comps)} 家 -> "
                  f"{[c['name'] for c in comps[:5]]}{' ...' if len(comps) > 5 else ''}")
        return 0

    # 3) 抓取 + 计算
    per_industry: dict[str, list[dict]] = {}
    report_industries: dict[str, dict] = {}
    t0 = time.time()
    done = 0
    for ind, comps in plan.items():
        results, stats = crawl_industry(
            ind, comps, args.cur, args.prev, raw_dir, args.delay
        )
        per_industry[ind] = results
        # ST 排除发生在进入 crawl_industry 之前，排除名单必须回到枚举全量算，
        # 并用纯函数 excluded_st_codes() 计算（有单测钉住 include_st 两条路径）
        report_industries[ind] = {**stats, "excluded_st":
            excluded_st_codes(universe[ind], args.include_st), "companies": {
            c["code"]: c for c in results
        }}
        done += len(comps)
        print(f"[完成] {ind}: {stats['crawled']} 家有数据，"
              f"缺报表 {len(stats['missing_statements'])} 家，"
              f"解析异常 {len(stats['parse_errors'])} 家，"
              f"交叉校验告警 {len(stats['flagged_checks'])} 家"
              f"（累计 {done}/{total}，耗时 {time.time() - t0:.0f}s）")

    # 4) 汇总（与已有簿合并，不覆盖未重抓的行业）
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict | None = None
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            existing = None      # 读不出来就当没有，重新生成好过崩掉
    if existing:
        kept = [i for i in (existing.get("industries") or {}) if i not in industries]
        if kept:
            print(f"[合并] 保留未重抓的行业：{kept}")
    book, detail = build_book(per_industry, args.min_peers, existing)
    for ind, det in detail["industries"].items():
        report_industries[ind]["metrics"] = det["metrics"]
    out_path.write_text(
        json.dumps(book, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # 报告同样合并：未重抓行业的明细原样保留
    report_path = Path(args.report)
    prev_report: dict = {}
    if report_path.exists():
        try:
            prev_report = json.loads(report_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            prev_report = {}
    prev_inds = prev_report.get("industries") or {}
    if isinstance(prev_inds, dict):
        for ind, body in prev_inds.items():
            if ind not in report_industries and isinstance(body, dict):
                report_industries[ind] = body
    report = {
        "generated_at": date.today().isoformat(),
        "params": {
            "industries": industries,
            "cur": args.cur,
            "prev": args.prev,
            "limit": args.limit,
            "include_st": args.include_st,
            "min_peers": args.min_peers,
        },
        "summary": {
            "industries_in_book": len(book["industries"]),
            "companies_in_book": sum(
                b["n_peers"] for b in book["industries"].values()
            ),
            "requests_this_run": total,
        },
        "industries": report_industries,
    }
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # 5) 自检：产出必须能被 finagent 正常加载
    from finagent import load_benchmarks  # noqa: E402

    check = load_benchmarks(out_path)
    print(f"[自检] load_benchmarks 读到 {len(check)} 条基准，"
          f"覆盖行业：{check.industries()}")
    print(f"[产出] {out_path}")
    print(f"[报告] {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
