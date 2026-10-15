"""审计工作底稿导出：把分析结果变成能直接进底稿归档的结构。

为什么不是再出一份 Markdown 报告
------------------------------
Markdown 报告是给**人读**的（投资者、管理层）；底稿是给**程序与复核人**用的。
两者回答的问题不同：

| | Markdown 报告 | 审计工作底稿 |
|---|---|---|
| 读者 | 投资者 / 管理层 | 项目合伙人 / 质控 / 监管检查 |
| 核心要求 | 看得懂 | 可追溯、可交叉索引、可复算 |
| 结构 | 叙述式五章节 | 编号化区块 + 索引号 + 签署栏 |
| 数字 | 四舍五入到两位 | 保留原始精度，便于反向核对 |

中国会计师事务所的底稿规范（可对照《中国注册会计师审计准则第 1131 号》）
要求每份底稿具备：**底稿编号、被审计单位、审计期间、审计过程与结论、
索引号（交叉引用）、编制人与复核人签署、审计标识**。
本模块把这些固化成数据模型，渲染层只负责排版。

设计取舍
--------
1. **模型与渲染分离**。`WorkingPaper` 是纯数据，先有模型再有渲染。
   同一份底稿可以出 Markdown、出 xlsx、出 HTML，不必改一行判定逻辑。
2. **xlsx 是可选依赖**。项目主线约定是零第三方依赖，所以 `render_workpaper_xlsx`
   需要 openpyxl 时才 import，缺依赖时**明确报错并说明装什么**，
   而不是静默退化成 Markdown——静默降级会让使用者以为拿到了 xlsx。
3. **审计标识有固定语义**。▲ 调整 / ● 重点核查 / ○ 一般关注 / √ 已核对 / × 不符。
   底稿里出现这些符号必须有统一含义，否则交叉索引时无法解读。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Sequence

from .models import AnalysisResult, Finding, Level, StatementSeries


# ---------------------------------------------------------------- 审计标识
#
# 底稿里这些符号是"压缩语言"：一个符号代表一类处理结论。
# 语义必须固定，否则复核人无法解读别人的底稿。

MARKS: dict[str, str] = {
    "▲": "调整事项（需提出审计调整）",
    "●": "重点核查事项（需专项程序）",
    "○": "一般关注事项（需说明或问询）",
    "√": "已核对相符",
    "×": "核对不符",
    "△": "差异已查明并说明",
}


def mark_for(level: Level) -> str:
    """严重度 → 审计标识。映射与等级语义绑定，不随心情改。"""
    return {Level.HIGH: "●", Level.WARN: "○", Level.INFO: "△"}.get(level, "△")


# ---------------------------------------------------------------- 数据模型


@dataclass
class Procedure:
    """一项已执行的审计程序。

    底稿的"过程"部分。程序 = 做了什么 + 范围 + 方法 + 结果，
    缺一项就不算完整记录——复核人无法判断程序是否充分。
    """

    code: str                       # 程序编号，如 "P1"
    name: str                       # 程序名称
    scope_: str = ""                # 范围（样本量 / 覆盖期间）
    method: str = ""                # 方法（检查 / 函证 / 分析程序 / 重新计算）
    result: str = ""                # 结果与发现
    marks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "name": self.name,
            "scope": self.scope_,
            "method": self.method,
            "result": self.result,
            "marks": list(self.marks),
        }


@dataclass
class CrossReference:
    """一条交叉索引：本底稿的某结论指向另一份底稿或原始资料。

    审计底稿的核心属性是**可追溯**：任何一个数字都要能顺藤摸到原始凭证。
    没有交叉索引的底稿是一份孤证。
    """

    from_ref: str                   # 本底稿内的位置，如 "结论-3"
    to_ref: str                     # 指向的底稿编号或资料名称
    note: str = ""

    def to_dict(self) -> dict:
        return {"from": self.from_ref, "to": self.to_ref, "note": self.note}


@dataclass
class Signoff:
    """编制 / 复核签署栏。底稿没有签署栏等于没有完成。"""

    prepared_by: str = ""
    prepared_date: str = ""
    reviewed_by: str = ""
    reviewed_date: str = ""

    def to_dict(self) -> dict:
        return {
            "prepared_by": self.prepared_by,
            "prepared_date": self.prepared_date,
            "reviewed_by": self.reviewed_by,
            "reviewed_date": self.reviewed_date,
        }


@dataclass
class WorkingPaper:
    """一份审计工作底稿。"""

    ref: str                                # 底稿编号，如 "A-3"
    title: str                              # 底稿标题
    entity: str                             # 被审计单位
    period: str                             # 审计期间
    objective: str = ""                     # 审计目标
    basis: str = ""                         # 编制依据（准则 / 制度）
    procedures: list[Procedure] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    conclusion: str = ""                    # 审计结论
    cross_refs: list[CrossReference] = field(default_factory=list)
    signoff: Signoff = field(default_factory=Signoff)
    meta: dict[str, Any] = field(default_factory=dict)

    # --- 派生 ---

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.level == Level.HIGH)

    @property
    def warn_count(self) -> int:
        return sum(1 for f in self.findings if f.level == Level.WARN)

    def index_refs(self) -> list[str]:
        """本底稿用到的全部索引号，供底稿目录交叉核对。"""
        return [c.to_ref for c in self.cross_refs]

    def to_dict(self) -> dict:
        return {
            "ref": self.ref,
            "title": self.title,
            "entity": self.entity,
            "period": self.period,
            "objective": self.objective,
            "basis": self.basis,
            "procedures": [p.to_dict() for p in self.procedures],
            "findings": [f.to_dict() for f in self.findings],
            "conclusion": self.conclusion,
            "cross_refs": [c.to_dict() for c in self.cross_refs],
            "signoff": self.signoff.to_dict(),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkingPaper":
        from .models import Finding as _F

        return cls(
            ref=d["ref"],
            title=d["title"],
            entity=d["entity"],
            period=d["period"],
            objective=d.get("objective", ""),
            basis=d.get("basis", ""),
            procedures=[
                Procedure(
                    code=p["code"],
                    name=p["name"],
                    scope_=p.get("scope", ""),
                    method=p.get("method", ""),
                    result=p.get("result", ""),
                    marks=list(p.get("marks", [])),
                )
                for p in d.get("procedures", [])
            ],
            findings=[_F.from_dict(f) for f in d.get("findings", [])],
            conclusion=d.get("conclusion", ""),
            cross_refs=[
                CrossReference(from_ref=c["from"], to_ref=c["to"], note=c.get("note", ""))
                for c in d.get("cross_refs", [])
            ],
            signoff=Signoff(**d.get("signoff", {})),
            meta=d.get("meta", {}),
        )


# ---------------------------------------------------------------- 构建


def build_working_paper(
    result: AnalysisResult,
    ref: str = "A-3",
    prepared_by: str = "",
    reviewed_by: str = "",
    series: StatementSeries | None = None,
) -> WorkingPaper:
    """把一次分析结果转成底稿。

    刻意保留 analysis 的**原始结论**（不改写、不润色）：底稿是证据，
    不是宣传材料。研判文字原样进入"审计结论"区，异常明细原样进入"发现"区。

    程序区是**按实际执行的动作反推**的，不是模板套话：
    - 勾稽校验执行了 → 记一条"重新计算"程序
    - 趋势分析执行了 → 记一条"分析程序"并注明趋势区间
    - 行业校准生效了 → 记一条程序说明阈值来源，并在索引里指向基准数据文件
    """
    today = date.today().isoformat()
    trend_on = series is not None and series.has_trend(3)

    procedures: list[Procedure] = [
        Procedure(
            code="P1",
            name="资产负债表勾稽关系复核",
            scope_="本期全部资产负债表科目",
            method="重新计算",
            result=(
                "资产 = 负债 + 所有者权益，容差千分之一。"
                + ("发现勾稽差异，见发现区。" if any(f.rule_id == "BS_IDENTITY" for f in result.findings)
                   else "勾稽平衡，未发现差异。")
            ),
            marks=["√"] if not any(f.rule_id == "BS_IDENTITY" for f in result.findings) else ["×"],
        ),
        Procedure(
            code="P2",
            name="财务指标横向与纵向分析",
            scope_=(
                f"{series.periods[0]}→{series.periods[-1]}（{len(series)} 期）"
                if trend_on else "本期与上期"
            ),
            method="分析程序（比率分析 / 趋势分析）",
            result=(
                f"执行 {len(result.findings)} 项规则判定，命中 {len(result.findings)} 项发现"
                f"（重大 {result.high_count} / 关注 {result.warn_count}）。"
                + ("含多期趋势规则与持续性升级。" if trend_on else "")
            ),
            marks=["●"] if result.high_count else ["○"],
        ),
    ]

    if trend_on:
        procedures.append(
            Procedure(
                code="P3",
                name="多期趋势与持续性判定",
                scope_=f"{' → '.join(series.periods)}",
                method="分析程序（连续同向变动检验）",
                result=(
                    "对最近 2 个相邻报告期均触发的信号升级一级严重度；"
                    "另按 QUALITY_DRIFT 检验多项盈利质量指标的连续同向变动。"
                ),
                marks=["●"] if any(f.rule_id == "QUALITY_DRIFT" for f in result.findings) else ["√"],
            )
        )

    calibrated = [
        f for f in result.findings
        if f.evidence.get("line_source", "absolute") != "absolute"
    ]
    if calibrated:
        procedures.append(
            Procedure(
                code="P4",
                name="行业基准比对",
                scope_=f"行业 {result.metrics.get('benchmarks', {}).get('industry', '—')}",
                method="分析程序（同行业分位数比对）",
                result=(
                    f"{len(calibrated)} 项发现的观察线按行业分位校准"
                    "（关注线取 p75、重大线取 p90，缺失时退回比例推导），"
                    "校准后阈值与同行区间已在发现区逐条披露。"
                ),
                marks=["●"],
            )
        )

    cross_refs: list[CrossReference] = [
        CrossReference("发现区", ref, "本底稿"),
        CrossReference("程序-P2", "A-1 财务报表分析程序表", "分析程序依据"),
    ]
    if trend_on:
        cross_refs.append(
            CrossReference("程序-P3", "A-2 多期趋势分析表", "趋势区间与各期指标明细")
        )
    if calibrated:
        bm_meta = result.metrics.get("benchmarks", {})
        cross_refs.append(
            CrossReference(
                "程序-P4",
                f"行业基准数据（{bm_meta.get('as_of', '—')}，来源：{bm_meta.get('source', '—')}）",
                "分位数取值的原始依据",
            )
        )
    cross_refs.append(
        CrossReference("结论", "D-1 审计结论汇总表", "重大异常汇总")
    )

    return WorkingPaper(
        ref=ref,
        title="财务报表异常识别",
        entity=result.company,
        period=result.period,
        objective=("识别财务报表中的异常事项，评估其对审计意见的影响，"
                   "并确定需要追加的审计程序。"),
        basis=("《中国注册会计师审计准则第 1131 号——审计工作底稿》；"
               "《中国注册会计师审计准则第 1301 号——审计证据》。"
               "异常判定阈值见 finagent RuleBook，可复算。"),
        procedures=procedures,
        findings=list(result.findings),
        conclusion=(
            f"{result.company} {result.period} 共识别 {len(result.findings)} 项异常"
            f"（重大 {result.high_count} 项、关注 {result.warn_count} 项），"
            f"风险分 {result.risk_score():.1f}/100，等级「{result.risk_grade()}」。"
            f"{result.risk_action()}。"
        ),
        cross_refs=cross_refs,
        signoff=Signoff(
            prepared_by=prepared_by or "finagent（自动编制）",
            prepared_date=today,
            reviewed_by=reviewed_by,
        ),
        meta={
            "risk_score": round(result.risk_score(), 2),
            "risk_grade": result.risk_grade(),
            "trend_on": trend_on,
            "periods": series.periods if series else [result.period],
            "generator": "finagent",
        },
    )


# ---------------------------------------------------------------- Markdown 渲染


def render_workpaper_markdown(wp: WorkingPaper) -> str:
    """底稿的 Markdown 渲染。纯标准库，永远可用。"""
    L: list[str] = []
    L.append(f"# 审计工作底稿 {wp.ref} · {wp.title}")
    L.append("")
    L.append(f"**被审计单位**：{wp.entity}　　**审计期间**：{wp.period}")
    L.append("")
    L.append("| 项目 | 内容 |")
    L.append("|---|---|")
    L.append(f"| 底稿编号 | {wp.ref} |")
    L.append(f"| 底稿标题 | {wp.title} |")
    L.append(f"| 被审计单位 | {wp.entity} |")
    L.append(f"| 审计期间 | {wp.period} |")
    L.append(f"| 索引号 | {', '.join(wp.index_refs()) or '—'} |")
    L.append(f"| 风险分 / 等级 | {wp.meta.get('risk_score', '—')} / {wp.meta.get('risk_grade', '—')} |")
    L.append("")

    if wp.objective:
        L.append("## 一、审计目标")
        L.append("")
        L.append(wp.objective)
        L.append("")

    if wp.basis:
        L.append("## 二、编制依据")
        L.append("")
        L.append(wp.basis)
        L.append("")

    L.append("## 三、审计过程")
    L.append("")
    L.append("| 程序编号 | 程序名称 | 范围 | 方法 | 标识 |")
    L.append("|---|---|---|---|---|")
    for p in wp.procedures:
        L.append(f"| {p.code} | {p.name} | {p.scope_} | {p.method} | {' '.join(p.marks) or '—'} |")
    L.append("")
    for p in wp.procedures:
        L.append(f"**{p.code}　{p.name}**")
        L.append("")
        if p.scope_:
            L.append(f"- 范围：{p.scope_}")
        if p.method:
            L.append(f"- 方法：{p.method}")
        L.append(f"- 结果：{p.result}")
        if p.marks:
            L.append(f"- 标识：{' '.join(p.marks)}")
        L.append("")

    L.append("## 四、审计发现")
    L.append("")
    if not wp.findings:
        L.append("未识别出异常事项。")
        L.append("")
    else:
        L.append("| 序号 | 标识 | 严重度 | 发现事项 | 规则编号 | 观察线来源 |")
        L.append("|---|---|---|---|---|---|")
        for i, f in enumerate(wp.findings, 1):
            L.append(
                f"| {i} | {mark_for(f.level)} | {f.level.label} | {f.title} "
                f"| `{f.rule_id}` | {f.evidence.get('line_source', 'absolute')} |"
            )
        L.append("")
        for i, f in enumerate(wp.findings, 1):
            L.append(f"### {i}. {mark_for(f.level)} {f.title}")
            L.append("")
            L.append(f"**严重度**：{f.level.label}　**规则编号**：`{f.rule_id}`"
                     f"　**观察线来源**：{f.evidence.get('line_source', 'absolute')}")
            L.append("")
            L.append(f.detail)
            L.append("")
            if f.basis:
                L.append(f"*判定依据*：{f.basis}")
                L.append("")

    if wp.cross_refs:
        L.append("## 五、交叉索引")
        L.append("")
        L.append("| 本底稿位置 | 指向 | 说明 |")
        L.append("|---|---|---|")
        for c in wp.cross_refs:
            L.append(f"| {c.from_ref} | {c.to_ref} | {c.note or '—'} |")
        L.append("")

    L.append("## 六、审计结论")
    L.append("")
    L.append(wp.conclusion)
    L.append("")

    L.append("## 七、签署")
    L.append("")
    L.append("| | 姓名 / 签章 | 日期 |")
    L.append("|---|---|---|")
    L.append(f"| 编制人 | {wp.signoff.prepared_by or '—'} | {wp.signoff.prepared_date or '—'} |")
    L.append(f"| 复核人 | {wp.signoff.reviewed_by or '—'} | {wp.signoff.reviewed_date or '—'} |")
    L.append("")

    used = {m for f in wp.findings for m in [mark_for(f.level)]}
    used |= {m for p in wp.procedures for m in p.marks}
    if used:
        L.append("**审计标识说明**")
        L.append("")
        for m in sorted(used):
            L.append(f"- {m}　{MARKS.get(m, '—')}")
        L.append("")

    L.append("---")
    L.append("")
    L.append("*本底稿由 finagent 自动生成。异常判定来自确定性规则引擎，同一输入可复算；"
              "底稿内容仅为识别结果，最终审计结论需经项目合伙人复核。*")
    return "\n".join(L)


# ---------------------------------------------------------------- xlsx 渲染（可选依赖）


def render_workpaper_xlsx(wp: WorkingPaper, path: str) -> str:
    """导出为 xlsx。需要 openpyxl——它是**可选依赖**，装不上不影响其他功能。

    刻意不做静默降级：调用方明确要 xlsx 却拿到 Markdown，
    比直接报错更危险（使用者会以为交付物是表格）。
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.utils import get_column_letter
    except ImportError as e:  # pragma: no cover - 取决于环境
        raise RuntimeError(
            "导出 xlsx 需要 openpyxl。安装：pip install openpyxl\n"
            "若环境不允许装第三方包，请改用 render_workpaper_markdown()——"
            "它不依赖任何外部库，产出可直接粘贴进 Excel。"
        ) from e

    wb = Workbook()
    thin = Side(style="thin", color="999999")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    head_fill = PatternFill("solid", fgColor="EDEDED")
    head_font = Font(bold=True)
    wrap = Alignment(wrap_text=True, vertical="top")

    def _style_header(ws, row: int, ncols: int) -> None:
        for c in range(1, ncols + 1):
            cell = ws.cell(row=row, column=c)
            cell.fill = head_fill
            cell.font = head_font
            cell.border = border
            cell.alignment = wrap

    # --- Sheet 1：底稿首页 ---
    ws = wb.active
    ws.title = "底稿首页"
    rows = [
        ("底稿编号", wp.ref),
        ("底稿标题", wp.title),
        ("被审计单位", wp.entity),
        ("审计期间", wp.period),
        ("索引号", ", ".join(wp.index_refs()) or "—"),
        ("审计目标", wp.objective),
        ("编制依据", wp.basis),
        ("风险分", wp.meta.get("risk_score", "—")),
        ("风险等级", wp.meta.get("risk_grade", "—")),
        ("编制人", wp.signoff.prepared_by),
        ("编制日期", wp.signoff.prepared_date),
        ("复核人", wp.signoff.reviewed_by),
        ("复核日期", wp.signoff.reviewed_date),
    ]
    ws.append(["项目", "内容"])
    _style_header(ws, 1, 2)
    for k, v in rows:
        ws.append([k, v])
    for r in range(2, ws.max_row + 1):
        for c in (1, 2):
            ws.cell(row=r, column=c).border = border
            ws.cell(row=r, column=c).alignment = wrap
    ws.column_dimensions["A"].width = 16
    ws.column_dimensions["B"].width = 78

    # --- Sheet 2：审计过程 ---
    ws2 = wb.create_sheet("审计过程")
    ws2.append(["程序编号", "程序名称", "范围", "方法", "标识", "结果"])
    _style_header(ws2, 1, 6)
    for p in wp.procedures:
        ws2.append([p.code, p.name, p.scope_, p.method, " ".join(p.marks), p.result])
    for r in range(2, ws2.max_row + 1):
        for c in range(1, 7):
            ws2.cell(row=r, column=c).border = border
            ws2.cell(row=r, column=c).alignment = wrap
    for col, w in zip("ABCDEF", (10, 26, 24, 22, 8, 60)):
        ws2.column_dimensions[col].width = w

    # --- Sheet 3：审计发现 ---
    ws3 = wb.create_sheet("审计发现")
    ws3.append(["序号", "标识", "严重度", "发现事项", "规则编号", "观察线来源", "明细", "判定依据"])
    _style_header(ws3, 1, 8)
    for i, f in enumerate(wp.findings, 1):
        ws3.append([
            i, mark_for(f.level), f.level.label, f.title, f.rule_id,
            f.evidence.get("line_source", "absolute"), f.detail, f.basis,
        ])
    for r in range(2, ws3.max_row + 1):
        for c in range(1, 9):
            ws3.cell(row=r, column=c).border = border
            ws3.cell(row=r, column=c).alignment = wrap
    for col, w in zip("ABCDEFGH", (6, 6, 8, 30, 16, 14, 70, 44)):
        ws3.column_dimensions[col].width = w

    # --- Sheet 4：交叉索引 ---
    ws4 = wb.create_sheet("交叉索引")
    ws4.append(["本底稿位置", "指向", "说明"])
    _style_header(ws4, 1, 3)
    for c in wp.cross_refs:
        ws4.append([c.from_ref, c.to_ref, c.note])
    for r in range(2, ws4.max_row + 1):
        for c in range(1, 4):
            ws4.cell(row=r, column=c).border = border
            ws4.cell(row=r, column=c).alignment = wrap
    for col, w in zip("ABC", (16, 44, 30)):
        ws4.column_dimensions[col].width = w

    # --- Sheet 5：审计结论 ---
    ws5 = wb.create_sheet("审计结论")
    ws5.append(["审计结论"])
    _style_header(ws5, 1, 1)
    ws5.append([wp.conclusion])
    ws5.cell(row=2, column=1).alignment = wrap
    ws5.column_dimensions["A"].width = 100
    ws5.row_dimensions[2].height = 90

    wb.save(path)
    return path
