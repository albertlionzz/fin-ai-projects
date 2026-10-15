"""A/B 对比：同一批 findings，模板研判 vs LLM 研判，哪个更可用。

为什么需要这一层
--------------
`ReportAgent` 有两条路：无 LLM 走确定性聚合模板，有 LLM 走模型生成。
README 里写着"未接真实 LLM 验证，生成质量未知"——这个模块就是把那句话兑现。

对比维度（全部可机器判定，不靠"读起来感觉"）
------------------------------------------
1. **事实保真度**：输出里每个数字必须能在 findings 里找到出处。
   财务场景里一个编造的数字就是事故，这一项的权重远高于其他。
2. **覆盖率**：每条 finding 是否都被提及。漏报比啰嗦危险。
3. **复述率**：finding.detail 的原文有多大比例被原样抄进输出。
   模板刻意做到接近 0（研判章节不能是明细的复述，这是我们自己修过的 bug）；
   模型如果只是换措辞重抄一遍，增量价值就是零。
4. **可复算性**：同一 prompt 跑 k 次，输出是否逐字节相同。
   这是整个项目的核心主张——判定要可复算。研判是措辞环节可以容忍波动，
   但波动幅度必须被量化，不能只说"可能有差异"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .models import Finding, Level


# ---------------------------------------------------------------- 数字抽取

# 匹配数字，含千分位逗号、小数、百分号。负号可选。
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")

# 行首列表标记 / 项目符号，抽取数字前先剥掉，否则 "1. xxx" 的 1 会被当成事实数字。
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[#>*\-+]|\d+\s*[.、)）])\s*")


def _to_float(tok: str) -> float | None:
    """把 "1,234.5%" / "56.3" 之类的 token 转成 float。失败返回 None。"""
    t = tok.replace(",", "").rstrip("%")
    if not t or t in ("-", "+", "."):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def extract_numbers(text: str) -> list[float]:
    """抽取文本中的全部数字（去重、保序）。

    先逐行剥掉列表标记，避免序号被误判为事实数字。
    """
    out: list[float] = []
    seen: set[float] = set()
    for raw in text.splitlines():
        line = _LIST_PREFIX_RE.sub("", raw)
        for m in _NUM_RE.finditer(line):
            v = _to_float(m.group(0))
            if v is None:
                continue
            if v not in seen:
                seen.add(v)
                out.append(v)
    return out


# ---------------------------------------------------------------- 事实保真度


def _collect_evidence_numbers(obj: Any, pool: set[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        pool.add(float(obj))
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_evidence_numbers(v, pool)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            _collect_evidence_numbers(v, pool)
    elif isinstance(obj, str):
        for v in extract_numbers(obj):
            pool.add(v)


def number_pool(
    findings: Sequence[Finding],
    n_high: int | None = None,
    n_warn: int | None = None,
) -> set[float]:
    """从 findings 反推"允许出现的数字"集合。

    三个来源，缺一不可：

    1. **detail / title / basis 的正文数字**——结论里引用的原始事实。
    2. **evidence 里的数值**——结构化证据。LLM 常把 evidence 的数字
       写进正文，这些数字是 grounded，不能算幻觉。
    3. **发现计数与列表序号**——"重大 2 项"里的 2、"共 7 项"里的 7。
       这些数字由 findings 本身决定，是合法的推导值。
       漏掉这一源会把所有计数表述都判成幻觉，误报率会高到没法用。
    """
    pool: set[float] = set()
    for f in findings:
        for text in (f.title, f.detail, f.basis or ""):
            for v in extract_numbers(text or ""):
                pool.add(v)
        _collect_evidence_numbers(f.evidence, pool)

    pool.add(float(len(findings)))
    for c in (n_high, n_warn):
        if c is not None:
            pool.add(float(c))
    for i in range(1, len(findings) + 2):
        pool.add(float(i))
    return pool


@dataclass
class FidelityReport:
    """事实保真度：输出中的数字有多少能在 findings 里找到出处。"""

    total: int
    grounded: int
    ungrounded: list[float] = field(default_factory=list)

    @property
    def score(self) -> float:
        return 1.0 if self.total == 0 else self.grounded / self.total

    def to_dict(self) -> dict:
        return {
            "total_numbers": self.total,
            "grounded": self.grounded,
            "ungrounded": self.ungrounded,
            "score": round(self.score, 4),
        }


def check_fidelity(
    text: str,
    findings: Sequence[Finding],
    n_high: int | None = None,
    n_warn: int | None = None,
) -> FidelityReport:
    """核查输出里的每个数字是否有出处。

    未标注出处的数字逐个列出——这是给人工复核的清单，
    不是只给一个分数。财务工具要能指出"具体哪个数字有问题"。
    """
    pool = number_pool(findings, n_high=n_high, n_warn=n_warn)
    nums = extract_numbers(text)
    ungrounded = [v for v in nums if v not in pool]
    return FidelityReport(
        total=len(nums),
        grounded=len(nums) - len(ungrounded),
        ungrounded=ungrounded,
    )


# ---------------------------------------------------------------- 覆盖率

# 汉字（CJK 统一表意文字）。用于"标题独有汉字"判定。
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass
class CoverageReport:
    """覆盖率：每条 finding 是否都在输出里被提及。"""

    total: int
    covered: int
    missing: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return 1.0 if self.total == 0 else self.covered / self.total

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "covered": self.covered,
            "missing": self.missing,
            "score": round(self.score, 4),
        }


def _strip_ws(s: str) -> str:
    """去掉全部空白。模型输出常把标题里的 "5 项" 写成 "5项"，
    精确子串匹配不该败在一个空格上。"""
    return re.sub(r"\s+", "", s or "")


def _title_chars(title: str) -> set[str]:
    """标题里的汉字集合。

    数字与标点不进：改写时数字通常会被保留（那是保真度检查的地盘），
    被同义替换掉的恰是汉字。
    """
    return set(_CJK_RE.findall(title or ""))


def _finding_numbers(f: Finding) -> set[float]:
    """单条 finding 自己的数字池：标题/依据正文 + evidence。

    与 `number_pool` 的区别：不含计数类全局数字（发现总数、列表序号），
    那些不属于任何一条发现，混进"独有数字"判定会互相污染。
    """
    pool: set[float] = set()
    for text in (f.title, f.detail, f.basis or ""):
        pool.update(extract_numbers(text or ""))
    _collect_evidence_numbers(f.evidence, pool)
    return pool


def _is_weak_number(v: float) -> bool:
    """区分度不足的数字，不进"独有数字"判定。两类：

    - 年份（1900–2100 的整数）：每份财务输出都会带报告期，
      命中它证明不了讨论过某条发现；
    - 10 以下的整数：计数类（"3 项指标""连续 4 期""权重 2"），
      行文里随处可见。本系统的指标值（百分比/比率/金额）总是小数
      或大量级，不受这条影响。
    """
    if not float(v).is_integer():
        return False
    iv = int(v)
    return 1900 <= iv <= 2100 or iv < 10


def check_coverage(text: str, findings: Sequence[Finding]) -> CoverageReport:
    """一条 finding 算"被提及"的条件（满足其一）：

    1. 标题（忽略空白差异）出现在输出里——stub 与保守模型的路径；
    2. 规则编号出现在输出里——模型回抄 ID 的路径；
    3. 标题的**独有汉字** ≥80% 出现，或该发现的**独有指标数字**出现——
       LLM 同义改写的路径。

    第 3 条是接真实模型后才补的。初版只有精确标题匹配，而 docstring 写着
    "LLM 完全可能用同义改写而不照抄标题"——实现与意图不符。stub 按构造
    照抄标题所以恒 100%；真实模型一改写措辞就整条判漏：实测 qwen3:4b 对
    示例制造 7 项发现的输出逐条覆盖了全部 7 项，旧指标报 0%。
    覆盖率对这个指标的主要用户群恒为 0，等于没有这个指标。

    为什么信"独有汉字/独有数字"：讨论一条发现，几乎不可能不碰它标题里
    的汉字组合或它特有的指标值。跨标题共享的词（"营收""净利率"）与共享的
    数字（营收增速 8.0% 同时出现在两条发现里）在去重阶段已被剔除，
    剩下的才构成证据。80% 的阈值留给"同向恶化→恶化"这类个别字词丢失。
    """
    bare = _strip_ws(text)
    char_sets = [_title_chars(f.title) for f in findings]
    char_union = set().union(*char_sets) if char_sets else set()
    num_pools = [_finding_numbers(f) for f in findings]
    num_union = set().union(*num_pools) if num_pools else set()
    text_nums = set(extract_numbers(text))

    missing: list[str] = []
    covered = 0
    for f, chars, nums in zip(findings, char_sets, num_pools):
        title_bare = _strip_ws(f.title)
        distinctive_chars = chars - (char_union - chars)
        distinctive_nums = {
            v for v in (nums - (num_union - nums)) if not _is_weak_number(v)
        }
        char_ratio = (
            sum(1 for c in distinctive_chars if c in text) / len(distinctive_chars)
            if len(distinctive_chars) >= 2 else 0.0
        )
        if (
            (title_bare and title_bare in bare)
            or (f.rule_id and f.rule_id in text)
            or char_ratio >= 0.8
            or bool(distinctive_nums & text_nums)
        ):
            covered += 1
        else:
            missing.append(f.rule_id or (f.title or "").strip())
    return CoverageReport(total=len(findings), covered=covered, missing=missing)


# ---------------------------------------------------------------- 复述率


def check_restatement(text: str, findings: Sequence[Finding]) -> float:
    """复述率：finding.detail 的原文被原样抄进输出的比例。

    这是"研判章节不能是明细的复述"这条约束的量化版。
    我们自己修过这个 bug（第一版模板就是逐条复述 detail），
    现在要能量化它，才能对比模板与模型谁在这件事上做得更好。

    实现：逐条 detail 按句切分，统计有多少句原样出现在输出里。
    用句子而非整段，是因为模型常常抄一半改一半——按句算能捕捉到部分复述。
    """
    if not findings:
        return 0.0
    total_sents = 0
    copied_sents = 0
    for f in findings:
        detail = (f.detail or "").strip()
        if not detail:
            continue
        # 按中文句读 / 分号切句，过滤太短的碎片
        sents = [s.strip() for s in re.split(r"[。；！？]", detail) if len(s.strip()) >= 6]
        for s in sents:
            total_sents += 1
            if s in text:
                copied_sents += 1
    if total_sents == 0:
        return 0.0
    return copied_sents / total_sents


# ---------------------------------------------------------------- 可复算性


def determinism_ratio(outputs: Sequence[str]) -> float:
    """k 次运行中输出完全一致的比例。

    1.0 = 完全可复算；0.0 = 每次都不一样。
    取"最常见的输出出现次数 / 总次数"，而不是"是否有一次与首次相同"——
    后者会把"偶尔一致"误判成稳定。
    """
    if not outputs:
        return 1.0
    counts: dict[str, int] = {}
    for o in outputs:
        counts[o] = counts.get(o, 0) + 1
    return max(counts.values()) / len(outputs)


# ---------------------------------------------------------------- 汇总


@dataclass
class ComparisonRow:
    """一个场景下一个研判器的全部指标。"""

    scenario: str
    reasoner: str
    fidelity: FidelityReport
    coverage: CoverageReport
    restatement: float
    length: int
    determinism: float
    n_runs: int

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "reasoner": self.reasoner,
            "fidelity": self.fidelity.to_dict(),
            "coverage": self.coverage.to_dict(),
            "restatement": round(self.restatement, 4),
            "length": self.length,
            "determinism": round(self.determinism, 4),
            "n_runs": self.n_runs,
        }


def evaluate(
    scenario: str,
    reasoner_name: str,
    outputs: Sequence[str],
    findings: Sequence[Finding],
) -> ComparisonRow:
    """对一组输出算全部指标。outputs 长度即重复运行次数。"""
    if not outputs:
        raise ValueError("outputs 不能为空")
    text = outputs[0]
    n_high = sum(1 for f in findings if f.level == Level.HIGH)
    n_warn = sum(1 for f in findings if f.level == Level.WARN)
    return ComparisonRow(
        scenario=scenario,
        reasoner=reasoner_name,
        fidelity=check_fidelity(text, findings, n_high=n_high, n_warn=n_warn),
        coverage=check_coverage(text, findings),
        restatement=check_restatement(text, findings),
        length=len(text),
        determinism=determinism_ratio(outputs),
        n_runs=len(outputs),
    )
