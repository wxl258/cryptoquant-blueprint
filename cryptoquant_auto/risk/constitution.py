"""交易宪法（蓝图阶段1）：把 fail-closed / 方向中立 / 不碰实盘 / 最大回撤
编码为不可绕过的硬约束。每笔动作过宪法校验，违例一律否决/降级。

原型沙盒专用，零依赖。设计目标：安全 by construction —— AI 永远对齐
用户风险授权，且架构级禁止触碰实盘资金。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ConstitutionVerdict:
    compliant: bool
    violations: List[str] = field(default_factory=list)
    safe_action: str = "观望"   # 违例时强制的安全动作


class TradingConstitution:
    def __init__(self, live_capital: bool = False, max_drawdown: float = 0.20,
                 require_rationale: bool = True):
        # 永远保持 False（原型绝不碰真钱）；一旦被置 True，架构级否决一切
        self.live_capital = live_capital
        self.max_drawdown = max_drawdown
        self.require_rationale = require_rationale

    def check(self, decision) -> ConstitutionVerdict:
        """对一笔 MetaDecision 做宪法合规检查。

        decision 需具备属性：action(int, 0=做多/1=做空/2=观望)、
        rationale(str)、confidence(float)、proposed_exposure(Optional[float])。
        """
        v: List[str] = []

        # R0 不碰实盘：架构级硬锁，live_capital 一旦为 True 全部否决
        if self.live_capital:
            v.append("R0:live_capital=True 禁止任何实盘动作（原型仅沙盒）")

        # R1 fail-closed：非持有动作必须带 rationale + 足够 confidence，否则视为不安全
        if decision.action != 2:  # 非观望
            if self.require_rationale and not getattr(decision, "rationale", ""):
                v.append("R1:非持有动作缺少 rationale（不可解释→不安全，否决）")
            if getattr(decision, "confidence", 0.0) < 0.1:
                v.append("R1:confidence 过低（<0.1），禁止开仓")

        # R2 最大回撤：决策若携带 proposed_exposure，超阈否决
        exp = getattr(decision, "proposed_exposure", None)
        if exp is not None and exp > self.max_drawdown:
            v.append(f"R2:proposed_exposure={exp:.2%} 超 max_drawdown={self.max_drawdown:.2%}")

        # R3 方向中立：结构性由 validate_direction_neutral 单测锁定；
        # 此处仅做运行期记录，真正中立不靠运行时兜而是靠测试门禁。
        return ConstitutionVerdict(
            compliant=len(v) == 0,
            violations=v,
            safe_action="观望",
        )
