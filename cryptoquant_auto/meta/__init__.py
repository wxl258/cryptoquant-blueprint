"""元认知层：三级环境聚合 + FinMem 分层记忆 + 四角色智能体。

- cognition : 规则式三级环境评分（阶段1 复用）
- memory    : FinMem 分层记忆（阶段3 任务16，替代裸 reflection）
- agents    : 接地 LLM 四角色（阶段3 任务18）
"""
from .cognition import assess, EnvAssessment, EnvRecord, record_env, load_env_history
from .memory import FinMemMemory, ReflectionLog, Profile, Episode, Insight
from .agents import (FourRoleCouncil, CouncilVerdict, Analyst, Researcher,
                     DecisionMaker, RiskController, LONG, SHORT, HOLD)

__all__ = [
    "assess", "EnvAssessment", "EnvRecord", "record_env", "load_env_history",
    "FinMemMemory", "ReflectionLog", "Profile", "Episode", "Insight",
    "FourRoleCouncil", "CouncilVerdict", "Analyst", "Researcher",
    "DecisionMaker", "RiskController", "LONG", "SHORT", "HOLD",
]
