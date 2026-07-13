"""信号层：自包含多因子评分引擎 + 均值回归（吸收生产系统 signals/ 优点 P0-A/P0-B）。"""
from .indicators import calc_adx, calc_rsi, calc_atr, volatility_regime
from .engine import gen_signal, SignalCandidate, MarketContext
from .mean_reversion import gen_mean_reversion, MeanReversionSignal
from .generator import generate_signals, candidate_to_signal

__all__ = [
    "calc_adx", "calc_rsi", "calc_atr", "volatility_regime",
    "gen_signal", "SignalCandidate", "MarketContext",
    "gen_mean_reversion", "MeanReversionSignal",
    "generate_signals", "candidate_to_signal",
]
