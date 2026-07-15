from __future__ import annotations

from cryptoquant_auto.risk.kill_switch import KillSwitch, KillLevel


def test_warn_band_daily_pnl():
    # 【P1-3】日亏中段预警带：落入 -1%~-3% 应进入 WARN，不触发 L1 暂停。
    ks = KillSwitch()
    lvl = ks.update(daily_pnl=-0.02)          # 中段
    assert lvl == KillLevel.WARN, lvl
    assert ks.allows_new() is True           # WARN 仍允许新开


def test_normal_when_flat():
    ks = KillSwitch()
    assert ks.update(daily_pnl=0.0) == KillLevel.NORMAL
    assert ks.update(daily_pnl=-0.005) == KillLevel.NORMAL  # 未达 -1% 预警线


def test_l1_hard_trigger_at_threshold():
    # 越过 -3% 硬阈值 → L1 暂停新开（与预警带不冲突）。
    ks = KillSwitch()
    assert ks.update(daily_pnl=-0.03) == KillLevel.L1_PAUSE_NEW
    assert ks.allows_new() is False


def test_warn_other_signals():
    # 中段信号：连亏 2 / 回撤 -3% / 波动 1.8σ / API 失败 5% 都应进 WARN。
    cases = dict(
        loss_streak=2, peak_dd=-0.03, btc_vol_sigma=1.8, api_fail_rate=0.05,
    )
    for k, v in cases.items():
        ks = KillSwitch()
        assert ks.update(**{k: v}) == KillLevel.WARN, (k, v)


def test_reduce_mode_false_on_warn():
    # WARN 不应被算作降险（reduce_mode 仅 L2+）。
    ks = KillSwitch()
    ks.update(daily_pnl=-0.02)
    assert ks.reduce_mode() is False
