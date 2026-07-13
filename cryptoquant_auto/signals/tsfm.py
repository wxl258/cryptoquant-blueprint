"""TSFM 时序基础模型骨架（蓝图阶段4 · 任务20 · 专家A）。

蓝图锚点：Time-MoE / Moirai 等预训练时序基础模型（零样本 + 原生预测区间）。
沙盒现实：无外网下载预训练权重，且 torch 虽可用但**必须保留降级路径**。

落地（诚实原型）：
  - TSFMForecaster   ：统一接口 fit(returns) / forecast(recent,h) → (point,lower,upper)
  - DistilledTSFM    ：纯 numpy 滞后岭回归 + 残差经验分位区间（常驻可用，零依赖）
  - TorchTSFM        ：torch 小蒸馏版（本地数据零样本拟合的小 MLP），原生点预测；
                      区间复用残差分位法（与 SPCI 共形互补）。**torch 缺失 → 构造即降级**
  - make_tsfm(backend)：auto 优先 torch，失败回退 numpy（纪律：torch 可选 + 降级）

原生预测区间与 risk/conformal.SPCI 天然互补：TSFM 给「点预测±分位带」，SPCI 给
「时依赖条件分位惊喜度」做不确定度软降级。

零依赖纪律：模块级仅 import numpy；torch 在 TorchTSFM 内惰性 import，缺失即降级。
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


class TSFMForecaster:
    """统一接口（所有后端共用契约）。"""

    name = "base"

    def fit(self, returns: np.ndarray) -> "TSFMForecaster":
        raise NotImplementedError

    def forecast(self, recent: np.ndarray, horizon: int = 1
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (point, lower, upper)，各长度 = horizon。"""
        raise NotImplementedError

    # ---- 区间覆盖评估（与 SPCI 共形同语义：经验覆盖率应≈标称 1-α）----
    def coverage(self, test_returns: np.ndarray, alpha: float = 0.10) -> float:
        """在 test_returns 上滚动 1 步预测，统计真值落入区间的比例。"""
        if len(test_returns) < self._lookback + 2:
            return 0.0
        hits = 0
        total = 0
        for t in range(self._lookback, len(test_returns)):
            recent = test_returns[t - self._lookback: t]
            pt, lo, hi = self.forecast(recent, horizon=1)
            actual = test_returns[t]
            total += 1
            if lo[0] <= actual <= hi[0]:
                hits += 1
        return hits / total if total else 0.0


class DistilledTSFM(TSFMForecaster):
    """纯 numpy 小蒸馏版：滞后岭回归 + 残差经验分位区间（常驻可用）。"""

    name = "distilled_numpy"

    def __init__(self, lookback: int = 24, alpha: float = 0.10, ridge: float = 1e-2,
                 seed: int = 0):
        self._lookback = lookback
        self._alpha = alpha
        self._ridge = ridge
        self._seed = seed
        self._coef = None          # 岭回归系数 (lookback,)
        self._intercept = 0.0
        self._q_lo = -1e-3
        self._q_hi = 1e-3

    def fit(self, returns: np.ndarray) -> "DistilledTSFM":
        r = np.asarray(returns, float)
        L = self._lookback
        if len(r) <= L + 2:
            # 数据不足：退化为均值预测
            self._intercept = float(r.mean()) if len(r) else 0.0
            self._coef = np.zeros(L)
            self._q_lo = self._q_hi = 0.0
            return self
        X, y = [], []
        for t in range(L, len(r) - 1):
            X.append(r[t - L: t])
            y.append(r[t])
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        # 岭回归闭式解
        A = X.T @ X + self._ridge * np.eye(L)
        b = X.T @ y
        self._coef = np.linalg.solve(A, b)
        self._intercept = float(y.mean() - self._coef @ X.mean(axis=0))
        resid = y - (X @ self._coef + self._intercept)
        self._q_lo = float(np.quantile(resid, self._alpha / 2))
        self._q_hi = float(np.quantile(resid, 1 - self._alpha / 2))
        return self

    def forecast(self, recent: np.ndarray, horizon: int = 1
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = np.asarray(recent, float)[-self._lookback:]
        if len(r) < self._lookback:
            r = np.pad(r, (self._lookback - len(r), 0), constant_values=0.0)
        point = np.zeros(horizon)
        cur = r.copy()
        for h in range(horizon):
            nxt = float(self._coef @ cur + self._intercept)
            point[h] = nxt
            # 多步：把预测值滚动进窗口（朴素递推）
            cur = np.roll(cur, -1)
            cur[-1] = nxt
        # 区间：残差分位 × √步长（独立残差假设）
        scale = np.sqrt(np.arange(1, horizon + 1))
        lo = point + self._q_lo * scale
        hi = point + self._q_hi * scale
        return point, lo, hi


class TorchTSFM(TSFMForecaster):
    """torch 小蒸馏版（本地数据零样本拟合）。torch 缺失 → 构造即抛 ImportError。

    原型期用一个小 MLP 替代巨型预训练 MoE；真实 Time-MoE/Moirai 权重经
    load_pretrained() 在阶段3-4 云环境注入（当前沙盒无外网 → 该钩子抛 NotImplementedError）。
    """

    name = "distilled_torch"

    def __init__(self, lookback: int = 24, alpha: float = 0.10,
                 hidden: int = 24, epochs: int = 60, lr: float = 1e-2,
                 seed: int = 0):
        try:
            import torch  # 惰性 import：缺失即降级
        except ImportError as e:
            raise ImportError("torch 不可用，TSFM 降级到 DistilledTSFM（numpy）") from e
        self._torch = torch
        self._lookback = lookback
        self._alpha = alpha
        self._hidden = hidden
        self._epochs = epochs
        self._lr = lr
        self._seed = seed
        self._net = None
        self._q_lo = -1e-3
        self._q_hi = 1e-3
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_pretrained(self, weights_path: str):
        """注入真实 Time-MoE/Moirai 权重（阶段3-4 云环境）。沙盒无外网 → 抛错。"""
        raise NotImplementedError(
            f"沙盒无外网，无法加载预训练权重 {weights_path!r}；"
            f"原型用 torch 小蒸馏版替代，真实权重留待云环境注入。")

    def fit(self, returns: np.ndarray) -> "TorchTSFM":
        torch = self._torch
        torch.manual_seed(self._seed)
        r = np.asarray(returns, float)
        L = self._lookback
        if len(r) <= L + 2:
            self._intercept = float(r.mean()) if len(r) else 0.0
            self._coef = np.zeros(L)
            self._q_lo = self._q_hi = 0.0
            # 仍建一个最小网络占位（保证 forecast 可用）
            self._net = torch.nn.Linear(L, 1).to(self._device)
            return self
        X, y = [], []
        for t in range(L, len(r) - 1):
            X.append(r[t - L: t])
            y.append(r[t])
        Xt = torch.tensor(np.array(X, dtype=float), dtype=torch.float32, device=self._device)
        yt = torch.tensor(np.array(y, dtype=float).reshape(-1, 1), dtype=torch.float32,
                          device=self._device)
        net = torch.nn.Sequential(
            torch.nn.Linear(L, self._hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(self._hidden, 1),
        ).to(self._device)
        opt = torch.optim.Adam(net.parameters(), lr=self._lr)
        loss_fn = torch.nn.MSELoss()
        for _ in range(self._epochs):
            opt.zero_grad()
            pred = net(Xt)
            loss = loss_fn(pred, yt)
            loss.backward()
            opt.step()
        self._net = net
        # 残差经验分位（区间）
        with torch.no_grad():
            resid = (yt - net(Xt)).cpu().numpy().ravel()
        self._q_lo = float(np.quantile(resid, self._alpha / 2))
        self._q_hi = float(np.quantile(resid, 1 - self._alpha / 2))
        return self

    def forecast(self, recent: np.ndarray, horizon: int = 1
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        torch = self._torch
        r = np.asarray(recent, float)[-self._lookback:]
        if len(r) < self._lookback:
            r = np.pad(r, (self._lookback - len(r), 0), constant_values=0.0)
        point = np.zeros(horizon)
        cur = r.copy()
        self._net.eval()
        with torch.no_grad():
            for h in range(horizon):
                x = torch.tensor(cur.reshape(1, -1), dtype=torch.float32,
                                 device=self._device)
                nxt = float(self._net(x).cpu().numpy().ravel()[0])
                point[h] = nxt
                cur = np.roll(cur, -1)
                cur[-1] = nxt
        scale = np.sqrt(np.arange(1, horizon + 1))
        lo = point + self._q_lo * scale
        hi = point + self._q_hi * scale
        return point, lo, hi


class PretrainedTSFM(TSFMForecaster):
    """真实预训练时序基础模型（Time-MoE / Moirai）零样本预测。

    沙箱无外网 + 无 timemoe/moirai 包：load_pretrained 抛 NotImplementedError。
    生产服务器（有外网 + GPU）部署时启用；缺失则 make_tsfm('pretrained') 自动降级 numpy。
    """

    name = "pretrained"

    def __init__(self, lookback: int = 24, alpha: float = 0.10,
                 model: str = "moirai", repo_id: Optional[str] = None,
                 cache_dir: str = "data/tsfm_cache"):
        self._lookback = lookback
        self._alpha = alpha
        self._backend = model              # "moirai" | "timemoe"
        self._repo_id = repo_id
        self._cache_dir = cache_dir
        self._model = None
        self._q_lo = -1e-3
        self._q_hi = 1e-3

    def load_pretrained(self, repo_id: Optional[str] = None):
        """在【有外网】的生产服务器下载并加载预训练权重。

        任何下载/认证/仓库/依赖失败都收敛为 NotImplementedError，便于 harness 干净降级。
        """
        repo_id = repo_id or self._repo_id
        if not repo_id:
            raise ValueError("须指定 repo_id（如 Salesforce/moirai-1.0-R-large）")
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise NotImplementedError(
                "生产环境需 pip install huggingface_hub timemoe（或 moirai）；"
                "沙箱无相关依赖，无法加载预训练权重") from e
        try:
            path = snapshot_download(repo_id, cache_dir=self._cache_dir)
            if self._backend == "moirai":
                from moirai.model.moirai import MoiraiForecast  # 视版本而定
                self._model = MoiraiForecast.from_pretrained(path)
            elif self._backend == "timemoe":
                from timemoe import TimeMoeForCausalLM
                self._model = TimeMoeForCausalLM.from_pretrained(path)
            else:
                raise ValueError(f"未知 backend: {self._backend}")
        except Exception as e:
            # 网络/认证/仓库不存在/依赖缺失 → 统一收敛，harness 据此降级 numpy
            raise NotImplementedError(
                f"预训练权重加载失败（网络/认证/仓库/依赖）：{e}；"
                f"请在【有外网】的生产服务器执行，并确认 repo_id 与 timemoe/moirai 已装。"
            ) from e
        return self

    def fit(self, returns: np.ndarray) -> "PretrainedTSFM":
        """零样本：预训练模型无需训练，仅缓存残差分位用于区间。"""
        r = np.asarray(returns, float)
        if len(r) >= 2:
            resid = r - np.roll(r, 1)
            resid = resid[1:]
            self._q_lo = float(np.quantile(resid, self._alpha / 2))
            self._q_hi = float(np.quantile(resid, 1 - self._alpha / 2))
        return self

    def forecast(self, recent: np.ndarray, horizon: int = 1):
        if self._model is None:
            raise RuntimeError("先调用 load_pretrained()（仅生产服务器有外网）")
        # TODO(prod): 在此调用真实模型推理，返回 (point, lo, hi)
        #   point = self._model.predict(context=recent[-self._lookback:], horizon=horizon)
        #   lo/hi 用 self._q_lo/self._q_hi 分位带（或模型原生区间）
        raise NotImplementedError(
            "真实模型前向须在生产服务器按官方 API 实现（见 load_pretrained 注释）。"
            "接口已对齐：返回 (point, lower, upper)。")


def make_tsfm(backend: str = "auto", **kw) -> TSFMForecaster:
    """统一入口：auto 优先 torch，缺失/失败 → numpy 降级（纪律：torch 可选 + 降级）。

    backend='pretrained'：返回真实权重骨架；权重加载须在生产服务器执行，
    加载失败/缺失时由调用方降级到 numpy（见 run_validation_stage4 的 A/B 降级逻辑）。
    """
    if backend == "numpy":
        return DistilledTSFM(**kw)
    if backend == "torch":
        try:
            return TorchTSFM(**kw)
        except ImportError:
            return DistilledTSFM(**kw)   # 降级路径
    if backend == "pretrained":
        try:
            return PretrainedTSFM(**kw)
        except Exception:
            return DistilledTSFM(**kw)   # 降级路径
    # auto
    try:
        return TorchTSFM(**kw)
    except ImportError:
        return DistilledTSFM(**kw)
