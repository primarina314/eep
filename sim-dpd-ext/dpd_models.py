#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dpd_models.py

DPD(디지털 프리디스토션) 모델들을 '플러그인' 형태로 교체할 수 있게 만든 모듈입니다.

지원하는 DPD 타입
-----------------
- MP-DPD  : Memory Polynomial, ILA(Indirect Learning Architecture) 기반 LS/Ridge 학습
- GMP-DPD : Generalized Memory Polynomial, ILA 기반 LS/Ridge 학습
- NN-DPD  : Neural Network(파이토치), ILA 방식(포스트디스토터 g(y) 학습 후 그대로 predistorter로 사용)

공통 인터페이스
---------------
모든 DPD는 아래 두 메서드를 제공합니다.

- fit_ila(x, y, ...)
    x: PA 입력(원 신호)
    y: PA 출력
    ILA: x ≈ g(y) 를 학습한 뒤 g를 predistorter로 사용

- forward(x_desired)
    x_desired: "선형"으로 보내고 싶은 신호
    return: predistorted u

주의
----
NN-DPD는 torch가 설치되어 있어야 합니다.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np

from pa_models import (
    mp_regressor,
    apply_mp,
    fit_mp,
    gmp_regressor,
    apply_gmp,
    fit_gmp,
)


def _as_1d(x: np.ndarray) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def mp_memory_vector(P: int, mem: int) -> np.ndarray:
    if P % 2 == 0:
        raise ValueError("P must be odd")
    K = (P + 1) // 2
    mem_i = int(mem)
    if mem_i <= 0:
        raise ValueError("mem must be >= 1")
    return np.full(K, mem_i, dtype=int)


class BaseDPD:
    """Minimal base class."""

    def fit_ila(self, x: np.ndarray, y: np.ndarray, **kwargs) -> "BaseDPD":
        raise NotImplementedError

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class MPDPD(BaseDPD):
    """Memory-Polynomial DPD (coeff learned by ILA)."""

    P: int
    M: np.ndarray
    ridge: float = 1e-6
    coeff: Optional[np.ndarray] = None

    def fit_ila(self, x: np.ndarray, y: np.ndarray, ridge: Optional[float] = None) -> "MPDPD":
        x = _as_1d(x)
        y = _as_1d(y)
        r = float(self.ridge if ridge is None else ridge)
        self.coeff = fit_mp(input_signal=y, target_signal=x, P=self.P, M=self.M, ridge=r)
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.coeff is None:
            raise RuntimeError("MPDPD.coeff is None. Call fit_ila() first.")
        x = _as_1d(x)
        return apply_mp(x, self.coeff, P=self.P, M=self.M)


@dataclass
class GMPDPD(BaseDPD):
    """GMP DPD (coeff learned by ILA)."""

    P: int
    M: int
    L_lag: int = 0
    L_lead: int = 0
    ridge: float = 1e-6
    coeff: Optional[np.ndarray] = None

    def fit_ila(self, x: np.ndarray, y: np.ndarray, ridge: Optional[float] = None) -> "GMPDPD":
        x = _as_1d(x)
        y = _as_1d(y)
        r = float(self.ridge if ridge is None else ridge)
        self.coeff = fit_gmp(y, x, P=self.P, M=self.M, L_lag=self.L_lag, L_lead=self.L_lead, ridge=r)
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.coeff is None:
            raise RuntimeError("GMPDPD.coeff is None. Call fit_ila() first.")
        x = _as_1d(x)
        return apply_gmp(x, self.coeff, P=self.P, M=self.M, L_lag=self.L_lag, L_lead=self.L_lead)


class NeuralNetDPD(BaseDPD):
    """Neural-network DPD.

    구조:
      입력: 최근 M개 샘플의 I/Q (shape [B, 2*M])
      출력: 현재 u[n]의 I/Q (shape [B, 2])

    학습(ILA):
      postdistorter g(y) ≈ x 를 학습한 후,
      predistorter로 u = g(x_desired) 사용.

    """

    def __init__(
        self,
        M: int = 5,
        hidden: int = 128,
        depth: int = 3,
        device: str = "cpu",
        ridge_like: float = 0.0,
    ):
        try:
            import torch
            import torch.nn as nn
        except Exception as e:
            raise ImportError("NeuralNetDPD requires PyTorch (torch).") from e

        self.torch = torch
        self.nn = nn
        self.M = int(M)
        self.device = torch.device(device)
        self.ridge_like = float(ridge_like)

        layers: List[nn.Module] = []
        in_dim = 2 * self.M
        if depth <= 1:
            layers.append(nn.Linear(in_dim, 2))
        else:
            for i in range(depth - 1):
                layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers).to(self.device)

    def _make_dataset(self, xin: np.ndarray, yout: np.ndarray) -> Tuple["torch.Tensor", "torch.Tensor"]:
        torch = self.torch
        xin = _as_1d(xin)
        yout = _as_1d(yout)
        N = len(xin)

        X = np.zeros((N, self.M), dtype=complex)
        for m in range(self.M):
            if m == 0:
                X[:, m] = xin
            else:
                X[m:, m] = xin[:-m]
                X[:m, m] = 0.0

        Xin = np.concatenate([X.real, X.imag], axis=1).astype(np.float32)
        Yout = np.stack([yout.real, yout.imag], axis=1).astype(np.float32)
        return torch.from_numpy(Xin).to(self.device), torch.from_numpy(Yout).to(self.device)

    def fit_ila(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lr: float = 1e-3,
        epochs: int = 5,
        batch: int = 4096,
        seed: int = 0,
    ) -> "NeuralNetDPD":
        """Fit postdistorter g(y) ≈ x using MSE."""
        torch = self.torch
        torch.manual_seed(int(seed))

        Xin, Yout = self._make_dataset(y, x)  # input=y, target=x

        opt = torch.optim.Adam(self.net.parameters(), lr=float(lr))
        loss_fn = torch.nn.MSELoss()

        N = Xin.shape[0]
        for _ep in range(int(epochs)):
            perm = torch.randperm(N, device=self.device)
            for i in range(0, N, int(batch)):
                idx = perm[i : i + int(batch)]
                pred = self.net(Xin[idx])
                loss = loss_fn(pred, Yout[idx])

                # optional tiny weight decay (ridge-like)
                if self.ridge_like > 0:
                    l2 = 0.0
                    for p in self.net.parameters():
                        l2 = l2 + (p * p).sum()
                    loss = loss + self.ridge_like * l2

                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply predistorter u = g(x)."""
        torch = self.torch
        x = _as_1d(x)
        N = len(x)

        X = np.zeros((N, self.M), dtype=complex)
        for m in range(self.M):
            if m == 0:
                X[:, m] = x
            else:
                X[m:, m] = x[:-m]
                X[:m, m] = 0.0

        Xin = np.concatenate([X.real, X.imag], axis=1).astype(np.float32)
        Xin_t = torch.from_numpy(Xin).to(self.device)
        with torch.no_grad():
            u2 = self.net(Xin_t).cpu().numpy()
        u = u2[:, 0] + 1j * u2[:, 1]
        return u
