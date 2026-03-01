#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pa_models.py

PA(전력증폭기) 행동모델들을 '플러그인' 형태로 교체할 수 있게 만든 모듈입니다.

- Memory Polynomial (MP)
- Generalized Memory Polynomial (GMP)
- Wiener (LTI filter + memoryless nonlinearity)
- Neural Net (PyTorch, 선택)

현재 제가 제공드린 dpd_siso_sim.py / hybrid_mimo_dpd_sim.py 에서는
'MP(메모리 폴리노미얼)' 형태의 PA를 사용합니다.

이 모듈을 사용하면, 시뮬레이터 코드에서
  y = pa.forward(x)
만 호출하도록 만들어두고, pa 객체만 바꿔 끼우면 됩니다.

주의
----
- MATLAB/Lund testbed 계수( PA_memory_models.mat / P_TX_Case1.mat )를 로드하는 기능은
  scipy.io.loadmat 기반입니다.
- NeuralNetPA는 torch가 설치되어 있을 때만 동작합니다(없으면 ImportError).

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np


# =========================
# 공통 유틸
# =========================

def clip_complex(sig: np.ndarray, peak: float) -> np.ndarray:
    """Magnitude clipper with phase preserved."""
    sig = np.asarray(sig)
    if peak <= 0:
        return sig
    mag = np.abs(sig)
    scale = np.minimum(1.0, peak / (mag + 1e-12))
    return sig * scale


# =========================
# 1) Memory Polynomial (MP)
# =========================

def mp_regressor(x: np.ndarray, P: int, M: np.ndarray) -> np.ndarray:
    """Regressor for Memory Polynomial (odd orders only).

    y[n] = sum_{p_idx=0..K-1} sum_{m=0..M[p_idx]-1} c[p_idx,m] * x[n-m] * |x[n-m]|^{2 p_idx}

    where P is the maximum odd order (1,3,...,P), K=(P+1)/2.
    M[p_idx] is the memory length for order (2*p_idx+1).
    """
    x = np.asarray(x).reshape(-1)
    N = len(x)
    if P % 2 == 0:
        raise ValueError("P must be odd")
    K = (P + 1) // 2

    M = np.asarray(M).astype(int).reshape(-1)
    if len(M) != K:
        raise ValueError(f"M must have length {K}, got {len(M)}")

    Phi = np.zeros((N, int(np.sum(M))), dtype=complex)
    col = 0
    for p_idx in range(K):
        basis = x * (np.abs(x) ** (2 * p_idx))
        for m in range(M[p_idx]):
            if m == 0:
                Phi[:, col] = basis
            else:
                Phi[m:, col] = basis[:-m]
                Phi[:m, col] = 0.0
            col += 1
    return Phi


def apply_mp(x: np.ndarray, coeff: np.ndarray, P: int, M: np.ndarray) -> np.ndarray:
    Phi = mp_regressor(x, P, M)
    return Phi @ np.asarray(coeff).reshape(-1)


def fit_mp(input_signal: np.ndarray, target_signal: np.ndarray, P: int, M: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """(Ridge) least-squares fit for MP coefficients."""
    x = np.asarray(input_signal).reshape(-1)
    y = np.asarray(target_signal).reshape(-1)
    Phi = mp_regressor(x, P, M)
    A = Phi.conj().T @ Phi
    b = Phi.conj().T @ y
    if ridge > 0:
        A = A + ridge * np.eye(A.shape[0], dtype=A.dtype)
    return np.linalg.solve(A, b)


@dataclass
class MPParam:
    Fs: float
    P: int
    M: np.ndarray
    coeff: np.ndarray
    Gain: float = 1.0


class MemoryPolynomialPA:
    """Memory Polynomial PA.

    - backoff_db: 입력 평균전력을 기준으로 backoff를 적용하기 위해 내부에서 스케일링.
    - Gain: branch gain(예: P_TX_Case1.mat 기반) 적용.

    forward(x): 복소 baseband 입력 -> 복소 baseband 출력
    """

    def __init__(self, param: MPParam, backoff_db: float = 0.0):
        self.param = param
        self.backoff_db = float(backoff_db)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).reshape(-1)
        Pin = np.mean(np.abs(x) ** 2)
        scale_in = 1.0
        if Pin > 0:
            scale_in = 1.0 / np.sqrt((10.0 ** (self.backoff_db / 10.0)) * Pin)
        x_scaled = scale_in * x

        y = apply_mp(x_scaled, self.param.coeff, self.param.P, self.param.M)
        y = self.param.Gain * y

        # scale back so output power tracks original scale
        y = (1.0 / scale_in) * y
        return y


def load_lund_mp_dataset(pa_memory_models_mat: str, p_tx_mat: Optional[str] = None, gains: str = "measured") -> Tuple[List[MPParam], float]:
    """Load the Lund testbed MP models (100 PAs) from MATLAB .mat files."""
    from scipy import io

    data = io.loadmat(pa_memory_models_mat)
    params_arr = data["parameters"]
    Fs = float(params_arr[0, 0]["Fs"][0, 0].squeeze())

    if gains == "equal":
        gain_vec = np.ones(100, dtype=float)
    else:
        if p_tx_mat is None:
            raise ValueError("p_tx_mat is required when gains='measured'")
        ptx = io.loadmat(p_tx_mat)["P_tx"].reshape(-1)
        gain_vec = np.sqrt(10.0 ** (ptx / 10.0))
        gain_vec = gain_vec / np.mean(gain_vec)

    models: List[MPParam] = []
    for i in range(100):
        p = params_arr[i, 0]
        pa = p["pa"][0, 0]
        P = int(pa["P"][0, 0].squeeze())
        M = np.array(pa["Lpa"][0, 0]).squeeze().astype(int)
        coeff = np.array(pa["coeff"][0, 0]).squeeze()
        models.append(MPParam(Fs=Fs, P=P, M=M, coeff=coeff, Gain=float(gain_vec[i])))
    return models, Fs


# =========================
# 2) Generalized Memory Polynomial (GMP)
# =========================

def gmp_regressor(
    x: np.ndarray,
    P: int,
    M: int,
    L_lag: int = 0,
    L_lead: int = 0,
) -> np.ndarray:
    """GMP regressor (one common parameterization).

    Base MP term:
      x[n-m] |x[n-m]|^{2p}

    Lagging cross-term:
      x[n-m] |x[n-m-l]|^{2p},  l=1..L_lag

    Leading cross-term:
      x[n-m] |x[n-m+l]|^{2p},  l=1..L_lead

    Parameters
    ----------
    P : max odd order (1,3,...,P)
    M : memory length for the main MP term (0..M-1)
    L_lag / L_lead : cross-term lags/leads

    Returns
    -------
    Phi : (N, n_features)
    """
    x = np.asarray(x).reshape(-1)
    N = len(x)
    if P % 2 == 0:
        raise ValueError("P must be odd")
    K = (P + 1) // 2

    feats: List[np.ndarray] = []

    # helper to time-shift a basis vector
    def shift(v: np.ndarray, m: int) -> np.ndarray:
        out = np.zeros_like(v)
        if m == 0:
            out[:] = v
        else:
            out[m:] = v[:-m]
        return out

    # main MP
    for p_idx in range(K):
        amp = (np.abs(x) ** (2 * p_idx))
        base = x * amp
        for m in range(M):
            feats.append(shift(base, m))

    # lagging envelope cross-terms
    for p_idx in range(K):
        for l in range(1, L_lag + 1):
            amp_l = (np.abs(np.roll(x, l)) ** (2 * p_idx))  # |x[n+l]| but roll shifts index; we'll handle edges later by masking
            # For causal lagging term |x[n-m-l]| we want envelope delayed more.
            # A safer approach is to explicitly build per-sample with shifts; we keep it simple here.
            # We'll zero out invalid edges by additional shifts.
            env = np.abs(x) ** (2 * p_idx)
            env_del = shift(env, l)  # env[n-l]
            base = x * env_del
            for m in range(M):
                feats.append(shift(base, m))

    # leading envelope cross-terms
    for p_idx in range(K):
        for l in range(1, L_lead + 1):
            env = np.abs(x) ** (2 * p_idx)
            # env[n+l] -> advance: env_adv[n] = env[n+l], last l samples invalid
            env_adv = np.zeros_like(env)
            env_adv[:-l] = env[l:]
            base = x * env_adv
            for m in range(M):
                feats.append(shift(base, m))

    Phi = np.stack(feats, axis=1).astype(complex)  # (N, F)
    return Phi


def apply_gmp(x: np.ndarray, coeff: np.ndarray, P: int, M: int, L_lag: int = 0, L_lead: int = 0) -> np.ndarray:
    Phi = gmp_regressor(x, P=P, M=M, L_lag=L_lag, L_lead=L_lead)
    return Phi @ np.asarray(coeff).reshape(-1)


def fit_gmp(x: np.ndarray, y: np.ndarray, P: int, M: int, L_lag: int = 0, L_lead: int = 0, ridge: float = 0.0) -> np.ndarray:
    Phi = gmp_regressor(x, P=P, M=M, L_lag=L_lag, L_lead=L_lead)
    A = Phi.conj().T @ Phi
    b = Phi.conj().T @ np.asarray(y).reshape(-1)
    if ridge > 0:
        A = A + ridge * np.eye(A.shape[0], dtype=A.dtype)
    return np.linalg.solve(A, b)


@dataclass
class GMPParam:
    Fs: float
    P: int
    M: int
    L_lag: int
    L_lead: int
    coeff: np.ndarray
    Gain: float = 1.0


class GMPPA:
    def __init__(self, param: GMPParam, backoff_db: float = 0.0):
        self.param = param
        self.backoff_db = float(backoff_db)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).reshape(-1)
        Pin = np.mean(np.abs(x) ** 2)
        scale_in = 1.0
        if Pin > 0:
            scale_in = 1.0 / np.sqrt((10.0 ** (self.backoff_db / 10.0)) * Pin)
        xs = scale_in * x

        y = apply_gmp(xs, self.param.coeff, self.param.P, self.param.M, self.param.L_lag, self.param.L_lead)
        y = self.param.Gain * y
        y = (1.0 / scale_in) * y
        return y


# =========================
# 3) Wiener model (LTI + memoryless nonlinearity)
# =========================

@dataclass
class WienerParam:
    Fs: float
    h: np.ndarray              # FIR taps (complex allowed)
    poly_a: np.ndarray         # memoryless odd-order polynomial coeffs: a0 (order1), a1(order3), ...
    Gain: float = 1.0


class WienerPA:
    """Simple complex-baseband Wiener PA:

    v[n] = sum_{m} h[m] x[n-m]
    y[n] = sum_{p_idx} a[p_idx] v[n] |v[n]|^{2 p_idx}

    (즉, LTI 메모리 + memoryless polynomial nonlinearity)
    """

    def __init__(self, param: WienerParam, backoff_db: float = 0.0):
        self.param = param
        self.backoff_db = float(backoff_db)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).reshape(-1)
        Pin = np.mean(np.abs(x) ** 2)
        scale_in = 1.0
        if Pin > 0:
            scale_in = 1.0 / np.sqrt((10.0 ** (self.backoff_db / 10.0)) * Pin)
        xs = scale_in * x

        v = np.convolve(self.param.h, xs)[: len(xs)]

        # memoryless odd-order poly
        y = np.zeros_like(v)
        for p_idx, a in enumerate(np.asarray(self.param.poly_a).reshape(-1)):
            y = y + a * v * (np.abs(v) ** (2 * p_idx))

        y = self.param.Gain * y
        y = (1.0 / scale_in) * y
        return y


# =========================
# 4) Neural Net PA (PyTorch, 선택)
# =========================

class NeuralNetPA:
    """Neural network PA (PyTorch).

    구현 방향(권장):
      - 입력: 최근 M개 샘플의 I/Q를 펼친 벡터 (shape: [B, 2*M])
      - 출력: 현재 y[n]의 I/Q (shape: [B, 2])

    아래 구현은 '모델/학습 루프'를 최소 뼈대만 제공합니다.
    실제 실험에서는 train/test split, early stopping, normalization 등을 추가하세요.

    Requirements:
      pip install torch
    """

    def __init__(self, M: int = 5, hidden: int = 128, depth: int = 3, backoff_db: float = 0.0, device: str = "cpu"):
        try:
            import torch
            import torch.nn as nn
        except Exception as e:
            raise ImportError("NeuralNetPA requires PyTorch (torch).") from e

        self.torch = torch
        self.nn = nn
        self.M = int(M)
        self.backoff_db = float(backoff_db)
        self.device = torch.device(device)

        layers: List[nn.Module] = []
        in_dim = 2 * self.M
        for i in range(depth - 1):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden if depth > 1 else in_dim, 2))
        self.net = nn.Sequential(*layers).to(self.device)

    def _make_dataset(self, x: np.ndarray, y: np.ndarray) -> Tuple["torch.Tensor", "torch.Tensor"]:
        torch = self.torch
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)
        N = len(x)

        # build tapped input: [x[n], x[n-1], ..., x[n-M+1]]
        X = np.zeros((N, self.M), dtype=complex)
        for m in range(self.M):
            if m == 0:
                X[:, m] = x
            else:
                X[m:, m] = x[:-m]
                X[:m, m] = 0.0

        Xin = np.concatenate([X.real, X.imag], axis=1)  # (N, 2M)
        Yout = np.stack([y.real, y.imag], axis=1)       # (N, 2)

        Xin_t = torch.from_numpy(Xin).float().to(self.device)
        Yout_t = torch.from_numpy(Yout).float().to(self.device)
        return Xin_t, Yout_t

    def fit(self, x: np.ndarray, y: np.ndarray, lr: float = 1e-3, epochs: int = 5, batch: int = 4096) -> None:
        torch = self.torch
        Xin, Yout = self._make_dataset(x, y)

        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        N = Xin.shape[0]
        for ep in range(epochs):
            perm = torch.randperm(N, device=self.device)
            for i in range(0, N, batch):
                idx = perm[i : i + batch]
                pred = self.net(Xin[idx])
                loss = loss_fn(pred, Yout[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

    def forward(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = np.asarray(x).reshape(-1)

        # backoff scaling
        Pin = np.mean(np.abs(x) ** 2)
        scale_in = 1.0
        if Pin > 0:
            scale_in = 1.0 / np.sqrt((10.0 ** (self.backoff_db / 10.0)) * Pin)
        xs = scale_in * x

        # build tapped input and run net
        N = len(xs)
        X = np.zeros((N, self.M), dtype=complex)
        for m in range(self.M):
            if m == 0:
                X[:, m] = xs
            else:
                X[m:, m] = xs[:-m]
                X[:m, m] = 0.0

        Xin = np.concatenate([X.real, X.imag], axis=1).astype(np.float32)
        Xin_t = torch.from_numpy(Xin).to(self.device)
        with torch.no_grad():
            y2 = self.net(Xin_t).cpu().numpy()
        y = y2[:, 0] + 1j * y2[:, 1]

        # scale back
        y = (1.0 / scale_in) * y
        return y
