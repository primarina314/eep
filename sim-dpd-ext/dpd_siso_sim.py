#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dpd_siso_sim.py

SISO PA + DPD 시뮬레이터 (QAM + RRC, baseband behavioral PA)

이번 버전에서 추가된 기능
------------------------
1) PA 모델 교체: --pa_type mp|gmp|wiener|nn
2) DPD 모델 교체: --dpd_type none|mp|gmp|nn

기본 동작
---------
- 입력: RRC-shaped square QAM (single-carrier), oversampling
- PA: 선택된 behavioral model
- DPD: ILA(Indirect Learning Architecture)
    postdistorter g(y) ≈ x 를 학습한 뒤, predistorter로 u=g(x_desired) 적용
- 성능: EVM, PSD, ACLR

Measured MP 모델 사용
---------------------
업로드해준 Lund testbed 계수를 그대로 쓰려면 아래 파일이 필요합니다.
- PA_memory_models.mat
- P_TX_Case1.mat (measured gain 사용 시)

파일이 없으면 자동으로 synthetic PA로 fallback 합니다.

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
from numpy.random import default_rng
from scipy import signal
import matplotlib.pyplot as plt

from pa_models import (
    MPParam,
    GMPParam,
    WienerParam,
    MemoryPolynomialPA,
    GMPPA,
    WienerPA,
    NeuralNetPA,
    load_lund_mp_dataset,
)

from dpd_models import (
    MPDPD,
    GMPDPD,
    NeuralNetDPD,
    mp_memory_vector,
)


# =========================
# Waveform: RRC shaping
# =========================


def rrc_taps(alpha: float, span: int, sps: int) -> np.ndarray:
    """Root Raised Cosine (RRC) FIR taps (unit energy)."""
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0,1].")
    if span <= 0 or sps <= 0:
        raise ValueError("span and sps must be positive")

    N = span * sps
    t = np.arange(-N / 2, N / 2 + 1) / sps
    h = np.zeros_like(t, dtype=float)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1.0 + alpha * (4.0 / np.pi - 1.0)
        elif abs(abs(4.0 * alpha * ti) - 1.0) < 1e-8:
            h[i] = (alpha / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
            )
        else:
            num = np.sin(np.pi * ti * (1.0 - alpha)) + 4.0 * alpha * ti * np.cos(np.pi * ti * (1.0 + alpha))
            den = np.pi * ti * (1.0 - (4.0 * alpha * ti) ** 2)
            h[i] = num / den
    h = h / np.sqrt(np.sum(h**2))
    return h


def pulse_shape(sym: np.ndarray, over: int, alpha: float, D: int) -> Tuple[np.ndarray, int, np.ndarray]:
    """RRC pulse shaping (MATLAB pulse_shape.m equivalent)."""
    sym = np.asarray(sym).reshape(-1)
    up = np.zeros(len(sym) * over, dtype=complex)
    up[::over] = sym
    rrc = rrc_taps(alpha=alpha, span=2 * D, sps=over)
    d = int(np.argmax(rrc))
    s_full = np.convolve(rrc, up)
    s = s_full[d : d + len(sym) * over]
    return s, d, rrc


def rx_match_filter_and_sample(wave: np.ndarray, rrc: np.ndarray, over: int, d: int, Nsym: int) -> np.ndarray:
    wave = np.asarray(wave).reshape(-1)
    mf_full = np.convolve(rrc, wave)
    mf = mf_full[d : d + Nsym * over]
    return mf[::over]


# =========================
# QAM
# =========================


def square_qam_alphabet(M: int, normalize_avg_power: bool = True) -> np.ndarray:
    m = int(np.sqrt(M))
    if m * m != M:
        raise ValueError("QAM order must be a perfect square (16/64/256/...) ")
    tmp = np.arange(-(m - 1), m, 2)
    alphabet = np.kron(tmp, np.ones(m)) + 1j * np.kron(np.ones(m), tmp)
    alphabet = alphabet.astype(complex)
    if normalize_avg_power:
        alphabet = alphabet / np.sqrt(np.mean(np.abs(alphabet) ** 2))
    return alphabet


def generate_qam_symbols(Nsym: int, M: int, seed: int = 1) -> np.ndarray:
    rng = default_rng(seed)
    a = square_qam_alphabet(M, normalize_avg_power=True)
    idx = rng.integers(0, len(a), size=Nsym)
    return a[idx]


# =========================
# Metrics
# =========================


def evm_rms(s_ref: np.ndarray, s_hat: np.ndarray, discard: int = 0) -> Tuple[float, complex]:
    """RMS EVM after best complex scalar equalization."""
    s_ref = np.asarray(s_ref).reshape(-1)
    s_hat = np.asarray(s_hat).reshape(-1)
    if len(s_ref) != len(s_hat):
        raise ValueError("length mismatch")
    if discard > 0:
        s_ref = s_ref[discard:-discard]
        s_hat = s_hat[discard:-discard]
    # minimize ||s_ref - a*s_hat|| -> a = (s_hat^H s_ref)/(s_hat^H s_hat)
    denom = np.vdot(s_hat, s_hat)
    a = np.vdot(s_hat, s_ref) / denom if abs(denom) > 0 else 1.0 + 0j
    e = s_ref - a * s_hat
    evm = np.sqrt(np.mean(np.abs(e) ** 2) / np.mean(np.abs(s_ref) ** 2))
    return float(evm), a


def welch_psd(x: np.ndarray, fs: float, nperseg: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x).reshape(-1)
    nper = int(min(int(nperseg), len(x)))
    nper = max(8, nper)
    nov = int(nper // 2)
    f, Pxx = signal.welch(
        x,
        fs=fs,
        window=("kaiser", 7.0),
        nperseg=nper,
        noverlap=nov,
        return_onesided=False,
        scaling="density",
    )
    # shift to [-fs/2, fs/2)
    f = np.fft.fftshift(f)
    Pxx = np.fft.fftshift(Pxx)
    return f, Pxx


def band_power_from_psd(f: np.ndarray, Pxx: np.ndarray, f0: float, bw: float) -> float:
    """Integrate PSD over [f0-bw/2, f0+bw/2]."""
    f = np.asarray(f)
    Pxx = np.asarray(Pxx)
    mask = (f >= (f0 - bw / 2)) & (f < (f0 + bw / 2))
    if not np.any(mask):
        return 0.0
    return float(np.trapz(Pxx[mask], f[mask]))


def aclr_from_psd(f: np.ndarray, Pxx: np.ndarray, ch_bw: float, adj_offset: float) -> Dict[str, float]:
    main = band_power_from_psd(f, Pxx, 0.0, ch_bw)
    upper = band_power_from_psd(f, Pxx, +adj_offset, ch_bw)
    lower = band_power_from_psd(f, Pxx, -adj_offset, ch_bw)
    aclr_u = 10 * np.log10((main + 1e-30) / (upper + 1e-30))
    aclr_l = 10 * np.log10((main + 1e-30) / (lower + 1e-30))
    return {
        "ACLR_u_dB": float(aclr_u),
        "ACLR_l_dB": float(aclr_l),
        "ACLR_worst_dB": float(min(aclr_u, aclr_l)),
    }


# =========================
# Helpers: build PA / DPD
# =========================


def _resolve_file(path_str: str) -> Optional[str]:
    p = Path(path_str)
    if p.exists():
        return str(p)
    # try relative to script
    try:
        here = Path(__file__).resolve().parent
        p2 = here / path_str
        if p2.exists():
            return str(p2)
    except Exception:
        pass
    return None


def build_pa(
    args: argparse.Namespace,
    x_for_nn_train: np.ndarray,
    seed: int,
):
    """Return (pa_object, Fs)."""
    rng = default_rng(seed)

    # 1) MP from measured dataset
    if args.pa_type == "mp":
        pa_path = _resolve_file(args.pa_memory_models_mat)
        ptx_path = _resolve_file(args.p_tx_mat) if args.gains == "measured" else None
        if pa_path is not None:
            if args.gains == "measured" and ptx_path is None:
                raise FileNotFoundError(f"'{args.p_tx_mat}' not found (needed for measured gains).")
            mp_models, Fs = load_lund_mp_dataset(pa_path, ptx_path, gains=args.gains)

            indTX = 2 * args.pa_index - 1 + args.rf
            indTX = int(np.clip(indTX, 1, 100))
            param = mp_models[indTX - 1]
            pa = MemoryPolynomialPA(param, backoff_db=args.backoff_db)
            return pa, Fs

        # fallback synthetic MP
        Fs = 120e6
        P = int(args.mp_P)
        M = np.array([max(1, int(args.mp_mem))] * ((P + 1) // 2), dtype=int)
        coeff = np.zeros(int(np.sum(M)), dtype=complex)
        coeff[0] = 1.0 + 0j
        if len(coeff) > 1:
            coeff[1] = (0.10 - 0.03j)  # weak memory on linear term
        if len(coeff) > M[0]:
            coeff[M[0] + 0] = (0.08 - 0.02j)  # 3rd-order
        param = MPParam(Fs=Fs, P=P, M=M, coeff=coeff, Gain=1.0)
        return MemoryPolynomialPA(param, backoff_db=args.backoff_db), Fs

    # 2) GMP synthetic
    if args.pa_type == "gmp":
        Fs = 120e6
        P = int(args.gmp_P)
        M = int(args.gmp_M)
        L_lag = int(args.gmp_L_lag)
        L_lead = int(args.gmp_L_lead)
        K = (P + 1) // 2
        n_feat = K * M * (1 + L_lag + L_lead)
        coeff = np.zeros(n_feat, dtype=complex)
        coeff[0] = 1.0 + 0j
        # small random nonlinear terms
        coeff[1:] = (0.03 * (rng.standard_normal(n_feat - 1) + 1j * rng.standard_normal(n_feat - 1)) / np.sqrt(2.0))
        param = GMPParam(Fs=Fs, P=P, M=M, L_lag=L_lag, L_lead=L_lead, coeff=coeff, Gain=1.0)
        return GMPPA(param, backoff_db=args.backoff_db), Fs

    # 3) Wiener synthetic
    if args.pa_type == "wiener":
        Fs = 120e6
        P = int(args.wiener_P)
        K = (P + 1) // 2
        L = int(args.wiener_fir_len)
        h = np.zeros(L, dtype=complex)
        h[0] = 1.0 + 0j
        if L > 1:
            decay = np.exp(-np.arange(1, L) / max(1.0, (L - 1) / 2.0))
            h[1:] = 0.08 * decay * (rng.standard_normal(L - 1) + 1j * rng.standard_normal(L - 1)) / np.sqrt(2.0)
        poly_a = np.zeros(K, dtype=complex)
        poly_a[0] = 1.0 + 0j
        if K > 1:
            poly_a[1] = 0.10 - 0.03j
        if K > 2:
            poly_a[2] = 0.03 + 0.01j
        param = WienerParam(Fs=Fs, h=h, poly_a=poly_a, Gain=1.0)
        return WienerPA(param, backoff_db=args.backoff_db), Fs

    # 4) Neural Net PA: train NN to mimic a teacher PA
    if args.pa_type == "nn":
        teacher = args.nn_teacher
        if teacher == "auto":
            # prefer measured MP if available
            pa_path = _resolve_file(args.pa_memory_models_mat)
            teacher = "mp" if pa_path is not None else "wiener"

        # build teacher PA recursively
        class _Tmp:
            pass

        tmp = _Tmp()
        for k, v in vars(args).items():
            setattr(tmp, k, v)
        tmp.pa_type = teacher
        pa_teacher, Fs = build_pa(tmp, x_for_nn_train=x_for_nn_train, seed=seed + 123)

        y_teacher = pa_teacher.forward(x_for_nn_train)

        pa_nn = NeuralNetPA(M=int(args.nn_M), hidden=int(args.nn_hidden), depth=int(args.nn_depth), backoff_db=args.backoff_db, device=args.nn_device)
        pa_nn.fit(x_for_nn_train, y_teacher, lr=args.nn_lr, epochs=int(args.nn_epochs), batch=int(args.nn_batch))
        return pa_nn, Fs

    raise ValueError("Unsupported --pa_type")


def build_dpd(args: argparse.Namespace, pa_obj, x: np.ndarray, y: np.ndarray, seed: int):
    """Return dpd object (already fitted) or None."""
    if args.dpd_type == "none":
        return None

    # training subset
    Ntrain = int(min(len(x), args.dpd_train_syms * args.osf)) if args.dpd_train_syms > 0 else len(x)
    x_tr = x[:Ntrain]
    y_tr = y[:Ntrain]

    if args.dpd_type == "mp":
        # choose order
        if args.dpd_order is not None:
            P = int(args.dpd_order)
        else:
            P = int(getattr(getattr(pa_obj, "param", None), "P", 9))

        # choose memory
        if args.dpd_mem > 0:
            M = mp_memory_vector(P, args.dpd_mem)
        else:
            # inherit PA memory if available
            paM = getattr(getattr(pa_obj, "param", None), "M", None)
            if paM is None:
                M = mp_memory_vector(P, 2)
            else:
                paM = np.asarray(paM).astype(int).reshape(-1)
                K = (P + 1) // 2
                if len(paM) >= K:
                    M = paM[:K]
                else:
                    M = np.concatenate([paM, np.full(K - len(paM), paM[-1], dtype=int)])

        dpd = MPDPD(P=P, M=M, ridge=float(args.ridge))
        dpd.fit_ila(x_tr, y_tr)
        return dpd

    if args.dpd_type == "gmp":
        P = int(args.dpd_gmp_P)
        M = int(args.dpd_gmp_M)
        dpd = GMPDPD(P=P, M=M, L_lag=int(args.dpd_gmp_L_lag), L_lead=int(args.dpd_gmp_L_lead), ridge=float(args.ridge))
        dpd.fit_ila(x_tr, y_tr)
        return dpd

    if args.dpd_type == "nn":
        dpd = NeuralNetDPD(M=int(args.dpd_nn_M), hidden=int(args.dpd_nn_hidden), depth=int(args.dpd_nn_depth), device=args.dpd_nn_device)
        dpd.fit_ila(x_tr, y_tr, lr=float(args.dpd_nn_lr), epochs=int(args.dpd_nn_epochs), batch=int(args.dpd_nn_batch), seed=int(seed + 999))
        return dpd

    raise ValueError("Unsupported --dpd_type")


def clip_to_rms(sig: np.ndarray, clip_rms: float) -> np.ndarray:
    sig = np.asarray(sig).reshape(-1)
    if clip_rms <= 0:
        return sig
    rms = float(np.sqrt(np.mean(np.abs(sig) ** 2)))
    peak = clip_rms * rms
    mag = np.abs(sig)
    scale = np.minimum(1.0, peak / (mag + 1e-12))
    return sig * scale


# =========================
# Main
# =========================


def main():
    ap = argparse.ArgumentParser()
    # waveform
    ap.add_argument("--qam_order", type=int, default=256)
    ap.add_argument("--Nsym", type=int, default=30000)
    ap.add_argument("--osf", type=int, default=8)
    ap.add_argument("--rolloff", type=float, default=0.22)
    ap.add_argument("--rrc_delay", type=int, default=12)
    ap.add_argument("--discard_syms", type=int, default=12)

    # PA selection
    ap.add_argument("--pa_type", type=str, default="mp", choices=["mp", "gmp", "wiener", "nn"])
    ap.add_argument("--backoff_db", type=float, default=0.0)

    # measured MP dataset options
    ap.add_argument("--pa_memory_models_mat", type=str, default="PA_memory_models.mat")
    ap.add_argument("--p_tx_mat", type=str, default="P_TX_Case1.mat")
    ap.add_argument("--gains", type=str, default="measured", choices=["equal", "measured"])
    ap.add_argument("--pa_index", type=int, default=1, help="USRP index Ind (1..50)")
    ap.add_argument("--rf", type=int, default=0, choices=[0, 1], help="0=TX0(odd), 1=TX1(even)")

    # synthetic PA params
    ap.add_argument("--mp_P", type=int, default=9)
    ap.add_argument("--mp_mem", type=int, default=2)
    ap.add_argument("--gmp_P", type=int, default=9)
    ap.add_argument("--gmp_M", type=int, default=3)
    ap.add_argument("--gmp_L_lag", type=int, default=1)
    ap.add_argument("--gmp_L_lead", type=int, default=1)
    ap.add_argument("--wiener_P", type=int, default=9)
    ap.add_argument("--wiener_fir_len", type=int, default=5)

    # NN PA params
    ap.add_argument("--nn_teacher", type=str, default="auto", choices=["auto", "mp", "gmp", "wiener"])
    ap.add_argument("--nn_M", type=int, default=7)
    ap.add_argument("--nn_hidden", type=int, default=128)
    ap.add_argument("--nn_depth", type=int, default=3)
    ap.add_argument("--nn_epochs", type=int, default=5)
    ap.add_argument("--nn_lr", type=float, default=1e-3)
    ap.add_argument("--nn_batch", type=int, default=4096)
    ap.add_argument("--nn_device", type=str, default="cpu")

    # DPD selection
    ap.add_argument("--dpd_type", type=str, default="mp", choices=["none", "mp", "gmp", "nn"])
    ap.add_argument("--dpd_train_syms", type=int, default=6000, help="0이면 전체 사용")
    ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument("--dpd_clip_rms", type=float, default=3.0)

    # MP DPD
    ap.add_argument("--dpd_order", type=int, default=None)
    ap.add_argument("--dpd_mem", type=int, default=-1, help=">0: 모든 차수 동일 메모리, -1: 가능하면 PA 메모리 상속")

    # GMP DPD
    ap.add_argument("--dpd_gmp_P", type=int, default=9)
    ap.add_argument("--dpd_gmp_M", type=int, default=3)
    ap.add_argument("--dpd_gmp_L_lag", type=int, default=1)
    ap.add_argument("--dpd_gmp_L_lead", type=int, default=1)

    # NN DPD
    ap.add_argument("--dpd_nn_M", type=int, default=7)
    ap.add_argument("--dpd_nn_hidden", type=int, default=128)
    ap.add_argument("--dpd_nn_depth", type=int, default=3)
    ap.add_argument("--dpd_nn_epochs", type=int, default=5)
    ap.add_argument("--dpd_nn_lr", type=float, default=1e-3)
    ap.add_argument("--dpd_nn_batch", type=int, default=4096)
    ap.add_argument("--dpd_nn_device", type=str, default="cpu")

    ap.add_argument("--plot", action="store_true")

    args = ap.parse_args()

    # -------------------------
    # Waveform
    # -------------------------
    sym = generate_qam_symbols(args.Nsym, args.qam_order, seed=1)
    x, d, rrc = pulse_shape(sym, over=args.osf, alpha=args.rolloff, D=args.rrc_delay)

    # -------------------------
    # PA
    # -------------------------
    pa, Fs = build_pa(args, x_for_nn_train=x, seed=0)

    y = pa.forward(x)

    # -------------------------
    # DPD
    # -------------------------
    dpd = build_dpd(args, pa_obj=pa, x=x, y=y, seed=0)
    if dpd is None:
        u = x
        y_dpd = y
    else:
        u = dpd.forward(x)
        u = clip_to_rms(u, args.dpd_clip_rms)
        y_dpd = pa.forward(u)

    # -------------------------
    # Metrics
    # -------------------------
    sym_hat = rx_match_filter_and_sample(y, rrc, over=args.osf, d=d, Nsym=args.Nsym)
    sym_hat_dpd = rx_match_filter_and_sample(y_dpd, rrc, over=args.osf, d=d, Nsym=args.Nsym)
    evm0, _ = evm_rms(sym, sym_hat, discard=args.discard_syms)
    evm1, _ = evm_rms(sym, sym_hat_dpd, discard=args.discard_syms)

    f0, P0 = welch_psd(y, Fs)
    f1, P1 = welch_psd(y_dpd, Fs)

    Rs = Fs / args.osf
    ch_bw = (1.0 + args.rolloff) * Rs
    adj = ch_bw
    aclr0 = aclr_from_psd(f0, P0, ch_bw=ch_bw, adj_offset=adj)
    aclr1 = aclr_from_psd(f1, P1, ch_bw=ch_bw, adj_offset=adj)

    print("=== SISO PA + DPD Simulation ===")
    print(f"Waveform: {args.qam_order}-QAM, Nsym={args.Nsym}, osf={args.osf}, rolloff={args.rolloff}, Fs={Fs/1e6:.1f} Msps")
    print(f"PA: type={args.pa_type}, backoff={args.backoff_db:.2f} dB")
    if args.pa_type == "mp":
        print(f"  gains={args.gains}, Ind={args.pa_index}, rf={args.rf}")
    print(f"DPD: type={args.dpd_type}, train_syms={args.dpd_train_syms}, ridge={args.ridge:g}, clip={args.dpd_clip_rms}*RMS")
    if args.dpd_type == "mp":
        print(f"  MP-DPD: order={args.dpd_order}, mem={args.dpd_mem}")
    if args.dpd_type == "gmp":
        print(f"  GMP-DPD: P={args.dpd_gmp_P}, M={args.dpd_gmp_M}, L_lag={args.dpd_gmp_L_lag}, L_lead={args.dpd_gmp_L_lead}")
    if args.dpd_type == "nn":
        print(f"  NN-DPD: M={args.dpd_nn_M}, hidden={args.dpd_nn_hidden}, depth={args.dpd_nn_depth}, epochs={args.dpd_nn_epochs}")
    print("")
    print(f"EVM (no DPD):   {100*evm0:.3f} %")
    print(f"EVM (with DPD): {100*evm1:.3f} %")
    print("")
    print(f"ACLR worst (no DPD):   {aclr0['ACLR_worst_dB']:.2f} dB")
    print(f"ACLR worst (with DPD): {aclr1['ACLR_worst_dB']:.2f} dB")

    if args.plot:
        plt.figure()
        plt.plot(f0 / 1e6, 10 * np.log10(np.maximum(P0, 1e-20)), label="PA out")
        plt.plot(f1 / 1e6, 10 * np.log10(np.maximum(P1, 1e-20)), label="DPD+PA out")
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("PSD [dB/Hz]")
        plt.title("PSD (Welch)")
        plt.grid(True)
        plt.legend()

        plt.figure()
        plt.plot(sym_hat.real, sym_hat.imag, ".", alpha=0.25, label="PA")
        plt.plot(sym_hat_dpd.real, sym_hat_dpd.imag, ".", alpha=0.25, label="DPD")
        plt.axis("equal")
        plt.grid(True)
        plt.title("RX Constellation")
        plt.legend()

        plt.show()


if __name__ == "__main__":
    main()
