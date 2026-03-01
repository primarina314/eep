#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hybrid_mimo_dpd_sim.py

Hybrid MU-MIMO Downlink Transmitter Simulation
--------------------------------------------

요청 반영 사항
--------------
1) Hybrid precoding (Digital precoder + Analog phase-shifter beamformer)
2) Multi-user (K users, 1 stream/user), QAM single-carrier + RRC
3) Rayleigh flat fading channel
4) PA 모델 스왑: --pa_type mp|gmp|wiener|nn
5) DPD 모델 스왑: --dpd_type mp|gmp|nn (mode는 --dpd)
6) MIMO에서 PA별 편차 주입 (AM/AM, AM/PM, memory diversity)
   - drive_db / gain_db / phase_deg / AMPM (amp-dependent phase)
   - (optional) output-side mild FIR (linear memory) per branch

Measured MP PA 모델
-------------------
업로드해준 Lund testbed MP 계수를 사용하려면:
  - PA_memory_models.mat
  - P_TX_Case1.mat (measured gain 사용 시)
가 필요합니다. 없으면 synthetic PA로 자동 fallback.

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List

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
    PAMismatchWrapper,
    random_mild_fir,
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
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0,1].")
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
        raise ValueError("QAM order must be a perfect square")
    tmp = np.arange(-(m - 1), m, 2)
    alphabet = np.kron(tmp, np.ones(m)) + 1j * np.kron(np.ones(m), tmp)
    alphabet = alphabet.astype(complex)
    if normalize_avg_power:
        alphabet = alphabet / np.sqrt(np.mean(np.abs(alphabet) ** 2))
    return alphabet


def generate_qam_symbols(Nsym: int, M: int, seed: int) -> np.ndarray:
    rng = default_rng(seed)
    a = square_qam_alphabet(M, normalize_avg_power=True)
    idx = rng.integers(0, len(a), size=Nsym)
    return a[idx]


# =========================
# Channel
# =========================


def rayleigh_channel(K: int, Nt: int, seed: int) -> np.ndarray:
    rng = default_rng(seed)
    H = (rng.standard_normal((K, Nt)) + 1j * rng.standard_normal((K, Nt))) / np.sqrt(2.0)
    return H


# =========================
# Hybrid precoding
# =========================


def quantize_phase(x: np.ndarray, bits: int, magnitude: Optional[float] = None) -> np.ndarray:
    if bits <= 0:
        return x
    levels = 2**bits
    step = 2 * np.pi / levels
    phi = step * np.round(np.angle(x) / step)
    mag = np.abs(x) if magnitude is None else float(magnitude)
    return mag * np.exp(1j * phi)


def zf_precoder(H: np.ndarray, reg: float = 0.0) -> np.ndarray:
    K = H.shape[0]
    HH = H @ H.conj().T
    if reg > 0:
        HH = HH + reg * np.eye(K, dtype=HH.dtype)
    return H.conj().T @ np.linalg.inv(HH)


def design_hybrid_precoder(
    H: np.ndarray,
    n_rf: int,
    method: str,
    phase_bits: int,
    rzf_reg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    K, Nt = H.shape
    if n_rf < K:
        raise ValueError("Need n_rf >= K")

    if method == "random_zf":
        rng = default_rng(0)
        FRF = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(Nt, n_rf))) / np.sqrt(Nt)
        FRF = quantize_phase(FRF, phase_bits, magnitude=1 / np.sqrt(Nt))
    else:
        # phase-only from right singular vectors
        _, _, Vh = np.linalg.svd(H, full_matrices=True)
        V = Vh.conj().T
        FRF = np.exp(1j * np.angle(V[:, :n_rf])) / np.sqrt(Nt)
        FRF = quantize_phase(FRF, phase_bits, magnitude=1 / np.sqrt(Nt))

    if method in ("svd_phase_zf", "random_zf"):
        Heff = H @ FRF  # K x n_rf
        HH = Heff @ Heff.conj().T
        if rzf_reg > 0:
            HH = HH + rzf_reg * np.eye(K, dtype=HH.dtype)
        FBB = Heff.conj().T @ np.linalg.inv(HH)  # n_rf x K
        return FRF, FBB

    if method == "fopt_altmin":
        # simple AltMin to approximate full-digital ZF
        Fopt = zf_precoder(H, reg=rzf_reg)  # Nt x K
        _, _, Vh = np.linalg.svd(H, full_matrices=True)
        V = Vh.conj().T
        FRF = np.exp(1j * np.angle(V[:, :n_rf])) / np.sqrt(Nt)
        FRF = quantize_phase(FRF, phase_bits, magnitude=1 / np.sqrt(Nt))
        for _ in range(10):
            # LS for FBB
            FBB = np.linalg.lstsq(FRF, Fopt, rcond=None)[0]
            # update FRF by phase projection
            FRF = np.exp(1j * np.angle(Fopt @ np.linalg.pinv(FBB))) / np.sqrt(Nt)
            FRF = quantize_phase(FRF, phase_bits, magnitude=1 / np.sqrt(Nt))
        # final baseband precoder on effective channel
        Heff = H @ FRF
        HH = Heff @ Heff.conj().T
        if rzf_reg > 0:
            HH = HH + rzf_reg * np.eye(K, dtype=HH.dtype)
        FBB = Heff.conj().T @ np.linalg.inv(HH)
        return FRF, FBB

    raise ValueError("Unknown hybrid precoder method")


def normalize_precoder(FRF: np.ndarray, FBB: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize so that ||FRF FBB||_F^2 = K."""
    F = FRF @ FBB
    p = np.sum(np.abs(F) ** 2)
    if p <= 0:
        return FRF, FBB
    return FRF, FBB * np.sqrt(K / p)


# =========================
# Metrics
# =========================


def evm_rms(s_ref: np.ndarray, s_hat: np.ndarray, discard: int = 0) -> Tuple[float, complex]:
    s_ref = np.asarray(s_ref).reshape(-1)
    s_hat = np.asarray(s_hat).reshape(-1)
    if discard > 0:
        s_ref = s_ref[discard:-discard]
        s_hat = s_hat[discard:-discard]
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
    return np.fft.fftshift(f), np.fft.fftshift(Pxx)


def average_psd_over_antennas(Y: np.ndarray, fs: float, nperseg: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """Average PSD over antennas (average of Welch periodograms)."""
    Nt = Y.shape[1]
    f0 = None
    Pacc = None
    for i in range(Nt):
        f, P = welch_psd(Y[:, i], fs=fs, nperseg=nperseg)
        if f0 is None:
            f0 = f
            Pacc = np.zeros_like(P)
        Pacc += P
    return f0, Pacc / Nt


def band_power_from_psd(f: np.ndarray, Pxx: np.ndarray, f0: float, bw: float) -> float:
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
# PA building / mismatch injection
# =========================


def _resolve_file(path_str: str) -> Optional[str]:
    p = Path(path_str)
    if p.exists():
        return str(p)
    try:
        here = Path(__file__).resolve().parent
        p2 = here / path_str
        if p2.exists():
            return str(p2)
    except Exception:
        pass
    return None


def _resize_mp_coeff(coeff: np.ndarray, M_old: np.ndarray, M_new: np.ndarray) -> np.ndarray:
    """Resize flattened MP coeff vector when memory vector changes."""
    coeff = np.asarray(coeff).reshape(-1)
    M_old = np.asarray(M_old).astype(int).reshape(-1)
    M_new = np.asarray(M_new).astype(int).reshape(-1)
    if len(M_old) != len(M_new):
        raise ValueError("M_old and M_new must have same length")
    out_parts: List[np.ndarray] = []
    idx = 0
    for mo, mn in zip(M_old, M_new):
        part = coeff[idx : idx + mo]
        idx += mo
        if mn <= mo:
            out_parts.append(part[:mn])
        else:
            pad = np.zeros(mn - mo, dtype=complex)
            out_parts.append(np.concatenate([part, pad]))
    return np.concatenate(out_parts)


def build_pa_array(
    args: argparse.Namespace,
    Nt: int,
    rng: np.random.Generator,
    x_for_nn_train: Optional[np.ndarray] = None,
) -> Tuple[List[object], float, Dict[str, np.ndarray]]:
    """Return (pa_list, Fs, mismatch_report).

    mismatch_report: per-antenna random draws for reproducibility/debug.
    """

    mismatch_report: Dict[str, np.ndarray] = {}

    # -------------------------
    # 1) Base PA objects (before mismatch wrapper)
    # -------------------------
    pa_base_list: List[object] = []
    Fs = 120e6

    if args.pa_type == "mp":
        pa_path = _resolve_file(args.pa_memory_models_mat)
        ptx_path = _resolve_file(args.p_tx_mat) if args.gains == "measured" else None
        measured_models = None
        if pa_path is not None:
            if args.gains == "measured" and ptx_path is None:
                raise FileNotFoundError(f"'{args.p_tx_mat}' not found (needed for measured gains).")
            measured_models, Fs = load_lund_mp_dataset(pa_path, ptx_path, gains=args.gains)

        if measured_models is None:
            # synthetic MP fallback (same for all antennas, then mismatches add diversity)
            P = int(args.mp_P)
            M = mp_memory_vector(P, int(args.mp_mem))
            coeff = np.zeros(int(np.sum(M)), dtype=complex)
            coeff[0] = 1.0 + 0j
            if len(coeff) > 1:
                coeff[1] = 0.08 - 0.02j
            if len(coeff) > int(M[0]):
                coeff[int(M[0]) + 0] = 0.08 - 0.02j
            base = MPParam(Fs=Fs, P=P, M=M, coeff=coeff, Gain=1.0)
            base_params = [base for _ in range(Nt)]
        else:
            # assign measured models to antennas
            if args.pa_assignment == "same":
                indTX = 2 * args.pa_index - 1 + args.rf
                indTX = int(np.clip(indTX, 1, 100))
                chosen = measured_models[indTX - 1]
                base_params = [chosen for _ in range(Nt)]
            elif args.pa_assignment == "sequential":
                base_params = [measured_models[i % len(measured_models)] for i in range(Nt)]
            elif args.pa_assignment == "random":
                idxs = rng.integers(0, len(measured_models), size=Nt)
                base_params = [measured_models[int(i)] for i in idxs]
            else:
                raise ValueError("pa_assignment must be same|sequential|random")

        # optional MP memory jitter / coeff perturb
        mem_jit = int(args.pa_mp_mem_jitter)
        coeff_sigma = float(args.pa_mp_coeff_sigma)
        for i in range(Nt):
            p = base_params[i]
            # copy
            M_old = np.asarray(p.M).astype(int)
            coeff = np.asarray(p.coeff).astype(complex)
            M_new = M_old.copy()
            if mem_jit > 0:
                jit = rng.integers(-mem_jit, mem_jit + 1, size=len(M_new))
                M_new = np.maximum(1, M_new + jit)
                if np.any(M_new != M_old):
                    coeff = _resize_mp_coeff(coeff, M_old, M_new)
            if coeff_sigma > 0:
                noise = (rng.standard_normal(len(coeff)) + 1j * rng.standard_normal(len(coeff))) / np.sqrt(2.0)
                # do not perturb the very first (main linear) coefficient too aggressively
                scale = np.ones(len(coeff), dtype=float)
                scale[0] = 0.0
                coeff = coeff * (1.0 + coeff_sigma * scale * noise)
            param_i = MPParam(Fs=float(p.Fs), P=int(p.P), M=M_new, coeff=coeff, Gain=float(getattr(p, "Gain", 1.0)))
            pa_base_list.append(MemoryPolynomialPA(param_i, backoff_db=args.backoff_db))

    elif args.pa_type == "gmp":
        Fs = 120e6
        P = int(args.gmp_P)
        M = int(args.gmp_M)
        L_lag = int(args.gmp_L_lag)
        L_lead = int(args.gmp_L_lead)
        K = (P + 1) // 2
        for _ in range(Nt):
            n_feat = K * M * (1 + L_lag + L_lead)
            coeff = np.zeros(n_feat, dtype=complex)
            coeff[0] = 1.0 + 0j
            coeff[1:] = 0.03 * (rng.standard_normal(n_feat - 1) + 1j * rng.standard_normal(n_feat - 1)) / np.sqrt(2.0)
            param = GMPParam(Fs=Fs, P=P, M=M, L_lag=L_lag, L_lead=L_lead, coeff=coeff, Gain=1.0)
            pa_base_list.append(GMPPA(param, backoff_db=args.backoff_db))

    elif args.pa_type == "wiener":
        Fs = 120e6
        P = int(args.wiener_P)
        K = (P + 1) // 2
        for _ in range(Nt):
            L = int(args.wiener_fir_len) + int(rng.integers(-args.wiener_fir_jitter, args.wiener_fir_jitter + 1))
            L = max(1, L)
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
            pa_base_list.append(WienerPA(param, backoff_db=args.backoff_db))

    elif args.pa_type == "nn":
        if x_for_nn_train is None:
            raise ValueError("x_for_nn_train is required for nn PA")
        teacher = args.pa_nn_teacher
        if teacher == "auto":
            pa_path = _resolve_file(args.pa_memory_models_mat)
            teacher = "mp" if pa_path is not None else "wiener"

        # build teacher PA list (shared)
        class _Tmp:
            pass

        tmp = _Tmp()
        for k, v in vars(args).items():
            setattr(tmp, k, v)
        tmp.pa_type = teacher
        tmp.pa_assignment = "same"  # teacher baseline

        # build one teacher PA (SISO) and train a NN on it
        pa_teacher_list, Fs, _ = build_pa_array(tmp, Nt=1, rng=rng, x_for_nn_train=None)
        pa_teacher = pa_teacher_list[0]
        y_teacher = pa_teacher.forward(x_for_nn_train)

        pa_nn = NeuralNetPA(
            M=int(args.pa_nn_M),
            hidden=int(args.pa_nn_hidden),
            depth=int(args.pa_nn_depth),
            backoff_db=args.backoff_db,
            device=args.pa_nn_device,
        )
        pa_nn.fit(x_for_nn_train, y_teacher, lr=float(args.pa_nn_lr), epochs=int(args.pa_nn_epochs), batch=int(args.pa_nn_batch))

        # shared NN base for all antennas
        for _ in range(Nt):
            pa_base_list.append(pa_nn)

    else:
        raise ValueError("Unsupported pa_type")

    # -------------------------
    # 2) Mismatch injection wrapper
    # -------------------------
    drive = rng.standard_normal(Nt) * float(args.pa_drive_sigma_db)
    gain = rng.standard_normal(Nt) * float(args.pa_gain_sigma_db)
    phase = rng.standard_normal(Nt) * float(args.pa_phase_sigma_deg)
    ampm = rng.standard_normal(Nt) * float(args.pa_ampm_sigma_deg)

    mismatch_report["drive_db"] = drive
    mismatch_report["gain_db"] = gain
    mismatch_report["phase_deg"] = phase
    mismatch_report["ampm_deg"] = ampm

    pa_list: List[object] = []
    for i in range(Nt):
        hmem = None
        if int(args.pa_mem_fir_len) > 1 and float(args.pa_mem_fir_sigma) > 0:
            hmem = random_mild_fir(rng, int(args.pa_mem_fir_len), sigma=float(args.pa_mem_fir_sigma))
        pa_list.append(
            PAMismatchWrapper(
                pa_base_list[i],
                drive_db=float(drive[i]),
                gain_db=float(gain[i]),
                phase_deg=float(phase[i]),
                ampm_deg=float(ampm[i]),
                lin_mem_fir=hmem,
            )
        )
    return pa_list, Fs, mismatch_report


def apply_pa_array(X: np.ndarray, pa_list: List[object]) -> np.ndarray:
    """Apply per-antenna PA model."""
    X = np.asarray(X)
    Ns, Nt = X.shape
    if len(pa_list) != Nt:
        raise ValueError("pa_list length mismatch")
    Y = np.zeros_like(X, dtype=complex)
    for i in range(Nt):
        Y[:, i] = pa_list[i].forward(X[:, i])
    return Y


# =========================
# DPD helpers
# =========================


def _infer_mp_dpd_cfg(args: argparse.Namespace, pa_hint: Optional[object]) -> Tuple[int, np.ndarray]:
    """Infer (P, Mvec) for MP-DPD.

    Rules:
    - P:
        * if --dpd_order is provided -> use it
        * else if pa_hint is a MemoryPolynomialPA -> inherit PA order
        * else default 9
    - M:
        * if --dpd_mem > 0 -> constant memory for all odd orders
        * else if pa_hint has param.M -> inherit (truncate/extend)
        * else default mem=2
    """
    # order
    if getattr(args, "dpd_order", None) is not None:
        P = int(args.dpd_order)
    else:
        P = int(getattr(getattr(pa_hint, "param", None), "P", 9))

    # memory
    if int(args.dpd_mem) > 0:
        M = mp_memory_vector(P, int(args.dpd_mem))
    else:
        paM = getattr(getattr(pa_hint, "param", None), "M", None)
        if paM is None:
            M = mp_memory_vector(P, 2)
        else:
            paM = np.asarray(paM).astype(int).reshape(-1)
            K = (P + 1) // 2
            if len(paM) >= K:
                M = paM[:K]
            else:
                M = np.concatenate([paM, np.full(K - len(paM), paM[-1], dtype=int)])
    return P, M


def build_dpd_instance(args: argparse.Namespace, seed: int, pa_hint: Optional[object] = None) -> object:
    """Create an *unfitted* DPD instance (MP/GMP/NN).

    pa_hint is used only for MP-DPD default inheritance.
    """
    if args.dpd_type == "mp":
        P, M = _infer_mp_dpd_cfg(args, pa_hint)
        return MPDPD(P=P, M=M, ridge=float(args.ridge))
    if args.dpd_type == "gmp":
        return GMPDPD(
            P=int(args.dpd_gmp_P),
            M=int(args.dpd_gmp_M),
            L_lag=int(args.dpd_gmp_L_lag),
            L_lead=int(args.dpd_gmp_L_lead),
            ridge=float(args.ridge),
        )
    if args.dpd_type == "nn":
        return NeuralNetDPD(
            M=int(args.dpd_nn_M),
            hidden=int(args.dpd_nn_hidden),
            depth=int(args.dpd_nn_depth),
            device=args.dpd_nn_device,
        )
    raise ValueError("Unsupported dpd_type")


# =========================
# Simulation
# =========================


def simulate(args: argparse.Namespace) -> Dict[str, object]:
    rng = default_rng(args.seed)

    Nt = int(args.nt)
    K = int(args.k)
    n_rf = int(args.nrf)
    if n_rf < K:
        raise ValueError("Need nrf >= K")

    # -------------------------
    # Channel and precoder
    # -------------------------
    H = rayleigh_channel(K=K, Nt=Nt, seed=args.seed + 10)
    FRF, FBB = design_hybrid_precoder(H, n_rf=n_rf, method=args.precoder, phase_bits=args.phase_bits, rzf_reg=args.rzf_reg)
    FRF, FBB = normalize_precoder(FRF, FBB, K=K)

    # -------------------------
    # Waveform per user
    # -------------------------
    syms = np.zeros((args.Nsym, K), dtype=complex)
    for k in range(K):
        syms[:, k] = generate_qam_symbols(args.Nsym, args.qam_order, seed=args.seed + 100 + k)

    S = np.zeros((args.Nsym * args.osf, K), dtype=complex)
    d = None
    rrc = None
    for k in range(K):
        s_k, d_k, rrc_k = pulse_shape(syms[:, k], over=args.osf, alpha=args.rolloff, D=args.rrc_delay)
        S[:, k] = s_k
        if d is None:
            d, rrc = d_k, rrc_k

    # -------------------------
    # Apply hybrid precoder (linear)
    # -------------------------
    Xrf = S @ FBB.T        # Nsamp x n_rf
    Xant = Xrf @ FRF.T     # Nsamp x Nt

    # -------------------------
    # Build PA array + mismatches
    # -------------------------
    pa_list, Fs, mismatch = build_pa_array(args, Nt=Nt, rng=rng, x_for_nn_train=Xant[:, 0])

    # -------------------------
    # PA only
    # -------------------------
    Y_pa = apply_pa_array(Xant, pa_list)

    # -------------------------
    # DPD
    # -------------------------
    if args.dpd == "none":
        Uant = Xant
        Y_dpd = Y_pa
        dpd_models = None

    elif args.dpd == "per_antenna":
        Uant = np.zeros_like(Xant)
        dpd_models = []

        # training subset
        Ntrain = int(min(len(Xant), args.dpd_train_syms * args.osf)) if args.dpd_train_syms > 0 else len(Xant)

        if args.dpd_type == "nn" and args.dpd_nn_mode == "shared":
            # stack multiple antennas to train a shared inverse
            Xtr = Xant[:Ntrain, :].reshape(-1)
            Ytr = Y_pa[:Ntrain, :].reshape(-1)
            dpd_shared = build_dpd_instance(args, seed=args.seed + 999, pa_hint=getattr(pa_list[0], "pa", None))
            dpd_shared.fit_ila(Xtr, Ytr, lr=args.dpd_nn_lr, epochs=args.dpd_nn_epochs, batch=args.dpd_nn_batch, seed=args.seed + 999)
            for i in range(Nt):
                u_i = dpd_shared.forward(Xant[:, i])
                u_i = clip_to_rms(u_i, args.dpd_clip_rms)
                Uant[:, i] = u_i
            dpd_models = dpd_shared
        else:
            # per-antenna DPD (MP/GMP always per-antenna; NN per_antenna mode)
            for i in range(Nt):
                dpd_i = build_dpd_instance(args, seed=args.seed + 2000 + i, pa_hint=getattr(pa_list[i], "pa", None))
                if args.dpd_type == "nn":
                    dpd_i.fit_ila(Xant[:Ntrain, i], Y_pa[:Ntrain, i], lr=args.dpd_nn_lr, epochs=args.dpd_nn_epochs, batch=args.dpd_nn_batch, seed=args.seed + 2000 + i)
                else:
                    dpd_i.fit_ila(Xant[:Ntrain, i], Y_pa[:Ntrain, i])
                u_i = dpd_i.forward(Xant[:, i])
                u_i = clip_to_rms(u_i, args.dpd_clip_rms)
                Uant[:, i] = u_i
                dpd_models.append(dpd_i)

        Y_dpd = apply_pa_array(Uant, pa_list)

    elif args.dpd == "per_rf":
        # low-complexity: train one DPD per RF chain based on combined observation
        Ntrain = int(min(len(Xrf), args.dpd_train_syms * args.osf)) if args.dpd_train_syms > 0 else len(Xrf)

        # combined feedback (simple): z_r = sum_i conj(FRF[i,r]) * y_i
        Z = Y_pa @ np.conj(FRF)
        Urf = np.zeros_like(Xrf)
        dpd_models = []
        for r in range(n_rf):
            dpd_r = build_dpd_instance(args, seed=args.seed + 3000 + r, pa_hint=getattr(pa_list[0], "pa", None))
            if args.dpd_type == "nn":
                dpd_r.fit_ila(Xrf[:Ntrain, r], Z[:Ntrain, r], lr=args.dpd_nn_lr, epochs=args.dpd_nn_epochs, batch=args.dpd_nn_batch, seed=args.seed + 3000 + r)
            else:
                dpd_r.fit_ila(Xrf[:Ntrain, r], Z[:Ntrain, r])
            u_r = dpd_r.forward(Xrf[:, r])
            u_r = clip_to_rms(u_r, args.dpd_clip_rms)
            Urf[:, r] = u_r
            dpd_models.append(dpd_r)
        Uant = Urf @ FRF.T
        Y_dpd = apply_pa_array(Uant, pa_list)

    else:
        raise ValueError("dpd must be none|per_antenna|per_rf")

    # -------------------------
    # RX (multi-user) + AWGN
    # -------------------------
    R_lin = Xant @ H.T
    R_pa = Y_pa @ H.T
    R_dpd = Y_dpd @ H.T

    # estimate symbol-rate power at MF output for SNR calibration
    symhat_lin0 = np.zeros((args.Nsym, K), dtype=complex)
    for k in range(K):
        symhat_lin0[:, k] = rx_match_filter_and_sample(R_lin[:, k], rrc, over=args.osf, d=d, Nsym=args.Nsym)
    sigpow = float(np.mean(np.abs(symhat_lin0) ** 2))
    sigma2 = sigpow / (10.0 ** (args.snr_db / 10.0))
    noise = (rng.standard_normal(R_lin.shape) + 1j * rng.standard_normal(R_lin.shape)) * np.sqrt(sigma2 / 2.0)

    R_lin_n = R_lin + noise
    R_pa_n = R_pa + noise
    R_dpd_n = R_dpd + noise

    symhat_lin = np.zeros((args.Nsym, K), dtype=complex)
    symhat_pa = np.zeros((args.Nsym, K), dtype=complex)
    symhat_dpd = np.zeros((args.Nsym, K), dtype=complex)
    for k in range(K):
        symhat_lin[:, k] = rx_match_filter_and_sample(R_lin_n[:, k], rrc, over=args.osf, d=d, Nsym=args.Nsym)
        symhat_pa[:, k] = rx_match_filter_and_sample(R_pa_n[:, k], rrc, over=args.osf, d=d, Nsym=args.Nsym)
        symhat_dpd[:, k] = rx_match_filter_and_sample(R_dpd_n[:, k], rrc, over=args.osf, d=d, Nsym=args.Nsym)

    discard = int(args.rrc_delay)
    evm_lin = np.zeros(K)
    evm_pa = np.zeros(K)
    evm_dpd = np.zeros(K)
    for k in range(K):
        evm_lin[k], _ = evm_rms(syms[:, k], symhat_lin[:, k], discard=discard)
        evm_pa[k], _ = evm_rms(syms[:, k], symhat_pa[:, k], discard=discard)
        evm_dpd[k], _ = evm_rms(syms[:, k], symhat_dpd[:, k], discard=discard)

    # -------------------------
    # TX PSD / ACLR
    # -------------------------
    Rs = Fs / args.osf
    ch_bw = (1.0 + args.rolloff) * Rs
    adj = ch_bw
    f_pa, P_pa = average_psd_over_antennas(Y_pa, fs=Fs, nperseg=2048)
    f_d, P_d = average_psd_over_antennas(Y_dpd, fs=Fs, nperseg=2048)
    aclr_pa = aclr_from_psd(f_pa, P_pa, ch_bw=ch_bw, adj_offset=adj)
    aclr_d = aclr_from_psd(f_d, P_d, ch_bw=ch_bw, adj_offset=adj)

    return {
        "H": H,
        "FRF": FRF,
        "FBB": FBB,
        "syms": syms,
        "S": S,
        "Xrf": Xrf,
        "Xant": Xant,
        "Y_pa": Y_pa,
        "Uant": Uant,
        "Y_dpd": Y_dpd,
        "symhat_lin": symhat_lin,
        "symhat_pa": symhat_pa,
        "symhat_dpd": symhat_dpd,
        "evm_lin": evm_lin,
        "evm_pa": evm_pa,
        "evm_dpd": evm_dpd,
        "Fs": Fs,
        "sigma2": sigma2,
        "psd_pa": (f_pa, P_pa),
        "psd_dpd": (f_d, P_d),
        "aclr_pa": aclr_pa,
        "aclr_dpd": aclr_d,
        "mismatch": mismatch,
        "dpd_models": dpd_models,
    }


# =========================
# CLI
# =========================


def main():
    ap = argparse.ArgumentParser()
    # system size
    ap.add_argument("--nt", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--nrf", type=int, default=4)

    # waveform
    ap.add_argument("--qam_order", type=int, default=256)
    ap.add_argument("--Nsym", type=int, default=8000)
    ap.add_argument("--osf", type=int, default=8)
    ap.add_argument("--rolloff", type=float, default=0.22)
    ap.add_argument("--rrc_delay", type=int, default=12)
    ap.add_argument("--snr_db", type=float, default=25.0)

    # precoder
    ap.add_argument("--precoder", type=str, default="svd_phase_zf", choices=["svd_phase_zf", "random_zf", "fopt_altmin"])
    ap.add_argument("--phase_bits", type=int, default=0)
    ap.add_argument("--rzf_reg", type=float, default=0.0)

    # PA
    ap.add_argument("--pa_type", type=str, default="mp", choices=["mp", "gmp", "wiener", "nn"])
    ap.add_argument("--backoff_db", type=float, default=0.0)

    # measured MP dataset options
    ap.add_argument("--pa_memory_models_mat", type=str, default="PA_memory_models.mat")
    ap.add_argument("--p_tx_mat", type=str, default="P_TX_Case1.mat")
    ap.add_argument("--gains", type=str, default="measured", choices=["equal", "measured"])
    ap.add_argument("--pa_assignment", type=str, default="random", choices=["same", "sequential", "random"])
    ap.add_argument("--pa_index", type=int, default=1)
    ap.add_argument("--rf", type=int, default=0, choices=[0, 1])

    # synthetic PA params
    ap.add_argument("--mp_P", type=int, default=9)
    ap.add_argument("--mp_mem", type=int, default=2)
    ap.add_argument("--gmp_P", type=int, default=9)
    ap.add_argument("--gmp_M", type=int, default=3)
    ap.add_argument("--gmp_L_lag", type=int, default=1)
    ap.add_argument("--gmp_L_lead", type=int, default=1)
    ap.add_argument("--wiener_P", type=int, default=9)
    ap.add_argument("--wiener_fir_len", type=int, default=5)
    ap.add_argument("--wiener_fir_jitter", type=int, default=0)

    # nn PA
    ap.add_argument("--pa_nn_teacher", type=str, default="auto", choices=["auto", "mp", "gmp", "wiener"])
    ap.add_argument("--pa_nn_M", type=int, default=7)
    ap.add_argument("--pa_nn_hidden", type=int, default=128)
    ap.add_argument("--pa_nn_depth", type=int, default=3)
    ap.add_argument("--pa_nn_epochs", type=int, default=5)
    ap.add_argument("--pa_nn_lr", type=float, default=1e-3)
    ap.add_argument("--pa_nn_batch", type=int, default=4096)
    ap.add_argument("--pa_nn_device", type=str, default="cpu")

    # PA mismatch injection (set std devs; 0 disables each component)
    ap.add_argument("--pa_drive_sigma_db", type=float, default=0.5, help="input drive std dev [dB]")
    ap.add_argument("--pa_gain_sigma_db", type=float, default=0.5, help="output gain std dev [dB]")
    ap.add_argument("--pa_phase_sigma_deg", type=float, default=5.0, help="const phase std dev [deg]")
    ap.add_argument("--pa_ampm_sigma_deg", type=float, default=3.0, help="AM/PM coef std dev [deg] at RMS")
    ap.add_argument("--pa_mem_fir_len", type=int, default=3, help="extra linear memory FIR length (<=1 disables)")
    ap.add_argument("--pa_mem_fir_sigma", type=float, default=0.02, help="extra memory FIR tap scale")
    # MP internal mismatch
    ap.add_argument("--pa_mp_mem_jitter", type=int, default=0, help="MP memory length jitter (+/-J per order)")
    ap.add_argument("--pa_mp_coeff_sigma", type=float, default=0.0, help="relative coeff perturb std (excluding main tap)")

    # DPD
    ap.add_argument("--dpd", type=str, default="none", choices=["none", "per_antenna", "per_rf"], help="DPD mode")
    ap.add_argument("--dpd_type", type=str, default="mp", choices=["mp", "gmp", "nn"], help="DPD model type")
    ap.add_argument("--dpd_train_syms", type=int, default=4000)
    ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument("--dpd_clip_rms", type=float, default=3.0)

    # MP DPD
    ap.add_argument("--dpd_order", type=int, default=None, help="MP DPD odd order (None -> inherit if possible)")
    ap.add_argument("--dpd_mem", type=int, default=-1, help=">0: MP DPD memory (all orders), -1/0: inherit if possible")
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
    ap.add_argument("--dpd_nn_mode", type=str, default="per_antenna", choices=["per_antenna", "shared"], help="only for dpd=per_antenna, dpd_type=nn")

    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    res = simulate(args)

    print("=== Hybrid MU-MIMO TX Simulation ===")
    print(f"Nt={args.nt}, K={args.k}, nRF={args.nrf}, precoder={args.precoder}, phase_bits={args.phase_bits}")
    print(f"Waveform: {args.qam_order}-QAM, Nsym={args.Nsym}, osf={args.osf}, rolloff={args.rolloff}, Fs={res['Fs']/1e6:.1f} Msps")
    print(f"Channel: Rayleigh flat fading, SNR={args.snr_db:.1f} dB")
    print(f"PA: type={args.pa_type}, backoff={args.backoff_db:.2f} dB, gains={args.gains}, assignment={args.pa_assignment}")
    print(
        "PA mismatch (std): "
        f"drive={args.pa_drive_sigma_db} dB, gain={args.pa_gain_sigma_db} dB, "
        f"phase={args.pa_phase_sigma_deg} deg, AMPM={args.pa_ampm_sigma_deg} deg, "
        f"memFIR(L={args.pa_mem_fir_len}, sigma={args.pa_mem_fir_sigma})"
    )
    print(f"DPD: mode={args.dpd}, type={args.dpd_type}, train_syms={args.dpd_train_syms}, ridge={args.ridge:g}, clip={args.dpd_clip_rms}*RMS")
    print("")
    for k in range(args.k):
        print(
            f"User {k+1:02d} EVM: "
            f"Linear={100*res['evm_lin'][k]:6.3f}% | "
            f"PA={100*res['evm_pa'][k]:6.3f}% | "
            f"DPD={100*res['evm_dpd'][k]:6.3f}%"
        )
    print("")
    print(f"Avg EVM: Linear={100*np.mean(res['evm_lin']):.3f}% | PA={100*np.mean(res['evm_pa']):.3f}% | DPD={100*np.mean(res['evm_dpd']):.3f}%")
    print("")
    print("TX ACLR (avg PSD over antennas):")
    print(
        f"  PA only:  worst={res['aclr_pa']['ACLR_worst_dB']:.2f} dB "
        f"(U={res['aclr_pa']['ACLR_u_dB']:.2f}, L={res['aclr_pa']['ACLR_l_dB']:.2f})"
    )
    print(
        f"  With DPD: worst={res['aclr_dpd']['ACLR_worst_dB']:.2f} dB "
        f"(U={res['aclr_dpd']['ACLR_u_dB']:.2f}, L={res['aclr_dpd']['ACLR_l_dB']:.2f})"
    )

    if args.plot:
        f_pa, P_pa = res["psd_pa"]
        f_d, P_d = res["psd_dpd"]
        plt.figure()
        plt.plot(f_pa / 1e6, 10 * np.log10(np.maximum(P_pa, 1e-20)), label="PA only")
        plt.plot(f_d / 1e6, 10 * np.log10(np.maximum(P_d, 1e-20)), label="DPD+PA")
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("PSD [dB/Hz]")
        plt.title("TX PSD (avg over antennas)")
        plt.grid(True)
        plt.legend()

        # constellation of user 1
        k0 = 0
        plt.figure()
        plt.plot(res["symhat_pa"][:, k0].real, res["symhat_pa"][:, k0].imag, ".", alpha=0.25, label="PA")
        plt.plot(res["symhat_dpd"][:, k0].real, res["symhat_dpd"][:, k0].imag, ".", alpha=0.25, label="DPD")
        plt.axis("equal")
        plt.grid(True)
        plt.title(f"RX Constellation (User {k0+1})")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
