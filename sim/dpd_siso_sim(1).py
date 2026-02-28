#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dpd_siso_sim.py

SISO PA + DPD simulation with **swappable PA behavioral models**.

- Waveform: RRC-shaped square QAM (single-carrier)
- Channel: none (TX-only PA/DPD evaluation)
- PA model (selectable):
    * MP (Memory Polynomial)  [measured Lund dataset if .mat available, else synthetic]
    * GMP (Generalized Memory Polynomial) [synthetic]
    * Wiener (LTI + memoryless polynomial) [synthetic]
    * NN (NeuralNetPA, PyTorch) [trained to mimic a teacher model]
- DPD: MP predistorter trained with ILA (least squares + ridge)
- Metrics: EVM, PSD, ACLR

This is an updated version of the earlier script, extended so you can switch
PA types via CLI arguments.

Requirements
------------
- numpy, scipy, matplotlib
- pa_models.py (provided alongside this script)
- (optional) torch for NN-based PA

Examples
--------
# Measured MP PA (requires PA_memory_models.mat and P_TX_Case1.mat)
python dpd_siso_sim.py --pa_type mp --gains measured --plot

# Synthetic GMP PA
python dpd_siso_sim.py --pa_type gmp --gmp_P 7 --gmp_M 3 --gmp_L_lag 1 --gmp_L_lead 1 --plot

# Synthetic Wiener PA
python dpd_siso_sim.py --pa_type wiener --wiener_P 7 --wiener_fir_len 5 --plot

# NN-based PA (train NN to mimic a teacher; auto selects teacher)
python dpd_siso_sim.py --pa_type nn --nn_teacher auto --nn_epochs 5 --plot

Notes
-----
- Polynomial behavioral models can "blow up" if DPD increases peaks too much.
  A magnitude clipper is provided for predistorted samples.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
from numpy.random import default_rng
from scipy import signal
import matplotlib.pyplot as plt

# --- import PA models module (must be placed next to this script) ---
try:
    from pa_models import (
        MPParam,
        MemoryPolynomialPA,
        load_lund_mp_dataset,
        GMPParam,
        GMPPA,
        WienerParam,
        WienerPA,
        NeuralNetPA,
        apply_mp,
        fit_mp,
        clip_complex,
    )
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "[ERROR] pa_models.py import failed. Put pa_models.py in the same folder as this script.\n"
        f"Reason: {e}"
    )


# =============================
# Pulse shaping (RRC)
# =============================

def rrc_taps(alpha: float, span: int, sps: int) -> np.ndarray:
    """Root Raised Cosine (RRC) FIR taps (unit-energy normalization)."""
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1].")
    if span <= 0 or sps <= 0:
        raise ValueError("span and sps must be positive.")

    N = span * sps
    t = np.arange(-N / 2, N / 2 + 1) / sps  # time in symbol periods (T=1)
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
    sym_over = np.zeros(len(sym) * over, dtype=complex)
    sym_over[::over] = sym

    rrc = rrc_taps(alpha=alpha, span=2 * D, sps=over)
    d = int(np.argmax(rrc))

    s_full = np.convolve(rrc, sym_over)
    s = s_full[d : d + len(sym) * over]
    return s, d, rrc


def rx_match_filter_and_sample(wave: np.ndarray, rrc: np.ndarray, over: int, d: int, Nsym: int) -> np.ndarray:
    """Matched filter with the same RRC pulse and downsample to symbol rate."""
    wave = np.asarray(wave).reshape(-1)
    mf_full = np.convolve(rrc, wave)
    mf = mf_full[d : d + Nsym * over]
    return mf[::over]


# =============================
# QAM
# =============================

def square_qam_alphabet(M: int, normalize_avg_power: bool = False) -> np.ndarray:
    """MATLAB-style square QAM alphabet."""
    m = int(np.sqrt(M))
    if m * m != M:
        raise ValueError("M must be a perfect square (e.g., 16, 64, 256).")
    tmp = np.arange(-(m - 1), m, 2)
    alphabet = np.kron(tmp, np.ones(m)) + 1j * np.kron(np.ones(m), tmp)
    alphabet = alphabet.astype(complex)
    if normalize_avg_power:
        alphabet = alphabet / np.sqrt(np.mean(np.abs(alphabet) ** 2))
    return alphabet


def generate_qam_symbols(Nsym: int, M: int, seed: int = 0, normalize_avg_power: bool = False) -> np.ndarray:
    rng = default_rng(seed)
    alphabet = square_qam_alphabet(M, normalize_avg_power=normalize_avg_power)
    idx = rng.integers(low=0, high=len(alphabet), size=Nsym)
    return alphabet[idx]


# =============================
# Metrics
# =============================

def estimate_scalar_ls(ref: np.ndarray, meas: np.ndarray) -> complex:
    ref = np.asarray(ref).reshape(-1)
    meas = np.asarray(meas).reshape(-1)
    denom = np.vdot(ref, ref)
    if denom == 0:
        return 0.0 + 0.0j
    return np.vdot(ref, meas) / denom


def evm_rms(ref: np.ndarray, meas: np.ndarray, discard: int = 0) -> Tuple[float, complex]:
    ref = np.asarray(ref).reshape(-1)
    meas = np.asarray(meas).reshape(-1)
    if discard > 0:
        ref = ref[discard:-discard]
        meas = meas[discard:-discard]
    g = estimate_scalar_ls(ref, meas)
    err = meas - g * ref
    evm = np.sqrt(np.mean(np.abs(err) ** 2) / np.mean(np.abs(g * ref) ** 2))
    return float(evm), g


def welch_psd(x: np.ndarray, fs: float, nperseg: int = 2048, beta: float = 7.0, noverlap: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x).reshape(-1)
    if noverlap is None:
        noverlap = nperseg // 2
    f, Pxx = signal.welch(
        x,
        fs=fs,
        window=("kaiser", beta),
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
        scaling="density",
    )
    idx = np.argsort(f)
    return f[idx], Pxx[idx]


def integrate_band(f: np.ndarray, Pxx: np.ndarray, f1: float, f2: float) -> float:
    mask = (f >= f1) & (f <= f2)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(Pxx[mask], f[mask]))


def aclr_from_psd(f: np.ndarray, Pxx: np.ndarray, ch_bw: float, adj_offset: float) -> Dict[str, float]:
    P_main = integrate_band(f, Pxx, -ch_bw / 2, ch_bw / 2)
    P_adj_u = integrate_band(f, Pxx, adj_offset - ch_bw / 2, adj_offset + ch_bw / 2)
    P_adj_l = integrate_band(f, Pxx, -adj_offset - ch_bw / 2, -adj_offset + ch_bw / 2)

    ACLR_u = 10 * np.log10(P_main / P_adj_u) if P_adj_u > 0 else np.inf
    ACLR_l = 10 * np.log10(P_main / P_adj_l) if P_adj_l > 0 else np.inf
    ACLR_worst = float(min(ACLR_u, ACLR_l))

    return {
        "P_main": float(P_main),
        "P_adj_u": float(P_adj_u),
        "P_adj_l": float(P_adj_l),
        "ACLR_u_dB": float(ACLR_u),
        "ACLR_l_dB": float(ACLR_l),
        "ACLR_worst_dB": ACLR_worst,
    }


# =============================
# PA builders
# =============================

def _resolve_existing(path_str: str) -> Optional[str]:
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


def try_load_measured_mp_models(pa_memory_models_mat: str, p_tx_mat: str, gains: str) -> Tuple[Optional[list], Optional[float]]:
    """Try loading the Lund measured MP dataset.

    Returns (models, Fs). If loading fails, returns (None, None).
    """
    try:
        pa_path = _resolve_existing(pa_memory_models_mat)
        if pa_path is None:
            return None, None
        ptx_path = _resolve_existing(p_tx_mat) if gains == "measured" else None
        if gains == "measured" and ptx_path is None:
            return None, None
        models, Fs = load_lund_mp_dataset(pa_path, ptx_path, gains=gains)
        return models, float(Fs)
    except Exception:
        return None, None


def make_synthetic_mp(Fs: float, P: int, mem_other_orders: int, strength: float, seed: int = 0) -> MPParam:
    """Create a stable-ish synthetic MP PA parameter."""
    if P % 2 == 0:
        raise ValueError("MP P must be odd")
    K = (P + 1) // 2
    M = np.ones(K, dtype=int)
    if K >= 2:
        M[1:] = int(mem_other_orders)

    rng = default_rng(seed)
    coeff = np.zeros(int(np.sum(M)), dtype=complex)

    col = 0
    for p_idx in range(K):
        for m in range(M[p_idx]):
            if p_idx == 0 and m == 0:
                coeff[col] = 1.0 + 0.0j
            else:
                mag = (0.12 * (0.55**p_idx) * (0.65**m)) * float(strength)
                phase = rng.uniform(-np.pi, np.pi)
                coeff[col] = -mag * np.exp(1j * phase)
            col += 1

    return MPParam(Fs=float(Fs), P=int(P), M=M, coeff=coeff, Gain=1.0)


def make_synthetic_gmp(Fs: float, P: int, M: int, L_lag: int, L_lead: int, strength: float, seed: int = 0) -> GMPParam:
    if P % 2 == 0:
        raise ValueError("GMP P must be odd")
    K = (P + 1) // 2
    n_feat = K * int(M) * (1 + int(L_lag) + int(L_lead))

    rng = default_rng(seed)
    coeff = np.zeros(n_feat, dtype=complex)
    coeff[0] = 1.0 + 0.0j
    for i in range(1, n_feat):
        mag = float(strength) * 0.05 * (0.985**i)
        phase = rng.uniform(-np.pi, np.pi)
        coeff[i] = -mag * np.exp(1j * phase)

    return GMPParam(Fs=float(Fs), P=int(P), M=int(M), L_lag=int(L_lag), L_lead=int(L_lead), coeff=coeff, Gain=1.0)


def make_synthetic_wiener(Fs: float, P: int, fir_len: int, strength: float, seed: int = 0) -> WienerParam:
    if P % 2 == 0:
        raise ValueError("Wiener P must be odd")
    K = (P + 1) // 2

    rng = default_rng(seed)

    L = int(max(1, fir_len))
    taps = np.zeros(L, dtype=complex)
    for i in range(L):
        mag = 1.0 if i == 0 else (0.35 * (0.45 ** (i - 1)))
        phase = rng.uniform(-np.pi, np.pi)
        taps[i] = mag * np.exp(1j * phase)
    taps = taps / np.sqrt(np.sum(np.abs(taps) ** 2))

    poly_a = np.zeros(K, dtype=complex)
    poly_a[0] = 1.0 + 0.0j
    base = 0.14 * float(strength)
    for p_idx in range(1, K):
        mag = base * (0.35 ** (p_idx - 1))
        phase = rng.uniform(-0.2, 0.2)
        poly_a[p_idx] = ((-1) ** p_idx) * mag * np.exp(1j * phase)

    return WienerParam(Fs=float(Fs), h=taps, poly_a=poly_a, Gain=1.0)


def build_pa_siso(args, x_for_training: np.ndarray, mp_models: Optional[list], Fs_hint: float) -> Tuple[object, float, Dict[str, object], Optional[np.ndarray], Optional[int]]:
    """Build the PA object (with .forward) depending on args.pa_type.

    Returns
    -------
    pa : object with forward(x)->y
    Fs : sampling rate for PSD/ACLR
    info : dict for printing
    mp_mem_vec : if PA is MP and has per-order memory (for default DPD), else None
    pa_order : odd order if meaningful (MP/GMP/Wiener), else None
    """
    rng = default_rng(args.seed)

    pa_type = args.pa_type.lower()
    Fs = float(Fs_hint)

    info: Dict[str, object] = {"pa_type": pa_type}

    indTX = 2 * args.pa_index - 1 + args.rf
    if indTX < 1 or indTX > 100:
        raise ValueError("indTX out of range. Use pa_index in 1..50 and rf in {0,1}.")

    if pa_type == "mp":
        if mp_models is not None:
            param: MPParam = mp_models[indTX - 1]
            Fs = float(param.Fs)
            pa = MemoryPolynomialPA(param, backoff_db=args.backoff_db)
            info.update({
                "source": "measured_mat" if _resolve_existing(args.pa_memory_models_mat) else "measured_loaded",
                "indTX": indTX,
                "P": param.P,
                "M": param.M.tolist(),
                "Gain": float(param.Gain),
            })
            return pa, Fs, info, param.M.copy(), int(param.P)

        param = make_synthetic_mp(Fs=Fs, P=int(args.mp_P), mem_other_orders=args.mp_mem, strength=args.pa_strength, seed=args.seed)
        pa = MemoryPolynomialPA(param, backoff_db=args.backoff_db)
        info.update({"source": "synthetic", "P": param.P, "M": param.M.tolist(), "Gain": float(param.Gain)})
        return pa, Fs, info, param.M.copy(), int(param.P)

    if pa_type == "gmp":
        param = make_synthetic_gmp(
            Fs=Fs,
            P=int(args.gmp_P),
            M=int(args.gmp_M),
            L_lag=int(args.gmp_L_lag),
            L_lead=int(args.gmp_L_lead),
            strength=args.pa_strength,
            seed=args.seed,
        )
        pa = GMPPA(param, backoff_db=args.backoff_db)
        info.update({
            "source": "synthetic",
            "P": param.P,
            "M": param.M,
            "L_lag": param.L_lag,
            "L_lead": param.L_lead,
        })
        return pa, Fs, info, None, int(param.P)

    if pa_type == "wiener":
        param = make_synthetic_wiener(Fs=Fs, P=int(args.wiener_P), fir_len=int(args.wiener_fir_len), strength=args.pa_strength, seed=args.seed)
        pa = WienerPA(param, backoff_db=args.backoff_db)
        info.update({"source": "synthetic", "P": int(args.wiener_P), "fir_len": int(len(param.h))})
        return pa, Fs, info, None, int(args.wiener_P)

    if pa_type == "nn":
        teacher = args.nn_teacher.lower()
        if teacher == "auto":
            teacher = "mp" if mp_models is not None else "wiener"

        info["nn_teacher"] = teacher

        # teacher PA
        if teacher == "mp":
            if mp_models is not None:
                param = mp_models[indTX - 1]
                Fs = float(param.Fs)
                teacher_pa = MemoryPolynomialPA(param, backoff_db=args.backoff_db)
                info.update({"teacher_source": "measured_mp", "teacher_P": int(param.P)})
            else:
                param = make_synthetic_mp(Fs=Fs, P=int(args.mp_P), mem_other_orders=args.mp_mem, strength=args.pa_strength, seed=args.seed)
                teacher_pa = MemoryPolynomialPA(param, backoff_db=args.backoff_db)
                info.update({"teacher_source": "synthetic_mp", "teacher_P": int(param.P)})

        elif teacher == "gmp":
            param = make_synthetic_gmp(
                Fs=Fs,
                P=int(args.gmp_P),
                M=int(args.gmp_M),
                L_lag=int(args.gmp_L_lag),
                L_lead=int(args.gmp_L_lead),
                strength=args.pa_strength,
                seed=args.seed,
            )
            teacher_pa = GMPPA(param, backoff_db=args.backoff_db)
            info.update({"teacher_source": "synthetic_gmp", "teacher_P": int(param.P)})

        elif teacher == "wiener":
            param = make_synthetic_wiener(Fs=Fs, P=int(args.wiener_P), fir_len=int(args.wiener_fir_len), strength=args.pa_strength, seed=args.seed)
            teacher_pa = WienerPA(param, backoff_db=args.backoff_db)
            info.update({"teacher_source": "synthetic_wiener", "teacher_P": int(args.wiener_P)})

        else:
            raise ValueError("nn_teacher must be one of {auto, mp, gmp, wiener}")

        # training subset
        N = len(x_for_training)
        n_train_syms = int(min(max(200, args.nn_train_syms), args.Nsym))
        n_train = int(min(N, n_train_syms * args.osf))
        x_train = x_for_training[:n_train]
        y_train = teacher_pa.forward(x_train)

        device = args.nn_device
        try:
            import torch

            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
        except Exception:
            device = "cpu"

        try:
            import torch

            torch.manual_seed(int(args.seed))
        except Exception:
            pass

        nn_pa = NeuralNetPA(
            M=int(args.nn_mem),
            hidden=int(args.nn_hidden),
            depth=int(args.nn_depth),
            backoff_db=float(args.backoff_db),
            device=device,
        )
        nn_pa.fit(x_train, y_train, lr=float(args.nn_lr), epochs=int(args.nn_epochs), batch=int(args.nn_batch))

        info.update({
            "source": "nn_trained",
            "nn_mem": int(args.nn_mem),
            "nn_hidden": int(args.nn_hidden),
            "nn_depth": int(args.nn_depth),
            "nn_epochs": int(args.nn_epochs),
            "nn_device": device,
            "train_samples": int(n_train),
        })
        return nn_pa, Fs, info, None, None

    raise ValueError("pa_type must be one of {mp,gmp,wiener,nn}")


# =============================
# Main
# =============================

def main() -> None:
    ap = argparse.ArgumentParser()

    # waveform
    ap.add_argument("--Nsym", type=int, default=30000)
    ap.add_argument("--osf", type=int, default=8)
    ap.add_argument("--qam_order", type=int, default=256)
    ap.add_argument("--rolloff", type=float, default=0.22)
    ap.add_argument("--rrc_delay", type=int, default=12)

    # PA selection
    ap.add_argument("--pa_type", type=str, default="mp", choices=["mp", "gmp", "wiener", "nn"], help="PA model type")
    ap.add_argument("--fs", type=float, default=120e6, help="Sample rate for synthetic PA types")
    ap.add_argument("--pa_strength", type=float, default=1.0, help="Nonlinearity strength (synthetic models)")

    # measured MP dataset (used when pa_type=mp)
    ap.add_argument("--pa_memory_models_mat", type=str, default="PA_memory_models.mat")
    ap.add_argument("--p_tx_mat", type=str, default="P_TX_Case1.mat")
    ap.add_argument("--gains", type=str, choices=["equal", "measured"], default="measured")
    ap.add_argument("--pa_index", type=int, default=1)
    ap.add_argument("--rf", type=int, default=0, choices=[0, 1])

    # synthetic MP
    ap.add_argument("--mp_P", type=int, default=9)
    ap.add_argument("--mp_mem", type=int, default=2)

    # synthetic GMP
    ap.add_argument("--gmp_P", type=int, default=7)
    ap.add_argument("--gmp_M", type=int, default=3)
    ap.add_argument("--gmp_L_lag", type=int, default=1)
    ap.add_argument("--gmp_L_lead", type=int, default=1)

    # synthetic Wiener
    ap.add_argument("--wiener_P", type=int, default=7)
    ap.add_argument("--wiener_fir_len", type=int, default=5)

    # NN
    ap.add_argument("--nn_teacher", type=str, default="auto", choices=["auto", "mp", "gmp", "wiener"])
    ap.add_argument("--nn_mem", type=int, default=7)
    ap.add_argument("--nn_hidden", type=int, default=128)
    ap.add_argument("--nn_depth", type=int, default=3)
    ap.add_argument("--nn_epochs", type=int, default=5)
    ap.add_argument("--nn_lr", type=float, default=1e-3)
    ap.add_argument("--nn_batch", type=int, default=4096)
    ap.add_argument("--nn_device", type=str, default="cpu")
    ap.add_argument("--nn_train_syms", type=int, default=6000)

    # backoff & DPD
    ap.add_argument("--backoff_db", type=float, default=0.0)
    ap.add_argument("--dpd_order", type=int, default=None)
    ap.add_argument("--dpd_mem", type=int, default=-1, help="DPD memory length (>=1). -1 inherits PA memory if available")
    ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument("--discard_syms", type=int, default=12)
    ap.add_argument("--dpd_clip_rms", type=float, default=3.0)
    ap.add_argument("--match_input_power", action="store_true")

    # misc
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--plot", action="store_true")

    args = ap.parse_args()

    # Load measured MP models if available
    mp_models, Fs_meas = try_load_measured_mp_models(args.pa_memory_models_mat, args.p_tx_mat, gains=args.gains)
    Fs_in = float(Fs_meas) if Fs_meas is not None else float(args.fs)

    # Waveform
    sym = generate_qam_symbols(args.Nsym, args.qam_order, seed=args.seed + 1, normalize_avg_power=False)
    x, d, rrc = pulse_shape(sym, over=args.osf, alpha=args.rolloff, D=args.rrc_delay)

    # PA
    pa, Fs_in, pa_info, pa_mem_vec, pa_order = build_pa_siso(args, x_for_training=x, mp_models=mp_models, Fs_hint=Fs_in)

    y = pa.forward(x)

    # DPD defaults
    if args.dpd_order is not None:
        P_dpd = int(args.dpd_order)
    elif pa_order is not None:
        P_dpd = int(pa_order)
    else:
        P_dpd = 9
    if P_dpd % 2 == 0:
        raise ValueError("dpd_order must be odd")

    Kd = (P_dpd + 1) // 2

    if args.dpd_mem is not None and args.dpd_mem > 0:
        M_dpd = np.full(Kd, int(args.dpd_mem), dtype=int)
        M_dpd[0] = max(1, M_dpd[0])
    elif pa_mem_vec is not None:
        if len(pa_mem_vec) == Kd:
            M_dpd = pa_mem_vec.astype(int)
        elif len(pa_mem_vec) > Kd:
            M_dpd = pa_mem_vec[:Kd].astype(int)
        else:
            M_dpd = np.concatenate([pa_mem_vec.astype(int), np.ones(Kd - len(pa_mem_vec), dtype=int)])
    else:
        M_dpd = np.full(Kd, 2, dtype=int)
        M_dpd[0] = 1

    # ILA: fit postdistorter x ≈ g(y)
    c_dpd = fit_mp(input_signal=y, target_signal=x, P=P_dpd, M=M_dpd, ridge=args.ridge)
    u = apply_mp(x, c_dpd, P=P_dpd, M=M_dpd)

    if args.dpd_clip_rms > 0:
        u_rms = float(np.sqrt(np.mean(np.abs(u) ** 2)))
        u = clip_complex(u, peak=args.dpd_clip_rms * u_rms)

    if args.match_input_power:
        p_u = np.mean(np.abs(u) ** 2)
        p_x = np.mean(np.abs(x) ** 2)
        if p_u > 0:
            u = u * np.sqrt(p_x / p_u)

    y_dpd = pa.forward(u)

    # EVM
    sym_hat = rx_match_filter_and_sample(y, rrc, over=args.osf, d=d, Nsym=args.Nsym)
    sym_hat_dpd = rx_match_filter_and_sample(y_dpd, rrc, over=args.osf, d=d, Nsym=args.Nsym)
    evm0, _ = evm_rms(sym, sym_hat, discard=args.discard_syms)
    evm1, _ = evm_rms(sym, sym_hat_dpd, discard=args.discard_syms)

    # PSD + ACLR
    f0, P0 = welch_psd(y, Fs_in)
    f1, P1 = welch_psd(y_dpd, Fs_in)

    Rs = Fs_in / args.osf
    ch_bw = (1.0 + args.rolloff) * Rs
    adj_offset = ch_bw

    aclr0 = aclr_from_psd(f0, P0, ch_bw=ch_bw, adj_offset=adj_offset)
    aclr1 = aclr_from_psd(f1, P1, ch_bw=ch_bw, adj_offset=adj_offset)

    # Print
    print("=== SISO PA + DPD (MP-ILA, swappable PA) ===")
    print(f"PA type: {pa_info.get('pa_type')}")
    if args.pa_type == "mp":
        print(f"  MP source: {pa_info.get('source')}  (Ind={args.pa_index}, rf={args.rf}, indTX={pa_info.get('indTX')})")
        print(f"  MP P={pa_info.get('P')}, M={pa_info.get('M')}, Gain={pa_info.get('Gain'):.4f}")
    elif args.pa_type == "gmp":
        print(f"  GMP P={pa_info.get('P')}, M={pa_info.get('M')}, L_lag={pa_info.get('L_lag')}, L_lead={pa_info.get('L_lead')}")
    elif args.pa_type == "wiener":
        print(f"  Wiener P={pa_info.get('P')}, fir_len={pa_info.get('fir_len')}")
    elif args.pa_type == "nn":
        print(
            "  NN trained PA: "
            f"teacher={pa_info.get('nn_teacher')} ({pa_info.get('teacher_source')}), "
            f"mem={pa_info.get('nn_mem')}, hidden={pa_info.get('nn_hidden')}, depth={pa_info.get('nn_depth')}, "
            f"epochs={pa_info.get('nn_epochs')}, device={pa_info.get('nn_device')}, train_samples={pa_info.get('train_samples')}"
        )

    print(f"Fs={Fs_in/1e6:.2f} Msps, backoff={args.backoff_db:.2f} dB")
    print(f"DPD: P={P_dpd}, M={M_dpd.tolist()}, ridge={args.ridge:g}, clip={args.dpd_clip_rms}*RMS")
    print("")
    print(f"EVM (no DPD):   {100*evm0:.3f} %")
    print(f"EVM (with DPD): {100*evm1:.3f} %")
    print("")
    print(f"ACLR worst (no DPD):   {aclr0['ACLR_worst_dB']:.2f} dB")
    print(f"ACLR worst (with DPD): {aclr1['ACLR_worst_dB']:.2f} dB")

    if args.plot:
        plt.figure()
        plt.plot(f0 / 1e6, 10 * np.log10(np.maximum(P0, 1e-20)), label="PA out (no DPD)")
        plt.plot(f1 / 1e6, 10 * np.log10(np.maximum(P1, 1e-20)), label="PA out (with DPD)")
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("PSD [dB/Hz]")
        plt.title("PSD (Welch)")
        plt.grid(True)
        plt.legend()

        plt.figure()
        plt.plot(sym_hat.real, sym_hat.imag, ".", alpha=0.3, label="no DPD")
        plt.plot(sym_hat_dpd.real, sym_hat_dpd.imag, ".", alpha=0.3, label="with DPD")
        plt.xlabel("I")
        plt.ylabel("Q")
        plt.title("RX Constellation (SISO)")
        plt.grid(True)
        plt.axis("equal")
        plt.legend()

        plt.show()


if __name__ == "__main__":
    main()
