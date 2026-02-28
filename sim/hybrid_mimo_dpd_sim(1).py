#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hybrid_mimo_dpd_sim.py

Hybrid MU-MIMO downlink transmitter simulation with **swappable PA models**.

System
------
- Multi-user downlink: K single-antenna users, 1 stream/user
- Channel: Rayleigh flat fading (i.i.d. CN(0,1))
- Hybrid precoding: analog phase-shifter beamformer FRF + digital baseband precoder FBB
- Waveform: RRC-shaped square QAM (single-carrier), oversampling osf

PA / DPD
--------
- PA model (selectable via --pa_type):
    * mp     : Memory Polynomial (measured Lund dataset if .mat available, else synthetic)
    * gmp    : Generalized Memory Polynomial (synthetic)
    * wiener : Wiener (LTI + memoryless poly) (synthetic)
    * nn     : NeuralNetPA (PyTorch) trained to mimic a teacher model

- DPD: Memory-Polynomial predistorter trained with ILA (least squares + ridge)
    * none
    * per_antenna
    * per_rf (simple combined feedback)

Metrics
-------
- RX EVM per user (matched filter + downsample + best scalar EQ)
- TX PSD and ACLR (Welch PSD averaged over antennas; ACLR by band integration)

Examples
--------
# MP PA (measured if .mat exists)
python hybrid_mimo_dpd_sim.py --pa_type mp --dpd per_antenna --plot

# GMP PA
python hybrid_mimo_dpd_sim.py --pa_type gmp --gmp_P 7 --gmp_M 3 --gmp_L_lag 1 --gmp_L_lead 1 --plot

# Wiener PA
python hybrid_mimo_dpd_sim.py --pa_type wiener --wiener_P 7 --wiener_fir_len 5 --plot

# NN PA (train NN to mimic teacher)
python hybrid_mimo_dpd_sim.py --pa_type nn --nn_teacher auto --nn_epochs 5 --plot

Notes
-----
- If the measured .mat files are missing, the script falls back to a synthetic PA.
- NN-based PA requires torch.
- DPD is kept as MP-ILA for now (PA model is swappable; DPD model can be extended later).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List

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


# =========================
# Pulse shaping (RRC)
# =========================

def rrc_taps(alpha: float, span: int, sps: int) -> np.ndarray:
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1].")
    if span <= 0 or sps <= 0:
        raise ValueError("span and sps must be positive.")

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
    sym_over = np.zeros(len(sym) * over, dtype=complex)
    sym_over[::over] = sym

    rrc = rrc_taps(alpha=alpha, span=2 * D, sps=over)
    d = int(np.argmax(rrc))

    s_full = np.convolve(rrc, sym_over)
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
        raise ValueError("M must be a perfect square (e.g., 16, 64, 256).")
    tmp = np.arange(-(m - 1), m, 2)
    alphabet = np.kron(tmp, np.ones(m)) + 1j * np.kron(np.ones(m), tmp)
    alphabet = alphabet.astype(complex)

    if normalize_avg_power:
        p = np.mean(np.abs(alphabet) ** 2)
        alphabet = alphabet / np.sqrt(p)
    return alphabet


def generate_qam_symbols(Nsym: int, M: int, seed: int, normalize_avg_power: bool = True) -> np.ndarray:
    rng = default_rng(seed)
    alphabet = square_qam_alphabet(M, normalize_avg_power=normalize_avg_power)
    idx = rng.integers(low=0, high=len(alphabet), size=Nsym)
    return alphabet[idx]


# =========================
# Metrics
# =========================

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


def welch_psd(x: np.ndarray, fs: float, nperseg: int = 2048, beta: float = 7.0) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x).reshape(-1)
    f, Pxx = signal.welch(
        x,
        fs=fs,
        window=("kaiser", beta),
        nperseg=nperseg,
        noverlap=nperseg // 2,
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


# =========================
# Hybrid precoding (TX)
# =========================

def quantize_phase(x: np.ndarray, bits: int, magnitude: Optional[float] = None) -> np.ndarray:
    if bits <= 0:
        return x
    levels = 2**bits
    step = 2 * np.pi / levels
    phi_q = step * np.round(np.angle(x) / step)
    mag = np.abs(x) if magnitude is None else float(magnitude)
    return mag * np.exp(1j * phi_q)


def zf_precoder(H: np.ndarray, reg: float = 0.0) -> np.ndarray:
    K = H.shape[0]
    HH = H @ H.conj().T
    if reg > 0:
        HH = HH + reg * np.eye(K, dtype=HH.dtype)
    F = H.conj().T @ np.linalg.inv(HH)
    return F


def design_hybrid_precoder(
    H: np.ndarray,
    n_rf: int,
    method: str = "svd_phase_zf",
    phase_bits: int = 0,
    rzf_reg: float = 0.0,
    power_norm: str = "K",
) -> Tuple[np.ndarray, np.ndarray]:
    """Design hybrid precoder (FRF, FBB)."""
    K, Nt = H.shape
    if n_rf < K:
        raise ValueError(f"Need n_rf >= K (streams). Got n_rf={n_rf}, K={K}.")

    if method == "random_zf":
        rng = default_rng(0)
        FRF = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(Nt, n_rf))) / np.sqrt(Nt)
        if phase_bits > 0:
            FRF = quantize_phase(FRF, phase_bits, magnitude=1 / np.sqrt(Nt))

    elif method in ("svd_phase_zf", "fopt_altmin"):
        U, S, Vh = np.linalg.svd(H, full_matrices=True)
        V = Vh.conj().T
        FRF = np.exp(1j * np.angle(V[:, :n_rf])) / np.sqrt(Nt)
        if phase_bits > 0:
            FRF = quantize_phase(FRF, phase_bits, magnitude=1 / np.sqrt(Nt))

    else:
        raise ValueError(f"Unknown method: {method}")

    if method in ("svd_phase_zf", "random_zf"):
        Heff = H @ FRF
        HH = Heff @ Heff.conj().T
        if rzf_reg > 0:
            HH = HH + rzf_reg * np.eye(K, dtype=HH.dtype)
        FBB = Heff.conj().T @ np.linalg.inv(HH)  # n_rf x K

    elif method == "fopt_altmin":
        Fopt = zf_precoder(H, reg=rzf_reg)  # Nt x K
        n_iter = 20
        for _ in range(n_iter):
            FBB = np.linalg.pinv(FRF) @ Fopt
            T = Fopt @ FBB.conj().T
            FRF = np.exp(1j * np.angle(T)) / np.sqrt(Nt)
            if phase_bits > 0:
                FRF = quantize_phase(FRF, phase_bits, magnitude=1 / np.sqrt(Nt))
        FBB = np.linalg.pinv(FRF) @ Fopt

    # power normalization
    F = FRF @ FBB
    fro2 = float(np.linalg.norm(F, "fro") ** 2)
    if fro2 == 0:
        raise RuntimeError("Precoder has zero norm.")

    target = float(K) if power_norm.upper() == "K" else 1.0
    scale = np.sqrt(target / fro2)
    FBB = scale * FBB
    return FRF, FBB


# =========================
# Channel
# =========================

def rayleigh_channel(K: int, Nt: int, seed: int = 0) -> np.ndarray:
    rng = default_rng(seed)
    H = (rng.standard_normal((K, Nt)) + 1j * rng.standard_normal((K, Nt))) / np.sqrt(2.0)
    return H


# =========================
# PA builders
# =========================

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


def build_pa_list(
    args,
    Xant: np.ndarray,
    mp_models: Optional[list],
    Fs_hint: float,
    seed: int,
) -> Tuple[List[object], float, Dict[str, object], Optional[int], Optional[np.ndarray]]:
    """Return (pa_list, Fs_in, pa_info, pa_order, pa_mem_vec)."""
    rng = default_rng(seed)

    pa_type = args.pa_type.lower()
    Fs_in = float(Fs_hint)

    pa_info: Dict[str, object] = {"pa_type": pa_type}

    # mapping used by MATLAB scripts
    indTX = int(2 * args.pa_index - 1 + args.rf)
    indTX = int(np.clip(indTX, 1, 100))

    Nt = Xant.shape[1]

    def assign_params_from_pool(pool: list, assignment: str) -> List:
        if assignment == "same":
            return [pool[indTX - 1] for _ in range(Nt)]
        if assignment == "sequential":
            return [pool[i % len(pool)] for i in range(Nt)]
        if assignment == "random":
            idxs = rng.integers(0, len(pool), size=Nt)
            return [pool[int(i)] for i in idxs]
        raise ValueError("pa_assignment must be one of {same,sequential,random}")

    if pa_type == "mp":
        if mp_models is not None:
            Fs_in = float(mp_models[0].Fs)
            params = assign_params_from_pool(mp_models, args.pa_assignment)
            pa_list = [MemoryPolynomialPA(p, backoff_db=args.backoff_db) for p in params]
            pa_order = int(params[0].P)
            pa_mem = np.asarray(params[0].M).astype(int)
            pa_info.update({
                "source": "measured_mat" if _resolve_existing(args.pa_memory_models_mat) else "measured_loaded",
                "Gain_mode": args.gains,
                "pa_assignment": args.pa_assignment,
                "P": pa_order,
                "M": pa_mem.tolist(),
            })
            return pa_list, Fs_in, pa_info, pa_order, pa_mem

        # synthetic MP
        Fs_in = float(args.fs)
        # Create a pool to mimic assignment semantics
        pool = [make_synthetic_mp(Fs=Fs_in, P=int(args.mp_P), mem_other_orders=int(args.mp_mem), strength=args.pa_strength, seed=seed + i) for i in range(100)]
        params = assign_params_from_pool(pool, args.pa_assignment)
        pa_list = [MemoryPolynomialPA(p, backoff_db=args.backoff_db) for p in params]
        pa_order = int(params[0].P)
        pa_mem = np.asarray(params[0].M).astype(int)
        pa_info.update({"source": "synthetic", "pa_assignment": args.pa_assignment, "P": pa_order, "M": pa_mem.tolist()})
        return pa_list, Fs_in, pa_info, pa_order, pa_mem

    if pa_type == "gmp":
        Fs_in = float(args.fs)
        pool = [
            make_synthetic_gmp(
                Fs=Fs_in,
                P=int(args.gmp_P),
                M=int(args.gmp_M),
                L_lag=int(args.gmp_L_lag),
                L_lead=int(args.gmp_L_lead),
                strength=args.pa_strength,
                seed=seed + i,
            )
            for i in range(100)
        ]
        params = assign_params_from_pool(pool, args.pa_assignment)
        pa_list = [GMPPA(p, backoff_db=args.backoff_db) for p in params]
        pa_order = int(params[0].P)
        pa_info.update({
            "source": "synthetic",
            "pa_assignment": args.pa_assignment,
            "P": pa_order,
            "M": int(params[0].M),
            "L_lag": int(params[0].L_lag),
            "L_lead": int(params[0].L_lead),
        })
        return pa_list, Fs_in, pa_info, pa_order, None

    if pa_type == "wiener":
        Fs_in = float(args.fs)
        pool = [make_synthetic_wiener(Fs=Fs_in, P=int(args.wiener_P), fir_len=int(args.wiener_fir_len), strength=args.pa_strength, seed=seed + i) for i in range(100)]
        params = assign_params_from_pool(pool, args.pa_assignment)
        pa_list = [WienerPA(p, backoff_db=args.backoff_db) for p in params]
        pa_order = int(args.wiener_P)
        pa_info.update({"source": "synthetic", "pa_assignment": args.pa_assignment, "P": pa_order, "fir_len": int(len(params[0].h))})
        return pa_list, Fs_in, pa_info, pa_order, None

    if pa_type == "nn":
        Fs_in = float(args.fs) if mp_models is None else float(mp_models[0].Fs)
        teacher = args.nn_teacher.lower()
        if teacher == "auto":
            teacher = "mp" if mp_models is not None else "wiener"

        pa_info.update({"source": "nn_trained", "nn_teacher": teacher, "nn_mode": args.nn_mode})

        # Build teacher PA
        if teacher == "mp":
            if mp_models is not None:
                if args.pa_assignment == "same":
                    tparam = mp_models[indTX - 1]
                else:
                    tparam = mp_models[0]
                Fs_in = float(tparam.Fs)
                teacher_pa = MemoryPolynomialPA(tparam, backoff_db=args.backoff_db)
                pa_info.update({"teacher_source": "measured_mp", "teacher_P": int(tparam.P)})
            else:
                tparam = make_synthetic_mp(Fs=Fs_in, P=int(args.mp_P), mem_other_orders=int(args.mp_mem), strength=args.pa_strength, seed=seed)
                teacher_pa = MemoryPolynomialPA(tparam, backoff_db=args.backoff_db)
                pa_info.update({"teacher_source": "synthetic_mp", "teacher_P": int(tparam.P)})

        elif teacher == "gmp":
            tparam = make_synthetic_gmp(
                Fs=Fs_in,
                P=int(args.gmp_P),
                M=int(args.gmp_M),
                L_lag=int(args.gmp_L_lag),
                L_lead=int(args.gmp_L_lead),
                strength=args.pa_strength,
                seed=seed,
            )
            teacher_pa = GMPPA(tparam, backoff_db=args.backoff_db)
            pa_info.update({"teacher_source": "synthetic_gmp", "teacher_P": int(tparam.P)})

        elif teacher == "wiener":
            tparam = make_synthetic_wiener(Fs=Fs_in, P=int(args.wiener_P), fir_len=int(args.wiener_fir_len), strength=args.pa_strength, seed=seed)
            teacher_pa = WienerPA(tparam, backoff_db=args.backoff_db)
            pa_info.update({"teacher_source": "synthetic_wiener", "teacher_P": int(args.wiener_P)})

        else:
            raise ValueError("nn_teacher must be one of {auto, mp, gmp, wiener}")

        # training subset
        N = Xant.shape[0]
        n_train_syms = int(min(max(200, args.nn_train_syms), args.Nsym))
        n_train = int(min(N, n_train_syms * args.osf))

        # resolve device
        device = args.nn_device
        try:
            import torch

            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
        except Exception:
            device = "cpu"

        # torch seed
        try:
            import torch

            torch.manual_seed(int(seed))
        except Exception:
            pass

        def train_one_nn(x_train: np.ndarray, seed_offset: int = 0) -> NeuralNetPA:
            try:
                import torch

                torch.manual_seed(int(seed + seed_offset))
            except Exception:
                pass

            y_train = teacher_pa.forward(x_train)
            nn_pa = NeuralNetPA(
                M=int(args.nn_mem),
                hidden=int(args.nn_hidden),
                depth=int(args.nn_depth),
                backoff_db=float(args.backoff_db),
                device=device,
            )
            nn_pa.fit(x_train, y_train, lr=float(args.nn_lr), epochs=int(args.nn_epochs), batch=int(args.nn_batch))
            return nn_pa

        if args.nn_mode == "shared":
            x_train = Xant[:n_train, 0]
            nn_pa = train_one_nn(x_train)
            pa_list = [nn_pa for _ in range(Nt)]

        elif args.nn_mode == "per_antenna":
            pa_list = []
            for i in range(Nt):
                x_train = Xant[:n_train, i]
                pa_list.append(train_one_nn(x_train, seed_offset=i + 1))

        else:
            raise ValueError("nn_mode must be one of {shared, per_antenna}")

        pa_info.update({
            "nn_mem": int(args.nn_mem),
            "nn_hidden": int(args.nn_hidden),
            "nn_depth": int(args.nn_depth),
            "nn_epochs": int(args.nn_epochs),
            "nn_device": device,
            "train_samples": int(n_train),
        })

        return pa_list, Fs_in, pa_info, None, None

    raise ValueError("pa_type must be one of {mp,gmp,wiener,nn}")


# =========================
# Simulation blocks
# =========================

def apply_pa_array(Xant: np.ndarray, pa_list: List[object]) -> np.ndarray:
    Ns, Nt = Xant.shape
    if len(pa_list) != Nt:
        raise ValueError("pa_list length must equal Nt")

    Y = np.zeros_like(Xant, dtype=complex)
    for i in range(Nt):
        Y[:, i] = pa_list[i].forward(Xant[:, i])
    return Y


def average_psd_over_antennas(Yant: np.ndarray, fs: float, nperseg: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    Ns, Nt = Yant.shape
    f_ref = None
    P_acc = None
    for i in range(Nt):
        f, P = welch_psd(Yant[:, i], fs=fs, nperseg=nperseg)
        if f_ref is None:
            f_ref = f
            P_acc = np.zeros_like(P)
        P_acc += P
    return f_ref, P_acc / Nt


def simulate(args, mp_models: Optional[list], seed: int) -> Dict[str, object]:
    rng = default_rng(seed)

    # -------------------------
    # Channel
    # -------------------------
    H = rayleigh_channel(K=args.k, Nt=args.nt, seed=seed + 10)

    # -------------------------
    # Precoder
    # -------------------------
    FRF, FBB = design_hybrid_precoder(
        H=H,
        n_rf=args.nrf,
        method=args.precoder,
        phase_bits=args.phase_bits,
        rzf_reg=args.rzf_reg,
        power_norm="K",
    )

    # -------------------------
    # Multi-user waveform generation
    # -------------------------
    syms = np.zeros((args.Nsym, args.k), dtype=complex)
    for k in range(args.k):
        syms[:, k] = generate_qam_symbols(args.Nsym, args.qam_order, seed=seed + 100 + k, normalize_avg_power=True)

    S = np.zeros((args.Nsym * args.osf, args.k), dtype=complex)
    d = None
    rrc = None
    for k in range(args.k):
        s_k, d_k, rrc_k = pulse_shape(syms[:, k], over=args.osf, alpha=args.rolloff, D=args.rrc_delay)
        S[:, k] = s_k
        if d is None:
            d, rrc = d_k, rrc_k

    # -------------------------
    # Apply hybrid precoder (linear)
    # -------------------------
    Xrf = S @ FBB.T           # Nsamp x n_rf
    Xant = Xrf @ FRF.T        # Nsamp x Nt

    # -------------------------
    # Build PA list (swappable)
    # -------------------------
    Fs_hint = float(args.fs)
    if mp_models is not None:
        Fs_hint = float(mp_models[0].Fs)

    pa_list, Fs_in, pa_info, pa_order, pa_mem_vec = build_pa_list(args, Xant=Xant, mp_models=mp_models, Fs_hint=Fs_hint, seed=seed)

    # -------------------------
    # PA only
    # -------------------------
    Y_pa = apply_pa_array(Xant, pa_list)

    # -------------------------
    # DPD (optional)
    # -------------------------
    if args.dpd == "none":
        Uant = Xant
        Y_dpd = Y_pa
        dpd_coeff = None

    else:
        # Choose DPD order
        if args.dpd_order is not None:
            P_dpd = int(args.dpd_order)
        elif pa_order is not None:
            P_dpd = int(pa_order)
        else:
            P_dpd = 9

        if P_dpd % 2 == 0:
            raise ValueError("dpd_order must be odd")
        Kd = (P_dpd + 1) // 2

        # Choose DPD memory vector
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

        if args.dpd == "per_antenna":
            Uant = np.zeros_like(Xant)
            dpd_coeff = []
            for i in range(args.nt):
                c_i = fit_mp(input_signal=Y_pa[:, i], target_signal=Xant[:, i], P=P_dpd, M=M_dpd, ridge=args.ridge)
                u_i = apply_mp(Xant[:, i], c_i, P=P_dpd, M=M_dpd)
                if args.dpd_clip_rms > 0:
                    u_rms = float(np.sqrt(np.mean(np.abs(u_i) ** 2)))
                    u_i = clip_complex(u_i, peak=args.dpd_clip_rms * u_rms)
                Uant[:, i] = u_i
                dpd_coeff.append((c_i, P_dpd, M_dpd))

            Y_dpd = apply_pa_array(Uant, pa_list)

        elif args.dpd == "per_rf":
            # Combined observation per RF chain
            Z = Y_pa @ np.conj(FRF)  # Nsamp x n_rf

            Urf = np.zeros_like(Xrf)
            dpd_coeff = []
            for r in range(args.nrf):
                c_r = fit_mp(input_signal=Z[:, r], target_signal=Xrf[:, r], P=P_dpd, M=M_dpd, ridge=args.ridge)
                u_r = apply_mp(Xrf[:, r], c_r, P=P_dpd, M=M_dpd)
                if args.dpd_clip_rms > 0:
                    u_rms = float(np.sqrt(np.mean(np.abs(u_r) ** 2)))
                    u_r = clip_complex(u_r, peak=args.dpd_clip_rms * u_rms)
                Urf[:, r] = u_r
                dpd_coeff.append((c_r, P_dpd, M_dpd))

            Uant = Urf @ FRF.T
            Y_dpd = apply_pa_array(Uant, pa_list)

        else:
            raise ValueError("dpd must be one of {none, per_antenna, per_rf}")

    # -------------------------
    # RX (multi-user)
    # -------------------------
    R_lin = Xant @ H.T
    R_pa = Y_pa @ H.T
    R_dpd = Y_dpd @ H.T

    # AWGN at receivers (interpreted at symbol rate after MF+sample)
    symhat_lin0 = np.zeros((args.Nsym, args.k), dtype=complex)
    for k in range(args.k):
        symhat_lin0[:, k] = rx_match_filter_and_sample(R_lin[:, k], rrc, over=args.osf, d=d, Nsym=args.Nsym)
    sigpow_sym = float(np.mean(np.abs(symhat_lin0) ** 2))
    sigma2 = sigpow_sym / (10.0 ** (args.snr_db / 10.0))

    noise = (rng.standard_normal(R_lin.shape) + 1j * rng.standard_normal(R_lin.shape)) * np.sqrt(sigma2 / 2.0)

    R_lin_n = R_lin + noise
    R_pa_n = R_pa + noise
    R_dpd_n = R_dpd + noise

    symhat_lin = np.zeros((args.Nsym, args.k), dtype=complex)
    symhat_pa = np.zeros((args.Nsym, args.k), dtype=complex)
    symhat_dpd = np.zeros((args.Nsym, args.k), dtype=complex)
    for k in range(args.k):
        symhat_lin[:, k] = rx_match_filter_and_sample(R_lin_n[:, k], rrc, over=args.osf, d=d, Nsym=args.Nsym)
        symhat_pa[:, k] = rx_match_filter_and_sample(R_pa_n[:, k], rrc, over=args.osf, d=d, Nsym=args.Nsym)
        symhat_dpd[:, k] = rx_match_filter_and_sample(R_dpd_n[:, k], rrc, over=args.osf, d=d, Nsym=args.Nsym)

    discard = args.rrc_delay
    evm_lin = np.zeros(args.k)
    evm_pa = np.zeros(args.k)
    evm_dpd = np.zeros(args.k)
    for k in range(args.k):
        evm_lin[k], _ = evm_rms(syms[:, k], symhat_lin[:, k], discard=discard)
        evm_pa[k], _ = evm_rms(syms[:, k], symhat_pa[:, k], discard=discard)
        evm_dpd[k], _ = evm_rms(syms[:, k], symhat_dpd[:, k], discard=discard)

    # -------------------------
    # TX PSD / ACLR (average over antennas)
    # -------------------------
    Rs = Fs_in / args.osf
    ch_bw = (1.0 + args.rolloff) * Rs
    adj_offset = ch_bw

    f_pa, P_pa = average_psd_over_antennas(Y_pa, fs=Fs_in, nperseg=2048)
    aclr_pa = aclr_from_psd(f_pa, P_pa, ch_bw=ch_bw, adj_offset=adj_offset)

    f_dpd, P_dpd = average_psd_over_antennas(Y_dpd, fs=Fs_in, nperseg=2048)
    aclr_dpd = aclr_from_psd(f_dpd, P_dpd, ch_bw=ch_bw, adj_offset=adj_offset)

    return {
        "H": H,
        "FRF": FRF,
        "FBB": FBB,
        "syms": syms,
        "S": S,
        "Xrf": Xrf,
        "Xant": Xant,
        "Uant": Uant,
        "Y_pa": Y_pa,
        "Y_dpd": Y_dpd,
        "R_lin": R_lin_n,
        "R_pa": R_pa_n,
        "R_dpd": R_dpd_n,
        "symhat_lin": symhat_lin,
        "symhat_pa": symhat_pa,
        "symhat_dpd": symhat_dpd,
        "evm_lin": evm_lin,
        "evm_pa": evm_pa,
        "evm_dpd": evm_dpd,
        "aclr_pa": aclr_pa,
        "aclr_dpd": aclr_dpd,
        "psd_pa": (f_pa, P_pa),
        "psd_dpd": (f_dpd, P_dpd),
        "Fs": Fs_in,
        "sigma2": sigma2,
        "pa_info": pa_info,
    }


# =========================
# CLI / main
# =========================

def main() -> None:
    ap = argparse.ArgumentParser()

    # system sizes
    ap.add_argument("--nt", type=int, default=16, help="Number of TX antennas (Nt)")
    ap.add_argument("--k", type=int, default=4, help="Number of users/streams (K)")
    ap.add_argument("--nrf", type=int, default=4, help="Number of RF chains (nRF), must be >= K")

    # waveform
    ap.add_argument("--qam_order", type=int, default=256)
    ap.add_argument("--Nsym", type=int, default=8000)
    ap.add_argument("--osf", type=int, default=8)
    ap.add_argument("--rolloff", type=float, default=0.22)
    ap.add_argument("--rrc_delay", type=int, default=12)

    # channel
    ap.add_argument("--snr_db", type=float, default=25.0)

    # hybrid precoder
    ap.add_argument("--precoder", type=str, default="svd_phase_zf", choices=["svd_phase_zf", "random_zf", "fopt_altmin"])
    ap.add_argument("--phase_bits", type=int, default=0)
    ap.add_argument("--rzf_reg", type=float, default=0.0)

    # PA selection
    ap.add_argument("--pa_type", type=str, default="mp", choices=["mp", "gmp", "wiener", "nn"])
    ap.add_argument("--fs", type=float, default=120e6, help="Sample rate for synthetic PA types")
    ap.add_argument("--pa_strength", type=float, default=1.0)
    ap.add_argument("--backoff_db", type=float, default=0.0)

    # measured MP dataset paths (used when pa_type=mp, and optionally as NN teacher)
    ap.add_argument("--pa_memory_models_mat", type=str, default="PA_memory_models.mat")
    ap.add_argument("--p_tx_mat", type=str, default="P_TX_Case1.mat")
    ap.add_argument("--gains", type=str, default="measured", choices=["equal", "measured"])

    # assignment (applies to measured pool OR synthetic pool)
    ap.add_argument("--pa_assignment", type=str, default="random", choices=["same", "sequential", "random"])
    ap.add_argument("--pa_index", type=int, default=1, help="USRP index Ind (1..50) if pa_assignment='same'")
    ap.add_argument("--rf", type=int, default=0, choices=[0, 1])

    # synthetic MP args
    ap.add_argument("--mp_P", type=int, default=9)
    ap.add_argument("--mp_mem", type=int, default=2)

    # synthetic GMP args
    ap.add_argument("--gmp_P", type=int, default=7)
    ap.add_argument("--gmp_M", type=int, default=3)
    ap.add_argument("--gmp_L_lag", type=int, default=1)
    ap.add_argument("--gmp_L_lead", type=int, default=1)

    # synthetic Wiener args
    ap.add_argument("--wiener_P", type=int, default=7)
    ap.add_argument("--wiener_fir_len", type=int, default=5)

    # NN args
    ap.add_argument("--nn_teacher", type=str, default="auto", choices=["auto", "mp", "gmp", "wiener"])
    ap.add_argument("--nn_mode", type=str, default="shared", choices=["shared", "per_antenna"])
    ap.add_argument("--nn_mem", type=int, default=7)
    ap.add_argument("--nn_hidden", type=int, default=128)
    ap.add_argument("--nn_depth", type=int, default=3)
    ap.add_argument("--nn_epochs", type=int, default=5)
    ap.add_argument("--nn_lr", type=float, default=1e-3)
    ap.add_argument("--nn_batch", type=int, default=4096)
    ap.add_argument("--nn_device", type=str, default="cpu")
    ap.add_argument("--nn_train_syms", type=int, default=2000)

    # DPD
    ap.add_argument("--dpd", type=str, default="none", choices=["none", "per_antenna", "per_rf"])
    ap.add_argument("--dpd_order", type=int, default=None)
    ap.add_argument("--dpd_mem", type=int, default=-1, help="DPD MP memory length (>=1). -1 inherits PA memory if available.")
    ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument("--dpd_clip_rms", type=float, default=3.0)

    # misc
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--plot", action="store_true")

    args = ap.parse_args()

    # Basic checks
    if args.nrf < args.k:
        raise SystemExit("[ERROR] Need nrf >= k (streams).")

    # Load measured MP pool if possible (used when pa_type=mp or nn_teacher=mp)
    mp_models, Fs_meas = try_load_measured_mp_models(args.pa_memory_models_mat, args.p_tx_mat, gains=args.gains)
    if args.pa_type == "mp" and mp_models is None:
        print(f"[WARN] Measured MP dataset not found/loaded. Falling back to synthetic MP.")

    if args.pa_type == "nn" and args.nn_teacher in ("auto", "mp") and mp_models is None and args.nn_teacher == "mp":
        print(f"[WARN] nn_teacher='mp' requested but measured MP dataset not found -> using synthetic MP teacher.")

    res = simulate(args, mp_models=mp_models, seed=args.seed)

    evm_lin = res["evm_lin"]
    evm_pa = res["evm_pa"]
    evm_dpd = res["evm_dpd"]

    aclr_pa = res["aclr_pa"]
    aclr_dpd = res["aclr_dpd"]

    pa_info = res["pa_info"]

    print("=== Hybrid MU-MIMO TX Simulation (QAM + Rayleigh, swappable PA) ===")
    print(f"Nt={args.nt}, K={args.k}, nRF={args.nrf}, precoder={args.precoder}, phase_bits={args.phase_bits}")
    print(f"Waveform: {args.qam_order}-QAM, Nsym={args.Nsym}, osf={args.osf}, rolloff={args.rolloff}, Fs={res['Fs']/1e6:.1f} Msps")
    print(f"Channel: Rayleigh flat fading, SNR={args.snr_db:.1f} dB")
    print(f"PA: type={args.pa_type}, backoff={args.backoff_db:.2f} dB, assignment={args.pa_assignment}, strength={args.pa_strength}")

    if args.pa_type == "mp":
        print(f"  MP source={pa_info.get('source')} gains={pa_info.get('Gain_mode')} P={pa_info.get('P')} M={pa_info.get('M')}")
    elif args.pa_type == "gmp":
        print(f"  GMP P={pa_info.get('P')} M={pa_info.get('M')} L_lag={pa_info.get('L_lag')} L_lead={pa_info.get('L_lead')}")
    elif args.pa_type == "wiener":
        print(f"  Wiener P={pa_info.get('P')} fir_len={pa_info.get('fir_len')}")
    elif args.pa_type == "nn":
        print(
            "  NN trained PA: "
            f"teacher={pa_info.get('nn_teacher')} ({pa_info.get('teacher_source')}), "
            f"mode={pa_info.get('nn_mode')}, mem={pa_info.get('nn_mem')}, hidden={pa_info.get('nn_hidden')}, depth={pa_info.get('nn_depth')}, "
            f"epochs={pa_info.get('nn_epochs')}, device={pa_info.get('nn_device')}, train_samples={pa_info.get('train_samples')}"
        )

    print(f"DPD: mode={args.dpd}, order={args.dpd_order}, dpd_mem={args.dpd_mem}, ridge={args.ridge:g}, clip={args.dpd_clip_rms}*RMS")
    print("")

    for k in range(args.k):
        print(
            f"User {k+1:02d} EVM: "
            f"Linear={100*evm_lin[k]:6.3f}% | "
            f"PA={100*evm_pa[k]:6.3f}% | "
            f"DPD={100*evm_dpd[k]:6.3f}%"
        )

    print("")
    print(f"Avg EVM: Linear={100*np.mean(evm_lin):.3f}% | PA={100*np.mean(evm_pa):.3f}% | DPD={100*np.mean(evm_dpd):.3f}%")
    print("")
    print("TX ACLR (avg PSD over antennas):")
    print(f"  PA only:  worst={aclr_pa['ACLR_worst_dB']:.2f} dB (U={aclr_pa['ACLR_u_dB']:.2f}, L={aclr_pa['ACLR_l_dB']:.2f})")
    print(f"  With DPD: worst={aclr_dpd['ACLR_worst_dB']:.2f} dB (U={aclr_dpd['ACLR_u_dB']:.2f}, L={aclr_dpd['ACLR_l_dB']:.2f})")

    if args.plot:
        f_pa, P_pa = res["psd_pa"]
        f_d, P_d = res["psd_dpd"]

        plt.figure()
        plt.plot(f_pa / 1e6, 10 * np.log10(np.maximum(P_pa, 1e-20)), label="PA only (avg)")
        plt.plot(f_d / 1e6, 10 * np.log10(np.maximum(P_d, 1e-20)), label="DPD+PA (avg)")
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("PSD [dB/Hz]")
        plt.title("TX PSD (Welch, averaged over antennas)")
        plt.grid(True)
        plt.legend()

        # Constellation for user 1
        k0 = 0
        plt.figure()
        plt.plot(res["symhat_pa"][:, k0].real, res["symhat_pa"][:, k0].imag, ".", alpha=0.3, label="PA")
        plt.plot(res["symhat_dpd"][:, k0].real, res["symhat_dpd"][:, k0].imag, ".", alpha=0.3, label="DPD")
        plt.xlabel("I")
        plt.ylabel("Q")
        plt.title(f"RX Constellation (User {k0+1})")
        plt.grid(True)
        plt.axis("equal")
        plt.legend()

        plt.show()


if __name__ == "__main__":
    main()
