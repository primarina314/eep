
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid MU-MIMO Transmitter Simulation (QAM + Rayleigh, Hybrid Precoding + PA/DPD)

What this script does
---------------------
- Multi-user downlink (K users, 1 stream per user), narrowband/flat-fading Rayleigh channel
- Hybrid precoding:
    * Analog beamforming FRF (phase-shifters, constant modulus)
    * Digital precoding FBB (ZF / RZF on the effective channel)
- Per-antenna PA nonlinearity using the Lund massive MIMO testbed memory-polynomial models
  (same PA dataset used in the MATLAB code you uploaded)
- Optional DPD (Memory Polynomial, ILA) with two modes:
    * per_antenna : one DPD per antenna branch (upper bound / best-case)
    * per_rf      : one DPD per RF chain using a simple "combined feedback" observation
                    (approximation of low-complexity hybrid-MIMO DPD ideas)

Waveform
--------
- Square QAM + root-raised-cosine (RRC) pulse shaping (single-carrier)
- Oversampling factor osf

Metrics
-------
- RX EVM per user (after matched filter + downsample, best scalar equalization)
- TX PSD and ACLR (Welch PSD; ACLR via band integration around main and adjacent channels)

Notes
-----
1) The PA models are behavioral polynomials identified for a limited input range.
   Aggressive DPD can push the input outside that range and cause unrealistic spikes.
   Therefore an optional magnitude clipper is provided for the predistorted samples.

2) The Rayleigh channel here is narrowband i.i.d. CN(0,1) (flat fading).
   OFDM and frequency-selective channels can be added later by replacing the waveform/channel blocks.

Usage examples
--------------
# Default (hybrid precoding, PA only)
python hybrid_mimo_dpd_sim.py

# Enable per-antenna DPD and show plots
python hybrid_mimo_dpd_sim.py --dpd per_antenna --plot

# Change antennas/users/RF-chains, SNR, QAM order
python hybrid_mimo_dpd_sim.py --nt 32 --k 8 --nrf 8 --snr_db 25 --qam_order 256 --plot

# Quantized phase shifters (e.g., 6-bit)
python hybrid_mimo_dpd_sim.py --phase_bits 6

By (re)using your uploaded dataset:
- PA_memory_models.mat
- P_TX_Case1.mat

This script will auto-load them if present in the current directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
from numpy.random import default_rng
from scipy import io, signal
import matplotlib.pyplot as plt


# =========================
# Pulse shaping (RRC)
# =========================
def rrc_taps(alpha: float, span: int, sps: int) -> np.ndarray:
    """
    Root Raised Cosine (RRC) FIR taps (unit-energy normalization).

    Parameters
    ----------
    alpha : roll-off (0<alpha<=1)
    span  : filter span in symbols (MATLAB rcosdesign uses 'span')
    sps   : samples per symbol

    Returns
    -------
    h : real ndarray, length span*sps + 1
    """
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

    # unit energy
    h = h / np.sqrt(np.sum(h**2))
    return h


def pulse_shape(sym: np.ndarray, over: int, alpha: float, D: int) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    MATLAB pulse_shape.m equivalent for RRC pulse shaping.

    Returns
    -------
    s : complex ndarray of length len(sym)*over
    d : index of maximum tap (0-index)
    rrc : taps
    """
    sym = np.asarray(sym).reshape(-1)
    sym_over = np.zeros(len(sym) * over, dtype=complex)
    sym_over[::over] = sym

    rrc = rrc_taps(alpha=alpha, span=2 * D, sps=over)
    d = int(np.argmax(rrc))

    s_full = np.convolve(rrc, sym_over)
    s = s_full[d : d + len(sym) * over]
    return s, d, rrc


def rx_match_filter_and_sample(wave: np.ndarray, rrc: np.ndarray, over: int, d: int, Nsym: int) -> np.ndarray:
    """
    Matched filter with the same RRC pulse and downsample to symbol rate.
    """
    wave = np.asarray(wave).reshape(-1)
    mf_full = np.convolve(rrc, wave)
    mf = mf_full[d : d + Nsym * over]
    sym_hat = mf[::over]
    return sym_hat


# =========================
# QAM
# =========================
def square_qam_alphabet(M: int, normalize_avg_power: bool = True) -> np.ndarray:
    """
    MATLAB-style square QAM alphabet:
      tmp = -(sqrt(M)-1):2:(sqrt(M)-1)
      alphabet = kron(tmp,ones(1,sqrt(M))) + 1i*kron(ones(1,sqrt(M)),tmp)

    If normalize_avg_power=True, the alphabet is normalized to unit average power.
    """
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
# Memory polynomial PA / DPD
# =========================
def mp_regressor(x: np.ndarray, P: int, M: np.ndarray) -> np.ndarray:
    """
    Regressor for a memory polynomial:
      y[n] = Σ_{p=0..K-1} Σ_{m=0..M[p]-1} c[p,m] * x[n-m]*|x[n-m]|^(2p)
    with odd orders 1,3,...,P (K=(P+1)/2).

    Matches your PAmodel.m structure (varying memory length per odd order).
    """
    x = np.asarray(x).reshape(-1)
    N = len(x)
    if P % 2 == 0:
        raise ValueError("P must be odd.")
    K = (P + 1) // 2

    M = np.asarray(M).astype(int).reshape(-1)
    if len(M) != K:
        raise ValueError(f"M must have length {K}, got {len(M)}")

    Phi = np.zeros((N, int(np.sum(M))), dtype=complex)
    col = 0
    for p_idx in range(K):
        basis = x * (np.abs(x) ** (2 * p_idx))  # x |x|^{2p_idx}
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
    """
    (Ridge) LS fit of MP coefficients c for mapping: input_signal -> target_signal
      min_c ||Phi(input)c - target||^2 + ridge||c||^2
    """
    x = np.asarray(input_signal).reshape(-1)
    y = np.asarray(target_signal).reshape(-1)
    Phi = mp_regressor(x, P, M)
    A = Phi.conj().T @ Phi
    b = Phi.conj().T @ y
    if ridge > 0:
        A = A + ridge * np.eye(A.shape[0], dtype=A.dtype)
    c = np.linalg.solve(A, b)
    return c


def train_dpd_ila(x: np.ndarray, y: np.ndarray, P_dpd: int, M_dpd: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """
    Indirect Learning Architecture (ILA):
      fit postdistorter: x ≈ g(y)  -> copy g as predistorter.
    """
    return fit_mp(input_signal=y, target_signal=x, P=P_dpd, M=M_dpd, ridge=ridge)


def clip_complex(sig: np.ndarray, peak: float) -> np.ndarray:
    """
    Magnitude clipper with phase preserved:
      if |x| > peak -> peak*exp(j*angle(x))
    """
    sig = np.asarray(sig)
    if peak <= 0:
        return sig
    mag = np.abs(sig)
    scale = np.minimum(1.0, peak / (mag + 1e-12))
    return sig * scale


@dataclass
class PAModelParam:
    Fs: float
    P: int
    Lpa: np.ndarray
    coeff: np.ndarray
    Gain: float = 1.0


def load_pa_dataset(pa_memory_models_mat: str, p_tx_mat: Optional[str], gains: str) -> Tuple[List[PAModelParam], float]:
    """
    Load the 100 PA models (parameters) and apply either equal or measured gains.

    MATLAB reference:
      - PA_memory_models.mat -> 'parameters'
      - P_TX_Case1.mat       -> 'P_tx' used to compute normalized amplitude gains
    """
    data = io.loadmat(pa_memory_models_mat)
    params_arr = data["parameters"]
    Fs = float(params_arr[0, 0]["Fs"][0, 0].squeeze())

    # gains
    if gains == "equal":
        gain_vec = np.ones(100, dtype=float)
    else:
        if p_tx_mat is None:
            raise ValueError("p_tx_mat must be provided when gains='measured'.")
        ptx = io.loadmat(p_tx_mat)["P_tx"].reshape(-1)
        gain_vec = np.sqrt(10.0 ** (ptx / 10.0))
        gain_vec = gain_vec / np.mean(gain_vec)

    models: List[PAModelParam] = []
    for i in range(100):
        p = params_arr[i, 0]
        pa = p["pa"][0, 0]
        P = int(pa["P"][0, 0].squeeze())
        Lpa = np.array(pa["Lpa"][0, 0]).squeeze().astype(int)
        coeff = np.array(pa["coeff"][0, 0]).squeeze()
        models.append(PAModelParam(Fs=Fs, P=P, Lpa=Lpa, coeff=coeff, Gain=float(gain_vec[i])))
    return models, Fs


def pa_model(x: np.ndarray, param: PAModelParam, backoff_db: float = 0.0) -> np.ndarray:
    """
    Python equivalent of PAmodel.m (Lund testbed MP model).
    """
    x = np.asarray(x).reshape(-1)

    Pin = np.mean(np.abs(x) ** 2)
    scale_input = 1.0 / np.sqrt((10.0 ** (backoff_db / 10.0)) * Pin) if Pin > 0 else 1.0
    x_scaled = scale_input * x

    y = apply_mp(x_scaled, param.coeff, param.P, param.Lpa)
    y = param.Gain * y

    # scale back to original power
    y = (1.0 / scale_input) * y
    return y


# =========================
# Hybrid precoding (TX)
# =========================
def quantize_phase(x: np.ndarray, bits: int, magnitude: Optional[float] = None) -> np.ndarray:
    """
    Quantize complex phases to B-bit phase shifters.

    Parameters
    ----------
    x : complex ndarray
        Input matrix/vector whose phases are quantized.
    bits : int
        Number of quantization bits (0 disables quantization).
    magnitude : float or None
        If given, output has constant magnitude = magnitude (typical for phase shifters).
        If None, preserves |x|.

    Returns
    -------
    y : complex ndarray
        Phase-quantized output.
    """
    if bits <= 0:
        return x
    levels = 2 ** bits
    step = 2 * np.pi / levels
    phi_q = step * np.round(np.angle(x) / step)
    mag = np.abs(x) if magnitude is None else float(magnitude)
    return mag * np.exp(1j * phi_q)


def zf_precoder(H: np.ndarray, reg: float = 0.0) -> np.ndarray:
    """
    ZF/RZF precoder for MU-MIMO downlink:
      F = H^H (H H^H + reg*I)^{-1}
    H: K x Nt  (single-antenna users)
    F: Nt x K
    """
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
    """
    Design (FRF, FBB) for hybrid precoding.

    H: K x Nt
    FRF: Nt x n_rf (constant-modulus phase shifters)
    FBB: n_rf x K  (digital baseband precoder)

    Methods
    -------
    - svd_phase_zf: FRF from phase of right singular vectors of H, FBB by ZF on Heff
    - random_zf:    random FRF phases, FBB by ZF on Heff
    - fopt_altmin:  approximate full-digital ZF Fopt with a simple AltMin decomposition
    """
    K, Nt = H.shape
    if n_rf < K:
        raise ValueError(f"Need n_rf >= K (streams). Got n_rf={n_rf}, K={K}.")

    if method == "random_zf":
        rng = default_rng(0)
        FRF = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(Nt, n_rf))) / np.sqrt(Nt)
        if phase_bits > 0:
            FRF = quantize_phase(FRF, phase_bits, magnitude=1/np.sqrt(Nt))
        Heff = H @ FRF
        FBB = zf_precoder(Heff, reg=rzf_reg).T  # zf_precoder returns n_rf x K? Wait Heff is K x n_rf -> returns n_rf x K then .T? let's do carefully below
    else:
        # SVD-based analog beamformer
        U, S, Vh = np.linalg.svd(H, full_matrices=True)
        V = Vh.conj().T  # Nt x Nt
        FRF = np.exp(1j * np.angle(V[:, :n_rf])) / np.sqrt(Nt)
        if phase_bits > 0:
            FRF = quantize_phase(FRF, phase_bits, magnitude=1/np.sqrt(Nt))

    if method in ("svd_phase_zf", "random_zf"):
        Heff = H @ FRF  # K x n_rf
        # Digital ZF / RZF on effective channel:
        # FBB (n_rf x K) = Heff^H (Heff Heff^H + reg I)^{-1}
        HH = Heff @ Heff.conj().T
        if rzf_reg > 0:
            HH = HH + rzf_reg * np.eye(K, dtype=HH.dtype)
        FBB = Heff.conj().T @ np.linalg.inv(HH)  # n_rf x K

    elif method == "fopt_altmin":
        # Full-digital ZF target
        Fopt = zf_precoder(H, reg=rzf_reg)  # Nt x K

        # Initialize FRF from the phase of the first n_rf columns of V (SVD of H) for stability
        U, S, Vh = np.linalg.svd(H, full_matrices=True)
        V = Vh.conj().T
        FRF = np.exp(1j * np.angle(V[:, :n_rf])) / np.sqrt(Nt)
        if phase_bits > 0:
            FRF = quantize_phase(FRF, phase_bits, magnitude=1/np.sqrt(Nt))

        # Simple alternating minimization (heuristic)
        n_iter = 20
        for _ in range(n_iter):
            # LS digital: FBB = pinv(FRF) Fopt
            FBB = np.linalg.pinv(FRF) @ Fopt  # n_rf x K

            # Update FRF phases to match Fopt FBB^H
            T = Fopt @ FBB.conj().T  # Nt x n_rf
            FRF = np.exp(1j * np.angle(T)) / np.sqrt(Nt)
            if phase_bits > 0:
                FRF = quantize_phase(FRF, phase_bits, magnitude=1/np.sqrt(Nt))

        # final FBB
        FBB = np.linalg.pinv(FRF) @ Fopt

    else:
        raise ValueError(f"Unknown method: {method}")

    # Power normalization: scale FBB so that ||FRF FBB||_F^2 equals either K or 1.
    F = FRF @ FBB
    fro2 = float(np.linalg.norm(F, "fro") ** 2)
    if fro2 == 0:
        raise RuntimeError("Precoder has zero norm.")

    if power_norm.upper() == "K":
        target = float(K)
    elif power_norm.upper() == "1":
        target = 1.0
    else:
        raise ValueError("power_norm must be 'K' or '1'.")
    scale = np.sqrt(target / fro2)
    FBB = scale * FBB
    return FRF, FBB


# =========================
# Metrics
# =========================
def estimate_scalar_ls(ref: np.ndarray, meas: np.ndarray) -> complex:
    """
    Best scalar g minimizing ||meas - g*ref||_2.
    """
    ref = np.asarray(ref).reshape(-1)
    meas = np.asarray(meas).reshape(-1)
    denom = np.vdot(ref, ref)
    if denom == 0:
        return 0.0 + 0.0j
    return np.vdot(ref, meas) / denom


def evm_rms(ref: np.ndarray, meas: np.ndarray, discard: int = 0) -> Tuple[float, complex]:
    """
    RMS EVM after best complex scalar equalization.
    Returns (evm_linear, g).
    """
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
    """
    Welch PSD for complex baseband, returns fftshift-sorted f and Pxx.
    """
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
    """
    ACLR from PSD integration.

    main: [-ch_bw/2, +ch_bw/2]
    upper: [adj_offset - ch_bw/2, adj_offset + ch_bw/2]
    lower: [-adj_offset - ch_bw/2, -adj_offset + ch_bw/2]
    """
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
# Simulation blocks
# =========================
def rayleigh_channel(K: int, Nt: int, seed: int = 0) -> np.ndarray:
    """
    Flat-fading Rayleigh MU-MIMO channel, i.i.d. CN(0,1).
    Returns H of shape (K, Nt).
    """
    rng = default_rng(seed)
    H = (rng.standard_normal((K, Nt)) + 1j * rng.standard_normal((K, Nt))) / np.sqrt(2.0)
    return H


def apply_pa_array(Xant: np.ndarray, pa_params: List[PAModelParam], backoff_db: float) -> np.ndarray:
    """
    Apply PA model per antenna.

    Xant: Nsamp x Nt
    Returns: Yant Nsamp x Nt
    """
    Ns, Nt = Xant.shape
    if len(pa_params) != Nt:
        raise ValueError("pa_params length must equal Nt.")

    Y = np.zeros_like(Xant, dtype=complex)
    for i in range(Nt):
        Y[:, i] = pa_model(Xant[:, i], pa_params[i], backoff_db=backoff_db)
    return Y


def average_psd_over_antennas(Yant: np.ndarray, fs: float, nperseg: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Welch PSD per antenna and average (power-average) across antennas.
    """
    Ns, Nt = Yant.shape
    f_ref = None
    P_acc = None
    for i in range(Nt):
        f, P = welch_psd(Yant[:, i], fs=fs, nperseg=nperseg)
        if f_ref is None:
            f_ref = f
            P_acc = np.zeros_like(P)
        P_acc += P
    P_mean = P_acc / Nt
    return f_ref, P_mean


def simulate(
    Nt: int,
    K: int,
    n_rf: int,
    qam_order: int,
    Nsym: int,
    osf: int,
    rolloff: float,
    rrc_delay: int,
    snr_db: float,
    precoder_method: str,
    phase_bits: int,
    rzf_reg: float,
    backoff_db: float,
    dpd_mode: str,
    dpd_order: Optional[int],
    ridge: float,
    dpd_clip_rms: float,
    pa_models: Optional[List[PAModelParam]],
    pa_assignment: str,
    pa_index: int,
    rf: int,
    gains: str,
    seed: int,
) -> Dict[str, object]:
    """
    Run one Monte-Carlo drop (one channel realization + one waveform).
    Returns a dict with waveforms, metrics, and precoders.
    """
    rng = default_rng(seed)

    # -------------------------
    # Channel
    # -------------------------
    H = rayleigh_channel(K=K, Nt=Nt, seed=seed + 10)

    # -------------------------
    # Precoder
    # -------------------------
    FRF, FBB = design_hybrid_precoder(
        H=H,
        n_rf=n_rf,
        method=precoder_method,
        phase_bits=phase_bits,
        rzf_reg=rzf_reg,
        power_norm="K",
    )

    # -------------------------
    # Multi-user waveform generation
    # -------------------------
    syms = np.zeros((Nsym, K), dtype=complex)
    for k in range(K):
        syms[:, k] = generate_qam_symbols(Nsym, qam_order, seed=seed + 100 + k, normalize_avg_power=True)

    # pulse shape each stream
    # (use same rrc across users)
    S = np.zeros((Nsym * osf, K), dtype=complex)
    d = None
    rrc = None
    for k in range(K):
        s_k, d_k, rrc_k = pulse_shape(syms[:, k], over=osf, alpha=rolloff, D=rrc_delay)
        S[:, k] = s_k
        if d is None:
            d, rrc = d_k, rrc_k

    # -------------------------
    # Apply hybrid precoder (linear)
    # -------------------------
    # Xrf: Nsamp x n_rf
    Xrf = S @ FBB.T
    # Xant: Nsamp x Nt
    Xant = Xrf @ FRF.T

    # -------------------------
    # Build PA parameter list for antennas
    # -------------------------
    if pa_models is None:
        # Synthetic PA models (fallback)
        # A mild MP nonlinearity; same for all antennas
        Fs = 120e6
        P = 9
        Lpa = np.array([1, 2, 2, 2, 2], dtype=int)
        # linear + weak 3rd + 5th, with small memory terms
        c = np.zeros(np.sum(Lpa), dtype=complex)
        c[0] = 1.0 + 0j
        c[1] = 0.08 - 0.02j
        c[3] = 0.02 + 0.01j
        pa_params = [PAModelParam(Fs=Fs, P=P, Lpa=Lpa, coeff=c, Gain=1.0) for _ in range(Nt)]
        Fs_in = Fs
    else:
        # Use the provided 100 measured PA models, assign them across Nt antennas
        Fs_in = float(pa_models[0].Fs)
        if pa_assignment == "same":
            indTX = 2 * pa_index - 1 + rf
            indTX = int(np.clip(indTX, 1, 100))
            chosen = pa_models[indTX - 1]
            pa_params = [chosen for _ in range(Nt)]
        elif pa_assignment == "sequential":
            pa_params = [pa_models[i % len(pa_models)] for i in range(Nt)]
        elif pa_assignment == "random":
            idxs = rng.integers(0, len(pa_models), size=Nt)
            pa_params = [pa_models[int(i)] for i in idxs]
        else:
            raise ValueError("pa_assignment must be one of {'same','sequential','random'}")

    # -------------------------
    # PA only
    # -------------------------
    Y_pa = apply_pa_array(Xant, pa_params, backoff_db=backoff_db)

    # -------------------------
    # DPD (optional)
    # -------------------------
    if dpd_mode == "none":
        Uant = Xant
        Y_dpd = Y_pa
        dpd_coeff = None

    elif dpd_mode == "per_antenna":
        # Train one postdistorter per antenna: x_ant_i ≈ g_i(y_pa_i)
        Uant = np.zeros_like(Xant)
        dpd_coeff = []
        for i in range(Nt):
            Ppa = pa_params[i].P
            Mpa = pa_params[i].Lpa
            P_dpd = dpd_order if dpd_order is not None else Ppa
            # memory lengths: if order differs, simple pad/truncate
            Kd = (P_dpd + 1) // 2
            if Kd <= len(Mpa):
                M_dpd = Mpa[:Kd]
            else:
                M_dpd = np.concatenate([Mpa, np.ones(Kd - len(Mpa), dtype=int)])
            c_i = train_dpd_ila(Xant[:, i], Y_pa[:, i], P_dpd=P_dpd, M_dpd=M_dpd, ridge=ridge)
            u_i = apply_mp(Xant[:, i], c_i, P=P_dpd, M=M_dpd)
            if dpd_clip_rms > 0:
                u_rms = float(np.sqrt(np.mean(np.abs(u_i) ** 2)))
                u_i = clip_complex(u_i, peak=dpd_clip_rms * u_rms)
            Uant[:, i] = u_i
            dpd_coeff.append((c_i, P_dpd, M_dpd))

        Y_dpd = apply_pa_array(Uant, pa_params, backoff_db=backoff_db)

    elif dpd_mode == "per_rf":
        # Approximate low-complexity: train one DPD per RF chain using a simple combined feedback:
        #   z_r[n] = Σ_i conj(FRF[i,r]) * y_i[n]
        # Then fit x_rf_r ≈ g_r(z_r) and apply g_r to x_rf_r before FRF.
        Ppa = pa_params[0].P
        Mpa = pa_params[0].Lpa
        P_dpd = dpd_order if dpd_order is not None else Ppa
        Kd = (P_dpd + 1) // 2
        if Kd <= len(Mpa):
            M_dpd = Mpa[:Kd]
        else:
            M_dpd = np.concatenate([Mpa, np.ones(Kd - len(Mpa), dtype=int)])

        # Combined observation per RF chain
        Z = Y_pa @ np.conj(FRF)  # Nsamp x n_rf

        Urf = np.zeros_like(Xrf)
        dpd_coeff = []
        for r in range(n_rf):
            c_r = train_dpd_ila(Xrf[:, r], Z[:, r], P_dpd=P_dpd, M_dpd=M_dpd, ridge=ridge)
            u_r = apply_mp(Xrf[:, r], c_r, P=P_dpd, M=M_dpd)
            if dpd_clip_rms > 0:
                u_rms = float(np.sqrt(np.mean(np.abs(u_r) ** 2)))
                u_r = clip_complex(u_r, peak=dpd_clip_rms * u_rms)
            Urf[:, r] = u_r
            dpd_coeff.append((c_r, P_dpd, M_dpd))

        Uant = Urf @ FRF.T
        Y_dpd = apply_pa_array(Uant, pa_params, backoff_db=backoff_db)

    else:
        raise ValueError("dpd_mode must be one of {'none','per_antenna','per_rf'}")

    # -------------------------
    # RX (multi-user)
    # -------------------------
    # noiseless received waveforms
    R_lin = Xant @ H.T
    R_pa = Y_pa @ H.T
    R_dpd = Y_dpd @ H.T

    # Add AWGN at the receivers.
    # Here snr_db is interpreted at *symbol rate* after the matched filter + sampling
    # (more intuitive than sample-rate SNR when osf>1).
    # We estimate the symbol-rate power from the *noiseless* linear received waveform.
    symhat_lin0 = np.zeros((Nsym, K), dtype=complex)
    for k in range(K):
        symhat_lin0[:, k] = rx_match_filter_and_sample(R_lin[:, k], rrc, over=osf, d=d, Nsym=Nsym)
    sigpow_sym = float(np.mean(np.abs(symhat_lin0) ** 2))
    sigma2 = sigpow_sym / (10.0 ** (snr_db / 10.0)) if snr_db is not None else 0.0
    noise = (rng.standard_normal(R_lin.shape) + 1j * rng.standard_normal(R_lin.shape)) * np.sqrt(sigma2 / 2.0)

    R_lin_n = R_lin + noise
    R_pa_n = R_pa + noise
    R_dpd_n = R_dpd + noise

    # Matched filter + symbol sampling per user
    symhat_lin = np.zeros((Nsym, K), dtype=complex)
    symhat_pa = np.zeros((Nsym, K), dtype=complex)
    symhat_dpd = np.zeros((Nsym, K), dtype=complex)
    for k in range(K):
        symhat_lin[:, k] = rx_match_filter_and_sample(R_lin_n[:, k], rrc, over=osf, d=d, Nsym=Nsym)
        symhat_pa[:, k] = rx_match_filter_and_sample(R_pa_n[:, k], rrc, over=osf, d=d, Nsym=Nsym)
        symhat_dpd[:, k] = rx_match_filter_and_sample(R_dpd_n[:, k], rrc, over=osf, d=d, Nsym=Nsym)

    # EVM per user
    discard = rrc_delay  # discard a few symbols at edges
    evm_lin = np.zeros(K)
    evm_pa = np.zeros(K)
    evm_dpd = np.zeros(K)
    for k in range(K):
        evm_lin[k], _ = evm_rms(syms[:, k], symhat_lin[:, k], discard=discard)
        evm_pa[k], _ = evm_rms(syms[:, k], symhat_pa[:, k], discard=discard)
        evm_dpd[k], _ = evm_rms(syms[:, k], symhat_dpd[:, k], discard=discard)

    # -------------------------
    # TX PSD / ACLR (average over antennas)
    # -------------------------
    Rs = Fs_in / osf
    ch_bw = (1.0 + rolloff) * Rs
    adj_offset = ch_bw

    f_pa, P_pa = average_psd_over_antennas(Y_pa, fs=Fs_in, nperseg=2048)
    aclr_pa = aclr_from_psd(f_pa, P_pa, ch_bw=ch_bw, adj_offset=adj_offset)

    f_dpd, P_dpd = average_psd_over_antennas(Y_dpd, fs=Fs_in, nperseg=2048)
    aclr_dpd = aclr_from_psd(f_dpd, P_dpd, ch_bw=ch_bw, adj_offset=adj_offset)

    # -------------------------
    # Return
    # -------------------------
    out: Dict[str, object] = {
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
        "dpd_coeff": dpd_coeff,
    }
    return out


# =========================
# CLI / main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nt", type=int, default=16, help="Number of TX antennas (Nt)")
    ap.add_argument("--k", type=int, default=4, help="Number of users/streams (K)")
    ap.add_argument("--nrf", type=int, default=4, help="Number of RF chains (n_rf), must be >= K")
    ap.add_argument("--qam_order", type=int, default=256)
    ap.add_argument("--Nsym", type=int, default=8000)
    ap.add_argument("--osf", type=int, default=8)
    ap.add_argument("--rolloff", type=float, default=0.22)
    ap.add_argument("--rrc_delay", type=int, default=12)
    ap.add_argument("--snr_db", type=float, default=25.0)
    ap.add_argument("--precoder", type=str, default="svd_phase_zf", choices=["svd_phase_zf", "random_zf", "fopt_altmin"])
    ap.add_argument("--phase_bits", type=int, default=0, help="Quantize phase shifters to B bits (0 disables)")
    ap.add_argument("--rzf_reg", type=float, default=0.0, help="RZF regularization (0 -> ZF)")
    ap.add_argument("--backoff_db", type=float, default=0.0)

    ap.add_argument("--dpd", type=str, default="none", choices=["none", "per_antenna", "per_rf"])
    ap.add_argument("--dpd_order", type=int, default=None, help="Odd order for DPD (default: PA order)")
    ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument("--dpd_clip_rms", type=float, default=3.0, help="Clip predistorted samples to (clip*rms). <=0 disables")

    ap.add_argument("--pa_memory_models_mat", type=str, default="PA_memory_models.mat")
    ap.add_argument("--p_tx_mat", type=str, default="P_TX_Case1.mat")
    ap.add_argument("--gains", type=str, default="measured", choices=["equal", "measured"])
    ap.add_argument("--pa_assignment", type=str, default="random", choices=["same", "sequential", "random"])
    ap.add_argument("--pa_index", type=int, default=1, help="USRP index Ind (1..50) if pa_assignment='same'")
    ap.add_argument("--rf", type=int, default=0, choices=[0, 1], help="0=TX0(odd), 1=TX1(even)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    # Load PA dataset if files exist, otherwise fallback to synthetic model.
    # We try:
    #   1) the provided path as-is
    #   2) relative to the script location
    #   3) (optional) current working directory (already covered by #1 for relative paths)
    from pathlib import Path

    pa_models = None
    Fs = 120e6

    def _resolve_candidate(path_str: str) -> Optional[str]:
        p = Path(path_str)
        if p.exists():
            return str(p)
        # try relative to this script
        try:
            here = Path(__file__).resolve().parent
            p2 = here / path_str
            if p2.exists():
                return str(p2)
        except Exception:
            pass
        return None

    try:
        pa_path = _resolve_candidate(args.pa_memory_models_mat)
        ptx_path = _resolve_candidate(args.p_tx_mat) if args.gains == "measured" else None

        if pa_path is not None:
            if args.gains == "measured" and ptx_path is None:
                raise FileNotFoundError(f"'{args.p_tx_mat}' not found (needed for measured gains).")
            pa_models, Fs = load_pa_dataset(pa_path, ptx_path, gains=args.gains)
        else:
            print(f"[WARN] '{args.pa_memory_models_mat}' not found -> using synthetic PA models.")
    except Exception as e:
        print(f"[WARN] Failed to load PA dataset ({e}) -> using synthetic PA models.")
        pa_models = None

    res = simulate(
        Nt=args.nt,
        K=args.k,
        n_rf=args.nrf,
        qam_order=args.qam_order,
        Nsym=args.Nsym,
        osf=args.osf,
        rolloff=args.rolloff,
        rrc_delay=args.rrc_delay,
        snr_db=args.snr_db,
        precoder_method=args.precoder,
        phase_bits=args.phase_bits,
        rzf_reg=args.rzf_reg,
        backoff_db=args.backoff_db,
        dpd_mode=args.dpd,
        dpd_order=args.dpd_order,
        ridge=args.ridge,
        dpd_clip_rms=args.dpd_clip_rms,
        pa_models=pa_models,
        pa_assignment=args.pa_assignment,
        pa_index=args.pa_index,
        rf=args.rf,
        gains=args.gains,
        seed=args.seed,
    )

    evm_lin = res["evm_lin"]
    evm_pa = res["evm_pa"]
    evm_dpd = res["evm_dpd"]

    aclr_pa = res["aclr_pa"]
    aclr_dpd = res["aclr_dpd"]

    print("=== Hybrid MU-MIMO TX Simulation (QAM + Rayleigh) ===")
    print(f"Nt={args.nt}, K={args.k}, nRF={args.nrf}, precoder={args.precoder}, phase_bits={args.phase_bits}")
    print(f"Waveform: {args.qam_order}-QAM, Nsym={args.Nsym}, osf={args.osf}, rolloff={args.rolloff}, Fs={res['Fs']/1e6:.1f} Msps")
    print(f"Channel: Rayleigh flat fading, SNR={args.snr_db:.1f} dB")
    print(f"PA: backoff={args.backoff_db:.2f} dB, gains={args.gains}, PA assignment={args.pa_assignment}")
    print(f"DPD mode: {args.dpd} (order={args.dpd_order}, ridge={args.ridge:g}, clip={args.dpd_clip_rms}*RMS)")
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
    print(f"TX ACLR (avg PSD over antennas):")
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

        # Constellation for a user (pick user 1)
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