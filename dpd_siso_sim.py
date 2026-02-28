#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SISO PA + DPD simulation (Memory Polynomial, Indirect Learning Architecture)

This code is written to closely match the MATLAB reference files from:
"Digital Predistortion for Hybrid MIMO Transmitters"
  - PAmodel.m
  - pulse_shape.m
  - PA_memory_models.mat / P_TX_Case1.mat

Main features
-------------
- Generates RRC-shaped square QAM (MATLAB-style alphabet)
- PA model: Memory Polynomial PA extracted from Lund massive MIMO testbed
- DPD: Memory Polynomial predistorter trained with ILA (least-squares + ridge)
- Metrics: EVM (symbol-domain), ACLR (PSD integration), PSD plot, AM/AM plot

Important practical note
------------------------
The provided PA models are polynomial behavioral models identified for a limited
input amplitude range. DPD can increase peaks (PAPR) and may push the input
outside the model validity range, causing unrealistic "blow-ups" (huge spikes).
To keep the simulation numerically/physically meaningful, this script includes
an optional peak limiter on the predistorted samples.

Run
---
python dpd_siso_sim.py --pa_memory_models_mat PA_memory_models.mat --p_tx_mat P_TX_Case1.mat

Example (show plots)
--------------------
python dpd_siso_sim.py --plot

By default, it picks:
  pa_index=1, rf=0  -> indTX = 1 (same mapping as MATLAB test_PA_models.m)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
from numpy.random import default_rng
from scipy import signal, io
import matplotlib.pyplot as plt


# -----------------------------
# Root Raised Cosine pulse
# -----------------------------
def rrc_taps(alpha: float, span: int, sps: int) -> np.ndarray:
    """
    Root Raised Cosine (RRC) FIR taps.

    Parameters
    ----------
    alpha : float
        Roll-off (0 < alpha <= 1)
    span : int
        Filter span in symbols (MATLAB rcosdesign uses "span" = 2*D)
    sps : int
        Samples per symbol (oversampling factor)

    Returns
    -------
    h : np.ndarray
        Real-valued RRC filter taps of length span*sps + 1
    """
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1].")

    N = span * sps
    t = np.arange(-N / 2, N / 2 + 1) / sps  # time in symbol periods (T=1)
    h = np.zeros_like(t, dtype=float)

    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            # t = 0 special case
            h[i] = (1.0 + alpha * (4.0 / np.pi - 1.0))
        elif abs(abs(4.0 * alpha * ti) - 1.0) < 1e-8:
            # t = ±1/(4α) special case
            h[i] = (alpha / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
            )
        else:
            num = (
                np.sin(np.pi * ti * (1.0 - alpha))
                + 4.0 * alpha * ti * np.cos(np.pi * ti * (1.0 + alpha))
            )
            den = np.pi * ti * (1.0 - (4.0 * alpha * ti) ** 2)
            h[i] = num / den

    # Match MATLAB-style normalization (unit energy for sqrt pulse is common)
    h = h / np.sqrt(np.sum(h**2))
    return h


def pulse_shape(sym: np.ndarray, over: int, alpha: float, D: int) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    MATLAB pulse_shape.m equivalent:
      sym_over = kron(sym,[1; zeros(over-1,1)])
      rrc = rcosdesign(alpha, D*2, over, 'sqrt')
      [~,d] = max(rrc)
      s = conv(rrc, sym_over)
      s = s(d:d+length(sym)*over-1)

    Returns
    -------
    s : np.ndarray
        Shaped baseband samples of length len(sym)*over
    d : int
        Index of maximum tap (0-indexed)
    rrc : np.ndarray
        RRC taps
    """
    sym = np.asarray(sym).reshape(-1)
    sym_over = np.zeros(len(sym) * over, dtype=complex)
    sym_over[::over] = sym

    rrc = rrc_taps(alpha=alpha, span=2 * D, sps=over)
    d = int(np.argmax(rrc))  # 0-index (MATLAB is 1-index)

    s_full = np.convolve(rrc, sym_over)
    s = s_full[d : d + len(sym) * over]
    return s, d, rrc


# -----------------------------
# Memory polynomial PA / DPD
# -----------------------------
def mp_regressor(x: np.ndarray, P: int, M: np.ndarray) -> np.ndarray:
    """
    Build regressor matrix Phi(x) for memory polynomial:
      y[n] = Σ_{p odd} Σ_{m=0}^{M_p-1} c_{p,m} x[n-m] |x[n-m]|^{p-1}

    This matches PAmodel.m (zero-padding for n-m < 0).
    """
    x = np.asarray(x).reshape(-1)
    N = len(x)
    if P % 2 == 0:
        raise ValueError("P must be odd.")
    K = (P + 1) // 2

    M = np.asarray(M).astype(int).reshape(-1)
    if len(M) != K:
        raise ValueError(f"M must have length {(P + 1) // 2}, got {len(M)}")

    Phi = np.zeros((N, int(np.sum(M))), dtype=complex)
    col = 0
    for p_idx in range(K):
        # odd order = 1,3,5,...,P
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
    """
    Fit MP coefficients c for mapping input_signal -> target_signal by (ridge) least squares:
      min_c ||Phi(input)c - target||^2 + ridge ||c||^2
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


@dataclass
class PAModelParam:
    Fs: float
    P: int
    Lpa: np.ndarray
    coeff: np.ndarray
    Gain: float = 1.0


def load_pa_models(pa_memory_models_mat: str) -> Tuple[np.ndarray, float]:
    """
    Load PA models from PA_memory_models.mat (MATLAB variable: 'parameters').
    """
    data = io.loadmat(pa_memory_models_mat)
    params = data["parameters"]
    Fs = float(params[0, 0]["Fs"][0, 0].squeeze())
    return params, Fs


def matlab_param_to_dataclass(p) -> PAModelParam:
    """
    Convert one MATLAB struct element to PAModelParam.
    """
    Fs = float(p["Fs"][0, 0].squeeze())
    pa = p["pa"][0, 0]
    P = int(pa["P"][0, 0].squeeze())
    Lpa = np.array(pa["Lpa"][0, 0]).squeeze().astype(int)
    coeff = np.array(pa["coeff"][0, 0]).squeeze()
    return PAModelParam(Fs=Fs, P=P, Lpa=Lpa, coeff=coeff, Gain=1.0)


def pa_model(x: np.ndarray, param: PAModelParam, backoff_db: float = 0.0) -> np.ndarray:
    """
    Python equivalent of PAmodel.m.
    """
    x = np.asarray(x).reshape(-1)

    Pin = np.mean(np.abs(x) ** 2)
    scale_input = 1.0 / np.sqrt((10.0 ** (backoff_db / 10.0)) * Pin) if Pin > 0 else 1.0
    x_scaled = scale_input * x

    y = apply_mp(x_scaled, param.coeff, param.P, param.Lpa)
    y = param.Gain * y

    # scale power back to original
    y = (1.0 / scale_input) * y
    return y


def train_dpd_ila(x: np.ndarray, y: np.ndarray, P_dpd: int, M_dpd: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """
    Indirect Learning Architecture (ILA):
    Fit postdistorter: x ≈ g(y). Copy g as predistorter.
    """
    return fit_mp(input_signal=y, target_signal=x, P=P_dpd, M=M_dpd, ridge=ridge)


def scale_to_match_power(sig: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Scale sig so that mean power matches ref.
    (Optional: in many DPD setups you might not want to force this.)
    """
    sig = np.asarray(sig)
    ref = np.asarray(ref)
    p_sig = np.mean(np.abs(sig) ** 2)
    p_ref = np.mean(np.abs(ref) ** 2)
    if p_sig == 0:
        return sig
    return sig * np.sqrt(p_ref / p_sig)


def clip_complex(sig: np.ndarray, peak: float) -> np.ndarray:
    """
    Magnitude clipping with phase preserved:
      if |x| > peak -> x := peak * exp(j*angle(x))
    """
    sig = np.asarray(sig)
    if peak <= 0:
        return sig
    mag = np.abs(sig)
    scale = np.minimum(1.0, peak / (mag + 1e-12))
    return sig * scale


# -----------------------------
# Metrics
# -----------------------------
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


def rx_match_filter_and_sample(wave: np.ndarray, rrc: np.ndarray, over: int, d: int, Nsym: int) -> np.ndarray:
    """
    Matched filtering with the same RRC pulse, then downsample.

    wave is expected to have length Nsym*over (same as pulse_shape output).
    """
    wave = np.asarray(wave).reshape(-1)
    mf_full = np.convolve(rrc, wave)
    mf = mf_full[d : d + Nsym * over]
    sym_hat = mf[::over]
    return sym_hat


def welch_psd(x: np.ndarray, fs: float, nperseg: int = 2048, beta: float = 7.0, noverlap: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Welch PSD for complex baseband, returns shifted frequency axis.
    """
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
    idx = np.argsort(f)  # sort (fftshift-equivalent)
    return f[idx], Pxx[idx]


def integrate_band(f: np.ndarray, Pxx: np.ndarray, f1: float, f2: float) -> float:
    mask = (f >= f1) & (f <= f2)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(Pxx[mask], f[mask]))


def aclr_from_psd(f: np.ndarray, Pxx: np.ndarray, ch_bw: float, adj_offset: float) -> Dict[str, float]:
    """
    ACLR = 10log10(P_main/P_adj) for upper/lower adjacent.
    main: [-ch_bw/2, ch_bw/2]
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


# -----------------------------
# Signal generation
# -----------------------------
def square_qam_alphabet(M: int) -> np.ndarray:
    """
    MATLAB-style square QAM alphabet (unnormalized):
      tmp = -(sqrt(M)-1):2:(sqrt(M)-1)
      alphabet = kron(tmp,ones(1,sqrt(M))) + 1i*kron(ones(1,sqrt(M)),tmp)
    """
    m = int(np.sqrt(M))
    if m * m != M:
        raise ValueError("M must be a perfect square (e.g., 16, 64, 256).")
    tmp = np.arange(-(m - 1), m, 2)
    alphabet = np.kron(tmp, np.ones(m)) + 1j * np.kron(np.ones(m), tmp)
    return alphabet.astype(complex)


def generate_qam_symbols(Nsym: int, M: int, seed: int = 0) -> np.ndarray:
    rng = default_rng(seed)
    alphabet = square_qam_alphabet(M)
    idx = rng.integers(low=0, high=len(alphabet), size=Nsym)
    return alphabet[idx]


# -----------------------------
# Main demo
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pa_memory_models_mat", type=str, default="PA_memory_models.mat")
    ap.add_argument("--p_tx_mat", type=str, default="P_TX_Case1.mat")
    ap.add_argument("--gains", type=str, choices=["equal", "measured"], default="measured")
    ap.add_argument("--pa_index", type=int, default=1, help="USRP index Ind in MATLAB (1..50)")
    ap.add_argument("--rf", type=int, default=0, choices=[0, 1], help="0=TX0 (odd), 1=TX1 (even)")
    ap.add_argument("--backoff_db", type=float, default=0.0)
    ap.add_argument("--Nsym", type=int, default=30000)
    ap.add_argument("--osf", type=int, default=8)
    ap.add_argument("--qam_order", type=int, default=256)
    ap.add_argument("--rolloff", type=float, default=0.22)
    ap.add_argument("--rrc_delay", type=int, default=12)
    ap.add_argument("--dpd_order", type=int, default=None, help="DPD odd order (default: PA order)")
    ap.add_argument("--ridge", type=float, default=1e-6, help="Ridge regularization for LS")
    ap.add_argument("--discard_syms", type=int, default=12, help="Symbols to discard at edges for EVM")
    ap.add_argument(
        "--dpd_clip_rms",
        type=float,
        default=3.0,
        help="Clip predistorted samples to (dpd_clip_rms * RMS). Set <=0 to disable.",
    )
    ap.add_argument(
        "--match_input_power",
        action="store_true",
        help="Optionally scale predistorted waveform to have same mean power as the original x.",
    )
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    # Load PA models
    params_arr, Fs = load_pa_models(args.pa_memory_models_mat)

    indTX = 2 * args.pa_index - 1 + args.rf  # MATLAB mapping
    if indTX < 1 or indTX > 100:
        raise ValueError("indTX out of range. Choose pa_index in 1..50 and rf in {0,1}.")
    param = matlab_param_to_dataclass(params_arr[indTX - 1, 0])

    # Gains
    if args.gains == "equal":
        param.Gain = 1.0
    else:
        ptx = io.loadmat(args.p_tx_mat)["P_tx"].reshape(-1)
        Gain_tx = np.sqrt(10.0 ** (ptx / 10.0))
        Gain_tx = Gain_tx / np.mean(Gain_tx)
        param.Gain = float(Gain_tx[indTX - 1])

    # Signal
    sym = generate_qam_symbols(args.Nsym, args.qam_order, seed=1)
    x, d, rrc = pulse_shape(sym, over=args.osf, alpha=args.rolloff, D=args.rrc_delay)
    Fs_in = Fs  # 120e6 in the provided models

    # PA output without DPD
    y = pa_model(x, param, backoff_db=args.backoff_db)

    # Train DPD (ILA) using the same waveform (you may split train/test if desired)
    P_dpd = args.dpd_order if args.dpd_order is not None else param.P
    M_dpd = param.Lpa.copy()
    if P_dpd != param.P:
        # simple heuristic: keep same memory length for each odd order, truncate/extend as needed
        K = (P_dpd + 1) // 2
        if K <= len(M_dpd):
            M_dpd = M_dpd[:K]
        else:
            M_dpd = np.concatenate([M_dpd, np.ones(K - len(M_dpd), dtype=int)])

    c_dpd = train_dpd_ila(x=x, y=y, P_dpd=P_dpd, M_dpd=M_dpd, ridge=args.ridge)

    # Apply DPD
    u = apply_mp(x, c_dpd, P_dpd, M_dpd)

    # Optional peak limiter (recommended for polynomial PA models)
    if args.dpd_clip_rms > 0:
        u_rms = float(np.sqrt(np.mean(np.abs(u) ** 2)))
        u = clip_complex(u, peak=args.dpd_clip_rms * u_rms)

    # Optional mean-power matching (not always desired)
    if args.match_input_power:
        u = scale_to_match_power(u, x)

    y_dpd = pa_model(u, param, backoff_db=args.backoff_db)

    # EVM (symbol-domain)
    sym_hat = rx_match_filter_and_sample(y, rrc, over=args.osf, d=d, Nsym=args.Nsym)
    sym_hat_dpd = rx_match_filter_and_sample(y_dpd, rrc, over=args.osf, d=d, Nsym=args.Nsym)
    evm0, _ = evm_rms(sym, sym_hat, discard=args.discard_syms)
    evm1, _ = evm_rms(sym, sym_hat_dpd, discard=args.discard_syms)

    # PSD + ACLR
    f0, P0 = welch_psd(y, Fs_in)
    f1, P1 = welch_psd(y_dpd, Fs_in)

    Rs = Fs_in / args.osf
    # For RRC-shaped single-carrier QAM, approximate occupied BW ≈ (1+alpha)*Rs
    ch_bw = (1.0 + args.rolloff) * Rs
    adj_offset = ch_bw  # adjacent channel spaced by one channel bandwidth

    aclr0 = aclr_from_psd(f0, P0, ch_bw=ch_bw, adj_offset=adj_offset)
    aclr1 = aclr_from_psd(f1, P1, ch_bw=ch_bw, adj_offset=adj_offset)

    print("=== SISO PA + DPD (MP-ILA) ===")
    print(f"PA index (Ind) = {args.pa_index}, rf = {args.rf} -> indTX = {indTX}")
    print(f"Backoff = {args.backoff_db:.2f} dB, Gains = {args.gains}, Gain = {param.Gain:.4f}")
    print(f"PA MP:  P = {param.P}, Lpa = {param.Lpa.tolist()}, #coeff = {len(param.coeff)}")
    print(f"DPD MP: P = {P_dpd}, M = {M_dpd.tolist()}, #coeff = {len(c_dpd)}")
    print(f"DPD clip: {args.dpd_clip_rms} * RMS (<=0 disables)")
    print("")
    print(f"EVM (no DPD):   {100*evm0:.3f} %")
    print(f"EVM (with DPD): {100*evm1:.3f} %")
    print("")
    print(f"ACLR worst (no DPD):   {aclr0['ACLR_worst_dB']:.2f} dB")
    print(f"ACLR worst (with DPD): {aclr1['ACLR_worst_dB']:.2f} dB")

    if args.plot:
        # PSD plot
        plt.figure()
        plt.plot(f0 / 1e6, 10 * np.log10(np.maximum(P0, 1e-20)), label="PA out (no DPD)")
        plt.plot(f1 / 1e6, 10 * np.log10(np.maximum(P1, 1e-20)), label="PA out (with DPD)")
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("PSD [dB/Hz]")
        plt.title("PSD (Welch)")
        plt.grid(True)
        plt.legend()

        # AM/AM plot (scatter)
        plt.figure()
        plt.plot(np.abs(x), np.abs(y), ".", markersize=1, alpha=0.25, label="no DPD")
        plt.plot(np.abs(x), np.abs(y_dpd), ".", markersize=1, alpha=0.25, label="with DPD")
        plt.xlabel("|x|")
        plt.ylabel("|y|")
        plt.title("AM/AM")
        plt.grid(True)
        plt.legend()

        plt.show()


if __name__ == "__main__":
    main()
