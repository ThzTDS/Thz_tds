"""
thz_tds.spectral
================
Canonical FFT helpers for THz-TDS signals.

Every other module that needs a Fourier transform imports from here so there
is exactly one implementation throughout the package.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def fft_field(
    time_ps: np.ndarray,
    y: np.ndarray,
    pad_factor: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the one-sided complex FFT of a THz electric-field trace.

    Parameters
    ----------
    time_ps : np.ndarray, shape (N,)
        Uniformly-sampled time axis in picoseconds.
    y : np.ndarray, shape (N,)
        Electric-field amplitude samples.
    pad_factor : int
        Zero-padding factor.  ``pad_factor=1`` means no padding (default,
        matching the usage in :mod:`thz_tds.refractive`).  Use larger values
        (e.g. 4) for a smoother spectral display.

    Returns
    -------
    freq_Hz : np.ndarray, shape (M,)
        Non-negative frequency axis in Hz.  DC (0 Hz) is included at index 0.
    E_complex : np.ndarray, shape (M,)
        Complex electric-field spectrum (output of ``numpy.fft.rfft``).
    """
    time_ps = np.asarray(time_ps, dtype=float)
    y = np.asarray(y, dtype=float)

    dt_s = float(np.mean(np.diff(time_ps))) * 1e-12
    n_pad = len(y) * int(pad_factor)

    E_complex = np.fft.rfft(y, n=n_pad)
    freq_Hz = np.fft.rfftfreq(n_pad, d=dt_s)

    return freq_Hz, E_complex


def power_spectrum_dB(
    time_ps: np.ndarray,
    y: np.ndarray,
    pad_factor: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the normalised power spectrum in dB relative to its peak.

    Parameters
    ----------
    time_ps : np.ndarray, shape (N,)
        Time axis in picoseconds.
    y : np.ndarray, shape (N,)
        Electric-field amplitude samples.
    pad_factor : int
        Zero-padding factor (default 4 for a smooth spectral display).

    Returns
    -------
    freq_THz : np.ndarray
        Non-negative frequency axis in THz.
    power_dB : np.ndarray
        ``10 * log10(|E|² / max(|E|²))``, clipped to avoid ``-inf``.
    """
    freq_Hz, E = fft_field(time_ps, y, pad_factor=pad_factor)

    power = np.abs(E) ** 2
    pmax = max(float(np.max(power)), 1e-30)
    power_dB = 10.0 * np.log10(np.maximum(power, 1e-30) / pmax)

    return freq_Hz / 1e12, power_dB


def ensure_common_time_axis(
    t_ref: np.ndarray,
    y_ref: np.ndarray,
    t_sam: np.ndarray,
    y_sam: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return a common time axis and both signals interpolated onto it.

    The reference time axis is used as the common grid.  If the sample has
    a different time axis it is re-interpolated (with zero-padding at the
    edges) onto the reference grid.

    Parameters
    ----------
    t_ref : np.ndarray
        Reference time axis in picoseconds.
    y_ref : np.ndarray
        Reference electric-field samples.
    t_sam : np.ndarray
        Sample time axis in picoseconds.
    y_sam : np.ndarray
        Sample electric-field samples.

    Returns
    -------
    t_common : np.ndarray
        Common time axis (equals ``t_ref``).
    y_ref_out : np.ndarray
        Reference signal on the common axis (unchanged).
    y_sam_out : np.ndarray
        Sample signal interpolated onto the common axis.
    """
    t_ref = np.asarray(t_ref, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    t_sam = np.asarray(t_sam, dtype=float)
    y_sam = np.asarray(y_sam, dtype=float)

    if len(t_ref) == len(t_sam) and np.allclose(t_ref, t_sam, atol=1e-9):
        return t_ref, y_ref, y_sam

    y_sam_interp = np.interp(t_ref, t_sam, y_sam, left=0.0, right=0.0)
    return t_ref, y_ref, y_sam_interp
