"""
thz_tds.refractive_selected
===========================
Refractive index and absorption extraction at selected THz frequencies.

This module does NOT filter the time-domain signal and does NOT create
narrowband signals. It first performs the normal broadband THz-TDS
refractive-index/absorption extraction using ``thz_tds.refractive`` and
then selects the requested frequency points.

The requested frequencies can be supplied from a YAML file, for example:

refractive_index:
  f_low_THz: 0.4
  f_high_THz: 1.1
  frequencies_THz: [0.5, 0.7, 0.9]

Any number of frequencies may be provided.

Selection methods
-----------------
"nearest"
    Return the FFT bin nearest to each requested frequency.
    This is the recommended/default mode because no extra interpolation
    is introduced.

"interpolate"
    Linearly interpolate n, alpha, phase, and the real/imaginary parts
    of the complex transfer function to the exact requested frequencies.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .refractive import (
    compute_refractive_index,
    compute_refractive_index_arrays,
)


def _validate_frequencies(
    frequencies_THz: Sequence[float],
    f_low_THz: float,
    f_high_THz: float,
) -> np.ndarray:
    """Validate and return requested frequencies as a 1-D float array."""
    f_req = np.asarray(frequencies_THz, dtype=float)

    if f_req.ndim != 1 or f_req.size == 0:
        raise ValueError(
            "frequencies_THz must be a non-empty 1-D sequence, "
            "e.g. [0.5, 0.7, 0.9]."
        )

    if not np.all(np.isfinite(f_req)):
        raise ValueError("frequencies_THz contains NaN or infinite values.")

    if np.any(f_req <= 0.0):
        raise ValueError("All selected frequencies must be > 0 THz.")

    outside = (f_req < f_low_THz) | (f_req > f_high_THz)
    if np.any(outside):
        bad = ", ".join(f"{f:.6g}" for f in f_req[outside])
        raise ValueError(
            f"Selected frequencies [{bad}] THz lie outside the reliable "
            f"band [{f_low_THz}, {f_high_THz}] THz."
        )

    return f_req


def _select_from_spectrum(
    f_THz: np.ndarray,
    n_f: np.ndarray,
    T: np.ndarray,
    phi_full: np.ndarray,
    alpha_cm: np.ndarray,
    frequencies_THz: Sequence[float],
    method: str = "nearest",
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Select requested frequencies from an already calculated broadband spectrum.

    Returns
    -------
    f_selected_THz : np.ndarray
        Actual frequencies returned. For ``nearest``, these are the FFT-bin
        frequencies. For ``interpolate``, these equal the requested values.
    n_selected : np.ndarray
        Refractive index at selected frequencies.
    T_selected : np.ndarray
        Complex transfer function at selected frequencies.
    phi_selected : np.ndarray
        Unwrapped phase at selected frequencies.
    alpha_selected_cm : np.ndarray
        Absorption coefficient at selected frequencies in cm^-1.
    """
    f_req = np.asarray(frequencies_THz, dtype=float)
    method = method.lower().strip()

    if method == "nearest":
        indices = np.array(
            [int(np.argmin(np.abs(f_THz - f))) for f in f_req],
            dtype=int,
        )

        return (
            f_THz[indices],
            n_f[indices],
            T[indices],
            phi_full[indices],
            alpha_cm[indices],
        )

    if method == "interpolate":
        # np.interp works on real arrays, so interpolate real and imaginary
        # parts of the complex transfer function separately.
        T_real = np.interp(f_req, f_THz, np.real(T))
        T_imag = np.interp(f_req, f_THz, np.imag(T))
        T_selected = T_real + 1j * T_imag

        return (
            f_req.copy(),
            np.interp(f_req, f_THz, n_f),
            T_selected,
            np.interp(f_req, f_THz, phi_full),
            np.interp(f_req, f_THz, alpha_cm),
        )

    raise ValueError(
        f"Unknown selection method {method!r}. "
        "Use 'nearest' or 'interpolate'."
    )


def compute_refractive_index_selected(
    ref_ds,
    sam_ds,
    thickness_m: float,
    frequencies_THz: Sequence[float],
    f_low_THz: float = 0.4,
    f_high_THz: float = 1.1,
    method: str = "nearest",
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Extract n and alpha only at selected THz frequencies from THZDataset objects.

    The normal broadband calculation is performed first so that phase
    unwrapping remains identical to :func:`compute_refractive_index`.
    Only afterward are the requested frequency points selected.

    Parameters
    ----------
    ref_ds : THZDataset
        Reference (air) dataset.
    sam_ds : THZDataset
        Sample dataset.
    thickness_m : float
        Sample thickness in metres.
    frequencies_THz : sequence of float
        Requested frequencies in THz, e.g. [0.5, 0.7, 0.9].
        The list may contain any number of frequencies.
    f_low_THz : float
        Lower bound of the reliable phase-unwrapping band.
    f_high_THz : float
        Upper bound of the reliable phase-unwrapping band.
    method : {"nearest", "interpolate"}
        How requested frequencies are taken from the broadband result.

    Returns
    -------
    f_selected_THz, n_selected, T_selected, phi_selected, alpha_selected_cm
    """
    f_req = _validate_frequencies(
        frequencies_THz,
        f_low_THz,
        f_high_THz,
    )

    f_THz, n_f, T, phi_full, alpha_cm = compute_refractive_index(
        ref_ds=ref_ds,
        sam_ds=sam_ds,
        thickness_m=thickness_m,
        f_low_THz=f_low_THz,
        f_high_THz=f_high_THz,
    )

    return _select_from_spectrum(
        f_THz=f_THz,
        n_f=n_f,
        T=T,
        phi_full=phi_full,
        alpha_cm=alpha_cm,
        frequencies_THz=f_req,
        method=method,
    )


def compute_refractive_index_arrays_selected(
    time_ps: np.ndarray,
    y_ref: np.ndarray,
    y_sam: np.ndarray,
    thickness_m: float,
    frequencies_THz: Sequence[float],
    f_low_THz: float = 0.4,
    f_high_THz: float = 1.1,
    method: str = "nearest",
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Array-based version of :func:`compute_refractive_index_selected`.

    Useful when reference and sample waveforms are already available as
    NumPy arrays on a common time axis.
    """
    f_req = _validate_frequencies(
        frequencies_THz,
        f_low_THz,
        f_high_THz,
    )

    f_THz, n_f, T, phi_full, alpha_cm = compute_refractive_index_arrays(
        time_ps=time_ps,
        y_ref=y_ref,
        y_sam=y_sam,
        thickness_m=thickness_m,
        f_low_THz=f_low_THz,
        f_high_THz=f_high_THz,
    )

    return _select_from_spectrum(
        f_THz=f_THz,
        n_f=n_f,
        T=T,
        phi_full=phi_full,
        alpha_cm=alpha_cm,
        frequencies_THz=f_req,
        method=method,
    )
