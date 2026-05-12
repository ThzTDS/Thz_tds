"""
thz_tds.refractive
==================
Refractive index and absorption coefficient extraction from THz-TDS data.

Implements the method of Withayachumnankul & Naftaly (2014), Sections 2.2–2.3:

* The transfer function T(ω) = E_sam(ω) / E_ref(ω) is computed.
* Phase is unwrapped only in the reliable frequency band (f_low – f_high THz)
  and linearly extrapolated to DC from a fit in that band.
* The refractive index follows Eq. (9a):  n(ω) = 1 − c φ(ω) / (ω l).
* The absorption coefficient follows the standard formula:
  α = −(2/l) ln[(n+1)² / (4n |T|)]  and is converted to cm⁻¹.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .spectral import fft_field, ensure_common_time_axis

# Speed of light in vacuum [m/s]
_C0 = 2.99792458e8


def compute_refractive_index(
    ref_ds,
    sam_ds,
    thickness_m: float,
    f_low_THz: float = 0.4,
    f_high_THz: float = 1.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract refractive index and absorption coefficient from two
    :class:`~thz_tds.dataset.THZDataset` objects.

    Parameters
    ----------
    ref_ds : THZDataset
        Reference (air) dataset.  Must have ``x_ps`` and ``y_avg`` populated.
    sam_ds : THZDataset
        Sample dataset.  Must have ``x_ps`` and ``y_avg`` populated.
    thickness_m : float
        Sample thickness in metres.
    f_low_THz : float
        Lower bound (THz) of the reliable frequency band used for phase
        unwrapping.  Below this, the phase is linearly extrapolated from a
        linear fit in the good band.
    f_high_THz : float
        Upper bound (THz) of the reliable band.

    Returns
    -------
    f_THz : np.ndarray
        Frequency axis in THz (DC removed).
    n_f : np.ndarray
        Refractive index as a function of frequency.
    T : np.ndarray
        Complex transfer function E_sam / E_ref.
    phi_full : np.ndarray
        Unwrapped, linearly extrapolated phase of T.
    alpha_cm : np.ndarray
        Absorption coefficient in cm⁻¹.
    """
    _, y_ref, y_sam = ensure_common_time_axis(
        ref_ds.x_ps, ref_ds.y_avg, sam_ds.x_ps, sam_ds.y_avg
    )
    return compute_refractive_index_arrays(
        ref_ds.x_ps, y_ref, y_sam, thickness_m, f_low_THz, f_high_THz
    )


def compute_refractive_index_arrays(
    time_ps: np.ndarray,
    y_ref: np.ndarray,
    y_sam: np.ndarray,
    thickness_m: float,
    f_low_THz: float = 0.4,
    f_high_THz: float = 1.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract refractive index and absorption coefficient from raw arrays.

    This lower-level variant is useful when data has already been loaded
    through a different pipeline (e.g. ``thzpy``) and is available as plain
    NumPy arrays on a common time axis.

    Parameters
    ----------
    time_ps : np.ndarray
        Common time axis in picoseconds (DC removed).
    y_ref : np.ndarray
        Reference electric-field amplitude on ``time_ps``.
    y_sam : np.ndarray
        Sample electric-field amplitude on ``time_ps``.
    thickness_m : float
        Sample thickness in metres.
    f_low_THz : float
        Lower frequency bound of the reliable band (THz).
    f_high_THz : float
        Upper frequency bound of the reliable band (THz).

    Returns
    -------
    f_THz, n_f, T, phi_full, alpha_cm
        Same outputs as :func:`compute_refractive_index`.

    Raises
    ------
    RuntimeError
        If no frequencies fall within the specified good band.
    """
    # 1 — FFT on the common time grid (no padding — keeps phase meaningful)
    f_Hz, E_ref = fft_field(time_ps, y_ref, pad_factor=1)
    _, E_sam = fft_field(time_ps, y_sam, pad_factor=1)

    # Remove DC bin to avoid division by zero
    f_Hz = f_Hz[1:]
    E_ref = E_ref[1:]
    E_sam = E_sam[1:]
    f_THz = f_Hz / 1e12

    # 2 — Transfer function
    T = E_sam / E_ref
    phi_wrapped = np.angle(T)

    # 3 — Phase unwrapping in the reliable band only (Withayachumnankul & Naftaly Sec. 2.3)
    band_good = (f_THz >= f_low_THz) & (f_THz <= f_high_THz)
    if not np.any(band_good):
        raise RuntimeError(
            f"No frequencies fall within the good band "
            f"[{f_low_THz}, {f_high_THz}] THz.  "
            "Check f_low_THz and f_high_THz settings."
        )

    phi_good_unwrapped = np.unwrap(phi_wrapped[band_good])

    # Linear fit in the good band to extrapolate to DC
    f_good_Hz = f_Hz[band_good]
    p = np.polyfit(f_good_Hz, phi_good_unwrapped, 1)

    # 4 — Build the full unwrapped phase vector
    phi_full = np.empty_like(phi_wrapped)
    phi_full[:] = np.nan
    phi_full[band_good] = phi_good_unwrapped

    # Below the good band: linear extrapolation
    band_low = f_THz < f_low_THz
    if np.any(band_low):
        phi_full[band_low] = np.polyval(p, f_Hz[band_low])

    # Remaining NaNs (high-frequency tail): fall back to wrapped phase
    nan_mask = np.isnan(phi_full)
    if np.any(nan_mask):
        phi_full[nan_mask] = phi_wrapped[nan_mask]

    # Force φ(lowest frequency) = 0 (Withayachumnankul & Naftaly convention)
    phi_full -= phi_full[0]

    # 5 — Refractive index: n(ω) = 1 − c φ(ω) / (ω l)  [Eq. 9a]
    omega = 2.0 * np.pi * f_Hz
    n_f = 1.0 - _C0 * phi_full / (omega * thickness_m)

    # 6 — Absorption coefficient: α = −(2/l) ln[(n+1)² / (4n |T|)]
    T_mag = np.maximum(np.abs(T), 1e-15)
    n_safe = np.maximum(n_f, 1e-6)

    alpha_m = -(2.0 / thickness_m) * np.log(
        ((n_safe + 1.0) ** 2 * T_mag) / (4.0 * n_safe + 1e-15)
    )
    alpha_cm = alpha_m / 100.0  # convert m⁻¹ → cm⁻¹

    return f_THz, n_f, T, phi_full, alpha_cm
