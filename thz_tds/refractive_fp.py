"""
thz_tds.refractive_fp
=====================
Fabry–Pérot-aware refractive index and absorption coefficient extraction
from THz-TDS data at normal incidence with known sample thickness.

Physical model
--------------
The sample is treated as a homogeneous planar slab of known thickness *L*
surrounded by air. Multiple internal reflections (Fabry–Pérot echoes) are
included via the exact FP slab transfer function.

The complex refractive index is:

    ñ(ω) = n(ω) − j·κ(ω)

Fresnel coefficients at normal incidence:

    t₀₁ = 2 n_air / (n_air + ñ)          [air → sample transmission]
    t₁₀ = 2 ñ     / (n_air + ñ)          [sample → air transmission]
    r₁₀ = (n_air − ñ) / (n_air + ñ)      [sample → air reflection]

FP slab transfer function:

    H(ω) = t₀₁ t₁₀ exp[−jω(ñ − n_air)L/c]
            / {1 − r₁₀² exp[−2jω ñ L/c]}

Workflow
--------
1. Pre-process time-domain traces onto a common time axis.
2. FFT both traces (no zero-padding, preserving phase).
3. Compute the measured transfer function H_meas(ω) = E_sam / E_ref.
4. Obtain initial estimates n⁰(ω) and α⁰(ω) from the no-FP method
   (:func:`~thz_tds.refractive.compute_refractive_index_arrays`).
5. Convert α⁰ → κ⁰.
6. Within the reliable frequency band, refine n and κ by minimising the
   complex residual H_model(n, κ) − H_meas, frequency by frequency, using
   ``scipy.optimize.least_squares`` (real + imaginary parts stacked).
7. Outside the reliable band, keep the initial (no-FP) values.
8. Optionally smooth n_fp and κ_fp with a Savitzky–Golay filter.
9. Reconstruct the modelled sample spectrum and time-domain trace.

Notes
-----
- This is a Version-1 implementation: fitting is done frequency-by-frequency.
- The fitted result at the previous frequency is reused as the initial guess
  for the next frequency inside the fitting band, which improves continuity.
- Division by very small reference-spectrum values is safeguarded.
- Time-domain reconstruction assumes that ``fft_field`` is compatible with
  NumPy's rFFT/irFFT convention.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from .refractive import compute_refractive_index_arrays
from .spectral import ensure_common_time_axis, fft_field

log = logging.getLogger(__name__)

# Speed of light in vacuum [m/s]
_C0 = 2.99792458e8


# ---------------------------------------------------------------------------
# Public result container
# ---------------------------------------------------------------------------

class FPResult(NamedTuple):
    """
    Result of Fabry–Pérot-aware optical-constant extraction.

    All array fields share the same frequency axis ``f_THz`` (DC removed,
    positive frequencies only).

    Attributes
    ----------
    f_THz : np.ndarray
        Frequency axis in THz.
    n_fp : np.ndarray
        FP-refined real refractive index n(ω).
    kappa_fp : np.ndarray
        FP-refined extinction coefficient κ(ω).
    alpha_cm_fp : np.ndarray
        FP-refined absorption coefficient α(ω) in cm⁻¹.
    H_meas : np.ndarray
        Measured complex transfer function E_sam / E_ref.
    H_model : np.ndarray
        FP model transfer function evaluated at the refined n, κ.
    E_sam_model : np.ndarray
        Modelled sample spectrum: H_model × E_ref (complex, DC-free).
    y_sam_model : np.ndarray
        Modelled sample time-domain trace (inverse FFT of E_sam_model).
    n_init : np.ndarray
        Initial (no-FP) refractive index used as the starting guess.
    alpha_init_cm : np.ndarray
        Initial (no-FP) absorption coefficient in cm⁻¹.
    """

    f_THz: np.ndarray
    n_fp: np.ndarray
    kappa_fp: np.ndarray
    alpha_cm_fp: np.ndarray
    H_meas: np.ndarray
    H_model: np.ndarray
    E_sam_model: np.ndarray
    y_sam_model: np.ndarray
    n_init: np.ndarray
    alpha_init_cm: np.ndarray


# ---------------------------------------------------------------------------
# Physical helper functions
# ---------------------------------------------------------------------------

def alpha_cm_to_kappa(f_Hz: np.ndarray, alpha_cm: np.ndarray) -> np.ndarray:
    """
    Convert absorption coefficient α [cm⁻¹] to extinction coefficient κ.

    Derived from the imaginary part of the plane-wave phase factor:

        E ∝ exp[−jωñz/c] = exp[−jωnz/c] · exp[−ωκz/c]
        ⟹  α(m⁻¹) = 2ω κ / c
        ⟹  κ = α(m⁻¹) · c / (2ω) = α(cm⁻¹) · 100 · c / (2ω)

    Parameters
    ----------
    f_Hz : np.ndarray
        Frequency axis in Hz (positive, DC-free).
    alpha_cm : np.ndarray
        Absorption coefficient in cm⁻¹, shape matching ``f_Hz``.

    Returns
    -------
    kappa : np.ndarray
        Extinction coefficient (dimensionless, clipped to ≥ 0).
    """
    alpha_m = alpha_cm * 100.0  # cm⁻¹ → m⁻¹
    omega = 2.0 * np.pi * f_Hz
    kappa = alpha_m * _C0 / np.maximum(2.0 * omega, 1e-30)
    return np.maximum(kappa, 0.0)


def kappa_to_alpha_cm(f_Hz: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """
    Convert extinction coefficient κ to absorption coefficient α [cm⁻¹].

        α(m⁻¹) = 2ω κ / c
        α(cm⁻¹) = α(m⁻¹) / 100

    Parameters
    ----------
    f_Hz : np.ndarray
        Frequency axis in Hz (positive, DC-free).
    kappa : np.ndarray
        Extinction coefficient (dimensionless), shape matching ``f_Hz``.

    Returns
    -------
    alpha_cm : np.ndarray
        Absorption coefficient in cm⁻¹.
    """
    omega = 2.0 * np.pi * f_Hz
    alpha_m = 2.0 * omega * kappa / _C0
    return alpha_m / 100.0


def fp_transfer_function_normal_incidence(
    omega: np.ndarray,
    n: np.ndarray,
    kappa: np.ndarray,
    thickness_m: float,
    n_air: float = 1.0,
) -> np.ndarray:
    """
    Compute the Fabry–Pérot slab transfer function at normal incidence.

    With ñ = n − j·κ, the Fresnel coefficients are:

        t₀₁ = 2 n_air / (n_air + ñ)
        t₁₀ = 2 ñ     / (n_air + ñ)
        r₁₀ = (n_air − ñ) / (n_air + ñ)

    and the transfer function is:

        H(ω) = t₀₁ t₁₀ exp[−jω(ñ − n_air)L/c]
                / {1 − r₁₀² exp[−2jω ñ L/c]}

    Parameters
    ----------
    omega : np.ndarray
        Angular frequency in rad/s (positive, DC-free).
    n : np.ndarray
        Real refractive index, shape matching ``omega``.
    kappa : np.ndarray
        Extinction coefficient, shape matching ``omega``.
    thickness_m : float
        Sample thickness in metres.
    n_air : float
        Refractive index of the surrounding medium (default 1.0).

    Returns
    -------
    H : np.ndarray
        Complex FP transfer function, shape matching ``omega``.
    """
    n_tilde = n - 1j * kappa  # complex refractive index  ñ = n − j κ

    t01 = (2.0 * n_air) / (n_air + n_tilde)
    t10 = (2.0 * n_tilde) / (n_air + n_tilde)
    r10 = (n_air - n_tilde) / (n_air + n_tilde)

    # Single-pass propagation through the slab relative to air propagation
    #   Phase argument: ω(ñ − n_air)L/c
    delta_phase = omega * (n_tilde - n_air) * thickness_m / _C0
    # Round-trip phase inside the slab: 2ω ñ L/c
    rt_phase = 2.0 * omega * n_tilde * thickness_m / _C0

    numerator = t01 * t10 * np.exp(-1j * delta_phase)
    denominator = 1.0 - (r10 ** 2) * np.exp(-1j * rt_phase)

    return numerator / denominator


# ---------------------------------------------------------------------------
# High-level wrapper (operates on THZDataset objects)
# ---------------------------------------------------------------------------

def compute_refractive_index_fp(
    ref_ds,
    sam_ds,
    thickness_m: float,
    f_low_THz: float = 0.4,
    f_high_THz: float = 1.1,
    n_air: float = 1.0,
    n_bounds: Tuple[float, float] = (0.5, 10.0),
    kappa_bounds: Tuple[float, float] = (0.0, 10.0),
    smooth: bool = False,
    smooth_window: int = 11,
    smooth_polyorder: int = 3,
    ref_floor_rel: float = 1e-3,
    ref_floor_abs: float = 1e-15,
) -> FPResult:
    """
    Extract FP-refined optical constants from two dataset objects.

    Parameters
    ----------
    ref_ds : THZDataset
        Reference (air) dataset.
    sam_ds : THZDataset
        Sample dataset.
    thickness_m : float
        Sample thickness in metres.
    f_low_THz : float
        Lower bound (THz) of the reliable fitting band.
    f_high_THz : float
        Upper bound (THz) of the reliable fitting band.
    n_air : float
        Refractive index of the surrounding medium.
    n_bounds : tuple of float
        Bounds on n during fitting.
    kappa_bounds : tuple of float
        Bounds on kappa during fitting.
    smooth : bool
        If True, apply Savitzky–Golay smoothing to in-band fitted curves.
    smooth_window : int
        Window length for Savitzky–Golay filtering.
    smooth_polyorder : int
        Polynomial order for Savitzky–Golay filtering.
    ref_floor_rel : float
        Relative threshold for rejecting weak-reference bins in fitting.
    ref_floor_abs : float
        Absolute floor used to stabilise division by E_ref.

    Returns
    -------
    FPResult
        Named tuple with extracted quantities.
    """
    time_ps_common, y_ref, y_sam = ensure_common_time_axis(
        ref_ds.x_ps, ref_ds.y_avg, sam_ds.x_ps, sam_ds.y_avg
    )

    return compute_refractive_index_fp_arrays(
        time_ps=time_ps_common,
        y_ref=y_ref,
        y_sam=y_sam,
        thickness_m=thickness_m,
        f_low_THz=f_low_THz,
        f_high_THz=f_high_THz,
        n_air=n_air,
        n_bounds=n_bounds,
        kappa_bounds=kappa_bounds,
        smooth=smooth,
        smooth_window=smooth_window,
        smooth_polyorder=smooth_polyorder,
        ref_floor_rel=ref_floor_rel,
        ref_floor_abs=ref_floor_abs,
    )


# ---------------------------------------------------------------------------
# Array-level function
# ---------------------------------------------------------------------------

def compute_refractive_index_fp_arrays(
    time_ps: np.ndarray,
    y_ref: np.ndarray,
    y_sam: np.ndarray,
    thickness_m: float,
    f_low_THz: float = 0.4,
    f_high_THz: float = 1.1,
    n_air: float = 1.0,
    n_bounds: Tuple[float, float] = (0.5, 10.0),
    kappa_bounds: Tuple[float, float] = (0.0, 10.0),
    smooth: bool = False,
    smooth_window: int = 11,
    smooth_polyorder: int = 3,
    ref_floor_rel: float = 1e-3,
    ref_floor_abs: float = 1e-15,
) -> FPResult:
    """
    Extract FP-refined optical constants from raw NumPy arrays on a common
    time axis.

    Parameters
    ----------
    time_ps : np.ndarray
        Common time axis in picoseconds.
    y_ref : np.ndarray
        Reference electric-field amplitude on ``time_ps``.
    y_sam : np.ndarray
        Sample electric-field amplitude on ``time_ps``.
    thickness_m : float
        Sample thickness in metres.
    f_low_THz : float
        Lower frequency bound of the reliable fitting band (THz).
    f_high_THz : float
        Upper frequency bound of the reliable fitting band (THz).
    n_air : float
        Refractive index of the surrounding medium.
    n_bounds : tuple of float
        Bounds on n for the nonlinear fit.
    kappa_bounds : tuple of float
        Bounds on kappa for the nonlinear fit.
    smooth : bool
        If True, apply Savitzky–Golay smoothing to in-band fitted curves.
    smooth_window : int
        Window length for Savitzky–Golay filtering.
    smooth_polyorder : int
        Polynomial order for Savitzky–Golay filtering.
    ref_floor_rel : float
        Relative threshold for rejecting weak-reference bins in fitting.
    ref_floor_abs : float
        Absolute floor used to stabilise division by E_ref.

    Returns
    -------
    FPResult
        Named tuple with fields:
        f_THz, n_fp, kappa_fp, alpha_cm_fp, H_meas, H_model,
        E_sam_model, y_sam_model, n_init, alpha_init_cm.

    Raises
    ------
    RuntimeError
        If no frequencies fall within the fit band after masking.
    """
    n_time = len(time_ps)

    # 1 — FFT on the common time grid (no padding — keeps phase meaningful)
    f_Hz, E_ref = fft_field(time_ps, y_ref, pad_factor=1)
    _, E_sam = fft_field(time_ps, y_sam, pad_factor=1)

    # Remove DC bin to avoid division by zero
    f_Hz = f_Hz[1:]
    E_ref = E_ref[1:]
    E_sam = E_sam[1:]
    f_THz = f_Hz / 1e12
    omega = 2.0 * np.pi * f_Hz

    # 2 — Measured transfer function with safe denominator
    E_ref_safe = np.where(np.abs(E_ref) < ref_floor_abs, ref_floor_abs + 0j, E_ref)
    H_meas = E_sam / E_ref_safe

    # 3 — Initial (no-FP) guess for n and alpha
    _, n_init, _, _, alpha_init_cm = compute_refractive_index_arrays(
        time_ps=time_ps,
        y_ref=y_ref,
        y_sam=y_sam,
        thickness_m=thickness_m,
        f_low_THz=f_low_THz,
        f_high_THz=f_high_THz,
    )

    # 4 — Convert initial alpha -> kappa
    kappa_init = alpha_cm_to_kappa(f_Hz, alpha_init_cm)

    # 5 — Identify the reliable fitting band
    ref_mag = np.abs(E_ref)
    ref_threshold = max(ref_floor_abs, ref_floor_rel * np.max(ref_mag))

    band_fit = (
        (f_THz >= f_low_THz) &
        (f_THz <= f_high_THz) &
        (ref_mag > ref_threshold)
    )

    if not np.any(band_fit):
        raise RuntimeError(
            f"No frequencies fall within the fit band "
            f"[{f_low_THz}, {f_high_THz}] THz after reference-magnitude masking. "
            f"Try widening the band or lowering ref_floor_rel={ref_floor_rel}."
        )

    # 6 — Frequency-by-frequency nonlinear fit inside the reliable band
    n_fp = n_init.copy()
    kappa_fp = np.maximum(kappa_init.copy(), 0.0)

    lb = np.array([float(n_bounds[0]), float(kappa_bounds[0])], dtype=float)
    ub = np.array([float(n_bounds[1]), float(kappa_bounds[1])], dtype=float)

    fit_indices = np.where(band_fit)[0]
    log.debug(
        "FP fit: %d frequency points in [%.3f, %.3f] THz",
        len(fit_indices), f_low_THz, f_high_THz,
    )

    prev_x = None

    for idx in fit_indices:
        H_target = H_meas[idx]
        om_val = float(omega[idx])

        def _residuals(x: np.ndarray, _om=om_val, _Ht=H_target) -> np.ndarray:
            """Stack [Re, Im] of H_model(n, kappa) − H_target."""
            H_mod = fp_transfer_function_normal_incidence(
                omega=np.array([_om], dtype=float),
                n=np.array([x[0]], dtype=float),
                kappa=np.array([x[1]], dtype=float),
                thickness_m=thickness_m,
                n_air=n_air,
            )[0]
            diff = H_mod - _Ht
            return np.array([diff.real, diff.imag], dtype=float)

        if prev_x is None:
            x0 = np.array(
                [
                    float(np.clip(n_fp[idx], lb[0], ub[0])),
                    float(np.clip(kappa_fp[idx], lb[1], ub[1])),
                ],
                dtype=float,
            )
        else:
            x0 = np.array(
                [
                    float(np.clip(prev_x[0], lb[0], ub[0])),
                    float(np.clip(prev_x[1], lb[1], ub[1])),
                ],
                dtype=float,
            )

        result = least_squares(
            _residuals,
            x0=x0,
            bounds=(lb, ub),
            method="trf",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=200,
        )

        n_fp[idx] = result.x[0]
        kappa_fp[idx] = result.x[1]
        prev_x = result.x.copy()

    # 7 — Optional Savitzky–Golay smoothing within the fit band
    if smooth:
        n_fp, kappa_fp = _savgol_smooth_band(
            n_fp, kappa_fp, band_fit, smooth_window, smooth_polyorder
        )

    # 8 — Derive alpha from the refined kappa
    alpha_cm_fp = kappa_to_alpha_cm(f_Hz, kappa_fp)

    # 9 — Reconstruct modelled transfer function, sample spectrum, and time trace
    H_model = fp_transfer_function_normal_incidence(
        omega=omega,
        n=n_fp,
        kappa=kappa_fp,
        thickness_m=thickness_m,
        n_air=n_air,
    )
    E_sam_model = H_model * E_ref

    # Re-insert zero DC bin and transform back to time domain.
    E_full = np.concatenate(([0.0 + 0.0j], E_sam_model))
    y_sam_model = np.fft.irfft(E_full, n=n_time)

    return FPResult(
        f_THz=f_THz,
        n_fp=n_fp,
        kappa_fp=kappa_fp,
        alpha_cm_fp=alpha_cm_fp,
        H_meas=H_meas,
        H_model=H_model,
        E_sam_model=E_sam_model,
        y_sam_model=y_sam_model,
        n_init=n_init,
        alpha_init_cm=alpha_init_cm,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _savgol_smooth_band(
    n_arr: np.ndarray,
    kappa_arr: np.ndarray,
    band_mask: np.ndarray,
    window: int,
    polyorder: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Savitzky–Golay smoothing to n and kappa within ``band_mask``.

    Values outside the band are left unchanged. The window length is clamped
    to the number of in-band samples and forced to be odd.

    Parameters
    ----------
    n_arr : np.ndarray
        Real refractive index array (full frequency grid).
    kappa_arr : np.ndarray
        Extinction coefficient array (full frequency grid).
    band_mask : np.ndarray of bool
        Boolean mask selecting the in-band frequencies.
    window : int
        Desired Savitzky–Golay window length.
    polyorder : int
        Polynomial order for the Savitzky–Golay filter.

    Returns
    -------
    n_smooth : np.ndarray
        Smoothed refractive index (in-band region only modified).
    kappa_smooth : np.ndarray
        Smoothed extinction coefficient (clipped to >= 0, in-band only).
    """
    n_out = n_arr.copy()
    kappa_out = kappa_arr.copy()

    n_in_band = int(np.sum(band_mask))
    win = min(window, n_in_band)

    if win % 2 == 0:
        win = max(win - 1, 1)

    if win <= polyorder or n_in_band < 3:
        log.warning(
            "Savitzky–Golay skipped: effective window (%d), polyorder (%d), in-band points (%d).",
            win, polyorder, n_in_band,
        )
        return n_out, kappa_out

    n_out[band_mask] = savgol_filter(n_arr[band_mask], win, polyorder)
    kappa_out[band_mask] = savgol_filter(kappa_arr[band_mask], win, polyorder)
    kappa_out = np.maximum(kappa_out, 0.0)

    return n_out, kappa_out
