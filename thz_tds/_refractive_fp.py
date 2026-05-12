"""
thz_tds.refractive_fp
=====================
Fabry–Pérot-aware refractive index and absorption coefficient extraction
from THz-TDS data at normal incidence with known sample thickness.

Physical model
--------------
The sample is treated as a homogeneous planar slab of known thickness *L*
surrounded by air (n_air = 1).  Multiple internal reflections (Fabry–Pérot
echoes) are included via the exact FP slab transfer function.

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
1.  Pre-process time-domain traces onto a common time axis.
2.  FFT both traces (no zero-padding, preserving phase).
3.  Compute the measured transfer function H_meas(ω) = E_sam / E_ref.
4.  Obtain initial estimates n⁰(ω) and α⁰(ω) from the no-FP method
    (:func:`~thz_tds.refractive.compute_refractive_index_arrays`).
5.  Convert α⁰ → κ⁰.
6.  Within the reliable frequency band, refine n and κ by minimising the
    complex residual H_model(n, κ) − H_meas, frequency by frequency, using
    ``scipy.optimize.least_squares`` (real + imaginary parts stacked).
7.  Outside the reliable band, keep the initial (no-FP) values.
8.  Optionally smooth n_fp and κ_fp with a Savitzky–Golay filter.
9.  Reconstruct the modelled sample spectrum and time-domain trace.

References
----------
Withayachumnankul, W. & Naftaly, M. (2014). Fundamentals of measurement in
terahertz time-domain spectroscopy. *Journal of Infrared, Millimeter, and
Terahertz Waves*, 35(8), 610–637.
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
    kappa = alpha_m * _C0 / (2.0 * omega)
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
    denominator = 1.0 - r10 ** 2 * np.exp(-1j * rt_phase)

    return numerator / denominator


# ---------------------------------------------------------------------------
# High-level wrapper  (operates on THZDataset objects)
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
) -> FPResult:
    """
    Extract FP-refined optical constants from two
    :class:`~thz_tds.dataset.THZDataset` objects.

    This is a thin convenience wrapper around
    :func:`compute_refractive_index_fp_arrays` that handles interpolation onto
    a common time axis from dataset objects.

    Parameters
    ----------
    ref_ds : THZDataset
        Reference (air) dataset.  Must have ``x_ps`` and ``y_avg`` populated.
    sam_ds : THZDataset
        Sample dataset.  Must have ``x_ps`` and ``y_avg`` populated.
    thickness_m : float
        Sample thickness in metres.
    f_low_THz : float
        Lower bound (THz) of the reliable frequency band used for fitting.
    f_high_THz : float
        Upper bound (THz) of the reliable band.
    n_air : float
        Refractive index of the surrounding medium (default 1.0).
    n_bounds : tuple of float
        ``(lower, upper)`` bounds on the real refractive index n during fitting.
    kappa_bounds : tuple of float
        ``(lower, upper)`` bounds on the extinction coefficient κ during fitting.
    smooth : bool
        If ``True``, apply Savitzky–Golay smoothing to n_fp and kappa_fp
        within the fit band after fitting.
    smooth_window : int
        Window length for the Savitzky–Golay filter (must be odd).
    smooth_polyorder : int
        Polynomial order for the Savitzky–Golay filter.

    Returns
    -------
    FPResult
        Named tuple with all extracted quantities.  See :class:`FPResult`.
    """
    _, y_ref, y_sam = ensure_common_time_axis(
        ref_ds.x_ps, ref_ds.y_avg, sam_ds.x_ps, sam_ds.y_avg
    )
    return compute_refractive_index_fp_arrays(
        time_ps=ref_ds.x_ps,
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
) -> FPResult:
    """
    Extract FP-refined optical constants from raw NumPy arrays on a common
    time axis.

    This is the core implementation.  It follows the workflow described in the
    module docstring.

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
        Refractive index of the surrounding medium (default 1.0).
    n_bounds : tuple of float
        ``(lower, upper)`` bounds on n for the nonlinear fit.
    kappa_bounds : tuple of float
        ``(lower, upper)`` bounds on κ for the nonlinear fit.
    smooth : bool
        If ``True``, apply Savitzky–Golay smoothing to n_fp and kappa_fp
        within the fit band after fitting.
    smooth_window : int
        Window length for the Savitzky–Golay filter (must be odd).
    smooth_polyorder : int
        Polynomial order for the Savitzky–Golay filter.

    Returns
    -------
    FPResult
        Named tuple with fields: f_THz, n_fp, kappa_fp, alpha_cm_fp,
        H_meas, H_model, E_sam_model, y_sam_model, n_init, alpha_init_cm.

    Raises
    ------
    RuntimeError
        If no frequencies fall within ``[f_low_THz, f_high_THz]``.
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

    # 2 — Measured transfer function
    H_meas = E_sam / E_ref

    # 3 — Initial (no-FP) guess for n and α via the standard simple formula
    _, n_init, _, _, alpha_init_cm = compute_refractive_index_arrays(
        time_ps=time_ps,
        y_ref=y_ref,
        y_sam=y_sam,
        thickness_m=thickness_m,
        f_low_THz=f_low_THz,
        f_high_THz=f_high_THz,
    )

    # 4 — Convert initial α → κ (used as starting point for the FP fit)
    kappa_init = alpha_cm_to_kappa(f_Hz, alpha_init_cm)

    # 5 — Identify the reliable fitting band
    band_fit = (f_THz >= f_low_THz) & (f_THz <= f_high_THz)
    if not np.any(band_fit):
        raise RuntimeError(
            f"No frequencies fall within the fit band "
            f"[{f_low_THz}, {f_high_THz}] THz.  "
            "Check f_low_THz and f_high_THz settings."
        )

    # 6 — Frequency-by-frequency nonlinear fit inside the reliable band.
    #     Outside the band, the initial no-FP values are kept unchanged.
    n_fp = n_init.copy()
    kappa_fp = kappa_init.copy()

    lb = [float(n_bounds[0]), float(kappa_bounds[0])]
    ub = [float(n_bounds[1]), float(kappa_bounds[1])]

    fit_indices = np.where(band_fit)[0]
    log.debug(
        "FP fit: %d frequency points in [%.3f, %.3f] THz",
        len(fit_indices), f_low_THz, f_high_THz,
    )

    for idx in fit_indices:
        H_target = H_meas[idx]   # complex scalar for this frequency bin
        om_val = float(omega[idx])

        def _residuals(x: np.ndarray, _om=om_val, _Ht=H_target) -> np.ndarray:
            """Stack [Re, Im] of H_model(n, κ) − H_target for least_squares."""
            H_mod = fp_transfer_function_normal_incidence(
                omega=np.array([_om]),
                n=np.array([x[0]]),
                kappa=np.array([x[1]]),
                thickness_m=thickness_m,
                n_air=n_air,
            )[0]
            diff = H_mod - _Ht
            return np.array([diff.real, diff.imag])

        # Initial guess — clipped to stay inside the bounds
        x0 = np.array([
            float(np.clip(n_fp[idx], lb[0], ub[0])),
            float(np.clip(kappa_fp[idx], lb[1], ub[1])),
        ])

        result = least_squares(
            _residuals,
            x0=x0,
            bounds=(lb, ub),
            method="trf",       # Trust Region Reflective handles bounds natively
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=200,
        )

        n_fp[idx] = result.x[0]
        kappa_fp[idx] = result.x[1]

    # 7 — Optional Savitzky–Golay smoothing within the fit band
    if smooth:
        n_fp, kappa_fp = _savgol_smooth_band(
            n_fp, kappa_fp, band_fit, smooth_window, smooth_polyorder
        )

    # 8 — Derive α from the refined κ
    alpha_cm_fp = kappa_to_alpha_cm(f_Hz, kappa_fp)

    # 9 — Reconstruct modelled transfer function, sample spectrum, and
    #     time-domain trace using the full frequency grid.
    H_model = fp_transfer_function_normal_incidence(
        omega=omega,
        n=n_fp,
        kappa=kappa_fp,
        thickness_m=thickness_m,
        n_air=n_air,
    )
    E_sam_model = H_model * E_ref

    # Inverse FFT: re-insert the DC bin (set to zero) that was stripped, then
    # call irfft with the original time-axis length to recover the time domain.
    E_full = np.concatenate([[0.0 + 0.0j], E_sam_model])
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
    Apply Savitzky–Golay smoothing to n and κ within ``band_mask``.

    Values outside the band are left unchanged.  The window length is clamped
    to the number of in-band samples and forced to be odd so that
    ``savgol_filter`` does not raise.

    Parameters
    ----------
    n_arr : np.ndarray
        Real refractive index array (full frequency grid).
    kappa_arr : np.ndarray
        Extinction coefficient array (full frequency grid).
    band_mask : np.ndarray of bool
        Boolean mask selecting the in-band frequencies.
    window : int
        Desired Savitzky–Golay window length (will be clamped if necessary).
    polyorder : int
        Polynomial order for the Savitzky–Golay filter.

    Returns
    -------
    n_smooth : np.ndarray
        Smoothed refractive index (in-band region only modified).
    kappa_smooth : np.ndarray
        Smoothed extinction coefficient (clipped to ≥ 0, in-band only).
    """
    n_out = n_arr.copy()
    kappa_out = kappa_arr.copy()

    n_in_band = int(np.sum(band_mask))
    win = min(window, n_in_band)
    if win % 2 == 0:
        win = max(win - 1, 1)

    if win <= polyorder:
        log.warning(
            "Savitzky–Golay: effective window (%d) ≤ polyorder (%d); "
            "skipping smoothing.",
            win, polyorder,
        )
        return n_out, kappa_out

    n_out[band_mask] = savgol_filter(n_arr[band_mask], win, polyorder)
    kappa_out[band_mask] = savgol_filter(kappa_arr[band_mask], win, polyorder)
    kappa_out = np.maximum(kappa_out, 0.0)  # extinction coefficient must be ≥ 0

    return n_out, kappa_out
