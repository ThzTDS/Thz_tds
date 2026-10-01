"""
thz_tds.refractive_complex_iterative
=====================================
Refractive index and absorption coefficient extraction from THz-TDS data
using the *full complex* Fresnel transmission factor

    F(ntilde) = 4*ntilde / (1 + ntilde)**2

instead of the real-valued approximation F_real(n) = 4n/(1+n)**2 used in
:mod:`thz_tds.refractive`.  Because the complex index

    ntilde(f) = n(f) - i*kappa(f)

depends on kappa, which itself depends on the absorption coefficient alpha
that we are trying to extract, n, kappa and alpha must be found by fixed-
point iteration rather than in one shot.

Sign convention
----------------
This project's FFT helper (:func:`thz_tds.spectral.fft_field`) is a thin
wrapper around ``numpy.fft.rfft``, i.e. it uses the basis
``exp(-i*2*pi*f*t)``.  The existing (real-Fresnel) module
:mod:`thz_tds.refractive` combines that FFT convention with the extraction
formula

    n(f) = 1 - c*phi(f) / (omega*d)                              (refractive.py, Eq. 9a)

where ``phi = arg(T)`` and ``T = E_sam/E_ref``.  Note that this is the
*opposite* sign from the more commonly published ``n = 1 + c*phi/(omega*d)``
that appears in papers assuming the opposite FFT sign convention.

Working through what transfer function is consistent with
``n = 1 - c*phi/(omega*d)`` shows that, for a slab of thickness ``d`` with
complex refractive index ``ntilde`` and complex Fresnel transmission factor
``F(ntilde)``, the transfer function must be of the form

    H(f) = F(ntilde) * exp[-i*(omega*d/c)*(ntilde - 1)]                      (*)

so that, writing ``ntilde = n - i*kappa`` with ``kappa >= 0``,

    exp[-i*(omega*d/c)*(ntilde - 1)]
        = exp[-i*(omega*d/c)*(n - 1)] * exp[-i*(omega*d/c)*(-i*kappa)]
        = exp[-i*(omega*d/c)*(n - 1)] * exp[-(omega*d/c)*kappa]
        = exp[-i*(omega*d/c)*(n - 1)] * exp(-alpha*d/2)

with ``alpha = 2*omega*kappa/c = 4*pi*f*kappa/c``, i.e.
``kappa = c*alpha/(4*pi*f)``.  This reproduces the amplitude-attenuation
term ``exp(-alpha*d/2)`` used throughout the project (see also
``thz_tds.refractive_fp.alpha_cm_to_kappa``, which independently derives and
uses the *same* sign: ``ntilde = n - j*kappa``).  Therefore, in this
codebase:

    ntilde = n - i*kappa,   kappa = c*alpha / (4*pi*f) >= 0 for an
                                     attenuating (absorbing) medium.

Using the opposite sign (``ntilde = n + i*kappa``) would flip the sign of
the exponent above and turn absorption into *gain* — see
``_check_attenuation_sign()`` at the bottom of this module for a numerical
proof that ``ntilde = n - i*kappa`` is the physically correct choice for
this project's conventions.

From (*):

    arg H = arg F(ntilde) - (omega*d/c)*(n - 1)
    |H|   = |F(ntilde)| * exp(-alpha*d/2)

which rearranges to the update rules used in the iteration below:

    n     = 1 - (c/(omega*d)) * (phi - arg F(ntilde))
    alpha = (2/d) * ln(|F(ntilde)| / |H|)
    kappa = c*alpha / (4*pi*f)

Algorithm
---------
1. Compute an initial estimate ``n(0)`` exactly as in
   :mod:`thz_tds.refractive` (real Fresnel factor), then
   ``alpha(0) = (2/d) * ln[4*n(0) / ((1+n(0))**2 * |H|)]`` and
   ``kappa(0) = c*alpha(0) / (4*pi*f)``.
2. At each iteration, build ``ntilde = n - i*kappa``, evaluate the complex
   Fresnel factor ``F(ntilde)``, and use its magnitude/phase to update ``n``
   and ``alpha`` via the formulas above.
3. Apply optional relaxation ``n <- (1-lambda)*n + lambda*n_new`` (and
   likewise for alpha) to stabilise the fixed point.
4. Stop when both ``max|n_next - n_current|`` and
   ``max|alpha_next - alpha_current|`` (evaluated over the reliable
   frequency band) fall below the requested tolerances, or after
   ``max_iter`` iterations (in which case a ``RuntimeWarning`` is issued).

All internal absorption values are kept in m^-1; the returned ``alpha_cm``
is converted to cm^-1 only at the very end, matching ``refractive.py``.

Phase unwrapping and frequency-band handling are copied unmodified from
``refractive.py`` — the complex-Fresnel correction only changes how ``n``
and ``alpha`` are derived from the (already unwrapped) phase and magnitude
of ``T``, so there is no mathematical reason to change that part.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np

from .spectral import fft_field, ensure_common_time_axis

# Speed of light in vacuum [m/s]
_C0 = 2.99792458e8

_ResultNoDiag = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
_ResultWithDiag = Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]
]


def compute_refractive_index(
    ref_ds,
    sam_ds,
    thickness_m: float,
    f_low_THz: float = 0.4,
    f_high_THz: float = 1.1,
    max_iter: int = 100,
    tol_n: float = 1e-8,
    tol_alpha: float = 1e-6,
    relaxation: float = 1.0,
    clip_negative_alpha: bool = False,
    return_diagnostics: bool = False,
) -> Union[_ResultNoDiag, _ResultWithDiag]:
    """
    Extract refractive index and absorption coefficient from two
    :class:`~thz_tds.dataset.THZDataset` objects, using the iterative
    complex-Fresnel method.

    Parameters
    ----------
    ref_ds, sam_ds : THZDataset
        Reference (air) and sample datasets.  Must have ``x_ps``/``y_avg``.
    thickness_m : float
        Sample thickness in metres. Must be > 0.
    f_low_THz, f_high_THz : float
        Reliable-band bounds (THz) used for phase unwrapping. ``f_low_THz``
        must be < ``f_high_THz``.
    max_iter : int
        Maximum number of fixed-point iterations. Must be > 0.
    tol_n : float
        Convergence tolerance on ``n`` (dimensionless). Must be > 0.
    tol_alpha : float
        Convergence tolerance on alpha, in m^-1. Must be > 0.
    relaxation : float
        Relaxation factor ``lambda`` in ``(0, 1]``; ``1.0`` = no relaxation.
    clip_negative_alpha : bool
        If True, clip negative alpha values to zero at every iteration.
        Default False — negative absorption can legitimately arise from
        measurement noise and should not be silently discarded unless
        explicitly requested.
    return_diagnostics : bool
        If True, return a 6th element: a diagnostics dictionary.

    Returns
    -------
    f_THz, n_f, T, phi_full, alpha_cm [, diagnostics]
        Same first five outputs as :func:`thz_tds.refractive.compute_refractive_index`.
        ``n_f`` is the converged real part of the refractive index.
    """
    _, y_ref, y_sam = ensure_common_time_axis(
        ref_ds.x_ps, ref_ds.y_avg, sam_ds.x_ps, sam_ds.y_avg
    )
    return compute_refractive_index_arrays(
        y_ref=y_ref,
        y_sam=y_sam,
        time_ps=ref_ds.x_ps,
        thickness_m=thickness_m,
        f_low_THz=f_low_THz,
        f_high_THz=f_high_THz,
        max_iter=max_iter,
        tol_n=tol_n,
        tol_alpha=tol_alpha,
        relaxation=relaxation,
        clip_negative_alpha=clip_negative_alpha,
        return_diagnostics=return_diagnostics,
    )


def compute_refractive_index_arrays(
    time_ps: np.ndarray,
    y_ref: np.ndarray,
    y_sam: np.ndarray,
    thickness_m: float,
    f_low_THz: float = 0.4,
    f_high_THz: float = 1.1,
    max_iter: int = 100,
    tol_n: float = 1e-8,
    tol_alpha: float = 1e-6,
    relaxation: float = 1.0,
    clip_negative_alpha: bool = False,
    return_diagnostics: bool = False,
) -> Union[_ResultNoDiag, _ResultWithDiag]:
    """
    Extract refractive index and absorption coefficient from raw arrays
    using the iterative complex-Fresnel method.  See
    :func:`compute_refractive_index` for parameter documentation, and the
    module docstring for the sign convention and algorithm.

    Raises
    ------
    ValueError
        If any input parameter fails validation.
    RuntimeError
        If no frequencies fall within the specified good band.
    """
    # ---- input validation ------------------------------------------------
    if not (np.isfinite(thickness_m) and thickness_m > 0):
        raise ValueError(f"thickness_m must be > 0, got {thickness_m!r}")
    if not (isinstance(max_iter, (int, np.integer)) and max_iter > 0):
        raise ValueError(f"max_iter must be a positive integer, got {max_iter!r}")
    if not (np.isfinite(tol_n) and tol_n > 0):
        raise ValueError(f"tol_n must be > 0, got {tol_n!r}")
    if not (np.isfinite(tol_alpha) and tol_alpha > 0):
        raise ValueError(f"tol_alpha must be > 0, got {tol_alpha!r}")
    if not (0 < relaxation <= 1):
        raise ValueError(f"relaxation must be in (0, 1], got {relaxation!r}")
    if not (f_low_THz < f_high_THz):
        raise ValueError(
            f"f_low_THz must be < f_high_THz, got f_low_THz={f_low_THz!r}, "
            f"f_high_THz={f_high_THz!r}"
        )

    # ---- 1: FFT on the common time grid (no padding, phase-preserving) ---
    f_Hz, E_ref = fft_field(time_ps, y_ref, pad_factor=1)
    _, E_sam = fft_field(time_ps, y_sam, pad_factor=1)

    # Remove DC bin to avoid division by zero
    f_Hz = f_Hz[1:]
    E_ref = E_ref[1:]
    E_sam = E_sam[1:]
    f_THz = f_Hz / 1e12

    # ---- 2: transfer function ---------------------------------------------
    T = E_sam / E_ref
    phi_wrapped = np.angle(T)

    # ---- 3: phase unwrapping in the reliable band (unchanged from refractive.py) ---
    band_good = (f_THz >= f_low_THz) & (f_THz <= f_high_THz)
    if not np.any(band_good):
        raise RuntimeError(
            f"No frequencies fall within the good band "
            f"[{f_low_THz}, {f_high_THz}] THz.  "
            "Check f_low_THz and f_high_THz settings."
        )

    phi_good_unwrapped = np.unwrap(phi_wrapped[band_good])

    f_good_Hz = f_Hz[band_good]
    p = np.polyfit(f_good_Hz, phi_good_unwrapped, 1)

    phi_full = np.empty_like(phi_wrapped)
    phi_full[:] = np.nan
    phi_full[band_good] = phi_good_unwrapped

    band_low = f_THz < f_low_THz
    if np.any(band_low):
        phi_full[band_low] = np.polyval(p, f_Hz[band_low])

    nan_mask = np.isnan(phi_full)
    if np.any(nan_mask):
        phi_full[nan_mask] = phi_wrapped[nan_mask]

    phi_full -= phi_full[0]

    omega = 2.0 * np.pi * f_Hz
    T_mag = np.maximum(np.abs(T), 1e-15)

    # ---- 4: initial estimate (real-Fresnel), matching refractive.py ------
    n0 = 1.0 - _C0 * phi_full / (omega * thickness_m)
    n0_safe = np.maximum(n0, 1e-6)
    alpha0 = (2.0 / thickness_m) * np.log(
        (4.0 * n0_safe) / (((1.0 + n0_safe) ** 2) * T_mag + 1e-300)
    )
    if clip_negative_alpha:
        alpha0 = np.maximum(alpha0, 0.0)

    # ---- 5: fixed-point iteration on the complex Fresnel factor ----------
    n_current = n0.copy()
    alpha_current = alpha0.copy()

    converged = False
    iterations_run = 0
    delta_n = np.nan
    delta_alpha = np.nan

    for iteration in range(1, max_iter + 1):
        iterations_run = iteration

        kappa_current = _C0 * alpha_current / (4.0 * np.pi * f_Hz)
        n_tilde = n_current - 1j * kappa_current

        denom = (1.0 + n_tilde) ** 2
        denom_mag = np.abs(denom)
        denom_floor = 1e-300
        denom_safe = np.where(
            denom_mag < denom_floor,
            denom_floor * np.exp(1j * np.angle(denom)),
            denom,
        )
        F = 4.0 * n_tilde / denom_safe
        F_mag = np.maximum(np.abs(F), 1e-300)
        F_phase = np.angle(F)

        n_new = 1.0 - (_C0 / (omega * thickness_m)) * (phi_full - F_phase)
        alpha_new = (2.0 / thickness_m) * np.log(F_mag / T_mag)
        if clip_negative_alpha:
            alpha_new = np.maximum(alpha_new, 0.0)

        n_next = (1.0 - relaxation) * n_current + relaxation * n_new
        alpha_next = (1.0 - relaxation) * alpha_current + relaxation * alpha_new

        valid = (
            band_good
            & np.isfinite(n_current) & np.isfinite(n_next)
            & np.isfinite(alpha_current) & np.isfinite(alpha_next)
        )
        if np.any(valid):
            delta_n = float(np.nanmax(np.abs(n_next[valid] - n_current[valid])))
            delta_alpha = float(
                np.nanmax(np.abs(alpha_next[valid] - alpha_current[valid]))
            )
        else:
            delta_n = np.nan
            delta_alpha = np.nan

        n_current, alpha_current = n_next, alpha_next

        if np.isfinite(delta_n) and np.isfinite(delta_alpha):
            if delta_n < tol_n and delta_alpha < tol_alpha:
                converged = True
                break

    if not converged:
        warnings.warn(
            f"compute_refractive_index_arrays: complex-Fresnel iteration did "
            f"not converge within max_iter={max_iter} iterations "
            f"(final delta_n={delta_n:.3e}, final delta_alpha={delta_alpha:.3e} m^-1). "
            f"Results may be inaccurate; consider raising max_iter, loosening "
            f"tolerances, or reducing 'relaxation'.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ---- 6: final diagnostics quantities (recompute F at the converged point) ---
    kappa_final = _C0 * alpha_current / (4.0 * np.pi * f_Hz)
    n_tilde_final = n_current - 1j * kappa_final
    denom_final = (1.0 + n_tilde_final) ** 2
    denom_final_mag = np.abs(denom_final)
    denom_final_safe = np.where(
        denom_final_mag < 1e-300,
        1e-300 * np.exp(1j * np.angle(denom_final)),
        denom_final,
    )
    F_final = 4.0 * n_tilde_final / denom_final_safe
    F_mag_final = np.abs(F_final)
    F_phase_final = np.angle(F_final)

    alpha_cm = alpha_current / 100.0  # m^-1 -> cm^-1

    if not return_diagnostics:
        return f_THz, n_current, T, phi_full, alpha_cm

    diagnostics: Dict[str, object] = {
        "converged": converged,
        "iterations": iterations_run,
        "delta_n": delta_n,
        "delta_alpha_m": delta_alpha,
        "kappa": kappa_final,
        "fresnel_magnitude": F_mag_final,
        "fresnel_phase": F_phase_final,
    }
    return f_THz, n_current, T, phi_full, alpha_cm, diagnostics


# ---------------------------------------------------------------------------
# Self-test: prove that the chosen sign convention (ntilde = n - i*kappa)
# gives attenuation, not gain, for kappa > 0.
# ---------------------------------------------------------------------------

def _check_attenuation_sign() -> None:
    """
    Synthetic sanity check (no I/O, no plotting): build a slab transfer
    function using this module's sign convention, ntilde = n - i*kappa,
    and verify that increasing kappa (> 0) strictly *decreases* |H| (i.e.
    attenuates), never increases it.

    Raises
    ------
    AssertionError
        If the convention does not produce attenuation for kappa > 0.
    """
    f_Hz = np.linspace(0.2e12, 1.2e12, 50)
    omega = 2.0 * np.pi * f_Hz
    d = 3.0e-3
    n = 1.8

    def _H_mag(kappa: np.ndarray) -> np.ndarray:
        n_tilde = n - 1j * kappa
        F = 4.0 * n_tilde / (1.0 + n_tilde) ** 2
        prop = np.exp(-1j * (omega * d / _C0) * (n_tilde - 1.0))
        return np.abs(F * prop)

    H_mag_zero = _H_mag(np.zeros_like(f_Hz))
    H_mag_pos = _H_mag(np.full_like(f_Hz, 5.0))  # kappa > 0

    if not np.all(H_mag_pos < H_mag_zero):
        raise AssertionError(
            "Sign convention check failed: positive kappa did not attenuate "
            "|H| relative to kappa=0 at all test frequencies — the sign of "
            "ntilde in this module is inconsistent with the project's FFT "
            "and phase conventions."
        )

    # Also confirm alpha = 2*omega*kappa/c reproduces exp(-alpha*d/2)
    kappa_test = np.full_like(f_Hz, 3.0)
    alpha_test = 2.0 * omega * kappa_test / _C0
    n_tilde = n - 1j * kappa_test
    prop = np.exp(-1j * (omega * d / _C0) * (n_tilde - 1.0))
    expected_amplitude = np.exp(-alpha_test * d / 2.0)
    if not np.allclose(np.abs(prop), expected_amplitude, rtol=1e-10):
        raise AssertionError(
            "Sign convention check failed: |exp[-i*omega*d/c*(ntilde-1)]| "
            "does not match exp(-alpha*d/2) for ntilde = n - i*kappa."
        )


if __name__ == "__main__":
    _check_attenuation_sign()
    print("Sign convention self-test passed: ntilde = n - i*kappa attenuates "
          "for kappa > 0, consistent with this project's FFT/phase convention.")
