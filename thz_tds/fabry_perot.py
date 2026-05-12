"""
thz_tds.fabry_perot
===================
Fabry-Perot (FP) echo removal using the Liu et al. time-domain model.

Workflow
--------
1. Call :func:`remove_fabry_perot` (high-level convenience function), or:
2. Call :func:`liu_fit_over_thickness_grid` to find the best-fit thickness,
   average refractive index, attenuation, and distortion parameters.
3. Subtract the fitted FP echoes from the measured sample signal.
4. Call :func:`extract_optical_constants` on the cleaned signal to obtain
   n(f) and α(f).

References
----------
Liu, H. B. et al. (2006). Identification and characterization of Fabry-Perot
resonances in terahertz time-domain spectroscopy. *Applied Physics Letters*.

Withayachumnankul, W. & Naftaly, M. (2014). Fundamentals of measurement in
terahertz time-domain spectroscopy. *Journal of Infrared, Millimeter, and
Terahertz Waves*, 35(8), 610–637.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from .spectral import fft_field

log = logging.getLogger(__name__)

_C0 = 299_792_458.0  # speed of light in vacuum [m/s]


# ---------------------------------------------------------------------------
# Public data containers
# ---------------------------------------------------------------------------

@dataclass
class THzTrace:
    """
    A single THz time-domain trace.

    Parameters
    ----------
    time_ps : np.ndarray
        Time axis in picoseconds.
    y : np.ndarray
        Electric-field amplitude samples.
    """
    time_ps: np.ndarray
    y: np.ndarray


@dataclass
class LiuFitResult:
    """
    Result of a Liu model grid-search fit.

    Attributes
    ----------
    thickness_m : float
        Best-fit sample thickness in metres.
    nav : float
        Best-fit average refractive index.
    gamma : float
        Best-fit attenuation parameter.
    delta : float
        Best-fit temporal distortion parameter.
    mse : float
        Mean-squared time-domain error of the best fit.
    r2 : float
        Coefficient of determination (R²) of the best fit.
    echo_count : int
        Number of FP echoes included in the model.
    y_sim : np.ndarray
        Simulated sample signal from the best-fit parameters.
    y_fp : np.ndarray
        Simulated FP echo contribution only (to be subtracted).
    y_clean : np.ndarray
        FP-cleaned sample signal (``y_sam − y_fp``).
    """
    thickness_m: float
    nav: float
    gamma: float
    delta: float
    mse: float
    r2: float
    echo_count: int
    y_sim: np.ndarray
    y_fp: np.ndarray
    y_clean: np.ndarray


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def liu_fit_over_thickness_grid(
    ref: THzTrace,
    sam: THzTrace,
    thickness_min_m: float,
    thickness_max_m: float,
    thickness_step_m: float,
    n_air: float = 1.0,
    echo_count: int = 3,
) -> LiuFitResult:
    """
    Scan a thickness grid, optimise nav/gamma/delta at each point, and
    return the result with the lowest time-domain error.

    Parameters
    ----------
    ref : THzTrace
        Reference (air) trace.
    sam : THzTrace
        Measured sample trace.
    thickness_min_m : float
        Minimum sample thickness to test [m].
    thickness_max_m : float
        Maximum sample thickness to test [m].
    thickness_step_m : float
        Thickness step [m] (e.g. ``1e-6`` for 1 µm resolution).
    n_air : float
        Refractive index of air (default 1.0).
    echo_count : int
        Number of FP echoes included in the model.

    Returns
    -------
    LiuFitResult
        Best-fit result including the cleaned signal.
    """
    t_ps, y_ref, y_sam = _ensure_same_time_axis(ref, sam)

    thickness_values = np.arange(
        thickness_min_m,
        thickness_max_m + 0.5 * thickness_step_m,
        thickness_step_m,
    )

    best: Optional[Dict] = None
    x0: Optional[np.ndarray] = None

    for L in thickness_values:
        out = _liu_fit_for_fixed_thickness(
            t_ps=t_ps,
            y_ref=y_ref,
            y_sam=y_sam,
            thickness_m=L,
            n_air=n_air,
            echo_count=echo_count,
            x0=x0,
        )
        # Warm-start the next iteration from this solution
        x0 = np.array([out["nav"], out["gamma"], out["delta"]], dtype=float)

        if best is None or out["mse"] < best["mse"]:
            best = out

    y_clean = y_sam - best["y_fp"]

    return LiuFitResult(
        thickness_m=best["thickness_m"],
        nav=best["nav"],
        gamma=best["gamma"],
        delta=best["delta"],
        mse=best["mse"],
        r2=best["r2"],
        echo_count=echo_count,
        y_sim=best["y_sim"],
        y_fp=best["y_fp"],
        y_clean=y_clean,
    )


def extract_optical_constants(
    ref: THzTrace,
    clean_sample: THzTrace,
    thickness_m: float,
    n_air: float = 1.0,
    f_min_THz: float = 0.2,
    f_max_THz: float = 2.0,
) -> Dict[str, np.ndarray]:
    """
    Extract n(f) and α(f) from a FP-cleaned sample signal via the standard
    transfer-function approach (analogous to Withayachumnankul & Naftaly
    Eqs. 5–6 applied to the FP-free signal).

    Parameters
    ----------
    ref : THzTrace
        Reference (air) trace.
    clean_sample : THzTrace
        FP-cleaned sample trace (i.e. ``y_sam − y_fp``).
    thickness_m : float
        Sample thickness in metres.
    n_air : float
        Refractive index of air (default 1.0).
    f_min_THz : float
        Lower frequency bound of the output (THz).
    f_max_THz : float
        Upper frequency bound of the output (THz).

    Returns
    -------
    dict with keys:
        ``f_THz``, ``H`` (complex transfer function),
        ``n`` (refractive index), ``alpha_1_per_m``, ``alpha_1_per_cm``
    """
    t_ps, y_ref, y_sam_clean = _ensure_same_time_axis(ref, clean_sample)

    f_Hz, E_ref = fft_field(t_ps, y_ref, pad_factor=1)
    _, E_sam = fft_field(t_ps, y_sam_clean, pad_factor=1)

    # Remove DC
    f_Hz = f_Hz[1:]
    E_ref = E_ref[1:]
    E_sam = E_sam[1:]

    H = E_sam / (E_ref + 1e-30)

    omega = 2.0 * np.pi * f_Hz
    mag_H = np.abs(H)
    phase = np.unwrap(np.angle(H))

    n = n_air - (_C0 / (omega * thickness_m)) * phase
    alpha = (2.0 / thickness_m) * np.log(
        (4.0 * n) / (mag_H * (1.0 + n) ** 2 + 1e-30)
    )

    f_THz = f_Hz * 1e-12
    mask = (f_THz >= f_min_THz) & (f_THz <= f_max_THz)

    return {
        "f_THz": f_THz[mask],
        "H": H[mask],
        "n": np.real(n[mask]),
        "alpha_1_per_m": np.real(alpha[mask]),
        "alpha_1_per_cm": np.real(alpha[mask]) / 100.0,
    }


def remove_fabry_perot(
    ref: THzTrace,
    sam: THzTrace,
    thickness_min_m: float,
    thickness_max_m: float,
    thickness_step_m: float,
    n_air: float = 1.0,
    echo_count: int = 3,
    f_min_THz: float = 0.2,
    f_max_THz: float = 2.0,
) -> Dict[str, object]:
    """
    High-level convenience: run the Liu grid search and optical-constant
    extraction in a single call.

    Parameters
    ----------
    ref : THzTrace
        Reference trace.
    sam : THzTrace
        Measured sample trace.
    thickness_min_m, thickness_max_m, thickness_step_m : float
        Thickness grid parameters [m].
    n_air : float
        Refractive index of air.
    echo_count : int
        Number of FP echoes in the model.
    f_min_THz, f_max_THz : float
        Frequency range for optical-constant extraction.

    Returns
    -------
    dict with keys:
        ``fit`` (:class:`LiuFitResult`),
        ``clean_trace`` (:class:`THzTrace`),
        ``spectrum`` (dict from :func:`extract_optical_constants`)
    """
    fit = liu_fit_over_thickness_grid(
        ref=ref,
        sam=sam,
        thickness_min_m=thickness_min_m,
        thickness_max_m=thickness_max_m,
        thickness_step_m=thickness_step_m,
        n_air=n_air,
        echo_count=echo_count,
    )

    clean_trace = THzTrace(time_ps=np.asarray(ref.time_ps), y=fit.y_clean)

    spec = extract_optical_constants(
        ref=ref,
        clean_sample=clean_trace,
        thickness_m=fit.thickness_m,
        n_air=n_air,
        f_min_THz=f_min_THz,
        f_max_THz=f_max_THz,
    )

    return {"fit": fit, "clean_trace": clean_trace, "spectrum": spec}


# ---------------------------------------------------------------------------
# Private implementation helpers
# ---------------------------------------------------------------------------

def _ensure_same_time_axis(
    ref: THzTrace, sam: THzTrace
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a common time axis and both signals on that axis."""
    t_ref = np.asarray(ref.time_ps, dtype=float)
    y_ref = np.asarray(ref.y, dtype=float)
    t_sam = np.asarray(sam.time_ps, dtype=float)
    y_sam = np.asarray(sam.y, dtype=float)

    if len(t_ref) != len(t_sam) or not np.allclose(t_ref, t_sam, atol=1e-9):
        y_sam = np.interp(t_ref, t_sam, y_sam, left=0.0, right=0.0)

    return t_ref, y_ref, y_sam


def _shift_and_scale_time(
    t_ps: np.ndarray,
    y_ref: np.ndarray,
    delay_ps: float,
    delta: float,
) -> np.ndarray:
    """
    Implements the Liu distortion: E_ref( (t − delay) / delta ).

    delta > 1 broadens; delta < 1 compresses the pulse in time.
    """
    tau = (t_ps - delay_ps) / delta
    return np.interp(tau, t_ps, y_ref, left=0.0, right=0.0)


def _find_main_peak_time_ps(y: np.ndarray, t_ps: np.ndarray) -> float:
    return float(t_ps[int(np.argmax(np.abs(y)))])


def _find_main_peak_amp(y: np.ndarray) -> float:
    return float(np.max(np.abs(y)))


def _estimate_width_ps(y: np.ndarray, t_ps: np.ndarray) -> float:
    """
    Heuristic pulse-width estimate: distance from the maximum-|E| point to the
    nearest opposite-sign extremum within ±200 samples.
    """
    idx0 = int(np.argmax(np.abs(y)))
    sign0 = np.sign(y[idx0]) if y[idx0] != 0 else 1.0

    left = max(0, idx0 - 200)
    right = min(len(y), idx0 + 200)
    segment = y[left:right]
    seg_t = t_ps[left:right]

    idx_opp = int(np.argmin(segment) if sign0 >= 0 else np.argmax(segment))
    return max(float(abs(seg_t[idx_opp] - t_ps[idx0])), 1e-6)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def _liu_simulate(
    t_ps: np.ndarray,
    y_ref: np.ndarray,
    thickness_m: float,
    nav: float,
    gamma: float,
    delta: float,
    n_air: float = 1.0,
    echo_count: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate the sample signal using the Liu time-domain model.

    Returns ``(y_sim, y_fp)`` where ``y_fp`` is the FP echo contribution only.
    """
    delay_main_ps = ((nav - n_air) * thickness_m / _C0) * 1e12
    trans = 4.0 * nav / (nav + 1.0) ** 2

    y_main = trans * gamma * _shift_and_scale_time(t_ps, y_ref, delay_main_ps, delta)

    refl = (nav - 1.0) / (nav + 1.0)
    y_fp = np.zeros_like(y_ref, dtype=float)

    for m in range(1, echo_count + 1):
        extra_ps = (nav * 2.0 * thickness_m * m / _C0) * 1e12
        total_delay_ps = delay_main_ps + extra_ps
        amp = (refl ** (2 * m)) * (gamma ** (2 * m + 1))
        delta_m = delta ** (2 * m + 1)
        y_fp += trans * amp * _shift_and_scale_time(t_ps, y_ref, total_delay_ps, delta_m)

    return y_main + y_fp, y_fp


def _liu_initial_guesses(
    t_ps: np.ndarray,
    y_ref: np.ndarray,
    y_sam: np.ndarray,
    thickness_m: float,
    n_air: float = 1.0,
) -> np.ndarray:
    """Return initial [nav, gamma, delta] guess based on time-delay and amplitude ratio."""
    t_ref_peak = _find_main_peak_time_ps(y_ref, t_ps)
    t_sam_peak = _find_main_peak_time_ps(y_sam, t_ps)
    t_delay_s = (t_sam_peak - t_ref_peak) * 1e-12

    nav0 = float(np.clip(n_air + _C0 * t_delay_s / thickness_m, 1.01, 10.0))

    a_ref = _find_main_peak_amp(y_ref)
    a_sam = _find_main_peak_amp(y_sam)
    gamma0 = float(np.clip(
        (a_sam / max(a_ref, 1e-12)) * ((nav0 + 1.0) ** 2 / (4.0 * nav0)),
        0.01, 3.0,
    ))

    w_ref = _estimate_width_ps(y_ref, t_ps)
    w_sam = _estimate_width_ps(y_sam, t_ps)
    delta0 = float(np.clip(w_ref / max(w_sam, 1e-12), 0.5, 2.0))

    return np.array([nav0, gamma0, delta0], dtype=float)


def _liu_fit_for_fixed_thickness(
    t_ps: np.ndarray,
    y_ref: np.ndarray,
    y_sam: np.ndarray,
    thickness_m: float,
    n_air: float = 1.0,
    echo_count: int = 3,
    x0: Optional[np.ndarray] = None,
) -> Dict:
    """Optimise nav, gamma, delta for one fixed thickness using Nelder-Mead."""
    if x0 is None:
        x0 = _liu_initial_guesses(t_ps, y_ref, y_sam, thickness_m, n_air=n_air)

    def objective(x: np.ndarray) -> float:
        nav, gamma, delta = x
        if nav <= 1.0 or gamma <= 0.0 or delta <= 0.0 or nav > 20.0 or gamma > 10.0 or delta > 5.0:
            return 1e30
        y_sim, _ = _liu_simulate(t_ps, y_ref, thickness_m, nav, gamma, delta, n_air, echo_count)
        return float(np.mean((y_sim - y_sam) ** 2))

    res = minimize(
        objective,
        x0=x0,
        method="Nelder-Mead",
        options={"maxiter": 3000, "xatol": 1e-8, "fatol": 1e-12},
    )

    nav, gamma, delta = res.x
    y_sim, y_fp = _liu_simulate(t_ps, y_ref, thickness_m, nav, gamma, delta, n_air, echo_count)
    mse = float(np.mean((y_sim - y_sam) ** 2))
    r2 = _r2_score(y_sam, y_sim)

    return {
        "thickness_m": thickness_m,
        "nav": float(nav),
        "gamma": float(gamma),
        "delta": float(delta),
        "mse": mse,
        "r2": r2,
        "y_sim": y_sim,
        "y_fp": y_fp,
    }
