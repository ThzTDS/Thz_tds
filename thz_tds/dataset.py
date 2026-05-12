"""
thz_tds.dataset
===============
:class:`THZDataset` — the central data container for a single THz-TDS measurement
folder — and :func:`scan_samples` for batch loading of multi-sample experiments.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .spectral import fft_field, power_spectrum_dB

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private file helpers
# ---------------------------------------------------------------------------

def _list_files(folder: str | Path) -> List[str]:
    """Return filenames, not full paths, of all regular files in *folder*."""
    folder = str(folder)
    try:
        return [
            f for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
        ]
    except OSError:
        return []


# ---------------------------------------------------------------------------
# THZDataset
# ---------------------------------------------------------------------------

class THZDataset:
    """
    Load, optionally align, and average all THz traces in a folder.

    Parameters
    ----------
    path : str or Path
        Folder containing tab-separated THz trace files.

    correction_factor : float
        Multiplicative scaling applied to the raw amplitude column.
        Set this to the instrument-specific conversion factor
        e.g. ``0.0414`` for nA output.

    align : bool
        If ``True``, apply jitter correction before averaging.

    align_method : str
        Algorithm used for alignment. One of:

        * ``"correlation"``
          Integer-sample cross-correlation to the first trace.

        * ``"correlation_subsample"``
          Cross-correlation to the first trace with sub-sample refinement
          via parabolic interpolation. Recommended default.

        * ``"gaussian_minimum_mean"``
          Fit a Gaussian around the negative minimum of each trace, compute
          the mean fitted minimum time, and align all traces to this mean.
          This is the method suggested by your friend.

        * ``"gaussian_median_integer"``
          Fit a Gaussian to the strongest peak of each trace, use the
          median of fitted centers as reference, and shift by an integer
          number of samples (no interpolation).  Robust to outlier traces;
          avoids interpolation artefacts at the cost of ≤ 0.5-sample
          residual misalignment.

    corr_window_ps : float or None
        Width in ps of the cross-correlation window centred on the strongest
        peak. ``None`` uses the full trace.

    fit_method : str or None
        Optional fit applied to the averaged trace after loading.
        Currently only ``"gaussian"`` is supported, or ``None`` to skip.

    fit_window_ps : float
        Legacy alias for *fallback_fit_window_ps*.  Ignored when
        *fallback_fit_window_ps* is given explicitly.

    adaptive_fit_window : bool
        When ``True`` (default) the Gaussian fitting window is determined by
        the estimated pulse width: ``x0_ini ± fit_window_sigma_factor *
        sigma_ini``.  When ``False``, or when the sigma estimate fails, the
        fixed *fallback_fit_window_ps* half-width is used instead.

    fit_window_sigma_factor : float
        Multiplier applied to the estimated sigma to define the adaptive
        half-width.  For example ``5.0`` gives ``x0_ini ± 5 σ``.

    fallback_fit_window_ps : float or None
        Half-width in ps of the fixed fitting window used when the adaptive
        window is disabled or its sigma estimate fails.  Defaults to the
        value of *fit_window_ps* for backward compatibility.

    min_sigma_ps : float
        Minimum sigma [ps] accepted for adaptive window calculation.

    max_sigma_ps : float
        Maximum sigma [ps] accepted for adaptive window calculation.

    min_fit_points : int
        Minimum number of samples required inside the adaptive window.
        Falls back to the fixed window if fewer points would be included.

    max_center_shift_ps : float
        Maximum allowed shift of the fitted Gaussian centre from the initial
        estimate, used as the ``t0`` bound in ``curve_fit``.

    Attributes
    ----------
    x_ps : np.ndarray
        Common time axis in picoseconds.

    all_y_raw : np.ndarray, shape (M, N)
        Matrix of all raw individual traces before alignment.

    all_y : np.ndarray, shape (M, N)
        Matrix of all aligned individual traces.

    y_avg : np.ndarray, shape (N,)
        Average of all traces.

    jitter_shifts_ps : np.ndarray or None
        Time shifts applied to each trace in ps. ``None`` if alignment was
        disabled.

    t_min_ref : float or None
        Alignment reference time. For ``gaussian_minimum_mean`` this is the
        mean fitted minimum time.

    fit_result : dict or None
        Result of the optional Gaussian fit on the averaged trace.

    freq_THz : np.ndarray or None
        Frequency axis set by :meth:`compute_fft`.

    spectrum_dB : np.ndarray or None
        Power spectrum in dB set by :meth:`compute_fft`.

    spectrum_complex : np.ndarray or None
        Complex spectrum set by :meth:`compute_fft`.
    """

    def __init__(
        self,
        path: str | Path,
        correction_factor: float = 0.0414,
        align: bool = False,
        align_method: str = "correlation_subsample",
        corr_window_ps: Optional[float] = None,
        fit_method: Optional[str] = None,
        fit_window_ps: float = 15.0,
        # Adaptive Gaussian fitting window parameters
        adaptive_fit_window: bool = True,
        fit_window_sigma_factor: float = 5.0,
        fallback_fit_window_ps: Optional[float] = None,
        min_sigma_ps: float = 0.05,
        max_sigma_ps: float = 10.0,
        min_fit_points: int = 10,
        max_center_shift_ps: float = 5.0,
        max_traces: Optional[int] = None,
    ) -> None:
        self.path = str(path)
        self.name = os.path.basename(self.path.rstrip("/\\"))
        self.correction_factor = float(correction_factor)
        self.do_align = bool(align)
        self.align_method = align_method.lower()
        self.corr_window_ps = corr_window_ps
        self.fit_method = None if fit_method is None else fit_method.lower()

        # fallback_fit_window_ps takes priority; fall back to fit_window_ps for
        # backward compatibility with callers that still use the old parameter name.
        self.fit_window_ps = float(
            fallback_fit_window_ps
            if fallback_fit_window_ps is not None
            else fit_window_ps
        )
        self.fallback_fit_window_ps = self.fit_window_ps

        # Adaptive Gaussian fitting window parameters
        self.adaptive_fit_window = bool(adaptive_fit_window)
        self.fit_window_sigma_factor = float(fit_window_sigma_factor)
        self.min_sigma_ps = float(min_sigma_ps)
        self.max_sigma_ps = float(max_sigma_ps)
        self.min_fit_points = int(min_fit_points)
        self.max_center_shift_ps = float(max_center_shift_ps)
        self.max_traces = None if max_traces is None else int(max_traces)

        # Populated by load()
        self.x_ps: Optional[np.ndarray] = None
        self.all_y_raw: Optional[np.ndarray] = None
        self.all_y: Optional[np.ndarray] = None
        self.y_avg: Optional[np.ndarray] = None

        # Alignment diagnostics
        self.jitter_shifts_ps: Optional[np.ndarray] = None
        self.t_min_ref: Optional[float] = None
        self.gaussian_minimum_fits: Optional[List[Dict]] = None

        # Fit diagnostics
        self.fit_result: Optional[Dict] = None

        # FFT results
        self.freq_THz: Optional[np.ndarray] = None
        self.spectrum_dB: Optional[np.ndarray] = None
        self.spectrum_complex: Optional[np.ndarray] = None

        self.load()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Re-load all traces from :attr:`path`, apply optional alignment,
        and compute the average.
        """
        files = _list_files(self.path)
        if not files:
            raise RuntimeError(f"No files found in {self.path!r}")
        if self.max_traces is not None:
            files = sorted(files)[: self.max_traces]

        all_y: List[np.ndarray] = []
        x_ref: Optional[np.ndarray] = None

        for fname in files:
            full = os.path.join(self.path, fname)

            try:
                df = pd.read_csv(full, sep="\t", header=0, comment="#")
            except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                log.warning("Skipping %s: %s", fname, exc)
                continue

            if df.shape[1] < 2:
                log.warning("Skipping %s: fewer than 2 columns.", fname)
                continue

            try:
                x = df.iloc[:, 0].to_numpy(dtype=float)
                y = df.iloc[:, 1].to_numpy(dtype=float) * self.correction_factor
            except (TypeError, ValueError) as exc:
                log.warning("Skipping %s: %s", fname, exc)
                continue

            if x_ref is None:
                x_ref = x
            elif len(x) != len(x_ref) or not np.allclose(x, x_ref, atol=1e-9):
                y = np.interp(x_ref, x, y, left=0.0, right=0.0)

            all_y.append(y)

        if x_ref is None or not all_y:
            raise RuntimeError(f"No readable THz files found in {self.path!r}")

        all_y_arr = np.vstack(all_y)
        self.all_y_raw = all_y_arr.copy()

        if self.do_align:
            all_y_arr, shifts_ps, t_min_ref = self._run_alignment(x_ref, all_y_arr)
            self.jitter_shifts_ps = shifts_ps
            self.t_min_ref = t_min_ref
        else:
            self.jitter_shifts_ps = None
            self.t_min_ref = None
            self.gaussian_minimum_fits = None

        self.x_ps = x_ref
        self.all_y = all_y_arr
        self.y_avg = np.mean(all_y_arr, axis=0)

        self.compute_fit()

    def compute_fft(self, pad_factor: int = 4):
        """
        Compute the power spectrum of the averaged trace.

        Parameters
        ----------
        pad_factor : int
            Zero-padding factor.

        Returns
        -------
        freq_THz : np.ndarray
        spectrum_dB : np.ndarray
        spectrum_complex : np.ndarray
        """
        if self.y_avg is None or self.x_ps is None:
            raise RuntimeError("Call load() before compute_fft().")

        freq_THz, spectrum_dB = power_spectrum_dB(
            self.x_ps,
            self.y_avg,
            pad_factor=pad_factor,
        )

        _, spectrum_complex = fft_field(
            self.x_ps,
            self.y_avg,
            pad_factor=pad_factor,
        )

        self.freq_THz = freq_THz
        self.spectrum_dB = spectrum_dB
        self.spectrum_complex = spectrum_complex

        return freq_THz, spectrum_dB, spectrum_complex

    def compute_fit(self) -> Optional[Dict]:
        """
        Compute an optional fit on the averaged trace.
        """
        self.fit_result = None

        if self.fit_method is None:
            return None

        if self.fit_method == "gaussian":
            if self.x_ps is None or self.y_avg is None:
                raise RuntimeError("Cannot fit before data are loaded.")
            self.fit_result = self.fit_gaussian_to_trace(self.x_ps, self.y_avg)
            return self.fit_result

        raise ValueError(
            f"Unknown fit_method={self.fit_method!r}. Use None or 'gaussian'."
        )

    # ------------------------------------------------------------------
    # Alignment methods
    # ------------------------------------------------------------------

    def align_by_correlation_to_first(
        self,
        t_ps: np.ndarray,
        all_y: np.ndarray,
        subsample: bool = False,
    ):
        """
        Align all traces to the first trace using cross-correlation.
        """
        all_y = np.asarray(all_y)
        n_scans, _ = all_y.shape

        if len(t_ps) < 2:
            raise ValueError("t_ps must have at least 2 points.")

        dt_ps = float(np.mean(np.diff(t_ps)))
        y_ref = all_y[0]

        idx_ref = int(np.argmin(y_ref))
        t_min_ref = float(t_ps[idx_ref])

        aligned = np.zeros_like(all_y)
        shifts_ps = np.zeros(n_scans)

        aligned[0] = y_ref

        mask = self._get_correlation_window_mask(t_ps, y_ref)
        ref_seg = y_ref[mask] - np.mean(y_ref[mask])

        m = len(ref_seg)
        if m < 2:
            raise ValueError("Correlation window is too short.")

        nfft = 2 * m - 1
        F_ref = np.fft.fft(ref_seg, n=nfft)
        lags = np.arange(-(m - 1), m, dtype=float)

        for i in range(1, n_scans):
            y_seg = all_y[i][mask] - np.mean(all_y[i][mask])

            Fy = np.fft.fft(y_seg, n=nfft)
            corr = np.fft.ifft(F_ref * np.conj(Fy)).real
            corr = np.concatenate((corr[-(m - 1):], corr[:m]))

            k0 = int(np.argmax(corr))
            lag_best = lags[k0]

            if subsample and 0 < k0 < len(corr) - 1:
                lag_best += self._parabolic_peak_offset(
                    corr[k0 - 1],
                    corr[k0],
                    corr[k0 + 1],
                )

            delay_ps = lag_best * dt_ps
            shifts_ps[i] = delay_ps

            aligned[i] = np.interp(
                t_ps,
                t_ps + delay_ps,
                all_y[i],
                left=0.0,
                right=0.0,
            )

        return aligned, shifts_ps, t_min_ref

    def align_by_gaussian_minimum_to_mean(
        self,
        t_ps: np.ndarray,
        all_y: np.ndarray,
    ):
        """
        Align traces by fitting a Gaussian around the negative minimum
        of each trace and aligning all fitted minimum positions to their mean.

        This follows the idea:

            1. For each trace, find the rough discrete minimum.
            2. Fit a Gaussian around that minimum.
            3. Extract the fitted centre t0_i.
            4. Compute mean(t0_i).
            5. Shift every trace so t0_i becomes mean(t0_i).

        This is useful when the main THz pulse is negative and isolated.
        """
        all_y = np.asarray(all_y)
        n_scans = all_y.shape[0]

        fitted_t0 = np.zeros(n_scans)
        fits: List[Dict] = []

        for i in range(n_scans):
            fit_i = self.fit_gaussian_to_minimum(t_ps, all_y[i])
            fitted_t0[i] = float(fit_i["t0_ps"])
            fits.append(fit_i)

        t0_mean = float(np.mean(fitted_t0))

        aligned = np.zeros_like(all_y)
        shifts_ps = np.zeros(n_scans)

        for i in range(n_scans):
            delay_ps = t0_mean - fitted_t0[i]
            shifts_ps[i] = delay_ps

            aligned[i] = np.interp(
                t_ps,
                t_ps + delay_ps,
                all_y[i],
                left=0.0,
                right=0.0,
            )

        self.gaussian_minimum_fits = fits

        return aligned, shifts_ps, t0_mean

    def align_by_gaussian_median_integer(
        self,
        t_ps: np.ndarray,
        all_y: np.ndarray,
    ):
        """
        Align traces using Gaussian-fitted peak positions, integer-sample
        shifts, and the median of all fitted centers as the reference.

        Steps:

            1. Fit a Gaussian to the strongest peak of each trace to get t0_i.
            2. Compute t0_ref = median(t0_i)  — robust to outlier traces.
            3. For each trace compute lag = round((t0_i − t0_ref) / dt).
            4. Shift by that integer number of samples using array slicing
               (no interpolation); edges are padded with zeros.

        This avoids interpolation artefacts at the cost of ≤ 0.5-sample
        residual misalignment.
        """
        all_y = np.asarray(all_y)
        n_scans, N = all_y.shape
        dt = float(np.median(np.diff(t_ps)))

        fitted_t0 = np.zeros(n_scans)
        for i in range(n_scans):
            fit_i = self.fit_gaussian_to_trace(t_ps, all_y[i])
            fitted_t0[i] = float(fit_i["t0_ps"])

        t0_ref = float(np.median(fitted_t0))

        aligned = np.zeros_like(all_y)
        shifts_ps = np.zeros(n_scans)

        for i in range(n_scans):
            lag = int(np.round((fitted_t0[i] - t0_ref) / dt))
            shifts_ps[i] = float(-lag) * dt
            y = all_y[i]
            if lag > 0:
                aligned[i, :-lag] = y[lag:]
            elif lag < 0:
                aligned[i, -lag:] = y[:lag]
            else:
                aligned[i] = y

        return aligned, shifts_ps, t0_ref

    # ------------------------------------------------------------------
    # Gaussian fitting
    # ------------------------------------------------------------------

    @staticmethod
    def gaussian_with_offset(
        t: np.ndarray,
        A: float,
        t0: float,
        sigma: float,
        C: float,
    ) -> np.ndarray:
        """
        Gaussian with a constant baseline:

            A * exp(-(t - t0)^2 / (2 sigma^2)) + C
        """
        return A * np.exp(-((t - t0) ** 2) / (2.0 * sigma ** 2)) + C

    def fit_gaussian_to_trace(
        self,
        t_ps: np.ndarray,
        y: np.ndarray,
    ) -> Dict:
        """
        Fit a Gaussian with constant offset around the strongest absolute pulse.

        Works for both positive and negative THz pulses by searching for the
        largest absolute deviation from the median baseline.

        When adaptive_fit_window is True the fitting window is determined by
        the estimated pulse width: x0_ini ± fit_window_sigma_factor * sigma_ini.
        If the sigma estimate fails, the fixed fallback_fit_window_ps is used.

        Returns
        -------
        dict with keys: A, t0_ps, sigma_ps, C, fwhm_ps, success, plus
        diagnostic keys: x0_ini_ps, A_ini, C_ini, sigma_ini_ps, fwhm_ini_ps,
        fit_window_left_ps, fit_window_right_ps, fit_window_n_points,
        adaptive_half_width_ps, fit_window_sigma_factor, used_adaptive_window.
        """
        # --- Step 1: Initial estimates on the full trace ---
        C_ini = float(np.median(y))
        idx_peak = int(np.argmax(np.abs(y - C_ini)))
        x0_ini = float(t_ps[idx_peak])
        A_ini = float(y[idx_peak]) - C_ini

        # --- Step 2: Sigma estimate from FWHM on the full trace ---
        sigma_ini, fwhm_ini = self._estimate_sigma(t_ps, y, C_ini, A_ini)

        # --- Step 3: Adaptive or fallback fit window ---
        mask, half_width, used_adaptive = self._compute_fit_window(
            t_ps, x0_ini, sigma_ini
        )
        n_pts = int(np.count_nonzero(mask))

        log.debug(
            "fit_gaussian_to_trace: x0_ini=%.4f ps  A_ini=%.4e  C_ini=%.4e  "
            "sigma_ini=%s ps  half_width=%.4f ps  n_pts=%d  adaptive=%s",
            x0_ini, A_ini, C_ini,
            f"{sigma_ini:.4f}" if sigma_ini is not None else "N/A",
            half_width, n_pts, used_adaptive,
        )

        if n_pts < 5:
            raise RuntimeError("Fit window too small for Gaussian fit.")

        t_fit = t_ps[mask]
        y_fit = y[mask]

        # --- Step 4: Re-estimate initial guesses within the window ---
        baseline = float(np.median(y_fit))
        idx_peak_w = int(np.argmax(np.abs(y_fit - baseline)))
        t0_guess = float(t_fit[idx_peak_w])
        A_guess = float(y_fit[idx_peak_w]) - baseline

        above = np.where(np.abs(y_fit - baseline) >= 0.5 * abs(A_guess))[0]
        if len(above) >= 2:
            sigma_guess = max(
                (float(t_fit[above[-1]]) - float(t_fit[above[0]])) / 2.355,
                1e-3,
            )
        elif sigma_ini is not None and sigma_ini > 0:
            sigma_guess = max(float(sigma_ini), 1e-3)
        else:
            sigma_guess = max((float(t_fit[-1]) - float(t_fit[0])) / 6.0, 1e-3)

        # --- Step 5: Bounds — allow both positive and negative amplitude ---
        # Parameter order in gaussian_with_offset: (t, A, t0, sigma, C)
        t_lo = max(x0_ini - self.max_center_shift_ps, float(t_fit[0]))
        t_hi = min(x0_ini + self.max_center_shift_ps, float(t_fit[-1]))
        lower = [-np.inf, t_lo, self.min_sigma_ps, -np.inf]
        upper = [np.inf,  t_hi, self.max_sigma_ps,  np.inf]

        popt, _ = curve_fit(
            self.gaussian_with_offset,
            t_fit,
            y_fit,
            p0=[A_guess, t0_guess, sigma_guess, baseline],
            bounds=(lower, upper),
            maxfev=50000,
            method="trf",
        )

        A, t0, sigma, C = popt
        sigma = abs(float(sigma))
        fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma

        return {
            "A": float(A),
            "t0_ps": float(t0),
            "sigma_ps": sigma,
            "C": float(C),
            "fwhm_ps": float(fwhm),
            "success": True,
            # Diagnostics
            "x0_ini_ps": x0_ini,
            "A_ini": A_ini,
            "C_ini": C_ini,
            "sigma_ini_ps": float(sigma_ini) if sigma_ini is not None else float("nan"),
            "fwhm_ini_ps": float(fwhm_ini) if fwhm_ini is not None else float("nan"),
            "fit_window_left_ps": float(t_fit[0]),
            "fit_window_right_ps": float(t_fit[-1]),
            "fit_window_n_points": n_pts,
            "adaptive_half_width_ps": half_width,
            "fit_window_sigma_factor": self.fit_window_sigma_factor,
            "used_adaptive_window": used_adaptive,
        }

    def fit_gaussian_to_minimum(
        self,
        t_ps: np.ndarray,
        y: np.ndarray,
    ) -> Dict:
        """
        Fit a Gaussian with constant offset around the negative minimum
        of a THz trace.

        Unlike fit_gaussian_to_trace this always uses the minimum sample as the
        initial centre candidate, which is correct when the main THz lobe is
        negative.  A <= 0 is enforced in the bounds so the fitted Gaussian
        always opens downward.

        Uses the same adaptive fitting window as fit_gaussian_to_trace.

        Returns
        -------
        dict with the same keys as fit_gaussian_to_trace.
        """
        # --- Step 1: Explicit minimum search ---
        idx_min = int(np.argmin(y))
        x0_ini = float(t_ps[idx_min])
        C_ini = float(np.median(y))
        A_ini = float(y[idx_min]) - C_ini  # negative for a genuine negative pulse

        # --- Step 2: Sigma estimate from FWHM on the full trace ---
        sigma_ini, fwhm_ini = self._estimate_sigma(t_ps, y, C_ini, A_ini)

        # --- Step 3: Adaptive or fallback fit window ---
        mask, half_width, used_adaptive = self._compute_fit_window(
            t_ps, x0_ini, sigma_ini
        )
        n_pts = int(np.count_nonzero(mask))

        log.debug(
            "fit_gaussian_to_minimum: x0_ini=%.4f ps  A_ini=%.4e  C_ini=%.4e  "
            "sigma_ini=%s ps  half_width=%.4f ps  n_pts=%d  adaptive=%s",
            x0_ini, A_ini, C_ini,
            f"{sigma_ini:.4f}" if sigma_ini is not None else "N/A",
            half_width, n_pts, used_adaptive,
        )

        if n_pts < 5:
            raise RuntimeError("Fit window too small for Gaussian minimum fit.")

        t_fit = t_ps[mask]
        y_fit = y[mask]

        # --- Step 4: Re-estimate initial guesses within the window ---
        baseline = float(np.median(y_fit))
        idx_min_local = int(np.argmin(y_fit))
        t0_guess = float(t_fit[idx_min_local])
        A_guess = float(y_fit[idx_min_local]) - baseline

        # Ensure the initial amplitude guess is negative (minimum fit)
        if A_guess >= 0:
            A_guess = -abs(float(np.max(y_fit) - np.min(y_fit)))
            if A_guess == 0:
                A_guess = -1e-12

        above = np.where(np.abs(y_fit - baseline) >= 0.5 * abs(A_guess))[0]
        if len(above) >= 2:
            sigma_guess = max(
                (float(t_fit[above[-1]]) - float(t_fit[above[0]])) / 2.355,
                1e-3,
            )
        elif sigma_ini is not None and sigma_ini > 0:
            sigma_guess = max(float(sigma_ini), 1e-3)
        else:
            sigma_guess = max((float(t_fit[-1]) - float(t_fit[0])) / 6.0, 1e-3)

        # --- Step 5: Bounds — force A <= 0 for the minimum fit ---
        # Parameter order in gaussian_with_offset: (t, A, t0, sigma, C)
        t_lo = max(x0_ini - self.max_center_shift_ps, float(t_fit[0]))
        t_hi = min(x0_ini + self.max_center_shift_ps, float(t_fit[-1]))
        lower = [-np.inf, t_lo, self.min_sigma_ps, -np.inf]
        upper = [0.0,     t_hi, self.max_sigma_ps,  np.inf]

        popt, _ = curve_fit(
            self.gaussian_with_offset,
            t_fit,
            y_fit,
            p0=[A_guess, t0_guess, sigma_guess, baseline],
            bounds=(lower, upper),
            maxfev=50000,
            method="trf",
        )

        A, t0, sigma, C = popt
        sigma = abs(float(sigma))
        fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma

        return {
            "A": float(A),
            "t0_ps": float(t0),
            "sigma_ps": sigma,
            "C": float(C),
            "fwhm_ps": float(fwhm),
            "success": True,
            # Diagnostics
            "x0_ini_ps": x0_ini,
            "A_ini": A_ini,
            "C_ini": C_ini,
            "sigma_ini_ps": float(sigma_ini) if sigma_ini is not None else float("nan"),
            "fwhm_ini_ps": float(fwhm_ini) if fwhm_ini is not None else float("nan"),
            "fit_window_left_ps": float(t_fit[0]),
            "fit_window_right_ps": float(t_fit[-1]),
            "fit_window_n_points": n_pts,
            "adaptive_half_width_ps": half_width,
            "fit_window_sigma_factor": self.fit_window_sigma_factor,
            "used_adaptive_window": used_adaptive,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _estimate_sigma(
        self,
        x: np.ndarray,
        y: np.ndarray,
        C_ini: float,
        A_ini: float,
    ):
        """
        Estimate sigma and FWHM from half-amplitude crossings.

        Handles both positive (A_ini > 0) and negative (A_ini < 0) pulses.
        Returns (sigma_ini, fwhm_ini) or (None, None) when estimation fails.
        """
        if abs(A_ini) < 1e-30:
            return None, None

        half_level = C_ini + A_ini / 2.0

        # For a positive pulse, find where signal is above the half-level.
        # For a negative pulse, find where signal is below the half-level.
        if A_ini > 0:
            idx_half = np.where(y >= half_level)[0]
        else:
            idx_half = np.where(y <= half_level)[0]

        if len(idx_half) < 2:
            return None, None

        fwhm_ini = float(x[idx_half[-1]] - x[idx_half[0]])
        if fwhm_ini <= 0:
            return None, None

        sigma_ini = fwhm_ini / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        return sigma_ini, fwhm_ini

    def _compute_fit_window(
        self,
        x: np.ndarray,
        x0_ini: float,
        sigma_ini,
    ):
        """
        Compute the Gaussian fitting window mask.

        When adaptive_fit_window is True and sigma_ini passes all sanity checks,
        uses x0_ini ± fit_window_sigma_factor * sigma_ini.
        Otherwise falls back to x0_ini ± 0.5 * fit_window_ps.

        Returns
        -------
        mask : np.ndarray of bool
        half_width : float
            Actual half-width used (adaptive or fallback).
        used_adaptive : bool
            True when the adaptive window was chosen.
        """
        fallback_half = 0.5 * self.fit_window_ps

        # Try adaptive window
        if self.adaptive_fit_window and sigma_ini is not None:
            sigma_val = float(sigma_ini)
            if (
                np.isfinite(sigma_val)
                and self.min_sigma_ps < sigma_val < self.max_sigma_ps
            ):
                half_width = self.fit_window_sigma_factor * sigma_val
                mask = (x >= x0_ini - half_width) & (x <= x0_ini + half_width)
                if int(np.count_nonzero(mask)) >= self.min_fit_points:
                    return mask, half_width, True

        # Fallback to fixed window centred on the initial peak estimate
        half_width = fallback_half
        mask = (x >= x0_ini - half_width) & (x <= x0_ini + half_width)
        if int(np.count_nonzero(mask)) < 5:
            # Last resort: use the entire trace
            mask = np.ones(len(x), dtype=bool)
            half_width = 0.5 * float(x[-1] - x[0])
        return mask, half_width, False

    def _get_correlation_window_mask(
        self,
        t_ps: np.ndarray,
        y_ref: np.ndarray,
    ) -> np.ndarray:
        if self.corr_window_ps is None:
            return np.ones(len(t_ps), dtype=bool)

        idx_peak = int(np.argmax(np.abs(y_ref)))
        t0 = float(t_ps[idx_peak])
        half = 0.5 * float(self.corr_window_ps)

        mask = (t_ps >= t0 - half) & (t_ps <= t0 + half)

        if int(np.count_nonzero(mask)) < 5:
            return np.ones(len(t_ps), dtype=bool)

        return mask

    def _get_fit_window_mask(
        self,
        t_ps: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        idx_peak = int(np.argmax(np.abs(y)))
        t0 = float(t_ps[idx_peak])
        half = 0.5 * self.fit_window_ps

        mask = (t_ps >= t0 - half) & (t_ps <= t0 + half)

        if int(np.count_nonzero(mask)) < 5:
            return np.ones(len(t_ps), dtype=bool)

        return mask

    @staticmethod
    def _parabolic_peak_offset(
        y_m1: float,
        y_0: float,
        y_p1: float,
    ) -> float:
        """
        Sub-sample refinement using a parabolic fit around the correlation peak.
        """
        denom = y_m1 - 2.0 * y_0 + y_p1

        if abs(denom) < 1e-30:
            return 0.0

        return 0.5 * (y_m1 - y_p1) / denom

    def _run_alignment(
        self,
        t_ps: np.ndarray,
        all_y: np.ndarray,
    ):
        method = self.align_method

        if method == "correlation":
            return self.align_by_correlation_to_first(
                t_ps,
                all_y,
                subsample=False,
            )

        if method == "correlation_subsample":
            return self.align_by_correlation_to_first(
                t_ps,
                all_y,
                subsample=True,
            )

        if method == "gaussian_minimum_mean":
            return self.align_by_gaussian_minimum_to_mean(t_ps, all_y)

        if method == "gaussian_median_integer":
            return self.align_by_gaussian_median_integer(t_ps, all_y)

        raise ValueError(
            f"Unknown align_method={method!r}. "
            "Use 'correlation', 'correlation_subsample', "
            "'gaussian_minimum_mean', or 'gaussian_median_integer'."
        )


# ---------------------------------------------------------------------------
# Directory scanner
# ---------------------------------------------------------------------------

def scan_samples(
    base_dir: str | Path,
    correction_factor: float = 0.0414,
    exclude_reference: bool = True,
    align: bool = False,
    align_method: str = "correlation_subsample",
    corr_window_ps: Optional[float] = None,
    fit_method: Optional[str] = None,
    fit_window_ps: float = 8.0,
    adaptive_fit_window: bool = True,
    fit_window_sigma_factor: float = 5.0,
    fallback_fit_window_ps: Optional[float] = None,
    min_sigma_ps: float = 0.05,
    max_sigma_ps: float = 10.0,
    min_fit_points: int = 10,
    max_center_shift_ps: float = 5.0,
) -> List[THZDataset]:
    """
    Scan all subdirectories of *base_dir* and return one :class:`THZDataset`
    for each sample folder.
    """
    base_dir = Path(base_dir)
    datasets: List[THZDataset] = []

    reference_names = {"reference", "refrence"}

    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue

        if exclude_reference and folder.name.lower() in reference_names:
            continue

        try:
            ds = THZDataset(
                path=folder,
                correction_factor=correction_factor,
                align=align,
                align_method=align_method,
                corr_window_ps=corr_window_ps,
                fit_method=fit_method,
                fit_window_ps=fit_window_ps,
                adaptive_fit_window=adaptive_fit_window,
                fit_window_sigma_factor=fit_window_sigma_factor,
                fallback_fit_window_ps=fallback_fit_window_ps,
                min_sigma_ps=min_sigma_ps,
                max_sigma_ps=max_sigma_ps,
                min_fit_points=min_fit_points,
                max_center_shift_ps=max_center_shift_ps,
            )
            datasets.append(ds)

        except RuntimeError as exc:
            log.warning("Skipping %s: %s", folder.name, exc)

    return datasets