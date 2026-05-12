"""
thz_tds.dataset_diag
====================
Robust dataset loader and diagnostic module for THz-TDS data.

Physics note on alignment
--------------------------
Two distinct alignment operations exist and must never be confused:

  * **Internal alignment** corrects timing jitter between repeated sweeps
    *within* a single measurement folder.  It improves the SNR of the
    averaged trace without altering the physical propagation delay.
    ``THZDatasetDiag(align=True)`` does this.

  * **Reference–sample alignment** would shift a sample trace to overlap
    a reference trace, erasing the propagation delay that encodes the
    complex refractive index.  **This module never does this.**
    The reference–sample delay is a physical observable; preserve it.

Typical workflow
----------------
::

    from thz_tds.dataset_diag import (
        THZDatasetDiag, compare_reference_sample_timing,
        print_dataset_report, plot_ref_sample_overlay,
    )

    ref = THZDatasetDiag("data/air1",   align=True)
    sam = THZDatasetDiag("data/point3", align=True)

    print_dataset_report(ref)
    print_dataset_report(sam)

    report = compare_reference_sample_timing(ref, sam, thickness_m=0.5e-3)
    print(report)

    plot_ref_sample_overlay(ref, sam)
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_C0: float = 2.99792458e8  # speed of light [m/s]

# Folder names that indicate reference/background measurements.
# Excluded from sample scans; searched by scan_references().
_REFERENCE_NAMES: frozenset = frozenset({
    "reference", "refrence",   # legacy spelling preserved for compat
    "ref", "refs",
    "air", "airs",
    "background", "backgrounds",
})

_JITTER_LARGE_PS: float = 0.5       # warn if any single |shift| exceeds this
_JITTER_STD_LARGE_PS: float = 0.2   # warn if std of shifts exceeds this
_SNR_LOW: float = 10.0              # warn if SNR estimate falls below this
_N_SUSPICIOUS_LOW: float = 1.0      # n < this is physically unrealistic
_N_SUSPICIOUS_HIGH: float = 15.0    # n > this is highly suspicious


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AlignmentDiagnostics:
    """Jitter statistics from an internal alignment run."""
    method: str
    shifts_ps: np.ndarray    # shape (n_traces,); shifts[0] is always 0
    mean_ps: float           # mean of shifts[1:]
    std_ps: float            # std  of shifts[1:]
    max_abs_ps: float        # max|shifts[1:]|
    reference_time_ps: float # time of the anchor feature in trace 0

    def __str__(self) -> str:
        return (
            f"AlignmentDiagnostics(method={self.method!r}, "
            f"mean={self.mean_ps:.4f} ps, std={self.std_ps:.4f} ps, "
            f"max_abs={self.max_abs_ps:.4f} ps)"
        )


@dataclass
class TimingComparisonReport:
    """Result of :func:`compare_reference_sample_timing`."""
    ref_peak_time_ps: float
    sam_peak_time_ps: float
    delay_ps: float
    n_from_delay: float
    expected_delay_ps: Optional[float] = None
    timing_error_ps: Optional[float] = None
    air_path_error_mm: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "Reference–sample timing comparison",
            f"  t_ref_peak         : {self.ref_peak_time_ps:.4f} ps",
            f"  t_sam_peak         : {self.sam_peak_time_ps:.4f} ps",
            f"  Δt (delay)         : {self.delay_ps:.4f} ps",
            f"  n (from peak delay): {self.n_from_delay:.4f}",
        ]
        if self.expected_delay_ps is not None:
            lines.append(f"  Expected delay     : {self.expected_delay_ps:.4f} ps")
        if self.timing_error_ps is not None:
            lines.append(f"  Timing error       : {self.timing_error_ps:.4f} ps")
        if self.air_path_error_mm is not None:
            lines.append(f"  Air-path error     : {self.air_path_error_mm:.3f} mm")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# THZDatasetDiag
# ---------------------------------------------------------------------------

class THZDatasetDiag:
    """
    Enhanced THz-TDS dataset loader with alignment and full diagnostics.

    Loads all traces from a folder, optionally applies **internal** jitter
    correction (within the same folder only), averages, and computes a
    comprehensive set of diagnostic quantities.

    .. important::

       ``align=True`` corrects timing jitter *between repeated scans of the
       same measurement*.  Never use this to align a sample against a
       reference — that erases the physical propagation delay.

    Parameters
    ----------
    path : str or Path
        Folder containing tab-separated THz trace files.
    correction_factor : float
        Multiplicative factor applied to raw y-values (default 0.0414).
    align : bool
        Enable internal jitter correction (default False).
    align_method : str
        One of: ``'none'``, ``'minimum'``, ``'maximum'``, ``'absolute_peak'``,
        ``'correlation'``, ``'correlation_subsample'``, ``'gaussian'``.
    corr_window_ps : float or None
        Half-width of the cross-correlation window in ps (None = full trace).
    fit_method : str or None
        Optional fit on averaged trace: ``None`` or ``'gaussian'``.
    fit_window_ps : float
        Width of the Gaussian fit window in ps (default 8.0).
    """

    _EXTENSIONS = frozenset({".txt", ".dat", ".csv", ".tsv"})

    def __init__(
        self,
        path: Union[str, Path],
        correction_factor: float = 0.0414,
        align: bool = False,
        align_method: str = "correlation_subsample",
        corr_window_ps: Optional[float] = None,
        fit_method: Optional[str] = None,
        fit_window_ps: float = 8.0,
    ) -> None:
        self.path = str(path)
        self.name = os.path.basename(self.path.rstrip("/\\"))
        self.correction_factor = float(correction_factor)
        self.do_align = bool(align)
        self.align_method = align_method.lower()
        self.corr_window_ps = corr_window_ps
        self.fit_method = None if fit_method is None else fit_method.lower()
        self.fit_window_ps = float(fit_window_ps)

        # Trace arrays
        self.x_ps: Optional[np.ndarray] = None
        self.all_y: Optional[np.ndarray] = None    # post-alignment
        self.all_y_raw: Optional[np.ndarray] = None  # pre-alignment (raw stack)
        self.y_avg: Optional[np.ndarray] = None

        # Load diagnostics
        self.loaded_filenames: List[str] = []
        self.skipped_filenames: Dict[str, str] = {}  # filename → reason string
        self.n_traces: int = 0
        self.time_axes_same_before_interpolation: bool = True

        # Time-axis properties
        self.dt_ps: float = float("nan")
        self.scan_length_ps: float = float("nan")
        self.n_time_samples: int = 0
        self.min_time_ps: float = float("nan")
        self.max_time_ps: float = float("nan")
        self.time_axis_monotonic: bool = False
        self.time_axis_uniform: bool = False

        # Amplitude / SNR diagnostics
        self.peak_abs_time_ps: float = float("nan")
        self.peak_abs_amplitude: float = float("nan")
        self.baseline_mean: float = float("nan")
        self.baseline_std: float = float("nan")
        self.snr_estimate: float = float("nan")

        # Alignment diagnostics
        self.alignment: Optional[AlignmentDiagnostics] = None
        # Legacy compatibility attributes (match THZDataset API)
        self.jitter_shifts_ps: Optional[np.ndarray] = None
        self.t_min_ref: Optional[float] = None

        # Fit and FFT results
        self.fit_result: Optional[Dict] = None
        self.freq_THz: Optional[np.ndarray] = None
        self.spectrum_dB: Optional[np.ndarray] = None
        self.spectrum_complex: Optional[np.ndarray] = None

        # All warnings accumulated during load()
        self.warnings: List[str] = []

        self.load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        (Re-)load all traces, apply internal alignment if requested,
        and compute all diagnostic quantities.  Called automatically by
        ``__init__``.
        """
        self.warnings = []
        files = self._list_files(self.path)

        if not files:
            raise RuntimeError(f"No THz files found in {self.path!r}")

        all_y: List[np.ndarray] = []
        x_ref: Optional[np.ndarray] = None
        x_ref_orig: Optional[np.ndarray] = None
        self.loaded_filenames = []
        self.skipped_filenames = {}
        self.time_axes_same_before_interpolation = True

        for fname in files:
            full = os.path.join(self.path, fname)
            try:
                df = pd.read_csv(full, sep="\t", header=0, comment="#")
            except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                self.skipped_filenames[fname] = f"read error: {exc}"
                log.warning("Skipping %s: %s", fname, exc)
                continue

            if df.shape[1] < 2:
                self.skipped_filenames[fname] = "fewer than 2 columns"
                log.warning("Skipping %s: fewer than 2 columns.", fname)
                continue

            try:
                x = df.iloc[:, 0].to_numpy(dtype=float)
                y = df.iloc[:, 1].to_numpy(dtype=float) * self.correction_factor
            except (TypeError, ValueError) as exc:
                self.skipped_filenames[fname] = f"conversion error: {exc}"
                log.warning("Skipping %s: %s", fname, exc)
                continue

            if x_ref is None:
                x_ref = x
                x_ref_orig = x.copy()
            else:
                if not (len(x) == len(x_ref_orig) and np.allclose(x, x_ref_orig, atol=1e-9)):
                    self.time_axes_same_before_interpolation = False
                    self.warnings.append(
                        f"{fname}: time axis differs from first trace; interpolating."
                    )
                    y = np.interp(x_ref, x, y, left=0.0, right=0.0)

            all_y.append(y)
            self.loaded_filenames.append(fname)

        if not all_y:
            raise RuntimeError(f"No readable THz files found in {self.path!r}")

        if self.skipped_filenames:
            self.warnings.append(
                f"{len(self.skipped_filenames)} file(s) skipped: "
                + ", ".join(self.skipped_filenames.keys())
            )

        all_y_arr = np.vstack(all_y)
        self.n_traces = all_y_arr.shape[0]
        self._compute_time_axis_diagnostics(x_ref)

        # Store raw (pre-alignment) traces before any shifting
        self.all_y_raw = all_y_arr.copy()

        # Internal alignment: corrects jitter between scans in THIS folder only.
        # Never align sample traces against the reference — that erases the
        # physical propagation delay required for refractive index extraction.
        if self.do_align and self.align_method != "none":
            all_y_arr, al = self._run_alignment(x_ref, all_y_arr)
            self.alignment = al
            self.jitter_shifts_ps = al.shifts_ps
            self.t_min_ref = al.reference_time_ps
            self._check_alignment_warnings(al)
        else:
            self.alignment = None
            self.jitter_shifts_ps = None
            self.t_min_ref = None

        self.x_ps = x_ref
        self.all_y = all_y_arr
        self.y_avg = np.mean(all_y_arr, axis=0)

        self._compute_amplitude_diagnostics()
        self.compute_fit()

    def _list_files(self, path: str) -> List[str]:
        try:
            entries = os.listdir(path)
        except OSError as exc:
            raise RuntimeError(f"Cannot list directory {path!r}: {exc}") from exc
        return sorted(
            f for f in entries
            if os.path.isfile(os.path.join(path, f))
            and os.path.splitext(f)[1].lower() in self._EXTENSIONS
        )

    def _compute_time_axis_diagnostics(self, t: np.ndarray) -> None:
        self.n_time_samples = len(t)
        self.min_time_ps = float(t[0])
        self.max_time_ps = float(t[-1])
        self.scan_length_ps = float(t[-1] - t[0])

        diffs = np.diff(t)
        self.dt_ps = float(np.mean(diffs))
        self.time_axis_monotonic = bool(np.all(diffs > 0))

        if not self.time_axis_monotonic:
            self.warnings.append("Time axis is not monotonically increasing.")

        tol = 0.01 * abs(self.dt_ps) if self.dt_ps != 0 else 1e-9
        self.time_axis_uniform = bool(np.max(np.abs(diffs - self.dt_ps)) < tol)
        if not self.time_axis_uniform:
            self.warnings.append(
                f"Time axis non-uniform "
                f"(max step deviation = {float(np.max(np.abs(diffs - self.dt_ps))):.4g} ps)."
            )

    def _compute_amplitude_diagnostics(self) -> None:
        y, t = self.y_avg, self.x_ps
        peak_idx = int(np.argmax(np.abs(y)))
        self.peak_abs_time_ps = float(t[peak_idx])
        self.peak_abs_amplitude = float(np.abs(y[peak_idx]))

        # Baseline: quieter of the first 20% or last 20% of the scan window.
        # This avoids accidentally using a region that contains the pulse.
        n = len(y)
        n_base = max(5, n // 5)
        pre, post = y[:n_base], y[n - n_base:]
        baseline_region = pre if np.max(np.abs(pre)) <= np.max(np.abs(post)) else post

        self.baseline_mean = float(np.mean(baseline_region))
        self.baseline_std = float(np.std(baseline_region))
        self.snr_estimate = (
            self.peak_abs_amplitude / self.baseline_std
            if self.baseline_std > 0 else float("inf")
        )
        if self.snr_estimate < _SNR_LOW:
            self.warnings.append(
                f"Low SNR: {self.snr_estimate:.1f} (threshold = {_SNR_LOW:.0f})."
            )

    def _check_alignment_warnings(self, al: AlignmentDiagnostics) -> None:
        if al.max_abs_ps > _JITTER_LARGE_PS:
            self.warnings.append(
                f"Large alignment shift: max|shift| = {al.max_abs_ps:.4f} ps "
                f"(threshold {_JITTER_LARGE_PS} ps). Check for outlier scans."
            )
        if al.std_ps > _JITTER_STD_LARGE_PS:
            self.warnings.append(
                f"Large jitter spread: std = {al.std_ps:.4f} ps "
                f"(threshold {_JITTER_STD_LARGE_PS} ps)."
            )

    # ------------------------------------------------------------------
    # FFT and fit
    # ------------------------------------------------------------------

    def compute_fft(self, pad_factor: int = 4):
        """Compute the power spectrum of the averaged trace."""
        if self.y_avg is None or self.x_ps is None:
            raise RuntimeError("Call load() before compute_fft().")
        from .spectral import fft_field, power_spectrum_dB
        freq_THz, spectrum_dB = power_spectrum_dB(
            self.x_ps, self.y_avg, pad_factor=pad_factor
        )
        _, spectrum_complex = fft_field(self.x_ps, self.y_avg, pad_factor=pad_factor)
        self.freq_THz = freq_THz
        self.spectrum_dB = spectrum_dB
        self.spectrum_complex = spectrum_complex
        return freq_THz, spectrum_dB, spectrum_complex

    def compute_fit(self) -> Optional[Dict]:
        """Compute optional Gaussian fit on averaged trace."""
        self.fit_result = None
        if self.fit_method is None:
            return None
        if self.fit_method == "gaussian":
            try:
                self.fit_result = self.fit_gaussian_to_trace(self.x_ps, self.y_avg)
            except Exception as exc:
                self.warnings.append(f"Gaussian fit failed: {exc}")
            return self.fit_result
        raise ValueError(
            f"Unknown fit_method={self.fit_method!r}. Use None or 'gaussian'."
        )

    # ------------------------------------------------------------------
    # Summary / report
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a multi-line diagnostic summary string."""
        lines = [
            f"THZDatasetDiag: {self.path}",
            f"  Loaded traces     : {self.n_traces}",
            f"  Skipped files     : {len(self.skipped_filenames)}",
            f"  Time range        : [{self.min_time_ps:.3f}, {self.max_time_ps:.3f}] ps",
            f"  Scan length       : {self.scan_length_ps:.3f} ps",
            f"  dt                : {self.dt_ps:.5f} ps",
            f"  N samples         : {self.n_time_samples}",
            f"  Monotonic axis    : {self.time_axis_monotonic}",
            f"  Uniform step      : {self.time_axis_uniform}",
            f"  Same axes         : {self.time_axes_same_before_interpolation}",
            f"  Peak time         : {self.peak_abs_time_ps:.4f} ps",
            f"  Peak amplitude    : {self.peak_abs_amplitude:.4e}",
            f"  Baseline std      : {self.baseline_std:.4e}",
            f"  SNR estimate      : {self.snr_estimate:.1f}",
            f"  Alignment         : "
            + (f"enabled ({self.align_method})" if self.do_align else "disabled"),
        ]
        if self.alignment is not None:
            al = self.alignment
            lines += [
                f"    Jitter mean     : {al.mean_ps:.5f} ps",
                f"    Jitter std      : {al.std_ps:.5f} ps",
                f"    Jitter max|Δ|   : {al.max_abs_ps:.5f} ps",
            ]
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    ! {w}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Alignment dispatch
    # ------------------------------------------------------------------

    def _run_alignment(
        self, t_ps: np.ndarray, all_y: np.ndarray
    ) -> Tuple[np.ndarray, AlignmentDiagnostics]:
        """Dispatch to the chosen internal alignment method."""
        m = self.align_method
        if m in ("minimum", "maximum", "absolute_peak"):
            aligned, shifts, t_ref = self._align_by_feature(t_ps, all_y, feature=m)
        elif m in ("correlation", "correlation_subsample"):
            aligned, shifts, t_ref = self._align_by_correlation(
                t_ps, all_y, subsample=(m == "correlation_subsample")
            )
        elif m == "gaussian":
            aligned, shifts, t_ref = self._align_by_gaussian(t_ps, all_y)
        else:
            raise ValueError(
                f"Unknown align_method={m!r}. Use one of: 'none', 'minimum', "
                "'maximum', 'absolute_peak', 'correlation', "
                "'correlation_subsample', 'gaussian'."
            )
        tail = shifts[1:] if len(shifts) > 1 else np.zeros(1)
        diag = AlignmentDiagnostics(
            method=m,
            shifts_ps=shifts,
            mean_ps=float(np.mean(tail)),
            std_ps=float(np.std(tail)),
            max_abs_ps=float(np.max(np.abs(tail))),
            reference_time_ps=float(t_ref),
        )
        return aligned, diag

    def _feature_time(self, t_ps: np.ndarray, y: np.ndarray, feature: str) -> float:
        if feature == "minimum":
            return float(t_ps[int(np.argmin(y))])
        if feature == "maximum":
            return float(t_ps[int(np.argmax(y))])
        # absolute_peak
        return float(t_ps[int(np.argmax(np.abs(y)))])

    def _align_by_feature(
        self, t_ps: np.ndarray, all_y: np.ndarray, feature: str
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        n = all_y.shape[0]
        t_ref = self._feature_time(t_ps, all_y[0], feature)
        aligned = np.zeros_like(all_y)
        shifts = np.zeros(n)
        aligned[0] = all_y[0]
        for i in range(1, n):
            shift = t_ref - self._feature_time(t_ps, all_y[i], feature)
            shifts[i] = shift
            aligned[i] = np.interp(t_ps, t_ps + shift, all_y[i], left=0.0, right=0.0)
        return aligned, shifts, t_ref

    def _align_by_correlation(
        self, t_ps: np.ndarray, all_y: np.ndarray, subsample: bool
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        n_scans = all_y.shape[0]
        dt = float(np.mean(np.diff(t_ps)))
        y0 = all_y[0]
        t_ref = float(t_ps[int(np.argmin(y0))])

        aligned = np.zeros_like(all_y)
        shifts = np.zeros(n_scans)
        aligned[0] = y0

        mask = self._corr_window_mask(t_ps, y0)
        seg0 = y0[mask] - np.mean(y0[mask])
        m = len(seg0)
        if m < 2:
            raise ValueError("Correlation window too short.")
        nfft = 2 * m - 1
        F0 = np.fft.fft(seg0, n=nfft)
        lags = np.arange(-(m - 1), m, dtype=float)

        for i in range(1, n_scans):
            seg = all_y[i][mask] - np.mean(all_y[i][mask])
            corr = np.fft.ifft(F0 * np.conj(np.fft.fft(seg, n=nfft))).real
            corr = np.concatenate((corr[-(m - 1):], corr[:m]))
            k0 = int(np.argmax(corr))
            lag = lags[k0]
            if subsample and 0 < k0 < len(corr) - 1:
                lag += _parabolic_offset(corr[k0 - 1], corr[k0], corr[k0 + 1])
            delay = lag * dt
            shifts[i] = delay
            aligned[i] = np.interp(t_ps, t_ps + delay, all_y[i], left=0.0, right=0.0)

        return aligned, shifts, t_ref

    def _align_by_gaussian(
        self, t_ps: np.ndarray, all_y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        n = all_y.shape[0]
        fit0 = self.fit_gaussian_to_trace(t_ps, all_y[0])
        t0_ref = fit0["t0_ps"]
        t_ref = float(t_ps[int(np.argmin(all_y[0]))])

        aligned = np.zeros_like(all_y)
        shifts = np.zeros(n)
        aligned[0] = all_y[0]

        for i in range(1, n):
            try:
                shift = t0_ref - self.fit_gaussian_to_trace(t_ps, all_y[i])["t0_ps"]
            except Exception:
                shift = 0.0
            shifts[i] = shift
            aligned[i] = np.interp(t_ps, t_ps + shift, all_y[i], left=0.0, right=0.0)

        return aligned, shifts, t_ref

    def _corr_window_mask(self, t_ps: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.corr_window_ps is None:
            return np.ones(len(t_ps), dtype=bool)
        idx = int(np.argmax(np.abs(y)))
        t0 = float(t_ps[idx])
        half = 0.5 * float(self.corr_window_ps)
        mask = (t_ps >= t0 - half) & (t_ps <= t0 + half)
        return mask if int(np.count_nonzero(mask)) >= 5 else np.ones(len(t_ps), dtype=bool)

    # ------------------------------------------------------------------
    # Gaussian fitting
    # ------------------------------------------------------------------

    @staticmethod
    def _gaussian(t, A, t0, sigma, C):
        return A * np.exp(-((t - t0) ** 2) / (2.0 * sigma ** 2)) + C

    def fit_gaussian_to_trace(self, t_ps: np.ndarray, y: np.ndarray) -> Dict:
        """Fit a Gaussian with constant offset to a THz trace."""
        mask = self._fit_window_mask(t_ps, y)
        t_fit, y_fit = t_ps[mask], y[mask]
        if len(t_fit) < 5:
            raise RuntimeError("Fit window too small for Gaussian fit.")
        baseline = float(np.median(y_fit))
        idx = int(np.argmax(np.abs(y_fit - baseline)))
        t0_g = float(t_fit[idx])
        A_g = float(y_fit[idx]) - baseline
        above = np.where(np.abs(y_fit - baseline) >= 0.5 * abs(A_g))[0]
        sigma_g = (
            max((float(t_fit[above[-1]]) - float(t_fit[above[0]])) / 2.355, 1e-3)
            if len(above) >= 2
            else max((float(t_fit[-1]) - float(t_fit[0])) / 6.0, 1e-3)
        )
        popt, _ = curve_fit(
            self._gaussian, t_fit, y_fit,
            p0=[A_g, t0_g, sigma_g, baseline],
            bounds=(
                [-np.inf, float(t_fit[0]),  1e-4,   -np.inf],
                [ np.inf, float(t_fit[-1]), float(t_fit[-1] - t_fit[0]), np.inf],
            ),
            maxfev=50000, method="trf",
        )
        A, t0, sigma, C = popt
        return {
            "A": A, "t0_ps": t0, "sigma_ps": abs(sigma), "C": C,
            "fwhm_ps": 2.355 * abs(sigma), "success": True,
        }

    def _fit_window_mask(self, t_ps: np.ndarray, y: np.ndarray) -> np.ndarray:
        idx = int(np.argmax(np.abs(y)))
        t0 = float(t_ps[idx])
        half = 0.5 * self.fit_window_ps
        mask = (t_ps >= t0 - half) & (t_ps <= t0 + half)
        return mask if int(np.count_nonzero(mask)) >= 5 else np.ones(len(t_ps), dtype=bool)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parabolic_offset(y_m: float, y_0: float, y_p: float) -> float:
    d = y_m - 2.0 * y_0 + y_p
    return 0.0 if abs(d) < 1e-30 else 0.5 * (y_m - y_p) / d


def _ensure_axes(
    ax: Optional[plt.Axes],
    figsize: Tuple[float, float] = (10, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    return ax.get_figure(), ax


def _color_cycle(n: int) -> List:
    cmap = plt.get_cmap("tab20")
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------

def load_reference(
    path: Union[str, Path],
    correction_factor: float = 0.0414,
    align: bool = True,
    align_method: str = "correlation_subsample",
    corr_window_ps: Optional[float] = None,
) -> THZDatasetDiag:
    """Load a reference (air) measurement folder."""
    return THZDatasetDiag(
        path=path,
        correction_factor=correction_factor,
        align=align,
        align_method=align_method,
        corr_window_ps=corr_window_ps,
    )


def load_sample(
    path: Union[str, Path],
    correction_factor: float = 0.0414,
    align: bool = True,
    align_method: str = "correlation_subsample",
    corr_window_ps: Optional[float] = None,
) -> THZDatasetDiag:
    """Load a sample measurement folder.

    The alignment applied here is **internal** jitter correction within the
    sample folder only.  Do not align the returned dataset against the
    reference — that would erase the propagation delay.
    """
    return THZDatasetDiag(
        path=path,
        correction_factor=correction_factor,
        align=align,
        align_method=align_method,
        corr_window_ps=corr_window_ps,
    )


def scan_samples(
    base_dir: Union[str, Path],
    correction_factor: float = 0.0414,
    exclude_reference: bool = True,
    align: bool = False,
    align_method: str = "correlation_subsample",
    corr_window_ps: Optional[float] = None,
    max_samples: Optional[int] = None,
) -> List[THZDatasetDiag]:
    """Scan a directory and load all sample subfolders.

    Folders whose names match the reference/air/background exclusion list
    are skipped automatically.  See module-level ``_REFERENCE_NAMES``.
    """
    base_dir = Path(base_dir)
    datasets: List[THZDatasetDiag] = []

    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue
        if exclude_reference and folder.name.lower() in _REFERENCE_NAMES:
            log.info("Skipping reference folder: %s", folder.name)
            continue
        try:
            ds = THZDatasetDiag(
                path=folder,
                correction_factor=correction_factor,
                align=align,
                align_method=align_method,
                corr_window_ps=corr_window_ps,
            )
            datasets.append(ds)
        except RuntimeError as exc:
            log.warning("Skipping %s: %s", folder.name, exc)
        if max_samples is not None and len(datasets) >= max_samples:
            break

    return datasets


def scan_references(
    base_dir: Union[str, Path],
    correction_factor: float = 0.0414,
    align: bool = True,
    align_method: str = "correlation_subsample",
    corr_window_ps: Optional[float] = None,
) -> List[THZDatasetDiag]:
    """Scan a directory and return datasets from recognised reference/air folders only."""
    base_dir = Path(base_dir)
    datasets: List[THZDatasetDiag] = []

    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name.lower() not in _REFERENCE_NAMES:
            continue
        try:
            ds = THZDatasetDiag(
                path=folder,
                correction_factor=correction_factor,
                align=align,
                align_method=align_method,
                corr_window_ps=corr_window_ps,
            )
            datasets.append(ds)
        except RuntimeError as exc:
            log.warning("Skipping %s: %s", folder.name, exc)

    return datasets


def load_experiment(
    reference_path: Optional[Union[str, Path]] = None,
    sample_paths: Optional[List[Union[str, Path]]] = None,
    base_dir: Optional[Union[str, Path]] = None,
    correction_factor: float = 0.0414,
    align: bool = True,
    align_method: str = "correlation_subsample",
    corr_window_ps: Optional[float] = None,
) -> Tuple[Optional[THZDatasetDiag], List[THZDatasetDiag]]:
    """Load a complete experiment (one reference + one or more samples).

    Parameters
    ----------
    reference_path : str or Path, optional
        Explicit path to the reference folder.
    sample_paths : list of paths, optional
        Explicit paths to sample folders.
    base_dir : str or Path, optional
        If given, scans for references and samples automatically.
    align : bool
        Enable internal jitter correction within each folder.
    align_method : str
        Alignment algorithm.
    corr_window_ps : float or None
        Correlation window in ps.

    Returns
    -------
    ref_ds : THZDatasetDiag or None
    sample_dss : list[THZDatasetDiag]
    """
    kwargs = dict(
        correction_factor=correction_factor,
        align=align,
        align_method=align_method,
        corr_window_ps=corr_window_ps,
    )
    ref_ds: Optional[THZDatasetDiag] = None
    sample_dss: List[THZDatasetDiag] = []

    if reference_path is not None:
        ref_ds = load_reference(reference_path, **kwargs)

    if sample_paths:
        for p in sample_paths:
            try:
                sample_dss.append(load_sample(p, **kwargs))
            except RuntimeError as exc:
                log.warning("Could not load sample %s: %s", p, exc)

    if base_dir is not None:
        base = Path(base_dir)
        if ref_ds is None:
            refs = scan_references(base, **kwargs)
            if refs:
                ref_ds = refs[0]
                if len(refs) > 1:
                    log.warning(
                        "Multiple reference folders found; using %s.", refs[0].name
                    )
        if not sample_paths:
            sample_dss = scan_samples(base, **kwargs)

    return ref_ds, sample_dss


# ---------------------------------------------------------------------------
# Reference–sample comparison
# ---------------------------------------------------------------------------

def compute_peak_delay(
    ref_ds: THZDatasetDiag, sam_ds: THZDatasetDiag
) -> float:
    """Return peak delay Δt = t_sam_peak − t_ref_peak in ps."""
    t_ref = float(ref_ds.x_ps[int(np.argmax(np.abs(ref_ds.y_avg)))])
    t_sam = float(sam_ds.x_ps[int(np.argmax(np.abs(sam_ds.y_avg)))])
    return t_sam - t_ref


def estimate_n_from_peak_delay(delay_ps: float, thickness_m: float) -> float:
    """Estimate refractive index from peak delay: n = 1 + c·Δt / L."""
    return 1.0 + (delay_ps * 1e-12 * _C0) / thickness_m


def expected_delay_from_n(expected_n: float, thickness_m: float) -> float:
    """Expected peak delay for a given n and sample thickness [ps]."""
    return (expected_n - 1.0) * thickness_m / _C0 * 1e12


def timing_error_to_air_path_mm(timing_error_ps: float) -> float:
    """Convert a timing error in ps to the equivalent free-space path length in mm."""
    return timing_error_ps * 1e-12 * _C0 * 1e3


def compare_reference_sample_timing(
    ref_ds: THZDatasetDiag,
    sam_ds: THZDatasetDiag,
    thickness_m: float,
    expected_n: Optional[float] = None,
) -> TimingComparisonReport:
    """Detailed reference–sample timing comparison.

    Parameters
    ----------
    ref_ds : THZDatasetDiag
        Reference (air) dataset.
    sam_ds : THZDatasetDiag
        Sample dataset.
    thickness_m : float
        Sample thickness [m].
    expected_n : float or None
        If given, used to check plausibility and compute timing error.

    Returns
    -------
    TimingComparisonReport
    """
    t_ref = float(ref_ds.x_ps[int(np.argmax(np.abs(ref_ds.y_avg)))])
    t_sam = float(sam_ds.x_ps[int(np.argmax(np.abs(sam_ds.y_avg)))])
    delay = t_sam - t_ref
    n_peak = estimate_n_from_peak_delay(delay, thickness_m)

    exp_delay = t_error = path_err_mm = None
    if expected_n is not None:
        exp_delay = expected_delay_from_n(expected_n, thickness_m)
        t_error = delay - exp_delay
        path_err_mm = timing_error_to_air_path_mm(t_error)

    warnings: List[str] = []

    if delay <= 0:
        warnings.append(
            f"Sample peak is BEFORE reference peak (Δt = {delay:.4f} ps). "
            "Check measurement paths or time-zero calibration."
        )
    if n_peak < _N_SUSPICIOUS_LOW:
        warnings.append(
            f"n_peak = {n_peak:.4f} < 1 is physically unrealistic. "
            "Check reference/sample pairing and thickness."
        )
    if n_peak > _N_SUSPICIOUS_HIGH:
        warnings.append(
            f"n_peak = {n_peak:.4f} is very large. Verify thickness "
            f"({thickness_m * 1e3:.3f} mm) and reference/sample assignment."
        )
    if expected_n is not None and t_error is not None:
        if abs(t_error) > 1.0:
            warnings.append(
                f"Timing error = {t_error:.4f} ps "
                f"(≡ {path_err_mm:.2f} mm air path). "
                "May indicate wrong sample, wrong thickness, or time-zero shift."
            )
        if abs(n_peak - expected_n) / max(expected_n, 1e-6) > 0.10:
            warnings.append(
                f"n_peak = {n_peak:.4f} differs from expected n = {expected_n:.4f} "
                f"by > 10%."
            )

    return TimingComparisonReport(
        ref_peak_time_ps=t_ref,
        sam_peak_time_ps=t_sam,
        delay_ps=delay,
        n_from_delay=n_peak,
        expected_delay_ps=exp_delay,
        timing_error_ps=t_error,
        air_path_error_mm=path_err_mm,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Fabry–Pérot echo window check
# ---------------------------------------------------------------------------

def check_fp_echo_window(
    ds: THZDatasetDiag,
    n_expected: float,
    thickness_m: float,
    n_echoes: int = 3,
) -> Dict:
    """Check whether the scan window captures expected Fabry–Pérot echoes.

    Parameters
    ----------
    ds : THZDatasetDiag
        Sample dataset.
    n_expected : float
        Expected refractive index.
    thickness_m : float
        Sample thickness [m].
    n_echoes : int
        Number of echoes to check (default 3).

    Returns
    -------
    dict
        ``echo_delay_ps``, ``echo_times_ps``, ``scan_end_ps``,
        ``echoes_in_window``, ``scan_long_enough``, ``warnings``.
    """
    echo_delay = 2.0 * n_expected * thickness_m / _C0 * 1e12
    t_pulse = float(ds.x_ps[int(np.argmax(np.abs(ds.y_avg)))])
    echo_times = [t_pulse + k * echo_delay for k in range(1, n_echoes + 1)]
    scan_end = ds.max_time_ps
    in_window = [t <= scan_end for t in echo_times]
    fp_warnings: List[str] = []
    if not all(in_window):
        first_miss = next(k for k, ok in enumerate(in_window) if not ok) + 1
        fp_warnings.append(
            f"Scan ends at {scan_end:.1f} ps. "
            f"Echo {first_miss} expected at {echo_times[first_miss - 1]:.1f} ps "
            "— cut off. May cause frequency-domain artifacts."
        )
    return {
        "echo_delay_ps": echo_delay,
        "echo_times_ps": echo_times,
        "scan_end_ps": scan_end,
        "echoes_in_window": in_window,
        "scan_long_enough": all(in_window),
        "warnings": fp_warnings,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_raw_traces(
    ds: THZDatasetDiag,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot all raw (pre-alignment) traces."""
    fig, ax = _ensure_axes(ax, (10, 5))
    traces = ds.all_y_raw if ds.all_y_raw is not None else ds.all_y
    n = traces.shape[0]
    colors = _color_cycle(n)
    for i, y in enumerate(traces):
        ax.plot(ds.x_ps, y, color=colors[i], lw=0.8, alpha=0.7,
                label=f"Trace {i + 1}" if n <= 10 else None)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Amplitude [nA]")
    ax.set_title(title or f"Raw traces — {ds.name}  ({n} traces)")
    ax.grid(True, alpha=0.3)
    if n <= 10:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_aligned_traces(
    ds: THZDatasetDiag,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot all (post-alignment) traces with the average overlaid."""
    fig, ax = _ensure_axes(ax, (10, 5))
    n = ds.all_y.shape[0]
    colors = _color_cycle(n)
    for i, y in enumerate(ds.all_y):
        ax.plot(ds.x_ps, y, color=colors[i], lw=0.8, alpha=0.45,
                label=f"Trace {i + 1}" if n <= 10 else None)
    ax.plot(ds.x_ps, ds.y_avg, color="black", lw=2, label="Average", zorder=10)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Amplitude [nA]")
    ax.set_title(title or f"Aligned traces — {ds.name}  ({n} traces)")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if n <= 10:
        ax.legend(fontsize=8)
    else:
        ax.legend(["Average"], fontsize=8)
    fig.tight_layout()
    return fig


def plot_average_trace(
    ds: THZDatasetDiag,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot the averaged trace."""
    fig, ax = _ensure_axes(ax, (10, 5))
    ax.plot(ds.x_ps, ds.y_avg, color="tab:blue", lw=1.5)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Amplitude [nA]")
    ax.set_title(title or f"Average trace — {ds.name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_ref_sample_overlay(
    ref_ds: THZDatasetDiag,
    sam_ds: THZDatasetDiag,
    normalize: bool = True,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Overlay reference and sample average traces.

    When ``normalize=True``, both signals are divided by the reference peak so
    the reference reaches ±1 and the sample shows its true relative attenuation.
    This preserves the physical delay between the two pulses.
    """
    fig, ax = _ensure_axes(ax, (10, 5))
    scale = float(np.max(np.abs(ref_ds.y_avg))) if normalize else 1.0
    ax.plot(ref_ds.x_ps, ref_ds.y_avg / scale,
            color="tab:blue",   lw=1.5, label=f"Reference ({ref_ds.name})")
    ax.plot(sam_ds.x_ps, sam_ds.y_avg / scale,
            color="tab:orange", lw=1.5, label=f"Sample ({sam_ds.name})")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Amplitude (normalised) [nA]" if normalize else "Amplitude [nA]")
    ax.set_title(title or "Reference vs. sample overlay")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_jitter_shifts(
    ds: THZDatasetDiag,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of per-trace alignment shifts."""
    fig, ax = _ensure_axes(ax, (10, 4))
    if ds.alignment is None:
        ax.text(0.5, 0.5, "No alignment was performed.",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title(title or f"Jitter shifts — {ds.name} (alignment disabled)")
        fig.tight_layout()
        return fig

    shifts = ds.alignment.shifts_ps
    x = np.arange(len(shifts))
    ax.bar(x, shifts, color="tab:blue", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(ds.alignment.mean_ps, color="tab:red", lw=1.5, ls="--",
               label=f"Mean = {ds.alignment.mean_ps:.4f} ps")
    ax.set_xlabel("Trace index")
    ax.set_ylabel("Shift applied (ps)")
    ax.set_title(title or f"Alignment shifts — {ds.name}  (method: {ds.alignment.method})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_trace_residuals(
    ds: THZDatasetDiag,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot deviation of each aligned trace from the average."""
    fig, ax = _ensure_axes(ax, (10, 5))
    n = ds.all_y.shape[0]
    colors = _color_cycle(n)
    for i, y in enumerate(ds.all_y):
        ax.plot(ds.x_ps, y - ds.y_avg, color=colors[i], lw=0.8, alpha=0.7,
                label=f"Trace {i + 1}" if n <= 10 else None)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Residual [nA]")
    ax.set_title(title or f"Trace residuals — {ds.name}")
    ax.grid(True, alpha=0.3)
    if n <= 10:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_fft_magnitude(
    ds: THZDatasetDiag,
    pad_factor: int = 4,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot the FFT power spectrum of the averaged trace."""
    fig, ax = _ensure_axes(ax, (10, 5))
    freq_THz, spec_dB, _ = ds.compute_fft(pad_factor=pad_factor)
    ax.plot(freq_THz, spec_dB, color="tab:blue", lw=1.5)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(title or f"FFT power spectrum — {ds.name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_transfer_function(
    ref_ds: THZDatasetDiag,
    sam_ds: THZDatasetDiag,
    pad_factor: int = 4,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot magnitude and unwrapped phase of H(f) = E_sam / E_ref."""
    from .spectral import fft_field, ensure_common_time_axis

    _, y_ref, y_sam = ensure_common_time_axis(
        ref_ds.x_ps, ref_ds.y_avg, sam_ds.x_ps, sam_ds.y_avg
    )
    f_Hz, E_ref = fft_field(ref_ds.x_ps, y_ref, pad_factor=pad_factor)
    _, E_sam    = fft_field(ref_ds.x_ps, y_sam, pad_factor=pad_factor)
    f_THz = f_Hz / 1e12
    floor = 1e-30
    H = E_sam / np.where(np.abs(E_ref) < floor, floor, E_ref)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(f_THz, np.abs(H), color="tab:blue", lw=1.5)
    axes[0].set_ylabel("|H(f)|")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(title or f"Transfer function — {sam_ds.name}")

    axes[1].plot(f_THz, np.unwrap(np.angle(H)), color="tab:orange", lw=1.5)
    axes[1].set_ylabel("Phase (rad, unwrapped)")
    axes[1].set_xlabel("Frequency (THz)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def print_dataset_report(ds: THZDatasetDiag) -> None:
    """Print the diagnostic summary of a dataset."""
    print(ds.summary())


def print_experiment_report(
    refs: List[THZDatasetDiag],
    samples: List[THZDatasetDiag],
) -> None:
    """Print a summary report for all reference and sample datasets."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"EXPERIMENT REPORT   ({len(refs)} reference(s), {len(samples)} sample(s))")
    print(sep)
    print("\n--- REFERENCES ---")
    for ds in refs:
        print(ds.summary())
        print()
    print("--- SAMPLES ---")
    for ds in samples:
        print(ds.summary())
        print()


def save_dataset_report_json(
    ds: THZDatasetDiag,
    output_path: Union[str, Path],
) -> None:
    """Save a dataset diagnostic report as a JSON file."""
    report = {
        "name": ds.name,
        "path": ds.path,
        "n_traces": ds.n_traces,
        "loaded_filenames": ds.loaded_filenames,
        "skipped_filenames": ds.skipped_filenames,
        "time": {
            "min_ps":           ds.min_time_ps,
            "max_ps":           ds.max_time_ps,
            "scan_length_ps":   ds.scan_length_ps,
            "dt_ps":            ds.dt_ps,
            "n_samples":        ds.n_time_samples,
            "monotonic":        ds.time_axis_monotonic,
            "uniform":          ds.time_axis_uniform,
            "same_before_interp": ds.time_axes_same_before_interpolation,
        },
        "amplitude": {
            "peak_time_ps":     ds.peak_abs_time_ps,
            "peak_amplitude":   ds.peak_abs_amplitude,
            "baseline_mean":    ds.baseline_mean,
            "baseline_std":     ds.baseline_std,
            "snr_estimate":     (
                ds.snr_estimate if np.isfinite(ds.snr_estimate) else None
            ),
        },
        "alignment": None if ds.alignment is None else {
            "method":              ds.alignment.method,
            "mean_ps":             ds.alignment.mean_ps,
            "std_ps":              ds.alignment.std_ps,
            "max_abs_ps":          ds.alignment.max_abs_ps,
            "reference_time_ps":   ds.alignment.reference_time_ps,
            "shifts_ps":           ds.alignment.shifts_ps.tolist(),
        },
        "warnings": ds.warnings,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("Dataset report saved: %s", out)


def save_timing_report_csv(
    reports: List[TimingComparisonReport],
    output_path: Union[str, Path],
    labels: Optional[List[str]] = None,
) -> None:
    """Save a list of timing comparison reports to a CSV file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label", "ref_peak_ps", "sam_peak_ps", "delay_ps",
        "n_from_delay", "expected_delay_ps", "timing_error_ps",
        "air_path_error_mm", "warnings",
    ]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, r in enumerate(reports):
            lbl = labels[i] if labels and i < len(labels) else str(i)
            writer.writerow({
                "label":            lbl,
                "ref_peak_ps":      f"{r.ref_peak_time_ps:.6f}",
                "sam_peak_ps":      f"{r.sam_peak_time_ps:.6f}",
                "delay_ps":         f"{r.delay_ps:.6f}",
                "n_from_delay":     f"{r.n_from_delay:.6f}",
                "expected_delay_ps":
                    f"{r.expected_delay_ps:.6f}" if r.expected_delay_ps is not None else "",
                "timing_error_ps":
                    f"{r.timing_error_ps:.6f}" if r.timing_error_ps is not None else "",
                "air_path_error_mm":
                    f"{r.air_path_error_mm:.4f}" if r.air_path_error_mm is not None else "",
                "warnings": "; ".join(r.warnings),
            })
    log.info("Timing report saved: %s", out)


# ---------------------------------------------------------------------------
# Alignment method comparison
# ---------------------------------------------------------------------------

_ALL_ALIGNMENT_METHODS: List[str] = [
    "none",
    "minimum",
    "maximum",
    "absolute_peak",
    "correlation",
    "correlation_subsample",
    "gaussian",
]


def compare_alignment_methods(
    path: Union[str, Path],
    methods: Optional[List[str]] = None,
    correction_factor: float = 0.0414,
    corr_window_ps: Optional[float] = None,
) -> Dict[str, THZDatasetDiag]:
    """Load the same folder with every alignment method and return a results dict.

    Parameters
    ----------
    path : str or Path
        Measurement folder to analyse.
    methods : list of str or None
        Which methods to compare.  Defaults to all seven available methods.
    correction_factor : float
        Passed to each :class:`THZDatasetDiag`.
    corr_window_ps : float or None
        Correlation window for the correlation-based methods.

    Returns
    -------
    dict mapping method name → THZDatasetDiag
    """
    if methods is None:
        methods = _ALL_ALIGNMENT_METHODS

    results: Dict[str, THZDatasetDiag] = {}
    for m in methods:
        use_align = m != "none"
        ds = THZDatasetDiag(
            path=path,
            correction_factor=correction_factor,
            align=use_align,
            align_method=m if use_align else "correlation_subsample",
            corr_window_ps=corr_window_ps,
        )
        results[m] = ds
        log.debug("Alignment method %r: jitter std = %.4f ps", m,
                  ds.alignment.std_ps if ds.alignment else 0.0)
    return results


def plot_alignment_comparison(
    method_results: Dict[str, THZDatasetDiag],
    title_prefix: str = "",
) -> Tuple[plt.Figure, plt.Figure]:
    """Plot a side-by-side comparison of all alignment methods.

    Produces two figures:

    * **Figure A** — average traces overlaid for every method plus a
      difference panel (each method minus 'none' baseline).
    * **Figure B** — jitter shift bars for every aligned method arranged
      in a grid, with a summary statistics table below.

    Parameters
    ----------
    method_results : dict
        Output of :func:`compare_alignment_methods`.
    title_prefix : str
        Optional prefix for figure titles (e.g. the folder name).

    Returns
    -------
    fig_traces : plt.Figure
    fig_jitter : plt.Figure
    """
    methods = list(method_results.keys())
    n_methods = len(methods)

    # ── Figure A: average traces + difference from 'none' ────────────────
    has_none = "none" in method_results
    n_rows = 2 if has_none else 1
    fig_traces, axes_t = plt.subplots(
        n_rows, 1, figsize=(12, 4 * n_rows), sharex=True,
        squeeze=False,
    )
    ax_avg = axes_t[0, 0]
    colors = _color_cycle(n_methods)

    for color, (m, ds) in zip(colors, method_results.items()):
        ax_avg.plot(ds.x_ps, ds.y_avg, lw=1.2, alpha=0.85, color=color, label=m)

    ax_avg.set_ylabel("Amplitude [nA]")
    ax_avg.set_title(f"{title_prefix}Average trace — all alignment methods")
    ax_avg.legend(fontsize=8, ncol=2)
    ax_avg.grid(True, alpha=0.3)

    if has_none:
        ax_diff = axes_t[1, 0]
        y_none = method_results["none"].y_avg
        t_none = method_results["none"].x_ps
        for color, (m, ds) in zip(colors, method_results.items()):
            if m == "none":
                continue
            diff = ds.y_avg - np.interp(ds.x_ps, t_none, y_none)
            ax_diff.plot(ds.x_ps, diff, lw=1.0, alpha=0.8, color=color, label=m)
        ax_diff.axhline(0, color="black", lw=0.8)
        ax_diff.set_xlabel("Time (ps)")
        ax_diff.set_ylabel("ΔAmplitude [nA]")
        ax_diff.set_title("Difference from 'none' baseline")
        ax_diff.legend(fontsize=8, ncol=2)
        ax_diff.grid(True, alpha=0.3)
    else:
        axes_t[0, 0].set_xlabel("Time (ps)")

    fig_traces.tight_layout()

    # ── Figure B: jitter bars grid + stats table ─────────────────────────
    aligned_methods = [m for m in methods if method_results[m].alignment is not None]
    n_aligned = len(aligned_methods)

    if n_aligned == 0:
        fig_jitter, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No alignment was performed for any method.",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        fig_jitter.tight_layout()
        return fig_traces, fig_jitter

    ncols = min(3, n_aligned)
    nrows = (n_aligned + ncols - 1) // ncols

    fig_jitter, axes_j = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 3.5 * nrows + 1.5),
        squeeze=False,
    )

    for idx, m in enumerate(aligned_methods):
        row, col = divmod(idx, ncols)
        ax = axes_j[row, col]
        al = method_results[m].alignment
        shifts = al.shifts_ps
        x = np.arange(len(shifts))
        ax.bar(x, shifts, color="tab:blue", alpha=0.75, edgecolor="black", lw=0.4)
        ax.axhline(0, color="black", lw=0.7)
        ax.axhline(al.mean_ps, color="tab:red", lw=1.2, ls="--",
                   label=f"μ={al.mean_ps:.3f} ps")
        ax.set_title(f"{m}\nstd={al.std_ps:.4f} ps  max|Δ|={al.max_abs_ps:.4f} ps",
                     fontsize=9)
        ax.set_xlabel("Trace", fontsize=8)
        ax.set_ylabel("Shift (ps)", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, axis="y")

    # Hide unused subplots
    for idx in range(n_aligned, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes_j[row, col].set_visible(False)

    # Summary stats table below the grid
    col_labels = ["Method", "mean (ps)", "std (ps)", "max|Δ| (ps)"]
    table_data = [
        [m,
         f"{method_results[m].alignment.mean_ps:.5f}",
         f"{method_results[m].alignment.std_ps:.5f}",
         f"{method_results[m].alignment.max_abs_ps:.5f}"]
        for m in aligned_methods
    ]
    ax_table = fig_jitter.add_axes([0.05, 0.01, 0.90, 0.18])
    ax_table.axis("off")
    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)

    fig_jitter.suptitle(
        f"{title_prefix}Alignment method comparison — jitter shifts", fontsize=11
    )
    fig_jitter.subplots_adjust(top=0.92, bottom=0.22, hspace=0.55, wspace=0.35)

    return fig_traces, fig_jitter


def plot_alignment_method_detail(
    ds: THZDatasetDiag,
    method_name: str,
) -> plt.Figure:
    """Four-panel diagnostic figure for a single alignment method.

    Panels
    ------
    Top-left  : raw (pre-alignment) traces
    Top-right : aligned traces + average
    Bottom-left : per-trace jitter shifts
    Bottom-right : per-trace residuals from average
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    n = ds.all_y.shape[0]
    colors = _color_cycle(n)

    # ── Top-left: raw traces ────────────────────────────────────────────
    ax = axes[0, 0]
    raw = ds.all_y_raw if ds.all_y_raw is not None else ds.all_y
    for i, y in enumerate(raw):
        ax.plot(ds.x_ps, y, color=colors[i], lw=0.8, alpha=0.6,
                label=f"Trace {i+1}" if n <= 8 else None)
    ax.set_title("Raw traces (pre-alignment)")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Amplitude [nA]")
    ax.grid(True, alpha=0.3)
    if n <= 8:
        ax.legend(fontsize=7)

    # ── Top-right: aligned traces + average ────────────────────────────
    ax = axes[0, 1]
    for i, y in enumerate(ds.all_y):
        ax.plot(ds.x_ps, y, color=colors[i], lw=0.8, alpha=0.45)
    ax.plot(ds.x_ps, ds.y_avg, color="black", lw=2, label="Average", zorder=10)
    ax.set_title("Aligned traces + average")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Amplitude [nA]")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Bottom-left: jitter shifts ──────────────────────────────────────
    ax = axes[1, 0]
    if ds.alignment is not None:
        al = ds.alignment
        x = np.arange(len(al.shifts_ps))
        ax.bar(x, al.shifts_ps, color="tab:blue", alpha=0.8,
               edgecolor="black", lw=0.5)
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(al.mean_ps, color="tab:red", lw=1.5, ls="--",
                   label=f"mean = {al.mean_ps:.4f} ps")
        ax.axhline(al.mean_ps + al.std_ps, color="tab:red", lw=0.8,
                   ls=":", alpha=0.6, label=f"±std = {al.std_ps:.4f} ps")
        ax.axhline(al.mean_ps - al.std_ps, color="tab:red", lw=0.8,
                   ls=":", alpha=0.6)
        ax.set_title(f"Jitter shifts  (max|Δ| = {al.max_abs_ps:.4f} ps)")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No alignment", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title("Jitter shifts")
    ax.set_xlabel("Trace index")
    ax.set_ylabel("Shift applied (ps)")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Bottom-right: residuals ─────────────────────────────────────────
    ax = axes[1, 1]
    for i, y in enumerate(ds.all_y):
        ax.plot(ds.x_ps, y - ds.y_avg, color=colors[i], lw=0.8, alpha=0.6,
                label=f"Trace {i+1}" if n <= 8 else None)
    ax.axhline(0, color="black", lw=0.8)
    rms = float(np.sqrt(np.mean((ds.all_y - ds.y_avg[None, :]) ** 2)))
    ax.set_title(f"Residuals from average  (RMS = {rms:.4e})")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("ΔAmplitude [nA]")
    ax.grid(True, alpha=0.3)
    if n <= 8:
        ax.legend(fontsize=7)

    fig.suptitle(
        f"Alignment: {method_name}  —  {ds.name}  ({n} traces)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
