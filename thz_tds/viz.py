"""
thz_tds.viz
===========
Matplotlib figure functions for THz-TDS data.

Design principles:

* Every function accepts an optional ``ax`` parameter so it can be embedded
  in a caller-managed subplot layout (e.g. inside a Jupyter notebook).
* No computation happens here — all inputs are pre-computed arrays or result
  dicts from :mod:`thz_tds.workflows`.
* :func:`save_figure` saves to multiple formats in one call.

Example::

    import thz_tds

    cfg = thz_tds.load_config("experiment.yaml")
    results = thz_tds.run_spectral_survey(cfg)
    fig = thz_tds.viz.plot_survey_figure(results, cfg)
    thz_tds.viz.save_figure(fig, cfg.paths.output, formats=cfg.plot.formats)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Time domain
# ---------------------------------------------------------------------------

def plot_time_domain(
    datasets,
    labels: Optional[List[str]] = None,
    zoom_xlim: Optional[Tuple[float, float]] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Overlay the averaged time-domain traces of multiple datasets.

    Parameters
    ----------
    datasets : list[THZDataset]
        Datasets to plot.  Each must have ``x_ps`` and ``y_avg``.
    labels : list[str] or None
        Legend labels.  Defaults to ``ds.name`` for each dataset.
    zoom_xlim : (float, float) or None
        If given, set the x-axis limits to this range [ps].
    ax : Axes or None
        Axes to plot into.  A new figure is created if ``None``.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    if labels is None:
        labels = [ds.name for ds in datasets]

    for ds, label in zip(datasets, labels):
        ax.plot(ds.x_ps, ds.y_avg, label=label, linewidth=2)

    ax.set_xlabel("Time [ps]")
    ax.set_ylabel("Amplitude [nA]")
    ax.set_title("THz Time-Domain Signal (Averages)")
    ax.grid(True)
    ax.legend()
    if zoom_xlim is not None:
        ax.set_xlim(*zoom_xlim)

    return ax


def plot_raw_and_average(
    dataset,
    zoom_xlim: Optional[Tuple[float, float]] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot all individual raw traces (faint grey) plus the average (red).

    Parameters
    ----------
    dataset : THZDataset
        Dataset with ``x_ps``, ``all_y``, and ``y_avg`` populated.
    zoom_xlim : (float, float) or None
        Optional x-axis limits [ps].
    ax : Axes or None
        Target axes; a new figure is created if ``None``.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6))

    n_traces = dataset.all_y.shape[0]
    for y in dataset.all_y:
        ax.plot(dataset.x_ps, y, color="gray", alpha=0.10, linewidth=0.8)

    ax.plot(
        dataset.x_ps,
        dataset.y_avg,
        color="red",
        linewidth=2.5,
        label=f"Average (n={n_traces})",
    )

    ax.set_title(f"All Raw Pulses + Average\nSample: {dataset.name} (n={n_traces})")
    ax.set_xlabel("Time [ps]")
    ax.set_ylabel("Amplitude [nA]")
    ax.grid(True)
    ax.legend()

    if zoom_xlim is not None:
        ax.set_xlim(*zoom_xlim)

    return ax


# ---------------------------------------------------------------------------
# Frequency domain
# ---------------------------------------------------------------------------

def plot_power_spectra(
    results: List[dict],
    f_max_THz: float = 2.0,
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Overlay normalised power spectra from a spectral-survey result list.

    Parameters
    ----------
    results : list[dict]
        Output of :func:`thz_tds.workflows.run_spectral_survey`.
    f_max_THz : float
        Upper limit of the frequency axis.
    ax : Axes or None

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    for r in results:
        mask = r["freq_THz"] <= f_max_THz
        label = f"{r['name']} (ref)" if r.get("is_reference") else r["name"]
        ax.plot(r["freq_THz"][mask], r["P_dB"][mask], label=label, linewidth=2)

    ax.set_xlabel("Frequency [THz]")
    ax.set_ylabel("Power [dB]")
    ax.set_title("Normalised Power Spectra")
    ax.grid(True)
    ax.legend()

    return ax


def plot_spectrum_difference(
    results: List[dict],
    f_max_THz: float = 2.0,
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot spectrum-difference curves (sample − reference) in dB.

    Parameters
    ----------
    results : list[dict]
        Output of :func:`thz_tds.workflows.run_spectral_survey`.
        Only entries with ``is_reference=False`` are plotted.
    f_max_THz : float
        Upper limit of the frequency axis.
    ax : Axes or None

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    ax.axhline(0, color="gray", linewidth=1, alpha=0.4)

    for r in results:
        if r.get("is_reference") or r.get("P_diff_dB") is None:
            continue
        freq = r["freq_THz"]
        diff = r["P_diff_dB"]
        mask = freq <= f_max_THz
        ax.plot(freq[mask], diff[mask], label=f"{r['name']} − reference", linewidth=2)

    ax.set_xlabel("Frequency [THz]")
    ax.set_ylabel("Δ Power [dB]")
    ax.set_title("Spectrum Difference (Sample – Reference)")
    ax.grid(True)
    ax.legend()

    return ax


# ---------------------------------------------------------------------------
# Optical constants
# ---------------------------------------------------------------------------

def plot_refractive_index(
    results: List[dict],
    f_min_THz: float = 0.0,
    f_max_THz: float = 2.0,
    ylim: Optional[Tuple[float, float]] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot n(f) for each sample in *results*.

    Parameters
    ----------
    results : list[dict]
        Output of :func:`thz_tds.workflows.run_refractive_survey`.
        Each dict must contain ``f_THz`` and ``n_f``.
    f_min_THz, f_max_THz : float
        Frequency axis limits.
    ylim : (float, float) or None
        Optional y-axis limits.
    ax : Axes or None

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for r in results:
        f = r["f_THz"]
        n = r["n_f"]
        mask = (f >= f_min_THz) & (f <= f_max_THz)
        ax.plot(f[mask], n[mask], label=r["name"], linewidth=2)

    ax.set_xlabel("Frequency [THz]")
    ax.set_ylabel("Refractive index n(f)")
    ax.set_title("Refractive Index")
    ax.grid(True)
    ax.legend()
    if ylim is not None:
        ax.set_ylim(*ylim)

    return ax


def plot_absorption(
    results: List[dict],
    f_min_THz: float = 0.0,
    f_max_THz: float = 2.0,
    ylim: Optional[Tuple[float, float]] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot α(f) [cm⁻¹] for each sample in *results*.

    Parameters
    ----------
    results : list[dict]
        Output of :func:`thz_tds.workflows.run_refractive_survey`.
        Each dict must contain ``f_THz`` and ``alpha_cm``.
    f_min_THz, f_max_THz : float
        Frequency axis limits.
    ylim : (float, float) or None
        Optional y-axis limits.
    ax : Axes or None

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for r in results:
        f = r["f_THz"]
        alpha = r["alpha_cm"]
        mask = (f >= f_min_THz) & (f <= f_max_THz)
        ax.plot(f[mask], alpha[mask], label=r["name"], linewidth=2)

    ax.set_xlabel("Frequency [THz]")
    ax.set_ylabel("Absorption coefficient α [cm⁻¹]")
    ax.set_title("Absorption Coefficient")
    ax.grid(True)
    ax.legend()
    if ylim is not None:
        ax.set_ylim(*ylim)

    return ax


# ---------------------------------------------------------------------------
# Fabry-Perot
# ---------------------------------------------------------------------------

def plot_fp_fit(
    ref_trace,
    sam_trace,
    fit_result,
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Time-domain comparison: measured reference, measured sample, Liu-simulated
    signal, and FP-cleaned signal.

    Parameters
    ----------
    ref_trace : THzTrace
    sam_trace : THzTrace
    fit_result : LiuFitResult
    ax : Axes or None

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    t = ref_trace.time_ps
    ax.plot(t, ref_trace.y, label="Reference", linewidth=1.5, alpha=0.8)
    ax.plot(t, sam_trace.y, label="Measured sample", linewidth=1.5, alpha=0.8)
    ax.plot(t, fit_result.y_sim, "--", label="Liu simulated", linewidth=1.5)
    ax.plot(t, fit_result.y_clean, label="FP-cleaned", linewidth=2)

    ax.set_xlabel("Time [ps]")
    ax.set_ylabel("Amplitude [nA]")
    ax.set_title(
        f"Fabry-Perot Fit  |  "
        f"d = {fit_result.thickness_m * 1e3:.3f} mm  |  "
        f"n_av = {fit_result.nav:.4f}  |  "
        f"R² = {fit_result.r2:.4f}"
    )
    ax.grid(True)
    ax.legend()

    return ax


# ---------------------------------------------------------------------------
# Survey figure (4-panel)
# ---------------------------------------------------------------------------

def plot_survey_figure(results: List[dict], cfg) -> Figure:
    """
    Create the standard 4-panel survey figure.

    Panels:
    1. Time-domain averages (zoomed if ``cfg.plot.zoom_xlim_ps`` is set)
    2. Full power spectra
    3. Power spectra limited to ≤ 1 THz (if ``cfg.plot.limit_to_1THz``)
    4. Spectrum difference (sample − reference)

    Parameters
    ----------
    results : list[dict]
        Output of :func:`thz_tds.workflows.run_spectral_survey`.
    cfg : TDSConfig
        Configuration; ``cfg.plot`` controls layout options.

    Returns
    -------
    Figure
    """
    figsize = tuple(cfg.plot.figure_size)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 1, hspace=0.45)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    zoom = cfg.plot.zoom_xlim_ps

    # Panel 1: time domain
    for r in results:
        n_traces = r["all_y"].shape[0] if r.get("all_y") is not None else "?"
        suffix = " (ref)" if r.get("is_reference") else ""
        label = f"{r['name']}{suffix} (n={n_traces})"
        ax1.plot(r["x_ps"], r["y_avg"], label=label, linewidth=2)
    ax1.set_title("Time Domain THz Signal (Averages)")
    ax1.set_xlabel("Time [ps]")
    ax1.set_ylabel("Amplitude [nA]")
    ax1.grid(True)
    ax1.legend()
    if zoom is not None:
        ax1.set_xlim(*zoom)

    # Panel 2: full spectra
    for r in results:
        suffix = " (ref)" if r.get("is_reference") else ""
        ax2.plot(r["freq_THz"], r["P_dB"], label=f"{r['name']}{suffix}", linewidth=2)
    ax2.set_title("Normalised Power Spectrum (Full)")
    ax2.set_xlabel("Frequency [THz]")
    ax2.set_ylabel("Power [dB]")
    ax2.grid(True)
    ax2.legend()

    # Panel 3: spectra limited to ≤ 1 THz
    for r in results:
        f, P = r["freq_THz"], r["P_dB"]
        suffix = " (ref)" if r.get("is_reference") else ""
        m = f <= 1.0
        ax3.plot(f[m], P[m], label=f"{r['name']}{suffix} (≤1 THz)", linewidth=2)
    if cfg.plot.limit_to_1THz:
        ax3.set_xlim(0, 1.0)
    ax3.set_title("Normalised Power Spectrum (≤ 1 THz)")
    ax3.set_xlabel("Frequency [THz]")
    ax3.set_ylabel("Power [dB]")
    ax3.grid(True)
    ax3.legend()

    # Panel 4: difference curves
    ax4.axhline(0, color="gray", linewidth=1, alpha=0.4)
    for r in results:
        if r.get("is_reference") or r.get("P_diff_dB") is None:
            continue
        ax4.plot(
            r["freq_THz"], r["P_diff_dB"],
            label=f"{r['name']} – reference", linewidth=2,
        )
    ax4.set_title("Spectrum Difference (Sample – Reference)")
    ax4.set_xlabel("Frequency [THz]")
    ax4.set_ylabel("Δ Power [dB]")
    ax4.grid(True)
    ax4.legend()

    return fig


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_figure(
    fig: Figure,
    output_dir: str | Path,
    stem: str = "figure",
    dpi: int = 300,
    formats: Sequence[str] = ("png", "pdf"),
) -> List[Path]:
    """
    Save a figure to one or more formats.

    Parameters
    ----------
    fig : Figure
    output_dir : str or Path
        Directory where files are saved (created if needed).
    stem : str
        Base filename without extension.
    dpi : int
        Resolution for raster formats.
    formats : sequence of str
        File extensions to save (e.g. ``["png", "pdf"]``).

    Returns
    -------
    list[Path]
        Paths of the saved files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    for fmt in formats:
        p = output_dir / f"{stem}.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        log.info("Saved figure to %s", p)
        paths.append(p)

    return paths
