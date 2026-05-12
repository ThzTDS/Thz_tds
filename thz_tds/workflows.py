"""
thz_tds.workflows
=================
High-level batch processing pipelines for multi-sample experiments.

Each function loads data, runs the requested computation, and returns a list
of result dicts — with **no** side effects (no plots, no files written).
Visualisation and saving are handled separately by :mod:`thz_tds.viz`.

Typical usage::

    import thz_tds

    cfg = thz_tds.load_config("experiment.yaml")
    results = thz_tds.run_spectral_survey(cfg)
    fig = thz_tds.viz.plot_survey_figure(results, cfg)
    thz_tds.viz.save_figure(fig, cfg.paths.output)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import TDSConfig
from .dataset import THZDataset, scan_samples
from .fabry_perot import THzTrace, remove_fabry_perot
from .refractive import compute_refractive_index
from .spectral import power_spectrum_dB, ensure_common_time_axis

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_reference(cfg: TDSConfig) -> THZDataset:
    return THZDataset(
        path=cfg.paths.reference,
        correction_factor=cfg.loading.correction_factor,
        align=cfg.alignment.enabled,
        align_method=cfg.alignment.method,
        corr_window_ps=cfg.alignment.corr_window_ps,
    )


def _load_samples(cfg: TDSConfig) -> List[THZDataset]:
    """Load all sample datasets from the base directory or the explicit list."""
    if cfg.paths.samples:
        datasets = []
        for p in cfg.paths.samples:
            try:
                ds = THZDataset(
                    path=p,
                    correction_factor=cfg.loading.correction_factor,
                    align=cfg.alignment.enabled,
                    align_method=cfg.alignment.method,
                    corr_window_ps=cfg.alignment.corr_window_ps,
                )
                datasets.append(ds)
            except RuntimeError as exc:
                log.warning("Skipping sample %s: %s", p, exc)
        return datasets

    return scan_samples(
        base_dir=cfg.paths.base_dir,
        correction_factor=cfg.loading.correction_factor,
        exclude_reference=True,
        align=cfg.alignment.enabled,
        align_method=cfg.alignment.method,
        corr_window_ps=cfg.alignment.corr_window_ps,
    )


# ---------------------------------------------------------------------------
# Spectral survey
# ---------------------------------------------------------------------------

def run_spectral_survey(cfg: TDSConfig) -> List[Dict]:
    """
    Load reference and all samples, compute power spectra and dB differences.

    Parameters
    ----------
    cfg : TDSConfig
        Configuration loaded with :func:`thz_tds.load_config`.

    Returns
    -------
    list[dict]
        One dict per sample, plus the first element being the reference.
        Each dict contains:

        * ``name`` – folder name
        * ``x_ps`` – time axis [ps]
        * ``y_avg`` – averaged electric-field amplitude
        * ``all_y`` – matrix of individual traces
        * ``freq_THz`` – frequency axis [THz]
        * ``P_dB`` – normalised power spectrum [dB]
        * ``P_diff_dB`` – spectrum difference relative to reference [dB]
          (``None`` for the reference entry)
        * ``is_reference`` – ``True`` for the reference entry
    """
    ref = _load_reference(cfg)
    samples = _load_samples(cfg)

    if not samples:
        raise RuntimeError(
            f"No sample folders found in {cfg.paths.base_dir!r}. "
            "Check paths.base_dir in your configuration."
        )

    # Reference spectrum (the baseline for difference curves)
    freq_THz_ref, P_ref_dB = power_spectrum_dB(
        ref.x_ps, ref.y_avg, pad_factor=cfg.fft.pad_factor
    )

    results: List[Dict] = []

    # Reference entry
    results.append({
        "name": ref.name,
        "x_ps": ref.x_ps,
        "y_avg": ref.y_avg,
        "all_y": ref.all_y,
        "freq_THz": freq_THz_ref,
        "P_dB": P_ref_dB,
        "P_diff_dB": None,
        "is_reference": True,
    })

    # Sample entries
    for sam in samples:
        try:
            _, _, y_sam_aligned = ensure_common_time_axis(
                ref.x_ps, ref.y_avg, sam.x_ps, sam.y_avg
            )
            freq_THz_sam, P_sam_dB = power_spectrum_dB(
                ref.x_ps, y_sam_aligned, pad_factor=cfg.fft.pad_factor
            )
            P_diff_dB = P_sam_dB - P_ref_dB

            results.append({
                "name": sam.name,
                "x_ps": sam.x_ps,
                "y_avg": y_sam_aligned,
                "all_y": sam.all_y,
                "freq_THz": freq_THz_sam,
                "P_dB": P_sam_dB,
                "P_diff_dB": P_diff_dB,
                "is_reference": False,
            })
        except Exception as exc:
            log.warning("Skipping sample %s in spectral survey: %s", sam.name, exc)

    return results


# ---------------------------------------------------------------------------
# Refractive index survey
# ---------------------------------------------------------------------------

def run_refractive_survey(
    cfg: TDSConfig,
    thickness_m: float | None = None,
    f_low_THz: float | None = None,
    f_high_THz: float | None = None,
) -> List[Dict]:
    """
    Load reference and all samples, compute n(f) and α(f) for each.

    Parameters default to the values in ``cfg.refractive_index`` when not
    explicitly provided.

    Parameters
    ----------
    cfg : TDSConfig
        Configuration loaded with :func:`thz_tds.load_config`.
    thickness_m : float or None
        Override ``cfg.refractive_index.thickness_m``.
    f_low_THz : float or None
        Override ``cfg.refractive_index.f_low_THz``.
    f_high_THz : float or None
        Override ``cfg.refractive_index.f_high_THz``.

    Returns
    -------
    list[dict]
        One dict per sample containing:

        * ``name``, ``x_ps``, ``y_avg``
        * ``f_THz``, ``n_f``, ``T``, ``phi_full``, ``alpha_cm``
    """
    thickness_m = thickness_m or cfg.refractive_index.thickness_m
    f_low_THz = f_low_THz or cfg.refractive_index.f_low_THz
    f_high_THz = f_high_THz or cfg.refractive_index.f_high_THz

    ref = _load_reference(cfg)
    samples = _load_samples(cfg)

    results: List[Dict] = []
    for sam in samples:
        try:
            f_THz, n_f, T, phi_full, alpha_cm = compute_refractive_index(
                ref, sam,
                thickness_m=thickness_m,
                f_low_THz=f_low_THz,
                f_high_THz=f_high_THz,
            )
            results.append({
                "name": sam.name,
                "x_ps": sam.x_ps,
                "y_avg": sam.y_avg,
                "f_THz": f_THz,
                "n_f": n_f,
                "T": T,
                "phi_full": phi_full,
                "alpha_cm": alpha_cm,
            })
        except Exception as exc:
            log.warning("Skipping sample %s in refractive survey: %s", sam.name, exc)

    if not results:
        raise RuntimeError("No samples were processed successfully in the refractive survey.")

    return results


# ---------------------------------------------------------------------------
# Fabry-Perot survey
# ---------------------------------------------------------------------------

def run_fp_survey(
    cfg: TDSConfig,
    thickness_min_m: float | None = None,
    thickness_max_m: float | None = None,
    thickness_step_m: float | None = None,
    echo_count: int | None = None,
    f_min_THz: float | None = None,
    f_max_THz: float | None = None,
) -> List[Dict]:
    """
    Load reference and all samples, run FP removal and optical-constant
    extraction for each.

    All parameters default to ``cfg.fabry_perot.*`` when not given.

    Parameters
    ----------
    cfg : TDSConfig
        Configuration loaded with :func:`thz_tds.load_config`.
    thickness_min_m, thickness_max_m, thickness_step_m : float or None
        Override the thickness grid from the config.
    echo_count : int or None
        Number of FP echoes; overrides ``cfg.fabry_perot.echo_count``.
    f_min_THz, f_max_THz : float or None
        Frequency range for optical-constant extraction.

    Returns
    -------
    list[dict]
        One dict per sample containing:

        * ``name``
        * ``fit`` (:class:`~thz_tds.fabry_perot.LiuFitResult`)
        * ``clean_trace`` (:class:`~thz_tds.fabry_perot.THzTrace`)
        * ``spectrum`` (dict from :func:`~thz_tds.fabry_perot.extract_optical_constants`)
    """
    fp = cfg.fabry_perot
    thickness_min_m = thickness_min_m or fp.thickness_min_m
    thickness_max_m = thickness_max_m or fp.thickness_max_m
    thickness_step_m = thickness_step_m or fp.thickness_step_m
    echo_count = echo_count or fp.echo_count
    f_min_THz = f_min_THz or fp.f_min_THz
    f_max_THz = f_max_THz or fp.f_max_THz

    ref = _load_reference(cfg)
    samples = _load_samples(cfg)

    ref_trace = THzTrace(time_ps=ref.x_ps, y=ref.y_avg)

    results: List[Dict] = []
    for sam in samples:
        try:
            sam_trace = THzTrace(time_ps=sam.x_ps, y=sam.y_avg)
            out = remove_fabry_perot(
                ref=ref_trace,
                sam=sam_trace,
                thickness_min_m=thickness_min_m,
                thickness_max_m=thickness_max_m,
                thickness_step_m=thickness_step_m,
                echo_count=echo_count,
                f_min_THz=f_min_THz,
                f_max_THz=f_max_THz,
            )
            results.append({"name": sam.name, **out})
        except Exception as exc:
            log.warning("Skipping sample %s in FP survey: %s", sam.name, exc)

    if not results:
        raise RuntimeError("No samples were processed successfully in the FP survey.")

    return results
