"""
thz_tds.config
==============
YAML configuration loading and validation.

Users copy ``config/default_config.yaml``, fill in their paths and parameters,
then call :func:`load_config` to get a validated :class:`TDSConfig` object that
can be passed to every other module.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


class ConfigError(ValueError):
    """Raised when the configuration file is missing required fields or has invalid values."""


# ---------------------------------------------------------------------------
# Nested dataclasses — one per YAML section
# ---------------------------------------------------------------------------

@dataclass
class PathsConfig:
    reference: str
    base_dir: str
    output: str
    samples: List[str] = field(default_factory=list)


@dataclass
class LoadingConfig:
    correction_factor: float = 0.0414


@dataclass
class AlignmentConfig:
    enabled: bool = True
    method: str = "correlation_subsample"
    corr_window_ps: Optional[float] = None
    corr_window_sweep: List[Optional[float]] = field(default_factory=list)

    # Adaptive Gaussian fitting window
    adaptive_fit_window: bool = True
    fit_window_sigma_factor: float = 5.0
    fallback_fit_window_ps: float = 15.0
    min_sigma_ps: float = 0.05
    max_sigma_ps: float = 10.0
    min_fit_points: int = 10
    max_center_shift_ps: float = 5.0


@dataclass
class FftConfig:
    pad_factor: int = 4


@dataclass
class RefractiveIndexConfig:
    thickness_m: float = 1e-3
    f_low_THz: float = 0.4
    f_high_THz: float = 1.1


@dataclass
class FabryPerotConfig:
    thickness_min_m: float = 2.90e-3
    thickness_max_m: float = 3.01e-3
    thickness_step_m: float = 1.0e-6
    echo_count: int = 3
    f_min_THz: float = 0.2
    f_max_THz: float = 1.8


@dataclass
class PlotConfig:
    zoom_xlim_ps: Optional[Tuple[float, float]] = None
    limit_to_1THz: bool = True
    figure_size: Tuple[float, float] = (16.0, 16.0)
    dpi: int = 300
    formats: List[str] = field(default_factory=lambda: ["png", "pdf"])
    n_traces_plot: Optional[int] = 10


@dataclass
class TDSConfig:
    """
    Top-level configuration object returned by :func:`load_config`.

    Attributes
    ----------
    paths : PathsConfig
        File-system paths for reference, base directory, output, and optional
        explicit sample list.
    loading : LoadingConfig
        Amplitude correction factor.
    alignment : AlignmentConfig
        Jitter-alignment settings.
    fft : FftConfig
        FFT zero-padding factor.
    refractive_index : RefractiveIndexConfig
        Frequency band and thickness for n(f)/alpha(f) extraction.
    fabry_perot : FabryPerotConfig
        Thickness grid and frequency band for Liu FP removal.
    plot : PlotConfig
        Figure appearance settings.
    """
    paths: PathsConfig
    loading: LoadingConfig
    alignment: AlignmentConfig
    fft: FftConfig
    refractive_index: RefractiveIndexConfig
    fabry_perot: FabryPerotConfig
    plot: PlotConfig


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> TDSConfig:
    """
    Load and validate a YAML configuration file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    TDSConfig
        Validated configuration object.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ConfigError
        If required fields are missing or values are out of range.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ConfigError(f"Configuration file is not a valid YAML mapping: {path}")

    return _parse(raw)


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------

def _require(d: dict, key: str, section: str) -> object:
    if key not in d:
        raise ConfigError(f"Missing required field '{key}' in [{section}] section.")
    return d[key]


def _parse(raw: dict) -> TDSConfig:
    paths = _parse_paths(raw.get("paths", {}))
    loading = _parse_loading(raw.get("loading", {}))
    alignment = _parse_alignment(raw.get("alignment", {}))
    fft = _parse_fft(raw.get("fft", {}))
    ri = _parse_refractive_index(raw.get("refractive_index", {}))
    fp = _parse_fabry_perot(raw.get("fabry_perot", {}))
    plot = _parse_plot(raw.get("plot", {}))

    return TDSConfig(
        paths=paths,
        loading=loading,
        alignment=alignment,
        fft=fft,
        refractive_index=ri,
        fabry_perot=fp,
        plot=plot,
    )


def _parse_paths(d: dict) -> PathsConfig:
    ref = _require(d, "reference", "paths")
    base = _require(d, "base_dir", "paths")
    out = _require(d, "output", "paths")
    samples = d.get("samples", [])
    return PathsConfig(reference=str(ref), base_dir=str(base), output=str(out), samples=samples)


def _parse_loading(d: dict) -> LoadingConfig:
    cf = float(d.get("correction_factor", 0.0414))
    if cf <= 0:
        raise ConfigError("loading.correction_factor must be positive.")
    return LoadingConfig(correction_factor=cf)


def _parse_alignment(d: dict) -> AlignmentConfig:
    enabled = bool(d.get("enabled", True))
    method = str(d.get("method", "correlation_subsample")).strip()
    valid = {"correlation", "correlation_subsample", "gaussian_minimum_mean", "gaussian_median_integer"}
    if method not in valid:
        raise ConfigError(
            f"alignment.method '{method}' is not valid. Choose from: {sorted(valid)}"
        )
    corr_window_ps = d.get("corr_window_ps", None)
    if corr_window_ps is not None:
        corr_window_ps = float(corr_window_ps)
    raw_sweep = d.get("corr_window_sweep", [])
    corr_window_sweep = [float(v) if v is not None else None for v in raw_sweep]

    # Adaptive Gaussian fitting window parameters
    adaptive_fit_window = bool(d.get("adaptive_fit_window", True))
    fit_window_sigma_factor = float(d.get("fit_window_sigma_factor", 5.0))
    fallback_fit_window_ps = float(d.get("fallback_fit_window_ps", 15.0))
    min_sigma_ps = float(d.get("min_sigma_ps", 0.05))
    max_sigma_ps = float(d.get("max_sigma_ps", 10.0))
    min_fit_points = int(d.get("min_fit_points", 10))
    max_center_shift_ps = float(d.get("max_center_shift_ps", 5.0))

    return AlignmentConfig(
        enabled=enabled,
        method=method,
        corr_window_ps=corr_window_ps,
        corr_window_sweep=corr_window_sweep,
        adaptive_fit_window=adaptive_fit_window,
        fit_window_sigma_factor=fit_window_sigma_factor,
        fallback_fit_window_ps=fallback_fit_window_ps,
        min_sigma_ps=min_sigma_ps,
        max_sigma_ps=max_sigma_ps,
        min_fit_points=min_fit_points,
        max_center_shift_ps=max_center_shift_ps,
    )


def _parse_fft(d: dict) -> FftConfig:
    pad = int(d.get("pad_factor", 4))
    if pad < 1:
        raise ConfigError("fft.pad_factor must be >= 1.")
    return FftConfig(pad_factor=pad)


def _parse_refractive_index(d: dict) -> RefractiveIndexConfig:
    thickness = float(d.get("thickness_m", 1e-3))
    f_low = float(d.get("f_low_THz", 0.4))
    f_high = float(d.get("f_high_THz", 1.1))
    if f_low >= f_high:
        raise ConfigError("refractive_index.f_low_THz must be less than f_high_THz.")
    return RefractiveIndexConfig(thickness_m=thickness, f_low_THz=f_low, f_high_THz=f_high)


def _parse_fabry_perot(d: dict) -> FabryPerotConfig:
    return FabryPerotConfig(
        thickness_min_m=float(d.get("thickness_min_m", 2.90e-3)),
        thickness_max_m=float(d.get("thickness_max_m", 3.01e-3)),
        thickness_step_m=float(d.get("thickness_step_m", 1.0e-6)),
        echo_count=int(d.get("echo_count", 3)),
        f_min_THz=float(d.get("f_min_THz", 0.2)),
        f_max_THz=float(d.get("f_max_THz", 1.8)),
    )


def _parse_plot(d: dict) -> PlotConfig:
    zoom = d.get("zoom_xlim_ps", None)
    if zoom is not None:
        zoom = (float(zoom[0]), float(zoom[1]))
    limit = bool(d.get("limit_to_1THz", True))
    figsize_raw = d.get("figure_size", [16, 16])
    figsize = (float(figsize_raw[0]), float(figsize_raw[1]))
    dpi = int(d.get("dpi", 300))
    formats = list(d.get("formats", ["png", "pdf"]))
    raw_ntp = d.get("n_traces_plot", 10)
    n_traces_plot = None if raw_ntp is None else int(raw_ntp)
    return PlotConfig(
        zoom_xlim_ps=zoom,
        limit_to_1THz=limit,
        figure_size=figsize,
        dpi=dpi,
        formats=formats,
        n_traces_plot=n_traces_plot,
    )
