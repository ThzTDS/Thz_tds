"""
thz_tds
=======
Professional Python package for Terahertz Time-Domain Spectroscopy (THz-TDS)
data processing.

Quick start
-----------
>>> import thz_tds
>>> cfg = thz_tds.load_config("experiment.yaml")
>>> ref = thz_tds.THZDataset(cfg.paths.reference,
...     correction_factor=cfg.loading.correction_factor,
...     align=cfg.alignment.enabled,
...     align_method=cfg.alignment.method)
>>> sam = thz_tds.THZDataset(cfg.paths.samples[0], ...)
>>> f, n, T, phi, alpha = thz_tds.compute_refractive_index(ref, sam,
...     thickness_m=cfg.refractive_index.thickness_m)

Batch workflows
---------------
>>> results = thz_tds.run_spectral_survey(cfg)
>>> fig = thz_tds.viz.plot_survey_figure(results, cfg)
>>> thz_tds.viz.save_figure(fig, cfg.paths.output)
"""

from ._version import __version__

# Config
from .config import load_config, TDSConfig, ConfigError

# Core data class
from .dataset import THZDataset, scan_samples

# Data containers
from .fabry_perot import THzTrace, LiuFitResult

# Spectral helpers
from .spectral import fft_field, power_spectrum_dB

# Physics computations
from .refractive import compute_refractive_index, compute_refractive_index_arrays
from .fabry_perot import remove_fabry_perot, liu_fit_over_thickness_grid, extract_optical_constants

# High-level workflows
from .workflows import run_spectral_survey, run_refractive_survey, run_fp_survey

# Visualisation — import the module (not individual functions) to keep the
# top-level namespace clean.  Access as: thz_tds.viz.plot_time_domain(...)
from . import viz

__all__ = [
    "__version__",
    # Config
    "load_config",
    "TDSConfig",
    "ConfigError",
    # Data classes
    "THZDataset",
    "scan_samples",
    "THzTrace",
    "LiuFitResult",
    # Spectral
    "fft_field",
    "power_spectrum_dB",
    # Physics
    "compute_refractive_index",
    "compute_refractive_index_arrays",
    "remove_fabry_perot",
    "liu_fit_over_thickness_grid",
    "extract_optical_constants",
    # Workflows
    "run_spectral_survey",
    "run_refractive_survey",
    "run_fp_survey",
    # Sub-module
    "viz",
]
