"""Sparse-only THz extraction for every sample in the YAML configuration.

No broadband H(f), n(f), or alpha(f) is calculated.  The complex Fourier
coefficient is evaluated directly at the configured frequencies.  Phase-pair
delays provide n, attenuation provides alpha at the sparse points, and simple
models provide approximation curves for plotting:

    n(f)     = n0 + n_slope * (f - f_center)
    alpha(f) = alpha0 + beta * f**2

The dense plot lines are model evaluations, not broadband measurements.
"""

from __future__ import annotations

import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import least_squares
from tqdm import tqdm

import thz_tds
from thz_tds.refractive_fp import fp_transfer_function_normal_incidence
from thz_tds.spectral import ensure_common_time_axis


CONFIG_DEFAULT = (
    Path(__file__).parent.parent.parent
    / "config"
    / "all_samples_sparse_fp_batch.yaml"
)
C0 = 2.99792458e8


def _label(name: str) -> str:
    parts = name.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else parts[0]


def load_yaml(path: str | Path) -> dict:
    with Path(path).open() as fh:
        return yaml.safe_load(fh)


def discover_pairs(
    sample_dir: Path,
    air_prefix: str,
    point_prefix: str,
    n_points: int | None,
) -> list[tuple[int, Path, Path]]:
    """Find matching airN and pointN folders."""

    air_re = re.compile(rf"^{re.escape(air_prefix)}(\d+)$", re.IGNORECASE)
    pairs: list[tuple[int, Path, Path]] = []
    for child in sorted(sample_dir.iterdir()):
        if not child.is_dir():
            continue
        match = air_re.match(child.name)
        if not match:
            continue
        index = int(match.group(1))
        point_path = sample_dir / f"{point_prefix}{index}"
        if point_path.is_dir():
            pairs.append((index, child, point_path))

    pairs.sort(key=lambda item: item[0])
    return pairs if n_points is None else pairs[:n_points]


def transfer_at_sparse_frequencies(
    ref_ds,
    sam_ds,
    frequencies_THz: np.ndarray,
    remove_dc: bool,
    reference_floor_rel: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the DFT only at the requested frequencies and return H=Es/Er."""

    time_ps, y_ref, y_sam = ensure_common_time_axis(
        ref_ds.x_ps,
        ref_ds.y_avg,
        sam_ds.x_ps,
        sam_ds.y_avg,
    )
    time_s = (np.asarray(time_ps, dtype=float) - float(time_ps[0])) * 1e-12
    y_ref = np.asarray(y_ref, dtype=float)
    y_sam = np.asarray(y_sam, dtype=float)
    if remove_dc:
        y_ref = y_ref - np.mean(y_ref)
        y_sam = y_sam - np.mean(y_sam)

    f_Hz = np.asarray(frequencies_THz, dtype=float) * 1e12
    kernel = np.exp(-2j * np.pi * f_Hz[:, None] * time_s[None, :])
    E_ref = kernel @ y_ref
    E_sam = kernel @ y_sam

    floor = max(
        float(reference_floor_rel) * float(np.max(np.abs(E_ref))),
        np.finfo(float).tiny,
    )
    weak = np.abs(E_ref) <= floor
    if np.any(weak):
        bad = ", ".join(f"{f:.6f}" for f in frequencies_THz[weak])
        raise RuntimeError(f"Reference signal is too weak at [{bad}] THz")

    return E_ref, E_sam, E_sam / E_ref


def _phase_pair_indices(
    frequencies_THz: np.ndarray,
    sparse_cfg: dict,
) -> list[tuple[int, int]]:
    configured = sparse_cfg.get("phase_pairs_THz")
    if configured is None:
        configured = (
            [frequencies_THz[:2], frequencies_THz[-2:]]
            if frequencies_THz.size >= 4
            else [frequencies_THz[:2]]
        )

    pairs: list[tuple[int, int]] = []
    for pair in configured:
        if len(pair) != 2:
            raise ValueError("Every phase_pairs_THz entry needs two frequencies")
        i = int(np.argmin(np.abs(frequencies_THz - float(pair[0]))))
        j = int(np.argmin(np.abs(frequencies_THz - float(pair[1]))))
        if i == j:
            raise ValueError(f"Phase pair {pair!r} maps to one frequency")
        pairs.append((min(i, j), max(i, j)))
    return pairs


def calculate_phase_pairs(
    frequencies_THz: np.ndarray,
    H: np.ndarray,
    thickness_m: float,
    sparse_cfg: dict,
) -> list[dict]:
    """Calculate wrapped phase difference, tau, and group index for each pair."""

    n_bounds = tuple(map(float, sparse_cfg.get("n_bounds", [1.0, 3.0])))
    target_n = float(sparse_cfg.get("expected_n", np.mean(n_bounds)))
    max_cycles = int(sparse_cfg.get("max_phase_cycles", 20))
    results: list[dict] = []

    for i, j in _phase_pair_indices(frequencies_THz, sparse_cfg):
        df_Hz = (float(frequencies_THz[j]) - float(frequencies_THz[i])) * 1e12
        delta_wrapped = float(np.angle(H[j] * np.conj(H[i])))
        candidates = []

        for cycle in range(-max_cycles, max_cycles + 1):
            delta = delta_wrapped + 2.0 * np.pi * cycle
            # NumPy FFT convention: phi = -omega*(n-1)*d/c.
            tau_s = -delta / (2.0 * np.pi * df_Hz)
            n_group = 1.0 + C0 * tau_s / thickness_m
            if n_bounds[0] <= n_group <= n_bounds[1]:
                candidates.append((abs(n_group - target_n), cycle, delta, n_group))

        if not candidates:
            raise RuntimeError(
                "No phase branch gives n inside "
                f"{n_bounds} for {frequencies_THz[i]:.6f}-"
                f"{frequencies_THz[j]:.6f} THz"
            )

        _, cycle, delta, n_group = min(candidates, key=lambda row: row[0])
        tau_s = -delta / (2.0 * np.pi * df_Hz)
        result = {
            "f1_THz": float(frequencies_THz[i]),
            "f2_THz": float(frequencies_THz[j]),
            "center_THz": 0.5
            * (float(frequencies_THz[i]) + float(frequencies_THz[j])),
            "delta_f_GHz": df_Hz / 1e9,
            "delta_phi_wrapped_rad": delta_wrapped,
            "cycle": int(cycle),
            "delta_phi_selected_rad": float(delta),
            "tau_ps": float(tau_s * 1e12),
            "n_group": float(n_group),
        }
        results.append(result)
        target_n = result["n_group"]

    return results


def fit_n_model(
    frequencies_THz: np.ndarray,
    pair_results: list[dict],
    sparse_cfg: dict,
) -> tuple[float, float]:
    """Fit n(f)=n0+s*(f-fc) from the phase-pair group indices."""

    n_bounds = tuple(map(float, sparse_cfg.get("n_bounds", [1.0, 3.0])))
    slope_bounds = tuple(
        map(float, sparse_cfg.get("n_slope_bounds_per_THz", [-0.5, 0.5]))
    )
    f_center = float(np.mean(frequencies_THz))

    if len(pair_results) == 1:
        n0, slope = pair_results[0]["n_group"], 0.0
    else:
        # If n=n0+s*(f-fc), a phase-pair slope measures
        # n_group(fp)=n0+s*(2*fp-fc).
        design = np.array(
            [
                [1.0, 2.0 * row["center_THz"] - f_center]
                for row in pair_results
            ],
            dtype=float,
        )
        target = np.array([row["n_group"] for row in pair_results])
        n0, slope = np.linalg.lstsq(design, target, rcond=None)[0]

    return float(np.clip(n0, *n_bounds)), float(np.clip(slope, *slope_bounds))


def material_curves(
    frequencies_THz: np.ndarray,
    n0: float,
    n_slope_per_THz: float,
    alpha0_cm: float,
    beta_cm_per_THz2: float,
    center_THz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the approximation equations; this does not calculate H(f)."""

    f = np.asarray(frequencies_THz, dtype=float)
    n = n0 + n_slope_per_THz * (f - center_THz)
    alpha = alpha0_cm + beta_cm_per_THz2 * f**2
    return n, alpha


def transfer_from_material(
    frequencies_THz: np.ndarray,
    n: np.ndarray,
    alpha_cm: np.ndarray,
    thickness_m: float,
    model: str,
) -> np.ndarray:
    """Calculate H only at supplied sparse frequencies."""

    f_Hz = np.asarray(frequencies_THz, dtype=float) * 1e12
    omega = 2.0 * np.pi * f_Hz
    kappa = np.asarray(alpha_cm) * 100.0 * C0 / (4.0 * np.pi * f_Hz)

    if model == "fp":
        return fp_transfer_function_normal_incidence(
            omega=omega,
            n=np.asarray(n, dtype=float),
            kappa=kappa,
            thickness_m=thickness_m,
            n_air=1.0,
        )
    if model == "no_fp":
        n_tilde = np.asarray(n, dtype=float) - 1j * kappa
        fresnel = 4.0 * n_tilde / (1.0 + n_tilde) ** 2
        propagation = np.exp(
            -1j * omega * (n_tilde - 1.0) * thickness_m / C0
        )
        return fresnel * propagation
    raise ValueError("selected_frequencies.model must be 'fp' or 'no_fp'")


def fit_alpha_points(
    frequencies_THz: np.ndarray,
    H: np.ndarray,
    n_points: np.ndarray,
    thickness_m: float,
    sparse_cfg: dict,
) -> np.ndarray:
    """Fit one alpha value from |H| at each measured sparse frequency."""

    model = str(sparse_cfg.get("model", "fp")).lower()
    alpha_bounds = tuple(
        map(float, sparse_cfg.get("point_alpha_bounds_cm", [0.0, 300.0]))
    )
    starts = max(2, int(sparse_cfg.get("absorption_multistart_count", 9)))
    max_nfev = int(sparse_cfg.get("max_nfev", 1000))
    residual_floor = float(sparse_cfg.get("relative_residual_floor", 0.05))
    alpha_points = np.empty_like(frequencies_THz)

    for k, (frequency, H_target, n_value) in enumerate(
        zip(frequencies_THz, H, n_points)
    ):
        scale = max(abs(H_target), residual_floor)

        def residual(alpha_vector: np.ndarray) -> np.ndarray:
            H_model = transfer_from_material(
                np.array([frequency]),
                np.array([n_value]),
                np.array([alpha_vector[0]]),
                thickness_m,
                model,
            )[0]
            return np.array([(abs(H_model) - abs(H_target)) / scale])

        best = None
        for start in np.linspace(alpha_bounds[0], alpha_bounds[1], starts):
            result = least_squares(
                residual,
                x0=np.array([start]),
                bounds=(np.array([alpha_bounds[0]]), np.array([alpha_bounds[1]])),
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
                max_nfev=max_nfev,
            )
            if best is None or result.cost < best.cost:
                best = result

        assert best is not None
        alpha_points[k] = best.x[0]

    return alpha_points


def fit_alpha_model(
    frequencies_THz: np.ndarray,
    alpha_points: np.ndarray,
    sparse_cfg: dict,
) -> tuple[float, float]:
    """Fit alpha(f)=alpha0+beta*f^2 through the sparse alpha values."""

    alpha0_bounds = tuple(
        map(float, sparse_cfg.get("alpha0_bounds_cm", [0.0, 100.0]))
    )
    beta_bounds = tuple(
        map(float, sparse_cfg.get("beta_bounds_cm_per_THz2", [0.0, 500.0]))
    )
    design = np.column_stack(
        [np.ones_like(frequencies_THz), frequencies_THz**2]
    )
    initial = np.linalg.lstsq(design, alpha_points, rcond=None)[0]
    initial = np.array(
        [
            np.clip(initial[0], *alpha0_bounds),
            np.clip(initial[1], *beta_bounds),
        ]
    )
    result = least_squares(
        lambda values: design @ values - alpha_points,
        x0=initial,
        bounds=(
            np.array([alpha0_bounds[0], beta_bounds[0]]),
            np.array([alpha0_bounds[1], beta_bounds[1]]),
        ),
    )
    return float(result.x[0]), float(result.x[1])


def extract_sparse_material(
    frequencies_THz: np.ndarray,
    H: np.ndarray,
    thickness_m: float,
    sparse_cfg: dict,
) -> dict:
    """Calculate tau, n, and alpha using only the supplied sparse H values."""

    frequencies_THz = np.asarray(frequencies_THz, dtype=float)
    H = np.asarray(H, dtype=complex)
    if frequencies_THz.size < 3 or H.shape != frequencies_THz.shape:
        raise ValueError("At least three matching frequency and H values are required")
    if np.any(np.diff(frequencies_THz) <= 0.0):
        raise ValueError("Sparse frequencies must be strictly increasing")

    pair_results = calculate_phase_pairs(
        frequencies_THz, H, thickness_m, sparse_cfg
    )
    n0, n_slope = fit_n_model(frequencies_THz, pair_results, sparse_cfg)
    center_THz = float(np.mean(frequencies_THz))
    n_points, _ = material_curves(
        frequencies_THz, n0, n_slope, 0.0, 0.0, center_THz
    )
    alpha_points = fit_alpha_points(
        frequencies_THz, H, n_points, thickness_m, sparse_cfg
    )
    alpha0, beta = fit_alpha_model(frequencies_THz, alpha_points, sparse_cfg)
    _, alpha_fit_points = material_curves(
        frequencies_THz, n0, n_slope, alpha0, beta, center_THz
    )
    H_fit = transfer_from_material(
        frequencies_THz,
        n_points,
        alpha_fit_points,
        thickness_m,
        str(sparse_cfg.get("model", "fp")).lower(),
    )
    scale = np.maximum(
        np.abs(H), float(sparse_cfg.get("relative_residual_floor", 0.05))
    )

    return {
        "frequencies_THz": frequencies_THz,
        "H_meas": H,
        "H_fit": H_fit,
        "phase_pairs": pair_results,
        "pair_centers_THz": np.array([x["center_THz"] for x in pair_results]),
        "tau_ps": np.array([x["tau_ps"] for x in pair_results]),
        "n_pair": np.array([x["n_group"] for x in pair_results]),
        "n_points": n_points,
        "alpha_points_cm": alpha_points,
        "alpha_fit_points_cm": alpha_fit_points,
        "n0": n0,
        "n_slope_per_THz": n_slope,
        "alpha0_cm": alpha0,
        "beta_cm_per_THz2": beta,
        "center_THz": center_THz,
        "relative_complex_rmse": float(
            np.sqrt(np.mean(np.abs((H_fit - H) / scale) ** 2))
        ),
    }


def _dataset_kwargs(loading: dict, alignment: dict) -> dict:
    return {
        "correction_factor": loading["correction_factor"],
        "align": alignment["enabled"],
        "align_method": alignment["method"],
        "corr_window_ps": alignment.get("corr_window_ps"),
        "adaptive_fit_window": alignment.get("adaptive_fit_window", True),
        "fit_window_sigma_factor": alignment.get("fit_window_sigma_factor", 5.0),
        "fallback_fit_window_ps": alignment.get("fallback_fit_window_ps", 15.0),
        "min_sigma_ps": alignment.get("min_sigma_ps", 0.05),
        "max_sigma_ps": alignment.get("max_sigma_ps", 10.0),
        "min_fit_points": alignment.get("min_fit_points", 10),
        "max_center_shift_ps": alignment.get("max_center_shift_ps", 5.0),
        "max_traces": loading.get("max_traces"),
    }


def run_sample(
    sample_entry: dict,
    cfg: dict,
    output_dir: Path,
    timestamp: str,
) -> dict | None:
    """Process one sample without any broadband spectral calculation."""

    name = sample_entry["name"]
    sample_dir = Path(sample_entry["dir"])
    thickness_m = float(sample_entry["thickness_m"])
    loading = cfg["loading"]
    alignment = cfg["alignment"]
    sparse_cfg = cfg["selected_frequencies"]
    frequencies_THz = np.asarray(sparse_cfg["frequencies_THz"], dtype=float)

    pairs = discover_pairs(
        sample_dir,
        alignment.get("air_prefix", "air"),
        alignment.get("point_prefix", "point"),
        alignment.get("n_points"),
    )
    if not pairs:
        tqdm.write(f"  [{name}] No matched air/point folders — skipping")
        return None

    tqdm.write(
        f"\n{'=' * 70}\n"
        f"  {name} | {len(pairs)} positions | {thickness_m * 1e3:.3f} mm\n"
        f"{'=' * 70}"
    )
    kwargs = _dataset_kwargs(loading, alignment)
    results: list[dict] = []

    for index, air_path, point_path in pairs:
        position = f"{alignment.get('point_prefix', 'point')}{index}"
        ref = thz_tds.THZDataset(path=air_path, **kwargs)
        sam = thz_tds.THZDataset(path=point_path, **kwargs)
        E_ref, E_sam, H = transfer_at_sparse_frequencies(
            ref,
            sam,
            frequencies_THz,
            bool(sparse_cfg.get("remove_dc", True)),
            float(sparse_cfg.get("reference_floor_rel", 1e-8)),
        )
        result = extract_sparse_material(
            frequencies_THz, H, thickness_m, sparse_cfg
        )
        result.update({"position": position, "E_ref": E_ref, "E_sam": E_sam})
        results.append(result)

        pairs_text = ", ".join(
            f"tau={x['tau_ps']:.3f} ps, n={x['n_group']:.4f}"
            for x in result["phase_pairs"]
        )
        tqdm.write(
            f"  {position}: {pairs_text}; alpha0={result['alpha0_cm']:.4f}, "
            f"beta={result['beta_cm_per_THz2']:.4f}"
        )

    pair_centers = results[0]["pair_centers_THz"]
    H_all = np.array([row["H_meas"] for row in results])
    n_pair_all = np.array([row["n_pair"] for row in results])
    tau_all = np.array([row["tau_ps"] for row in results])
    alpha_all = np.array([row["alpha_points_cm"] for row in results])

    plot_low, plot_high = map(
        float,
        sparse_cfg.get(
            "plot_frequency_range_THz",
            [float(frequencies_THz[0]), float(frequencies_THz[-1])],
        ),
    )
    plot_grid = np.linspace(
        plot_low, plot_high, int(sparse_cfg.get("plot_grid_points", 500))
    )
    n_grid_all, alpha_grid_all = [], []
    for row in results:
        n_grid, alpha_grid = material_curves(
            plot_grid,
            row["n0"],
            row["n_slope_per_THz"],
            row["alpha0_cm"],
            row["beta_cm_per_THz2"],
            row["center_THz"],
        )
        n_grid_all.append(n_grid)
        alpha_grid_all.append(alpha_grid)
    n_grid_all = np.asarray(n_grid_all)
    alpha_grid_all = np.asarray(alpha_grid_all)

    points_path = output_dir / f"{name}_sparse_points_{timestamp}.tsv"
    with points_path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            [
                "position",
                "frequency_THz",
                "H_real",
                "H_imag",
                "H_abs",
                "H_phase_rad",
                "n_approx",
                "alpha_point_cm-1",
                "alpha_curve_cm-1",
            ]
        )
        for row in results:
            for k, frequency in enumerate(frequencies_THz):
                writer.writerow(
                    [
                        row["position"],
                        frequency,
                        row["H_meas"][k].real,
                        row["H_meas"][k].imag,
                        abs(row["H_meas"][k]),
                        np.angle(row["H_meas"][k]),
                        row["n_points"][k],
                        row["alpha_points_cm"][k],
                        row["alpha_fit_points_cm"][k],
                    ]
                )

    pairs_path = output_dir / f"{name}_phase_pairs_{timestamp}.tsv"
    with pairs_path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            [
                "position",
                "f1_THz",
                "f2_THz",
                "center_THz",
                "delta_f_GHz",
                "delta_phi_wrapped_rad",
                "selected_cycle",
                "delta_phi_selected_rad",
                "tau_ps",
                "n_group",
            ]
        )
        for row in results:
            for pair in row["phase_pairs"]:
                writer.writerow(
                    [
                        row["position"],
                        pair["f1_THz"],
                        pair["f2_THz"],
                        pair["center_THz"],
                        pair["delta_f_GHz"],
                        pair["delta_phi_wrapped_rad"],
                        pair["cycle"],
                        pair["delta_phi_selected_rad"],
                        pair["tau_ps"],
                        pair["n_group"],
                    ]
                )

    tqdm.write(f"  Saved: {points_path}")
    tqdm.write(f"  Saved: {pairs_path}")

    return {
        "name": name,
        "thickness_mm": thickness_m * 1e3,
        "n_positions": len(results),
        "frequencies_THz": frequencies_THz,
        "pair_centers_THz": pair_centers,
        "plot_grid_THz": plot_grid,
        "avg_H": H_all.mean(axis=0),
        "avg_n_pair": n_pair_all.mean(axis=0),
        "std_n_pair": n_pair_all.std(axis=0),
        "avg_tau_ps": tau_all.mean(axis=0),
        "std_tau_ps": tau_all.std(axis=0),
        "avg_alpha_points_cm": alpha_all.mean(axis=0),
        "std_alpha_points_cm": alpha_all.std(axis=0),
        "avg_n_grid": n_grid_all.mean(axis=0),
        "avg_alpha_grid_cm": alpha_grid_all.mean(axis=0),
        "n0_mean": float(np.mean([row["n0"] for row in results])),
        "n_slope_mean": float(
            np.mean([row["n_slope_per_THz"] for row in results])
        ),
        "alpha0_mean_cm": float(
            np.mean([row["alpha0_cm"] for row in results])
        ),
        "beta_mean_cm_per_THz2": float(
            np.mean([row["beta_cm_per_THz2"] for row in results])
        ),
        "relative_complex_rmse_mean": float(
            np.mean([row["relative_complex_rmse"] for row in results])
        ),
    }


def _finish_axis(ax, xlabel, ylabel, title, ylim, plot_cfg, legend_loc):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if plot_cfg.get("font_title", 10):
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    if plot_cfg.get("use_legend", True):
        ax.legend(
            loc=legend_loc,
            ncol=int(plot_cfg.get("legend_ncol", 1)),
            fontsize=float(plot_cfg.get("font_legend", 9)),
        )


def create_sparse_plots(samples: list[dict], cfg: dict, output_dir: Path) -> None:
    """Plot sparse points and fitted approximation lines only."""

    plot_cfg = cfg["plot"]
    ylim = cfg.get("ylim", {}) or {}
    figure_size = tuple(plot_cfg.get("figure_size", [5.0, 3.3]))
    line_width = float(plot_cfg.get("lw_curve", 1.2))
    marker_size = float(plot_cfg.get("marker_size", 4.5))
    capsize = float(plot_cfg.get("capsize", 2.0))

    fig_n, ax_n = plt.subplots(figsize=figure_size)
    fig_alpha, ax_alpha = plt.subplots(figsize=figure_size)
    fig_tau, ax_tau = plt.subplots(figsize=figure_size)
    colors = cm.tab10(np.linspace(0, 1, max(len(samples), 1)))

    for sample, color in zip(samples, colors):
        label = _label(sample["name"])

        # These continuous lines are equations fitted from sparse points.
        ax_n.plot(
            sample["plot_grid_THz"],
            sample["avg_n_grid"],
            color=color,
            lw=line_width,
            label=f"{label} approximation",
        )
        ax_n.errorbar(
            sample["pair_centers_THz"],
            sample["avg_n_pair"],
            yerr=sample["std_n_pair"],
            fmt="o",
            color=color,
            ms=marker_size,
            capsize=capsize,
            label=f"{label} sparse points",
        )

        ax_alpha.plot(
            sample["plot_grid_THz"],
            sample["avg_alpha_grid_cm"],
            color=color,
            lw=line_width,
            label=f"{label} approximation",
        )
        ax_alpha.errorbar(
            sample["frequencies_THz"],
            sample["avg_alpha_points_cm"],
            yerr=sample["std_alpha_points_cm"],
            fmt="o",
            color=color,
            ms=marker_size,
            capsize=capsize,
            label=f"{label} sparse points",
        )

        ax_tau.errorbar(
            sample["pair_centers_THz"],
            sample["avg_tau_ps"],
            yerr=sample["std_tau_ps"],
            fmt="o-",
            color=color,
            lw=line_width,
            ms=marker_size,
            capsize=capsize,
            label=label,
        )

    _finish_axis(
        ax_n,
        "Frequency (THz)",
        "Refractive index $n$",
        "Sparse-frequency refractive-index approximation",
        ylim.get("n"),
        plot_cfg,
        plot_cfg.get("legend_loc", "best"),
    )
    _finish_axis(
        ax_alpha,
        "Frequency (THz)",
        r"$\alpha$ (cm$^{-1}$)",
        "Sparse-frequency absorption approximation",
        ylim.get("alpha"),
        plot_cfg,
        plot_cfg.get("legend_loc_alpha", "best"),
    )
    _finish_axis(
        ax_tau,
        "Phase-pair centre frequency (THz)",
        r"Delay $\tau$ (ps)",
        "Delay from sparse phase pairs",
        ylim.get("tau_ps"),
        plot_cfg,
        plot_cfg.get("legend_loc", "best"),
    )

    figures = [
        (fig_n, "batch_sparse_n_approximation"),
        (fig_alpha, "batch_sparse_alpha_approximation"),
        (fig_tau, "batch_sparse_tau"),
    ]
    formats = plot_cfg.get("formats", ["svg", "pdf", "png"])
    dpi = int(plot_cfg.get("dpi", 300))
    for figure, stem in figures:
        figure.tight_layout(pad=0.3)
        for file_format in formats:
            path = output_dir / f"{stem}.{file_format}"
            bbox = None if file_format in ("svg", "pdf") else "tight"
            figure.savefig(path, dpi=dpi, bbox_inches=bbox)
        print(f"  Saved: {output_dir / f'{stem}.{formats[0]}'}")

    if plot_cfg.get("show", True):
        plt.show()
    else:
        for figure, _ in figures:
            plt.close(figure)


def main(config_path: str | Path = CONFIG_DEFAULT) -> None:
    cfg = load_yaml(config_path)
    sparse_cfg = cfg.get("selected_frequencies", {}) or {}
    if not sparse_cfg.get("enabled", False):
        raise ValueError("selected_frequencies.enabled must be true")

    frequencies_THz = np.asarray(sparse_cfg["frequencies_THz"], dtype=float)
    plot_cfg = cfg["plot"]
    plt.rcParams.update(
        {
            "svg.fonttype": "path",
            "axes.labelsize": plot_cfg.get("font_axis_label", 10),
            "axes.titlesize": plot_cfg.get("font_title", 10),
            "xtick.labelsize": plot_cfg.get("font_tick", 10),
            "ytick.labelsize": plot_cfg.get("font_tick", 10),
            "legend.fontsize": plot_cfg.get("font_legend", 9),
            "axes.linewidth": plot_cfg.get("lw_spine", 0.8),
            "grid.linewidth": plot_cfg.get("lw_grid", 0.4),
        }
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg["output"]) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config       : {config_path}")
    print(f"Samples      : {len(cfg['samples'])}")
    print(f"Output       : {output_dir}")
    print(f"Sparse THz   : {', '.join(f'{f:.3f}' for f in frequencies_THz)}")
    print("Broadband    : disabled (no broadband H, n, or alpha)\n")

    samples: list[dict] = []
    for entry in tqdm(cfg["samples"], desc="Samples", unit="sample"):
        result = run_sample(entry, cfg, output_dir, timestamp)
        if result is not None:
            samples.append(result)

    if not samples:
        print("No samples processed")
        return

    print(f"\n{'=' * 100}")
    print("SPARSE-ONLY CROSS-SAMPLE SUMMARY")
    print(f"{'=' * 100}")
    print(
        f"{'Sample':<18} {'mm':>6} {'pos':>4} {'n0':>9} {'dn/df':>10} "
        f"{'alpha0':>10} {'beta':>10} {'H RMSE':>10}"
    )
    print("-" * 100)
    for sample in samples:
        print(
            f"{sample['name']:<18} {sample['thickness_mm']:>6.2f} "
            f"{sample['n_positions']:>4} {sample['n0_mean']:>9.4f} "
            f"{sample['n_slope_mean']:>10.4f} "
            f"{sample['alpha0_mean_cm']:>10.4f} "
            f"{sample['beta_mean_cm_per_THz2']:>10.4f} "
            f"{sample['relative_complex_rmse_mean']:>10.4g}"
        )

    summary_path = output_dir / f"sparse_cross_sample_{timestamp}.tsv"
    with summary_path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            [
                "sample",
                "thickness_mm",
                "n_positions",
                "n0_mean",
                "n_slope_mean_per_THz",
                "alpha0_mean_cm-1",
                "beta_mean_cm-1_THz-2",
                "relative_complex_rmse_mean",
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    sample["name"],
                    sample["thickness_mm"],
                    sample["n_positions"],
                    sample["n0_mean"],
                    sample["n_slope_mean"],
                    sample["alpha0_mean_cm"],
                    sample["beta_mean_cm_per_THz2"],
                    sample["relative_complex_rmse_mean"],
                ]
            )
    print(f"  Saved: {summary_path}")

    create_sparse_plots(samples, cfg, output_dir)
    print(f"\nAll sparse-only results saved in: {output_dir}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else CONFIG_DEFAULT
    main(path)
