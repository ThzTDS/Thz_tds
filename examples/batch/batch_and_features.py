"""
batch_and_features.py
=====================
Combined pipeline: runs fp_all_samples_batch + material_features for every
sample in the YAML config in a single pass.

Pipeline
--------
1. FP batch  — per sample: compute n(f) and α(f) (with / without FP
   correction), save TSVs and cross-sample overlay figures.
2. Features  — per individual measurement: extract n_mean and β (absorption
   slope), save per-measurement spectra, feature tables, and 2D material
   classification maps.

Both pipelines write into the same timestamped output directory:

    output_root/
      {timestamp}/
        batch/       ← per-sample TSVs + cross-sample overlay figures
        features/    ← per_measurement_features.tsv, sample_summary.tsv
        spectra/     ← per-measurement n / alpha / fit TSVs
        plots/       ← per-sample curves + material map figures

Usage
-----
    python examples/batch_and_features.py
    python examples/batch_and_features.py config/all_samples_fp_batch.yaml
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from tqdm import tqdm

import thz_tds
from thz_tds.refractive_fp import compute_refractive_index_fp
from thz_tds.refractive import compute_refractive_index, compute_refractive_index_arrays
from thz_tds.spectral import ensure_common_time_axis
from thz_tds import viz


CONFIG_DEFAULT = Path(__file__).parent.parent.parent / "config" / "all_samples_fp_batch.yaml"
METHOD = "no_fp"


# ── shared helpers ────────────────────────────────────────────────────────────

def load_yaml(path: str | Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def discover_pairs(
    sample_dir: Path,
    air_prefix: str,
    point_prefix: str,
    n_points: Optional[int],
) -> list[tuple[int, Path, Path]]:
    air_re = re.compile(rf"^{re.escape(air_prefix)}(\d+)$", re.IGNORECASE)
    pairs: list[tuple[int, Path, Path]] = []
    for child in sorted(sample_dir.iterdir()):
        if not child.is_dir():
            continue
        m = air_re.match(child.name)
        if not m:
            continue
        idx = int(m.group(1))
        point_path = sample_dir / f"{point_prefix}{idx}"
        if point_path.is_dir():
            pairs.append((idx, child, point_path))
    pairs.sort(key=lambda t: t[0])
    if n_points is not None:
        pairs = pairs[:n_points]
    return pairs


def _label(name: str) -> str:
    """'PA6_B_3' → 'PA6_B'."""
    parts = name.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else parts[0]


_VARIANTS = {"B": "Black", "N": "Natural", "W": "White"}

def _material_from_name(name: str) -> str:
    parts = name.split("_")
    base  = parts[0]
    if len(parts) >= 2 and parts[1] in _VARIANTS:
        return f"{base}_{parts[1]}"
    return base


def _build_ds_kwargs(load_cfg: dict, align_cfg: dict) -> dict:
    return dict(
        correction_factor       = load_cfg["correction_factor"],
        align                   = align_cfg["enabled"],
        align_method            = align_cfg["method"],
        corr_window_ps          = align_cfg["corr_window_ps"],
        adaptive_fit_window     = align_cfg.get("adaptive_fit_window",     True),
        fit_window_sigma_factor = align_cfg.get("fit_window_sigma_factor", 5.0),
        fallback_fit_window_ps  = align_cfg.get("fallback_fit_window_ps",  15.0),
        min_sigma_ps            = align_cfg.get("min_sigma_ps",            0.05),
        max_sigma_ps            = align_cfg.get("max_sigma_ps",            10.0),
        min_fit_points          = align_cfg.get("min_fit_points",          10),
        max_center_shift_ps     = align_cfg.get("max_center_shift_ps",     5.0),
        max_traces              = load_cfg.get("max_traces"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline 1 — FP batch
# ══════════════════════════════════════════════════════════════════════════════

def run_sample_fp(
    sample_entry: dict,
    shared: dict,
    output_root: Path,
    timestamp: str = "",
) -> dict | None:
    """
    Load all positions for one sample, compute n(f)/α(f) with/without FP
    correction, print summary table, save TSV.
    Returns band statistics for cross-sample plotting.
    """
    name        = sample_entry["name"]
    sample_dir  = Path(sample_entry["dir"])
    thickness   = float(sample_entry["thickness_m"])
    load_cfg    = shared["loading"]
    align_cfg   = shared["alignment"]
    ri_cfg      = shared["refractive_index"]

    air_prefix   = align_cfg.get("air_prefix",   "air")
    point_prefix = align_cfg.get("point_prefix", "point")
    n_points     = align_cfg.get("n_points",     None)
    f_low        = ri_cfg["f_low_THz"]
    f_high       = ri_cfg["f_high_THz"]
    skip_fp      = ri_cfg.get("skip_fp", False)

    output_root.mkdir(parents=True, exist_ok=True)
    pairs = discover_pairs(sample_dir, air_prefix, point_prefix, n_points)
    if not pairs:
        tqdm.write(f"  [{name}] No matched pairs found — skipping.")
        return None

    ds_kw = _build_ds_kwargs(load_cfg, align_cfg)
    traces_str = str(ds_kw["max_traces"]) if ds_kw["max_traces"] else "all"
    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"  {name}  |  {len(pairs)} positions  |  {thickness*1e3:.3f} mm  |  {traces_str} traces/folder")
    tqdm.write(f"{'='*60}")

    pos_colors = cm.tab10(np.linspace(0, 1, len(pairs)))
    positions  = []

    for (idx, air_path, point_path), pos_col in zip(pairs, pos_colors):
        pos_name = f"{point_prefix}{idx}"
        ref = thz_tds.THZDataset(path=air_path,   **ds_kw)
        sam = thz_tds.THZDataset(path=point_path, **ds_kw)

        if skip_fp:
            f_arr, n_arr, _, _, a_arr = compute_refractive_index(
                ref_ds=ref, sam_ds=sam,
                thickness_m=thickness,
                f_low_THz=f_low, f_high_THz=f_high,
            )
            n_init, n_fp         = n_arr, n_arr
            alpha_init, alpha_fp = a_arr, a_arr
        else:
            fp           = compute_refractive_index_fp(
                ref_ds=ref, sam_ds=sam,
                thickness_m=thickness,
                f_low_THz=f_low, f_high_THz=f_high,
            )
            f_arr                = fp.f_THz
            n_init, n_fp         = fp.n_init, fp.n_fp
            alpha_init, alpha_fp = fp.alpha_init_cm, fp.alpha_cm_fp

        band = (f_arr >= f_low) & (f_arr <= f_high)
        if skip_fp:
            tqdm.write(f"  {pos_name}: n={n_init[band].mean():.4f}")
        else:
            tqdm.write(
                f"  {pos_name}: n={n_init[band].mean():.4f}  "
                f"n_fp={n_fp[band].mean():.4f}"
            )
        positions.append({
            "name":       pos_name,
            "color":      pos_col,
            "f_THz":      f_arr,
            "n_fp":       n_fp,      "n_init":     n_init,
            "alpha_fp":   alpha_fp,  "alpha_init": alpha_init,
            "band":       band,
        })

    # band statistics
    f_band         = positions[0]["f_THz"][positions[0]["band"]]
    n_fp_all       = np.array([p["n_fp"][p["band"]]       for p in positions])
    n_init_all     = np.array([p["n_init"][p["band"]]     for p in positions])
    alpha_fp_all   = np.array([p["alpha_fp"][p["band"]]   for p in positions])
    alpha_init_all = np.array([p["alpha_init"][p["band"]] for p in positions])

    avg_n_init = n_init_all.mean(axis=0);    std_n_init = n_init_all.std(axis=0)
    avg_n_fp   = n_fp_all.mean(axis=0);      std_n_fp   = n_fp_all.std(axis=0)
    avg_a_init = alpha_init_all.mean(axis=0); std_a_init = alpha_init_all.std(axis=0)
    avg_a_fp   = alpha_fp_all.mean(axis=0);   std_a_fp   = alpha_fp_all.std(axis=0)

    per_n_init = n_init_all.mean(axis=1)
    per_n_fp   = n_fp_all.mean(axis=1)

    if skip_fp:
        tqdm.write(f"\n  {'Pos':<10}  {'n':>8}")
        tqdm.write(f"  {'-'*22}")
        for i, p in enumerate(positions):
            tqdm.write(f"  {p['name']:<10}  {per_n_init[i]:>8.4f}")
        total_mae_n = total_mae_a = 0.0
    else:
        per_mae_n = np.abs(n_fp_all - n_init_all).mean(axis=1)
        per_mae_a = np.abs(alpha_fp_all - alpha_init_all).mean(axis=1)
        tqdm.write(f"\n  {'Pos':<10}  {'n':>8}  {'n_fp':>10}  {'Δn':>8}  {'|Δn|(f)':>9}  {'|Δα|(f)':>10}")
        tqdm.write(f"  {'-'*64}")
        for i, p in enumerate(positions):
            dn = per_n_fp[i] - per_n_init[i]
            tqdm.write(
                f"  {p['name']:<10}  {per_n_init[i]:>8.4f}  {per_n_fp[i]:>10.4f}"
                f"  {dn:>+8.4f}  {per_mae_n[i]:>9.5f}  {per_mae_a[i]:>10.4f}"
            )
        total_mae_n = np.abs(n_fp_all - n_init_all).mean()
        total_mae_a = np.abs(alpha_fp_all - alpha_init_all).mean()
        tqdm.write(f"  {'TOTAL MAE':<10}  {'':>8}  {'':>10}  {'':>8}  {total_mae_n:>9.5f}  {total_mae_a:>10.4f}")

    # save TSV
    header = ["frequency_THz"]
    cols   = [f_band]
    for p in positions:
        pn = p["name"]
        if skip_fp:
            header += [f"{pn}_n", f"{pn}_alpha"]
            cols   += [p["n_init"][p["band"]], p["alpha_init"][p["band"]]]
        else:
            header += [f"{pn}_n", f"{pn}_n_withfp", f"{pn}_alpha", f"{pn}_alpha_withfp"]
            cols   += [p["n_init"][p["band"]], p["n_fp"][p["band"]],
                       p["alpha_init"][p["band"]], p["alpha_fp"][p["band"]]]
    if skip_fp:
        header += ["avg_n", "avg_alpha"]
        cols   += [avg_n_init, avg_a_init]
    else:
        header += ["avg_n", "avg_n_withfp", "avg_alpha", "avg_alpha_withfp"]
        cols   += [avg_n_init, avg_n_fp, avg_a_init, avg_a_fp]

    tsv_tag = "results" if skip_fp else "fp_results"
    tsv = output_root / f"{name}_{tsv_tag}_{timestamp}.tsv"
    np.savetxt(tsv, np.column_stack(cols),
               delimiter="\t", header="\t".join(header), comments="")
    tqdm.write(f"  Saved: {tsv}")

    return {
        "name":         name,
        "thickness_mm": thickness * 1e3,
        "n_positions":  len(pairs),
        "positions":    positions,
        "f_band":       f_band,
        "avg_n_init":   avg_n_init,  "std_n_init": std_n_init,
        "avg_n_fp":     avg_n_fp,    "std_n_fp":   std_n_fp,
        "avg_a_init":   avg_a_init,  "std_a_init": std_a_init,
        "avg_a_fp":     avg_a_fp,    "std_a_fp":   std_a_fp,
        "n_nofp_mean":  per_n_init.mean(), "n_nofp_std": per_n_init.std(),
        "n_fp_mean":    per_n_fp.mean(),   "n_fp_std":   per_n_fp.std(),
        "total_mae_n":  total_mae_n,       "total_mae_a": total_mae_a,
    }


def _run_fp_batch(cfg: dict, batch_dir: Path, timestamp: str) -> None:
    """Run the FP batch pipeline and save cross-sample overlay figures."""
    ri_cfg   = cfg["refractive_index"]
    skip_fp  = ri_cfg.get("skip_fp", False)
    plot_cfg = cfg["plot"]
    shared   = {k: cfg[k] for k in ("loading", "alignment", "refractive_index", "plot")}
    shared["ylim"] = cfg.get("ylim", {})

    fig_size       = tuple(plot_cfg["figure_size"])
    dpi            = plot_cfg["dpi"]
    formats        = plot_cfg["formats"]
    errorbar_step  = plot_cfg.get("errorbar_step",     15)
    line_width     = plot_cfg.get("lw_curve",          1.8)
    trace_width    = plot_cfg.get("lw_trace",          0.8)
    errorbar_width = plot_cfg.get("lw_errorbar",       1.0)
    label_fontsize = plot_cfg.get("font_line_label",    8)
    font_title     = plot_cfg.get("font_title",        11)
    font_legend    = plot_cfg.get("font_legend",        9)
    capsize        = plot_cfg.get("capsize",            3)
    xlim           = plot_cfg.get("xlim",               None)
    xticks         = plot_cfg.get("xticks",             None)
    use_inline_labels    = plot_cfg.get("use_inline_labels",    True)
    label_x              = plot_cfg.get("label_x",              None)
    use_legend           = plot_cfg.get("use_legend",           False)
    legend_loc           = plot_cfg.get("legend_loc",           "best")
    legend_loc_alpha     = plot_cfg.get("legend_loc_alpha",     legend_loc)
    legend_ncol          = plot_cfg.get("legend_ncol",          1)
    legend_handlelength  = plot_cfg.get("legend_handlelength",  1.5)
    legend_columnspacing = plot_cfg.get("legend_columnspacing", 0.8)
    legend_labelspacing  = plot_cfg.get("legend_labelspacing",  0.3)

    _ylim_cfg = shared.get("ylim", {}) or {}
    ylim_n    = _ylim_cfg.get("n")

    n_samples     = len(cfg["samples"])
    sample_colors = cm.tab10(np.linspace(0, 1, max(n_samples, 1)))
    cross_sample  = []

    for entry, s_col in zip(
        tqdm(cfg["samples"], desc="[1/2] FP batch", unit="sample"),
        sample_colors,
    ):
        summary = run_sample_fp(entry, shared, batch_dir, timestamp=timestamp)
        if summary:
            summary["color"] = s_col
            cross_sample.append(summary)

    if not cross_sample:
        print("  No samples processed in FP batch.")
        return

    # cross-sample summary table
    if skip_fp:
        W = 52
        print(f"\n{'=' * W}\nFP BATCH CROSS-SAMPLE SUMMARY\n{'=' * W}")
        print(f"{'Sample':<18}  {'mm':>5}  {'pts':>4}  {'n':>8}  {'±':>6}")
        print("-" * W)
        for s in cross_sample:
            print(f"{s['name']:<18}  {s['thickness_mm']:>5.2f}  {s['n_positions']:>4}  "
                  f"{s['n_nofp_mean']:>8.4f}  {s['n_nofp_std']:>6.4f}")
    else:
        W = 94
        print(f"\n{'=' * W}\nFP BATCH CROSS-SAMPLE SUMMARY\n{'=' * W}")
        print(f"{'Sample':<18}  {'mm':>5}  {'pts':>4}  "
              f"{'n':>8}  {'±':>6}  {'n_fp':>10}  {'±':>6}  "
              f"{'|Δn| MAE':>10}  {'|Δα| MAE':>10}")
        print("-" * W)
        for s in cross_sample:
            print(f"{s['name']:<18}  {s['thickness_mm']:>5.2f}  {s['n_positions']:>4}  "
                  f"{s['n_nofp_mean']:>8.4f}  {s['n_nofp_std']:>6.4f}  "
                  f"{s['n_fp_mean']:>10.4f}  {s['n_fp_std']:>6.4f}  "
                  f"{s['total_mae_n']:>10.5f}  {s['total_mae_a']:>10.4f}")
        all_mae_n = np.mean([s["total_mae_n"] for s in cross_sample])
        all_mae_a = np.mean([s["total_mae_a"] for s in cross_sample])
        print("-" * W)
        print(f"{'GRAND MEAN':<18}  {'':>5}  {'':>4}  "
              f"{'':>8}  {'':>6}  {'':>10}  {'':>6}  "
              f"{all_mae_n:>10.5f}  {all_mae_a:>10.4f}")

    # overlay figures
    single = len(cross_sample) == 1

    def _overlay(ax, qty_avg, qty_std, qty_pos_key, ylabel, title,
                 errorbars=True, ylim=None, loc=None):
        for s in cross_sample:
            f   = s["f_band"]
            avg = s[qty_avg]
            std = s[qty_std]
            col = s["color"]
            lbl = _label(s["name"])
            if single:
                for p in s["positions"]:
                    ax.plot(
                        p["f_THz"][p["band"]], p[qty_pos_key][p["band"]],
                        color=col, alpha=0.3, lw=trace_width,
                        label="_nolegend_",
                    )
            ax.plot(f, avg, color=col, lw=line_width, label=lbl)
            if errorbars:
                ax.errorbar(
                    f[::errorbar_step], avg[::errorbar_step], yerr=std[::errorbar_step],
                    fmt="none", ecolor=col,
                    elinewidth=errorbar_width, capsize=capsize, capthick=errorbar_width,
                    alpha=0.9, label="_nolegend_",
                )
            if use_inline_labels:
                x_lbl = label_x if label_x is not None else f[-1]
                ax.text(x_lbl, avg[-1], lbl,
                        color=col, fontsize=label_fontsize,
                        va="center", ha="left", clip_on=True)
        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel(ylabel)
        if font_title:
            ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)
        if xticks is not None:
            ax.set_xticks(xticks)
        ax.grid(True, alpha=0.3)
        if use_legend:
            ax.legend(
                loc=loc if loc is not None else legend_loc,
                ncol=legend_ncol, fontsize=font_legend,
                handlelength=legend_handlelength,
                columnspacing=legend_columnspacing,
                labelspacing=legend_labelspacing,
                frameon=True, borderpad=0.4, handletextpad=0.4,
            )

    fig_n,      ax_n      = plt.subplots(figsize=fig_size)
    fig_a,      ax_a      = plt.subplots(figsize=fig_size)
    fig_n_mean, ax_n_mean = plt.subplots(figsize=fig_size)
    fig_a_mean, ax_a_mean = plt.subplots(figsize=fig_size)

    _overlay(ax_n,      "avg_n_init", "std_n_init", "n_init",
             "Refractive index $n$",  "Refractive index",
             ylim=ylim_n)
    _overlay(ax_a,      "avg_a_init", "std_a_init", "alpha_init",
             r"Absorption coefficient, $\alpha$ [cm$^{-1}$]", "Absorption",
             loc=legend_loc_alpha)
    _overlay(ax_n_mean, "avg_n_init", "std_n_init", "n_init",
             "Refractive index $n$",  "Mean refractive index",
             errorbars=False, ylim=ylim_n)
    _overlay(ax_a_mean, "avg_a_init", "std_a_init", "alpha_init",
             r"Absorption coefficient, $\alpha$ [cm$^{-1}$]", "Mean absorption",
             errorbars=False, loc=legend_loc_alpha)

    figures_to_save = [
        (fig_n,      "n"),
        (fig_a,      "alpha"),
        (fig_n_mean, "n_mean"),
        (fig_a_mean, "alpha_mean"),
    ]

    if not skip_fp:
        fig_n_fp,      ax_n_fp      = plt.subplots(figsize=fig_size)
        fig_a_fp,      ax_a_fp      = plt.subplots(figsize=fig_size)
        fig_n_fp_mean, ax_n_fp_mean = plt.subplots(figsize=fig_size)
        fig_a_fp_mean, ax_a_fp_mean = plt.subplots(figsize=fig_size)

        _overlay(ax_n_fp,      "avg_n_fp", "std_n_fp", "n_fp",
                 "Refractive index $n$",  "Refractive index (FP corrected)",
                 ylim=ylim_n)
        _overlay(ax_a_fp,      "avg_a_fp", "std_a_fp", "alpha_fp",
                 r"Absorption coefficient, $\alpha$ [cm$^{-1}$]", "Absorption (FP corrected)",
                 loc=legend_loc_alpha)
        _overlay(ax_n_fp_mean, "avg_n_fp", "std_n_fp", "n_fp",
                 "Refractive index $n$",  "Mean refractive index (FP corrected)",
                 errorbars=False, ylim=ylim_n)
        _overlay(ax_a_fp_mean, "avg_a_fp", "std_a_fp", "alpha_fp",
                 r"Absorption coefficient, $\alpha$ [cm$^{-1}$]", "Mean absorption (FP corrected)",
                 errorbars=False, loc=legend_loc_alpha)

        figures_to_save += [
            (fig_n_fp,      "n_withfp"),
            (fig_a_fp,      "alpha_withfp"),
            (fig_n_fp_mean, "n_mean_withfp"),
            (fig_a_fp_mean, "alpha_mean_withfp"),
        ]

    for fig_obj, _ in figures_to_save:
        fig_obj.tight_layout(pad=0.3)

    for fig_obj, tag in figures_to_save:
        stem = f"batch_{tag}"
        for fmt in formats:
            path = batch_dir / f"{stem}.{fmt}"
            bbox = None if fmt in ("svg", "pdf") else "tight"
            fig_obj.savefig(path, dpi=dpi, bbox_inches=bbox)
        print(f"  Saved: {batch_dir / stem}.{formats[0]}")
        plt.close(fig_obj)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline 2 — material features
# ══════════════════════════════════════════════════════════════════════════════

def _alpha_model(f_THz: np.ndarray, beta: float, alpha0: float) -> np.ndarray:
    return beta * f_THz ** 2 + alpha0


def fit_absorption_beta(
    f_THz: np.ndarray,
    alpha_cm1: np.ndarray,
    f_low: float = 0.4,
    f_high: float = 1.0,
) -> dict:
    band   = (f_THz >= f_low) & (f_THz <= f_high)
    finite = np.isfinite(alpha_cm1)
    mask   = band & finite
    n_pts  = int(mask.sum())
    alpha_fit = np.full_like(f_THz, np.nan)
    residual  = np.full_like(f_THz, np.nan)

    if n_pts < 4:
        return dict(beta=np.nan, alpha0=np.nan,
                    alpha_fit=alpha_fit, residual=residual,
                    rmse=np.nan, n_points_used=n_pts,
                    success=False, error=f"only {n_pts} valid points (need ≥ 4)")
    try:
        popt, _ = curve_fit(_alpha_model, f_THz[mask], alpha_cm1[mask],
                            p0=[10.0, 1.0], maxfev=10_000)
        beta_v, alpha0_v = float(popt[0]), float(popt[1])
        alpha_fit[band]  = _alpha_model(f_THz[band], beta_v, alpha0_v)
        residual[mask]   = alpha_cm1[mask] - alpha_fit[mask]
        rmse             = float(np.sqrt(np.mean(residual[mask] ** 2)))
        return dict(beta=beta_v, alpha0=alpha0_v,
                    alpha_fit=alpha_fit, residual=residual,
                    rmse=rmse, n_points_used=n_pts,
                    success=True, error=None)
    except Exception as exc:
        return dict(beta=np.nan, alpha0=np.nan,
                    alpha_fit=alpha_fit, residual=residual,
                    rmse=np.nan, n_points_used=n_pts,
                    success=False, error=str(exc))


def extract_features(
    f_THz: np.ndarray,
    n: np.ndarray,
    alpha_cm1: np.ndarray,
    material: str,
    sample_id: str,
    measurement_id: str,
    f_low: float,
    f_high: float,
) -> tuple[dict, dict]:
    band   = (f_THz >= f_low) & (f_THz <= f_high)
    good   = band & np.isfinite(n) & np.isfinite(alpha_cm1)
    n_mean = float(np.mean(n[good])) if good.any() else np.nan
    fit    = fit_absorption_beta(f_THz, alpha_cm1, f_low, f_high)

    feature_row = dict(
        material=material, sample_id=sample_id,
        measurement_id=measurement_id, method=METHOD,
        n_mean=n_mean, beta=fit["beta"], alpha0=fit["alpha0"],
        fit_rmse=fit["rmse"], f_low_THz=f_low, f_high_THz=f_high,
        n_points_used=fit["n_points_used"],
    )
    spectrum = dict(
        f_THz=f_THz, n=n, alpha_cm1=alpha_cm1,
        alpha_fit=fit["alpha_fit"], residual=fit["residual"],
        band=band, fit_success=fit["success"], fit_error=fit["error"],
    )
    return feature_row, spectrum


def save_spectrum_tsv(spectra_dir: Path, sample_id: str,
                      measurement_id: str, spectrum: dict) -> None:
    out_dir = spectra_dir / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    f    = spectrum["f_THz"]
    band = spectrum["band"]
    used = (band & np.isfinite(spectrum["n"]) & np.isfinite(spectrum["alpha_cm1"])).astype(float)
    header = "frequency_THz\tn\talpha_cm1\talpha_fit_cm1\talpha_residual_cm1\tused_for_fit"
    np.savetxt(
        out_dir / f"{measurement_id}_n_alpha_fit.tsv",
        np.column_stack([f, spectrum["n"], spectrum["alpha_cm1"],
                         spectrum["alpha_fit"], spectrum["residual"], used]),
        delimiter="\t", header=header, comments="",
    )


def plot_sample_curves(
    plots_dir: Path, sample_id: str,
    records: list[tuple[dict, dict]],
    f_low: float, f_high: float,
    fig_size: tuple, dpi: int, formats: list[str],
) -> None:
    out_dir = plots_dir / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = cm.tab10(np.linspace(0, 1, max(len(records), 1)))

    fig_n,  ax_n  = plt.subplots(figsize=fig_size)
    fig_a,  ax_a  = plt.subplots(figsize=fig_size)
    fig_af, ax_af = plt.subplots(figsize=fig_size)

    for (row, sp), col in zip(records, colors):
        f    = sp["f_THz"]
        band = sp["band"]
        lbl  = row["measurement_id"]
        ax_n.plot( f[band], sp["n"][band],        color=col, lw=1.0, label=lbl)
        ax_a.plot( f[band], sp["alpha_cm1"][band], color=col, lw=1.0, label=lbl)
        ax_af.plot(f[band], sp["alpha_cm1"][band], color=col, lw=1.0, label=lbl)
        if sp["fit_success"]:
            ax_af.plot(f[band], sp["alpha_fit"][band], "--", color=col, lw=1.3,
                       label=f"{lbl} fit")

    for ax, ylabel, title, stem in [
        (ax_n,  "Refractive index $n$",  f"{sample_id} — n(f)",       "n_curves"),
        (ax_a,  r"Absorption coefficient, $\alpha$ [cm$^{-1}$]", f"{sample_id} — α(f)",       "alpha_curves"),
        (ax_af, r"Absorption coefficient, $\alpha$ [cm$^{-1}$]", f"{sample_id} — α(f) + fit", "alpha_fit_curves"),
    ]:
        ax.axvspan(f_low, f_high, alpha=0.07, color="grey")
        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    for fig_obj, stem in [(fig_n, "n_curves"), (fig_a, "alpha_curves"), (fig_af, "alpha_fit_curves")]:
        fig_obj.tight_layout()
        viz.save_figure(fig_obj, out_dir / stem, formats=formats, dpi=dpi)
        plt.close(fig_obj)


def _material_palette(materials: list[str]) -> dict[str, np.ndarray]:
    unique  = sorted(set(materials))
    palette = cm.tab20(np.linspace(0, 1, max(len(unique), 1)))
    return {m: palette[i] for i, m in enumerate(unique)}


def plot_material_map(
    plots_dir: Path, df: pd.DataFrame,
    fig_size: tuple, dpi: int, formats: list[str],
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    df_v = df.dropna(subset=["n_mean", "beta"])
    if df_v.empty:
        tqdm.write("  [material map] No valid (n_mean, beta) pairs — skipping.")
        return

    pal = _material_palette(df_v["material"].tolist())

    fig1, ax1 = plt.subplots(figsize=fig_size)
    for mat, grp in df_v.groupby("material"):
        ax1.scatter(grp["n_mean"], grp["beta"],
                    c=[pal[mat]], s=28, alpha=0.7, label=mat, zorder=3)
    ax1.set_xlabel("$n_{\\mathrm{mean}}$")
    ax1.set_ylabel(r"$\beta$")
    ax1.set_title("2D material map — individual measurements")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()

    summary = (
        df_v.groupby(["material", "sample_id"])[["n_mean", "beta"]]
        .agg(["mean", "std"]).reset_index()
    )
    summary.columns = ["material", "sample_id",
                        "n_mean_avg", "n_mean_std", "beta_avg", "beta_std"]
    summary["n_mean_std"] = summary["n_mean_std"].fillna(0)
    summary["beta_std"]   = summary["beta_std"].fillna(0)

    fig2, ax2 = plt.subplots(figsize=fig_size)
    for mat, grp in summary.groupby("material"):
        col = pal[mat]
        ax2.errorbar(grp["n_mean_avg"], grp["beta_avg"],
                     xerr=grp["n_mean_std"], yerr=grp["beta_std"],
                     fmt="o", color=col, ecolor=col,
                     capsize=4, capthick=1.2, elinewidth=1.0,
                     markersize=7, label=mat, zorder=3)
        for _, row in grp.iterrows():
            ax2.annotate(row["sample_id"],
                         (row["n_mean_avg"], row["beta_avg"]),
                         textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax2.set_xlabel("$n_{\\mathrm{mean}}$")
    ax2.set_ylabel(r"$\beta$")
    ax2.set_title("2D material map — sample mean ± std")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    viz.save_figure(fig1, plots_dir / "material_map_nmean_beta",  formats=formats, dpi=dpi)
    plt.close(fig1)
    viz.save_figure(fig2, plots_dir / "material_map_errorbars",   formats=formats, dpi=dpi)
    plt.close(fig2)


def process_position_features(
    ref_ds, sam_ds,
    thickness_m: float, material: str, sample_id: str, pos_name: str,
    f_low: float, f_high: float, skipped_log: list[str],
) -> Optional[tuple[dict, dict]]:
    try:
        t_com, y_ref_c, y_sam_c = ensure_common_time_axis(
            ref_ds.x_ps, ref_ds.y_avg, sam_ds.x_ps, sam_ds.y_avg,
        )
        f_THz, n_f, _, _, alpha_cm = compute_refractive_index_arrays(
            t_com, y_ref_c, y_sam_c, thickness_m, f_low, f_high,
        )
    except Exception as exc:
        msg = f"  SKIP {sample_id}/{pos_name}: {exc}"
        tqdm.write(msg); skipped_log.append(msg)
        return None

    if not np.any(np.isfinite(n_f)) or not np.any(np.isfinite(alpha_cm)):
        msg = f"  SKIP {sample_id}/{pos_name}: all-NaN result"
        tqdm.write(msg); skipped_log.append(msg)
        return None

    feat, spec = extract_features(f_THz, n_f, alpha_cm,
                                   material, sample_id, pos_name,
                                   f_low, f_high)
    if not spec["fit_success"]:
        msg = f"  WARN {sample_id}/{pos_name}: fit failed — {spec['fit_error']}"
        tqdm.write(msg); skipped_log.append(msg)

    return feat, spec


def process_sample_features(
    sample_entry: dict, shared: dict, out_dirs: dict,
    f_low: float, f_high: float,
    fig_size: tuple, dpi: int, formats: list[str],
    skipped_log: list[str],
) -> list[dict]:
    name        = sample_entry["name"]
    sample_dir  = Path(sample_entry["dir"])
    thickness_m = float(sample_entry["thickness_m"])
    material    = sample_entry.get("material", _material_from_name(name))
    load_cfg    = shared["loading"]
    align_cfg   = shared["alignment"]

    ds_kw = _build_ds_kwargs(load_cfg, align_cfg)
    pairs = discover_pairs(
        sample_dir,
        align_cfg.get("air_prefix",   "air"),
        align_cfg.get("point_prefix", "point"),
        align_cfg.get("n_points",     None),
    )
    if not pairs:
        tqdm.write(f"  [{name}] No matched pairs — skipping.")
        return []

    traces_str   = str(ds_kw["max_traces"]) if ds_kw["max_traces"] else "all"
    point_prefix = align_cfg.get("point_prefix", "point")
    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"  {name}  |  {len(pairs)} positions  |  {thickness_m*1e3:.3f} mm  |  {traces_str} traces/folder")
    tqdm.write(f"{'='*60}")

    all_records: list[tuple[dict, dict]] = []
    for idx, air_path, point_path in pairs:
        pos_name = f"{point_prefix}{idx}"
        ref_ds   = thz_tds.THZDataset(path=air_path,   **ds_kw)
        sam_ds   = thz_tds.THZDataset(path=point_path, **ds_kw)
        result   = process_position_features(
            ref_ds, sam_ds, thickness_m,
            material, name, pos_name,
            f_low, f_high, skipped_log,
        )
        if result is None:
            continue
        feat, spec = result
        save_spectrum_tsv(out_dirs["spectra"], name, feat["measurement_id"], spec)
        status = "" if spec["fit_success"] else "  [fit failed]"
        tqdm.write(f"  {feat['measurement_id']}: n_mean={feat['n_mean']:.4f}  "
                   f"beta={feat['beta']:.3f}  alpha0={feat['alpha0']:.3f}" + status)
        all_records.append((feat, spec))

    if all_records:
        plot_sample_curves(out_dirs["plots"], name, all_records,
                           f_low, f_high, fig_size, dpi, formats)

    return [feat for feat, _ in all_records]


def _run_features(cfg: dict, feat_dirs: dict) -> None:
    """Run the material-features pipeline."""
    band_cfg = cfg.get("features", cfg["refractive_index"])
    plot_cfg = cfg["plot"]
    shared   = {k: cfg[k] for k in ("loading", "alignment", "refractive_index", "plot")}

    f_low    = float(band_cfg.get("f_low_THz",  0.4))
    f_high   = float(band_cfg.get("f_high_THz", 1.0))
    fig_size = tuple(plot_cfg["figure_size"])
    dpi      = plot_cfg["dpi"]
    formats  = plot_cfg["formats"]

    skipped_log:  list[str]  = []
    all_features: list[dict] = []

    for entry in tqdm(cfg["samples"], desc="[2/2] Features", unit="sample"):
        feats = process_sample_features(
            entry, shared, feat_dirs,
            f_low, f_high, fig_size, dpi, formats,
            skipped_log,
        )
        all_features.extend(feats)

    if not all_features:
        print("  No features extracted.")
        return

    df = pd.DataFrame(all_features)

    feat_cols = ["material", "sample_id", "measurement_id", "method",
                 "n_mean", "beta", "alpha0", "fit_rmse",
                 "f_low_THz", "f_high_THz", "n_points_used"]
    df[feat_cols].to_csv(
        feat_dirs["features"] / "per_measurement_features.tsv",
        sep="\t", index=False, float_format="%.6f",
    )

    summary = (
        df.groupby(["material", "sample_id", "method"])
        .agg(n_mean_avg=("n_mean","mean"), n_mean_std=("n_mean","std"),
             beta_avg=("beta","mean"),     beta_std=("beta","std"),
             alpha0_avg=("alpha0","mean"), alpha0_std=("alpha0","std"),
             n_measurements=("n_mean","count"))
        .reset_index()
    )
    summary.to_csv(
        feat_dirs["features"] / "sample_summary_features.tsv",
        sep="\t", index=False, float_format="%.6f",
    )

    plot_material_map(feat_dirs["plots"], df, fig_size, dpi, formats)

    # printed summary
    W = 90
    print(f"\n{'=' * W}\nFEATURE SUMMARY (per sample)\n{'=' * W}")
    print(f"{'Sample':<18}  {'Material':<8}  {'#':>4}  "
          f"{'n_mean':>8}  {'±':>6}  {'beta':>8}  {'±':>7}  {'alpha0':>8}  {'±':>7}")
    print("-" * W)
    for _, row in summary.iterrows():
        print(f"{row['sample_id']:<18}  {row['material']:<8}  "
              f"{int(row['n_measurements']):>4}  "
              f"{row['n_mean_avg']:>8.4f}  {row.get('n_mean_std', 0.0):>6.4f}  "
              f"{row['beta_avg']:>8.3f}  {row.get('beta_std', 0.0):>7.3f}  "
              f"{row['alpha0_avg']:>8.3f}  {row.get('alpha0_std', 0.0):>7.3f}")

    if skipped_log:
        print(f"\n{'─' * W}\nSKIPPED / WARNINGS ({len(skipped_log)}):")
        for msg in skipped_log:
            print(msg)


# ══════════════════════════════════════════════════════════════════════════════
# Combined entry point
# ══════════════════════════════════════════════════════════════════════════════

def main(config_path: str | Path = CONFIG_DEFAULT) -> None:
    cfg = load_yaml(config_path)

    output_root = Path(cfg["output"])
    plot_cfg    = cfg["plot"]
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir     = output_root / timestamp

    # sub-directories
    batch_dir = run_dir / "batch"
    feat_dirs = {
        "features": run_dir / "features",
        "spectra":  run_dir / "spectra",
        "plots":    run_dir / "plots",
    }
    for d in [batch_dir] + list(feat_dirs.values()):
        d.mkdir(parents=True, exist_ok=True)

    # apply rcParams once for both pipelines
    plt.rcParams.update({
        "svg.fonttype":       "path",
        "axes.labelsize":     plot_cfg.get("font_axis_label", 10),
        "axes.titlesize":     plot_cfg.get("font_title",      11),
        "xtick.labelsize":    plot_cfg.get("font_tick",        9),
        "ytick.labelsize":    plot_cfg.get("font_tick",        9),
        "legend.fontsize":    plot_cfg.get("font_legend",      9),
        "axes.linewidth":     plot_cfg.get("lw_spine",        0.8),
        "grid.linewidth":     plot_cfg.get("lw_grid",         0.4),
        "xtick.major.size":   plot_cfg.get("tick_length",     3.0),
        "ytick.major.size":   plot_cfg.get("tick_length",     3.0),
        "xtick.major.width":  plot_cfg.get("tick_width",      0.8),
        "ytick.major.width":  plot_cfg.get("tick_width",      0.8),
    })

    band_cfg = cfg.get("features", cfg["refractive_index"])
    print(f"Config   : {config_path}")
    print(f"Samples  : {len(cfg['samples'])}")
    print(f"Band     : {band_cfg.get('f_low_THz', 0.4)}–{band_cfg.get('f_high_THz', 1.0)} THz")
    print(f"Output   : {run_dir}\n")

    _run_fp_batch(cfg, batch_dir, timestamp)
    _run_features(cfg, feat_dirs)

    print(f"\n{'=' * 60}")
    print(f"All results saved in: {run_dir}")
    print(f"  batch/               — TSVs + overlay figures")
    print(f"  features/            — per_measurement_features.tsv, sample_summary_features.tsv")
    print(f"  spectra/<sample>/    — per-measurement n / alpha / fit TSVs")
    print(f"  plots/<sample>/      — n_curves, alpha_curves, alpha_fit_curves")
    print(f"  plots/               — material_map_nmean_beta, material_map_errorbars")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else CONFIG_DEFAULT
    main(path)
