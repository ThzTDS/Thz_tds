"""
cielecki_material_features.py
==============================
Per-measurement feature extraction for 2D material classification maps.

For every individual THz trace in each sample folder this script:
  1. Computes n(f) and α(f) without FP correction.
  2. Extracts n_mean = mean n  in [f_low, f_high] THz.
  3. Fits α(f) = beta·f² + alpha0  in [f_low, f_high] THz.
  4. Saves per-measurement spectra (n, α, α_fit, residual) as TSV.
  5. Saves per-measurement and sample-summary feature tables.
  6. Generates n(f), α(f), α+fit plots per sample and 2D material maps.

Usage
-----
    python examples/cielecki_material_features.py
    python examples/cielecki_material_features.py config/all_samples_fp_batch.yaml
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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from tqdm import tqdm

import thz_tds
from thz_tds.refractive import compute_refractive_index_arrays
from thz_tds.spectral import ensure_common_time_axis
from thz_tds import viz


CONFIG_DEFAULT = Path(__file__).parent.parent / "config" / "all_samples_fp_batch.yaml"
METHOD = "no_fp"


# ── YAML / path helpers ──────────────────────────────────────────────────────

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


def _material_from_name(name: str) -> str:
    """Derive material label from sample name (part before first '_')."""
    return name.split("_")[0]


# ── absorption model and fitting ─────────────────────────────────────────────

def _alpha_model(f_THz: np.ndarray, beta: float, alpha0: float) -> np.ndarray:
    return beta * f_THz ** 2 + alpha0


def fit_absorption_beta(
    f_THz: np.ndarray,
    alpha_cm1: np.ndarray,
    f_low: float = 0.4,
    f_high: float = 1.0,
) -> dict:
    """
    Fit alpha(f) = beta * f^2 + alpha0 in [f_low, f_high] THz.
    No linear term.

    Returns dict with keys:
        beta, alpha0, alpha_fit, residual, rmse, n_points_used, success, error
    """
    band   = (f_THz >= f_low) & (f_THz <= f_high)
    finite = np.isfinite(alpha_cm1)
    mask   = band & finite

    n_pts      = int(mask.sum())
    alpha_fit  = np.full_like(f_THz, np.nan)
    residual   = np.full_like(f_THz, np.nan)

    if n_pts < 4:
        return dict(
            beta=np.nan, alpha0=np.nan,
            alpha_fit=alpha_fit, residual=residual,
            rmse=np.nan, n_points_used=n_pts,
            success=False, error=f"only {n_pts} valid band points (need ≥ 4)",
        )

    try:
        popt, _ = curve_fit(
            _alpha_model, f_THz[mask], alpha_cm1[mask],
            p0=[10.0, 1.0], maxfev=10_000,
        )
        beta_v, alpha0_v = float(popt[0]), float(popt[1])
        alpha_fit[band]  = _alpha_model(f_THz[band], beta_v, alpha0_v)
        residual[mask]   = alpha_cm1[mask] - alpha_fit[mask]
        rmse             = float(np.sqrt(np.mean(residual[mask] ** 2)))
        return dict(
            beta=beta_v, alpha0=alpha0_v,
            alpha_fit=alpha_fit, residual=residual,
            rmse=rmse, n_points_used=n_pts,
            success=True, error=None,
        )
    except Exception as exc:
        return dict(
            beta=np.nan, alpha0=np.nan,
            alpha_fit=alpha_fit, residual=residual,
            rmse=np.nan, n_points_used=n_pts,
            success=False, error=str(exc),
        )


# ── feature extraction ───────────────────────────────────────────────────────

def extract_cielecki_features(
    f_THz: np.ndarray,
    n: np.ndarray,
    alpha_cm1: np.ndarray,
    material: str,
    sample_id: str,
    measurement_id: str,
    method: str,
    f_low: float = 0.4,
    f_high: float = 1.0,
) -> tuple[dict, dict]:
    """
    Compute n_mean and absorption-fit parameters for one measurement.

    Returns
    -------
    feature_row : dict  — flat dict suitable as a DataFrame row.
    spectrum    : dict  — full spectral arrays for saving / plotting.
    """
    band   = (f_THz >= f_low) & (f_THz <= f_high)
    good   = band & np.isfinite(n) & np.isfinite(alpha_cm1)
    n_mean = float(np.mean(n[good])) if good.any() else np.nan

    fit = fit_absorption_beta(f_THz, alpha_cm1, f_low, f_high)

    feature_row = dict(
        material=material,
        sample_id=sample_id,
        measurement_id=measurement_id,
        method=method,
        n_mean=n_mean,
        beta=fit["beta"],
        alpha0=fit["alpha0"],
        fit_rmse=fit["rmse"],
        f_low_THz=f_low,
        f_high_THz=f_high,
        n_points_used=fit["n_points_used"],
    )

    spectrum = dict(
        f_THz=f_THz,
        n=n,
        alpha_cm1=alpha_cm1,
        alpha_fit=fit["alpha_fit"],
        residual=fit["residual"],
        band=band,
        fit_success=fit["success"],
        fit_error=fit["error"],
    )

    return feature_row, spectrum


# ── per-measurement spectrum TSV ─────────────────────────────────────────────

def save_measurement_spectrum_with_fit(
    spectra_dir: Path,
    sample_id: str,
    measurement_id: str,
    spectrum: dict,
) -> None:
    """
    Columns: frequency_THz  n  alpha_cm1  alpha_fit_cm1  alpha_residual_cm1  used_for_fit
    """
    out_dir = spectra_dir / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    f      = spectrum["f_THz"]
    band   = spectrum["band"]
    finite = np.isfinite(spectrum["n"]) & np.isfinite(spectrum["alpha_cm1"])
    used   = (band & finite).astype(float)

    header = (
        "frequency_THz\tn\talpha_cm1\t"
        "alpha_fit_cm1\talpha_residual_cm1\tused_for_fit"
    )
    data = np.column_stack([
        f,
        spectrum["n"],
        spectrum["alpha_cm1"],
        spectrum["alpha_fit"],
        spectrum["residual"],
        used,
    ])
    np.savetxt(
        out_dir / f"{measurement_id}_n_alpha_fit.tsv",
        data, delimiter="\t", header=header, comments="",
    )


# ── per-sample plots ─────────────────────────────────────────────────────────

def plot_sample_curves(
    plots_dir: Path,
    sample_id: str,
    records: list[tuple[dict, dict]],
    f_low: float,
    f_high: float,
    fig_size: tuple,
    dpi: int,
    formats: list[str],
) -> None:
    """
    Three figures saved to plots_dir / sample_id /:
      n_curves.png, alpha_curves.png, alpha_fit_curves.png
    """
    out_dir = plots_dir / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    n_meas = len(records)
    colors = cm.tab10(np.linspace(0, 1, max(n_meas, 1)))

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
        (ax_n,  "Refractive index $n$",   f"{sample_id} — n(f)",       "n_curves"),
        (ax_a,  r"$\alpha$ (cm$^{-1}$)",  f"{sample_id} — α(f)",       "alpha_curves"),
        (ax_af, r"$\alpha$ (cm$^{-1}$)",  f"{sample_id} — α(f) + fit", "alpha_fit_curves"),
    ]:
        ax.axvspan(f_low, f_high, alpha=0.07, color="grey")
        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    for fig_obj, stem in [
        (fig_n,  "n_curves"),
        (fig_a,  "alpha_curves"),
        (fig_af, "alpha_fit_curves"),
    ]:
        fig_obj.tight_layout()
        viz.save_figure(fig_obj, out_dir / stem, formats=formats, dpi=dpi)
        plt.close(fig_obj)


# ── 2D material map plots ────────────────────────────────────────────────────

def _material_palette(materials: list[str]) -> dict[str, np.ndarray]:
    unique  = sorted(set(materials))
    palette = cm.tab10(np.linspace(0, 1, max(len(unique), 1)))
    return {m: palette[i] for i, m in enumerate(unique)}


def plot_material_map(
    plots_dir: Path,
    df: pd.DataFrame,
    fig_size: tuple,
    dpi: int,
    formats: list[str],
) -> None:
    """
    material_map_nmean_beta.png  — scatter of every individual measurement.
    material_map_errorbars.png   — sample mean ± std with labels.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    df_v = df.dropna(subset=["n_mean", "beta"])
    if df_v.empty:
        tqdm.write("  [material map] No valid (n_mean, beta) pairs — skipping.")
        return

    pal = _material_palette(df_v["material"].tolist())

    # ── all measurements scatter ──────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=fig_size)
    for mat, grp in df_v.groupby("material"):
        ax1.scatter(grp["n_mean"], grp["beta"],
                    c=[pal[mat]], s=28, alpha=0.7, label=mat, zorder=3)
    ax1.set_xlabel("$n_{\\mathrm{mean}}$")
    ax1.set_ylabel(r"$\beta$ (cm$^{-1}$ THz$^{-2}$)")
    ax1.set_title("2D material map — individual measurements")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    viz.save_figure(fig1, plots_dir / "material_map_nmean_beta", formats=formats, dpi=dpi)
    plt.close(fig1)

    # ── sample mean ± std errorbars ───────────────────────────────────────
    summary = (
        df_v
        .groupby(["material", "sample_id"])[["n_mean", "beta"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["material", "sample_id",
                        "n_mean_avg", "n_mean_std",
                        "beta_avg",   "beta_std"]
    summary["n_mean_std"] = summary["n_mean_std"].fillna(0)
    summary["beta_std"]   = summary["beta_std"].fillna(0)

    fig2, ax2 = plt.subplots(figsize=fig_size)
    for mat, grp in summary.groupby("material"):
        col = pal[mat]
        ax2.errorbar(
            grp["n_mean_avg"], grp["beta_avg"],
            xerr=grp["n_mean_std"], yerr=grp["beta_std"],
            fmt="o", color=col, ecolor=col,
            capsize=4, capthick=1.2, elinewidth=1.0,
            markersize=7, label=mat, zorder=3,
        )
        for _, row in grp.iterrows():
            ax2.annotate(
                row["sample_id"],
                (row["n_mean_avg"], row["beta_avg"]),
                textcoords="offset points", xytext=(6, 4), fontsize=7,
            )
    ax2.set_xlabel("$n_{\\mathrm{mean}}$")
    ax2.set_ylabel(r"$\beta$ (cm$^{-1}$ THz$^{-2}$)")
    ax2.set_title("2D material map — sample mean ± std")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    viz.save_figure(fig2, plots_dir / "material_map_errorbars", formats=formats, dpi=dpi)
    plt.close(fig2)


# ── per-position individual trace loop ───────────────────────────────────────

def process_position(
    ref_ds,
    sam_ds,
    thickness_m: float,
    material: str,
    sample_id: str,
    pos_name: str,
    f_low: float,
    f_high: float,
    skipped_log: list[str],
) -> list[tuple[dict, dict]]:
    """
    Compute n/alpha for each individual trace at one air/point position.
    Pairs ref[i] with sam[i] when counts match; uses avg ref otherwise.
    """
    n_ref   = ref_ds.all_y.shape[0]
    n_sam   = sam_ds.all_y.shape[0]
    results = []

    for j in range(n_sam):
        meas_id = f"{pos_name}_m{j:02d}"
        y_sam_j = sam_ds.all_y[j]
        y_ref_j = ref_ds.all_y[j] if j < n_ref else ref_ds.y_avg

        try:
            t_com, y_ref_c, y_sam_c = ensure_common_time_axis(
                ref_ds.x_ps, y_ref_j, sam_ds.x_ps, y_sam_j,
            )
            f_THz, n_f, _, _, alpha_cm = compute_refractive_index_arrays(
                t_com, y_ref_c, y_sam_c, thickness_m, f_low, f_high,
            )
        except Exception as exc:
            msg = f"  SKIP {sample_id}/{meas_id}: {exc}"
            tqdm.write(msg)
            skipped_log.append(msg)
            continue

        if not np.any(np.isfinite(n_f)) or not np.any(np.isfinite(alpha_cm)):
            msg = f"  SKIP {sample_id}/{meas_id}: all-NaN result"
            tqdm.write(msg)
            skipped_log.append(msg)
            continue

        feat, spec = extract_cielecki_features(
            f_THz, n_f, alpha_cm,
            material, sample_id, meas_id, METHOD,
            f_low, f_high,
        )

        if not spec["fit_success"]:
            msg = f"  WARN {sample_id}/{meas_id}: fit failed — {spec['fit_error']}"
            tqdm.write(msg)
            skipped_log.append(msg)

        results.append((feat, spec))

    return results


# ── per-sample orchestration ─────────────────────────────────────────────────

def process_sample(
    sample_entry: dict,
    shared: dict,
    out_dirs: dict,
    f_low: float,
    f_high: float,
    fig_size: tuple,
    dpi: int,
    formats: list[str],
    skipped_log: list[str],
) -> list[dict]:
    """
    Load all positions for one sample, extract per-trace features,
    save spectra TSVs, generate per-sample plots.
    Returns list of feature_row dicts.
    """
    name        = sample_entry["name"]
    sample_dir  = Path(sample_entry["dir"])
    thickness_m = float(sample_entry["thickness_m"])
    material    = sample_entry.get("material", _material_from_name(name))

    load_cfg  = shared["loading"]
    align_cfg = shared["alignment"]

    _ds_kwargs = dict(
        correction_factor       = load_cfg["correction_factor"],
        align                   = align_cfg["enabled"],
        align_method            = align_cfg["method"],
        corr_window_ps          = align_cfg["corr_window_ps"],
        adaptive_fit_window     = align_cfg.get("adaptive_fit_window", True),
        fit_window_sigma_factor = align_cfg.get("fit_window_sigma_factor", 5.0),
        fallback_fit_window_ps  = align_cfg.get("fallback_fit_window_ps", 15.0),
        min_sigma_ps            = align_cfg.get("min_sigma_ps", 0.05),
        max_sigma_ps            = align_cfg.get("max_sigma_ps", 10.0),
        min_fit_points          = align_cfg.get("min_fit_points", 10),
        max_center_shift_ps     = align_cfg.get("max_center_shift_ps", 5.0),
        max_traces              = load_cfg.get("max_traces"),
    )

    pairs = discover_pairs(
        sample_dir,
        align_cfg.get("air_prefix",   "air"),
        align_cfg.get("point_prefix", "point"),
        align_cfg.get("n_points",     None),
    )
    if not pairs:
        tqdm.write(f"  [{name}] No matched pairs — skipping.")
        return []

    traces_str = str(_ds_kwargs["max_traces"]) if _ds_kwargs["max_traces"] else "all"
    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"  {name}  |  {len(pairs)} positions  |  {thickness_m*1e3:.3f} mm  |  {traces_str} traces/folder")
    tqdm.write(f"{'='*60}")

    all_records: list[tuple[dict, dict]] = []
    point_prefix = align_cfg.get("point_prefix", "point")

    for idx, air_path, point_path in pairs:
        pos_name = f"{point_prefix}{idx}"
        ref_ds   = thz_tds.THZDataset(path=air_path,   **_ds_kwargs)
        sam_ds   = thz_tds.THZDataset(path=point_path, **_ds_kwargs)

        records = process_position(
            ref_ds, sam_ds, thickness_m,
            material, name, pos_name,
            f_low, f_high, skipped_log,
        )

        for feat, spec in records:
            save_measurement_spectrum_with_fit(
                out_dirs["spectra"], name, feat["measurement_id"], spec,
            )
            status = "" if spec["fit_success"] else "  [fit failed]"
            tqdm.write(
                f"  {feat['measurement_id']}: "
                f"n_mean={feat['n_mean']:.4f}  "
                f"beta={feat['beta']:.3f}  "
                f"alpha0={feat['alpha0']:.3f}"
                + status
            )

        all_records.extend(records)

    if all_records:
        plot_sample_curves(
            out_dirs["plots"], name, all_records,
            f_low, f_high, fig_size, dpi, formats,
        )

    return [feat for feat, _ in all_records]


# ── main ─────────────────────────────────────────────────────────────────────

def main(config_path: str | Path = CONFIG_DEFAULT) -> None:
    cfg = load_yaml(config_path)

    output_root = Path(cfg["output"])
    # `features` section overrides `refractive_index` band limits when present
    band_cfg = cfg.get("features", cfg["refractive_index"])
    plot_cfg = cfg["plot"]
    shared   = {k: cfg[k] for k in ("loading", "alignment", "refractive_index", "plot")}

    f_low  = float(band_cfg.get("f_low_THz",  0.4))
    f_high = float(band_cfg.get("f_high_THz", 1.0))

    fig_size = tuple(plot_cfg["figure_size"])
    dpi      = plot_cfg["dpi"]
    formats  = plot_cfg["formats"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = output_root / f"cielecki_{timestamp}"

    out_dirs = {
        "features": run_dir / "features",
        "spectra":  run_dir / "spectra",
        "plots":    run_dir / "plots",
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"Config   : {config_path}")
    print(f"Samples  : {len(cfg['samples'])}")
    print(f"Band     : {f_low}–{f_high} THz")
    print(f"Output   : {run_dir}\n")

    skipped_log:  list[str]  = []
    all_features: list[dict] = []

    for entry in tqdm(cfg["samples"], desc="Samples", unit="sample"):
        feats = process_sample(
            entry, shared, out_dirs,
            f_low, f_high, fig_size, dpi, formats,
            skipped_log,
        )
        all_features.extend(feats)

    if not all_features:
        print("No features extracted — nothing to save.")
        return

    df = pd.DataFrame(all_features)

    # ── per-measurement feature table ─────────────────────────────────────
    feat_cols = [
        "material", "sample_id", "measurement_id", "method",
        "n_mean", "beta", "alpha0", "fit_rmse",
        "f_low_THz", "f_high_THz", "n_points_used",
    ]
    df[feat_cols].to_csv(
        out_dirs["features"] / "per_measurement_features.tsv",
        sep="\t", index=False, float_format="%.6f",
    )

    # ── sample summary table ──────────────────────────────────────────────
    summary = (
        df.groupby(["material", "sample_id", "method"])
        .agg(
            n_mean_avg    = ("n_mean",   "mean"),
            n_mean_std    = ("n_mean",   "std"),
            beta_avg      = ("beta",     "mean"),
            beta_std      = ("beta",     "std"),
            alpha0_avg    = ("alpha0",   "mean"),
            alpha0_std    = ("alpha0",   "std"),
            n_measurements= ("n_mean",   "count"),
        )
        .reset_index()
    )
    summary.to_csv(
        out_dirs["features"] / "sample_summary_features.tsv",
        sep="\t", index=False, float_format="%.6f",
    )

    # ── material maps ─────────────────────────────────────────────────────
    plot_material_map(out_dirs["plots"], df, fig_size, dpi, formats)

    # ── printed cross-sample summary ──────────────────────────────────────
    W = 90
    print(f"\n{'=' * W}")
    print("FEATURE SUMMARY (per sample)")
    print(f"{'=' * W}")
    print(
        f"{'Sample':<18}  {'Material':<8}  {'#':>4}  "
        f"{'n_mean':>8}  {'±':>6}  {'beta':>8}  {'±':>7}  "
        f"{'alpha0':>8}  {'±':>7}"
    )
    print("-" * W)
    for _, row in summary.iterrows():
        n_std_str    = f"{row.get('n_mean_std', 0.0):>6.4f}"
        beta_std_str = f"{row.get('beta_std',   0.0):>7.3f}"
        a0_std_str   = f"{row.get('alpha0_std', 0.0):>7.3f}"
        print(
            f"{row['sample_id']:<18}  {row['material']:<8}  "
            f"{int(row['n_measurements']):>4}  "
            f"{row['n_mean_avg']:>8.4f}  {n_std_str}  "
            f"{row['beta_avg']:>8.3f}  {beta_std_str}  "
            f"{row['alpha0_avg']:>8.3f}  {a0_std_str}"
        )

    if skipped_log:
        print(f"\n{'─' * W}")
        print(f"SKIPPED / WARNINGS ({len(skipped_log)}):")
        for msg in skipped_log:
            print(msg)

    print(f"\nAll results saved in: {run_dir}")
    print(f"  features/per_measurement_features.tsv")
    print(f"  features/sample_summary_features.tsv")
    print(f"  spectra/<sample_id>/<meas_id>_n_alpha_fit.tsv")
    print(f"  plots/<sample_id>/{{n,alpha,alpha_fit}}_curves.png")
    print(f"  plots/material_map_nmean_beta.png")
    print(f"  plots/material_map_errorbars.png")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else CONFIG_DEFAULT
    main(path)
