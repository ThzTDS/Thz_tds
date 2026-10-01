"""
sparse_batch_and_features.py
============================
Sparse-frequency version of batch_and_features.py: runs the sparse batch
(fp_all_samples_saprse_batch) + material features for every sample in the
YAML config in a single pass.

No broadband H(f), n(f), or alpha(f) is calculated.  The complex Fourier
coefficient is evaluated only at the configured frequencies and the models

    n(f)     = n0 + n_slope * (f - f_center)
    alpha(f) = alpha0 + beta * f**2

are fitted per measurement.  Dense plot lines are model evaluations.

Pipeline
--------
1. Sparse batch — per sample: tau / n from phase pairs, alpha at the sparse
   points, fitted n(f) / alpha(f) models; save TSVs and cross-sample overlay
   figures.
2. Features     — per individual measurement: n_mean (= n0) and beta, save
   per-measurement sparse spectra, feature tables, and 2D material maps.

Both pipelines write into the same timestamped output directory:

    output_root/
      {timestamp}/
        batch/       ← per-sample TSVs + cross-sample overlay figures
        features/    ← per_measurement_features.tsv, sample_summary_features.tsv
        spectra/     ← per-measurement sparse n / alpha / fit TSVs
        plots/       ← per-sample curves + material map figures

Usage
-----
    python examples/batch/sparse_batch_and_features.py
    python examples/batch/sparse_batch_and_features.py config/all_samples_sparse_fp_batch.yaml
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

import thz_tds
from thz_tds import viz

from fp_all_samples_saprse_batch import (
    load_yaml,
    discover_pairs,
    _label,
    _dataset_kwargs,
    transfer_at_sparse_frequencies,
    extract_sparse_material,
    material_curves,
)
from batch_and_features import _material_from_name, plot_material_map


CONFIG_DEFAULT = Path(__file__).parent.parent.parent / "config" / "all_samples_sparse_fp_batch.yaml"


# ══════════════════════════════════════════════════════════════════════════════
# Per-sample sparse extraction (shared by both pipelines)
# ══════════════════════════════════════════════════════════════════════════════

def run_sample_sparse(
    sample_entry: dict,
    cfg: dict,
    batch_dir: Path,
    timestamp: str,
    skipped_log: list[str],
) -> dict | None:
    """
    Load all positions for one sample, extract sparse tau / n / alpha,
    print summary table, save TSVs.
    Returns band statistics for cross-sample plotting.
    """
    name        = sample_entry["name"]
    sample_dir  = Path(sample_entry["dir"])
    thickness_m = float(sample_entry["thickness_m"])
    load_cfg    = cfg["loading"]
    align_cfg   = cfg["alignment"]
    sparse_cfg  = cfg["selected_frequencies"]

    point_prefix    = align_cfg.get("point_prefix", "point")
    frequencies_THz = np.asarray(sparse_cfg["frequencies_THz"], dtype=float)
    remove_dc       = bool(sparse_cfg.get("remove_dc", True))
    floor_rel       = float(sparse_cfg.get("reference_floor_rel", 1e-8))

    pairs = discover_pairs(
        sample_dir,
        align_cfg.get("air_prefix", "air"),
        point_prefix,
        align_cfg.get("n_points"),
    )
    if not pairs:
        tqdm.write(f"  [{name}] No matched pairs found — skipping.")
        return None

    ds_kw = _dataset_kwargs(load_cfg, align_cfg)
    traces_str = str(ds_kw["max_traces"]) if ds_kw["max_traces"] else "all"
    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"  {name}  |  {len(pairs)} positions  |  {thickness_m*1e3:.3f} mm  |  {traces_str} traces/folder")
    tqdm.write(f"{'='*60}")

    plot_low, plot_high = map(float, sparse_cfg.get(
        "plot_frequency_range_THz", [frequencies_THz[0], frequencies_THz[-1]],
    ))
    plot_grid = np.linspace(plot_low, plot_high, int(sparse_cfg.get("plot_grid_points", 500)))

    positions: list[dict] = []
    for idx, air_path, point_path in pairs:
        pos_name = f"{point_prefix}{idx}"
        ref = thz_tds.THZDataset(path=air_path,   **ds_kw)
        sam = thz_tds.THZDataset(path=point_path, **ds_kw)
        try:
            E_ref, E_sam, H = transfer_at_sparse_frequencies(
                ref, sam, frequencies_THz, remove_dc, floor_rel,
            )
            res = extract_sparse_material(frequencies_THz, H, thickness_m, sparse_cfg)
        except (RuntimeError, ValueError) as exc:
            msg = f"  SKIP {name}/{pos_name}: {exc}"
            tqdm.write(msg); skipped_log.append(msg)
            continue

        n_grid, alpha_grid = material_curves(
            plot_grid, res["n0"], res["n_slope_per_THz"],
            res["alpha0_cm"], res["beta_cm_per_THz2"], res["center_THz"],
        )
        res.update({
            "name":       pos_name,
            "E_ref":      E_ref,
            "E_sam":      E_sam,
            "n_grid":     n_grid,
            "alpha_grid": alpha_grid,
        })
        positions.append(res)

        pairs_text = ", ".join(
            f"tau={x['tau_ps']:.3f} ps, n={x['n_group']:.4f}" for x in res["phase_pairs"]
        )
        tqdm.write(f"  {pos_name}: {pairs_text}; alpha0={res['alpha0_cm']:.4f}, "
                   f"beta={res['beta_cm_per_THz2']:.4f}")

    if not positions:
        tqdm.write(f"  [{name}] No positions extracted — skipping.")
        return None

    # statistics across positions
    n_pair_all     = np.array([p["n_pair"]          for p in positions])
    tau_all        = np.array([p["tau_ps"]          for p in positions])
    alpha_pts_all  = np.array([p["alpha_points_cm"] for p in positions])
    n_grid_all     = np.array([p["n_grid"]          for p in positions])
    alpha_grid_all = np.array([p["alpha_grid"]      for p in positions])

    per_n0     = np.array([p["n0"]                    for p in positions])
    per_slope  = np.array([p["n_slope_per_THz"]       for p in positions])
    per_alpha0 = np.array([p["alpha0_cm"]             for p in positions])
    per_beta   = np.array([p["beta_cm_per_THz2"]      for p in positions])
    per_rmse   = np.array([p["relative_complex_rmse"] for p in positions])

    tqdm.write(f"\n  {'Pos':<10}  {'n0':>8}  {'dn/df':>8}  {'alpha0':>9}  {'beta':>9}  {'H RMSE':>9}")
    tqdm.write(f"  {'-'*62}")
    for i, p in enumerate(positions):
        tqdm.write(f"  {p['name']:<10}  {per_n0[i]:>8.4f}  {per_slope[i]:>+8.4f}  "
                   f"{per_alpha0[i]:>9.4f}  {per_beta[i]:>9.4f}  {per_rmse[i]:>9.4g}")
    tqdm.write(f"  {'-'*62}")
    tqdm.write(f"  {'MEAN':<10}  {per_n0.mean():>8.4f}  {per_slope.mean():>+8.4f}  "
               f"{per_alpha0.mean():>9.4f}  {per_beta.mean():>9.4f}  {per_rmse.mean():>9.4g}")
    tqdm.write(f"  {'STD':<10}  {per_n0.std():>8.4f}  {per_slope.std():>8.4f}  "
               f"{per_alpha0.std():>9.4f}  {per_beta.std():>9.4f}")

    # save sparse-point TSV
    points_path = batch_dir / f"{name}_sparse_points_{timestamp}.tsv"
    with points_path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["position", "frequency_THz", "H_real", "H_imag", "H_abs",
                         "H_phase_rad", "n_approx", "alpha_point_cm-1", "alpha_curve_cm-1"])
        for p in positions:
            for k, frequency in enumerate(frequencies_THz):
                H_k = p["H_meas"][k]
                writer.writerow([p["name"], frequency, H_k.real, H_k.imag, abs(H_k),
                                 np.angle(H_k), p["n_points"][k],
                                 p["alpha_points_cm"][k], p["alpha_fit_points_cm"][k]])

    # save phase-pair TSV
    pairs_path = batch_dir / f"{name}_phase_pairs_{timestamp}.tsv"
    with pairs_path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["position", "f1_THz", "f2_THz", "center_THz", "delta_f_GHz",
                         "delta_phi_wrapped_rad", "selected_cycle",
                         "delta_phi_selected_rad", "tau_ps", "n_group"])
        for p in positions:
            for pair in p["phase_pairs"]:
                writer.writerow([p["name"], pair["f1_THz"], pair["f2_THz"],
                                 pair["center_THz"], pair["delta_f_GHz"],
                                 pair["delta_phi_wrapped_rad"], pair["cycle"],
                                 pair["delta_phi_selected_rad"], pair["tau_ps"],
                                 pair["n_group"]])

    # save model-curve TSV (same layout as the broadband batch results TSV)
    header = ["frequency_THz"]
    cols   = [plot_grid]
    for p in positions:
        header += [f"{p['name']}_n", f"{p['name']}_alpha"]
        cols   += [p["n_grid"], p["alpha_grid"]]
    header += ["avg_n", "avg_alpha"]
    cols   += [n_grid_all.mean(axis=0), alpha_grid_all.mean(axis=0)]
    curves_path = batch_dir / f"{name}_sparse_curves_{timestamp}.tsv"
    np.savetxt(curves_path, np.column_stack(cols),
               delimiter="\t", header="\t".join(header), comments="")

    for path in (points_path, pairs_path, curves_path):
        tqdm.write(f"  Saved: {path}")

    return {
        "name":             name,
        "material":         sample_entry.get("material", _material_from_name(name)),
        "thickness_mm":     thickness_m * 1e3,
        "n_positions":      len(positions),
        "positions":        positions,
        "frequencies_THz":  frequencies_THz,
        "pair_centers_THz": positions[0]["pair_centers_THz"],
        "plot_grid_THz":    plot_grid,
        "avg_n_pair":       n_pair_all.mean(axis=0),    "std_n_pair":      n_pair_all.std(axis=0),
        "avg_tau_ps":       tau_all.mean(axis=0),       "std_tau_ps":      tau_all.std(axis=0),
        "avg_alpha_points": alpha_pts_all.mean(axis=0), "std_alpha_points": alpha_pts_all.std(axis=0),
        "avg_n_grid":       n_grid_all.mean(axis=0),    "std_n_grid":      n_grid_all.std(axis=0),
        "avg_alpha_grid":   alpha_grid_all.mean(axis=0), "std_alpha_grid":  alpha_grid_all.std(axis=0),
        "n0_mean":          per_n0.mean(),     "n0_std":     per_n0.std(),
        "n_slope_mean":     per_slope.mean(),
        "alpha0_mean":      per_alpha0.mean(), "alpha0_std": per_alpha0.std(),
        "beta_mean":        per_beta.mean(),   "beta_std":   per_beta.std(),
        "rmse_mean":        per_rmse.mean(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline 1 — sparse batch
# ══════════════════════════════════════════════════════════════════════════════

def _run_sparse_batch(cross_sample: list[dict], cfg: dict, batch_dir: Path, timestamp: str) -> None:
    """Print/save the cross-sample summary and save cross-sample overlay figures."""
    plot_cfg = cfg["plot"]

    fig_size       = tuple(plot_cfg["figure_size"])
    dpi            = plot_cfg["dpi"]
    formats        = plot_cfg["formats"]
    line_width     = plot_cfg.get("lw_curve",          1.8)
    trace_width    = plot_cfg.get("lw_trace",          0.8)
    errorbar_width = plot_cfg.get("lw_errorbar",       1.0)
    marker_size    = plot_cfg.get("marker_size",       4.5)
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

    _ylim_cfg = cfg.get("ylim", {}) or {}
    ylim_n    = _ylim_cfg.get("n")
    ylim_a    = _ylim_cfg.get("alpha")
    ylim_tau  = _ylim_cfg.get("tau_ps")

    # cross-sample summary table
    W = 100
    print(f"\n{'=' * W}\nSPARSE BATCH CROSS-SAMPLE SUMMARY\n{'=' * W}")
    print(f"{'Sample':<18}  {'mm':>5}  {'pts':>4}  {'n0':>8}  {'±':>6}  {'dn/df':>8}  "
          f"{'alpha0':>9}  {'±':>7}  {'beta':>9}  {'±':>7}  {'H RMSE':>8}")
    print("-" * W)
    for s in cross_sample:
        print(f"{s['name']:<18}  {s['thickness_mm']:>5.2f}  {s['n_positions']:>4}  "
              f"{s['n0_mean']:>8.4f}  {s['n0_std']:>6.4f}  {s['n_slope_mean']:>+8.4f}  "
              f"{s['alpha0_mean']:>9.4f}  {s['alpha0_std']:>7.4f}  "
              f"{s['beta_mean']:>9.4f}  {s['beta_std']:>7.4f}  {s['rmse_mean']:>8.4g}")

    summary_path = batch_dir / f"sparse_cross_sample_{timestamp}.tsv"
    with summary_path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["sample", "thickness_mm", "n_positions",
                         "n0_mean", "n0_std", "n_slope_mean_per_THz",
                         "alpha0_mean_cm-1", "alpha0_std_cm-1",
                         "beta_mean_cm-1_THz-2", "beta_std_cm-1_THz-2",
                         "relative_complex_rmse_mean"])
        for s in cross_sample:
            writer.writerow([s["name"], s["thickness_mm"], s["n_positions"],
                             s["n0_mean"], s["n0_std"], s["n_slope_mean"],
                             s["alpha0_mean"], s["alpha0_std"],
                             s["beta_mean"], s["beta_std"], s["rmse_mean"]])
    print(f"  Saved: {summary_path}")

    # overlay figures
    single = len(cross_sample) == 1

    def _finish(ax, xlabel, ylabel, title, ylim, loc):
        ax.set_xlabel(xlabel)
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

    def _overlay(ax, grid_avg, grid_pos_key, pts_x, pts_avg, pts_std,
                 ylabel, title, points=True, ylim=None, loc=None):
        for s in cross_sample:
            f   = s["plot_grid_THz"]
            avg = s[grid_avg]
            col = s["color"]
            lbl = _label(s["name"])
            if single:
                for p in s["positions"]:
                    ax.plot(f, p[grid_pos_key], color=col, alpha=0.3,
                            lw=trace_width, label="_nolegend_")
            ax.plot(f, avg, color=col, lw=line_width, label=lbl)
            if points:
                ax.errorbar(
                    s[pts_x], s[pts_avg], yerr=s[pts_std],
                    fmt="o", color=col, ecolor=col, ms=marker_size,
                    elinewidth=errorbar_width, capsize=capsize, capthick=errorbar_width,
                    alpha=0.9, label="_nolegend_",
                )
            if use_inline_labels:
                x_lbl = label_x if label_x is not None else f[-1]
                ax.text(x_lbl, avg[-1], lbl,
                        color=col, fontsize=label_fontsize,
                        va="center", ha="left", clip_on=True)
        _finish(ax, "Frequency (THz)", ylabel, title, ylim, loc)

    fig_n,      ax_n      = plt.subplots(figsize=fig_size)
    fig_a,      ax_a      = plt.subplots(figsize=fig_size)
    fig_n_mean, ax_n_mean = plt.subplots(figsize=fig_size)
    fig_a_mean, ax_a_mean = plt.subplots(figsize=fig_size)
    fig_tau,    ax_tau    = plt.subplots(figsize=fig_size)

    _overlay(ax_n,      "avg_n_grid", "n_grid",
             "pair_centers_THz", "avg_n_pair", "std_n_pair",
             "Refractive index $n$", "Sparse refractive index",
             ylim=ylim_n)
    _overlay(ax_a,      "avg_alpha_grid", "alpha_grid",
             "frequencies_THz", "avg_alpha_points", "std_alpha_points",
             r"Absorption coefficient, $\alpha$ [cm$^{-1}$]", "Sparse absorption",
             ylim=ylim_a, loc=legend_loc_alpha)
    _overlay(ax_n_mean, "avg_n_grid", "n_grid",
             "pair_centers_THz", "avg_n_pair", "std_n_pair",
             "Refractive index $n$", "Mean sparse refractive index",
             points=False, ylim=ylim_n)
    _overlay(ax_a_mean, "avg_alpha_grid", "alpha_grid",
             "frequencies_THz", "avg_alpha_points", "std_alpha_points",
             r"Absorption coefficient, $\alpha$ [cm$^{-1}$]", "Mean sparse absorption",
             points=False, ylim=ylim_a, loc=legend_loc_alpha)

    for s in cross_sample:
        ax_tau.errorbar(
            s["pair_centers_THz"], s["avg_tau_ps"], yerr=s["std_tau_ps"],
            fmt="o-", color=s["color"], lw=line_width, ms=marker_size,
            elinewidth=errorbar_width, capsize=capsize, capthick=errorbar_width,
            label=_label(s["name"]),
        )
    _finish(ax_tau, "Phase-pair centre frequency (THz)", r"Delay $\tau$ (ps)",
            "Delay from sparse phase pairs", ylim_tau, legend_loc)

    figures_to_save = [
        (fig_n,      "n"),
        (fig_a,      "alpha"),
        (fig_n_mean, "n_mean"),
        (fig_a_mean, "alpha_mean"),
        (fig_tau,    "tau"),
    ]

    for fig_obj, _ in figures_to_save:
        fig_obj.tight_layout(pad=0.3)

    for fig_obj, tag in figures_to_save:
        stem = f"batch_sparse_{tag}"
        for fmt in formats:
            path = batch_dir / f"{stem}.{fmt}"
            bbox = None if fmt in ("svg", "pdf") else "tight"
            fig_obj.savefig(path, dpi=dpi, bbox_inches=bbox)
        print(f"  Saved: {batch_dir / stem}.{formats[0]}")
        plt.close(fig_obj)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline 2 — material features
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(sample: dict, position: dict, method: str) -> tuple[dict, dict]:
    """Build one feature row + sparse spectrum from a sparse extraction result."""
    f_THz     = position["frequencies_THz"]
    alpha     = position["alpha_points_cm"]
    alpha_fit = position["alpha_fit_points_cm"]
    residual  = alpha - alpha_fit

    feature_row = dict(
        material=sample["material"], sample_id=sample["name"],
        measurement_id=position["name"], method=method,
        n_mean=float(np.mean(position["n_points"])),
        beta=position["beta_cm_per_THz2"], alpha0=position["alpha0_cm"],
        fit_rmse=float(np.sqrt(np.mean(residual ** 2))),
        f_low_THz=float(f_THz[0]), f_high_THz=float(f_THz[-1]),
        n_points_used=int(f_THz.size),
    )
    spectrum = dict(
        f_THz=f_THz, n=position["n_points"], alpha_cm1=alpha,
        alpha_fit=alpha_fit, residual=residual,
        H=position["H_meas"],
        pair_centers_THz=position["pair_centers_THz"], n_pair=position["n_pair"],
        grid_THz=sample["plot_grid_THz"],
        n_grid=position["n_grid"], alpha_grid=position["alpha_grid"],
    )
    return feature_row, spectrum


def save_spectrum_tsv(spectra_dir: Path, sample_id: str,
                      measurement_id: str, spectrum: dict) -> None:
    out_dir = spectra_dir / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    header = ("frequency_THz\tn\talpha_cm1\talpha_fit_cm1\talpha_residual_cm1"
              "\tH_abs\tH_phase_rad")
    np.savetxt(
        out_dir / f"{measurement_id}_sparse_n_alpha_fit.tsv",
        np.column_stack([spectrum["f_THz"], spectrum["n"], spectrum["alpha_cm1"],
                         spectrum["alpha_fit"], spectrum["residual"],
                         np.abs(spectrum["H"]), np.angle(spectrum["H"])]),
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
        lbl = row["measurement_id"]
        ax_n.plot(sp["grid_THz"], sp["n_grid"], color=col, lw=1.0, label=lbl)
        ax_n.plot(sp["pair_centers_THz"], sp["n_pair"], "o", color=col, ms=4,
                  label="_nolegend_")
        ax_a.plot(sp["f_THz"], sp["alpha_cm1"], "o-", color=col, lw=1.0, ms=4, label=lbl)
        ax_af.plot(sp["f_THz"], sp["alpha_cm1"], "o", color=col, ms=4, label=lbl)
        ax_af.plot(sp["grid_THz"], sp["alpha_grid"], "--", color=col, lw=1.3,
                   label=f"{lbl} fit")

    for ax, ylabel, title in [
        (ax_n,  "Refractive index $n$",  f"{sample_id} — sparse n(f)"),
        (ax_a,  r"Absorption coefficient, $\alpha$ [cm$^{-1}$]", f"{sample_id} — sparse α(f)"),
        (ax_af, r"Absorption coefficient, $\alpha$ [cm$^{-1}$]", f"{sample_id} — sparse α(f) + fit"),
    ]:
        ax.axvspan(f_low, f_high, alpha=0.07, color="grey")
        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    for fig_obj, stem in [(fig_n, "n_curves"), (fig_a, "alpha_curves"), (fig_af, "alpha_fit_curves")]:
        fig_obj.tight_layout()
        viz.save_figure(fig_obj, out_dir, stem=stem, formats=formats, dpi=dpi)
        plt.close(fig_obj)


def _run_features(cross_sample: list[dict], cfg: dict, feat_dirs: dict,
                  skipped_log: list[str]) -> None:
    """Run the material-features pipeline on the sparse extraction results."""
    sparse_cfg = cfg["selected_frequencies"]
    plot_cfg   = cfg["plot"]
    method     = f"sparse_{str(sparse_cfg.get('model', 'fp')).lower()}"

    fig_size = tuple(plot_cfg["figure_size"])
    dpi      = plot_cfg["dpi"]
    formats  = plot_cfg["formats"]

    all_features: list[dict] = []

    for s in tqdm(cross_sample, desc="[2/2] Features", unit="sample"):
        records: list[tuple[dict, dict]] = []
        for p in s["positions"]:
            feat, spec = extract_features(s, p, method)
            save_spectrum_tsv(feat_dirs["spectra"], s["name"], feat["measurement_id"], spec)
            records.append((feat, spec))
        f_THz = s["frequencies_THz"]
        plot_sample_curves(feat_dirs["plots"], s["name"], records,
                           float(f_THz[0]), float(f_THz[-1]), fig_size, dpi, formats)
        all_features.extend(feat for feat, _ in records)

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
    sparse_cfg = cfg.get("selected_frequencies", {}) or {}
    if not sparse_cfg.get("enabled", False):
        raise ValueError("selected_frequencies.enabled must be true")

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

    frequencies_THz = np.asarray(sparse_cfg["frequencies_THz"], dtype=float)
    print(f"Config     : {config_path}")
    print(f"Samples    : {len(cfg['samples'])}")
    print(f"Sparse THz : {', '.join(f'{f:.3f}' for f in frequencies_THz)}")
    print(f"Model      : {sparse_cfg.get('model', 'fp')}")
    print(f"Output     : {run_dir}\n")

    skipped_log:  list[str]  = []
    cross_sample: list[dict] = []
    sample_colors = cm.tab10(np.linspace(0, 1, max(len(cfg["samples"]), 1)))

    for entry, s_col in zip(
        tqdm(cfg["samples"], desc="[1/2] Sparse batch", unit="sample"),
        sample_colors,
    ):
        summary = run_sample_sparse(entry, cfg, batch_dir, timestamp, skipped_log)
        if summary:
            summary["color"] = s_col
            cross_sample.append(summary)

    if not cross_sample:
        print("  No samples processed in sparse batch.")
        return

    _run_sparse_batch(cross_sample, cfg, batch_dir, timestamp)
    _run_features(cross_sample, cfg, feat_dirs, skipped_log)

    print(f"\n{'=' * 60}")
    print(f"All results saved in: {run_dir}")
    print(f"  batch/               — sparse points / phase pairs / curves TSVs + overlay figures")
    print(f"  features/            — per_measurement_features.tsv, sample_summary_features.tsv")
    print(f"  spectra/<sample>/    — per-measurement sparse n / alpha / fit TSVs")
    print(f"  plots/<sample>/      — n_curves, alpha_curves, alpha_fit_curves")
    print(f"  plots/               — material_map_nmean_beta, material_map_errorbars")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else CONFIG_DEFAULT
    main(path)
