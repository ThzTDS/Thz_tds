"""
fp_all_samples_batch_complex_iterative.py
==========================================
Run iterative complex-Fresnel refractive index extraction for every sample
listed in all_samples_fp_batch.yaml, optionally comparing against the
existing Fabry-Perot (FP) correction.

This is a variant of ``fp_all_samples_batch.py`` that replaces the
real-Fresnel extraction (``thz_tds.refractive.compute_refractive_index``)
with the iterative complex-Fresnel extraction
(``thz_tds.refractive_complex_iterative.compute_refractive_index``, see
that module's docstring for the sign convention and algorithm).

Two distinct, separate physical corrections are reported here — they must
not be confused:

* **Complex-Fresnel correction** (this script's new default): replaces the
  real-valued Fresnel transmission factor F_real(n) = 4n/(1+n)**2 with the
  full complex factor F(ntilde) = 4*ntilde/(1+ntilde)**2, evaluated at the
  complex refractive index ntilde = n - i*kappa found by fixed-point
  iteration. This changes the *interface* transmission magnitude and phase.
  It does **not** account for internal multiple reflections.

* **Fabry-Perot (FP) correction** (``thz_tds.refractive_fp``, unchanged):
  accounts for internal multiple reflections inside the slab via the full
  FP slab transfer function. It is a separate physical effect from the
  complex-Fresnel correction above, and is only applied when
  ``skip_fp: false`` in the YAML config.

Columns/labels used throughout this script:

    n_complex_fresnel, alpha_complex_fresnel   - iterative complex-Fresnel
                                                  result (this module)
    n_with_fp, alpha_with_fp                   - existing FP-corrected
                                                  result (refractive_fp.py),
                                                  only when skip_fp: false

IMPORTANT: ``thz_tds.refractive_fp.compute_refractive_index_fp`` seeds its
own nonlinear fit internally from the *real*-Fresnel initial guess (see
``compute_refractive_index_fp_arrays`` step 3, which calls
``thz_tds.refractive.compute_refractive_index_arrays``). Its public API does
not accept a caller-supplied initial n/alpha, so the FP-corrected result
here is **not** seeded from this script's complex-Fresnel iterative
estimate — the two columns are independent extractions of the same data,
not an init->refine pipeline. ``refractive_fp.py`` is left unmodified.

Usage
-----
    python examples/fp_all_samples_batch_complex_iterative.py
    python examples/fp_all_samples_batch_complex_iterative.py config/all_samples_fp_batch.yaml

Optional YAML keys (all have defaults; the script works if this section is
absent entirely):

    refractive_index:
      complex_fresnel:
        max_iter: 100
        tol_n: 1.0e-8
        tol_alpha: 1.0e-6
        relaxation: 1.0
        clip_negative_alpha: false
"""

import re
import sys
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

import thz_tds
from thz_tds.refractive_fp import compute_refractive_index_fp
from thz_tds.refractive_complex_iterative import compute_refractive_index
from thz_tds import viz


CONFIG_DEFAULT = Path(__file__).parent.parent.parent / "config" / "all_samples_fp_batch.yaml"


# ── helpers ──────────────────────────────────────────────────────────────────

def _label(name: str) -> str:
    """Keep material and variant, drop thickness: 'PA6_B_3' → 'PA6_B'."""
    parts = name.split('_')
    return '_'.join(parts[:2]) if len(parts) >= 2 else parts[0]


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def discover_pairs(
    sample_dir: Path,
    air_prefix: str,
    point_prefix: str,
    n_points: int | None,
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


def run_sample(
    sample_entry: dict,
    shared: dict,
    output_root: Path,
    timestamp: str = "",
) -> dict | None:
    """
    Process one sample: load datasets, compute the iterative complex-Fresnel
    result (and, unless ``skip_fp``, the existing FP-corrected result),
    print per-position convergence/log info and a summary table, save TSV.
    Returns all band data for combined plotting in main() — no figures are
    created here.
    """
    name        = sample_entry["name"]
    sample_dir  = Path(sample_entry["dir"])
    thickness   = float(sample_entry["thickness_m"])

    load_cfg  = shared["loading"]
    align_cfg = shared["alignment"]
    ri_cfg    = shared["refractive_index"]

    air_prefix   = align_cfg.get("air_prefix",   "air")
    point_prefix = align_cfg.get("point_prefix", "point")
    n_points     = align_cfg.get("n_points",     None)

    correction_factor       = load_cfg["correction_factor"]
    max_traces              = load_cfg.get("max_traces")
    align_enabled           = align_cfg["enabled"]
    align_method            = align_cfg["method"]
    corr_window_ps          = align_cfg["corr_window_ps"]
    adaptive_fit_window     = align_cfg.get("adaptive_fit_window", True)
    fit_window_sigma_factor = align_cfg.get("fit_window_sigma_factor", 5.0)
    fallback_fit_window_ps  = align_cfg.get("fallback_fit_window_ps", 15.0)
    min_sigma_ps            = align_cfg.get("min_sigma_ps", 0.05)
    max_sigma_ps            = align_cfg.get("max_sigma_ps", 10.0)
    min_fit_points          = align_cfg.get("min_fit_points", 10)
    max_center_shift_ps     = align_cfg.get("max_center_shift_ps", 5.0)

    f_low   = ri_cfg["f_low_THz"]
    f_high  = ri_cfg["f_high_THz"]
    skip_fp = ri_cfg.get("skip_fp", False)

    # ── optional iterative complex-Fresnel settings (all have defaults) ──
    cf_cfg               = (ri_cfg.get("complex_fresnel") or {})
    cf_max_iter           = cf_cfg.get("max_iter", 100)
    cf_tol_n               = cf_cfg.get("tol_n", 1.0e-8)
    cf_tol_alpha           = cf_cfg.get("tol_alpha", 1.0e-6)
    cf_relaxation           = cf_cfg.get("relaxation", 1.0)
    cf_clip_negative_alpha = cf_cfg.get("clip_negative_alpha", False)

    output_root.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(sample_dir, air_prefix, point_prefix, n_points)
    if not pairs:
        tqdm.write(f"  [{name}] No matched pairs found — skipping.")
        return None

    traces_str = str(max_traces) if max_traces is not None else "all"
    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"  {name}  |  {len(pairs)} positions  |  {thickness*1e3:.3f} mm  |  {traces_str} traces/folder")
    tqdm.write(f"{'='*60}")

    _ds_kwargs = dict(
        correction_factor=correction_factor,
        align=align_enabled,
        align_method=align_method,
        corr_window_ps=corr_window_ps,
        adaptive_fit_window=adaptive_fit_window,
        fit_window_sigma_factor=fit_window_sigma_factor,
        fallback_fit_window_ps=fallback_fit_window_ps,
        min_sigma_ps=min_sigma_ps,
        max_sigma_ps=max_sigma_ps,
        min_fit_points=min_fit_points,
        max_center_shift_ps=max_center_shift_ps,
        max_traces=max_traces,
    )

    pos_colors = cm.tab10(np.linspace(0, 1, len(pairs)))
    positions  = []
    n_not_converged = 0

    for (idx, air_path, point_path), pos_col in zip(pairs, pos_colors):
        pos_name = f"{point_prefix}{idx}"
        ref = thz_tds.THZDataset(path=air_path,   **_ds_kwargs)
        sam = thz_tds.THZDataset(path=point_path, **_ds_kwargs)

        f_arr, n_cf, _, _, a_cf, diag = compute_refractive_index(
            ref_ds=ref, sam_ds=sam,
            thickness_m=thickness,
            f_low_THz=f_low, f_high_THz=f_high,
            max_iter=cf_max_iter,
            tol_n=cf_tol_n,
            tol_alpha=cf_tol_alpha,
            relaxation=cf_relaxation,
            clip_negative_alpha=cf_clip_negative_alpha,
            return_diagnostics=True,
        )

        band = (f_arr >= f_low) & (f_arr <= f_high)

        if not diag["converged"]:
            n_not_converged += 1

        n_mean_cf = float(n_cf[band].mean())
        a_mean_cf = float(a_cf[band].mean())
        tqdm.write(
            f"  {pos_name}: converged={diag['converged']}, "
            f"iterations={diag['iterations']}, "
            f"n_mean={n_mean_cf:.4f}, alpha_mean={a_mean_cf:.4f}"
        )

        if skip_fp:
            n_fp, alpha_fp = None, None
        else:
            fp = compute_refractive_index_fp(
                ref_ds=ref, sam_ds=sam,
                thickness_m=thickness,
                f_low_THz=f_low, f_high_THz=f_high,
            )
            # fp.f_THz shares the same frequency grid/derivation as f_arr
            # (same underlying FFT), so the shared `band` mask applies.
            n_fp     = fp.n_fp
            alpha_fp = fp.alpha_cm_fp
            tqdm.write(
                f"    n with FP={n_fp[band].mean():.4f}  "
                f"(FP fit seeded from its own internal real-Fresnel guess, "
                f"not from the complex-Fresnel result above)"
            )

        positions.append({
            "name":                 pos_name,
            "color":                pos_col,
            "f_THz":                f_arr,
            "n_complex_fresnel":    n_cf,
            "alpha_complex_fresnel": a_cf,
            "n_with_fp":            n_fp,
            "alpha_with_fp":        alpha_fp,
            "converged":            diag["converged"],
            "iterations":           diag["iterations"],
            "band":                 band,
        })

    if n_not_converged:
        tqdm.write(
            f"\n  WARNING: {n_not_converged}/{len(positions)} position(s) in "
            f"'{name}' did not converge within max_iter={cf_max_iter} "
            f"(see RuntimeWarning(s) above)."
        )

    # ── band statistics ───────────────────────────────────────────────────
    f_band              = positions[0]["f_THz"][positions[0]["band"]]
    n_cf_all            = np.array([p["n_complex_fresnel"][p["band"]]     for p in positions])
    alpha_cf_all        = np.array([p["alpha_complex_fresnel"][p["band"]] for p in positions])

    avg_n_cf     = n_cf_all.mean(axis=0);     std_n_cf     = n_cf_all.std(axis=0)
    avg_alpha_cf = alpha_cf_all.mean(axis=0); std_alpha_cf = alpha_cf_all.std(axis=0)

    per_n_cf = n_cf_all.mean(axis=1)

    if not skip_fp:
        n_fp_all     = np.array([p["n_with_fp"][p["band"]]     for p in positions])
        alpha_fp_all = np.array([p["alpha_with_fp"][p["band"]] for p in positions])
        avg_n_fp     = n_fp_all.mean(axis=0);     std_n_fp     = n_fp_all.std(axis=0)
        avg_alpha_fp = alpha_fp_all.mean(axis=0); std_alpha_fp = alpha_fp_all.std(axis=0)
        per_n_fp     = n_fp_all.mean(axis=1)

    # ── per-position summary table ────────────────────────────────────────
    if skip_fp:
        tqdm.write(f"\n  {'Pos':<10}  {'n_complex_fresnel':>18}  {'converged':>10}  {'iters':>6}")
        tqdm.write(f"  {'-'*50}")
        for i, p in enumerate(positions):
            tqdm.write(
                f"  {p['name']:<10}  {per_n_cf[i]:>18.4f}  "
                f"{str(p['converged']):>10}  {p['iterations']:>6}"
            )
        total_mae_n = 0.0
        total_mae_a = 0.0
    else:
        per_mae_n = np.abs(n_fp_all - n_cf_all).mean(axis=1)
        per_mae_a = np.abs(alpha_fp_all - alpha_cf_all).mean(axis=1)
        tqdm.write(
            f"\n  {'Pos':<10}  {'n_complex_fresnel':>18}  {'n_with_fp':>10}  "
            f"{'Δn':>8}  {'|Δn|(f)':>9}  {'|Δα|(f)':>10}  {'converged':>10}  {'iters':>6}"
        )
        tqdm.write(f"  {'-'*90}")
        for i, p in enumerate(positions):
            dn = per_n_fp[i] - per_n_cf[i]
            tqdm.write(
                f"  {p['name']:<10}  {per_n_cf[i]:>18.4f}  {per_n_fp[i]:>10.4f}"
                f"  {dn:>+8.4f}  {per_mae_n[i]:>9.5f}  {per_mae_a[i]:>10.4f}"
                f"  {str(p['converged']):>10}  {p['iterations']:>6}"
            )
        total_mae_n = np.abs(n_fp_all - n_cf_all).mean()
        total_mae_a = np.abs(alpha_fp_all - alpha_cf_all).mean()
        tqdm.write(
            f"  {'TOTAL MAE':<10}  {'':>18}  {'':>10}  {'':>8}  "
            f"{total_mae_n:>9.5f}  {total_mae_a:>10.4f}  {'':>10}  {'':>6}"
        )

    # ── save TSV ──────────────────────────────────────────────────────────
    header = ["frequency_THz"]
    cols   = [f_band]
    for p in positions:
        pn = p["name"]
        if skip_fp:
            header += [f"{pn}_n_complex_fresnel", f"{pn}_alpha_complex_fresnel"]
            cols   += [p["n_complex_fresnel"][p["band"]], p["alpha_complex_fresnel"][p["band"]]]
        else:
            header += [
                f"{pn}_n_complex_fresnel", f"{pn}_n_with_fp",
                f"{pn}_alpha_complex_fresnel", f"{pn}_alpha_with_fp",
            ]
            cols   += [
                p["n_complex_fresnel"][p["band"]], p["n_with_fp"][p["band"]],
                p["alpha_complex_fresnel"][p["band"]], p["alpha_with_fp"][p["band"]],
            ]
    if skip_fp:
        header += ["avg_n_complex_fresnel", "avg_alpha_complex_fresnel"]
        cols   += [avg_n_cf, avg_alpha_cf]
    else:
        header += [
            "avg_n_complex_fresnel", "avg_n_with_fp",
            "avg_alpha_complex_fresnel", "avg_alpha_with_fp",
        ]
        cols   += [avg_n_cf, avg_n_fp, avg_alpha_cf, avg_alpha_fp]

    tsv_tag = "complex_fresnel_results" if skip_fp else "complex_fresnel_fp_results"
    tsv = output_root / f"{name}_{tsv_tag}_{timestamp}.tsv"
    np.savetxt(tsv, np.column_stack(cols),
               delimiter="\t", header="\t".join(header), comments="")
    tqdm.write(f"  Saved: {tsv}")

    result = {
        "name":            name,
        "thickness_mm":    thickness * 1e3,
        "n_positions":     len(pairs),
        "positions":       positions,
        "f_band":          f_band,
        "avg_n_cf":        avg_n_cf,     "std_n_cf":     std_n_cf,
        "avg_alpha_cf":    avg_alpha_cf, "std_alpha_cf": std_alpha_cf,
        "n_cf_mean":       per_n_cf.mean(),
        "n_cf_std":        per_n_cf.std(),
        "n_not_converged": n_not_converged,
        "total_mae_n":     total_mae_n,
        "total_mae_a":     total_mae_a,
    }
    if not skip_fp:
        result.update({
            "avg_n_fp":     avg_n_fp,     "std_n_fp":     std_n_fp,
            "avg_alpha_fp": avg_alpha_fp, "std_alpha_fp": std_alpha_fp,
            "n_fp_mean":    per_n_fp.mean(),
            "n_fp_std":     per_n_fp.std(),
        })
    return result


# ── main ─────────────────────────────────────────────────────────────────────

def main(config_path: str | Path = CONFIG_DEFAULT) -> None:
    cfg = load_yaml(config_path)

    output_root = Path(cfg["output"])
    ri_cfg      = cfg["refractive_index"]
    skip_fp     = ri_cfg.get("skip_fp", False)

    shared   = {k: cfg[k] for k in ("loading", "alignment", "refractive_index", "plot")}
    shared["ylim"] = cfg.get("ylim", {})
    plot_cfg = shared["plot"]
    fig_size       = tuple(plot_cfg["figure_size"])
    dpi            = plot_cfg["dpi"]
    formats        = plot_cfg["formats"]
    errorbar_step   = plot_cfg.get("errorbar_step",     15)
    font_axis_label = plot_cfg.get("font_axis_label",   10)
    font_tick       = plot_cfg.get("font_tick",          9)
    font_title      = plot_cfg.get("font_title",        11)
    font_legend     = plot_cfg.get("font_legend",        9)
    line_width      = plot_cfg.get("lw_curve",          1.8)
    trace_width     = plot_cfg.get("lw_trace",          0.8)
    errorbar_width  = plot_cfg.get("lw_errorbar",       1.0)
    label_fontsize  = plot_cfg.get("font_line_label",    8)
    capsize              = plot_cfg.get("capsize",              3)
    xlim                 = plot_cfg.get("xlim",                 None)
    xticks               = plot_cfg.get("xticks",               None)
    use_inline_labels    = plot_cfg.get("use_inline_labels",    True)
    label_x              = plot_cfg.get("label_x",              None)
    use_legend           = plot_cfg.get("use_legend",           False)
    legend_loc           = plot_cfg.get("legend_loc",           "best")
    legend_loc_alpha     = plot_cfg.get("legend_loc_alpha",     legend_loc)
    legend_ncol          = plot_cfg.get("legend_ncol",          1)
    legend_handlelength  = plot_cfg.get("legend_handlelength",  1.5)
    legend_columnspacing = plot_cfg.get("legend_columnspacing", 0.8)
    legend_labelspacing  = plot_cfg.get("legend_labelspacing",  0.3)

    plt.rcParams.update({
        "svg.fonttype":       "path",
        "axes.labelsize":     font_axis_label,
        "axes.titlesize":     font_title,
        "xtick.labelsize":    font_tick,
        "ytick.labelsize":    font_tick,
        "legend.fontsize":    font_legend,
        "axes.linewidth":     plot_cfg.get("lw_spine",    0.8),
        "grid.linewidth":     plot_cfg.get("lw_grid",     0.4),
        "xtick.major.size":   plot_cfg.get("tick_length", 3.0),
        "ytick.major.size":   plot_cfg.get("tick_length", 3.0),
        "xtick.major.width":  plot_cfg.get("tick_width",  0.8),
        "ytick.major.width":  plot_cfg.get("tick_width",  0.8),
    })

    _ylim_cfg  = shared.get("ylim", {}) or {}
    ylim_n     = _ylim_cfg.get("n")      # e.g. [1.0, 2.0] or None
    ylim_alpha = _ylim_cfg.get("alpha")  # None = auto

    timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_run_dir = output_root / timestamp
    output_run_dir.mkdir(parents=True, exist_ok=True)

    cf_cfg = (ri_cfg.get("complex_fresnel") or {})
    print(f"Config      : {config_path}")
    print(f"Samples     : {len(cfg['samples'])}")
    print(f"Output dir  : {output_run_dir}")
    print(f"skip_fp     : {skip_fp}")
    print(
        f"complex_fresnel settings : max_iter={cf_cfg.get('max_iter', 100)}, "
        f"tol_n={cf_cfg.get('tol_n', 1.0e-8)}, "
        f"tol_alpha={cf_cfg.get('tol_alpha', 1.0e-6)}, "
        f"relaxation={cf_cfg.get('relaxation', 1.0)}, "
        f"clip_negative_alpha={cf_cfg.get('clip_negative_alpha', False)}\n"
    )

    n_samples     = len(cfg["samples"])
    sample_colors = cm.tab10(np.linspace(0, 1, max(n_samples, 1)))
    cross_sample  = []

    for entry, s_col in zip(
        tqdm(cfg["samples"], desc="Samples", unit="sample"),
        sample_colors,
    ):
        summary = run_sample(entry, shared, output_run_dir, timestamp=timestamp)
        if summary:
            summary["color"] = s_col
            cross_sample.append(summary)

    if not cross_sample:
        print("No samples processed.")
        return

    total_not_converged = sum(s["n_not_converged"] for s in cross_sample)
    total_positions = sum(s["n_positions"] for s in cross_sample)

    # ── Cross-sample summary table ────────────────────────────────────────
    if skip_fp:
        W = 60
        print(f"\n{'=' * W}")
        print("CROSS-SAMPLE SUMMARY (complex-Fresnel, no FP)")
        print(f"{'=' * W}")
        print(f"{'Sample':<18}  {'mm':>5}  {'pts':>4}  {'n_complex_fresnel':>18}  {'±':>6}")
        print("-" * W)
        for s in cross_sample:
            print(
                f"{s['name']:<18}  {s['thickness_mm']:>5.2f}  {s['n_positions']:>4}  "
                f"{s['n_cf_mean']:>18.4f}  {s['n_cf_std']:>6.4f}"
            )
    else:
        W = 100
        print(f"\n{'=' * W}")
        print("CROSS-SAMPLE SUMMARY (complex-Fresnel vs. FP-corrected)")
        print(f"{'=' * W}")
        print(
            f"{'Sample':<18}  {'mm':>5}  {'pts':>4}  "
            f"{'n_complex_fresnel':>18}  {'±':>6}  {'n_with_fp':>10}  {'±':>6}  "
            f"{'|Δn| MAE':>10}  {'|Δα| MAE':>10}"
        )
        print("-" * W)
        for s in cross_sample:
            print(
                f"{s['name']:<18}  {s['thickness_mm']:>5.2f}  {s['n_positions']:>4}  "
                f"{s['n_cf_mean']:>18.4f}  {s['n_cf_std']:>6.4f}  "
                f"{s['n_fp_mean']:>10.4f}  {s['n_fp_std']:>6.4f}  "
                f"{s['total_mae_n']:>10.5f}  {s['total_mae_a']:>10.4f}"
            )
        all_mae_n = np.mean([s["total_mae_n"] for s in cross_sample])
        all_mae_a = np.mean([s["total_mae_a"] for s in cross_sample])
        print("-" * W)
        print(
            f"{'GRAND MEAN':<18}  {'':>5}  {'':>4}  "
            f"{'':>18}  {'':>6}  {'':>10}  {'':>6}  "
            f"{all_mae_n:>10.5f}  {all_mae_a:>10.4f}"
        )

    print(
        f"\nConvergence: {total_positions - total_not_converged}/{total_positions} "
        f"position(s) converged across all samples "
        f"({total_not_converged} failed to converge)."
    )

    # ── Combined figures ──────────────────────────────────────────────────
    single = len(cross_sample) == 1

    def _overlay(ax, qty_avg, qty_std, qty_pos_key, ylabel, title, errorbars=True, ylim=None, loc=None):
        for s in cross_sample:
            f   = s["f_band"]
            avg = s[qty_avg]
            std = s[qty_std]
            col = s["color"]
            lbl = _label(s['name'])
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
                    f[::errorbar_step],
                    avg[::errorbar_step],
                    yerr=std[::errorbar_step],
                    fmt="none",
                    ecolor=col,
                    elinewidth=errorbar_width,
                    capsize=capsize,
                    capthick=errorbar_width,
                    alpha=0.9,
                    label="_nolegend_",
                )
            if use_inline_labels:
                x_lbl = label_x if label_x is not None else f[-1]
                ax.text(
                    x_lbl, avg[-1], lbl,
                    color=col, fontsize=label_fontsize, va="center", ha="left",
                    clip_on=True,
                )
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
                ncol=legend_ncol,
                fontsize=font_legend,
                handlelength=legend_handlelength,
                columnspacing=legend_columnspacing,
                labelspacing=legend_labelspacing,
                frameon=True,
                borderpad=0.4,
                handletextpad=0.4,
            )

    # Always-present figures (complex-Fresnel n and alpha, with/without errorbars)
    fig_n,      ax_n      = plt.subplots(figsize=fig_size)
    fig_a,      ax_a      = plt.subplots(figsize=fig_size)
    fig_n_mean, ax_n_mean = plt.subplots(figsize=fig_size)
    fig_a_mean, ax_a_mean = plt.subplots(figsize=fig_size)

    _overlay(ax_n,      "avg_n_cf",     "std_n_cf",     "n_complex_fresnel",
             "Refractive index $n$",  "Refractive index — complex-Fresnel correction",
             ylim=ylim_n)
    _overlay(ax_a,      "avg_alpha_cf", "std_alpha_cf", "alpha_complex_fresnel",
             r"$\alpha$ (cm$^{-1}$)", "Absorption — complex-Fresnel correction",
             loc=legend_loc_alpha)
    _overlay(ax_n_mean, "avg_n_cf",     "std_n_cf",     "n_complex_fresnel",
             "Refractive index $n$",  "Mean refractive index — complex-Fresnel correction",
             errorbars=False, ylim=ylim_n)
    _overlay(ax_a_mean, "avg_alpha_cf", "std_alpha_cf", "alpha_complex_fresnel",
             r"$\alpha$ (cm$^{-1}$)", r"Mean absorption — complex-Fresnel correction",
             errorbars=False, loc=legend_loc_alpha)

    figures_to_save = [
        (fig_n,      "n_complex_fresnel"),
        (fig_a,      "alpha_complex_fresnel"),
        (fig_n_mean, "n_complex_fresnel_mean"),
        (fig_a_mean, "alpha_complex_fresnel_mean"),
    ]

    if not skip_fp:
        fig_n_fp,      ax_n_fp      = plt.subplots(figsize=fig_size)
        fig_a_fp,      ax_a_fp      = plt.subplots(figsize=fig_size)
        fig_n_fp_mean, ax_n_fp_mean = plt.subplots(figsize=fig_size)
        fig_a_fp_mean, ax_a_fp_mean = plt.subplots(figsize=fig_size)

        _overlay(ax_n_fp,      "avg_n_fp",     "std_n_fp",     "n_with_fp",
                 "Refractive index $n$",  "Refractive index — with Fabry-Pérot correction",
                 ylim=ylim_n)
        _overlay(ax_a_fp,      "avg_alpha_fp", "std_alpha_fp", "alpha_with_fp",
                 r"$\alpha$ (cm$^{-1}$)", "Absorption — with Fabry-Pérot correction",
                 loc=legend_loc_alpha)
        _overlay(ax_n_fp_mean, "avg_n_fp",     "std_n_fp",     "n_with_fp",
                 "Refractive index $n$",  "Mean refractive index — with Fabry-Pérot correction",
                 errorbars=False, ylim=ylim_n)
        _overlay(ax_a_fp_mean, "avg_alpha_fp", "std_alpha_fp", "alpha_with_fp",
                 r"$\alpha$ (cm$^{-1}$)", r"Mean absorption — with Fabry-Pérot correction",
                 errorbars=False, loc=legend_loc_alpha)

        figures_to_save += [
            (fig_n_fp,      "n_with_fp"),
            (fig_a_fp,      "alpha_with_fp"),
            (fig_n_fp_mean, "n_with_fp_mean"),
            (fig_a_fp_mean, "alpha_with_fp_mean"),
        ]

    for fig_obj, _ in figures_to_save:
        fig_obj.tight_layout(pad=0.3)

    plt.show()

    for fig_obj, tag in figures_to_save:
        stem = f"batch_{tag}"
        for fmt in formats:
            path = output_run_dir / f"{stem}.{fmt}"
            # svg/pdf: no bbox cropping so the file keeps the exact figure size
            bbox = None if fmt in ("svg", "pdf") else "tight"
            fig_obj.savefig(path, dpi=dpi, bbox_inches=bbox)
        print(f"  Saved: {output_run_dir / stem}.{formats[0]}")

    print(f"\nAll results saved in: {output_run_dir}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else CONFIG_DEFAULT
    main(path)
