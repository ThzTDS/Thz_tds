"""
Test alignment methods — shift statistics comparison (full control check).

For every sample in the batch YAML, loads every ``<prefix>N`` subfolder
(e.g. air1..air10, reference1..reference10 — prefixes from
``alignment_check.folder_prefixes``) with each alignment method, prints the
jitter shifts and their statistics, and saves the comparison figures.

Figures are never shown; they are saved to
``<alignment_check.output>/<sample>_<YYYY-MM-DD_HH-MM-SS>/<folder>/`` together with a
``summary.csv`` and summary figures at ``<alignment_check.output>/<sample>_<YYYY-MM-DD_HH-MM-SS>/``.

Note: valid align_method values are:
    "minimum", "correlation", "correlation_subsample", "gaussian"

Usage
-----
    python examples/diagnostics/test_alignment_shifts.py
    python examples/diagnostics/test_alignment_shifts.py config/my_experiment.yaml
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")   # plots off — save only
import matplotlib.pyplot as plt
import numpy as np
import yaml

import thz_tds
from thz_tds.dataset import THZDataset


# ── Config ────────────────────────────────────────────────────────────────────
_default = Path(__file__).resolve().parent.parent.parent / "config" / "all_samples_fp_batch.yaml"
config_path = sys.argv[1] if len(sys.argv) >= 2 else str(_default)

with open(config_path) as _f:
    _raw = yaml.safe_load(_f)
_check          = _raw.get("alignment_check", {}) or {}
out_root        = Path(_check.get("output", "/mnt/Data/results"))
# {prefix: [numbers] or None (= all)}
folder_select   = _check.get("folders") or {"air": None, "point": None}

try:
    # Strict format (default_config.yaml — has paths.reference, plot.n_traces_plot)
    cfg = thz_tds.load_config(config_path)
    correction_factor       = cfg.loading.correction_factor
    corr_window_ps          = cfg.alignment.corr_window_ps
    adaptive_fit_window     = cfg.alignment.adaptive_fit_window
    fit_window_sigma_factor = cfg.alignment.fit_window_sigma_factor
    fallback_fit_window_ps  = cfg.alignment.fallback_fit_window_ps
    min_sigma_ps            = cfg.alignment.min_sigma_ps
    max_sigma_ps            = cfg.alignment.max_sigma_ps
    min_fit_points          = cfg.alignment.min_fit_points
    max_center_shift_ps     = cfg.alignment.max_center_shift_ps
    n_traces_plot           = cfg.plot.n_traces_plot
    _ref = Path(cfg.paths.reference)
    jobs = [(_ref.parent.name, _ref)]
except (thz_tds.ConfigError, AttributeError):
    # Batch format (all_samples_fp_batch.yaml — has samples[].dir, no paths.reference)
    _loading = _raw.get("loading", {})
    _align   = _raw.get("alignment", {})
    _plot    = _raw.get("plot", {})
    correction_factor       = _loading.get("correction_factor", 1.0)
    corr_window_ps          = _align.get("corr_window_ps", None)
    adaptive_fit_window     = _align.get("adaptive_fit_window", True)
    fit_window_sigma_factor = _align.get("fit_window_sigma_factor", 5.0)
    fallback_fit_window_ps  = _align.get("fallback_fit_window_ps", 15.0)
    min_sigma_ps            = _align.get("min_sigma_ps", 0.05)
    max_sigma_ps            = _align.get("max_sigma_ps", 10.0)
    min_fit_points          = _align.get("min_fit_points", 10)
    max_center_shift_ps     = _align.get("max_center_shift_ps", 5.0)
    n_traces_plot           = _plot.get("n_traces_plot")

    # Discover every <prefix>N subfolder of every sample
    _samples = _raw.get("samples") or []
    if not _samples:
        raise RuntimeError("No samples found in batch YAML — cannot determine dataset paths.")
    jobs = []
    for _s in _samples:
        _sample_dir = Path(_s["dir"])
        _name       = _s.get("name", _sample_dir.name)
        for _prefix, _wanted in folder_select.items():
            _re    = re.compile(rf"^{re.escape(_prefix)}(\d+)$", re.IGNORECASE)
            _found = {int(_re.match(p.name).group(1)): p
                      for p in _sample_dir.iterdir() if p.is_dir() and _re.match(p.name)}
            _nums  = sorted(_found) if _wanted is None else [int(n) for n in _wanted]
            for _n in _nums:
                if _n in _found:
                    jobs.append((_name, _found[_n]))
                else:
                    print(f"WARNING: {_sample_dir / f'{_prefix}{_n}'} not found — skipped")

if not jobs:
    raise RuntimeError(f"No folders matching {folder_select} found.")

METHODS = [
    #"gaussian_minimum_mean",
    "gaussian_median_integer",
]

CORRECTION_FACTORS = [correction_factor]

ALPHA = 0.5

DS_KWARGS = dict(
    corr_window_ps=corr_window_ps,
    adaptive_fit_window=adaptive_fit_window,
    fit_window_sigma_factor=fit_window_sigma_factor,
    fallback_fit_window_ps=fallback_fit_window_ps,
    min_sigma_ps=min_sigma_ps,
    max_sigma_ps=max_sigma_ps,
    min_fit_points=min_fit_points,
    max_center_shift_ps=max_center_shift_ps,
)

print(f"Config                 : {config_path}")
print(f"Output                 : {out_root}")
print(f"Folders ({len(jobs):>2})           : {', '.join(f'{s}/{p.name}' for s, p in jobs)}")
print(f"correction_factor      : {correction_factor}")
print(f"corr_window_ps         : {corr_window_ps}")
print(f"adaptive_fit_window    : {adaptive_fit_window}")
print(f"fit_window_sigma_factor: {fit_window_sigma_factor}")
print(f"fallback_fit_window_ps : {fallback_fit_window_ps}")
print()


def save_fig(fig, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / f"{name}.png", dpi=150)
    plt.close(fig)


def fit_gaussian(ds, method, t, y):
    if "minimum" in method:
        return ds.fit_gaussian_to_minimum(t, y)
    return ds.fit_gaussian_to_trace(t, y)


def analyse_folder(path: Path, out_dir: Path) -> list[dict]:
    """Run every method on one folder, save its figures, return summary rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    label = f"{path.parent.name}/{path.name}"
    print(f"══ {label} " + "═" * max(0, 60 - len(label)))

    # ── Load each method and collect stats ───────────────────────────────────
    results: list[dict] = []
    for method in METHODS:
        print(f"Loading  align_method='{method}' ...")
        try:
            ds = THZDataset(
                path=str(path),
                correction_factor=correction_factor,
                align=True,
                align_method=method,
                **DS_KWARGS,
            )
        except Exception as exc:
            print(f"  FAILED: {exc}\n")
            continue

        shifts = ds.jitter_shifts_ps
        if shifts is None:
            shifts = np.zeros(ds.all_y.shape[0])

        print(f"  mean   : {shifts.mean():.5f} ps")
        print(f"  std    : {shifts.std():.5f} ps")
        print(f"  min    : {shifts.min():.5f} ps")
        print(f"  max    : {shifts.max():.5f} ps")
        print()
        results.append(dict(method=method, ds=ds, shifts=shifts))

    if not results:
        print("No results for this folder.\n")
        return []

    cmap   = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(results))]

    # ── Load raw (unaligned) traces ──────────────────────────────────────────
    ds_raw = THZDataset(
        path=str(path),
        correction_factor=correction_factor,
        align=False,
        **{k: v for k, v in DS_KWARGS.items() if k != "corr_window_ps"},
    )
    t          = ds_raw.x_ps
    raw_traces = ds_raw.all_y_raw if ds_raw.all_y_raw is not None else ds_raw.all_y
    n_traces   = raw_traces.shape[0]

    step     = 1 if (n_traces_plot is None or n_traces_plot >= n_traces) else max(1, n_traces // n_traces_plot)
    plot_idx = list(range(0, n_traces, step))

    rows: list[dict] = []

    # ── Per-method: raw vs aligned + average with p2p annotation ─────────────
    for r, col in zip(results, colors):
        ds  = r["ds"]
        avg = ds.y_avg
        p2p = float(avg.max() - avg.min())
        rms = float(np.sqrt(np.mean((ds.all_y - avg[None, :]) ** 2)))
        s   = r["shifts"]
        rows.append(dict(
            folder=path.name, method=r["method"], n_traces=n_traces,
            shift_mean_ps=float(s.mean()), shift_std_ps=float(s.std()),
            shift_min_ps=float(s.min()), shift_max_ps=float(s.max()),
            p2p_nA=p2p, rms_nA=rms,
        ))

        fig, ax = plt.subplots(figsize=(12, 5))
        for i in plot_idx:
            ax.plot(t, raw_traces[i], color="tab:gray", lw=0.7, alpha=ALPHA,
                    label="raw" if i == plot_idx[0] else None)
        for i in plot_idx:
            ax.plot(ds.x_ps, ds.all_y[i], color=col, lw=0.7, alpha=ALPHA,
                    label="aligned" if i == plot_idx[0] else None)
        ax.plot(ds.x_ps, avg, color="black", lw=2.2, label=f"average  p2p = {p2p:.4f} nA")

        # peak-to-peak arrow on the average
        t_max = ds.x_ps[np.argmax(avg)]
        ax.annotate("", xy=(t_max, avg.max()), xytext=(t_max, avg.min()),
                    arrowprops=dict(arrowstyle="<->", color="green", lw=1.8))
        ax.text(t_max + 0.3, (avg.max() + avg.min()) / 2,
                f"p2p = {p2p:.4f}", color="green", fontsize=9, va="center")

        ax.set_xlabel("Time [ps]")
        ax.set_ylabel("Amplitude [nA]")
        ax.set_title(
            f"Raw vs {r['method']}   std = {s.std():.5f} ps   RMS = {rms:.4e}"
            f"\n{len(plot_idx)} of {n_traces} traces (every {step}th)  —  {label}"
        )
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_fig(fig, out_dir, f"raw_vs_{r['method']}")

    # ── All averages overlaid with p2p in legend ─────────────────────────────
    fig_avg, ax_avg = plt.subplots(figsize=(11, 5))
    raw_avg = np.mean(raw_traces, axis=0)
    raw_p2p = float(raw_avg.max() - raw_avg.min())
    ax_avg.plot(t, raw_avg, color="black", lw=2, ls="--",
                label=f"raw average  p2p = {raw_p2p:.4f} nA")
    for r, col in zip(results, colors):
        avg = r["ds"].y_avg
        p2p = float(avg.max() - avg.min())
        ax_avg.plot(r["ds"].x_ps, avg, color=col, lw=1.8,
                    label=f"{r['method']}  p2p = {p2p:.4f} nA")
    ax_avg.set_xlabel("Time [ps]")
    ax_avg.set_ylabel("Amplitude [nA]")
    ax_avg.set_title(f"All averages — {label}")
    ax_avg.legend(fontsize=9, loc="upper right")
    ax_avg.grid(True, alpha=0.3)
    fig_avg.tight_layout()
    save_fig(fig_avg, out_dir, "averages")

    # ── Gaussian fit visualisation ───────────────────────────────────────────
    gauss_results = [r for r in results if "gaussian" in r["method"]]
    if gauss_results:
        n_g = len(gauss_results)
        fig_gf, axes_gf = plt.subplots(n_g, 2, figsize=(14, 4 * n_g), squeeze=False)

        y_demo = raw_traces[0]   # first raw trace used as demo

        for row, r in enumerate(gauss_results):
            ds     = r["ds"]
            method = r["method"]
            fit    = fit_gaussian(ds, method, t, y_demo)

            t0    = fit["t0_ps"]
            sigma = fit["sigma_ps"]
            A     = fit["A"]
            C     = fit["C"]

            # Read the actual fit window from the diagnostics returned by the fit method
            win_left  = fit.get("fit_window_left_ps",  t0 - 0.5 * ds.fit_window_ps)
            win_right = fit.get("fit_window_right_ps", t0 + 0.5 * ds.fit_window_ps)
            half_w    = fit.get("adaptive_half_width_ps", 0.5 * ds.fit_window_ps)
            used_adap = fit.get("used_adaptive_window", False)
            n_fit_pts = fit.get("fit_window_n_points", 0)
            sigma_ini = fit.get("sigma_ini_ps", float("nan"))
            win_mask  = (t >= win_left) & (t <= win_right)

            adap_str  = "adaptive" if used_adap else "fallback"
            win_label = f"Fit window ±{half_w:.2f} ps ({adap_str}, {n_fit_pts} pts)"

            # Full Gaussian curve on the full time axis
            y_gauss = ds.gaussian_with_offset(t, A, t0, sigma, C)

            # ── Left: full trace + window shading + Gaussian ─────────────────
            ax_l = axes_gf[row, 0]
            ax_l.plot(t, y_demo, color="tab:gray", lw=1, label="Raw trace (scan 0)")
            ylo, yhi = y_demo.min(), y_demo.max()
            ax_l.fill_between(t, ylo, yhi, where=win_mask,
                              color="gold", alpha=0.35, label=win_label)
            ax_l.plot(t, y_gauss, color="tab:red", lw=2, ls="--", label="Gaussian fit")
            ax_l.axvline(t0, color="tab:red", lw=1, ls=":", alpha=0.7)
            ax_l.set_xlabel("Time [ps]")
            ax_l.set_ylabel("Amplitude [nA]")
            ax_l.set_title(f"{method} — full trace")
            ax_l.legend(fontsize=8, loc="upper right")
            ax_l.grid(True, alpha=0.3)

            # ── Right: zoom to fit window — picked data vs Gaussian ──────────
            ax_r = axes_gf[row, 1]
            ax_r.plot(t[win_mask], y_demo[win_mask], color="tab:gray", lw=1.5,
                      marker="o", ms=3, label="Picked data (fit window)")
            ax_r.plot(t[win_mask], y_gauss[win_mask], color="tab:red", lw=2, ls="--",
                      label=f"Gaussian fit  t₀={t0:.3f} ps  σ={sigma:.3f} ps")
            ax_r.axvline(t0, color="tab:red", lw=1, ls=":", alpha=0.7,
                         label=f"t₀ = {t0:.3f} ps")
            ax_r.set_xlabel("Time [ps]")
            ax_r.set_ylabel("Amplitude [nA]")
            ax_r.set_title(
                f"{method} — fit window ±{half_w:.2f} ps ({adap_str})"
                + (f"  σ_ini={sigma_ini:.3f} ps" if not np.isnan(sigma_ini) else "")
            )
            ax_r.legend(fontsize=8, loc="upper right")
            ax_r.grid(True, alpha=0.3)

        fig_gf.suptitle(f"Gaussian fit visualisation — {label}", fontsize=11)
        fig_gf.tight_layout()
        save_fig(fig_gf, out_dir, "gaussian_fit")

        # ── All Gaussian fits overlaid in one plot ───────────────────────────
        gcolors = [cmap(i) for i in range(n_g)]

        fig_go, ax_go = plt.subplots(figsize=(12, 5))
        ax_go.plot(t, y_demo, color="tab:gray", lw=1.2, alpha=0.8, label="Raw trace (scan 0)")

        for r, gcol in zip(gauss_results, gcolors):
            ds     = r["ds"]
            method = r["method"]
            fit    = fit_gaussian(ds, method, t, y_demo)

            t0        = fit["t0_ps"]
            sigma     = fit["sigma_ps"]
            win_left  = fit.get("fit_window_left_ps",  t0 - 0.5 * ds.fit_window_ps)
            win_right = fit.get("fit_window_right_ps", t0 + 0.5 * ds.fit_window_ps)
            half_w    = fit.get("adaptive_half_width_ps", 0.5 * ds.fit_window_ps)
            used_adap = fit.get("used_adaptive_window", False)
            win_mask  = (t >= win_left) & (t <= win_right)
            adap_str  = "adaptive" if used_adap else "fallback"

            y_gauss = ds.gaussian_with_offset(t, fit["A"], t0, sigma, fit["C"])

            ax_go.fill_between(t, y_demo.min(), y_demo.max(), where=win_mask,
                               color=gcol, alpha=0.12)
            ax_go.plot(t, y_gauss, color=gcol, lw=2, ls="--",
                       label=f"{method}  t₀={t0:.3f} ps  σ={sigma:.3f} ps  "
                             f"win=±{half_w:.2f} ps ({adap_str})")
            ax_go.axvline(t0, color=gcol, lw=1, ls=":", alpha=0.7)

        ax_go.set_xlabel("Time [ps]")
        ax_go.set_ylabel("Amplitude [nA]")
        ax_go.set_title(f"All Gaussian fits overlaid — {label}")
        ax_go.legend(fontsize=9, loc="upper right")
        ax_go.grid(True, alpha=0.3)
        fig_go.tight_layout()
        save_fig(fig_go, out_dir, "gaussian_fits_overlay")

        # ── Gaussian fit on every trace: fitted curves + t₀ series ───────────
        trace_colors = plt.get_cmap("cool")(np.linspace(0, 1, len(plot_idx)))
        plot_set     = set(plot_idx)

        for r, gcol in zip(gauss_results, gcolors):
            ds     = r["ds"]
            method = r["method"]

            # Fit Gaussian on ALL raw traces to collect every t₀
            all_t0    = np.full(n_traces, np.nan)
            plot_fits = {}   # index → fit dict, only for plot_idx

            print(f"  Fitting {n_traces} traces for {method} ...")
            for i in range(n_traces):
                try:
                    fi = fit_gaussian(ds, method, t, raw_traces[i])
                    all_t0[i] = fi["t0_ps"]
                    if i in plot_set:
                        plot_fits[i] = fi
                except Exception:
                    pass

            if method == "gaussian_minimum_mean":
                t0_ref_plot = float(np.nanmean(all_t0))
                ref_name    = "Mean"
            else:  # gaussian_median_integer
                t0_ref_plot = float(np.nanmedian(all_t0))
                ref_name    = "Median"
            t0_std = float(np.nanstd(all_t0))
            for row_d in rows:
                if row_d["method"] == method:
                    row_d["t0_ref_ps"] = t0_ref_plot
                    row_d["t0_std_ps"] = t0_std

            fig_tr, (ax_tr, ax_t0) = plt.subplots(
                2, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [3, 1]},
            )

            # Top: raw traces (grey) + fitted Gaussians (colour gradient) + t₀ marks
            for j, i in enumerate(plot_idx):
                ax_tr.plot(t, raw_traces[i], color="tab:gray", lw=0.6, alpha=0.35,
                           label="Raw traces" if j == 0 else None)

            for j, i in enumerate(plot_idx):
                fi = plot_fits.get(i)
                if fi is None:
                    continue
                y_g = ds.gaussian_with_offset(t, fi["A"], fi["t0_ps"], fi["sigma_ps"], fi["C"])
                ax_tr.plot(t, y_g, color=trace_colors[j], lw=1.2, alpha=0.85,
                           label="Gaussian fits" if j == 0 else None)
                ax_tr.axvline(fi["t0_ps"], color=trace_colors[j], lw=0.7, ls=":", alpha=0.5)

            ax_tr.axvline(t0_ref_plot, color="red", lw=2, ls="--",
                          label=f"{ref_name} t₀ = {t0_ref_plot:.4f} ps")
            ax_tr.set_ylabel("Amplitude [nA]")
            ax_tr.set_title(
                f"{method} — Gaussian fit on every raw trace  "
                f"({len(plot_idx)} of {n_traces} shown)\n{label}"
            )
            ax_tr.legend(fontsize=8, loc="upper right")
            ax_tr.grid(True, alpha=0.3)

            # Bottom: fitted t₀ per trace index
            valid = ~np.isnan(all_t0)
            idx_v = np.where(valid)[0]
            ax_t0.plot(idx_v, all_t0[valid], "o", color=gcol, ms=3, alpha=0.8,
                       label=f"Fitted t₀  std = {t0_std:.5f} ps")
            ax_t0.axhline(t0_ref_plot, color="red", lw=1.5, ls="--",
                          label=f"{ref_name} = {t0_ref_plot:.4f} ps")
            ax_t0.set_xlabel("Trace index")
            ax_t0.set_ylabel("t₀ [ps]")
            ax_t0.set_title(f"Fitted pulse centre per trace — {ref_name} = {t0_ref_plot:.4f} ps   std = {t0_std:.5f} ps")
            ax_t0.legend(fontsize=8, loc="upper right")
            ax_t0.grid(True, alpha=0.3)

            fig_tr.tight_layout()
            save_fig(fig_tr, out_dir, f"gaussian_every_trace_{method}")

    # ── Correction factor comparison ─────────────────────────────────────────
    # Use the first method for this comparison
    cf_method  = METHODS[0]
    cf_colors  = ["tab:blue", "tab:red"]
    cf_results = []

    print(f"\nCorrection factor comparison  (method: {cf_method})")
    print(f"{'CF':>8}  {'peak-to-peak':>14}  {'max':>10}  {'min':>10}")
    print("-" * 48)

    for cf in CORRECTION_FACTORS:
        # Reuse the already-loaded dataset when the CF matches
        ds_cf = next((r["ds"] for r in results
                      if r["method"] == cf_method and cf == correction_factor), None)
        if ds_cf is None:
            ds_cf = THZDataset(
                path=str(path),
                correction_factor=cf,
                align=True,
                align_method=cf_method,
                **DS_KWARGS,
            )
        avg = ds_cf.y_avg
        p2p = float(avg.max() - avg.min())
        print(f"{cf:>8.4f}  {p2p:>14.6f}  {avg.max():>10.6f}  {avg.min():>10.6f}")
        cf_results.append(dict(cf=cf, ds=ds_cf, p2p=p2p))

    # One figure per correction factor: individual traces + average, peak-to-peak annotated
    for cr, col in zip(cf_results, cf_colors):
        ds  = cr["ds"]
        p2p = cr["p2p"]
        avg = ds.y_avg
        fig, ax = plt.subplots(figsize=(12, 5))

        for i in plot_idx:
            ax.plot(ds.x_ps, ds.all_y[i], color=col, lw=0.7, alpha=ALPHA,
                    label="traces" if i == plot_idx[0] else None)
        ax.plot(ds.x_ps, avg, color="black", lw=2.2, label="average")

        # peak-to-peak markers
        t_max = ds.x_ps[np.argmax(avg)]
        ax.annotate("", xy=(t_max, avg.max()), xytext=(t_max, avg.min()),
                    arrowprops=dict(arrowstyle="<->", color="green", lw=1.8))
        ax.text(t_max + 0.3, (avg.max() + avg.min()) / 2,
                f"p2p = {p2p:.4f}", color="green", fontsize=9, va="center")

        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Amplitude [nA]")
        ax.set_title(
            f"correction_factor = {cr['cf']}   peak-to-peak = {p2p:.6f}"
            f"\nmethod: {cf_method}   {len(plot_idx)} of {n_traces} traces  —  {label}"
        )
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_fig(fig, out_dir, f"cf_{cr['cf']}")

    # Overlay of all averages on one figure
    fig_cf, ax_cf = plt.subplots(figsize=(11, 5))
    for cr, col in zip(cf_results, cf_colors):
        ax_cf.plot(cr["ds"].x_ps, cr["ds"].y_avg, color=col, lw=2,
                   label=f"CF={cr['cf']}   p2p={cr['p2p']:.4f}")
    ax_cf.set_xlabel("Time (ps)")
    ax_cf.set_ylabel("Amplitude [nA]")
    ax_cf.set_title(f"Average trace: correction factor comparison  —  {label}")
    ax_cf.legend(fontsize=9, loc="upper right")
    ax_cf.grid(True, alpha=0.3)
    fig_cf.tight_layout()
    save_fig(fig_cf, out_dir, "cf_overlay")

    print(f"Saved figures to {out_dir}\n")
    return rows


# ── Run every folder ──────────────────────────────────────────────────────────
# One timestamp per run → new folder each run, nothing gets overwritten
RUN_STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
summary: dict[str, list[dict]] = {}
for sample_name, folder in jobs:
    try:
        rows = analyse_folder(folder, out_root / f"{sample_name}_{RUN_STAMP}" / folder.name)
    except Exception as exc:
        print(f"  FAILED on {folder}: {exc}\n")
        rows = []
    summary.setdefault(sample_name, []).extend(rows)

# ── Per-sample summary: CSV + shift std / p2p across all folders ─────────────
FIELDS = ["folder", "method", "n_traces", "shift_mean_ps", "shift_std_ps",
          "shift_min_ps", "shift_max_ps", "t0_ref_ps", "t0_std_ps", "p2p_nA", "rms_nA"]

for sample_name, rows in summary.items():
    if not rows:
        continue
    sample_dir = out_root / f"{sample_name}_{RUN_STAMP}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    with open(sample_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\n{sample_name}")
    print(f"{'Folder':<14}  {'Method':<25}  {'shift std (ps)':>14}  {'p2p (nA)':>10}  {'RMS (nA)':>10}")
    print("-" * 82)
    for r in rows:
        print(f"{r['folder']:<14}  {r['method']:<25}  {r['shift_std_ps']:>14.5f}"
              f"  {r['p2p_nA']:>10.4f}  {r['rms_nA']:>10.4e}")

    for method in METHODS:
        m_rows = [r for r in rows if r["method"] == method]
        if not m_rows:
            continue
        names = [r["folder"] for r in m_rows]
        x     = np.arange(len(names))
        fig, axes = plt.subplots(3, 1, figsize=(max(8, 0.6 * len(names)), 9), sharex=True)
        for ax, key, ylabel in zip(
            axes,
            ["shift_std_ps", "p2p_nA", "rms_nA"],
            ["shift std [ps]", "p2p [nA]", "RMS [nA]"],
        ):
            ax.plot(x, [r[key] for r in m_rows], "o-", color="tab:blue")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(names, rotation=45, ha="right")
        axes[0].set_title(f"Alignment control check — {sample_name} — {method}")
        fig.tight_layout()
        save_fig(fig, sample_dir, f"summary_{method}")

    print(f"Summary saved to {sample_dir}")
