"""
fp_all_samples_batch.py
=======================
Run FP-aware refractive index extraction for every sample listed in
all_samples_fp_batch.yaml.  For each sample the script:

  1. Auto-discovers air/point subfolder pairs.
  2. Computes n(f) and α(f) with FP correction AND without (no-FP).
  3. Saves per-sample figures and a TSV table.
  4. Prints a cross-sample summary table at the end.

Usage
-----
    python examples/fp_all_samples_batch.py
    python examples/fp_all_samples_batch.py config/all_samples_fp_batch.yaml
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
from thz_tds.refractive import compute_refractive_index
from thz_tds.spectral import fft_field
from thz_tds import viz


CONFIG_DEFAULT = Path(__file__).parent.parent / "config" / "all_samples_fp_batch.yaml"


# ── helpers ──────────────────────────────────────────────────────────────────

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
    Process one sample: load datasets, compute FP results, print per-position
    log and summary table, save TSV.  Returns all band data for combined
    plotting in main() — no figures are created here.
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

    for (idx, air_path, point_path), pos_col in zip(pairs, pos_colors):
        pos_name = f"{point_prefix}{idx}"
        ref = thz_tds.THZDataset(path=air_path,   **_ds_kwargs)
        sam = thz_tds.THZDataset(path=point_path, **_ds_kwargs)

        if skip_fp:
            f_arr, n_arr, _, _, a_arr = compute_refractive_index(
                ref_ds=ref, sam_ds=sam,
                thickness_m=thickness,
                f_low_THz=f_low, f_high_THz=f_high,
            )
            n_init, n_fp         = n_arr, n_arr
            alpha_init, alpha_fp = a_arr, a_arr
        else:
            fp                   = compute_refractive_index_fp(
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
                f"n with FP={n_fp[band].mean():.4f}"
            )

        positions.append({
            "name":       pos_name,
            "color":      pos_col,
            "f_THz":      f_arr,
            "n_fp":       n_fp,
            "n_init":     n_init,
            "alpha_fp":   alpha_fp,
            "alpha_init": alpha_init,
            "band":       band,
            "ref_t":      ref.x_ps,
            "ref_y":      ref.y_avg,
            "sam_t":      sam.x_ps,
            "sam_y":      sam.y_avg,
        })

    # ── per-sample waveform figures ──────────────────────────────────────
    plot_cfg  = shared["plot"]
    fig_size  = tuple(plot_cfg["figure_size"])
    dpi       = plot_cfg["dpi"]
    formats   = plot_cfg["formats"]

    fig_td,  ax_td  = plt.subplots(figsize=fig_size)
    fig_fft, ax_fft = plt.subplots(figsize=fig_size)

    for p in positions:
        col     = p["color"]
        p2p_ref = np.ptp(p["ref_y"])
        p2p_sam = np.ptp(p["sam_y"])
        ax_td.plot(p["ref_t"], p["ref_y"], "--", color=col, lw=0.9,
                   label=f"{p['name']} ref  (p2p={p2p_ref:.4f})")
        ax_td.plot(p["sam_t"], p["sam_y"], "-",  color=col, lw=1.2,
                   label=f"{p['name']} sam  (p2p={p2p_sam:.4f})")

        f_hz_r, E_r = fft_field(p["ref_t"], p["ref_y"], pad_factor=4)
        f_hz_s, E_s = fft_field(p["sam_t"], p["sam_y"], pad_factor=4)
        ax_fft.plot(f_hz_r / 1e12, np.abs(E_r), "--", color=col, lw=0.9,
                    label=f"{p['name']} ref")
        ax_fft.plot(f_hz_s / 1e12, np.abs(E_s), "-",  color=col, lw=1.2,
                    label=f"{p['name']} sam")

    ax_td.set_xlabel("Time (ps)")
    ax_td.set_ylabel("Amplitude (a.u.)")
    ax_td.set_title(f"{name} — time domain (aligned average)")
    ax_td.legend(fontsize=7)
    ax_td.grid(True, alpha=0.3)
    fig_td.tight_layout()

    ax_fft.set_xlabel("Frequency (THz)")
    ax_fft.set_ylabel("|E(f)| (a.u.)")
    ax_fft.set_title(f"{name} — frequency domain")
    ax_fft.set_xlim(0, f_high * 2)
    ax_fft.legend(fontsize=7)
    ax_fft.grid(True, alpha=0.3)
    fig_fft.tight_layout()

    for fig_obj, tag in [(fig_td, "timedomain"), (fig_fft, "fft")]:
        stem = output_root / f"{name}_{tag}_{timestamp}"
        viz.save_figure(fig_obj, stem, formats=formats, dpi=dpi)
        tqdm.write(f"  Saved: {stem}.{formats[0]}")
    plt.close(fig_td)
    plt.close(fig_fft)

    # ── band statistics ───────────────────────────────────────────────────
    f_band         = positions[0]["f_THz"][positions[0]["band"]]
    n_fp_all       = np.array([p["n_fp"][p["band"]]      for p in positions])
    n_init_all     = np.array([p["n_init"][p["band"]]    for p in positions])
    alpha_fp_all   = np.array([p["alpha_fp"][p["band"]]  for p in positions])
    alpha_init_all = np.array([p["alpha_init"][p["band"]] for p in positions])

    avg_n_init = n_init_all.mean(axis=0);   std_n_init = n_init_all.std(axis=0)
    avg_n_fp   = n_fp_all.mean(axis=0);     std_n_fp   = n_fp_all.std(axis=0)
    avg_a_init = alpha_init_all.mean(axis=0); std_a_init = alpha_init_all.std(axis=0)
    avg_a_fp   = alpha_fp_all.mean(axis=0);   std_a_fp   = alpha_fp_all.std(axis=0)

    # ── per-position summary table ────────────────────────────────────────
    per_n_init = n_init_all.mean(axis=1)
    per_n_fp   = n_fp_all.mean(axis=1)

    if skip_fp:
        tqdm.write(f"\n  {'Pos':<10}  {'n':>8}")
        tqdm.write(f"  {'-'*22}")
        for i, p in enumerate(positions):
            tqdm.write(f"  {p['name']:<10}  {per_n_init[i]:>8.4f}")
        total_mae_n = 0.0
        total_mae_a = 0.0
    else:
        per_mae_n = np.abs(n_fp_all - n_init_all).mean(axis=1)
        per_mae_a = np.abs(alpha_fp_all - alpha_init_all).mean(axis=1)
        tqdm.write(f"\n  {'Pos':<10}  {'n':>8}  {'n with FP':>10}  {'Δn':>8}  {'|Δn|(f)':>9}  {'|Δα|(f)':>10}")
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

    # ── save TSV ──────────────────────────────────────────────────────────
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
        "n_nofp_mean":  per_n_init.mean(),
        "n_nofp_std":   per_n_init.std(),
        "n_fp_mean":    per_n_fp.mean(),
        "n_fp_std":     per_n_fp.std(),
        "total_mae_n":  total_mae_n,
        "total_mae_a":  total_mae_a,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main(config_path: str | Path = CONFIG_DEFAULT) -> None:
    cfg = load_yaml(config_path)

    output_root = Path(cfg["output"])
    ri_cfg      = cfg["refractive_index"]
    skip_fp     = ri_cfg.get("skip_fp", False)

    shared   = {k: cfg[k] for k in ("loading", "alignment", "refractive_index", "plot")}
    plot_cfg = shared["plot"]
    fig_size      = tuple(plot_cfg["figure_size"])
    dpi           = plot_cfg["dpi"]
    formats       = plot_cfg["formats"]
    errorbar_step = plot_cfg.get("errorbar_step", 15)

    timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_run_dir = output_root / timestamp
    output_run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config      : {config_path}")
    print(f"Samples     : {len(cfg['samples'])}")
    print(f"Output dir  : {output_run_dir}")
    print(f"skip_fp     : {skip_fp}\n")

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

    # ── Cross-sample summary table ────────────────────────────────────────
    if skip_fp:
        W = 52
        print(f"\n{'=' * W}")
        print("CROSS-SAMPLE SUMMARY")
        print(f"{'=' * W}")
        print(f"{'Sample':<18}  {'mm':>5}  {'pts':>4}  {'n':>8}  {'±':>6}")
        print("-" * W)
        for s in cross_sample:
            print(
                f"{s['name']:<18}  {s['thickness_mm']:>5.2f}  {s['n_positions']:>4}  "
                f"{s['n_nofp_mean']:>8.4f}  {s['n_nofp_std']:>6.4f}"
            )
    else:
        W = 94
        print(f"\n{'=' * W}")
        print("CROSS-SAMPLE SUMMARY")
        print(f"{'=' * W}")
        print(
            f"{'Sample':<18}  {'mm':>5}  {'pts':>4}  "
            f"{'n':>8}  {'±':>6}  {'n with FP':>10}  {'±':>6}  "
            f"{'|Δn| MAE':>10}  {'|Δα| MAE':>10}"
        )
        print("-" * W)
        for s in cross_sample:
            print(
                f"{s['name']:<18}  {s['thickness_mm']:>5.2f}  {s['n_positions']:>4}  "
                f"{s['n_nofp_mean']:>8.4f}  {s['n_nofp_std']:>6.4f}  "
                f"{s['n_fp_mean']:>10.4f}  {s['n_fp_std']:>6.4f}  "
                f"{s['total_mae_n']:>10.5f}  {s['total_mae_a']:>10.4f}"
            )
        all_mae_n = np.mean([s["total_mae_n"] for s in cross_sample])
        all_mae_a = np.mean([s["total_mae_a"] for s in cross_sample])
        print("-" * W)
        print(
            f"{'GRAND MEAN':<18}  {'':>5}  {'':>4}  "
            f"{'':>8}  {'':>6}  {'':>10}  {'':>6}  "
            f"{all_mae_n:>10.5f}  {all_mae_a:>10.4f}"
        )

    # ── Combined figures ──────────────────────────────────────────────────
    single = len(cross_sample) == 1

    def _overlay(ax, qty_avg, qty_std, qty_pos_key, ylabel, title, errorbars=True):
        for s in cross_sample:
            f   = s["f_band"]
            avg = s[qty_avg]
            std = s[qty_std]
            col = s["color"]
            if single:
                for p in s["positions"]:
                    ax.plot(
                        p["f_THz"][p["band"]], p[qty_pos_key][p["band"]],
                        color=col, alpha=0.3, lw=0.8,
                    )
            ax.plot(f, avg, color=col, lw=1.8, label=s["name"])
            if errorbars:
                ax.errorbar(
                    f[::errorbar_step],
                    avg[::errorbar_step],
                    yerr=std[::errorbar_step],
                    fmt="none",
                    ecolor=col,
                    elinewidth=1.0,
                    capsize=3,
                    capthick=1.0,
                    alpha=0.9,
                )
        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Always-present figures (n and α without FP, with and without errorbars)
    fig_n,      ax_n      = plt.subplots(figsize=fig_size)
    fig_a,      ax_a      = plt.subplots(figsize=fig_size)
    fig_n_mean, ax_n_mean = plt.subplots(figsize=fig_size)
    fig_a_mean, ax_a_mean = plt.subplots(figsize=fig_size)

    _overlay(ax_n,      "avg_n_init", "std_n_init", "n_init",
             "Refractive index $n$",  "Refractive index — no FP correction")
    _overlay(ax_a,      "avg_a_init", "std_a_init", "alpha_init",
             r"$\alpha$ (cm$^{-1}$)", "Absorption — no FP correction")
    _overlay(ax_n_mean, "avg_n_init", "std_n_init", "n_init",
             "Refractive index $n$",  "Mean refractive index — no FP correction",
             errorbars=False)
    _overlay(ax_a_mean, "avg_a_init", "std_a_init", "alpha_init",
             r"$\alpha$ (cm$^{-1}$)", r"Mean absorption — no FP correction",
             errorbars=False)

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

        _overlay(ax_n_fp,      "avg_n_fp",   "std_n_fp",   "n_fp",
                 "Refractive index $n$",  "Refractive index — with FP correction")
        _overlay(ax_a_fp,      "avg_a_fp",   "std_a_fp",   "alpha_fp",
                 r"$\alpha$ (cm$^{-1}$)", "Absorption — with FP correction")
        _overlay(ax_n_fp_mean, "avg_n_fp",   "std_n_fp",   "n_fp",
                 "Refractive index $n$",  "Mean refractive index — with FP correction",
                 errorbars=False)
        _overlay(ax_a_fp_mean, "avg_a_fp",   "std_a_fp",   "alpha_fp",
                 r"$\alpha$ (cm$^{-1}$)", r"Mean absorption — with FP correction",
                 errorbars=False)

        figures_to_save += [
            (fig_n_fp,      "n_withfp"),
            (fig_a_fp,      "alpha_withfp"),
            (fig_n_fp_mean, "n_mean_withfp"),
            (fig_a_fp_mean, "alpha_mean_withfp"),
        ]

    for fig_obj, _ in figures_to_save:
        fig_obj.tight_layout()

    plt.show()

    for fig_obj, tag in figures_to_save:
        stem = f"batch_{tag}"
        viz.save_figure(fig_obj, output_run_dir / stem, formats=formats, dpi=dpi)
        print(f"  Saved: {output_run_dir / stem}.{formats[0]}")

    print(f"\nAll results saved in: {output_run_dir}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else CONFIG_DEFAULT
    main(path)
