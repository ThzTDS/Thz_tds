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

    f_low  = ri_cfg["f_low_THz"]
    f_high = ri_cfg["f_high_THz"]

    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)

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

        fp = compute_refractive_index_fp(
            ref_ds=ref, sam_ds=sam,
            thickness_m=thickness,
            f_low_THz=f_low, f_high_THz=f_high,
        )

        band = (fp.f_THz >= f_low) & (fp.f_THz <= f_high)
        tqdm.write(
            f"  {pos_name}: n={fp.n_init[band].mean():.4f}  "
            f"n with FP={fp.n_fp[band].mean():.4f}"
        )

        positions.append({
            "name":       pos_name,
            "color":      pos_col,
            "f_THz":      fp.f_THz,
            "n_fp":       fp.n_fp,
            "n_init":     fp.n_init,
            "alpha_fp":   fp.alpha_cm_fp,
            "alpha_init": fp.alpha_init_cm,
            "band":       band,
        })

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
    per_mae_n  = np.abs(n_fp_all - n_init_all).mean(axis=1)
    per_mae_a  = np.abs(alpha_fp_all - alpha_init_all).mean(axis=1)

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
        header += [f"{pn}_n", f"{pn}_n_withfp", f"{pn}_alpha", f"{pn}_alpha_withfp"]
        cols   += [p["n_init"][p["band"]], p["n_fp"][p["band"]],
                   p["alpha_init"][p["band"]], p["alpha_fp"][p["band"]]]
    header += ["avg_n", "avg_n_withfp", "avg_alpha", "avg_alpha_withfp"]
    cols   += [avg_n_init, avg_n_fp, avg_a_init, avg_a_fp]

    tsv = output_dir / f"{name}_fp_results_{timestamp}.tsv"
    np.savetxt(tsv, np.column_stack(cols),
               delimiter="\t", header="\t".join(header), comments="")
    tqdm.write(f"  Saved: {tsv}")

    return {
        # identification
        "name":          name,
        "thickness_mm":  thickness * 1e3,
        "n_positions":   len(pairs),
        "output_dir":    output_dir,
        # per-position traces (for detail view when single sample)
        "positions":     positions,
        # band-averaged statistics
        "f_band":        f_band,
        "avg_n_init":    avg_n_init,   "std_n_init":  std_n_init,
        "avg_n_fp":      avg_n_fp,     "std_n_fp":    std_n_fp,
        "avg_a_init":    avg_a_init,   "std_a_init":  std_a_init,
        "avg_a_fp":      avg_a_fp,     "std_a_fp":    std_a_fp,
        # cross-sample summary scalars
        "n_nofp_mean":   per_n_init.mean(),
        "n_nofp_std":    per_n_init.std(),
        "n_fp_mean":     per_n_fp.mean(),
        "n_fp_std":      per_n_fp.std(),
        "total_mae_n":   total_mae_n,
        "total_mae_a":   total_mae_a,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main(config_path: str | Path = CONFIG_DEFAULT) -> None:
    cfg = load_yaml(config_path)

    output_root = Path(cfg["output"])
    output_root.mkdir(parents=True, exist_ok=True)

    shared   = {k: cfg[k] for k in ("loading", "alignment", "refractive_index", "plot")}
    plot_cfg = shared["plot"]
    fig_size = tuple(plot_cfg["figure_size"])
    dpi      = plot_cfg["dpi"]
    formats  = plot_cfg["formats"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Config      : {config_path}")
    print(f"Samples     : {len(cfg['samples'])}")
    print(f"Output root : {output_root}")
    print(f"Timestamp   : {timestamp}\n")

    n_samples     = len(cfg["samples"])
    sample_colors = cm.tab10(np.linspace(0, 1, max(n_samples, 1)))
    cross_sample  = []

    for entry, s_col in zip(
        tqdm(cfg["samples"], desc="Samples", unit="sample"),
        sample_colors,
    ):
        summary = run_sample(entry, shared, output_root, timestamp=timestamp)
        if summary:
            summary["color"] = s_col
            cross_sample.append(summary)

    if not cross_sample:
        print("No samples processed.")
        return

    # ── Cross-sample summary table ────────────────────────────────────────
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

    def _overlay(ax, qty_avg, qty_std, qty_pos_key, ylabel, title):
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
            ax.fill_between(f, avg - std, avg + std, color=col, alpha=0.25)
        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    f1,  ax1          = plt.subplots(figsize=fig_size)
    f2,  ax2          = plt.subplots(figsize=fig_size)
    f3,  ax3          = plt.subplots(figsize=fig_size)
    f4,  ax4          = plt.subplots(figsize=fig_size)
    f5, (ax5a, ax5b)  = plt.subplots(1, 2, figsize=fig_size)
    f6, (ax6a, ax6b)  = plt.subplots(1, 2, figsize=fig_size)

    _overlay(ax1,  "avg_n_init", "std_n_init", "n_init",
             "Refractive index $n$",   "Refractive index — no FP correction")
    _overlay(ax2,  "avg_n_fp",   "std_n_fp",   "n_fp",
             "Refractive index $n$",   "Refractive index — with FP correction")
    _overlay(ax3,  "avg_a_init", "std_a_init", "alpha_init",
             r"$\alpha$ (cm$^{-1}$)",  "Absorption — no FP correction")
    _overlay(ax4,  "avg_a_fp",   "std_a_fp",   "alpha_fp",
             r"$\alpha$ (cm$^{-1}$)",  "Absorption — with FP correction")

    _overlay(ax5a, "avg_n_init", "std_n_init", "n_init",
             "Refractive index $n$",   "n — no FP")
    _overlay(ax5b, "avg_a_init", "std_a_init", "alpha_init",
             r"$\alpha$ (cm$^{-1}$)",  r"$\alpha$ — no FP")
    f5.suptitle("Grand average — no FP correction")
    f5.tight_layout()

    _overlay(ax6a, "avg_n_fp",   "std_n_fp",   "n_fp",
             "Refractive index $n$",   "n with FP")
    _overlay(ax6b, "avg_a_fp",   "std_a_fp",   "alpha_fp",
             r"$\alpha$ (cm$^{-1}$)",  r"$\alpha$ with FP")
    f6.suptitle("Grand average — with FP correction")
    f6.tight_layout()

    for fig_obj in (f1, f2, f3, f4):
        fig_obj.tight_layout()

    plt.show()

    for fig_obj, tag in [
        (f1, "n"),
        (f2, "n_withfp"),
        (f3, "alpha"),
        (f4, "alpha_withfp"),
        (f5, "grand_avg"),
        (f6, "grand_avg_withfp"),
    ]:
        stem = f"batch_{tag}_{timestamp}"
        viz.save_figure(fig_obj, output_root / stem, formats=formats, dpi=dpi)
        print(f"  Saved: {output_root / stem}.{formats[0]}")

    print(f"\nAll results saved in: {output_root}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else CONFIG_DEFAULT
    main(path)
