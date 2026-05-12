"""
fp_single_sample.py
===================
FP-aware refractive index extraction for all (or a chosen number of)
measurement positions in a single sample folder.

The script auto-discovers matching air/point subfolder pairs, so there
is nothing to edit here — just adjust the YAML config.

Usage
-----
    python examples/fp_single_sample.py
    python examples/fp_single_sample.py /path/to/single_sample_fp.yaml

Output (written to <output_dir>/)
------
    <sample_name>_n_fp_all_points.png/pdf     — FP-corrected n(f), all positions
    <sample_name>_n_init_all_points.png/pdf   — Standard n(f), all positions
    <sample_name>_fp_results.tsv              — Numerical results table
"""

import sys
import re
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


CONFIG_DEFAULT = Path(__file__).parent.parent / "config" / "all_measurment_single_sample_fp.yaml"


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
    """
    Return sorted list of (index, air_path, point_path) for every matched pair
    found in *sample_dir*.  Only pairs where BOTH subfolders exist are kept.
    If *n_points* is given, only the first *n_points* pairs are returned.
    """
    # Find all air subdirs and extract their numeric suffix
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


# ── main ─────────────────────────────────────────────────────────────────────

def main(config_path: str | Path = CONFIG_DEFAULT) -> None:
    cfg = load_yaml(config_path)

    sam_cfg   = cfg["sample"]
    load_cfg  = cfg["loading"]
    align_cfg = cfg["alignment"]
    ri_cfg    = cfg["refractive_index"]
    plot_cfg  = cfg["plot"]

    sample_dir    = Path(sam_cfg["dir"])
    thickness     = sam_cfg["thickness_m"]
    air_prefix    = sam_cfg.get("air_prefix", "air")
    point_prefix  = sam_cfg.get("point_prefix", "point")
    n_points      = sam_cfg.get("n_points")   # None means all

    correction_factor       = load_cfg["correction_factor"]
    max_traces              = load_cfg.get("max_traces")   # None = all traces
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

    sample_name = sample_dir.name

    output_dir = Path(cfg["output"]) if cfg.get("output") else sample_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    f_low  = ri_cfg["f_low_THz"]
    f_high = ri_cfg["f_high_THz"]

    # ── discover pairs ───────────────────────────────────────────────────────
    pairs = discover_pairs(sample_dir, air_prefix, point_prefix, n_points)

    if not pairs:
        print(f"No matched {air_prefix}N / {point_prefix}N pairs found in {sample_dir}")
        return

    indices_found = [idx for idx, _, _ in pairs]
    traces_str = str(max_traces) if max_traces is not None else "all"
    print(f"Sample       : {sample_name}")
    print(f"Thickness    : {thickness * 1e3:.3f} mm")
    print(f"Positions    : {len(pairs)}  ({air_prefix}{indices_found[0]} … "
          f"{air_prefix}{indices_found[-1]})")
    print(f"Traces/folder: {traces_str}")
    print(f"Output dir   : {output_dir}\n")

    colors = cm.tab10(np.linspace(0, 1, len(pairs)))

    results = []

    for (idx, air_path, point_path), color in tqdm(
        zip(pairs, colors), total=len(pairs), desc="Positions", unit="pos"
    ):
        pos_name = f"{point_prefix}{idx}"

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
        ref = thz_tds.THZDataset(path=air_path,   **_ds_kwargs)
        sam = thz_tds.THZDataset(path=point_path, **_ds_kwargs)

        tqdm.write(f"  Processing: {sample_name} / {pos_name}  |  thickness = {thickness * 1e3:.3f} mm")

        fp_result = compute_refractive_index_fp(
            ref_ds=ref,
            sam_ds=sam,
            thickness_m=thickness,
            f_low_THz=f_low,
            f_high_THz=f_high,
        )

        band = (fp_result.f_THz >= f_low) & (fp_result.f_THz <= f_high)
        tqdm.write(
            f"  {pos_name}: n_fp={fp_result.n_fp[band].mean():.4f}  "
            f"n_init={fp_result.n_init[band].mean():.4f}"
        )

        results.append({
            "name":       pos_name,
            "color":      color,
            "f_THz":      fp_result.f_THz,
            "n_fp":       fp_result.n_fp,
            "n_init":     fp_result.n_init,
            "alpha_fp":   fp_result.alpha_cm_fp,
            "alpha_init": fp_result.alpha_init_cm,
            "kappa_fp":   fp_result.kappa_fp,
        })

    # ── Compute band-interpolated arrays for averaging ───────────────────────
    f_ref  = results[0]["f_THz"]
    band   = (f_ref >= f_low) & (f_ref <= f_high)
    f_band = f_ref[band]

    n_fp_all    = np.array([r["n_fp"][band]    for r in results])
    n_init_all  = np.array([r["n_init"][band]  for r in results])
    alpha_fp_all   = np.array([r["alpha_fp"][band]   for r in results])
    alpha_init_all = np.array([r["alpha_init"][band] for r in results])

    avg_n_fp    = n_fp_all.mean(axis=0)
    std_n_fp    = n_fp_all.std(axis=0)
    avg_n_init  = n_init_all.mean(axis=0)
    std_n_init  = n_init_all.std(axis=0)
    avg_a_fp    = alpha_fp_all.mean(axis=0)
    std_a_fp    = alpha_fp_all.std(axis=0)
    avg_a_init  = alpha_init_all.mean(axis=0)
    std_a_init  = alpha_init_all.std(axis=0)

    dpi       = plot_cfg["dpi"]
    formats   = plot_cfg["formats"]
    fsize     = plot_cfg["figure_size"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_pos     = len(results)

    def _stem(tag: str) -> str:
        return f"{sample_name}_{tag}_{timestamp}"

    # ── Figure 1: n(f) all positions — FP only ───────────────────────────────
    fig1, ax1 = plt.subplots(figsize=fsize)
    for r in results:
        mask = (r["f_THz"] >= f_low) & (r["f_THz"] <= f_high)
        ax1.plot(r["f_THz"][mask], r["n_fp"][mask],
                 color=r["color"], lw=1.5, label=r["name"])
    ax1.plot(f_band, avg_n_fp, color="black", lw=2.5, label="average FP")
    ax1.set_xlabel("Frequency (THz)")
    ax1.set_ylabel("Refractive index  n(f)")
    ax1.set_title(f"{sample_name} — n(f)  FP-corrected  (all {n_pos} positions)")
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    ax1.grid(True, linestyle="--", alpha=0.5)
    fig1.tight_layout()

    # ── Figure 2: n(f) all positions — no-FP only ────────────────────────────
    fig2, ax2 = plt.subplots(figsize=fsize)
    for r in results:
        mask = (r["f_THz"] >= f_low) & (r["f_THz"] <= f_high)
        ax2.plot(r["f_THz"][mask], r["n_init"][mask],
                 color=r["color"], lw=1.5, label=r["name"])
    ax2.plot(f_band, avg_n_init, color="black", lw=2.5, label="average no-FP")
    ax2.set_xlabel("Frequency (THz)")
    ax2.set_ylabel("Refractive index  n(f)")
    ax2.set_title(f"{sample_name} — n(f)  no FP correction  (all {n_pos} positions)")
    ax2.legend(loc="upper right", fontsize=8, ncol=2)
    ax2.grid(True, linestyle="--", alpha=0.5)
    fig2.tight_layout()

    # ── Figure 3: α(f) all positions — FP only ───────────────────────────────
    fig3, ax3 = plt.subplots(figsize=fsize)
    for r in results:
        mask = (r["f_THz"] >= f_low) & (r["f_THz"] <= f_high)
        ax3.plot(r["f_THz"][mask], r["alpha_fp"][mask],
                 color=r["color"], lw=1.5, label=r["name"])
    ax3.plot(f_band, avg_a_fp, color="black", lw=2.5, label="average FP")
    ax3.set_xlabel("Frequency (THz)")
    ax3.set_ylabel("Absorption coefficient  α (cm⁻¹)")
    ax3.set_title(f"{sample_name} — α(f)  FP-corrected  (all {n_pos} positions)")
    ax3.legend(loc="upper right", fontsize=8, ncol=2)
    ax3.grid(True, linestyle="--", alpha=0.5)
    fig3.tight_layout()

    # ── Figure 4: α(f) all positions — no-FP only ────────────────────────────
    fig4, ax4 = plt.subplots(figsize=fsize)
    for r in results:
        mask = (r["f_THz"] >= f_low) & (r["f_THz"] <= f_high)
        ax4.plot(r["f_THz"][mask], r["alpha_init"][mask],
                 color=r["color"], lw=1.5, label=r["name"])
    ax4.plot(f_band, avg_a_init, color="black", lw=2.5, label="average no-FP")
    ax4.set_xlabel("Frequency (THz)")
    ax4.set_ylabel("Absorption coefficient  α (cm⁻¹)")
    ax4.set_title(f"{sample_name} — α(f)  no FP correction  (all {n_pos} positions)")
    ax4.legend(loc="upper right", fontsize=8, ncol=2)
    ax4.grid(True, linestyle="--", alpha=0.5)
    fig4.tight_layout()

    # ── Figure 5: average n(f) no-FP + ±1σ variation across positions ────────
    fig5, ax5 = plt.subplots(figsize=fsize)
    ax5.plot(f_band, avg_n_init, color="tab:blue", lw=2.2,
             label=f"mean n  ({n_pos} positions)")
    ax5.fill_between(f_band, avg_n_init - std_n_init, avg_n_init + std_n_init,
                     color="tab:blue", alpha=0.25, label="±1σ variation")
    for r in results:
        mask = (r["f_THz"] >= f_low) & (r["f_THz"] <= f_high)
        ax5.plot(r["f_THz"][mask], r["n_init"][mask],
                 color=r["color"], lw=0.9, alpha=0.55, label=r["name"])
    ax5.set_xlabel("Frequency (THz)")
    ax5.set_ylabel("Refractive index  n(f)")
    ax5.set_title(f"{sample_name} — Average n(f) no-FP  ±1σ across positions")
    ax5.legend(loc="upper right", fontsize=8, ncol=2)
    ax5.grid(True, linestyle="--", alpha=0.5)
    fig5.tight_layout()

    # ── Figure 6: average n(f) with FP + ±1σ variation across positions ──────
    fig6, ax6 = plt.subplots(figsize=fsize)
    ax6.plot(f_band, avg_n_fp, color="tab:red", lw=2.2,
             label=f"mean n FP  ({n_pos} positions)")
    ax6.fill_between(f_band, avg_n_fp - std_n_fp, avg_n_fp + std_n_fp,
                     color="tab:red", alpha=0.25, label="±1σ variation")
    for r in results:
        mask = (r["f_THz"] >= f_low) & (r["f_THz"] <= f_high)
        ax6.plot(r["f_THz"][mask], r["n_fp"][mask],
                 color=r["color"], lw=0.9, alpha=0.55, label=r["name"])
    ax6.set_xlabel("Frequency (THz)")
    ax6.set_ylabel("Refractive index  n(f)")
    ax6.set_title(f"{sample_name} — Average n(f) FP-corrected  ±1σ across positions")
    ax6.legend(loc="upper right", fontsize=8, ncol=2)
    ax6.grid(True, linestyle="--", alpha=0.5)
    fig6.tight_layout()

    # ── Figure 7: grand average FP — n(f) + α(f) 2-panel ────────────────────
    fig7, (ax7n, ax7a) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax7n.plot(f_band, avg_n_fp, color="tab:red", lw=2.2, label="FP average")
    ax7n.fill_between(f_band, avg_n_fp - std_n_fp, avg_n_fp + std_n_fp,
                      color="tab:red", alpha=0.2, label="±1σ")
    ax7n.set_ylabel("Refractive index  n(f)")
    ax7n.set_title(f"{sample_name} — Grand average FP  ±1σ")
    ax7n.legend(fontsize=9); ax7n.grid(True, linestyle="--", alpha=0.5)
    ax7a.plot(f_band, avg_a_fp, color="tab:red", lw=2.2, label="FP average")
    ax7a.fill_between(f_band, avg_a_fp - std_a_fp, avg_a_fp + std_a_fp,
                      color="tab:red", alpha=0.2, label="±1σ")
    ax7a.set_xlabel("Frequency (THz)")
    ax7a.set_ylabel("Absorption coefficient  α (cm⁻¹)")
    ax7a.legend(fontsize=9); ax7a.grid(True, linestyle="--", alpha=0.5)
    fig7.tight_layout()

    # ── Figure 8: grand average no-FP — n(f) + α(f) 2-panel ─────────────────
    fig8, (ax8n, ax8a) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax8n.plot(f_band, avg_n_init, color="tab:blue", lw=2.2, label="no-FP average")
    ax8n.fill_between(f_band, avg_n_init - std_n_init, avg_n_init + std_n_init,
                      color="tab:blue", alpha=0.2, label="±1σ")
    ax8n.set_ylabel("Refractive index  n(f)")
    ax8n.set_title(f"{sample_name} — Grand average no-FP  ±1σ")
    ax8n.legend(fontsize=9); ax8n.grid(True, linestyle="--", alpha=0.5)
    ax8a.plot(f_band, avg_a_init, color="tab:blue", lw=2.2, label="no-FP average")
    ax8a.fill_between(f_band, avg_a_init - std_a_init, avg_a_init + std_a_init,
                      color="tab:blue", alpha=0.2, label="±1σ")
    ax8a.set_xlabel("Frequency (THz)")
    ax8a.set_ylabel("Absorption coefficient  α (cm⁻¹)")
    ax8a.legend(fontsize=9); ax8a.grid(True, linestyle="--", alpha=0.5)
    fig8.tight_layout()

    plt.show()

    # ── Save figures ─────────────────────────────────────────────────────────
    for fig, tag in [
        (fig1, "n_fp_all_positions"),
        (fig2, "n_nofp_all_positions"),
        (fig3, "alpha_fp_all_positions"),
        (fig4, "alpha_nofp_all_positions"),
        (fig5, "n_avg_variation_nofp"),
        (fig6, "n_avg_variation_fp"),
        (fig7, "grand_avg_fp"),
        (fig8, "grand_avg_nofp"),
    ]:
        for path in viz.save_figure(fig, output_dir, stem=_stem(tag),
                                    dpi=dpi, formats=formats):
            print(f"Saved: {path}")

    # ── Save TSV ─────────────────────────────────────────────────────────────
    header_parts = ["frequency_THz"]
    for r in results:
        n = r["name"]
        header_parts += [f"{n}_n_fp", f"{n}_n_init",
                         f"{n}_alpha_fp_cm", f"{n}_alpha_init_cm",
                         f"{n}_kappa_fp"]

    columns = [f_band]
    for r in results:
        columns += [
            r["n_fp"][band],
            r["n_init"][band],
            r["alpha_fp"][band],
            r["alpha_init"][band],
            r["kappa_fp"][band],
        ]

    tsv_path = output_dir / f"{sample_name}_fp_results.tsv"
    np.savetxt(
        tsv_path,
        np.column_stack(columns),
        delimiter="\t",
        header="\t".join(header_parts),
        comments="",
    )
    print(f"Saved: {tsv_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'Position':<12}  {'n_fp':>8}  {'n_nofp':>8}  {'Δn':>8}  {'Δn %':>7}  "
          f"{'α_fp':>8}  {'α_nofp':>8}  {'Δα':>8}  {'mean|Δn|(f)':>12}")
    print("-" * 90)

    per_pos_n_fp    = n_fp_all.mean(axis=1)
    per_pos_n_init  = n_init_all.mean(axis=1)
    per_pos_a_fp    = alpha_fp_all.mean(axis=1)
    per_pos_a_init  = alpha_init_all.mean(axis=1)
    per_pos_mae_n   = np.abs(n_fp_all - n_init_all).mean(axis=1)

    for i, r in enumerate(results):
        dn      = per_pos_n_fp[i]  - per_pos_n_init[i]
        dn_pct  = dn / per_pos_n_init[i] * 100
        da      = per_pos_a_fp[i]  - per_pos_a_init[i]
        print(
            f"{r['name']:<12}  {per_pos_n_fp[i]:>8.4f}  {per_pos_n_init[i]:>8.4f}"
            f"  {dn:>+8.4f}  {dn_pct:>+6.2f}%"
            f"  {per_pos_a_fp[i]:>8.3f}  {per_pos_a_init[i]:>8.3f}"
            f"  {da:>+8.3f}  {per_pos_mae_n[i]:>12.5f}"
        )

    # ── Overall mean across all positions ────────────────────────────────────
    grand_n_fp    = per_pos_n_fp.mean()
    grand_n_init  = per_pos_n_init.mean()
    std_n_fp      = per_pos_n_fp.std()
    std_n_init    = per_pos_n_init.std()
    total_mae_n   = per_pos_mae_n.mean()          # mean over positions and frequency
    total_mae_a   = np.abs(alpha_fp_all - alpha_init_all).mean()

    print("-" * 90)
    print(f"{'MEAN ± STD':<12}  "
          f"{grand_n_fp:>8.4f}  {grand_n_init:>8.4f}"
          f"  {grand_n_fp - grand_n_init:>+8.4f}  "
          f"  ±{std_n_fp:.4f} / ±{std_n_init:.4f}")
    print(f"\nTotal mean |Δn(f)|  across all positions & frequencies : {total_mae_n:.5f}")
    print(f"Total mean |Δα(f)|  across all positions & frequencies : {total_mae_a:.4f} cm⁻¹")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else CONFIG_DEFAULT
    main(path)
